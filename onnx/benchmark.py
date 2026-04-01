"""
Accuracy + latency benchmark for ONNX T5 models.

Evaluates FP32 and INT8 variants against HuggingFace baseline using
BLEU-4, chrF++, and exact match on the shared 60-pair test set.

Usage:
  source onnx/.venv/bin/activate
  python onnx/benchmark.py
  python onnx/benchmark.py --verbose
  python onnx/benchmark.py --skip-int8   # only FP32
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parents[1]))
from shared.testset import ALL_PAIRS, TASK_EN_VI, TASK_VI_EN
from shared.metrics import compute_metrics_split, print_results_table

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parents[2] / "translation" / "model_pt"
ONNX_FP32 = Path(__file__).parent / "output" / "fp32"
ONNX_INT8 = Path(__file__).parent / "output" / "int8"

EOS_ID = 1
PAD_ID = 0


# ── Session loader ────────────────────────────────────────────────────────────

def load_sessions(model_dir: Path, num_threads: int = 4):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.inter_op_num_threads = num_threads
    opts.intra_op_num_threads = num_threads
    enc = ort.InferenceSession(str(model_dir / "encoder_model.onnx"),             opts)
    dec = ort.InferenceSession(str(model_dir / "decoder_model.onnx"),              opts)
    dwp = ort.InferenceSession(str(model_dir / "decoder_with_past_model.onnx"),    opts)
    return enc, dec, dwp


# ── Input helpers ─────────────────────────────────────────────────────────────

def make_enc_inputs(tokenizer, text: str, task_id: int):
    ids = [task_id] + tokenizer.encode(text, add_special_tokens=False)
    if ids[-1] != EOS_ID:
        ids.append(EOS_ID)
    input_ids    = np.array([ids], dtype=np.int64)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    return input_ids, attention_mask


# ── Inference ─────────────────────────────────────────────────────────────────

def decode_onnx(enc_sess, dec_sess, dwp_sess, tokenizer,
                text: str, task_id: int) -> str:
    input_ids, attn_mask = make_enc_inputs(tokenizer, text, task_id)

    enc_out       = enc_sess.run(None, {"input_ids": input_ids, "attention_mask": attn_mask})
    hidden_states = enc_out[0]

    dec_out = dec_sess.run(None, {
        "encoder_hidden_states":  hidden_states,
        "encoder_attention_mask": attn_mask,
        "input_ids":              np.array([[PAD_ID]], dtype=np.int64),
    })
    logits  = dec_out[0]
    present = dec_out[1:]

    next_tok = int(np.argmax(logits[0, 0, :]))
    if next_tok == EOS_ID:
        return ""
    gen = [next_tok]

    dec_out_names = [o.name for o in dec_sess.get_outputs()]
    dwp_out_names = [o.name for o in dwp_sess.get_outputs()]
    past = {n.replace("present.", "past_key_values."): t
            for n, t in zip(dec_out_names[1:], present)}

    for _ in range(63):
        dwp_out  = dwp_sess.run(None, {
            "encoder_attention_mask": attn_mask,
            "input_ids": np.array([[gen[-1]]], dtype=np.int64),
            **past,
        })
        next_tok = int(np.argmax(dwp_out[0][0, 0, :]))
        if next_tok == EOS_ID:
            break
        gen.append(next_tok)
        for out_name, tensor in zip(dwp_out_names[1:], dwp_out[1:]):
            past[out_name.replace("present.", "past_key_values.")] = tensor

    return tokenizer.decode(gen, skip_special_tokens=True)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_model(name: str, translate_fn, tokenizer,
              pairs: list, verbose: bool) -> dict:
    print(f"  Running {name} …")
    t0   = time.perf_counter()
    hyps = [translate_fn(tokenizer, src, task_id) for task_id, src, _ in pairs]
    elapsed = time.perf_counter() - t0

    refs     = [r for _, _, r in pairs]
    task_ids = [t for t, _, _  in pairs]
    metrics  = compute_metrics_split(hyps, refs, task_ids)
    metrics["elapsed_s"]   = round(elapsed, 1)
    metrics["ms_per_sent"] = round(elapsed * 1000 / len(pairs), 1)

    if verbose:
        print()
        for i, (task_id, src, ref) in enumerate(pairs):
            mark = "✓" if hyps[i].strip().lower() == ref.strip().lower() else "~"
            print(f"  [{task_id}] {src}")
            print(f"    ref : {ref}")
            print(f"    hyp : {hyps[i]}  {mark}")
        print()
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose",   action="store_true")
    parser.add_argument("--skip-int8", action="store_true")
    parser.add_argument("--threads",   type=int, default=4)
    args = parser.parse_args()

    pairs     = ALL_PAIRS
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    results   = {}

    print(f"Test set: {len(pairs)} pairs  "
          f"(EN→VI: {sum(1 for t,_,_ in pairs if t==TASK_EN_VI)}  "
          f"VI→EN: {sum(1 for t,_,_ in pairs if t==TASK_VI_EN)})")
    print()

    if ONNX_FP32.exists() and (ONNX_FP32 / "encoder_model.onnx").exists():
        print(f"Loading ONNX FP32 from {ONNX_FP32} …")
        e32, d32, w32 = load_sessions(ONNX_FP32, args.threads)
        results["ONNX FP32"] = run_model(
            "ONNX FP32",
            lambda tok, src, tid: decode_onnx(e32, d32, w32, tok, src, tid),
            tokenizer, pairs, args.verbose,
        )
    else:
        print(f"[skip] ONNX FP32 not found at {ONNX_FP32}")
        print("       Run  python onnx/export.py  first.")

    if not args.skip_int8 and ONNX_INT8.exists() and (ONNX_INT8 / "encoder_model.onnx").exists():
        print(f"Loading ONNX INT8 from {ONNX_INT8} …")
        e8, d8, w8 = load_sessions(ONNX_INT8, args.threads)
        results["ONNX INT8"] = run_model(
            "ONNX INT8",
            lambda tok, src, tid: decode_onnx(e8, d8, w8, tok, src, tid),
            tokenizer, pairs, args.verbose,
        )

    if not results:
        print("No ONNX models found. Run  python onnx/export.py  first.")
        return

    print_results_table(results, title="ONNX ACCURACY RESULTS")


if __name__ == "__main__":
    main()
