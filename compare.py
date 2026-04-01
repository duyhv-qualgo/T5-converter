"""
Cross-format benchmark: ONNX FP32 vs LiteRT Explicit FP32 vs LiteRT Stateful FP32.

Compares accuracy (BLEU-4, chrF++, exact match) and latency across all three
export formats on the shared 60-pair test set.

Requirements:
  - ONNX models exported via onnx/export.py
  - LiteRT explicit models converted via litert/explicit/convert.py
  - LiteRT stateful models converted via litert/stateful/convert.py

Both litert/.venv and onnx/.venv have different deps (litert-torch vs optimum).
This script requires BOTH to be installed in the same environment, OR run each
subset separately with --only-onnx / --only-litert.

Alternatively: activate litert/.venv (which has ai-edge-litert) and install
onnxruntime manually:
  source litert/.venv/bin/activate
  pip install onnxruntime

Usage:
  python compare.py
  python compare.py --only-onnx
  python compare.py --only-litert
  python compare.py --verbose
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from shared.testset import ALL_PAIRS, TASK_EN_VI, TASK_VI_EN
from shared.metrics import compute_metrics_split, print_results_table

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent.parent / "translation" / "model_pt"
OUT_DIR   = Path(__file__).parent / "litert" / "output"
ONNX_FP32 = Path(__file__).parent / "onnx" / "output" / "fp32"

SEQ_LEN  = 128
N_HEADS  = 8
D_KV     = 32
N_LAYERS = 4
EOS_ID   = 1
PAD_ID   = 0


# ── Input helpers ─────────────────────────────────────────────────────────────

def make_litert_inputs(tokenizer, text: str, task_id: int):
    ids = [task_id] + tokenizer.encode(text, add_special_tokens=False)
    if ids[-1] != EOS_ID:
        ids.append(EOS_ID)
    real = min(len(ids), SEQ_LEN)
    input_ids = np.zeros((1, SEQ_LEN), dtype=np.int32)
    input_ids[0, :real] = ids[:real]
    pad_mask = np.zeros(SEQ_LEN, dtype=np.float32)
    pad_mask[real:] = float("-inf")
    return input_ids, np.arange(SEQ_LEN, dtype=np.int32), pad_mask


def make_onnx_inputs(tokenizer, text: str, task_id: int):
    ids = [task_id] + tokenizer.encode(text, add_special_tokens=False)
    if ids[-1] != EOS_ID:
        ids.append(EOS_ID)
    input_ids    = np.array([ids], dtype=np.int64)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    return input_ids, attention_mask


# ── LiteRT loaders + inference ────────────────────────────────────────────────

def _load_litert(path: str):
    import ai_edge_litert.interpreter as m
    interp = m.Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp


def decode_explicit(enc_runner, dec_runner, tokenizer, text: str, task_id: int) -> str:
    ids, pos, mask = make_litert_inputs(tokenizer, text, task_id)
    hs  = enc_runner(**{"args_0": ids, "args_1": pos, "args_2": mask})["output_0"]
    kvc = {f"args_{4 + i}": np.zeros((1, SEQ_LEN, N_HEADS, D_KV), dtype=np.float32)
           for i in range(N_LAYERS * 2)}
    gen = []
    tok = PAD_ID
    for step in range(64):
        out      = dec_runner(**{"args_0": hs, "args_1": np.array([[tok]], dtype=np.int32),
                                 "args_2": np.array([step], dtype=np.int32),
                                 "args_3": mask, **kvc})
        next_tok = int(np.argmax(out["output_0"][0, 0, :]))
        if next_tok == EOS_ID:
            break
        gen.append(next_tok)
        tok = next_tok
        for i in range(N_LAYERS * 2):
            kvc[f"args_{4 + i}"] = out[f"output_{1 + i}"]
    return tokenizer.decode(gen, skip_special_tokens=True)


def decode_stateful(enc_runner, dec_model_path: str,
                    tokenizer, text: str, task_id: int) -> str:
    ids, pos, mask = make_litert_inputs(tokenizer, text, task_id)
    hs         = enc_runner(input_ids=ids, input_pos=pos, pad_mask=mask)["output_0"]
    dec_runner = _load_litert(dec_model_path).get_signature_runner("decode")
    gen = []
    tok = PAD_ID
    for step in range(64):
        out      = dec_runner(encoder_hidden_states=hs,
                               decoder_input_ids=np.array([[tok]], dtype=np.int32),
                               step=np.array(step, dtype=np.int32),
                               pad_mask=mask)
        next_tok = int(np.argmax(out["output_0"][0, 0, :]))
        if next_tok == EOS_ID:
            break
        gen.append(next_tok)
        tok = next_tok
    return tokenizer.decode(gen, skip_special_tokens=True)


# ── ONNX loader + inference ───────────────────────────────────────────────────

def _load_onnx(onnx_dir: Path, num_threads: int = 4):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.inter_op_num_threads = num_threads
    opts.intra_op_num_threads = num_threads
    enc = ort.InferenceSession(str(onnx_dir / "encoder_model.onnx"),             opts)
    dec = ort.InferenceSession(str(onnx_dir / "decoder_model.onnx"),              opts)
    dwp = ort.InferenceSession(str(onnx_dir / "decoder_with_past_model.onnx"),    opts)
    return enc, dec, dwp


def decode_onnx(enc_sess, dec_sess, dwp_sess, tokenizer,
                text: str, task_id: int) -> str:
    input_ids, attn_mask = make_onnx_inputs(tokenizer, text, task_id)
    enc_out = enc_sess.run(None, {"input_ids": input_ids, "attention_mask": attn_mask})
    hs      = enc_out[0]

    dec_out = dec_sess.run(None, {
        "encoder_hidden_states": hs, "encoder_attention_mask": attn_mask,
        "input_ids": np.array([[PAD_ID]], dtype=np.int64),
    })
    next_tok = int(np.argmax(dec_out[0][0, 0, :]))
    if next_tok == EOS_ID:
        return ""
    gen = [next_tok]

    dec_names = [o.name for o in dec_sess.get_outputs()]
    dwp_names = [o.name for o in dwp_sess.get_outputs()]
    past = {n.replace("present.", "past_key_values."): t
            for n, t in zip(dec_names[1:], dec_out[1:])}

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
        for n, t in zip(dwp_names[1:], dwp_out[1:]):
            past[n.replace("present.", "past_key_values.")] = t

    return tokenizer.decode(gen, skip_special_tokens=True)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_model(name: str, translate_fn, tokenizer,
              pairs: list, verbose: bool) -> dict:
    print(f"  [{name}] …")
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
            print(f"    [{task_id}] {src}")
            print(f"      ref : {ref}")
            print(f"      hyp : {hyps[i]}  {mark}")
        print()
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-onnx",   action="store_true")
    parser.add_argument("--only-litert", action="store_true")
    parser.add_argument("--verbose",     action="store_true")
    parser.add_argument("--threads",     type=int, default=4)
    args = parser.parse_args()

    pairs     = ALL_PAIRS
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    results   = {}

    print(f"Test set: {len(pairs)} pairs  "
          f"(EN→VI: {sum(1 for t,_,_ in pairs if t==TASK_EN_VI)}  "
          f"VI→EN: {sum(1 for t,_,_ in pairs if t==TASK_VI_EN)})")
    print()

    # ── ONNX ─────────────────────────────────────────────────────────────────
    if not args.only_litert and ONNX_FP32.exists() and (ONNX_FP32 / "encoder_model.onnx").exists():
        try:
            print(f"Loading ONNX FP32 from {ONNX_FP32} …")
            enc_o, dec_o, dwp_o = _load_onnx(ONNX_FP32, args.threads)
            results["ONNX FP32"] = run_model(
                "ONNX FP32",
                lambda tok, src, tid: decode_onnx(enc_o, dec_o, dwp_o, tok, src, tid),
                tokenizer, pairs, args.verbose,
            )
        except ImportError:
            print("  [skip] onnxruntime not installed in this env")
            print("         pip install onnxruntime")

    # ── LiteRT Explicit ───────────────────────────────────────────────────────
    explicit_fp32 = OUT_DIR / "t5_mini_explicit_fp32.tflite"
    if not args.only_onnx and explicit_fp32.exists():
        print(f"Loading LiteRT Explicit FP32 …")
        interp = _load_litert(str(explicit_fp32))
        er, dr = interp.get_signature_runner("encode"), interp.get_signature_runner("decode")
        results["LiteRT Explicit FP32"] = run_model(
            "LiteRT Explicit FP32",
            lambda tok, src, tid: decode_explicit(er, dr, tok, src, tid),
            tokenizer, pairs, args.verbose,
        )

    # ── LiteRT Stateful ───────────────────────────────────────────────────────
    stat_enc = OUT_DIR / "t5_mini_stateful_enc_fp32.tflite"
    stat_dec = OUT_DIR / "t5_mini_stateful_dec_fp32.tflite"
    if not args.only_onnx and stat_enc.exists() and stat_dec.exists():
        print(f"Loading LiteRT Stateful FP32 …")
        enc_runner = _load_litert(str(stat_enc)).get_signature_runner("encode")
        results["LiteRT Stateful FP32"] = run_model(
            "LiteRT Stateful FP32",
            lambda tok, src, tid: decode_stateful(enc_runner, str(stat_dec), tok, src, tid),
            tokenizer, pairs, args.verbose,
        )

    if not results:
        print("No models found. Run the export/convert scripts first.")
        return

    print_results_table(results, title="CROSS-FORMAT COMPARISON")


if __name__ == "__main__":
    main()
