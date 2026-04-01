"""
Accuracy + latency benchmark across all LiteRT model variants.

Evaluates against HuggingFace baseline using BLEU-4, chrF++, exact match.

Models evaluated (if present in output/):
  - Explicit KV FP32   (t5_mini_explicit_fp32.tflite)
  - Explicit KV INT8   (t5_mini_explicit_int8.tflite)
  - Stateful FP32      (t5_mini_stateful_enc/dec_fp32.tflite)

Usage:
  source litert/.venv/bin/activate
  python litert/benchmark.py
  python litert/benchmark.py --verbose
  python litert/benchmark.py --skip-hf
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import ai_edge_litert.interpreter as litert_interp
from transformers import AutoTokenizer, T5ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parents[1]))
from shared.testset import ALL_PAIRS, TASK_EN_VI, TASK_VI_EN
from shared.metrics import compute_metrics_split, print_results_table

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR  = Path(__file__).parents[2] / "translation" / "model_pt"
LITERT_DIR = Path(__file__).parent / "output"

SEQ_LEN  = 128
N_HEADS  = 8
D_KV     = 32
N_LAYERS = 4
EOS_ID   = 1
PAD_ID   = 0


# ── Input helpers ─────────────────────────────────────────────────────────────

def make_enc_inputs(tokenizer, text: str, task_id: int):
    ids = [task_id] + tokenizer.encode(text, add_special_tokens=False)
    if ids[-1] != EOS_ID:
        ids.append(EOS_ID)
    real = min(len(ids), SEQ_LEN)
    input_ids = np.zeros((1, SEQ_LEN), dtype=np.int32)
    input_ids[0, :real] = ids[:real]
    pad_mask  = np.zeros(SEQ_LEN, dtype=np.float32)
    pad_mask[real:] = float("-inf")
    return input_ids, np.arange(SEQ_LEN, dtype=np.int32), pad_mask


def load_interp(path: str):
    interp = litert_interp.Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp


# ── Inference: explicit KV cache ──────────────────────────────────────────────

def decode_explicit(enc_runner, dec_runner, tokenizer, text: str, task_id: int) -> str:
    ids, pos, mask = make_enc_inputs(tokenizer, text, task_id)
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


# ── Inference: stateful KV cache ──────────────────────────────────────────────

def decode_stateful(enc_runner, dec_model_path: str,
                    tokenizer, text: str, task_id: int) -> str:
    ids, pos, mask = make_enc_inputs(tokenizer, text, task_id)
    hs = enc_runner(input_ids=ids, input_pos=pos, pad_mask=mask)["output_0"]
    dec_runner = load_interp(dec_model_path).get_signature_runner("decode")
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


# ── HuggingFace baseline ──────────────────────────────────────────────────────

def hf_translate(model, tokenizer, text: str, task_id: int) -> str:
    ids = [task_id] + tokenizer.encode(text, add_special_tokens=False)
    if ids[-1] != EOS_ID:
        ids.append(EOS_ID)
    with torch.no_grad():
        out = model.generate(torch.tensor([ids]), max_new_tokens=64)
    return tokenizer.decode(out[0], skip_special_tokens=True)


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
            print(f"  [{TASK_EN_VI if task_id == TASK_EN_VI else TASK_VI_EN}] {src}")
            print(f"    ref : {ref}")
            print(f"    hyp : {hyps[i]}  {mark}")
        print()
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose",  action="store_true")
    parser.add_argument("--skip-hf",  action="store_true")
    args = parser.parse_args()

    pairs     = ALL_PAIRS
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    results   = {}

    print(f"Test set: {len(pairs)} pairs  "
          f"(EN→VI: {sum(1 for t,_,_ in pairs if t==TASK_EN_VI)}  "
          f"VI→EN: {sum(1 for t,_,_ in pairs if t==TASK_VI_EN)})")
    print()

    if not args.skip_hf:
        print("Loading HuggingFace baseline …")
        hf = T5ForConditionalGeneration.from_pretrained(str(MODEL_DIR)).eval()
        results["HF baseline"] = run_model(
            "HF baseline",
            lambda tok, src, tid: hf_translate(hf, tok, src, tid),
            tokenizer, pairs, args.verbose,
        )
        del hf

    # Explicit KV FP32
    fp32_path = LITERT_DIR / "t5_mini_explicit_fp32.tflite"
    if fp32_path.exists():
        print(f"Loading {fp32_path.name} …")
        interp = load_interp(str(fp32_path))
        er, dr = interp.get_signature_runner("encode"), interp.get_signature_runner("decode")
        results["Explicit FP32"] = run_model(
            "Explicit FP32",
            lambda tok, src, tid: decode_explicit(er, dr, tok, src, tid),
            tokenizer, pairs, args.verbose,
        )

    # Explicit KV INT8
    int8_path = LITERT_DIR / "t5_mini_explicit_int8.tflite"
    if int8_path.exists():
        print(f"Loading {int8_path.name} …")
        interp = load_interp(str(int8_path))
        er, dr = interp.get_signature_runner("encode"), interp.get_signature_runner("decode")
        results["Explicit INT8"] = run_model(
            "Explicit INT8",
            lambda tok, src, tid: decode_explicit(er, dr, tok, src, tid),
            tokenizer, pairs, args.verbose,
        )

    # Stateful FP32
    stat_enc = LITERT_DIR / "t5_mini_stateful_enc_fp32.tflite"
    stat_dec = LITERT_DIR / "t5_mini_stateful_dec_fp32.tflite"
    if stat_enc.exists() and stat_dec.exists():
        print(f"Loading stateful FP32 …")
        enc_runner = load_interp(str(stat_enc)).get_signature_runner("encode")
        dec_path   = str(stat_dec)
        results["Stateful FP32"] = run_model(
            "Stateful FP32",
            lambda tok, src, tid: decode_stateful(enc_runner, dec_path, tok, src, tid),
            tokenizer, pairs, args.verbose,
        )

    if not results:
        print("No models found in", LITERT_DIR)
        print("Run  python litert/explicit/convert.py  and/or  python litert/stateful/convert.py  first.")
        return

    print_results_table(results, title="LiteRT ACCURACY RESULTS")


if __name__ == "__main__":
    main()
