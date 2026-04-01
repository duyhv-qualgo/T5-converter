"""
Verify ONNX T5 model: correctness vs HuggingFace + latency benchmark.

Inference pattern (optimum-style with KV cache):
  step 0 : encoder_model + decoder_model → logits + present.*.{key,value}
  step k : decoder_with_past_model (past = previous present) → logits + updated present

Usage:
  source onnx/.venv/bin/activate
  python onnx/verify.py                    # FP32
  python onnx/verify.py --dir output/int8  # INT8
  python onnx/verify.py --runs 50
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort
from transformers import T5ForConditionalGeneration, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parents[2] / "translation" / "model_pt"
ONNX_DIR  = Path(__file__).parent / "output" / "fp32"

EOS_ID = 1
PAD_ID = 0

TASK_EN_VI = 20000
TASK_VI_EN = 20001
DIRECTION  = {TASK_EN_VI: "en→vi", TASK_VI_EN: "vi→en"}

TEST_CASES = [
    (TASK_EN_VI, "Hello, how are you?"),
    (TASK_EN_VI, "The weather is beautiful today."),
    (TASK_EN_VI, "I would like to order a coffee please."),
    (TASK_VI_EN, "Xin chào, bạn có khỏe không?"),
    (TASK_VI_EN, "Hôm nay trời đẹp quá."),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_enc_inputs(tokenizer, text: str, task_id: int):
    ids = [task_id] + tokenizer.encode(text, add_special_tokens=False)
    if ids[-1] != EOS_ID:
        ids.append(EOS_ID)
    input_ids    = np.array([ids], dtype=np.int64)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    return input_ids, attention_mask


def load_sessions(onnx_dir: Path, num_threads: int = 4):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.inter_op_num_threads = num_threads
    opts.intra_op_num_threads = num_threads
    enc = ort.InferenceSession(str(onnx_dir / "encoder_model.onnx"), opts)
    dec = ort.InferenceSession(str(onnx_dir / "decoder_model.onnx"), opts)
    dwp = ort.InferenceSession(str(onnx_dir / "decoder_with_past_model.onnx"), opts)
    return enc, dec, dwp


# ── HuggingFace reference ─────────────────────────────────────────────────────

def hf_translate(model, tokenizer, text: str, task_id: int) -> str:
    ids = [task_id] + tokenizer.encode(text, add_special_tokens=False)
    if ids[-1] != EOS_ID:
        ids.append(EOS_ID)
    with torch.no_grad():
        out = model.generate(torch.tensor([ids]), max_new_tokens=64)
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ── ONNX inference ────────────────────────────────────────────────────────────

def onnx_translate(enc_sess, dec_sess, dwp_sess, tokenizer,
                   text: str, task_id: int, max_new_tokens: int = 64) -> str:
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
    present_names = dec_out_names[1:]
    past = {n.replace("present.", "past_key_values."): t
            for n, t in zip(present_names, present)}

    for _ in range(max_new_tokens - 1):
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


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark(enc_sess, dec_sess, dwp_sess, tokenizer, n_runs: int = 20):
    input_ids, attn_mask = make_enc_inputs(tokenizer, "Hello, how are you?", TASK_EN_VI)

    for _ in range(3):
        enc_sess.run(None, {"input_ids": input_ids, "attention_mask": attn_mask})

    t0 = time.perf_counter()
    for _ in range(n_runs):
        enc_out = enc_sess.run(None, {"input_ids": input_ids, "attention_mask": attn_mask})
    enc_ms        = (time.perf_counter() - t0) * 1000 / n_runs
    hidden_states = enc_out[0]

    dec_out = dec_sess.run(None, {
        "encoder_hidden_states":  hidden_states,
        "encoder_attention_mask": attn_mask,
        "input_ids":              np.array([[PAD_ID]], dtype=np.int64),
    })
    dec_out_names = [o.name for o in dec_sess.get_outputs()]
    dwp_out_names = [o.name for o in dwp_sess.get_outputs()]
    past = {n.replace("present.", "past_key_values."): t
            for n, t in zip(dec_out_names[1:], dec_out[1:])}

    t0 = time.perf_counter()
    for _ in range(n_runs):
        dwp_out = dwp_sess.run(None, {
            "encoder_attention_mask": attn_mask,
            "input_ids": np.array([[1]], dtype=np.int64),
            **past,
        })
    dec_ms = (time.perf_counter() - t0) * 1000 / n_runs

    return enc_ms, dec_ms


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",     default=str(ONNX_DIR), help="ONNX model directory")
    parser.add_argument("--runs",    type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    onnx_dir = Path(args.dir)
    for fname in ["encoder_model.onnx", "decoder_model.onnx", "decoder_with_past_model.onnx"]:
        if not (onnx_dir / fname).exists():
            raise FileNotFoundError(f"{onnx_dir / fname} not found.\nRun  python onnx/export.py  first.")

    total_mb = sum((onnx_dir / f).stat().st_size for f in
                   ["encoder_model.onnx", "decoder_model.onnx", "decoder_with_past_model.onnx"]
                   ) / 1024 / 1024
    print(f"ONNX dir : {onnx_dir}  ({total_mb:.1f} MB total)")

    enc_sess, dec_sess, dwp_sess = load_sessions(onnx_dir, args.threads)

    print("\nLoading HuggingFace reference …")
    hf_model  = T5ForConditionalGeneration.from_pretrained(str(MODEL_DIR)).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    print("\n" + "=" * 70)
    print("Correctness check")
    print("=" * 70)
    passed = 0
    for task_id, text in TEST_CASES:
        hf_out   = hf_translate(hf_model, tokenizer, text, task_id)
        onnx_out = onnx_translate(enc_sess, dec_sess, dwp_sess, tokenizer, text, task_id)
        match    = hf_out.strip().lower() == onnx_out.strip().lower()
        status   = "PASS" if match else "DIFF"
        if match: passed += 1
        print(f"  [{status}] {DIRECTION[task_id]}  \"{text[:45]}\"")
        print(f"    ONNX : {onnx_out}")
        print(f"    HF   : {hf_out}")
    print(f"\n  Exact match: {passed}/{len(TEST_CASES)}")

    print("\n" + "=" * 70)
    print(f"Latency  ({args.runs} runs)")
    print("=" * 70)
    enc_ms, dec_ms = benchmark(enc_sess, dec_sess, dwp_sess, tokenizer, args.runs)
    assumed = 20
    print(f"  Encoder  / call : {enc_ms:7.2f} ms")
    print(f"  Decoder  / step : {dec_ms:7.2f} ms")
    print(f"  Estimated total : {enc_ms + assumed * dec_ms:7.2f} ms"
          f"  (enc + {assumed} decode steps)")
    print("\nDone.")


if __name__ == "__main__":
    main()
