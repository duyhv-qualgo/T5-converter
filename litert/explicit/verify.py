"""
Verify the explicit-KV LiteRT model: correctness vs HF + latency benchmark.

The decode signature threads k_i/v_i tensors as explicit inputs/outputs.
Caller allocates and threads the cache arrays between decode steps.

Usage:
  source litert/.venv/bin/activate
  python litert/explicit/verify.py
  python litert/explicit/verify.py --model t5_mini_explicit_fp32.tflite
  python litert/explicit/verify.py --runs 50
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR  = Path(__file__).parents[3] / "translation" / "model_pt"
LITERT_DIR = Path(__file__).parents[1] / "output"

SEQ_LEN        = 128
MAX_NEW_TOKENS = 64
NUM_HEADS      = 8      # must match convert.py
HEAD_DIM       = 32

TASK_EN_VI = 20000
TASK_VI_EN = 20001
EOS_ID     = 1
PAD_ID     = 0
DIRECTION  = {TASK_EN_VI: "en→vi", TASK_VI_EN: "vi→en"}

TEST_CASES = [
    (TASK_EN_VI, "Hello, how are you?"),
    (TASK_EN_VI, "The weather is beautiful today."),
    (TASK_EN_VI, "I would like to order a coffee please."),
    (TASK_VI_EN, "Xin chào, bạn có khỏe không?"),
    (TASK_VI_EN, "Hôm nay trời đẹp quá."),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_encoder_inputs(tokenizer, text: str, task_id: int):
    ids      = [task_id] + tokenizer.encode(text, add_special_tokens=False)
    if ids[-1] != EOS_ID:
        ids.append(EOS_ID)
    real_len = min(len(ids), SEQ_LEN)
    input_ids = np.zeros((1, SEQ_LEN), dtype=np.int32)
    input_ids[0, :real_len] = ids[:real_len]
    pad_mask = np.zeros(SEQ_LEN, dtype=np.float32)
    pad_mask[real_len:] = float("-inf")
    input_pos = np.arange(SEQ_LEN, dtype=np.int32)
    return input_ids, input_pos, pad_mask


def load_tflite(model_path: str):
    try:
        import ai_edge_litert.interpreter as m
        interp = m.Interpreter(model_path=model_path)
    except ImportError:
        import tflite_runtime.interpreter as m
        interp = m.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp.get_signature_runner("encode"), interp.get_signature_runner("decode")


# ── HuggingFace reference ─────────────────────────────────────────────────────

def hf_translate(model, tokenizer, text: str, task_id: int) -> str:
    ids = [task_id] + tokenizer.encode(text, add_special_tokens=False)
    if ids[-1] != EOS_ID:
        ids.append(EOS_ID)
    with torch.no_grad():
        out = model.generate(torch.tensor([ids]), max_new_tokens=MAX_NEW_TOKENS)
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ── LiteRT inference — explicit KV cache loop ─────────────────────────────────

def litert_translate(encode_runner, decode_runner, tokenizer,
                     text: str, task_id: int,
                     max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    input_ids, input_pos, pad_mask = build_encoder_inputs(tokenizer, text, task_id)

    enc_out       = encode_runner(args_0=input_ids, args_1=input_pos, args_2=pad_mask)
    hidden_states = enc_out["output_0"]

    # Discover how many k/v args the decode signature has (args_4 onward)
    dec_details = decode_runner.get_input_details()
    n_kv_args   = sum(1 for k in dec_details
                      if k.startswith("args_") and int(k.split("_")[1]) >= 4)
    kvc_shape   = (1, SEQ_LEN, NUM_HEADS, HEAD_DIM)
    kvc         = {f"args_{4 + i}": np.zeros(kvc_shape, dtype=np.float32)
                   for i in range(n_kv_args)}

    generated = []
    token_id  = PAD_ID

    for step in range(max_new_tokens):
        dec_out    = decode_runner(
            args_0=hidden_states,
            args_1=np.array([[token_id]], dtype=np.int32),
            args_2=np.array([step],       dtype=np.int32),
            args_3=pad_mask,
            **kvc,
        )
        next_token = int(np.argmax(dec_out["output_0"][0, 0, :]))
        if next_token == EOS_ID:
            break
        generated.append(next_token)
        token_id = next_token
        for i in range(n_kv_args):
            kvc[f"args_{4 + i}"] = dec_out[f"output_{1 + i}"]

    return tokenizer.decode(generated, skip_special_tokens=True)


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark(encode_runner, decode_runner, tokenizer, n_runs: int = 20):
    input_ids, input_pos, pad_mask = build_encoder_inputs(
        tokenizer, "Hello, how are you?", TASK_EN_VI)
    dec_details = decode_runner.get_input_details()
    n_kv_args   = sum(1 for k in dec_details
                      if k.startswith("args_") and int(k.split("_")[1]) >= 4)
    kvc_shape   = (1, SEQ_LEN, NUM_HEADS, HEAD_DIM)
    kvc         = {f"args_{4 + i}": np.zeros(kvc_shape, dtype=np.float32)
                   for i in range(n_kv_args)}

    # Warm-up
    for _ in range(3):
        enc_out = encode_runner(args_0=input_ids, args_1=input_pos, args_2=pad_mask)
        hs      = enc_out["output_0"]
        decode_runner(args_0=hs, args_1=np.array([[0]], dtype=np.int32),
                      args_2=np.array([0], dtype=np.int32), args_3=pad_mask, **kvc)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        enc_out = encode_runner(args_0=input_ids, args_1=input_pos, args_2=pad_mask)
    enc_ms = (time.perf_counter() - t0) * 1000 / n_runs
    hs     = enc_out["output_0"]

    t0 = time.perf_counter()
    for _ in range(n_runs):
        decode_runner(args_0=hs, args_1=np.array([[0]], dtype=np.int32),
                      args_2=np.array([0], dtype=np.int32), args_3=pad_mask, **kvc)
    dec_ms = (time.perf_counter() - t0) * 1000 / n_runs

    return enc_ms, dec_ms


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="t5_mini_explicit_int8.tflite")
    parser.add_argument("--runs",  type=int, default=20)
    args = parser.parse_args()

    model_path = LITERT_DIR / args.model
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found.\nRun  python litert/explicit/convert.py  first.")

    size_mb = model_path.stat().st_size / 1024 / 1024
    print(f"Model : {model_path.name}  ({size_mb:.1f} MB)  [explicit KV cache]")

    print("\nLoading HuggingFace reference …")
    hf_model  = T5ForConditionalGeneration.from_pretrained(str(MODEL_DIR)).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    encode_runner, decode_runner = load_tflite(str(model_path))

    print("\n" + "=" * 70)
    print("Correctness check")
    print("=" * 70)
    passed = 0
    for task_id, text in TEST_CASES:
        hf_out     = hf_translate(hf_model, tokenizer, text, task_id)
        litert_out = litert_translate(encode_runner, decode_runner, tokenizer, text, task_id)
        match  = hf_out.strip().lower() == litert_out.strip().lower()
        status = "PASS" if match else "DIFF"
        if match: passed += 1
        print(f"  [{status}] {DIRECTION[task_id]}  \"{text[:45]}\"")
        print(f"    LiteRT : {litert_out}")
        print(f"    HF     : {hf_out}")
    print(f"\n  Exact match: {passed}/{len(TEST_CASES)}")

    print("\n" + "=" * 70)
    print(f"Latency  ({args.runs} runs, seq_len={SEQ_LEN})")
    print("=" * 70)
    enc_ms, dec_ms = benchmark(encode_runner, decode_runner, tokenizer, args.runs)
    assumed = 20
    print(f"  Encoder  / call : {enc_ms:7.2f} ms")
    print(f"  Decoder  / step : {dec_ms:7.2f} ms")
    print(f"  Estimated total : {enc_ms + assumed * dec_ms:7.2f} ms"
          f"  (enc + {assumed} decode steps)")
    print("\nDone.")


if __name__ == "__main__":
    main()
