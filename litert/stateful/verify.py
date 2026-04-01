"""
Verify the stateful LiteRT model: correctness vs HF + latency benchmark.

The decode signature has NO k/v tensors. KV cache lives inside the TFLite
interpreter as resource variables. To reset between sentences, create a fresh
interpreter (new interpreter = zero-initialised resource variables).

Usage:
  source litert/.venv/bin/activate
  python litert/stateful/verify.py
  python litert/stateful/verify.py --enc t5_mini_stateful_enc_fp32.tflite \
                                   --dec t5_mini_stateful_dec_int8.tflite
  python litert/stateful/verify.py --runs 50
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
    pad_mask  = np.zeros(SEQ_LEN, dtype=np.float32)
    pad_mask[real_len:] = float("-inf")
    input_pos = np.arange(SEQ_LEN, dtype=np.int32)
    return input_ids, input_pos, pad_mask


def _make_interpreter(model_path: str):
    try:
        import ai_edge_litert.interpreter as m
        interp = m.Interpreter(model_path=model_path)
    except ImportError:
        import tflite_runtime.interpreter as m
        interp = m.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp


# ── HuggingFace reference ─────────────────────────────────────────────────────

def hf_translate(model, tokenizer, text: str, task_id: int) -> str:
    ids = [task_id] + tokenizer.encode(text, add_special_tokens=False)
    if ids[-1] != EOS_ID:
        ids.append(EOS_ID)
    with torch.no_grad():
        out = model.generate(torch.tensor([ids]), max_new_tokens=MAX_NEW_TOKENS)
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ── LiteRT inference — stateful (no k/v in signature) ────────────────────────

def litert_translate(enc_runner, dec_model_path: str, tokenizer,
                     text: str, task_id: int,
                     max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    input_ids, input_pos, pad_mask = build_encoder_inputs(tokenizer, text, task_id)

    enc_out       = enc_runner(input_ids=input_ids, input_pos=input_pos, pad_mask=pad_mask)
    hidden_states = enc_out["output_0"]

    # Fresh interpreter = zero-initialised KV cache
    dec_runner = _make_interpreter(dec_model_path).get_signature_runner("decode")

    generated = []
    token_id  = PAD_ID
    for step in range(max_new_tokens):
        dec_out    = dec_runner(
            encoder_hidden_states=hidden_states,
            decoder_input_ids=np.array([[token_id]], dtype=np.int32),
            step=np.array(step, dtype=np.int32),
            pad_mask=pad_mask,
        )
        next_token = int(np.argmax(dec_out["output_0"][0, 0, :]))
        if next_token == EOS_ID:
            break
        generated.append(next_token)
        token_id = next_token
        # No cache management — state lives inside dec_runner's interpreter

    return tokenizer.decode(generated, skip_special_tokens=True)


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark(enc_runner, dec_model_path: str, tokenizer, n_runs: int = 20):
    input_ids, input_pos, pad_mask = build_encoder_inputs(
        tokenizer, "Hello, how are you?", TASK_EN_VI)

    # Warm-up
    for _ in range(3):
        enc_out = enc_runner(input_ids=input_ids, input_pos=input_pos, pad_mask=pad_mask)
        hs      = enc_out["output_0"].copy()
        dr = _make_interpreter(dec_model_path).get_signature_runner("decode")
        dr(encoder_hidden_states=hs,
           decoder_input_ids=np.array([[0]], dtype=np.int32),
           step=np.array(0, dtype=np.int32),
           pad_mask=pad_mask)

    # Encoder
    t0 = time.perf_counter()
    for _ in range(n_runs):
        enc_out = enc_runner(input_ids=input_ids, input_pos=input_pos, pad_mask=pad_mask)
    enc_ms = (time.perf_counter() - t0) * 1000 / n_runs
    hs     = enc_out["output_0"].copy()

    # Decoder — use one persistent runner to measure per-step steady-state cost
    dr = _make_interpreter(dec_model_path).get_signature_runner("decode")
    dr(encoder_hidden_states=hs,
       decoder_input_ids=np.array([[0]], dtype=np.int32),
       step=np.array(0, dtype=np.int32),
       pad_mask=pad_mask)
    t0 = time.perf_counter()
    for s in range(1, n_runs + 1):
        dr(encoder_hidden_states=hs,
           decoder_input_ids=np.array([[0]], dtype=np.int32),
           step=np.array(s % SEQ_LEN, dtype=np.int32),
           pad_mask=pad_mask)
    dec_ms = (time.perf_counter() - t0) * 1000 / n_runs

    return enc_ms, dec_ms


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enc",  default="t5_mini_stateful_enc_fp32.tflite")
    parser.add_argument("--dec",  default="t5_mini_stateful_dec_fp32.tflite")
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()

    enc_path = LITERT_DIR / args.enc
    dec_path = LITERT_DIR / args.dec

    for p in [enc_path, dec_path]:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found.\nRun  python litert/stateful/convert.py  first.")

    enc_mb = enc_path.stat().st_size / 1024 / 1024
    dec_mb = dec_path.stat().st_size / 1024 / 1024
    print(f"Encoder : {enc_path.name}  ({enc_mb:.1f} MB)")
    print(f"Decoder : {dec_path.name}  ({dec_mb:.1f} MB)  [stateful VAR_HANDLE]")

    print("\nLoading HuggingFace reference …")
    hf_model  = T5ForConditionalGeneration.from_pretrained(str(MODEL_DIR)).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    enc_runner = _make_interpreter(str(enc_path)).get_signature_runner("encode")

    print("\n" + "=" * 70)
    print("Correctness check")
    print("=" * 70)
    passed = 0
    for task_id, text in TEST_CASES:
        hf_out     = hf_translate(hf_model, tokenizer, text, task_id)
        litert_out = litert_translate(enc_runner, str(dec_path), tokenizer, text, task_id)
        match  = hf_out.strip().lower() == litert_out.strip().lower()
        status = "PASS" if match else "DIFF"
        if match: passed += 1
        print(f"  [{status}] {DIRECTION[task_id]}  \"{text[:45]}\"")
        print(f"    Stateful LiteRT : {litert_out}")
        print(f"    HF              : {hf_out}")
    print(f"\n  Exact match: {passed}/{len(TEST_CASES)}")

    print("\n" + "=" * 70)
    print(f"Latency  ({args.runs} runs, seq_len={SEQ_LEN})")
    print("=" * 70)
    enc_ms, dec_ms = benchmark(enc_runner, str(dec_path), tokenizer, args.runs)
    assumed = 20
    print(f"  Encoder  / call : {enc_ms:7.2f} ms")
    print(f"  Decoder  / step : {dec_ms:7.2f} ms")
    print(f"  Estimated total : {enc_ms + assumed * dec_ms:7.2f} ms"
          f"  (enc + {assumed} decode steps)")
    print("\nDone.")


if __name__ == "__main__":
    main()
