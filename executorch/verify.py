"""
Verify the ExecuTorch T5 model: correctness vs HF + latency benchmark.

The model.pte produced by convert.py exposes three methods:
  encoder(input_ids: int64[1, seq_len])
      → encoder_hidden_states: float32[1, seq_len, 384]

  text_decoder(decoder_input_ids: int64[1, 1],
               encoder_hidden_states: float32[1, seq_len, 384],
               cache_position: int64[1])
      → lm_logits: float32[1, 1, vocab_size]

  sampler(logits: float32[1, 1, vocab_size]) → next_token: int64[1]

The decoder holds a static KV cache.  Passing cache_position=[0] at the first
decode step resets the cache, so a single loaded module handles multiple sentences
without reloading the file.

Usage:
  source executorch/.venv/bin/activate
  python executorch/verify.py
  python executorch/verify.py --model int8/model.pte
  python executorch/verify.py --runs 50
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR  = Path(__file__).parents[2] / "translation" / "model_pt"
OUT_DIR    = Path(__file__).parent / "output"

SEQ_LEN        = 512
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


# ── Runtime loader ────────────────────────────────────────────────────────────

def _load_pte(model_path: str):
    """
    Load an ExecuTorch .pte via the Python pybindings.

    optimum-executorch pulls in executorch as a dependency; the pybindings module
    is executorch.extension.pybindings.portable_lib.  For XNNPACK-lowered models
    the XNNPACK delegate is invoked automatically when the runtime was built with
    XNNPACK support (the default pip wheel).
    """
    try:
        from executorch.extension.pybindings import portable_lib as _lib
    except ImportError as e:
        raise RuntimeError(
            "executorch Python bindings not found.\n"
            "Install with: pip install executorch  (or optimum-executorch)"
        ) from e
    return _lib._load_for_executorch(model_path)


# ── Input helpers ─────────────────────────────────────────────────────────────

def build_encoder_input(tokenizer, text: str, task_id: int) -> torch.Tensor:
    ids = [task_id] + tokenizer.encode(text, add_special_tokens=False)
    if ids[-1] != EOS_ID:
        ids.append(EOS_ID)
    real_len = min(len(ids), SEQ_LEN)
    input_ids = torch.zeros((1, SEQ_LEN), dtype=torch.long)
    input_ids[0, :real_len] = torch.tensor(ids[:real_len], dtype=torch.long)
    return input_ids


# ── HuggingFace reference ─────────────────────────────────────────────────────

def hf_translate(model, tokenizer, text: str, task_id: int) -> str:
    ids = [task_id] + tokenizer.encode(text, add_special_tokens=False)
    if ids[-1] != EOS_ID:
        ids.append(EOS_ID)
    with torch.no_grad():
        out = model.generate(torch.tensor([ids]), max_new_tokens=MAX_NEW_TOKENS)
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ── ExecuTorch inference ──────────────────────────────────────────────────────

def et_translate(pte_module, tokenizer, text: str, task_id: int,
                 max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    input_ids = build_encoder_input(tokenizer, text, task_id)

    # Encode
    enc_out           = pte_module.run_method("encoder", (input_ids,))
    encoder_hidden_states = enc_out[0]   # float32[1, SEQ_LEN, d_model]

    # Decode — cache_position=0 resets the static KV cache for this sentence
    generated = []
    token_id  = PAD_ID
    for step in range(max_new_tokens):
        dec_out = pte_module.run_method(
            "text_decoder",
            (
                torch.tensor([[token_id]], dtype=torch.long),
                encoder_hidden_states,
                torch.tensor([step], dtype=torch.long),
            ),
        )
        logits   = dec_out[0]  # float32[1, 1, vocab_size]
        next_tok = int(torch.argmax(logits[0, 0, :]).item())
        if next_tok == EOS_ID:
            break
        generated.append(next_tok)
        token_id = next_tok

    return tokenizer.decode(generated, skip_special_tokens=True)


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark(pte_module, tokenizer, n_runs: int = 20):
    input_ids = build_encoder_input(tokenizer, "Hello, how are you?", TASK_EN_VI)

    # Warm-up
    for _ in range(3):
        enc_out = pte_module.run_method("encoder", (input_ids,))
        hs = enc_out[0]
        pte_module.run_method(
            "text_decoder",
            (torch.tensor([[0]], dtype=torch.long), hs, torch.tensor([0], dtype=torch.long)),
        )

    # Encoder latency
    t0 = time.perf_counter()
    for _ in range(n_runs):
        enc_out = pte_module.run_method("encoder", (input_ids,))
    enc_ms = (time.perf_counter() - t0) * 1000 / n_runs
    hs = enc_out[0]

    # Decoder latency — steady-state per-step cost (step > 0 to use cached state)
    pte_module.run_method(
        "text_decoder",
        (torch.tensor([[0]], dtype=torch.long), hs, torch.tensor([0], dtype=torch.long)),
    )
    t0 = time.perf_counter()
    for s in range(1, n_runs + 1):
        pte_module.run_method(
            "text_decoder",
            (
                torch.tensor([[0]], dtype=torch.long),
                hs,
                torch.tensor([s % SEQ_LEN], dtype=torch.long),
            ),
        )
    dec_ms = (time.perf_counter() - t0) * 1000 / n_runs

    return enc_ms, dec_ms


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="fp32/model.pte",
                        help="Path relative to executorch/output/ or absolute")
    parser.add_argument("--runs",  type=int, default=20)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = OUT_DIR / model_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found.\nRun  python executorch/convert.py  first."
        )

    quant_label = "INT8" if "int8" in str(model_path) else "FP32"
    size_mb = model_path.stat().st_size / 1024 / 1024
    print(f"Model : {model_path}  ({size_mb:.1f} MB)  [ExecuTorch {quant_label}, XNNPACK]")

    print("\nLoading ExecuTorch module …")
    pte_module = _load_pte(str(model_path))

    print("Loading HuggingFace reference …")
    hf_model  = T5ForConditionalGeneration.from_pretrained(str(MODEL_DIR)).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    print("\n" + "=" * 70)
    print("Correctness check")
    print("=" * 70)
    passed = 0
    for task_id, text in TEST_CASES:
        hf_out = hf_translate(hf_model, tokenizer, text, task_id)
        et_out = et_translate(pte_module, tokenizer, text, task_id)
        match  = hf_out.strip().lower() == et_out.strip().lower()
        status = "PASS" if match else "DIFF"
        if match: passed += 1
        print(f"  [{status}] {DIRECTION[task_id]}  \"{text[:45]}\"")
        print(f"    ExecuTorch : {et_out}")
        print(f"    HF         : {hf_out}")
    print(f"\n  Exact match: {passed}/{len(TEST_CASES)}")

    print("\n" + "=" * 70)
    print(f"Latency  ({args.runs} runs, seq_len={SEQ_LEN})")
    print("=" * 70)
    enc_ms, dec_ms = benchmark(pte_module, tokenizer, args.runs)
    assumed = 20
    print(f"  Encoder  / call : {enc_ms:7.2f} ms")
    print(f"  Decoder  / step : {dec_ms:7.2f} ms")
    print(f"  Estimated total : {enc_ms + assumed * dec_ms:7.2f} ms"
          f"  (enc + {assumed} decode steps)")
    print("\nDone.")


if __name__ == "__main__":
    main()
