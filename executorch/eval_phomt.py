"""
Evaluate ExecuTorch T5 against HuggingFace on 50 opus-100 en-vi samples.

Reports per-direction exact-match rate and BLEU score (via sacrebleu if
available, otherwise corpus-level n-gram overlap).

Usage:
  source executorch/.venv/bin/activate
  python executorch/eval_phomt.py
  python executorch/eval_phomt.py --samples 100 --model fp32/model.pte
"""

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

MODEL_DIR  = Path(__file__).parents[2] / "translation" / "model_pt"
OUT_DIR    = Path(__file__).parent / "output"

SEQ_LEN        = 128
MAX_NEW_TOKENS = 128
TASK_EN_VI     = 20000
TASK_VI_EN     = 20001
EOS_ID         = 1
PAD_ID         = 0


# ── Runtime ───────────────────────────────────────────────────────────────────

def _load_pte(model_path: str):
    from executorch.extension.pybindings import portable_lib as _lib
    return _lib._load_for_executorch(model_path)


# ── Input builder ─────────────────────────────────────────────────────────────

def build_encoder_input(tokenizer, text: str, task_id: int):
    ids = [task_id] + tokenizer.encode(text, add_special_tokens=False)
    if ids[-1] != EOS_ID:
        ids.append(EOS_ID)
    real_len = min(len(ids), SEQ_LEN)
    input_ids = torch.zeros((1, SEQ_LEN), dtype=torch.long)
    input_ids[0, :real_len] = torch.tensor(ids[:real_len], dtype=torch.long)
    return input_ids


# ── HF inference ──────────────────────────────────────────────────────────────

def hf_translate(model, tokenizer, text: str, task_id: int) -> str:
    ids = [task_id] + tokenizer.encode(text, add_special_tokens=False)
    if ids[-1] != EOS_ID:
        ids.append(EOS_ID)
    with torch.no_grad():
        out = model.generate(torch.tensor([ids]), max_new_tokens=MAX_NEW_TOKENS)
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ── ExecuTorch inference ──────────────────────────────────────────────────────

def et_translate(pte, tokenizer, text: str, task_id: int) -> str:
    input_ids = build_encoder_input(tokenizer, text, task_id)
    enc_mask  = (input_ids != PAD_ID).long()

    t0 = time.perf_counter()
    enc_hs = pte.run_method("encoder", (input_ids,))[0]
    enc_ms = (time.perf_counter() - t0) * 1000

    generated = []
    token_id  = PAD_ID
    t0 = time.perf_counter()
    for step in range(MAX_NEW_TOKENS):
        logits = pte.run_method(
            "text_decoder",
            (
                torch.tensor([[token_id]], dtype=torch.long),
                enc_hs,
                torch.tensor([step], dtype=torch.long),
                enc_mask,
            ),
        )[0]
        next_tok = int(torch.argmax(logits[0, 0]).item())
        if next_tok == EOS_ID:
            break
        generated.append(next_tok)
        token_id = next_tok
    dec_ms = (time.perf_counter() - t0) * 1000
    n_steps = len(generated)

    print(f"         ET timing: enc={enc_ms:.0f}ms  dec={dec_ms:.0f}ms ({n_steps} steps, {dec_ms/n_steps if n_steps else 0:.1f}ms/step)")
    return tokenizer.decode(generated, skip_special_tokens=True)


# ── BLEU ──────────────────────────────────────────────────────────────────────

def simple_bleu(hypotheses, references):
    """Corpus-level BLEU-1 (unigram) as fallback when sacrebleu not available."""
    from collections import Counter
    import math
    total_match = total_hyp = 0
    for hyp, ref in zip(hypotheses, references):
        h = hyp.lower().split()
        r = ref.lower().split()
        ref_counts = Counter(r)
        match = sum(min(c, ref_counts[w]) for w, c in Counter(h).items())
        total_match += match
        total_hyp   += len(h)
    prec = total_match / total_hyp if total_hyp else 0
    return round(prec * 100, 2)


def compute_bleu(hypotheses, references):
    try:
        from sacrebleu.metrics import BLEU
        bleu = BLEU(effective_order=True)
        result = bleu.corpus_score(hypotheses, [references])
        return round(result.score, 2)
    except ImportError:
        return simple_bleu(hypotheses, references)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="fp32/model.pte")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--et-only", action="store_true", help="Skip HF reference, only run ET")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = OUT_DIR / model_path

    print(f"Model : {model_path}  ({model_path.stat().st_size/1024/1024:.1f} MB)")
    print(f"Samples: {args.samples} per direction  ({args.samples*2} total)\n")

    print("Loading models …")
    pte       = _load_pte(str(model_path))
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    hf_model  = None
    if not args.et_only:
        hf_model = T5ForConditionalGeneration.from_pretrained(str(MODEL_DIR)).eval()

    from datasets import load_dataset
    ds = load_dataset("Helsinki-NLP/opus-100", "en-vi", split="test")

    import random
    random.seed(args.seed)
    indices = random.sample(range(len(ds)), args.samples)
    samples = [ds[i]["translation"] for i in indices]

    directions = [
        ("en→vi", TASK_EN_VI, "en", "vi"),
        ("vi→en", TASK_VI_EN, "vi", "en"),
    ]

    total_exact = 0
    total_cases = 0
    t0_all = time.perf_counter()

    for label, task_id, src_lang, tgt_lang in directions:
        et_hyps, hf_hyps, refs = [], [], []
        exact = 0

        print(f"{'='*60}")
        print(f"Direction: {label}  ({args.samples} samples)")
        print(f"{'='*60}")

        for i, pair in enumerate(samples):
            src  = pair[src_lang]
            ref  = pair[tgt_lang]

            t_et = time.perf_counter()
            et_out = et_translate(pte, tokenizer, src, task_id)
            t_et = time.perf_counter() - t_et

            hf_out = None
            if not args.et_only:
                t_hf = time.perf_counter()
                hf_out = hf_translate(hf_model, tokenizer, src, task_id)
                t_hf = time.perf_counter() - t_hf

            match = (hf_out is not None and hf_out.strip().lower() == et_out.strip().lower())
            if match:
                exact += 1

            et_hyps.append(et_out)
            if hf_out is not None:
                hf_hyps.append(hf_out)
            refs.append(ref)

            if i < 5:
                print(f"  src : {src[:60]}")
                print(f"  ET  : {et_out[:70]}  ({t_et*1000:.0f}ms)")
                if hf_out is not None:
                    print(f"  HF  : {hf_out[:70]}  ({t_hf*1000:.0f}ms)")
                print()

        et_bleu = compute_bleu(et_hyps, refs)
        print(f"  BLEU vs reference    : ET={et_bleu}")
        if not args.et_only:
            hf_bleu = compute_bleu(hf_hyps, refs)
            et_vs_hf_bleu = compute_bleu(et_hyps, hf_hyps)
            print(f"  Exact match (ET==HF) : {exact}/{args.samples}  ({exact/args.samples*100:.1f}%)")
            print(f"  BLEU vs reference    : HF={hf_bleu}")
            print(f"  BLEU ET vs HF        : {et_vs_hf_bleu}")
        print()

        total_exact += exact
        total_cases += args.samples

    elapsed = time.perf_counter() - t0_all
    print(f"{'='*60}")
    if not args.et_only:
        print(f"Overall exact match : {total_exact}/{total_cases}  ({total_exact/total_cases*100:.1f}%)")
    print(f"Total time          : {elapsed:.1f}s  ({elapsed/total_cases*1000:.0f}ms/sample)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
