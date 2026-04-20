"""
Evaluate ONNX T5 on 50 opus-100 en-vi samples (ET-only style, no HF reference).

Usage:
  source onnx/.venv/bin/activate
  python onnx/eval_phomt.py
  python onnx/eval_phomt.py --samples 50 --dir output/int8
"""

import argparse
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

MODEL_DIR = Path(__file__).parents[2] / "translation" / "model_pt"
ONNX_DIR  = Path(__file__).parent / "output" / "fp32"

MAX_NEW_TOKENS = 128
TASK_EN_VI     = 20000
TASK_VI_EN     = 20001
EOS_ID         = 1
PAD_ID         = 0


# ── Sessions ──────────────────────────────────────────────────────────────────

def load_sessions(onnx_dir: Path, num_threads: int = 4):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.inter_op_num_threads = num_threads
    opts.intra_op_num_threads = num_threads
    enc = ort.InferenceSession(str(onnx_dir / "encoder_model.onnx"), opts)
    dec = ort.InferenceSession(str(onnx_dir / "decoder_model.onnx"), opts)
    dwp = ort.InferenceSession(str(onnx_dir / "decoder_with_past_model.onnx"), opts)
    return enc, dec, dwp


# ── Inference ─────────────────────────────────────────────────────────────────

def onnx_translate(enc_sess, dec_sess, dwp_sess, tokenizer, text: str, task_id: int) -> str:
    ids = [task_id] + tokenizer.encode(text, add_special_tokens=False)
    if ids[-1] != EOS_ID:
        ids.append(EOS_ID)
    input_ids  = np.array([ids], dtype=np.int64)
    attn_mask  = np.ones_like(input_ids, dtype=np.int64)

    enc_hs = enc_sess.run(None, {"input_ids": input_ids, "attention_mask": attn_mask})[0]

    dec_out = dec_sess.run(None, {
        "encoder_hidden_states":  enc_hs,
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

    for _ in range(MAX_NEW_TOKENS - 1):
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


# ── BLEU ──────────────────────────────────────────────────────────────────────

def compute_bleu(hypotheses, references):
    try:
        from sacrebleu.metrics import BLEU
        bleu = BLEU(effective_order=True)
        return round(bleu.corpus_score(hypotheses, [references]).score, 2)
    except ImportError:
        from collections import Counter
        import math
        total_match = total_hyp = 0
        for hyp, ref in zip(hypotheses, references):
            h, r = hyp.lower().split(), ref.lower().split()
            ref_counts = Counter(r)
            total_match += sum(min(c, ref_counts[w]) for w, c in Counter(h).items())
            total_hyp   += len(h)
        return round((total_match / total_hyp if total_hyp else 0) * 100, 2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",     default=str(ONNX_DIR))
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    onnx_dir = Path(args.dir)
    total_mb = sum((onnx_dir / f).stat().st_size for f in
                   ["encoder_model.onnx", "decoder_model.onnx", "decoder_with_past_model.onnx"]
                   ) / 1024 / 1024
    print(f"ONNX dir : {onnx_dir}  ({total_mb:.1f} MB total)")
    print(f"Samples  : {args.samples} per direction  ({args.samples*2} total)\n")

    print("Loading ONNX sessions …")
    enc_sess, dec_sess, dwp_sess = load_sessions(onnx_dir, args.threads)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    from datasets import load_dataset
    import random
    ds = load_dataset("Helsinki-NLP/opus-100", "en-vi", split="test")
    random.seed(args.seed)
    indices = random.sample(range(len(ds)), args.samples)
    samples = [ds[i]["translation"] for i in indices]

    directions = [
        ("en→vi", TASK_EN_VI, "en", "vi"),
        ("vi→en", TASK_VI_EN, "vi", "en"),
    ]

    total_cases = 0
    t0_all = time.perf_counter()

    for label, task_id, src_lang, tgt_lang in directions:
        hyps, refs = [], []

        print(f"{'='*60}")
        print(f"Direction: {label}  ({args.samples} samples)")
        print(f"{'='*60}")

        for i, pair in enumerate(samples):
            src = pair[src_lang]
            ref = pair[tgt_lang]

            t0 = time.perf_counter()
            out = onnx_translate(enc_sess, dec_sess, dwp_sess, tokenizer, src, task_id)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            hyps.append(out)
            refs.append(ref)

            if i < 5:
                print(f"  src  : {src[:60]}")
                print(f"  ONNX : {out[:70]}  ({elapsed_ms:.0f}ms)")
                print()

        bleu = compute_bleu(hyps, refs)
        print(f"  BLEU vs reference : ONNX={bleu}")
        print()
        total_cases += args.samples

    elapsed = time.perf_counter() - t0_all
    print(f"{'='*60}")
    print(f"Total time : {elapsed:.1f}s  ({elapsed/total_cases*1000:.0f}ms/sample)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
