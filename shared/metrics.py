"""
Shared evaluation metrics for all T5 converter benchmarks.

Requires: sacrebleu
"""

import time
import sacrebleu
from .testset import TASK_EN_VI, TASK_VI_EN

DIR_LABEL = {TASK_EN_VI: "en→vi", TASK_VI_EN: "vi→en"}


def compute_metrics(hypotheses: list[str], references: list[str]) -> dict:
    """
    Corpus-level BLEU-4 and chrF++ (sacrebleu).
    tokenize='char' handles Vietnamese diacritics without a word tokenizer.
    chrF++ = chrF with beta=2, word_order=2.
    """
    bleu = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="char")
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], beta=2, word_order=2)
    exact = sum(
        h.strip().lower() == r.strip().lower()
        for h, r in zip(hypotheses, references)
    )
    return {
        "bleu4":  round(bleu.score, 2),
        "chrf++": round(chrf.score, 2),
        "exact":  exact,
        "n":      len(hypotheses),
    }


def compute_metrics_split(hypotheses: list[str], references: list[str],
                           task_ids: list[int]) -> dict:
    """Break metrics down by translation direction."""
    overall  = compute_metrics(hypotheses, references)
    env_idx  = [i for i, t in enumerate(task_ids) if t == TASK_EN_VI]
    vie_idx  = [i for i, t in enumerate(task_ids) if t == TASK_VI_EN]
    env = compute_metrics([hypotheses[i] for i in env_idx],
                          [references[i]  for i in env_idx])
    vie = compute_metrics([hypotheses[i] for i in vie_idx],
                          [references[i]  for i in vie_idx])
    return {"overall": overall, "en→vi": env, "vi→en": vie}


def run_and_score(name: str, translate_fn, tokenizer,
                  pairs: list, verbose: bool = False) -> dict:
    """
    Run translate_fn over pairs, compute metrics, return result dict.

    translate_fn signature: (tokenizer, src: str, task_id: int) -> str
    """
    print(f"  [{name}] translating {len(pairs)} sentences …")
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
            print(f"  [{DIR_LABEL[task_id]}] {src}")
            print(f"    ref : {ref}")
            print(f"    hyp : {hyps[i]}  {mark}")
        print()

    return metrics


def print_results_table(all_results: dict, title: str = "ACCURACY RESULTS",
                         test_pairs=None) -> None:
    """Print a formatted accuracy + speed table."""
    from .testset import ALL_PAIRS, TASK_EN_VI, TASK_VI_EN
    pairs = test_pairs or ALL_PAIRS
    models = list(all_results.keys())
    W = 22

    print()
    print("=" * 80)
    print(f"{title}  —  corpus-level sacrebleu (tokenize=char)")
    print(f"  Test set: {len(pairs)} sentences  "
          f"({sum(1 for t,_,_ in pairs if t==TASK_EN_VI)} en→vi  +  "
          f"{sum(1 for t,_,_ in pairs if t==TASK_VI_EN)} vi→en)")
    print("=" * 80)

    header = f"  {'Metric':<{W}}" + "".join(f"{m:>16}" for m in models)
    sep    = "  " + "-" * (W + 16 * len(models))

    def row(label, key, sub=None):
        vals = []
        for m in models:
            r = all_results[m]
            v = r[sub][key] if sub else r[key]
            if key == "exact":
                n = r[sub]["n"] if sub else r["overall"]["n"]
                vals.append(f"{v}/{n}")
            else:
                vals.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        print(f"  {label:<{W}}" + "".join(f"{v:>16}" for v in vals))

    for section, sub in [("── OVERALL", "overall"),
                          ("── EN→VI",  "en→vi"),
                          ("── VI→EN",  "vi→en")]:
        print(f"\n  {section}")
        print(header); print(sep)
        row("BLEU-4",      "bleu4",  sub)
        row("chrF++",      "chrf++", sub)
        row("Exact match", "exact",  sub)

    print(f"\n  ── SPEED")
    print(header); print(sep)
    row("Total time (s)",  "elapsed_s")
    row("ms / sentence",   "ms_per_sent")

    # Delta vs first model
    if len(models) > 1:
        ref_key = models[0]
        print(f"\n  ── DELTA  (vs {ref_key})")
        print(header); print(sep)
        for met in ["bleu4", "chrf++"]:
            label = f"Δ {met.upper().replace('BLEU4', 'BLEU-4').replace('CHRF++', 'chrF++')}"
            vals  = []
            for m in models:
                if m == ref_key:
                    vals.append("(ref)")
                else:
                    delta = all_results[m]["overall"][met] - all_results[ref_key]["overall"][met]
                    vals.append(f"{delta:>+.2f}")
            print(f"  {label:<{W}}" + "".join(f"{v:>16}" for v in vals))
    print()
