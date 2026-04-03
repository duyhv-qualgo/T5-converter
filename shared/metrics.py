"""
Shared evaluation metrics for all T5 converter benchmarks.

Requires: sacrebleu
Optional: bert-score (--bertscore), unbabel-comet (--comet)
"""

import time
import sacrebleu
from .testset import TASK_EN_VI, TASK_VI_EN

DIR_LABEL = {TASK_EN_VI: "en→vi", TASK_VI_EN: "vi→en"}


def compute_metrics(hypotheses: list[str], references: list[str],
                    sources: list[str] | None = None,
                    comet_model=None,
                    use_bertscore: bool = False,
                    lang: str = "en") -> dict:
    """
    Corpus-level BLEU-4, chrF++, exact match.
    Optionally adds BERTScore F1 and COMET if requested.
    tokenize='char' handles Vietnamese diacritics without a word tokenizer.
    chrF++ = chrF with beta=2, word_order=2.
    """
    bleu  = sacrebleu.corpus_bleu(hypotheses, [references], tokenize="char")
    chrf  = sacrebleu.corpus_chrf(hypotheses, [references], beta=2, word_order=2)
    exact = sum(
        h.strip().lower() == r.strip().lower()
        for h, r in zip(hypotheses, references)
    )
    result = {
        "bleu4":  round(bleu.score, 2),
        "chrf++": round(chrf.score, 2),
        "exact":  exact,
        "n":      len(hypotheses),
    }

    if use_bertscore and len(hypotheses) > 0:
        from bert_score import score as bs_score
        _, _, F1 = bs_score(
            hypotheses, references,
            model_type="bert-base-multilingual-cased",
            lang=lang, verbose=False, device="cpu",
        )
        result["bertscore"] = round(F1.mean().item() * 100, 2)

    if comet_model is not None and sources is not None and len(hypotheses) > 0:
        data = [{"src": s, "mt": h, "ref": r}
                for s, h, r in zip(sources, hypotheses, references)]
        output = comet_model.predict(data, batch_size=8, gpus=0, progress_bar=False)
        result["comet"] = round(output.system_score * 100, 2)

    return result


def compute_metrics_split(hypotheses: list[str], references: list[str],
                           task_ids: list[int],
                           sources: list[str] | None = None,
                           comet_model=None,
                           use_bertscore: bool = False) -> dict:
    """Break metrics down by translation direction."""
    overall = compute_metrics(hypotheses, references, sources,
                              comet_model, use_bertscore, lang="en")

    env_idx = [i for i, t in enumerate(task_ids) if t == TASK_EN_VI]
    vie_idx = [i for i, t in enumerate(task_ids) if t == TASK_VI_EN]

    env_hyps = [hypotheses[i] for i in env_idx]
    env_refs = [references[i]  for i in env_idx]
    env_srcs = [sources[i] for i in env_idx] if sources else None

    vie_hyps = [hypotheses[i] for i in vie_idx]
    vie_refs = [references[i]  for i in vie_idx]
    vie_srcs = [sources[i] for i in vie_idx] if sources else None

    env = compute_metrics(env_hyps, env_refs, env_srcs,
                          comet_model, use_bertscore, lang="vi")
    vie = compute_metrics(vie_hyps, vie_refs, vie_srcs,
                          comet_model, use_bertscore, lang="en")

    return {"overall": overall, "en→vi": env, "vi→en": vie}


def run_and_score(name: str, translate_fn, tokenizer,
                  pairs: list, verbose: bool = False,
                  sources: list[str] | None = None,
                  comet_model=None,
                  use_bertscore: bool = False) -> dict:
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
    metrics  = compute_metrics_split(hyps, refs, task_ids, sources,
                                     comet_model, use_bertscore)
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
    pairs  = test_pairs or ALL_PAIRS
    models = list(all_results.keys())
    W = 22

    # Detect which optional metrics are present
    sample = next(iter(all_results.values()))
    has_bertscore = "bertscore" in sample.get("overall", {})
    has_comet     = "comet"     in sample.get("overall", {})

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
            v = (r[sub][key] if sub else r[key]) if key in (r.get(sub) or r) else "—"
            if key == "exact" and v != "—":
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
        if has_bertscore:
            row("BERTScore F1", "bertscore", sub)
        if has_comet:
            row("COMET",        "comet",     sub)

    print(f"\n  ── SPEED")
    print(header); print(sep)
    row("Total time (s)",  "elapsed_s")
    row("ms / sentence",   "ms_per_sent")

    # Delta vs first model
    if len(models) > 1:
        ref_key = models[0]
        print(f"\n  ── DELTA  (vs {ref_key})")
        print(header); print(sep)
        delta_metrics = ["bleu4", "chrf++"]
        if has_bertscore:
            delta_metrics.append("bertscore")
        if has_comet:
            delta_metrics.append("comet")
        labels = {"bleu4": "Δ BLEU-4", "chrf++": "Δ chrF++",
                  "bertscore": "Δ BERTScore F1", "comet": "Δ COMET"}
        for met in delta_metrics:
            vals = []
            for m in models:
                if m == ref_key:
                    vals.append("(ref)")
                else:
                    ref_val = all_results[ref_key]["overall"].get(met)
                    cur_val = all_results[m]["overall"].get(met)
                    if ref_val is not None and cur_val is not None:
                        vals.append(f"{cur_val - ref_val:>+.2f}")
                    else:
                        vals.append("—")
            print(f"  {labels[met]:<{W}}" + "".join(f"{v:>16}" for v in vals))
    print()
