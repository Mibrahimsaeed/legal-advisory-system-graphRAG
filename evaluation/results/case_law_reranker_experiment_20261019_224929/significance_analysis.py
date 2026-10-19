"""
Phase E: paired query-level bootstrap comparison, baseline_hybrid vs. reranked_hybrid.
Seed=42, 5000 resamples, resampling at the query level (paired -- each resample redraws
the SAME set of query indices for both arms).
"""
import json, random
from pathlib import Path
from collections import Counter

NEW_DIR = Path(__file__).resolve().parent
rows = json.load(open(NEW_DIR / "metrics_per_query_raw.json"))
SEED = 42
N_RESAMPLES = 5000
random.seed(SEED)

METRICS = ["p_at_5", "ndcg_at_5", "ndcg_at_10", "recall_at_5", "recall_at_10",
           "mrr_at_5", "mrr_at_10", "hit_rate_at_5", "hit_rate_at_10"]

def paired_values(metric):
    a = {r["query_id"]: r[metric] for r in rows if r["method"] == "reranked_hybrid"}
    b = {r["query_id"]: r[metric] for r in rows if r["method"] == "baseline_hybrid"}
    qids = [q for q in a if a[q] is not None and b.get(q) is not None]
    return [(a[q], b[q]) for q in qids], qids

def bootstrap_diff(pairs, n_resamples=N_RESAMPLES):
    n = len(pairs)
    diffs = [a - b for a, b in pairs]
    obs = sum(diffs) / n
    boot = []
    for _ in range(n_resamples):
        sample = [diffs[random.randrange(n)] for _ in range(n)]
        boot.append(sum(sample) / n)
    boot.sort()
    lo = boot[int(0.025 * n_resamples)]
    hi = boot[int(0.975 * n_resamples)]
    return obs, lo, hi, n

results = {}
for metric in METRICS:
    pairs, qids = paired_values(metric)
    if not pairs:
        results[metric] = {"note": "no paired data"}
        continue
    a_vals = [p[0] for p in pairs]
    b_vals = [p[1] for p in pairs]
    mean_a = sum(a_vals) / len(a_vals)
    mean_b = sum(b_vals) / len(b_vals)
    obs, lo, hi, n = bootstrap_diff(pairs)
    sig = (lo > 0) or (hi < 0)
    rel_change = (obs / mean_b) if mean_b not in (0, None) else None

    improved = sum(1 for a, b in pairs if a > b)
    unchanged = sum(1 for a, b in pairs if a == b)
    worsened = sum(1 for a, b in pairs if a < b)

    results[metric] = {
        "n": n,
        "baseline_mean": round(mean_b, 4), "reranked_mean": round(mean_a, 4),
        "absolute_diff_reranked_minus_baseline": round(obs, 4),
        "relative_change_pct": round(100 * rel_change, 2) if rel_change is not None else None,
        "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
        "significant_at_5pct": sig,
        "n_queries_improved": improved, "n_queries_unchanged": unchanged, "n_queries_worsened": worsened,
    }
    print(metric, "baseline=", round(mean_b,4), "reranked=", round(mean_a,4),
          "diff=", round(obs,4), "CI=[", round(lo,4), ",", round(hi,4), "]", "sig=", sig,
          f"(+{improved}/={unchanged}/-{worsened})")

# per-query regression detail for the primary metric (nDCG@10)
regressions = []
pairs10, qids10 = paired_values("ndcg_at_10")
for (a, b), qid in zip(pairs10, qids10):
    if a < b:
        regressions.append({"query_id": qid, "baseline_ndcg10": b, "reranked_ndcg10": a, "delta": round(a-b, 4)})
regressions.sort(key=lambda x: x["delta"])

json.dump({
    "seed": SEED, "n_resamples": N_RESAMPLES,
    "procedure": "paired bootstrap over 50 case-law queries, resampling query indices, percentile CI",
    "results": results,
    "ndcg_at_10_regressions_detail": regressions,
}, open(NEW_DIR / "significance_analysis.json", "w"), indent=2)

print("\nnDCG@10 regressions (reranked worse than baseline):", len(regressions))
for r in regressions:
    print(" ", r)
