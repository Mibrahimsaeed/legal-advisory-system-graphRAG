import json, random
from pathlib import Path

NEW_DIR = Path(__file__).resolve().parent
rows = json.load(open(NEW_DIR / "metrics_per_query_expanded_raw.json"))
SEED = 42
N_RESAMPLES = 5000
random.seed(SEED)

def paired_values(method_a, method_b, metric):
    a = {r["query_id"]: r[metric] for r in rows if r["method"] == method_a}
    b = {r["query_id"]: r[metric] for r in rows if r["method"] == method_b}
    qids = [q for q in a if a[q] is not None and b.get(q) is not None]
    return [(a[q], b[q]) for q in qids]

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

comparisons = [
    ("hybrid", "dense", "ndcg_at_10"), ("hybrid", "sparse", "ndcg_at_10"), ("sparse", "dense", "ndcg_at_10"),
    ("hybrid", "dense", "recall_at_10"), ("hybrid", "sparse", "recall_at_10"), ("sparse", "dense", "recall_at_10"),
    ("hybrid", "dense", "mrr_at_10"), ("hybrid", "sparse", "mrr_at_10"),
    ("hybrid", "dense", "p_at_5"), ("hybrid", "sparse", "p_at_5"),
]

results = []
for ma, mb, metric in comparisons:
    pairs = paired_values(ma, mb, metric)
    obs, lo, hi, n = bootstrap_diff(pairs)
    sig = (lo > 0) or (hi < 0)
    results.append({"a": ma, "b": mb, "metric": metric, "n": n,
                     "observed_diff_a_minus_b": round(obs, 4), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
                     "significant_at_5pct": sig})
    print(ma, "vs", mb, metric, "diff=", round(obs, 4), "CI=[", round(lo, 4), ",", round(hi, 4), "]", "sig=", sig)

json.dump({"seed": SEED, "n_resamples": N_RESAMPLES, "procedure": "paired bootstrap over 50 case-law queries, percentile CI",
           "results": results}, open(NEW_DIR / "bootstrap_ci_results.json", "w"), indent=2)
