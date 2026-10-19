"""Paired bootstrap CIs for the principal per-corpus method-difference comparisons.
Seed fixed for reproducibility. Procedure: resample query indices with replacement
(same resample applied to both methods being compared, i.e. paired), recompute the
macro-average difference each resample, report the 2.5/97.5 percentile CI.
"""
import json, random
from pathlib import Path

AUDIT_DIR = Path(__file__).resolve().parent
rows = json.load(open(AUDIT_DIR / "metrics_per_query_raw.json"))
SEED = 42
N_RESAMPLES = 5000
random.seed(SEED)

def paired_values(corpus, method_a, method_b, metric):
    a = {r["query_id"]: r[metric] for r in rows if r["corpus"]==corpus and r["method"]==method_a}
    b = {r["query_id"]: r[metric] for r in rows if r["corpus"]==corpus and r["method"]==method_b}
    qids = [q for q in a if a[q] is not None and b.get(q) is not None]
    return [(a[q], b[q]) for q in qids], qids

def bootstrap_diff(pairs, n_resamples=N_RESAMPLES):
    n = len(pairs)
    diffs = [a-b for a,b in pairs]
    obs = sum(diffs)/n
    boot = []
    for _ in range(n_resamples):
        sample = [diffs[random.randrange(n)] for _ in range(n)]
        boot.append(sum(sample)/n)
    boot.sort()
    lo = boot[int(0.025*n_resamples)]
    hi = boot[int(0.975*n_resamples)]
    return obs, lo, hi, n

comparisons = [
    ("constitution", "hybrid", "dense", "ndcg_at_10"),
    ("constitution", "hybrid", "sparse", "ndcg_at_10"),
    ("statute", "hybrid", "dense", "ndcg_at_10"),
    ("statute", "hybrid", "sparse", "ndcg_at_10"),
    ("case", "hybrid", "dense", "doc_recall_at_10"),
    ("case", "hybrid", "sparse", "doc_recall_at_10"),
    ("case", "sparse", "dense", "doc_recall_at_10"),
    ("case", "hybrid", "dense", "doc_mrr_at_10"),
    ("case", "hybrid", "sparse", "doc_mrr_at_10"),
]

results = []
for corpus, ma, mb, metric in comparisons:
    pairs, qids = paired_values(corpus, ma, mb, metric)
    if not pairs:
        results.append({"corpus":corpus,"a":ma,"b":mb,"metric":metric,"n":0,"note":"no paired data"})
        continue
    obs, lo, hi, n = bootstrap_diff(pairs)
    sig = (lo > 0) or (hi < 0)
    results.append({
        "corpus": corpus, "a": ma, "b": mb, "metric": metric, "n": n,
        "observed_diff_a_minus_b": round(obs,4), "ci95_lo": round(lo,4), "ci95_hi": round(hi,4),
        "significant_at_5pct": sig,
    })
    print(corpus, ma, "vs", mb, metric, "diff=", round(obs,4), "CI=[",round(lo,4),",",round(hi,4),"]", "sig=",sig)

json.dump({"seed": SEED, "n_resamples": N_RESAMPLES, "procedure": "paired bootstrap over queries, percentile CI",
           "results": results}, open(AUDIT_DIR / "bootstrap_ci_results.json","w"), indent=2)
