"""
Phase D: evaluate baseline hybrid vs. reranked hybrid against the FROZEN expanded qrels
(case_law_qrels_expansion_20261019_220106/judgment_ledger.jsonl), reusing
calculate_metrics.py's pure precision_at_k/ndcg_at_k/recall_at_k functions unmodified.

Grading convention matches the qrels-expansion task exactly: relevance is a document-level
(case_key) property, propagated to every chunk of that case encountered in a ranking --
this resolves the "right case, wrong chunk" undercount for a multi-chunk case-law corpus,
and is the SAME convention already used to compute the reported baseline numbers this
experiment is checked against.

Unjudged treatment: a candidate's case is "unjudged" if it is not present at all in the
judgment ledger (relevance_grade is null there). It is NEVER coerced to 0. Where this would
fall within a metric's ranking window, P@k/nDCG@k are reported unavailable for that query,
exactly as calculate_metrics.py's own semantics already enforce.
"""
import json, importlib.util, csv
from pathlib import Path
from collections import defaultdict

NEW_DIR = Path(__file__).resolve().parent
QEXP_DIR = NEW_DIR.parents[0] / "case_law_qrels_expansion_20261019_220106"
REPO_ROOT = NEW_DIR.parents[2]

spec = importlib.util.spec_from_file_location("calc_metrics", REPO_ROOT / "evaluation" / "calculate_metrics.py")
calc_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calc_metrics)
precision_at_k, ndcg_at_k, recall_at_k = calc_metrics.precision_at_k, calc_metrics.ndcg_at_k, calc_metrics.recall_at_k

# ---- load the frozen judgment ledger (document-level grades, incl. explicit unjudged) ----
ledger = [json.loads(l) for l in open(QEXP_DIR / "judgment_ledger.jsonl")]
grade_by_query_case = defaultdict(dict)
for r in ledger:
    if r["relevance_grade"] is not None:
        grade_by_query_case[r["query_id"]][r["case_id"]] = r["relevance_grade"]
relevant_sets = {qid: {ck for ck, g in cases.items() if g >= 1} for qid, cases in grade_by_query_case.items()}

def mrr_at_k(ranked, relevant_set, k):
    for i, x in enumerate(ranked[:k], 1):
        if x in relevant_set:
            return 1.0 / i
    return 0.0

def hit_rate_at_k(ranked, relevant_set, k):
    return 1.0 if set(ranked[:k]) & relevant_set else 0.0

# ---- load baseline (original hybrid) and reranked rankings ----
baseline_rows = defaultdict(list)
for line in open(NEW_DIR / "baseline_hybrid_rankings.jsonl"):
    r = json.loads(line)
    baseline_rows[r["query_id"]].append(r)
for qid in baseline_rows:
    baseline_rows[qid].sort(key=lambda r: r["original_hybrid_rank"])

reranked_rows = defaultdict(list)
for line in open(NEW_DIR / "reranked_rankings.jsonl"):
    r = json.loads(line)
    reranked_rows[r["query_id"]].append(r)
for qid in reranked_rows:
    reranked_rows[qid].sort(key=lambda r: r["reranked_rank"])

qids = sorted(relevant_sets.keys(), key=lambda x: int(x[2:]))

def eval_method(qid, chunk_ids, case_ids, relevant_set, judged_cases):
    """Chunk-level AND document-level (case-id-propagated) views."""
    chunk_relevances = [judged_cases.get(ck) for ck in case_ids]   # document-grade applied per chunk-rank
    p5, p5note = precision_at_k(chunk_relevances, 5)
    p_at5_5, _ = precision_at_k(chunk_relevances, 5)  # alias, kept for clarity
    ndcg5, ndcg5note = ndcg_at_k(chunk_relevances, 5)
    ndcg10, ndcg10note = ndcg_at_k(chunk_relevances, 10)
    r5, _ = recall_at_k(case_ids, relevant_set, 5)
    r10, _ = recall_at_k(case_ids, relevant_set, 10)
    mrr5 = mrr_at_k(case_ids, relevant_set, 5)
    mrr10 = mrr_at_k(case_ids, relevant_set, 10)
    hr5 = hit_rate_at_k(case_ids, relevant_set, 5)
    hr10 = hit_rate_at_k(case_ids, relevant_set, 10)
    return {
        "p_at_5": p5, "p_at_5_note": p5note,
        "ndcg_at_5": ndcg5, "ndcg_at_5_note": ndcg5note,
        "ndcg_at_10": ndcg10, "ndcg_at_10_note": ndcg10note,
        "recall_at_5": r5, "recall_at_10": r10,
        "mrr_at_5": mrr5, "mrr_at_10": mrr10,
        "hit_rate_at_5": hr5, "hit_rate_at_10": hr10,
    }

per_query_rows = []
for qid in qids:
    relevant_set = relevant_sets[qid]
    judged_cases = grade_by_query_case[qid]

    b_rows = baseline_rows.get(qid, [])
    b_chunk_ids = [r["chunk_id"] for r in b_rows]
    b_case_ids = [r["case_id"] for r in b_rows]
    b_metrics = eval_method(qid, b_chunk_ids, b_case_ids, relevant_set, judged_cases)
    b_metrics.update({"query_id": qid, "method": "baseline_hybrid", "n_candidates": len(b_rows),
                      "n_known_relevant_total": len(relevant_set)})
    per_query_rows.append(b_metrics)

    rr_rows = reranked_rows.get(qid, [])
    rr_chunk_ids = [r["chunk_id"] for r in rr_rows]
    rr_case_ids = [r["case_id"] for r in rr_rows]
    rr_metrics = eval_method(qid, rr_chunk_ids, rr_case_ids, relevant_set, judged_cases)
    rr_metrics.update({"query_id": qid, "method": "reranked_hybrid", "n_candidates": len(rr_rows),
                       "n_known_relevant_total": len(relevant_set)})
    per_query_rows.append(rr_metrics)

json.dump(per_query_rows, open(NEW_DIR / "metrics_per_query_raw.json", "w"), indent=2)

fieldnames = ["query_id", "method", "n_candidates", "n_known_relevant_total",
              "p_at_5", "ndcg_at_5", "ndcg_at_10", "recall_at_5", "recall_at_10",
              "mrr_at_5", "mrr_at_10", "hit_rate_at_5", "hit_rate_at_10",
              "p_at_5_note", "ndcg_at_5_note", "ndcg_at_10_note"]
with open(NEW_DIR / "metrics_per_query.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    w.writerows(per_query_rows)

def avg(vals):
    vals = [v for v in vals if v is not None]
    return (sum(vals)/len(vals), len(vals)) if vals else (None, 0)

summary = {}
for method in ("baseline_hybrid", "reranked_hybrid"):
    subset = [r for r in per_query_rows if r["method"] == method]
    entry = {"n_eligible_queries": len(subset)}
    for metric in ("p_at_5", "ndcg_at_5", "ndcg_at_10", "recall_at_5", "recall_at_10",
                   "mrr_at_5", "mrr_at_10", "hit_rate_at_5", "hit_rate_at_10"):
        m, n = avg([r[metric] for r in subset])
        entry[metric] = {"macro_avg": round(m, 4) if m is not None else None, "denominator": n,
                          "n_missing_or_unavailable": len(subset) - n}
    summary[method] = entry

json.dump({
    "grading": "document-level (case_id), propagated from judged case to every chunk of that case",
    "qrels_source": "evaluation/results/case_law_qrels_expansion_20261019_220106/judgment_ledger.jsonl (frozen, read-only)",
    "n_queries": len(qids),
    "candidate_depth": 40,
    "methods": summary,
}, open(NEW_DIR / "metrics_summary.json", "w"), indent=2)

print(json.dumps(summary, indent=2))
