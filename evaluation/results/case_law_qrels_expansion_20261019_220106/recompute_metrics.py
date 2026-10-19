"""
Phase 4: recalculate case-law metrics using the expanded qrels (document-level grading,
since the judgment unit is the case, not a single chunk -- a document's relevance should
apply to any of ITS chunks encountered in a ranking, resolving the chunk-vs-document
granularity problem flagged in the original benchmark). Reuses the EXACT SAME frozen
dense/sparse/hybrid rankings from the original 150-query benchmark -- retrieval itself is
unchanged; only the relevance judgments feeding the metrics have been expanded.
"""
import json, sqlite3, importlib.util, csv
from pathlib import Path
from collections import defaultdict

NEW_DIR = Path(__file__).resolve().parent
OLD_DIR = NEW_DIR.parents[0] / "final_family_law_150_query_benchmark_20261019_212630"
REPO_ROOT = NEW_DIR.parents[2]

spec = importlib.util.spec_from_file_location("calc_metrics", REPO_ROOT / "evaluation" / "calculate_metrics.py")
calc_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calc_metrics)
precision_at_k, ndcg_at_k, recall_at_k = calc_metrics.precision_at_k, calc_metrics.ndcg_at_k, calc_metrics.recall_at_k

con = sqlite3.connect(REPO_ROOT / "var/rag/docstore.sqlite")
cur = con.cursor()
_case_key_cache = {}
def get_case_key(chunk_id):
    if chunk_id not in _case_key_cache:
        cur.execute("select json_extract(json,'$.case_key') from chunks where chunk_id=?", (chunk_id,))
        row = cur.fetchone()
        _case_key_cache[chunk_id] = row[0] if row else None
    return _case_key_cache[chunk_id]

# Build the FULL judged grade (0/1/2) and the relevant (grade>=1) sets per query, document-level.
ledger = [json.loads(l) for l in open(NEW_DIR / "judgment_ledger.jsonl")]
grade_by_query_case = defaultdict(dict)   # qid -> {case_key: grade}  (only where grade is not None)
for r in ledger:
    if r["relevance_grade"] is not None:
        grade_by_query_case[r["query_id"]][r["case_id"]] = r["relevance_grade"]

relevant_sets = {qid: {ck for ck, g in cases.items() if g >= 1} for qid, cases in grade_by_query_case.items()}

def mrr_at_10(ranked_case_keys, relevant_set):
    for i, ck in enumerate(ranked_case_keys[:10], 1):
        if ck in relevant_set:
            return 1.0 / i
    return 0.0

def hit_rate_at_k(ranked_case_keys, relevant_set, k):
    return 1.0 if set(ranked_case_keys[:k]) & relevant_set else 0.0

def load_rankings(path):
    out = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        out[r["query_id"]].append((r["rank"], r["chunk_id"]))
    for qid in out:
        out[qid] = [cid for _, cid in sorted(out[qid])]
    return out

methods = {
    "dense": load_rankings(OLD_DIR / "retrieval_dense_case.jsonl"),
    "sparse": load_rankings(OLD_DIR / "retrieval_sparse_case.jsonl"),
}
# hybrid file has k=1 throwaway rows for other corpora -- filter to source_type == case
hybrid_raw = defaultdict(list)
for line in open(OLD_DIR / "retrieval_hybrid_case.jsonl"):
    r = json.loads(line)
    if r.get("source_type") == "case":
        hybrid_raw[r["query_id"]].append((r["rank"], r["chunk_id"]))
methods["hybrid"] = {qid: [cid for _, cid in sorted(v)] for qid, v in hybrid_raw.items()}

qids = sorted(relevant_sets.keys(), key=lambda x: int(x[2:]))
per_query_rows = []
for qid in qids:
    relevant_set = relevant_sets[qid]
    judged_cases = grade_by_query_case[qid]
    for method in ("dense", "sparse", "hybrid"):
        ranked_chunks = methods[method].get(qid, [])
        ranked_case_keys = [get_case_key(c) for c in ranked_chunks]
        # document-level grading: a chunk's grade = its case's grade if judged, else None (unjudged)
        relevances = [judged_cases.get(ck) for ck in ranked_case_keys]
        p5, p5note = precision_at_k(relevances, 5)
        ndcg10, ndcgnote = ndcg_at_k(relevances, 10)
        r5, r5note = recall_at_k(ranked_case_keys, relevant_set, 5)
        r10, r10note = recall_at_k(ranked_case_keys, relevant_set, 10)
        mrr = mrr_at_10(ranked_case_keys, relevant_set)
        hr5 = hit_rate_at_k(ranked_case_keys, relevant_set, 5)
        hr10 = hit_rate_at_k(ranked_case_keys, relevant_set, 10)
        per_query_rows.append({
            "query_id": qid, "method": method, "n_retrieved": len(ranked_chunks),
            "n_known_relevant_total": len(relevant_set),
            "p_at_5": p5, "p_at_5_note": p5note,
            "ndcg_at_10": ndcg10, "ndcg_at_10_note": ndcgnote,
            "recall_at_5": r5, "recall_at_10": r10,
            "mrr_at_10": mrr, "hit_rate_at_5": hr5, "hit_rate_at_10": hr10,
        })

json.dump(per_query_rows, open(NEW_DIR / "metrics_per_query_expanded_raw.json", "w"), indent=2)

with open(NEW_DIR / "metrics_per_query.csv", "w", newline="") as f:
    fieldnames = list(per_query_rows[0].keys())
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(per_query_rows)

def avg(vals):
    vals = [v for v in vals if v is not None]
    return (sum(vals)/len(vals), len(vals)) if vals else (None, 0)

summary = {}
for method in ("dense", "sparse", "hybrid"):
    subset = [r for r in per_query_rows if r["method"] == method]
    entry = {"n_eligible_queries": len(subset)}
    for metric in ("p_at_5", "ndcg_at_10", "recall_at_5", "recall_at_10", "mrr_at_10", "hit_rate_at_5", "hit_rate_at_10"):
        m, n = avg([r[metric] for r in subset])
        entry[metric] = {"macro_avg": round(m, 4) if m is not None else None, "denominator": n, "n_missing": len(subset) - n}
    summary[method] = entry

json.dump({"corpus": "case", "grading": "document-level (case_key)", "corpora": {"case": summary}},
          open(NEW_DIR / "metrics_summary.json", "w"), indent=2)

print(json.dumps(summary, indent=2))
