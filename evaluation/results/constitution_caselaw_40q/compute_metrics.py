#!/usr/bin/env python3
"""Computes P@5, Recall@5, Recall@10, MRR, nDCG@10, HitRate@5, HitRate@10 for
all 3 retrieval methods (dense, sparse, hybrid) against the frozen qrels.
Reuses evaluation/calculate_metrics.py's pure functions unmodified (loaded
by file path, not by package import, consistent with this project's
existing workaround for the src/evaluation package-name collision). Writes
output only inside this audit directory.
"""
from __future__ import annotations
import importlib.util, json
from pathlib import Path
from collections import defaultdict

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parents[2]

spec = importlib.util.spec_from_file_location("eval_calculate_metrics", REPO_ROOT / "evaluation" / "calculate_metrics.py")
cm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cm)

qrels = json.load(open(AUDIT_DIR / "qrels_combined_simple.json"))
qrels_lookup = {}
for qid, by_source in qrels.items():
    for st, ids in by_source.items():
        qrels_lookup[(qid, st)] = set(ids)

queries = json.load(open(AUDIT_DIR / "queries_scored_for_retrieval.json"))
corpus_of = {q["query_id"]: q["corpus"] for q in queries}


def hit_rate_at_k(relevant_set, ranked_ids, k):
    if not relevant_set:
        return None
    return 1.0 if relevant_set.intersection(ranked_ids[:k]) else 0.0


def mrr_fn(relevant_set, ranked_ids):
    for i, cid in enumerate(ranked_ids, 1):
        if cid in relevant_set:
            return 1.0 / i
    return 0.0


def load_method(path, method_name):
    per_query = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        per_query[r["query_id"]].append(r)
    for qid in per_query:
        per_query[qid].sort(key=lambda r: r["rank"])
    return per_query


methods = {
    "hybrid": None,  # built from retrieval_hybrid.jsonl, filtered to own-corpus
    "dense": load_method(AUDIT_DIR / "retrieval_dense.jsonl", "dense"),
    "sparse": load_method(AUDIT_DIR / "retrieval_sparse.jsonl", "sparse"),
}

hybrid_raw = defaultdict(list)
for line in open(AUDIT_DIR / "retrieval_hybrid.jsonl"):
    r = json.loads(line)
    if r.get("source_type") == corpus_of.get(r["query_id"]):
        hybrid_raw[r["query_id"]].append(r)
for qid in hybrid_raw:
    hybrid_raw[qid].sort(key=lambda r: r["rank"])
methods["hybrid"] = hybrid_raw

results = {"constitution": {}, "case": {}}
per_query_detail = {"constitution": {}, "case": {}}
for method_name, per_query in methods.items():
    for st in ("constitution", "case"):
        results[st].setdefault(method_name, {})
        per_query_detail[st].setdefault(method_name, {})
        for qid, corpus in corpus_of.items():
            if corpus != st:
                continue
            rows = per_query.get(qid, [])
            ranked_ids = [r["chunk_id"] for r in rows]
            relevant = qrels_lookup.get((qid, st), set())
            relevances = [2 if cid in relevant else 0 for cid in ranked_ids]

            p5, p5note = cm.precision_at_k(relevances, 5)
            ndcg10, ndcgnote = cm.ndcg_at_k(relevances, 10)
            r5, r5note = cm.recall_at_k(ranked_ids, relevant, 5)
            r10, r10note = cm.recall_at_k(ranked_ids, relevant, 10)
            mrr = mrr_fn(relevant, ranked_ids)
            hr5 = hit_rate_at_k(relevant, ranked_ids, 5)
            hr10 = hit_rate_at_k(relevant, ranked_ids, 10)
            target_rank = next((i for i, cid in enumerate(ranked_ids, 1) if cid in relevant), None)

            per_query_detail[st][method_name][qid] = {
                "precision_at_5": p5, "ndcg_at_10": ndcg10, "recall_at_5": r5, "recall_at_10": r10,
                "mrr": mrr, "hit_rate_at_5": hr5, "hit_rate_at_10": hr10, "target_rank": target_rank,
                "n_relevant_in_qrels": len(relevant), "n_results": len(ranked_ids),
            }

for st in ("constitution", "case"):
    for method_name in methods:
        vals = per_query_detail[st][method_name]
        import statistics
        macro = {}
        for m in ("precision_at_5", "ndcg_at_10", "recall_at_5", "recall_at_10", "mrr", "hit_rate_at_5", "hit_rate_at_10"):
            xs = [v[m] for v in vals.values() if v[m] is not None]
            macro[m] = round(statistics.mean(xs), 4) if xs else None
        macro["n_queries"] = len(vals)
        results[st][method_name] = macro

out = {"macro": results, "per_query": per_query_detail}
json.dump(out, open(AUDIT_DIR / "retrieval_metrics.json", "w"), indent=2)
print(json.dumps(results, indent=2))
print(f"\n-> {AUDIT_DIR / 'retrieval_metrics.json'}")
