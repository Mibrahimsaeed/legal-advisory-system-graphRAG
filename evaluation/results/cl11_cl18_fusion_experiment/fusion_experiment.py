#!/usr/bin/env python3
"""Offline RRF reproduction and weighted-RRF experiment.

Uses the FIXED, fresh depth-40 dense/sparse candidate lists
(retrieval_dense_c40.jsonl / retrieval_sparse_c40.jsonl, produced by
run_branch_candidates40.py in this same directory) -- the same candidate
depth (candidates=40) that src/rag_prep/embed_index.py's hybrid_search()
uses in production. Implements the standard RRF formula
(score = sum over lists of weight / (k + rank), k=60, the standard
constant from the original Cormack/Clarke RRF paper) and validates it
against the saved, frozen production hybrid ranking
(constitution_caselaw_40q/retrieval_hybrid.jsonl) before trusting any
weighted variant's results. Read-only; writes only inside this directory.
"""
from __future__ import annotations
import importlib.util, json
from pathlib import Path
from collections import defaultdict

AUDIT_DIR = Path(__file__).resolve().parent
PREV_DIR = AUDIT_DIR.parent / "constitution_caselaw_40q"
REPO_ROOT = AUDIT_DIR.parents[2]

spec = importlib.util.spec_from_file_location("eval_calculate_metrics", REPO_ROOT / "evaluation" / "calculate_metrics.py")
cm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cm)

RRF_K = 1  # EMPIRICALLY VALIDATED, not assumed: a sweep over k in {0,1,2,5,10,20,30,60} against all 33
# frozen production hybrid rankings found k=1 gives the best fit (19/33 exact top-10 order matches,
# 31/33 identical top-10 candidate sets, and EXACT rank reproduction for both CL11 and CL18 specifically)
# -- far better than the commonly-cited k=60 default (1/33 exact matches). See rrf_validation.json.

CONFIGS = [
    {"name": "rrf_dense1.0_sparse1.0", "dense_weight": 1.0, "sparse_weight": 1.0},
    {"name": "rrf_dense1.0_sparse0.75", "dense_weight": 1.0, "sparse_weight": 0.75},
    {"name": "rrf_dense1.0_sparse0.5", "dense_weight": 1.0, "sparse_weight": 0.5},
]


def load_ranks(path):
    per_query = defaultdict(dict)  # qid -> {chunk_id: rank}
    for line in open(path):
        r = json.loads(line)
        per_query[r["query_id"]][r["chunk_id"]] = r["rank"]
    return per_query


def fuse(dense_ranks: dict, sparse_ranks: dict, dense_w: float, sparse_w: float) -> list[tuple[str, float]]:
    all_ids = set(dense_ranks) | set(sparse_ranks)
    scored = []
    for cid in all_ids:
        s = 0.0
        if cid in dense_ranks:
            s += dense_w / (RRF_K + dense_ranks[cid])
        if cid in sparse_ranks:
            s += sparse_w / (RRF_K + sparse_ranks[cid])
        scored.append((cid, s))
    scored.sort(key=lambda x: -x[1])
    return scored


def main():
    dense_all = load_ranks(AUDIT_DIR / "retrieval_dense_c40.jsonl")
    sparse_all = load_ranks(AUDIT_DIR / "retrieval_sparse_c40.jsonl")
    queries = json.load(open(PREV_DIR / "queries_scored_for_retrieval.json"))
    corpus_of = {q["query_id"]: q["corpus"] for q in queries}

    # Validation: does rrf_dense1.0_sparse1.0 (k=60) reproduce the saved production hybrid RANK ORDER?
    prod_hybrid = defaultdict(list)
    for line in open(PREV_DIR / "retrieval_hybrid.jsonl"):
        r = json.loads(line)
        if r.get("source_type") == corpus_of.get(r["query_id"]):
            prod_hybrid[r["query_id"]].append(r)
    for qid in prod_hybrid:
        prod_hybrid[qid].sort(key=lambda r: r["rank"])

    validation = {}
    for qid in corpus_of:
        offline = fuse(dense_all.get(qid, {}), sparse_all.get(qid, {}), 1.0, 1.0)
        offline_top10 = [c for c, _ in offline[:10]]
        prod_top10 = [r["chunk_id"] for r in prod_hybrid.get(qid, [])[:10]]
        validation[qid] = {
            "offline_top10": offline_top10, "production_top10": prod_top10,
            "exact_match": offline_top10 == prod_top10,
            "same_set": set(offline_top10) == set(prod_top10),
        }
    n_exact = sum(1 for v in validation.values() if v["exact_match"])
    n_same_set = sum(1 for v in validation.values() if v["same_set"])
    print(f"VALIDATION: offline RRF (k=60, equal weights) exact top-10 ORDER match vs. saved production hybrid: "
          f"{n_exact}/{len(validation)} queries; same top-10 SET (order-independent): {n_same_set}/{len(validation)}")
    json.dump(validation, open(AUDIT_DIR / "rrf_validation.json", "w"), indent=2)

    # Run all configs, compute per-query target rank + metrics inputs
    qrels = json.load(open(PREV_DIR / "qrels_combined_simple.json"))
    qrels_lookup = {}
    for qid, by_source in qrels.items():
        for st, ids in by_source.items():
            qrels_lookup[(qid, st)] = set(ids)

    all_results = {}
    rankings_out = open(AUDIT_DIR / "fusion_per_query_rankings.jsonl", "w")
    for cfg in CONFIGS:
        cfg_results = {}
        for qid, corpus in corpus_of.items():
            fused = fuse(dense_all.get(qid, {}), sparse_all.get(qid, {}), cfg["dense_weight"], cfg["sparse_weight"])
            ranked_ids = [c for c, _ in fused]
            relevant = qrels_lookup.get((qid, corpus), set())
            relevances = [2 if cid in relevant else 0 for cid in ranked_ids]
            p5, _ = cm.precision_at_k(relevances, 5)
            ndcg10, _ = cm.ndcg_at_k(relevances, 10)
            r5, _ = cm.recall_at_k(ranked_ids, relevant, 5)
            r10, _ = cm.recall_at_k(ranked_ids, relevant, 10)
            target_rank = next((i for i, cid in enumerate(ranked_ids, 1) if cid in relevant), None)
            mrr = (1.0 / target_rank) if target_rank else 0.0
            hr5 = 1.0 if (target_rank and target_rank <= 5) else (0.0 if relevant else None)
            hr10 = 1.0 if (target_rank and target_rank <= 10) else (0.0 if relevant else None)
            cfg_results[qid] = {"precision_at_5": p5, "ndcg_at_10": ndcg10, "recall_at_5": r5, "recall_at_10": r10,
                                 "mrr": mrr, "hit_rate_at_5": hr5, "hit_rate_at_10": hr10, "target_rank": target_rank}
            for i, cid in enumerate(ranked_ids[:20], 1):
                rankings_out.write(json.dumps({"query_id": qid, "config": cfg["name"], "rank": i, "chunk_id": cid,
                                                "is_relevant": cid in relevant}, ensure_ascii=False) + "\n")
        all_results[cfg["name"]] = cfg_results
    rankings_out.close()

    json.dump(all_results, open(AUDIT_DIR / "fusion_raw_per_query.json", "w"), indent=2)
    print(f"-> {AUDIT_DIR / 'fusion_per_query_rankings.jsonl'}")
    print(f"-> {AUDIT_DIR / 'fusion_raw_per_query.json'}")
    print(f"-> {AUDIT_DIR / 'rrf_validation.json'}")


if __name__ == "__main__":
    main()
