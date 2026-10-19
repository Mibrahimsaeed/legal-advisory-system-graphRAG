#!/usr/bin/env python3
"""SRI Phase 3 -- dense-only, BM25-only, and fused RRF rankings for all 16
benchmark queries, against the verified qrels. Reuses ONLY connect()/
load_dense()/load_sparse()/encode_dense()/hybrid_search() from
src.rag_prep.embed_index, unmodified -- the same isolated-diagnostic
pattern already validated in this project's earlier Phase 1 scripts
(dense/BM25 branches via plain, unfused client.query_points(); fused via
the real, unmodified hybrid_search()). No prefetch/fusion code is
reimplemented; no Qdrant write of any kind occurs.

    python evaluation/sri_phase3_branch_diagnostics.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rag_prep.embed_index import COLLECTION, connect, encode_dense, hybrid_search, load_dense, load_sparse  # noqa: E402

QUERIES_PATH = Path("evaluation/queries_statute_benchmark16.json")
QRELS_PATH = Path("evaluation/results/sri_phase1_qrels.json")
N_STATUTE = 240
CANDIDATES = 40
OUT = Path("evaluation/results/sri_phase3_branch_diagnostics.json")


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite existing path: {OUT}")

    queries = json.loads(QUERIES_PATH.read_text())
    qrels = json.loads(QRELS_PATH.read_text())

    from qdrant_client import models
    client = connect()
    dense = load_dense()
    sparse = load_sparse()
    statute_filter = models.Filter(must=[models.FieldCondition(key="source_type", match=models.MatchValue(value="statute"))])

    report = []
    for q in queries:
        qid, qtext = q["query_id"], q["query"]
        relevant = set(qrels.get(qid, {}).get("statute", []))

        qd = encode_dense(dense, [qtext], 1)[0].tolist()
        dense_hits = client.query_points(COLLECTION, query=qd, using="dense", query_filter=statute_filter, limit=N_STATUTE, with_payload=True).points

        qs_raw = next(iter(sparse.query_embed(qtext)))
        qs = models.SparseVector(indices=qs_raw.indices.tolist(), values=qs_raw.values.tolist())
        bm25_hits = client.query_points(COLLECTION, query=qs, using="bm25", query_filter=statute_filter, limit=N_STATUTE, with_payload=True).points

        fused_hits = hybrid_search(client, dense, sparse, qtext, {"statute": N_STATUTE}, candidates=CANDIDATES)

        def rank_of(hits, cid):
            return next((i for i, h in enumerate(hits, 1) if h.payload.get("chunk_id") == cid), None)

        per_target = {}
        for cid in relevant:
            d_rank = rank_of(dense_hits, cid)
            b_rank = rank_of(bm25_hits, cid)
            f_rank = rank_of(fused_hits, cid)
            per_target[cid] = {
                "dense_rank": d_rank, "in_dense_top40": d_rank is not None and d_rank <= 40,
                "bm25_rank": b_rank, "in_bm25_top40": b_rank is not None and b_rank <= 40,
                "fused_rank": f_rank,
                "rrf_effect": (
                    "improved" if (f_rank is not None and d_rank is not None and b_rank is not None and f_rank < min(d_rank, b_rank))
                    else "harmed" if (f_rank is not None and d_rank is not None and b_rank is not None and f_rank > max(d_rank, b_rank))
                    else "mixed_or_unavailable"
                ),
            }

        top5_fused = [{"rank": i, "chunk_id": h.payload.get("chunk_id"), "score": h.score, "is_relevant": h.payload.get("chunk_id") in relevant}
                       for i, h in enumerate(fused_hits[:5], 1)]
        hard_negative_outranks = [h for h in top5_fused if not h["is_relevant"]]

        entry = {
            "query_id": qid, "query": qtext, "type": q.get("type"), "statute": q.get("statute"),
            "verified_relevant_chunk_ids": sorted(relevant),
            "per_target_branch_analysis": per_target,
            "fused_top5": top5_fused,
            "hard_negatives_in_top5": [h["chunk_id"] for h in hard_negative_outranks],
        }
        report.append(entry)
        print(f"[{qid}] {q.get('statute','')[:30]:30s} ", {cid: v["fused_rank"] for cid, v in per_target.items()})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "candidates": CANDIDATES,
                                "ranking_depth": N_STATUTE, "queries": report}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
