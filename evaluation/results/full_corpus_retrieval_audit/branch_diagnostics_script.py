#!/usr/bin/env python3
"""Full-corpus audit -- dense-only/BM25-only/fused branch diagnostics for all
28 benchmark queries, generalized across case/statute/constitution. Reuses
ONLY connect()/load_dense()/load_sparse()/encode_dense()/hybrid_search()
from src.rag_prep.embed_index, unmodified -- same isolated-diagnostic
pattern already validated in sri_phase3_branch_diagnostics.py. Read-only.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.rag_prep.embed_index import COLLECTION, connect, encode_dense, hybrid_search, load_dense, load_sparse

AUDIT_DIR = Path(__file__).resolve().parent
QUERIES_PATH = AUDIT_DIR / "queries_combined28.json"
QRELS_PATH = AUDIT_DIR / "qrels_combined28.json"
DEPTH = {"case": 3000, "statute": 240, "constitution": 547}  # full retrievable count for statute/constitution;
# case law capped at 3000 (of 25,891) for tractability -- documented as a depth cap, not a full-corpus rank
CANDIDATES = 40
OUT = AUDIT_DIR / "branch_diagnostics.json"


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite: {OUT}")
    queries = json.loads(QUERIES_PATH.read_text())
    qrels = json.loads(QRELS_PATH.read_text())
    from qdrant_client import models
    client = connect()
    dense = load_dense()
    sparse = load_sparse()

    report = []
    for q in queries:
        qid, qtext = q["query_id"], q["query"]
        st = q.get("corpus") or "statute"
        relevant = set(qrels.get(qid, {}).get(st, []))
        if not relevant:
            continue
        depth = DEPTH[st]
        flt = models.Filter(must=[models.FieldCondition(key="source_type", match=models.MatchValue(value=st))])

        qd = encode_dense(dense, [qtext], 1)[0].tolist()
        dense_hits = client.query_points(COLLECTION, query=qd, using="dense", query_filter=flt, limit=depth, with_payload=True).points
        qs_raw = next(iter(sparse.query_embed(qtext)))
        qs = models.SparseVector(indices=qs_raw.indices.tolist(), values=qs_raw.values.tolist())
        bm25_hits = client.query_points(COLLECTION, query=qs, using="bm25", query_filter=flt, limit=depth, with_payload=True).points
        fused_hits = hybrid_search(client, dense, sparse, qtext, {st: depth}, candidates=CANDIDATES)

        def rank_of(hits, cid):
            return next((i for i, h in enumerate(hits, 1) if h.payload.get("chunk_id") == cid), None)

        per_target = {}
        for cid in relevant:
            d_rank, b_rank, f_rank = rank_of(dense_hits, cid), rank_of(bm25_hits, cid), rank_of(fused_hits, cid)
            per_target[cid] = {
                "dense_rank": d_rank, "in_dense_top40": d_rank is not None and d_rank <= 40,
                "bm25_rank": b_rank, "in_bm25_top40": b_rank is not None and b_rank <= 40,
                "fused_rank": f_rank, "fused_in_top10": f_rank is not None and f_rank <= 10,
            }
        entry = {"query_id": qid, "corpus": st, "query": qtext, "per_target": per_target}
        report.append(entry)
        print(f"[{qid}/{st}]", {cid: v["fused_rank"] for cid, v in per_target.items()})

    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
