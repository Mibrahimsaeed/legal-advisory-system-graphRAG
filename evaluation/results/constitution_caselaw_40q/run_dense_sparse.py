#!/usr/bin/env python3
"""Dense-only and sparse-only retrieval for the 33 scored queries (20
Constitution + 13 case law), isolated inside this audit directory. Reuses
ONLY connect()/load_dense()/load_sparse()/encode_dense() from
src.rag_prep.embed_index, unmodified. Read-only Qdrant calls only
(query_points, no prefetch/fusion). Output written only inside this
directory.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.rag_prep.embed_index import COLLECTION, connect, encode_dense, load_dense, load_sparse

AUDIT_DIR = Path(__file__).resolve().parent
QUERIES = json.loads((AUDIT_DIR / "queries_scored_for_retrieval.json").read_text())
DEPTH = 20

def main():
    from qdrant_client import models
    client = connect()
    dense = load_dense()
    sparse = load_sparse()

    dense_out = AUDIT_DIR / "retrieval_dense.jsonl"
    sparse_out = AUDIT_DIR / "retrieval_sparse.jsonl"
    with dense_out.open("w") as df, sparse_out.open("w") as sf:
        for q in QUERIES:
            qid, qtext, corpus = q["query_id"], q["query"], q["corpus"]
            flt = models.Filter(must=[models.FieldCondition(key="source_type", match=models.MatchValue(value=corpus))])

            qd = encode_dense(dense, [qtext], 1)[0].tolist()
            dense_hits = client.query_points(COLLECTION, query=qd, using="dense", query_filter=flt, limit=DEPTH, with_payload=True).points
            for i, h in enumerate(dense_hits, 1):
                p = h.payload or {}
                df.write(json.dumps({"query_id": qid, "method": "dense", "rank": i, "chunk_id": p.get("chunk_id"),
                                      "doc_id": p.get("doc_id"), "source_type": p.get("source_type"), "score": h.score,
                                      "context_header": p.get("context_header")}, ensure_ascii=False) + "\n")

            qs_raw = next(iter(sparse.query_embed(qtext)))
            qs = models.SparseVector(indices=qs_raw.indices.tolist(), values=qs_raw.values.tolist())
            sparse_hits = client.query_points(COLLECTION, query=qs, using="bm25", query_filter=flt, limit=DEPTH, with_payload=True).points
            for i, h in enumerate(sparse_hits, 1):
                p = h.payload or {}
                sf.write(json.dumps({"query_id": qid, "method": "sparse", "rank": i, "chunk_id": p.get("chunk_id"),
                                      "doc_id": p.get("doc_id"), "source_type": p.get("source_type"), "score": h.score,
                                      "context_header": p.get("context_header")}, ensure_ascii=False) + "\n")
            print(f"{qid}: dense={len(dense_hits)} sparse={len(sparse_hits)}")

    print(f"-> {dense_out}\n-> {sparse_out}")


if __name__ == "__main__":
    main()
