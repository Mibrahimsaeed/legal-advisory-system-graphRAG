"""
Dense-only and sparse-only retrieval (top 40) for the 150-query family-law benchmark.
Reuses connect(), load_dense(), load_sparse(), encode_dense() from src.rag_prep.embed_index
UNMODIFIED. This script only issues single-vector (not fused) Qdrant queries, filtered by
source_type exactly as production hybrid_search() filters, at the same candidate depth (40).
Read-only: query_points() only, no upsert/delete/recreate.
"""
import json, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.rag_prep.embed_index import connect, load_dense, load_sparse, encode_dense, COLLECTION
from qdrant_client import models

OUT_DIR = Path(__file__).resolve().parent
CANDIDATES = 40
PAYLOAD_FIELDS = ("chunk_id", "context_header", "canonical_text")

def dense_search(client, dense_model, query, source_type, limit):
    qd = encode_dense(dense_model, [query], 1)[0].tolist()
    flt = models.Filter(must=[models.FieldCondition(key="source_type", match=models.MatchValue(value=source_type))])
    res = client.query_points(COLLECTION, query=qd, using="dense", query_filter=flt, limit=limit, with_payload=True).points
    return res

def sparse_search(client, sparse_model, query, source_type, limit):
    qs = next(iter(sparse_model.query_embed(query)))
    qs = models.SparseVector(indices=qs.indices.tolist(), values=qs.values.tolist())
    flt = models.Filter(must=[models.FieldCondition(key="source_type", match=models.MatchValue(value=source_type))])
    res = client.query_points(COLLECTION, query=qs, using="bm25", query_filter=flt, limit=limit, with_payload=True).points
    return res

def run_batch(corpus, queries_path, out_prefix):
    client = connect()
    dense = load_dense()
    sparse = load_sparse()
    rows = json.load(open(queries_path))
    dense_out, sparse_out = [], []
    for row in rows:
        qid, qtext = row["query_id"], row["query"]
        for rank, h in enumerate(dense_search(client, dense, qtext, corpus, CANDIDATES), 1):
            p = h.payload or {}
            rec = {"query_id": qid, "query": qtext, "source_type": corpus, "rank": rank,
                   "point_id": str(h.id), "score": h.score}
            for f in PAYLOAD_FIELDS: rec[f] = p.get(f)
            dense_out.append(rec)
        for rank, h in enumerate(sparse_search(client, sparse, qtext, corpus, CANDIDATES), 1):
            p = h.payload or {}
            rec = {"query_id": qid, "query": qtext, "source_type": corpus, "rank": rank,
                   "point_id": str(h.id), "score": h.score}
            for f in PAYLOAD_FIELDS: rec[f] = p.get(f)
            sparse_out.append(rec)
        print(f"  {qid}: dense={sum(1 for r in dense_out if r['query_id']==qid)} sparse={sum(1 for r in sparse_out if r['query_id']==qid)}")
    with open(OUT_DIR / f"retrieval_dense_{out_prefix}.jsonl", "w") as f:
        for r in dense_out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT_DIR / f"retrieval_sparse_{out_prefix}.jsonl", "w") as f:
        for r in sparse_out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"corpus": corpus, "candidates": CANDIDATES, "n_queries": len(rows),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dense_rows": len(dense_out), "sparse_rows": len(sparse_out)}
    json.dump(meta, open(OUT_DIR / f"retrieval_dense_sparse_{out_prefix}.meta.json", "w"), indent=2)
    print(corpus, "done:", meta)

if __name__ == "__main__":
    for corpus, prefix in [("constitution", "constitution"), ("statute", "statute"), ("case", "case")]:
        run_batch(corpus, OUT_DIR / f"queries_{corpus}_input.json", prefix)
