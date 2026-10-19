#!/usr/bin/env python3
"""Read-only diagnostic: why is Family Courts Act 1964 S.10
(chunk_id 24520c344d2285555aa10988:s10) absent from Q01's top 10?

Reuses the EXISTING, unmodified src.rag_prep.embed_index functions
(connect, load_dense, load_sparse, encode_dense) -- never re-implements
embedding, chunking, or indexing. Performs three read-only operations
against the live Qdrant collection:

  1. points/scroll by chunk_id -- confirms the target chunk exists and
     inspects its stored payload.
  2. A dense-only query_points() call (using="dense", no prefetch, no
     fusion) with limit=240 (every retrievable statute chunk) -- finds
     the target's exact rank/score in a full, unfused dense ranking.
  3. A sparse-only query_points() call (using="bm25", no prefetch, no
     fusion) with limit=240 -- finds the target's exact rank/score in a
     full, unfused BM25 ranking, if it is returned at all.

No vector is written, no point is modified, and no embedding is
generated for anything other than the single Q01 query string (the same
query-time operation every search in this project already performs).

    python evaluation/diagnose_q01_dense_bm25.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rag_prep.embed_index import COLLECTION, connect, encode_dense, load_dense, load_sparse  # noqa: E402

QUERY = "Can a wife obtain khula without returning dower?"
TARGET_CHUNK_ID = "24520c344d2285555aa10988:s10"
N_STATUTE = 240  # every retrievable statute chunk -- a full ranking, not just top-k
OUT_PATH = Path("evaluation/results/q01_dense_vs_bm25_diagnostic.json")


def main() -> None:
    from qdrant_client import models

    client = connect()
    report: dict = {"query": QUERY, "target_chunk_id": TARGET_CHUNK_ID}

    # 1. Confirm existence + inspect payload (read-only scroll by chunk_id)
    scroll = client.scroll(
        COLLECTION,
        scroll_filter=models.Filter(must=[models.FieldCondition(key="chunk_id", match=models.MatchValue(value=TARGET_CHUNK_ID))]),
        limit=2,
        with_payload=True,
    )
    points = scroll[0]
    if not points:
        report["target_exists_in_qdrant"] = False
        print(f"[diagnose] TARGET NOT FOUND in Qdrant at all: {TARGET_CHUNK_ID}")
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return
    report["target_exists_in_qdrant"] = True
    report["target_point_id"] = str(points[0].id)
    report["target_payload"] = points[0].payload
    print(f"[diagnose] target exists: point_id={points[0].id}")
    print(f"[diagnose] target payload: {json.dumps(points[0].payload, indent=2)[:800]}")

    statute_filter = models.Filter(must=[models.FieldCondition(key="source_type", match=models.MatchValue(value="statute"))])

    # 2. Dense-only, full statute ranking
    dense = load_dense()
    qd = encode_dense(dense, [QUERY], 1)[0].tolist()
    dense_hits = client.query_points(
        COLLECTION, query=qd, using="dense", query_filter=statute_filter, limit=N_STATUTE, with_payload=True,
    ).points
    dense_rank = next((i for i, h in enumerate(dense_hits, 1) if h.payload.get("chunk_id") == TARGET_CHUNK_ID), None)
    dense_score = next((h.score for h in dense_hits if h.payload.get("chunk_id") == TARGET_CHUNK_ID), None)
    report["dense_only"] = {
        "target_rank": dense_rank,
        "target_score": dense_score,
        "n_candidates_ranked": len(dense_hits),
        "top10": [{"rank": i, "chunk_id": h.payload.get("chunk_id"), "score": h.score} for i, h in enumerate(dense_hits[:10], 1)],
    }
    print(f"\n[diagnose] DENSE-ONLY: target rank={dense_rank} score={dense_score} (out of {len(dense_hits)} statute chunks ranked)")
    for i, h in enumerate(dense_hits[:10], 1):
        print(f"  dense #{i}: score={h.score:.4f}  {h.payload.get('chunk_id')}")

    # 3. BM25-only (sparse), full statute ranking -- genuinely sparse-only, no fusion
    sparse = load_sparse()
    qs_raw = next(iter(sparse.query_embed(QUERY)))
    qs = models.SparseVector(indices=qs_raw.indices.tolist(), values=qs_raw.values.tolist())
    sparse_hits = client.query_points(
        COLLECTION, query=qs, using="bm25", query_filter=statute_filter, limit=N_STATUTE, with_payload=True,
    ).points
    sparse_rank = next((i for i, h in enumerate(sparse_hits, 1) if h.payload.get("chunk_id") == TARGET_CHUNK_ID), None)
    sparse_score = next((h.score for h in sparse_hits if h.payload.get("chunk_id") == TARGET_CHUNK_ID), None)
    report["bm25_only"] = {
        "target_rank": sparse_rank,
        "target_score": sparse_score,
        "n_candidates_ranked": len(sparse_hits),
        "top10": [{"rank": i, "chunk_id": h.payload.get("chunk_id"), "score": h.score} for i, h in enumerate(sparse_hits[:10], 1)],
    }
    print(f"\n[diagnose] BM25-ONLY: target rank={sparse_rank} score={sparse_score} (out of {len(sparse_hits)} statute chunks ranked)")
    for i, h in enumerate(sparse_hits[:10], 1):
        print(f"  bm25 #{i}: score={h.score:.4f}  {h.payload.get('chunk_id')}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[diagnose] full report -> {OUT_PATH}")


if __name__ == "__main__":
    main()
