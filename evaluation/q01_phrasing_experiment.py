#!/usr/bin/env python3
"""Controlled Q01 query-phrasing experiment: does rewording change whether
Family Courts Act 1964 S.10 (chunk_id 24520c344d2285555aa10988:s10) is
retrieved, under dense-only, BM25-only, and the existing production hybrid
retrieval -- all three held to the same statute filter and the same
candidate depth, varying only the query text.

Reuses the EXISTING, unmodified src.rag_prep.embed_index functions
(connect, load_dense, load_sparse, encode_dense, hybrid_search) --
dense-only and BM25-only are built with the same unfused query_points()
pattern already used and validated in evaluation/diagnose_q01_dense_bm25.py;
hybrid uses the real, unmodified hybrid_search() function, not a
re-implementation of RRF.

Each (query, method) pair issues exactly one query_points() call with
limit=240 (every retrievable statute chunk) rather than limit=10 -- this
is the same choice already made and validated in the prior Q01 diagnostic:
the top 10 of a limit=240 call is identical to what a limit=10 call would
return (Qdrant's ranking is deterministic), so one call per pair gives
both "is it in the top 10" and "what is its exact rank if not" without
doubling the number of calls. The hybrid candidate-pool depth (`candidates`,
i.e. the per-source prefetch size before RRF fusion) stays fixed at the
production default (40) for every query/method -- only the *requested
final limit* is widened to 240 for inspection, exactly mirroring the
already-accepted methodology in the earlier Q04 diagnostic.

    python evaluation/q01_phrasing_experiment.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rag_prep.embed_index import (  # noqa: E402
    COLLECTION,
    connect,
    encode_dense,
    hybrid_search,
    load_dense,
    load_sparse,
)

QUERIES = [
    ("Q0", "Can a wife obtain khula without returning dower?"),
    ("Q1", "Under the Family Courts Act, what happens to Haq Mehr in a suit for dissolution of marriage?"),
    ("Q2", "What does section 10 of the Family Courts Act say about Haq Mehr in a suit for dissolution of marriage?"),
    ("Q3", "When a marriage is dissolved through the Family Court, can the wife be required to return her Haq Mehr?"),
]

TARGET_CHUNK_ID = "24520c344d2285555aa10988:s10"
N_STATUTE = 240  # every retrievable statute chunk -- see module docstring for why one call suffices
CANDIDATES = 40  # production default prefetch depth for hybrid_search -- held fixed across every run
METHODS = ("dense_only", "bm25_only", "hybrid")

DEFAULT_OUT_PATH = Path("evaluation/results/q01_phrasing_experiment.json")


def resolve_out_path(requested: Path) -> Path:
    """Never overwrites an existing file; a prior run's output is renamed
    aside with a unique, timestamped filename instead."""

    if not requested.exists():
        return requested
    stamped = requested.with_name(f"{requested.stem}_{time.strftime('%Y%m%dT%H%M%S')}{requested.suffix}")
    print(f"[q01_phrasing] {requested} already exists; writing to {stamped} instead")
    return stamped


def run_dense_only(client, dense_model, query_text: str, statute_filter):
    qd = encode_dense(dense_model, [query_text], 1)[0].tolist()
    hits = client.query_points(
        COLLECTION, query=qd, using="dense", query_filter=statute_filter, limit=N_STATUTE, with_payload=True,
    ).points
    return hits


def run_bm25_only(client, sparse_model, query_text: str, statute_filter):
    from qdrant_client import models

    qs_raw = next(iter(sparse_model.query_embed(query_text)))
    qs = models.SparseVector(indices=qs_raw.indices.tolist(), values=qs_raw.values.tolist())
    hits = client.query_points(
        COLLECTION, query=qs, using="bm25", query_filter=statute_filter, limit=N_STATUTE, with_payload=True,
    ).points
    return hits


def run_hybrid(client, dense_model, sparse_model, query_text: str):
    # hybrid_search() applies its own source_type filter internally via per_source's keys
    hits = hybrid_search(client, dense_model, sparse_model, query_text, {"statute": N_STATUTE}, candidates=CANDIDATES)
    return hits


def summarize(hits, method: str) -> dict:
    target_rank = next((i for i, h in enumerate(hits, 1) if h.payload.get("chunk_id") == TARGET_CHUNK_ID), None)
    target_score = next((h.score for h in hits if h.payload.get("chunk_id") == TARGET_CHUNK_ID), None)
    top10 = [
        {
            "rank": i,
            "point_id": str(h.id),
            "chunk_id": h.payload.get("chunk_id"),
            "score": h.score,
            "context_header": h.payload.get("context_header"),
            "canonical_text_preview": (h.payload.get("canonical_text") or "")[:160],
        }
        for i, h in enumerate(hits[:10], 1)
    ]
    return {
        "method": method,
        "n_candidates_ranked": len(hits),
        "target_rank": target_rank,
        "target_score": target_score,
        "target_in_top10": target_rank is not None and target_rank <= 10,
        "top10": top10,
    }


def main() -> None:
    from qdrant_client import models

    out_path = resolve_out_path(DEFAULT_OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = connect()
    dense = load_dense()
    sparse = load_sparse()
    statute_filter = models.Filter(must=[models.FieldCondition(key="source_type", match=models.MatchValue(value="statute"))])

    report: dict = {
        "target_chunk_id": TARGET_CHUNK_ID,
        "candidates_depth_hybrid": CANDIDATES,
        "ranking_depth": N_STATUTE,
        "source_filter": "statute",
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "queries": [],
    }

    errors: list[dict] = []

    for qid, qtext in QUERIES:
        print(f"\n=== {qid}: {qtext!r} ===")
        query_entry = {"query_id": qid, "query": qtext, "methods": {}}
        for method in METHODS:
            try:
                if method == "dense_only":
                    hits = run_dense_only(client, dense, qtext, statute_filter)
                elif method == "bm25_only":
                    hits = run_bm25_only(client, sparse, qtext, statute_filter)
                else:
                    hits = run_hybrid(client, dense, sparse, qtext)
                summary = summarize(hits, method)
                query_entry["methods"][method] = summary
                print(f"  {method:10s}: n_ranked={summary['n_candidates_ranked']:3d}  "
                      f"target_rank={summary['target_rank']}  target_score={summary['target_score']}  "
                      f"in_top10={summary['target_in_top10']}")
            except Exception as exc:  # noqa: BLE001 -- isolate this method's failure, continue, report it
                errors.append({"query_id": qid, "method": method, "error": str(exc)})
                query_entry["methods"][method] = {"method": method, "error": str(exc)}
                print(f"  {method:10s}: ERROR -- {exc}")
        report["queries"].append(query_entry)

    report["errors"] = errors
    report["run_completed_successfully"] = not errors

    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    meta = {
        "run_timestamp": report["run_timestamp"],
        "query_ids": [qid for qid, _ in QUERIES],
        "methods_run": list(METHODS),
        "collection": COLLECTION,
        "target_chunk_id": TARGET_CHUNK_ID,
        "source_filter": "statute",
        "ranking_depth": N_STATUTE,
        "hybrid_candidates_depth": CANDIDATES,
        "output_path": str(out_path),
        "run_completed_successfully": not errors,
        "errors": errors,
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n[q01_phrasing] full report -> {out_path}")
    print(f"[q01_phrasing] run metadata -> {meta_path}")
    if errors:
        print(f"[q01_phrasing] {len(errors)} error(s) occurred: {errors}")


if __name__ == "__main__":
    main()
