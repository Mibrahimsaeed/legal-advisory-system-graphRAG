#!/usr/bin/env python3
"""Dense-only (BAAI/bge-m3) retrieval for evaluating statute retrieval quality in isolation
from BM25/RRF fusion.

Reuses the EXISTING, unmodified retrieval stack in src.rag_prep.embed_index
(connect(), load_dense(), encode_dense(), COLLECTION, MODEL_NAME, DENSE_DIM) --
this script never re-implements, modifies, or alters embedding generation,
chunking, the Qdrant schema, or the production hybrid_search()/RRF pipeline.
It issues a single unfused dense-vector query per evaluation query, with no
BM25, no sparse vector, no FusionQuery/RRF, no query rewriting, and no
reranking -- if dense-only retrieval cannot be performed as configured, this
script raises an explicit error rather than silently falling back to hybrid
or sparse retrieval.

    python evaluation/retrieve_dense.py
    python evaluation/retrieve_dense.py --source statute --depth 10
    python evaluation/retrieve_dense.py --out evaluation/results/dense_statute_results.jsonl --overwrite

Output: one JSONL file under evaluation/results/ (one row per retrieved
chunk, per query) plus a sibling "<name>.meta.json" run-manifest. The output
schema intentionally matches evaluation/judge_results.py's expected input
fields (query_id, source_type, rank, point_id, chunk_id, context_header,
canonical_text) exactly, so the existing, unmodified judge_results.py and
calculate_metrics.py can consume this file without any code change -- no
relevance label is written, inferred, or reused by this script.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rag_prep.embed_index import (  # noqa: E402
    COLLECTION,
    DENSE_DIM,
    MODEL_NAME,
    QDRANT_URL,
    connect,
    encode_dense,
    load_dense,
)

DEFAULT_QUERIES_PATH = Path("evaluation/queries.json")
DEFAULT_RESULTS_DIR = Path("evaluation/results")
DEFAULT_OUT_PATH = DEFAULT_RESULTS_DIR / "dense_statute_results.jsonl"

# Fields copied from each hit's Qdrant payload into the export. The first
# four (chunk_id, context_header, canonical_text -- plus query_id/source_type/
# rank which come from the query loop, not the payload) are exactly what
# evaluation/judge_results.py reads from a retrieval JSONL; the rest are
# additional document/act/section identity fields this task asked for.
_PAYLOAD_FIELDS = (
    "chunk_id", "context_header", "canonical_text",
    "act_key", "act_name", "year", "section_number", "section_title", "section_status",
)


def _load_run_retrieval_module():
    """Loads evaluation/run_retrieval.py by file path, not by package name --
    src/evaluation/ already exists and shadows a dotted `import evaluation.X`
    for this repo-root evaluation/ directory (same gotcha documented in
    evaluation/README.md and tests/test_evaluation_metrics.py)."""

    path = REPO_ROOT / "evaluation" / "run_retrieval.py"
    spec = importlib.util.spec_from_file_location("eval_run_retrieval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_dense_vector_config(client) -> None:
    """Raises SystemExit with a clear message if the live collection does not
    have a named 'dense' vector with the expected dimension and distance --
    never silently proceeds with a mismatched or missing configuration."""

    from qdrant_client import models

    try:
        info = client.get_collection(COLLECTION)
    except Exception as exc:  # noqa: BLE001 -- surface the real connection error
        raise SystemExit(f"[retrieve_dense] could not reach Qdrant collection {COLLECTION!r}: {exc}") from exc

    vectors = info.config.params.vectors
    if not isinstance(vectors, dict) or "dense" not in vectors:
        raise SystemExit(
            f"[retrieve_dense] collection {COLLECTION!r} has no named 'dense' vector "
            f"(found: {list(vectors) if isinstance(vectors, dict) else vectors!r}) -- refusing to proceed"
        )
    dense_cfg = vectors["dense"]
    if dense_cfg.size != DENSE_DIM:
        raise SystemExit(
            f"[retrieve_dense] 'dense' vector size is {dense_cfg.size}, expected {DENSE_DIM} -- refusing to proceed"
        )
    if dense_cfg.distance != models.Distance.COSINE:
        raise SystemExit(
            f"[retrieve_dense] 'dense' vector distance is {dense_cfg.distance}, expected Cosine -- refusing to proceed"
        )
    print(f"[retrieve_dense] verified: collection={COLLECTION!r} dense vector size={dense_cfg.size} "
          f"distance={dense_cfg.distance}")


def run_query(client, dense_model, row: dict, source: str | None, depth: int) -> list[dict]:
    """One dense-only, unfused query_points() call -- no prefetch, no FusionQuery,
    no sparse vector involved at any point in this function."""

    from qdrant_client import models

    query_text = row.get("query")
    if not query_text:
        raise ValueError(f"query row {row!r} is missing a non-empty 'query' field")

    vec = encode_dense(dense_model, [query_text], 1)[0].tolist()

    query_filter = None
    if source:
        query_filter = models.Filter(must=[models.FieldCondition(key="source_type", match=models.MatchValue(value=source))])

    hits = client.query_points(
        COLLECTION,
        query=vec,
        using="dense",
        query_filter=query_filter,
        limit=depth,
        with_payload=True,
    ).points

    records = []
    for rank, h in enumerate(hits, start=1):
        payload = h.payload or {}
        record = {
            "query_id": row["query_id"],
            "query": query_text,
            "source_type": payload.get("source_type", source or "unknown"),
            "rank": rank,
            "point_id": str(h.id),
            "dense_score": h.score,
        }
        for f in _PAYLOAD_FIELDS:
            record[f] = payload.get(f)
        records.append(record)
    return records


def resolve_out_path(requested: Path | None, overwrite: bool) -> Path:
    """Never silently overwrites an existing file. Without --overwrite, an
    already-existing destination (default or explicitly given) is redirected
    to a unique, timestamped filename instead."""

    out_path = requested or DEFAULT_OUT_PATH
    if out_path.exists() and not overwrite:
        stamped = out_path.with_name(f"{out_path.stem}_{time.strftime('%Y%m%dT%H%M%S')}{out_path.suffix}")
        print(f"[retrieve_dense] {out_path} already exists and --overwrite was not given; "
              f"writing to {stamped} instead")
        return stamped
    return out_path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_PATH, help="path to queries.json")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"output .jsonl path (default: {DEFAULT_OUT_PATH})")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow overwriting an existing --out file (default: write a timestamped file instead)")
    ap.add_argument("--qdrant", default=QDRANT_URL, help="Qdrant URL (default: %(default)s)")
    ap.add_argument("--source", default="statute",
                    help="source_type filter, e.g. 'statute' (default: %(default)s); pass '' to disable filtering")
    ap.add_argument("--depth", type=int, default=10, help="results per query (default: %(default)s)")
    ap.add_argument("--device", default=None, help="dense model device override (cpu/cuda/mps)")
    ap.add_argument("--no-fp16", action="store_true", help="disable fp16 for the dense model")
    a = ap.parse_args(argv)

    run_retrieval = _load_run_retrieval_module()
    queries = run_retrieval.load_queries(a.queries)
    print(f"[retrieve_dense] {len(queries)} queries loaded from {a.queries}")

    source = a.source or None
    out_path = resolve_out_path(a.out, a.overwrite)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = connect(a.qdrant)
    verify_dense_vector_config(client)
    dense = load_dense(a.device, fp16=not a.no_fp16)

    all_records: list[dict] = []
    empty_queries: list[str] = []
    failed_queries: list[dict] = []
    run_ok = True

    with out_path.open("w", encoding="utf-8") as fh:
        for row in queries:
            try:
                records = run_query(client, dense, row, source, a.depth)
            except Exception as exc:  # noqa: BLE001 -- isolate this query's failure, continue the run, report it
                failed_queries.append({"query_id": row.get("query_id"), "error": str(exc)})
                run_ok = False
                print(f"  {row.get('query_id')}: FAILED -- {exc}")
                continue
            if not records:
                empty_queries.append(row["query_id"])
                print(f"  {row['query_id']}: 0 results")
            else:
                print(f"  {row['query_id']}: {len(records)} results")
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            all_records.extend(records)

    result_counts: dict[str, int] = {}
    for r in all_records:
        result_counts[r["source_type"]] = result_counts.get(r["source_type"], 0) + 1

    meta = {
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "queries_file": str(a.queries),
        "query_ids": [q["query_id"] for q in queries],
        "collection": COLLECTION,
        "dense_model": MODEL_NAME,
        "dense_vector_name": "dense",
        "dense_dim": DENSE_DIM,
        "dense_distance": "Cosine",
        "retrieval_method": "dense_only",
        "fusion": None,
        "sparse_model_used": None,
        "source_filter": source,
        "depth": a.depth,
        "qdrant_url": a.qdrant,
        "output_path": str(out_path),
        "query_count": len(queries),
        "result_counts": result_counts,
        "empty_result_queries": empty_queries,
        "failed_queries": failed_queries,
        "run_completed_successfully": run_ok,
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[retrieve_dense] wrote {len(all_records)} rows -> {out_path}")
    print(f"[retrieve_dense] run metadata -> {meta_path}")
    print(f"[retrieve_dense] result_counts: {result_counts}")
    if empty_queries:
        print(f"[retrieve_dense] WARNING: 0 results for query_id(s): {empty_queries}")
    if failed_queries:
        print(f"[retrieve_dense] ERROR: {len(failed_queries)} query(ies) failed: {failed_queries}")
        sys.exit(1)


if __name__ == "__main__":
    main()
