#!/usr/bin/env python3
"""Baseline retrieval export for the Pakistani legal Hybrid RAG system.

Reuses the EXISTING, unmodified retrieval stack in src.rag_prep.embed_index
(hybrid_search(), load_dense(), load_sparse(), connect()) -- this script never
re-implements or alters retrieval, embedding, chunking, or the Qdrant schema.

    python evaluation/run_retrieval.py
    python evaluation/run_retrieval.py --queries evaluation/queries.json \
        --k-case 10 --k-statute 8 --k-constitution 5

Output: one JSONL file under evaluation/results/ (one row per retrieved
chunk, per query, per source) plus a sibling "<name>.meta.json" run-manifest
recording the collection, model, retrieval parameters, timestamp, and result
counts. JSONL (not CSV) is used for this raw export specifically so that
canonical_text/context_header are preserved exactly, with no delimiter-
escaping or truncation risk -- evaluation/judge_results.py converts this into
the human-judging CSV.

Constitution results are fetched only if --k-constitution > 0 (default: not
fetched at all). When fetched, they are written to the SAME output file with
source_type="constitution" -- calculate_metrics.py explicitly excludes that
source_type from every metric, so including them here is purely for later,
separate analysis, never for scoring.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rag_prep.embed_index import (  # noqa: E402
    COLLECTION,
    MODEL_NAME,
    QDRANT_URL,
    connect,
    hybrid_search,
    load_dense,
    load_sparse,
)

DEFAULT_QUERIES_PATH = Path("evaluation/queries.json")
DEFAULT_RESULTS_DIR = Path("evaluation/results")

# Fields copied from each hit's Qdrant payload into the export, alongside
# query_id/query/source_type/rank/point_id/rrf_score (which are not payload
# fields -- they come from the query loop / the ScoredPoint itself).
_PAYLOAD_FIELDS = ("chunk_id", "context_header", "canonical_text")


def load_queries(path: Path) -> list[dict]:
    """Loads and validates queries.json. Raises ValueError with a precise
    message on any structural problem -- never silently skips a bad entry."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non-empty JSON array of query objects")

    seen_ids: set[str] = set()
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: entry {i} is not an object")
        missing = [k for k in ("query_id", "category", "query") if not row.get(k)]
        if missing:
            raise ValueError(f"{path}: entry {i} ({row!r}) is missing required field(s): {missing}")
        if row["query_id"] in seen_ids:
            raise ValueError(f"{path}: duplicate query_id {row['query_id']!r}")
        seen_ids.add(row["query_id"])
    return data


def run_query(client, dense, sparse, row: dict, per_source: dict[str, int], candidates: int) -> list[dict]:
    """Runs one query through hybrid_search() and returns one export record
    per retrieved chunk, with rank numbered 1..N independently PER SOURCE
    (hybrid_search's own result list is already grouped by source, in
    per_source iteration order, so a running per-source counter is enough --
    no re-sorting of hybrid_search's own ranking is performed here)."""

    hits = hybrid_search(client, dense, sparse, row["query"], per_source, candidates=candidates)
    records = []
    rank_by_source: dict[str, int] = {}
    for h in hits:
        payload = h.payload or {}
        st = payload.get("source_type", "unknown")
        rank_by_source[st] = rank_by_source.get(st, 0) + 1
        record = {
            "query_id": row["query_id"],
            "query": row["query"],
            "source_type": st,
            "rank": rank_by_source[st],
            "point_id": str(h.id),
            "rrf_score": h.score,
        }
        for f in _PAYLOAD_FIELDS:
            record[f] = payload.get(f)
        records.append(record)
    return records


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_PATH, help="path to queries.json")
    ap.add_argument("--out", type=Path, default=None,
                    help="output .jsonl path (default: evaluation/results/retrieval_<timestamp>.jsonl)")
    ap.add_argument("--qdrant", default=QDRANT_URL, help="Qdrant URL (default: %(default)s)")
    ap.add_argument("--k-case", type=int, default=10, help="case-law results per query (default: %(default)s)")
    ap.add_argument("--k-statute", type=int, default=8, help="statute results per query (default: %(default)s)")
    ap.add_argument("--k-constitution", type=int, default=0,
                    help="Constitution results per query (default: 0 -- not fetched; "
                         "excluded from all metrics regardless, even if >0)")
    ap.add_argument("--candidates", type=int, default=40, help="per-source prefetch depth before RRF fusion")
    ap.add_argument("--device", default=None, help="dense model device override (cpu/cuda/mps)")
    ap.add_argument("--no-fp16", action="store_true", help="disable fp16 for the dense model")
    a = ap.parse_args(argv)

    queries = load_queries(a.queries)
    print(f"[run_retrieval] {len(queries)} queries loaded from {a.queries}")

    per_source = {"case": a.k_case, "statute": a.k_statute}
    if a.k_constitution > 0:
        per_source["constitution"] = a.k_constitution

    out_path = a.out or (DEFAULT_RESULTS_DIR / f"retrieval_{time.strftime('%Y%m%dT%H%M%S')}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = connect(a.qdrant)
    dense = load_dense(a.device, fp16=not a.no_fp16)
    sparse = load_sparse()

    all_records: list[dict] = []
    with out_path.open("w", encoding="utf-8") as fh:
        for row in queries:
            records = run_query(client, dense, sparse, row, per_source, a.candidates)
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            all_records.extend(records)
            counts = {st: sum(1 for r in records if r["source_type"] == st) for st in per_source}
            print(f"  {row['query_id']}: {counts}")

    result_counts: dict[str, int] = {}
    for r in all_records:
        result_counts[r["source_type"]] = result_counts.get(r["source_type"], 0) + 1

    meta = {
        "collection": COLLECTION,
        "dense_model": MODEL_NAME,
        "sparse_model": "Qdrant/bm25",
        "qdrant_url": a.qdrant,
        "queries_file": str(a.queries),
        "per_source_limit": per_source,
        "candidates_per_source": a.candidates,
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "query_count": len(queries),
        "result_counts": result_counts,
        "excluded_from_metrics": ["constitution"],
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[run_retrieval] wrote {len(all_records)} rows -> {out_path}")
    print(f"[run_retrieval] run metadata -> {meta_path}")
    print(f"[run_retrieval] result_counts: {result_counts}")


if __name__ == "__main__":
    main()
