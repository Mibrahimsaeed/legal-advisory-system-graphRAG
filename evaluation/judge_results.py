#!/usr/bin/env python3
"""Builds a manual-relevance-judging CSV from a run_retrieval.py JSONL export.

    python evaluation/judge_results.py --in evaluation/results/retrieval_<ts>.jsonl \
        --out evaluation/results/judging_<ts>.csv

The output CSV has one row per retrieved chunk with a blank `relevance`
column for a human to fill in by hand:

    0 = irrelevant
    1 = partially relevant
    2 = highly relevant

This script NEVER writes a value into `relevance` itself -- it is always
left blank (an empty string) for manual judging. calculate_metrics.py is
the only place relevance labels are read back in, and it treats a blank
exactly as "not yet judged", never as "0 / irrelevant".

By default only case/statute rows are included, since those are the only
source types calculate_metrics.py scores -- pass --include-constitution to
also carry Constitution rows into the judging sheet (for separate analysis;
they still won't be scored).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDNAMES = ["query_id", "source_type", "rank", "point_id", "chunk_id",
              "context_header", "canonical_text", "relevance"]


def load_retrieval_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
    if not records:
        raise ValueError(f"{path}: no records found")
    return records


def build_judging_rows(records: list[dict], include_constitution: bool) -> list[dict]:
    rows = []
    for r in records:
        if not include_constitution and r.get("source_type") == "constitution":
            continue
        rows.append({
            "query_id": r.get("query_id"),
            "source_type": r.get("source_type"),
            "rank": r.get("rank"),
            "point_id": r.get("point_id"),
            "chunk_id": r.get("chunk_id"),
            "context_header": r.get("context_header"),
            "canonical_text": r.get("canonical_text"),
            "relevance": "",
        })
    rows.sort(key=lambda r: (r["query_id"] or "", r["source_type"] or "", r["rank"] or 0))
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="run_retrieval.py JSONL export")
    ap.add_argument("--out", type=Path, required=True, help="output judging .csv path")
    ap.add_argument("--include-constitution", action="store_true",
                    help="also include Constitution rows (never scored, but exportable for analysis)")
    a = ap.parse_args(argv)

    records = load_retrieval_jsonl(a.inp)
    rows = build_judging_rows(records, a.include_constitution)
    if not rows:
        raise SystemExit(f"[judge_results] no rows to judge after filtering (input: {a.inp})")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)

    by_source: dict[str, int] = {}
    for r in rows:
        by_source[r["source_type"]] = by_source.get(r["source_type"], 0) + 1
    print(f"[judge_results] wrote {len(rows)} rows -> {a.out}  ({by_source})")
    print("[judge_results] fill in the 'relevance' column (0/1/2) for every row before running calculate_metrics.py")


if __name__ == "__main__":
    main()
