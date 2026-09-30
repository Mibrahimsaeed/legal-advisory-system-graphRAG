#!/usr/bin/env python3
"""RAG cleaning stage: var/rag/raw/<doc_id>.json -> var/rag/processed/<doc_id>.json.

Case-law only, matching what's actually staged (every document under
var/rag/raw/ has source_type='case_html' -- verified against the database
before this script was written, not assumed; there is nothing of a
different document shape, e.g. a bare statute/Act text, to accidentally
process here).

    var/rag/raw/<doc_id>.json   (read-only input; never modified)
        -> clean_full_text()          [src/rag_prep/case_cleaner.py]
        -> extract_disposition()      [src/rag_prep/case_cleaner.py]
        -> derive_court_location()    [src/rag_prep/case_cleaner.py]
        -> extract_statutes_cited()   [src/rag_prep/statute_cleaner.py]
        -> extract_provisions_cited() [src/rag_prep/statute_cleaner.py]
        -> var/rag/processed/<doc_id>.json

Touches var/metadata.db not at all -- every input field comes from the
already-staged raw JSON, not a fresh database read. No classification
code, table, or config is imported or modified. No chunking, embedding,
BM25, or Neo4j step runs here.

Resumable: a doc_id whose processed output already exists is skipped.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.rag_prep.case_cleaner import (
    REQUIRED_METADATA_FIELDS,
    build_processed_metadata,
    clean_full_text,
)

RAW_DIR = REPO / "var" / "rag" / "raw"
OUT_DIR = REPO / "var" / "rag" / "processed"


def process_one(raw_path: Path) -> dict:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    cleaned_text = clean_full_text(raw.get("full_text") or "")
    metadata = build_processed_metadata(raw["metadata"], cleaned_text, raw["doc_id"])
    return {
        "doc_id": raw["doc_id"],
        "metadata": metadata,
        "full_text": cleaned_text,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(RAW_DIR.glob("*.json"))
    print(f"Input (var/rag/raw/): {len(raw_files)} documents")

    written = 0
    skipped_existing = 0
    failed: list[dict] = []

    for raw_path in raw_files:
        doc_id = raw_path.stem
        out_path = OUT_DIR / f"{doc_id}.json"
        if out_path.exists():
            skipped_existing += 1
            continue

        try:
            processed = process_one(raw_path)
        except Exception as exc:  # noqa: BLE001 -- one bad doc must not kill the batch
            failed.append({"doc_id": doc_id, "reason": f"{type(exc).__name__}: {exc}"})
            continue

        if not processed["full_text"].strip():
            failed.append({"doc_id": doc_id, "reason": "empty full_text after cleaning"})
            continue
        missing_fields = [f for f in REQUIRED_METADATA_FIELDS if f not in processed["metadata"]]
        if missing_fields:
            failed.append({"doc_id": doc_id, "reason": f"missing metadata fields: {missing_fields}"})
            continue

        tmp_path = out_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(processed, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(out_path)
        written += 1

        if (written + skipped_existing) % 300 == 0:
            print(f"  progress: {written} written, {skipped_existing} already existed, "
                  f"{len(failed)} failed / {len(raw_files)} total")

    print(f"\nDone. input={len(raw_files)} written={written} "
          f"skipped_existing={skipped_existing} failed={len(failed)}")

    if failed:
        (REPO / "var" / "rag" / "processed_failures.json").write_text(
            json.dumps(failed, indent=2), encoding="utf-8"
        )
        print(f"Failures written to var/rag/processed_failures.json ({len(failed)})")

    # ---- validation report -------------------------------------------
    processed_files = sorted(OUT_DIR.glob("*.json"))
    print(f"\n=== VALIDATION ({len(processed_files)} processed files) ===")

    raw_ids = {p.stem for p in raw_files}
    processed_ids = {p.stem for p in processed_files}
    missing_outputs = raw_ids - processed_ids - {f["doc_id"] for f in failed}
    print(f"input has corresponding output: {len(raw_ids & processed_ids)}/{len(raw_ids)}")
    if missing_outputs:
        print(f"  UNEXPECTED missing outputs (not in failures list either): {sorted(missing_outputs)[:10]}")

    bad_docid = bad_meta = bad_text = 0
    lens = []
    for p in processed_files:
        d = json.loads(p.read_text(encoding="utf-8"))
        if not d.get("doc_id") or d["doc_id"] != p.stem:
            bad_docid += 1
        missing = [f for f in REQUIRED_METADATA_FIELDS if f not in (d.get("metadata") or {})]
        if missing:
            bad_meta += 1
        ft = d.get("full_text") or ""
        if not ft.strip():
            bad_text += 1
        else:
            lens.append(len(ft))

    print(f"doc_id missing/mismatched: {bad_docid}")
    print(f"metadata missing required field(s): {bad_meta}")
    print(f"empty full_text: {bad_text}")
    if lens:
        print(f"full_text length (chars): min={min(lens)} median={sorted(lens)[len(lens)//2]} "
              f"max={max(lens)} mean={st.mean(lens):.0f}")

    # Raw input unmodified?
    raw_bytes_unchanged = all(
        (RAW_DIR / p.name).exists() for p in raw_files
    )
    print(f"\nvar/rag/raw/ untouched (all {len(raw_files)} originals still present): {raw_bytes_unchanged}")


if __name__ == "__main__":
    main()
