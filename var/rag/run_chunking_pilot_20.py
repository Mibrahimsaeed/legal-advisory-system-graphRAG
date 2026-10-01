#!/usr/bin/env python3
"""20-case real-data chunking pilot.

Reads the first 20 var/rag/structured_gemini/<doc_id>.json files in
sorted filename order (deterministic, reproducible selection), runs the
EXISTING, unmodified chunker (src.rag_prep.chunker.chunk_document) and
the EXISTING, unmodified validator (src.rag_prep.chunk_validate.validate_chunk_set)
on each, and writes the result to var/rag/chunked_pilot_20/<doc_id>.json.

Stops immediately if any of the 20 cases produces an invalid chunk set
(fail-fast, per the task brief) -- this pilot is a validation gate, not a
best-effort run.

Does not touch var/rag/structured_gemini/, var/rag/processed/,
var/rag/raw/, or var/metadata.db. Does not run BM25/embeddings/Neo4j.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.rag_prep.chunker import chunk_document
from src.rag_prep.chunk_validate import validate_chunk_set

STRUCTURED_DIR = REPO / "var" / "rag" / "structured_gemini"
OUT_DIR = REPO / "var" / "rag" / "chunked_pilot_20"
N_CASES = 20


def main() -> None:
    all_paths = sorted(
        p for p in STRUCTURED_DIR.glob("*.json")
        if p.name not in {"progress.json", "full_run_report.json"}
    )
    selected = all_paths[:N_CASES]

    print(f"cases_selected: {len(selected)}")
    if len(selected) != N_CASES:
        print(f"ABORTED: expected {N_CASES} cases, found {len(selected)}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    doc_ids: list[str] = []
    cases_processed = 0
    cases_failed = 0
    total_chunks = 0
    validation_passed = 0
    validation_failed = 0
    oversized_chunks = 0
    overlap_chunks = 0
    sections_encountered: Counter = Counter()

    for i, path in enumerate(selected, 1):
        doc_id = path.stem
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"  [{i}/{N_CASES}] {doc_id}: FAILED to load ({exc})")
            cases_failed += 1
            continue

        try:
            parent = chunk_document(doc)
        except Exception as exc:
            print(f"  [{i}/{N_CASES}] {doc_id}: chunk_document() FAILED ({exc})")
            cases_failed += 1
            print("ABORTING pilot: a case produced an invalid chunk set.")
            sys.exit(1)

        spans = doc.get("structure", {}).get("spans", [])
        result = validate_chunk_set(parent, spans)

        if not result.ok:
            validation_failed += 1
            print(f"  [{i}/{N_CASES}] {doc_id}: VALIDATION FAILED: {result.errors}")
            print("ABORTING pilot: a case failed validation.")
            sys.exit(1)

        validation_passed += 1
        doc_ids.append(doc_id)
        cases_processed += 1
        total_chunks += len(parent.chunks)
        oversized_chunks += result.diagnostics["oversized_single_paragraph_count"]
        overlap_chunks += result.diagnostics["overlap_chunk_count"]
        for c in parent.chunks:
            sections_encountered[c.section] += 1

        out_path = OUT_DIR / f"{doc_id}.json"
        tmp_path = out_path.with_suffix(".json.tmp")
        output = {
            "parent": parent.to_dict(),
            "validation": result.to_dict(),
        }
        tmp_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(out_path)

        print(f"  [{i}/{N_CASES}] {doc_id}: ok, {len(parent.chunks)} chunks, "
              f"validation_ok={result.ok}")

    report = {
        "cases_selected": len(selected),
        "cases_processed": cases_processed,
        "cases_failed": cases_failed,
        "total_chunks": total_chunks,
        "validation_passed": validation_passed,
        "validation_failed": validation_failed,
        "oversized_chunks": oversized_chunks,
        "overlap_chunks": overlap_chunks,
        "sections_encountered": dict(sections_encountered),
        "doc_ids": doc_ids,
    }
    (OUT_DIR / "pilot_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nPILOT COMPLETE.")
    for k, v in report.items():
        if k != "doc_ids":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
