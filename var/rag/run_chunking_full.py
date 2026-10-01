#!/usr/bin/env python3
"""Full 2,088-case chunking run.

Reads every var/rag/structured_gemini/<doc_id>.json, runs the EXISTING,
unmodified chunker (src.rag_prep.chunker.chunk_document) and validator
(src.rag_prep.chunk_validate.validate_chunk_set) on each, and writes
var/rag/chunked/<doc_id>.json in the same {"parent", "validation"} shape
the 20-case pilot used.

Resumable: an existing output is only skipped if
src.rag_prep.chunk_validate.is_valid_existing_output() says it's actually
valid (not merely present) -- a missing, corrupt, malformed, or
previously-failed-validation output is regenerated.

Per-document isolation: a ChunkingError or a failed validation on one
document is recorded and processing continues to the next document --
never silently hidden, never aborting the whole run over one bad case.

Does not touch var/rag/structured_gemini/, var/rag/processed/,
var/rag/raw/, var/metadata.db, or var/rag/chunked_pilot_20/. No LLM, no
BM25, no embeddings, no Neo4j.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.rag_prep.chunker import chunk_document
from src.rag_prep.chunk_validate import is_valid_existing_output, validate_chunk_set

STRUCTURED_DIR = REPO / "var" / "rag" / "structured_gemini"
OUT_DIR = REPO / "var" / "rag" / "chunked"


def main() -> None:
    all_paths = sorted(
        p for p in STRUCTURED_DIR.glob("*.json")
        if p.name not in {"progress.json", "full_run_report.json"}
    )
    total_input_cases = len(all_paths)
    print(f"total_input_cases: {total_input_cases}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    processed_cases = 0
    skipped_valid_cases = 0
    regenerated_cases = 0
    failed_cases: list[dict] = []
    total_chunks = 0
    validation_passed = 0
    validation_failed = 0
    oversized_chunks = 0
    overlap_chunks = 0
    chunks_per_case: list[int] = []

    for i, path in enumerate(all_paths, 1):
        doc_id = path.stem
        out_path = OUT_DIR / f"{doc_id}.json"

        had_existing = out_path.exists()
        if had_existing:
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                if is_valid_existing_output(existing.get("parent", {})) and existing.get("validation", {}).get("ok"):
                    skipped_valid_cases += 1
                    total_chunks += len(existing["parent"]["chunks"])
                    chunks_per_case.append(len(existing["parent"]["chunks"]))
                    validation_passed += 1
                    continue
            except (json.JSONDecodeError, KeyError, TypeError):
                pass  # falls through to regeneration below

        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failed_cases.append({"doc_id": doc_id, "error": f"could not load structured source: {exc}"})
            continue

        try:
            parent = chunk_document(doc)
        except Exception as exc:  # noqa: BLE001 -- isolate this doc's failure, continue the run
            failed_cases.append({"doc_id": doc_id, "error": f"chunk_document() failed: {exc}"})
            continue

        spans = doc.get("structure", {}).get("spans", [])
        result = validate_chunk_set(parent, spans)

        output = {"parent": parent.to_dict(), "validation": result.to_dict()}
        tmp_path = out_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(out_path)

        processed_cases += 1
        if had_existing:
            regenerated_cases += 1

        if result.ok:
            validation_passed += 1
            total_chunks += len(parent.chunks)
            chunks_per_case.append(len(parent.chunks))
            oversized_chunks += result.diagnostics["oversized_single_paragraph_count"]
            overlap_chunks += result.diagnostics["overlap_chunk_count"]
        else:
            validation_failed += 1
            failed_cases.append({"doc_id": doc_id, "error": result.errors})

        if i % 200 == 0:
            print(f"  ...{i}/{total_input_cases}")

    report = {
        "total_input_cases": total_input_cases,
        "processed_cases": processed_cases,
        "skipped_valid_cases": skipped_valid_cases,
        "regenerated_cases": regenerated_cases,
        "failed_cases": len(failed_cases),
        "total_chunks": total_chunks,
        "validation_passed": validation_passed,
        "validation_failed": validation_failed,
        "oversized_chunks": oversized_chunks,
        "overlap_chunks": overlap_chunks,
        "minimum_chunks_per_case": min(chunks_per_case) if chunks_per_case else None,
        "median_chunks_per_case": statistics.median(chunks_per_case) if chunks_per_case else None,
        "maximum_chunks_per_case": max(chunks_per_case) if chunks_per_case else None,
        "failed_doc_ids": failed_cases,
    }
    (OUT_DIR / "chunking_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nFULL CHUNKING RUN COMPLETE.")
    for k, v in report.items():
        if k != "failed_doc_ids":
            print(f"  {k}: {v}")
    if failed_cases:
        print(f"  failed_doc_ids: {[f['doc_id'] for f in failed_cases]}")


if __name__ == "__main__":
    main()
