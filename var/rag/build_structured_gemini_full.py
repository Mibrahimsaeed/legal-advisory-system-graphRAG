#!/usr/bin/env python3
"""Full-corpus case-law structuring run using the validated Gemini backend.

Reads every var/rag/processed/<doc_id>.json (the case-law-only cleaned
corpus -- confirmed 2,088 files, all with case-law metadata
(case_title/court/disposition/...), no statutes mixed in, so no separate
filtering is needed here). Writes var/rag/structured_gemini/<doc_id>.json.

Uses GeminiStructuringClient/GeminiStructuringConfig exactly as validated
by the 12-document pilot (var/rag/build_structured_gemini_pilot.py) --
same prompts, batching, schema, validation, fallback, pacing, and
retryDelay handling from src/rag_prep/structurer_llm_gemini.py, unmodified.

Resumable: any doc_id that already has an output file is skipped without
making an API call, so a killed/restarted run never re-pays for completed
documents. Per-document writes are atomic (.json.tmp + os.replace()), so a
kill mid-document never leaves a corrupted or partial file behind.

Does not touch var/rag/raw/, var/rag/processed/, var/rag/structured/,
var/rag/structured_openai_pilot/, var/rag/structured_gemini_pilot/, or
var/metadata.db. Does not run classification or apply-reviews.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.rag_prep.structurer import structure_document, to_output_document
from src.rag_prep.structurer_llm_gemini import (
    GeminiStructuringClient,
    GeminiStructuringConfig,
    to_rag_structuring_llm_config,
)

PROCESSED_DIR = REPO / "var" / "rag" / "processed"
OUT_DIR = REPO / "var" / "rag" / "structured_gemini"
PROGRESS_PATH = OUT_DIR / "progress.json"
REPORT_PATH = OUT_DIR / "full_run_report.json"


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _existing_output_doc_ids() -> set[str]:
    return {
        p.stem for p in OUT_DIR.glob("*.json")
        if p.name not in {PROGRESS_PATH.name, REPORT_PATH.name}
    }


def _aggregate_final_report(model: str, run_elapsed_seconds: float, session_usage: dict) -> dict:
    """Scans every completed output file (from this run AND any prior
    resumed runs) for the authoritative whole-corpus counts -- not just
    what this particular process session touched."""

    status_counts: dict[str, int] = {}
    validation_ok = 0
    validation_failed = 0
    fallback_docs = 0
    total_spans = 0
    doc_ids = _existing_output_doc_ids()

    for doc_id in doc_ids:
        try:
            doc = json.loads((OUT_DIR / f"{doc_id}.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        structure = doc.get("structure", {})
        status = structure.get("structure_status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        if structure.get("validation", {}).get("ok"):
            validation_ok += 1
        else:
            validation_failed += 1
        if structure.get("llm_fallback_reason"):
            fallback_docs += 1
        total_spans += len(structure.get("spans", []))

    return {
        "model": model,
        "corpus_size": len(list(PROCESSED_DIR.glob("*.json"))),
        "documents_completed": len(doc_ids),
        "status_counts": status_counts,
        "validation_ok": validation_ok,
        "validation_failed": validation_failed,
        "docs_with_fallback_batches": fallback_docs,
        "total_spans": total_spans,
        "this_session_usage": session_usage,
        "this_session_elapsed_seconds": round(run_elapsed_seconds, 2),
    }


def main() -> None:
    try:
        gemini_cfg = GeminiStructuringConfig()
        client = GeminiStructuringClient(cfg=gemini_cfg)
    except RuntimeError as exc:
        print(f"ABORTED before any API call: {exc}")
        sys.exit(1)

    llm_cfg = to_rag_structuring_llm_config(gemini_cfg)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_paths = sorted(PROCESSED_DIR.glob("*.json"))
    already_done = _existing_output_doc_ids()
    todo = [p for p in all_paths if p.stem not in already_done]

    print(f"Full corpus run ({gemini_cfg.model}): {len(all_paths)} total, "
          f"{len(already_done)} already done, {len(todo)} remaining")

    run_t0 = time.perf_counter()
    session_results = []
    for i, in_path in enumerate(todo, 1):
        doc_id = in_path.stem
        doc = json.loads(in_path.read_text(encoding="utf-8"))

        calls_before = client.call_count
        failed_before = client.failed_call_count

        t0 = time.perf_counter()
        try:
            result = structure_document(doc, llm_client=client, llm_cfg=llm_cfg)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"  [{i}/{len(todo)}] {doc_id}: FAILED ({exc}) elapsed={elapsed:.1f}s")
            session_results.append({"doc_id": doc_id, "status": "error", "error": str(exc)})
            continue
        elapsed = time.perf_counter() - t0

        output = to_output_document(doc, result)
        _atomic_write_json(OUT_DIR / f"{doc_id}.json", output)

        doc_calls = client.call_count - calls_before
        doc_failed_calls = client.failed_call_count - failed_before
        print(f"  [{i}/{len(todo)}] {doc_id}: status={result.structure_status} "
              f"n_spans={len(result.spans)} valid={result.validation['ok']} "
              f"fallback={result.llm_fallback_reason} llm_calls={doc_calls} "
              f"failed_calls={doc_failed_calls} elapsed={elapsed:.1f}s")
        session_results.append({
            "doc_id": doc_id, "status": result.structure_status,
            "validation_ok": result.validation["ok"],
            "llm_fallback_reason": result.llm_fallback_reason,
            "llm_calls": doc_calls, "llm_failed_calls": doc_failed_calls,
            "elapsed_seconds": round(elapsed, 2),
        })

        # Written every document (atomically) so a kill mid-run still
        # leaves an accurate, resumable progress snapshot behind.
        _atomic_write_json(PROGRESS_PATH, {
            "done_this_session": i, "remaining_this_session": len(todo) - i,
            "session_usage": client.usage_summary(),
            "session_elapsed_seconds": round(time.perf_counter() - run_t0, 2),
            "last_doc_id": doc_id,
        })

    run_elapsed = time.perf_counter() - run_t0
    final_report = _aggregate_final_report(gemini_cfg.model, run_elapsed, client.usage_summary())
    _atomic_write_json(REPORT_PATH, final_report)

    print(f"\nFULL RUN SESSION COMPLETE. Report: var/rag/structured_gemini/full_run_report.json")
    print(f"this session: {len(session_results)} documents processed, "
          f"{run_elapsed:.1f}s elapsed, usage={client.usage_summary()}")
    print(f"corpus-wide: {final_report['documents_completed']}/{final_report['corpus_size']} done, "
          f"validation_ok={final_report['validation_ok']}, "
          f"validation_failed={final_report['validation_failed']}, "
          f"status_counts={final_report['status_counts']}")


if __name__ == "__main__":
    main()
