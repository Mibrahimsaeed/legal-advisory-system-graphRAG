#!/usr/bin/env python3
"""Gemini-backend pilot run of the case-law structuring layer.

Mirrors build_structured_openai_pilot.py exactly (same PILOT_DOC_IDS, same
structure_document()/to_output_document() orchestration, same atomic
write pattern) with ONE difference: the LLM client passed to
structure_document() is GeminiStructuringClient (gemini-2.5-flash-lite)
instead of OpenAIStructuringClient or the default OllamaLLMClient.

Reads var/rag/processed/<doc_id>.json (same cleaned inputs the Qwen and
OpenAI pilots used). Writes var/rag/structured_gemini_pilot/<doc_id>.json
-- a SEPARATE directory from var/rag/structured/ and
var/rag/structured_openai_pilot/, so neither existing pilot's results are
touched or overwritten.

Does not touch var/rag/raw/, var/rag/processed/, var/rag/structured/,
var/rag/structured_openai_pilot/, or var/metadata.db. Does not process the
full 2,088-document corpus -- this is a pilot only, per the task brief.

Requires GEMINI_API_KEY to be set (shell env or a local .env file, never
committed). If it is not set, this script exits before making any network
call or writing any file.
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
OUT_DIR = REPO / "var" / "rag" / "structured_gemini_pilot"

# Identical to build_structured_pilot.py / build_structured_openai_pilot.py's
# PILOT_DOC_IDS -- same representative set, same cleaned inputs, so all
# three backends are compared on exactly the same documents.
PILOT_DOC_IDS = [
    "8dd0ca3d14364404d75b76fe",
    "00ad7248bf77883e6b5975ed",
    "a3ec28ba4900855d15a6e870",
    "1d0a3afecc805e3d71cb54b9",
    "0513f7c1139b729f4cccf83d",
    "04139076ff36792b38b72f8b",
    "177dbfa9f6115c8c4bb0cdcf",
    "f7afd3f61f59d8f9798b8742",
    "b3b83bb2ceb23d88cf08394a",
    "ae4a07835728a54c3dddb4ad",
    "2ee9704f426892ca99e2b92e",
    "008d928c2926b2a5855bcc36",
]


def main() -> None:
    # Fail loudly, before touching any output directory or making any
    # call, if the key isn't available -- per the task's "do not silently
    # mark success" instruction.
    try:
        gemini_cfg = GeminiStructuringConfig()
        client = GeminiStructuringClient(cfg=gemini_cfg)
    except RuntimeError as exc:
        print(f"ABORTED before any API call: {exc}")
        sys.exit(1)

    llm_cfg = to_rag_structuring_llm_config(gemini_cfg)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Gemini pilot ({gemini_cfg.model}): {len(PILOT_DOC_IDS)} documents")
    pilot_t0 = time.perf_counter()
    results = []
    for i, doc_id in enumerate(PILOT_DOC_IDS, 1):
        in_path = PROCESSED_DIR / f"{doc_id}.json"
        if not in_path.exists():
            print(f"  [{i}] {doc_id}: MISSING from var/rag/processed/ -- skipped")
            continue
        doc = json.loads(in_path.read_text(encoding="utf-8"))

        calls_before = client.call_count
        failed_before = client.failed_call_count
        prompt_tokens_before = client.total_prompt_tokens
        completion_tokens_before = client.total_completion_tokens

        t0 = time.perf_counter()
        try:
            result = structure_document(doc, llm_client=client, llm_cfg=llm_cfg)
        except Exception as exc:
            # structure_document itself does not raise on a per-batch LLM
            # failure (label_body_paragraphs catches that and falls back
            # per-batch) -- this only fires on something unexpected, and
            # per the task's conservative-failure instruction we record it
            # rather than fabricate a result for this document.
            elapsed = time.perf_counter() - t0
            print(f"  [{i}/{len(PILOT_DOC_IDS)}] {doc_id}: FAILED ({exc}) elapsed={elapsed:.1f}s")
            results.append({
                "doc_id": doc_id, "status": "error", "error": str(exc),
                "elapsed_seconds": round(elapsed, 2),
            })
            continue
        elapsed = time.perf_counter() - t0

        output = to_output_document(doc, result)
        out_path = OUT_DIR / f"{doc_id}.json"
        tmp_path = out_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(out_path)

        doc_calls = client.call_count - calls_before
        doc_failed_calls = client.failed_call_count - failed_before
        doc_prompt_tokens = client.total_prompt_tokens - prompt_tokens_before
        doc_completion_tokens = client.total_completion_tokens - completion_tokens_before

        print(f"  [{i}/{len(PILOT_DOC_IDS)}] {doc_id}: status={result.structure_status} "
              f"n_spans={len(result.spans)} used_llm={result.used_llm} "
              f"fallback={result.llm_fallback_reason} valid={result.validation['ok']} "
              f"llm_calls={doc_calls} failed_calls={doc_failed_calls} "
              f"tokens_in={doc_prompt_tokens} tokens_out={doc_completion_tokens} "
              f"elapsed={elapsed:.1f}s")
        results.append({
            "doc_id": doc_id, "status": result.structure_status,
            "n_spans": len(result.spans), "used_llm": result.used_llm,
            "llm_fallback_reason": result.llm_fallback_reason,
            "validation_ok": result.validation["ok"],
            "validation_errors": result.validation["errors"],
            "llm_calls": doc_calls,
            "llm_failed_calls": doc_failed_calls,
            "prompt_tokens": doc_prompt_tokens,
            "completion_tokens": doc_completion_tokens,
            "elapsed_seconds": round(elapsed, 2),
        })

    pilot_elapsed_seconds = time.perf_counter() - pilot_t0
    total_requests = client.call_count + client.failed_call_count
    requests_per_minute = (
        round(total_requests / (pilot_elapsed_seconds / 60.0), 2) if pilot_elapsed_seconds > 0 else 0.0
    )

    report = {
        "model": gemini_cfg.model,
        "max_paragraphs_per_batch": gemini_cfg.max_paragraphs_per_batch,
        "max_chars_per_batch": gemini_cfg.max_chars_per_batch,
        "documents": results,
        "usage_summary": client.usage_summary(),
        "pilot_elapsed_seconds": round(pilot_elapsed_seconds, 2),
        "approx_requests_per_minute": requests_per_minute,
    }
    (OUT_DIR / "pilot_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nPILOT COMPLETE. Report: var/rag/structured_gemini_pilot/pilot_report.json")
    n_ok = sum(1 for r in results if r.get("validation_ok"))
    print(f"validation ok: {n_ok}/{len(results)}")
    print(f"usage: {client.usage_summary()}")
    print(f"total pilot runtime: {pilot_elapsed_seconds:.1f}s, approx requests/minute: {requests_per_minute}")


if __name__ == "__main__":
    main()
