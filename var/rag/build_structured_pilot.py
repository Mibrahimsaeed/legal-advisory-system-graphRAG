#!/usr/bin/env python3
"""Pilot run of the case-law structuring layer (Phase A per the task brief).

Reads var/rag/processed/<doc_id>.json for a fixed, representative set of
doc_ids selected from var/rag/audits/ (not random -- see PILOT_DOC_IDS'
comments for why each one is here), runs the real Qwen-backed structurer,
and writes var/rag/structured/<doc_id>.json for exactly those documents.

Does not touch var/rag/raw/, var/rag/processed/, or var/metadata.db.
Does not process the full 2,088-document corpus -- that is Phase B,
gated on this pilot's own validation report.
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

PROCESSED_DIR = REPO / "var" / "rag" / "processed"
OUT_DIR = REPO / "var" / "rag" / "structured"

PILOT_DOC_IDS = [
    # headnotes + JUDGMENT marker, quoted lower-court order (visitation
    # schedule) embedded mid-judgment -- the example from the structure audit.
    "8dd0ca3d14364404d75b76fe",
    # headnotes + ORDER marker, prose body.
    "00ad7248bf77883e6b5975ed",
    # no standalone heading at all, continuous prose, substantial length
    # (22K chars) -- confirmed during the audit to be a COMPLETE judgment
    # despite lacking a JUDGMENT/ORDER marker.
    "a3ec28ba4900855d15a6e870",
    # no headnotes, numbered-paragraph body.
    "1d0a3afecc805e3d71cb54b9",
    # no headnotes, ORDER marker, continuous prose -- the Khula case from
    # the very first disposition-count audit of this corpus.
    "0513f7c1139b729f4cccf83d",
    # multi-matter: main appeal allowed + two C.M.A.s with different
    # outcomes in the same tail paragraph. Also very long (111K chars)
    # and has high roman-numeral quotation density mid-document.
    "04139076ff36792b38b72f8b",
    # the single longest document in the corpus (193K chars, 44 headnote
    # points, 101 case citations, 60 sub-items) -- the complexity extreme.
    "177dbfa9f6115c8c4bb0cdcf",
    # incomplete source: ends mid-headnote at a counsel-listing line, no
    # judgment body was ever scraped.
    "f7afd3f61f59d8f9798b8742",
    # incomplete source: ends at "Date of hearing:", no judgment text at all.
    "b3b83bb2ceb23d88cf08394a",
    # non_judgment_text: an academic law-review article (footnote
    # bibliography tail) swept into the corpus by classification --
    # discovered during THIS implementation, not in the original audit's
    # named categories, but a real corpus phenomenon the structurer must
    # not force into a case-law shape.
    "ae4a07835728a54c3dddb4ad",
    # shortest document in the entire corpus (1,698 chars).
    "2ee9704f426892ca99e2b92e",
    # headnotes + JUDGMENT + genuinely numbered-paragraph body, for
    # contrast with the prose-body headnote+JUDGMENT example above.
    "008d928c2926b2a5855bcc36",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Pilot: {len(PILOT_DOC_IDS)} documents")
    results = []
    for i, doc_id in enumerate(PILOT_DOC_IDS, 1):
        in_path = PROCESSED_DIR / f"{doc_id}.json"
        if not in_path.exists():
            print(f"  [{i}] {doc_id}: MISSING from var/rag/processed/ -- skipped")
            continue
        doc = json.loads(in_path.read_text(encoding="utf-8"))

        t0 = time.perf_counter()
        result = structure_document(doc)
        elapsed = time.perf_counter() - t0

        output = to_output_document(doc, result)
        out_path = OUT_DIR / f"{doc_id}.json"
        tmp_path = out_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(out_path)

        print(f"  [{i}/{len(PILOT_DOC_IDS)}] {doc_id}: status={result.structure_status} "
              f"n_spans={len(result.spans)} used_llm={result.used_llm} "
              f"fallback={result.llm_fallback_reason} valid={result.validation['ok']} "
              f"elapsed={elapsed:.1f}s")
        results.append({
            "doc_id": doc_id, "status": result.structure_status,
            "n_spans": len(result.spans), "used_llm": result.used_llm,
            "llm_fallback_reason": result.llm_fallback_reason,
            "validation_ok": result.validation["ok"],
            "validation_errors": result.validation["errors"],
            "elapsed_seconds": round(elapsed, 2),
        })

    (OUT_DIR / "pilot_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nPILOT COMPLETE. Report: var/rag/structured/pilot_report.json")
    n_ok = sum(1 for r in results if r["validation_ok"])
    print(f"validation ok: {n_ok}/{len(results)}")


if __name__ == "__main__":
    main()
