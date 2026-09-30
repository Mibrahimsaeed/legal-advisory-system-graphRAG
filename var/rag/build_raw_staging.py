#!/usr/bin/env python3
"""RAG raw staging, Family Law only.

    var/metadata.db  (SELECT, mode=ro -- structurally cannot write)
        -> classification_status='auto_accepted' AND primary_domain='family_law'
        -> load_case_folder() [existing, pure: reads case.html + metadata.json
           from disk only, never writes anywhere, never touches the DB]
        -> representation.full_text  (the COMPLETE extracted text --
           NFKC-normalized/whitespace-collapsed by normalize_case_text(),
           which is the HTML-side equivalent of basic decoding hygiene, NOT
           Phase 2's structural/procedural cleaning. No cause-list filtering,
           no drop logic, no chunking, no summarization is applied.)
        -> var/rag/raw/<doc_id>.json  {doc_id, metadata, full_text}

Reuses the project's own tested extraction function rather than writing a
second HTML parser, so "the complete original case.html content" means
exactly what Phase 1 itself means by it.

No classification logic runs. No signal/decision/review function is
imported or called. The DB connection never issues anything but SELECT,
and is opened via the mode=ro URI so a write would raise, not silently
succeed, if this script ever attempted one.

Resumable: a doc_id whose output JSON already exists is skipped.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.common.config import get_settings
from src.extraction.case_loader import load_case_folder

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "raw"
DB_PATH = REPO / "var" / "metadata.db"

SELECT_SQL = """
    SELECT doc_id, title, citation, court, decision_date, judges_json,
           case_number, primary_domain, classification_status,
           source_file, source_relpath, content_hash
      FROM document_representations
     WHERE classification_status = 'auto_accepted'
       AND primary_domain = 'family_law'
     ORDER BY doc_id
"""


def fetch_eligible() -> list[dict]:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(SELECT_SQL).fetchall()
    finally:
        con.close()
    return [dict(r) for r in rows]


def main() -> None:
    settings = get_settings()
    caselaw = settings.caselaw
    document = settings.document

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = fetch_eligible()
    print(f"Eligible (auto_accepted, family_law): {len(rows)}")

    written = 0
    skipped_existing = 0
    failed: list[dict] = []

    for row in rows:
        doc_id = row["doc_id"]
        out_path = OUT_DIR / f"{doc_id}.json"
        if out_path.exists():
            skipped_existing += 1
            continue

        source_file = row["source_file"]
        if not source_file:
            failed.append({"doc_id": doc_id, "reason": "no source_file recorded in DB"})
            continue

        case_folder = Path(source_file).parent

        representation = load_case_folder(
            case_folder,
            doc_id=doc_id,
            case_html_filename=caselaw.case_html_filename,
            metadata_filename=caselaw.metadata_filename,
            body_preview_char_limit=caselaw.body_preview_char_limit,
            max_headings=caselaw.max_headings,
            min_characters=document.min_characters,
        )

        if representation.status == "failed":
            failed.append({"doc_id": doc_id, "reason": representation.error,
                            "source_file": source_file})
            continue
        if not representation.full_text.strip():
            failed.append({"doc_id": doc_id, "reason": "empty full_text after extraction",
                            "source_file": source_file})
            continue

        try:
            judges = json.loads(row["judges_json"]) if row["judges_json"] else []
        except json.JSONDecodeError:
            judges = []

        payload = {
            "doc_id": doc_id,
            "metadata": {
                "title": row["title"],
                "citation": row["citation"],
                "court": row["court"],
                "decision_date": row["decision_date"],
                "judges": judges,
                "case_number": row["case_number"],
                "primary_domain": row["primary_domain"],
                "classification_status": row["classification_status"],
                "source_file": row["source_file"],
                "source_relpath": row["source_relpath"],
                "content_hash": row["content_hash"],
            },
            "full_text": representation.full_text,
        }

        tmp_path = out_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(out_path)
        written += 1

        if (written + skipped_existing) % 200 == 0:
            print(f"  progress: {written} written, {skipped_existing} already existed, "
                  f"{len(failed)} failed / {len(rows)} total")

    print(f"\nDone. eligible={len(rows)} written={written} "
          f"skipped_existing={skipped_existing} failed={len(failed)}")

    if failed:
        (HERE / "raw_staging_failures.json").write_text(
            json.dumps(failed, indent=2), encoding="utf-8"
        )
        print(f"Failures written to var/rag/raw_staging_failures.json ({len(failed)})")


if __name__ == "__main__":
    main()
