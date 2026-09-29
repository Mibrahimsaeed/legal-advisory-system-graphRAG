#!/usr/bin/env python3
"""Phase 1B, Part B: the 63 human-labelled documents at each prompt length.

Documents come from the 23-case review ledger (document_review_decisions)
plus the 40-case blind_validation_40.csv verified_domain column -- the
labels are used exactly as recorded, never modified or invented.

Same production path as Part A (assess_domain with prompt_chars varied),
same pinned settings, 1 worker, DB opened mode=ro. Nothing written to
var/metadata.db. Resumable per (doc_id, prompt_chars).
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.classification.case_representation import build_case_representation
from src.classification.domain_assessment import (
    ASSESSMENT_VERSION,
    assess_domain,
    render_domain_definitions,
)
from src.classification.taxonomy_registry import load_frozen_taxonomy
from src.common.config import get_settings
from src.common.llm_client import DEFAULT_SEED, DEFAULT_TEMPERATURE, get_llm_client
from src.extraction.representation_store import _row_to_representation

HERE = Path(__file__).resolve().parent
PROMPT_LENGTHS = (3000, 1500, 1000)
OUT = HERE / "human_label_results.jsonl"


def load_human_docs() -> list:
    labels = json.loads((HERE / "human_labels.json").read_text())["labels"]
    doc_ids = sorted(labels)
    con = sqlite3.connect(f"file:{REPO / 'var/metadata.db'}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    marks = ",".join("?" for _ in doc_ids)
    rows = con.execute(
        f"SELECT * FROM document_representations WHERE doc_id IN ({marks})", doc_ids
    ).fetchall()
    con.close()
    by_id = {r["doc_id"]: _row_to_representation(r) for r in rows}
    missing = [d for d in doc_ids if d not in by_id]
    if missing:
        raise SystemExit(f"human_labels.json references {len(missing)} unknown doc_id(s): {missing}")
    return [by_id[d] for d in doc_ids]


def already_done() -> set[tuple[str, int]]:
    if not OUT.exists():
        return set()
    done = set()
    for line in OUT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            done.add((r["doc_id"], r["prompt_chars"]))
    return done


def main() -> None:
    settings = get_settings()
    cfg = settings.domain_signals
    taxonomy = load_frozen_taxonomy(settings.classification.taxonomy_file)
    domain_block = render_domain_definitions(taxonomy)

    documents = load_human_docs()
    print(f"human-labelled sample: {len(documents)} documents", flush=True)

    client = get_llm_client(cfg.llm_model, max_tokens=cfg.llm_max_tokens)
    meta = {
        "phase": "1B-B-human-labelled",
        "model": cfg.llm_model,
        "temperature": DEFAULT_TEMPERATURE,
        "seed": DEFAULT_SEED,
        "think": False,
        "json_format": True,
        "num_predict": cfg.llm_max_tokens,
        "assessment_version": ASSESSMENT_VERSION,
        "workers": 1,
        "prompt_lengths": list(PROMPT_LENGTHS),
        "n_documents": len(documents),
    }
    (HERE / "run_metadata_human.json").write_text(
        json.dumps({**meta, "started_at": datetime.now(timezone.utc).isoformat()}, indent=2),
        encoding="utf-8",
    )
    print(f"settings: {meta}", flush=True)

    done = already_done()

    with OUT.open("a", encoding="utf-8") as fh:
        for prompt_chars in PROMPT_LENGTHS:
            pending = [d for d in documents if (d.doc_id, prompt_chars) not in done]
            print(f"\n=== prompt_chars={prompt_chars}: {len(pending)} to do "
                  f"({len(documents) - len(pending)} already recorded) ===", flush=True)

            for i, document in enumerate(pending, 1):
                representation = build_case_representation(
                    document, max_text_chars=cfg.max_text_chars,
                    max_headings=cfg.max_headings,
                )
                sent = len(representation.prompt_text(prompt_chars))

                t0 = time.perf_counter()
                assessment = assess_domain(
                    representation, taxonomy, client,
                    domain_block=domain_block,
                    prompt_chars=prompt_chars,
                    model_name=cfg.llm_model,
                )
                elapsed = time.perf_counter() - t0

                fh.write(json.dumps({
                    "doc_id": document.doc_id,
                    "prompt_chars": prompt_chars,
                    "text_chars_sent": sent,
                    "signal_text_chars": len(representation.signal_text),
                    "predicted_domain": assessment.domain,
                    "confidence": assessment.confidence,
                    "justification": assessment.reason,
                    "elapsed_seconds": round(elapsed, 3),
                    "success": not assessment.failed,
                    "error": assessment.error,
                    "model": cfg.llm_model,
                    "num_predict": cfg.llm_max_tokens,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }, ensure_ascii=False) + "\n")
                fh.flush()

                if i % 10 == 0 or i == len(pending):
                    print(f"  {i}/{len(pending)}  last={elapsed:.1f}s domain={assessment.domain}",
                          flush=True)

    print("\nHUMAN-LABEL BENCHMARK COMPLETE", flush=True)


if __name__ == "__main__":
    main()
