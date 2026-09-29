#!/usr/bin/env python3
"""Phase 1B, Part A: interleaved prompt-length runtime test.

Runs the SAME 100-document manifest as Phase 1 (sample_100.csv, unchanged),
but each document is put through all three prompt lengths back-to-back
before moving to the next document. This spreads any machine-throughput
drift evenly across all three conditions instead of confounding it with
prompt length, which is what happened when Phase 1 ran the three settings
as sequential blocks.

Order of the three settings within each document is randomized with a
fixed seed (random.Random(f"phase1b:{doc_id}")) and recorded per document,
so a systematic first/second/third-call bias is also averaged out rather
than always favouring one setting.

Calls the unmodified production path
(src.classification.domain_assessment.assess_domain) with prompt_chars
varied -- no other setting changes. DB is opened mode=ro. Nothing is
written to var/metadata.db. Resumable: a (doc_id, prompt_chars) pair
already in interleaved_results.jsonl is skipped.
"""

from __future__ import annotations

import json
import random
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
MANIFEST = HERE.parent / "sample_manifest.json"   # Phase 1's manifest, unchanged
PROMPT_LENGTHS = (3000, 1500, 1000)
OUT = HERE / "interleaved_results.jsonl"
ORDER_SEED = "phase1b-interleave"


def load_sample() -> list:
    doc_ids = json.loads(MANIFEST.read_text())["doc_ids"]
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
        raise SystemExit(f"manifest references {len(missing)} unknown doc_id(s)")
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

    documents = load_sample()
    print(f"sample: {len(documents)} documents (Phase 1's manifest, unchanged)", flush=True)

    client = get_llm_client(cfg.llm_model, max_tokens=cfg.llm_max_tokens)
    meta = {
        "phase": "1B-A-interleaved",
        "model": cfg.llm_model,
        "temperature": DEFAULT_TEMPERATURE,
        "seed": DEFAULT_SEED,
        "think": False,
        "json_format": True,
        "num_predict": cfg.llm_max_tokens,
        "assessment_version": ASSESSMENT_VERSION,
        "max_text_chars": cfg.max_text_chars,
        "max_headings": cfg.max_headings,
        "workers": 1,
        "prompt_lengths": list(PROMPT_LENGTHS),
        "order_randomization_seed": ORDER_SEED,
        "manifest": str(MANIFEST),
    }
    (HERE / "run_metadata_interleaved.json").write_text(
        json.dumps({**meta, "started_at": datetime.now(timezone.utc).isoformat()}, indent=2),
        encoding="utf-8",
    )
    print(f"settings: {meta}", flush=True)

    done = already_done()
    order_log = []

    with OUT.open("a", encoding="utf-8") as fh:
        for i, document in enumerate(documents, 1):
            rng = random.Random(f"{ORDER_SEED}:{document.doc_id}")
            order = list(PROMPT_LENGTHS)
            rng.shuffle(order)
            order_log.append({"doc_id": document.doc_id, "call_order": order})

            for position, prompt_chars in enumerate(order, 1):
                key = (document.doc_id, prompt_chars)
                if key in done:
                    continue

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
                    "call_position_within_doc": position,   # 1st/2nd/3rd call for this doc
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

            if i % 10 == 0 or i == len(documents):
                print(f"  {i}/{len(documents)} documents done "
                      f"(order for last: {order})", flush=True)

    (HERE / "call_order_log.json").write_text(
        json.dumps(order_log, indent=2), encoding="utf-8"
    )
    print("\nINTERLEAVED BENCHMARK COMPLETE", flush=True)


if __name__ == "__main__":
    main()
