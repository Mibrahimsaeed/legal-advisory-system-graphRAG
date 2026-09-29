#!/usr/bin/env python3
"""Phase 1 benchmark: does shortening the LLM case-text preview change anything?

Isolated by construction. It calls the PRODUCTION assessment path --
``src.classification.domain_assessment.assess_domain`` -- which already takes
``prompt_chars`` as a parameter, so the experiment needs no change to any
production module and no change to configuration. Nothing is written to
``var/metadata.db``: the database is opened read-only (``mode=ro``) to load
the sample, and every result goes to JSONL beside this file.

Every inference setting except ``prompt_chars`` is read from the live
configuration so the runs stay faithful to production.

Resumable: a doc_id already present in a results file is skipped, so an
interrupted run costs at most one document.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
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


def load_sample() -> list:
    doc_ids = json.loads((HERE / "sample_manifest.json").read_text())["doc_ids"]
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
    return [by_id[d] for d in doc_ids]          # manifest order, identical every run


def already_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["doc_id"])
    return done


def main() -> None:
    settings = get_settings()
    cfg = settings.domain_signals
    taxonomy = load_frozen_taxonomy(settings.classification.taxonomy_file)
    domain_block = render_domain_definitions(taxonomy)

    documents = load_sample()
    print(f"sample: {len(documents)} documents", flush=True)

    # Exactly the production client, exactly the production token budget.
    # ONE worker, sequential -- concurrency is deliberately not tested here.
    client = get_llm_client(cfg.llm_model, max_tokens=cfg.llm_max_tokens)
    meta = {
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
    }
    (HERE / "run_metadata.json").write_text(
        json.dumps({**meta, "started_at": datetime.now(timezone.utc).isoformat(),
                    "prompt_lengths": list(PROMPT_LENGTHS)}, indent=2), encoding="utf-8")
    print(f"settings: {meta}", flush=True)

    for prompt_chars in PROMPT_LENGTHS:
        out = HERE / f"results_{prompt_chars}.jsonl"
        done = already_done(out)
        pending = [d for d in documents if d.doc_id not in done]
        print(f"\n=== prompt_chars={prompt_chars}: {len(pending)} to do "
              f"({len(done)} already recorded) ===", flush=True)

        with out.open("a", encoding="utf-8") as fh:
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
                    "text_chars_sent": sent,          # not the text itself
                    "signal_text_chars": len(representation.signal_text),
                    "predicted_domain": assessment.domain,
                    "confidence": assessment.confidence,
                    "secondary_domains": [],          # the prompt forbids these
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
                    print(f"  {i}/{len(pending)}  last={elapsed:.1f}s "
                          f"domain={assessment.domain}", flush=True)

    print("\nBENCHMARK COMPLETE", flush=True)


if __name__ == "__main__":
    main()
