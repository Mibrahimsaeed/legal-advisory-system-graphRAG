#!/usr/bin/env python3
"""Concurrency Phase 1: 1 worker vs 2 concurrent workers at prompt_chars=1500.

Reuses Phase 1B's validated methodology and its exact 100-document sample
(var/benchmarks/prompt_length/sample_manifest.json -- not re-sampled).
prompt_chars is fixed at 1500 -- the one Phase 1B validated as
quality-neutral -- so this test isolates concurrency, not prompt length.

Calls the unmodified production path
(src.classification.domain_assessment.assess_domain) with an httpx-backed
OllamaLLMClient SHARED across worker threads (httpx.Client is documented
thread-safe and pools connections), which mirrors what a minimal
concurrency change to the real pipeline would look like: one client,
multiple in-flight requests. No production file is read for anything but
its already-public interface; nothing is written to var/metadata.db (the
DB is opened mode=ro to load the sample only, exactly once, up front).

Two configs run back-to-back, same order every time: 1 worker (sequential,
matches Phase 1B's methodology exactly), then 2 workers
(ThreadPoolExecutor(max_workers=2), one client instance shared). Each
config's WALL-CLOCK batch time is what answers "how much faster" --
individual call latencies are also recorded for mean/median/P90.

KNOWN CONSTRAINT, disclosed rather than worked around: the Ollama backend
observed running for this benchmark (`llama-server ... -np 1`) has only
ONE parallel inference slot. Two concurrent HTTP requests reaching that
server are expected to queue at the server rather than execute
side-by-side. This script does not change that -- Ollama's own
configuration is out of scope -- it measures the client-observed
consequence.

Resumable: a doc_id already recorded for a given worker-count config is
skipped on re-run.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
MANIFEST = REPO / "var/benchmarks/prompt_length/sample_manifest.json"   # Phase 1B's, unchanged
PROMPT_CHARS = 1500                                                     # fixed, per Phase 1B's finding
WORKER_CONFIGS = (1, 2)


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


def already_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        json.loads(l)["doc_id"]
        for l in path.read_text(encoding="utf-8").splitlines() if l.strip()
    }


def assess_one(document, taxonomy, client, domain_block, cfg):
    representation = build_case_representation(
        document, max_text_chars=cfg.max_text_chars, max_headings=cfg.max_headings
    )
    t0 = time.perf_counter()
    assessment = assess_domain(
        representation, taxonomy, client,
        domain_block=domain_block, prompt_chars=PROMPT_CHARS, model_name=cfg.llm_model,
    )
    elapsed = time.perf_counter() - t0
    return {
        "doc_id": document.doc_id,
        "prompt_chars": PROMPT_CHARS,
        "predicted_domain": assessment.domain,
        "confidence": assessment.confidence,
        "justification": assessment.reason,
        "elapsed_seconds": round(elapsed, 3),
        "success": not assessment.failed,
        "error": assessment.error,
        "model": cfg.llm_model,
        "num_predict": cfg.llm_max_tokens,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_config(n_workers: int, documents, taxonomy, domain_block, cfg) -> dict:
    out = HERE / f"results_{n_workers}worker.jsonl"
    done = already_done(out)
    pending = [d for d in documents if d.doc_id not in done]
    print(f"\n=== {n_workers}-worker config: {len(pending)} to do "
          f"({len(done)} already recorded) ===", flush=True)

    client = get_llm_client(cfg.llm_model, max_tokens=cfg.llm_max_tokens)
    write_lock = threading.Lock()
    completed = 0

    batch_start = time.perf_counter()
    if not pending:
        batch_wall_seconds = 0.0
    elif n_workers == 1:
        with out.open("a", encoding="utf-8") as fh:
            for document in pending:
                record = assess_one(document, taxonomy, client, domain_block, cfg)
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
                completed += 1
                if completed % 10 == 0 or completed == len(pending):
                    print(f"  {completed}/{len(pending)}  last={record['elapsed_seconds']:.1f}s "
                          f"domain={record['predicted_domain']}", flush=True)
        batch_wall_seconds = time.perf_counter() - batch_start
    else:
        with out.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(assess_one, document, taxonomy, client, domain_block, cfg): document
                for document in pending
            }
            for future in as_completed(futures):
                record = future.result()
                with write_lock:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fh.flush()
                    completed += 1
                    if completed % 10 == 0 or completed == len(pending):
                        print(f"  {completed}/{len(pending)}  last={record['elapsed_seconds']:.1f}s "
                              f"domain={record['predicted_domain']}", flush=True)
        batch_wall_seconds = time.perf_counter() - batch_start

    return {
        "n_workers": n_workers,
        "n_pending_this_run": len(pending),
        "n_already_done": len(done),
        "batch_wall_seconds_this_run": round(batch_wall_seconds, 2),
    }


def main() -> None:
    settings = get_settings()
    cfg = settings.domain_signals
    taxonomy = load_frozen_taxonomy(settings.classification.taxonomy_file)
    domain_block = render_domain_definitions(taxonomy)

    documents = load_sample()
    print(f"sample: {len(documents)} documents (Phase 1B's manifest, unchanged)", flush=True)

    meta = {
        "phase": "concurrency_phase1",
        "model": cfg.llm_model,
        "temperature": DEFAULT_TEMPERATURE,
        "seed": DEFAULT_SEED,
        "think": False,
        "json_format": True,
        "num_predict": cfg.llm_max_tokens,
        "prompt_chars": PROMPT_CHARS,
        "assessment_version": ASSESSMENT_VERSION,
        "worker_configs": list(WORKER_CONFIGS),
        "manifest": str(MANIFEST),
        "n_documents": len(documents),
        "note": (
            "Ollama backend observed for this run may have been launched with "
            "llama-server -np 1 (one parallel inference slot); see run log / "
            "ollama_server_check.json for the actual value captured at run time."
        ),
    }
    (HERE / "run_metadata.json").write_text(
        json.dumps({**meta, "started_at": datetime.now(timezone.utc).isoformat()}, indent=2),
        encoding="utf-8",
    )
    print(f"settings: {meta}", flush=True)

    batch_results = []
    for n_workers in WORKER_CONFIGS:
        batch_results.append(run_config(n_workers, documents, taxonomy, domain_block, cfg))

    (HERE / "batch_wall_clock.json").write_text(json.dumps(batch_results, indent=2), encoding="utf-8")
    print("\nCONCURRENCY BENCHMARK COMPLETE", flush=True)
    print(json.dumps(batch_results, indent=2))


if __name__ == "__main__":
    main()
