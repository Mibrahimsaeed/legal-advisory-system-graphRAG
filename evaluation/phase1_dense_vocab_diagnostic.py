#!/usr/bin/env python3
"""Phase 1 -- isolated dense-only vocabulary/authority-name diagnostic.

Isolates two factors that may affect dense retrieval of the Family Courts
Act 1964 S.10 Haq Mehr/dissolution proviso for a khula/dower query:
  (a) vocabulary substitution: "dower" -> "haq mehr"
  (b) statutory-authority injection: naming "Family Courts Act"

Reuses ONLY the existing, unmodified connect()/load_dense()/encode_dense()
from src.rag_prep.embed_index -- the same functions already used in
evaluation/diagnose_q01_dense_bm25.py and evaluation/q01_phrasing_experiment.py.
No BM25, no sparse vector, no hybrid fusion, no reranking anywhere in this
script. Every Qdrant call is a single, read-only, unfused query_points()
call using the "dense" named vector -- no upsert/delete/recreate call
exists in this file.

Target chunk identity is NOT re-derived by searching for a top result --
it is read directly from the prior diagnostic's own saved output
(evaluation/results/q01_dense_vs_bm25_diagnostic.json), which itself
confirmed the chunk's existence via a read-only Qdrant scroll.

    python evaluation/phase1_dense_vocab_diagnostic.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rag_prep.embed_index import (  # noqa: E402
    COLLECTION,
    DENSE_DIM,
    MODEL_NAME,
    connect,
    encode_dense,
    load_dense,
)

PRIOR_DIAGNOSTIC_PATH = Path("evaluation/results/q01_dense_vs_bm25_diagnostic.json")

QUERIES = [
    ("Q0", "Baseline", "Can a wife obtain khula without returning dower?"),
    ("Q1", "Replace dower with haq mehr", "Can a wife obtain khula without returning haq mehr?"),
    ("Q2", "Add Family Courts Act", "Can a wife obtain khula without returning dower under Family Courts Act?"),
    ("Q3", "Both changes", "Can a wife obtain khula without returning haq mehr under Family Courts Act?"),
]

N_STATUTE = 240  # same depth as the prior diagnostic this phase builds on
SOURCE_FILTER = "statute"

OUT_JSON = Path("evaluation/results/phase1_dense_vocab_diagnostic.json")
OUT_META = Path("evaluation/results/phase1_dense_vocab_diagnostic.meta.json")


def load_confirmed_target() -> tuple[str, str]:
    """Reads the target chunk_id/point_id from the prior diagnostic's own
    saved, already-verified output -- does not re-derive it from a fresh
    search. Fails loudly if that file is missing or doesn't confirm existence."""

    if not PRIOR_DIAGNOSTIC_PATH.exists():
        raise SystemExit(
            f"[phase1] cannot confirm target identity: {PRIOR_DIAGNOSTIC_PATH} not found. "
            "Stopping rather than guessing the target chunk."
        )
    prior = json.loads(PRIOR_DIAGNOSTIC_PATH.read_text(encoding="utf-8"))
    if not prior.get("target_exists_in_qdrant"):
        raise SystemExit(
            f"[phase1] prior diagnostic did not confirm the target exists in Qdrant "
            f"({PRIOR_DIAGNOSTIC_PATH}: target_exists_in_qdrant={prior.get('target_exists_in_qdrant')}). Stopping."
        )
    chunk_id = prior["target_chunk_id"]
    point_id = prior["target_point_id"]
    print(f"[phase1] target confirmed from {PRIOR_DIAGNOSTIC_PATH}: chunk_id={chunk_id} point_id={point_id}")
    return chunk_id, point_id


def git_status_short() -> str:
    try:
        out = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        return out.stdout
    except Exception as exc:  # noqa: BLE001
        return f"<could not run git status: {exc}>"


def run_dense_query(client, dense_model, query_text: str, statute_filter):
    vec = encode_dense(dense_model, [query_text], 1)[0].tolist()
    hits = client.query_points(
        COLLECTION, query=vec, using="dense", query_filter=statute_filter, limit=N_STATUTE, with_payload=True,
    ).points
    return hits


def summarize(hits, target_chunk_id: str) -> dict:
    target_rank = None
    target_score = None
    for i, h in enumerate(hits, 1):
        if h.payload.get("chunk_id") == target_chunk_id:
            target_rank, target_score = i, h.score
            break
    top10 = [
        {
            "rank": i,
            "point_id": str(h.id),
            "chunk_id": h.payload.get("chunk_id"),
            "score": h.score,
            "context_header": h.payload.get("context_header"),
            "canonical_text_preview": (h.payload.get("canonical_text") or "")[:160],
        }
        for i, h in enumerate(hits[:10], 1)
    ]
    return {
        "n_candidates_ranked": len(hits),
        "target_rank": target_rank,
        "target_score": target_score,
        "target_retrieved_within_depth": target_rank is not None,
        "target_in_top10": target_rank is not None and target_rank <= 10,
        "top10": top10,
    }


def main() -> None:
    from qdrant_client import models

    initial_git_status = git_status_short()
    print("[phase1] initial git status --short:")
    print(initial_git_status or "  (clean)")

    for p in (OUT_JSON, OUT_META):
        if p.exists():
            raise SystemExit(f"[phase1] refusing to overwrite existing path: {p}. Choose a new filename.")

    target_chunk_id, target_point_id = load_confirmed_target()

    client = connect()
    dense_model = load_dense()
    statute_filter = models.Filter(must=[models.FieldCondition(key="source_type", match=models.MatchValue(value=SOURCE_FILTER))])

    report: dict = {
        "experiment": "phase1_dense_vocab_diagnostic",
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "target_chunk_id": target_chunk_id,
        "target_point_id": target_point_id,
        "target_identity_source": str(PRIOR_DIAGNOSTIC_PATH),
        "model_name": MODEL_NAME,
        "dense_vector_name": "dense",
        "dense_dim": DENSE_DIM,
        "dense_distance": "Cosine",
        "query_embedding_normalized": True,
        "query_prefix_or_instruction": None,
        "collection": COLLECTION,
        "retrieval_depth": N_STATUTE,
        "source_filter": SOURCE_FILTER,
        "retrieval_method": "dense_only",
        "fusion": None,
        "sparse_model_used": None,
        "queries": [],
        "errors": [],
    }

    print(f"\n[phase1] running {len(QUERIES)} dense-only queries, depth={N_STATUTE}, filter=source_type:{SOURCE_FILTER}")
    for qid, change, qtext in QUERIES:
        entry = {"query_id": qid, "change_from_baseline": change, "query": qtext}
        try:
            hits = run_dense_query(client, dense_model, qtext, statute_filter)
            summary = summarize(hits, target_chunk_id)
            entry.update(summary)
            print(f"  {qid} ({change}): rank={summary['target_rank']} score={summary['target_score']} "
                  f"in_top10={summary['target_in_top10']} (n_ranked={summary['n_candidates_ranked']})")
        except Exception as exc:  # noqa: BLE001 -- isolate this query's failure, continue, report it
            entry["error"] = str(exc)
            report["errors"].append({"query_id": qid, "error": str(exc)})
            print(f"  {qid}: ERROR -- {exc}")
        report["queries"].append(entry)

    report["run_completed_successfully"] = not report["errors"]

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    final_git_status = git_status_short()
    meta = {
        "script_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "output_json": str(OUT_JSON),
        "output_meta": str(OUT_META),
        "target_identity_source": str(PRIOR_DIAGNOSTIC_PATH),
        "collection": COLLECTION,
        "model_name": MODEL_NAME,
        "dense_vector_name": "dense",
        "dense_dim": DENSE_DIM,
        "retrieval_depth": N_STATUTE,
        "source_filter": SOURCE_FILTER,
        "run_timestamp": report["run_timestamp"],
        "initial_git_status_short": initial_git_status,
        "final_git_status_short": final_git_status,
        "run_completed_successfully": report["run_completed_successfully"],
        "errors": report["errors"],
        "note": "final_git_status_short is expected to show exactly this script's own two new output "
                "files (plus whatever was already untracked before this run) and nothing else -- no "
                "tracked file should differ between initial and final status.",
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n[phase1] report -> {OUT_JSON}")
    print(f"[phase1] metadata -> {OUT_META}")
    print("\n[phase1] final git status --short:")
    print(final_git_status or "  (clean)")


if __name__ == "__main__":
    main()
