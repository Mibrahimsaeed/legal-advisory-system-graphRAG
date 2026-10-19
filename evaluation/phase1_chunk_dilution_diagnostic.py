#!/usr/bin/env python3
"""Phase 1, Test 2 -- isolated chunk-dilution diagnostic.

Tests whether Family Courts Act 1964 S.10's poor dense rank (81/240 for Q0)
is consistent with the target chunk's substantive Haq Mehr/dissolution
proviso being semantically diluted by three preceding, unrelated
pre-trial-procedure subsections.

Reuses ONLY connect()/load_dense()/encode_dense() from
src.rag_prep.embed_index, unmodified. The ONLY Qdrant operations are
read-only: client.retrieve() (stored vector, by point_id) and
client.query_points(..., using="dense", ...) (the same unfused dense-only
call already used and validated in evaluation/diagnose_q01_dense_bm25.py
and evaluation/phase1_dense_vocab_diagnostic.py). No point is ever
inserted, upserted, or deleted; the isolated-passage embedding exists
only in this process's memory and in this script's own output file.

    python evaluation/phase1_chunk_dilution_diagnostic.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

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
STATUTE_CHUNK_FILE = Path("var/rag/statutes_chunked/24520c344d2285555aa10988.json")
Q0 = "Can a wife obtain khula without returning dower?"
N_STATUTE = 240
SOURCE_FILTER = "statute"
PROVISO_ANCHOR = "Provided that"  # first word of the substantive rule, used to locate it exactly

OUT_JSON = Path("evaluation/results/phase1_chunk_dilution_diagnostic.json")
OUT_META = Path("evaluation/results/phase1_chunk_dilution_diagnostic.meta.json")


def git_status_short() -> str:
    out = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return out.stdout


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_confirmed_target() -> tuple[str, str]:
    if not PRIOR_DIAGNOSTIC_PATH.exists():
        raise SystemExit(f"[phase1-dilution] cannot confirm target identity: {PRIOR_DIAGNOSTIC_PATH} not found. Stopping.")
    prior = json.loads(PRIOR_DIAGNOSTIC_PATH.read_text(encoding="utf-8"))
    if not prior.get("target_exists_in_qdrant"):
        raise SystemExit(f"[phase1-dilution] prior diagnostic did not confirm target existence. Stopping.")
    return prior["target_chunk_id"], prior["target_point_id"]


def load_target_chunk(chunk_id: str) -> dict:
    """Reads the chunk's own stored fields from the existing chunk file --
    never reconstructs or retypes the text."""

    if not STATUTE_CHUNK_FILE.exists():
        raise SystemExit(f"[phase1-dilution] source chunk file not found: {STATUTE_CHUNK_FILE}. Stopping.")
    data = json.loads(STATUTE_CHUNK_FILE.read_text(encoding="utf-8"))
    for c in data["chunks"]:
        if c["chunk_id"] == chunk_id:
            return c
    raise SystemExit(f"[phase1-dilution] chunk_id {chunk_id!r} not found in {STATUTE_CHUNK_FILE}. Stopping.")


def extract_isolated_proviso(chunk: dict) -> dict:
    """Locates the substantive Haq Mehr/dissolution proviso within the
    chunk's own already-verified `embedding_text` field (the same
    bracket-stripped form the production pipeline actually embeds -- not
    a newly invented cleaning rule) and, separately, the corresponding raw
    span in `canonical_text` for provenance/offset reporting."""

    canonical_text = chunk["canonical_text"]
    embedding_text = chunk["embedding_text"]

    if PROVISO_ANCHOR not in embedding_text:
        raise SystemExit(f"[phase1-dilution] anchor {PROVISO_ANCHOR!r} not found in embedding_text -- cannot isolate the proviso unambiguously. Stopping.")
    if PROVISO_ANCHOR not in canonical_text:
        raise SystemExit(f"[phase1-dilution] anchor {PROVISO_ANCHOR!r} not found in canonical_text -- cannot locate the raw source span. Stopping.")

    clean_start = embedding_text.index(PROVISO_ANCHOR)
    isolated_proviso_clean = embedding_text[clean_start:].strip()

    raw_start = canonical_text.index(PROVISO_ANCHOR)
    # the proviso is the final sentence in this chunk -- its raw span runs to the end of canonical_text
    raw_source_span = canonical_text[raw_start:]

    return {
        "isolated_proviso_clean": isolated_proviso_clean,
        "raw_source_span": raw_source_span,
        "raw_span_char_offset_start": raw_start,
        "raw_span_char_offset_end": len(canonical_text),
        "extraction_rationale": (
            "The proviso sentence ('Provided that ... restore to the husband the Haq Mehr ...') "
            "is the final, self-contained sentence of the chunk: it already states WHO (the Family "
            "Court), the CONDITION (a suit for dissolution of marriage, if reconciliation fails), and "
            "the RULE (decree the dissolution and restore the Haq Mehr). Subsections (1)-(3), which "
            "precede it, are generic pre-trial procedure (written statement, hearing dates, examining "
            "the plaint, attempting compromise) with no bearing on the Haq Mehr/khula question and are "
            "excluded as the 'unrelated procedural boilerplate' the task instructed against including "
            "unless necessary -- it is not necessary here, since the proviso's own wording ('if "
            "reconciliation fails') already supplies the needed condition without them."
        ),
    }


def main() -> None:
    initial_git_status = git_status_short()
    print("[phase1-dilution] initial git status --short:")
    print(initial_git_status or "  (clean)")

    for p in (OUT_JSON, OUT_META):
        if p.exists():
            raise SystemExit(f"[phase1-dilution] refusing to overwrite existing path: {p}.")

    target_chunk_id, target_point_id = load_confirmed_target()
    print(f"[phase1-dilution] target confirmed from {PRIOR_DIAGNOSTIC_PATH}: chunk_id={target_chunk_id} point_id={target_point_id}")

    chunk = load_target_chunk(target_chunk_id)
    extraction = extract_isolated_proviso(chunk)
    isolated_proviso_clean = extraction["isolated_proviso_clean"]

    # Isolated passage for embedding: same context_header prefix the production pipeline
    # always prepends (holding that variable constant), followed by ONLY the proviso --
    # this isolates exactly one variable (presence of the 3 unrelated subsections).
    isolated_passage_text = f"{chunk['context_header']}\n{isolated_proviso_clean}"

    lengths = {
        "original_canonical_text_chars": len(chunk["canonical_text"]),
        "original_embedding_text_chars": len(chunk["embedding_text"]),
        "original_token_count_stored_field": chunk.get("token_count"),
        "isolated_proviso_clean_chars": len(isolated_proviso_clean),
        "isolated_passage_text_chars": len(isolated_passage_text),
        "raw_source_span_chars": len(extraction["raw_source_span"]),
    }

    print(f"[phase1-dilution] original embedding_text: {lengths['original_embedding_text_chars']} chars "
          f"(stored token_count={lengths['original_token_count_stored_field']})")
    print(f"[phase1-dilution] isolated passage text: {lengths['isolated_passage_text_chars']} chars")
    print(f"[phase1-dilution] isolated proviso (clean) is {lengths['isolated_proviso_clean_chars']} of "
          f"{lengths['original_embedding_text_chars']} embedding_text chars "
          f"({100 * lengths['isolated_proviso_clean_chars'] / lengths['original_embedding_text_chars']:.1f}%)")

    client = connect()
    dense_model = load_dense()

    # Q0 query embedding -- same pathway used everywhere else in this project
    q_vec = encode_dense(dense_model, [Q0], 1)[0].astype(np.float64)

    # --- A. Direct embedding similarity -------------------------------------------------
    direct: dict = {}
    retrieved = client.retrieve(COLLECTION, ids=[target_point_id], with_vectors=True)
    if not retrieved or "dense" not in (retrieved[0].vector or {}):
        direct["stored_vector_retrieved"] = False
        direct["limitation"] = "could not retrieve the stored 'dense' vector by point_id -- stored-vector comparison unavailable"
        stored_sim = None
        print("[phase1-dilution] LIMITATION: could not retrieve stored dense vector for the target point_id")
    else:
        stored_vec = np.array(retrieved[0].vector["dense"], dtype=np.float64)
        stored_sim = cosine(q_vec, stored_vec)
        direct["stored_vector_retrieved"] = True
        direct["stored_vector_cosine_similarity_to_q0"] = stored_sim

    isolated_vec = encode_dense(dense_model, [isolated_passage_text], 1)[0].astype(np.float64)
    isolated_sim = cosine(q_vec, isolated_vec)
    direct["isolated_passage_cosine_similarity_to_q0"] = isolated_sim
    if stored_sim is not None:
        direct["similarity_difference_isolated_minus_stored"] = isolated_sim - stored_sim

    print(f"[phase1-dilution] DIRECT SIMILARITY: stored_vector={stored_sim}  isolated_passage={isolated_sim:.6f}"
          + (f"  diff={isolated_sim - stored_sim:+.6f}" if stored_sim is not None else ""))

    # --- Optional control (section 6): locally re-embed the full original chunk ----------
    local_full_vec = encode_dense(dense_model, [chunk["embedding_text"]], 1)[0].astype(np.float64)
    local_full_sim = cosine(q_vec, local_full_vec)
    control = {
        "local_full_chunk_cosine_similarity_to_q0": local_full_sim,
        "stored_vs_local_full_chunk_similarity_diff": (local_full_sim - stored_sim) if stored_sim is not None else None,
    }
    if stored_sim is not None:
        # cross-validate the local pipeline against the known production rank/score for Q0 (0.3825, rank 81)
        print(f"[phase1-dilution] CONTROL: locally re-embedded full chunk similarity={local_full_sim:.6f} "
              f"vs. stored vector similarity={stored_sim:.6f} (diff={local_full_sim - stored_sim:+.6f})")

    # --- B. Controlled retrieval comparison -------------------------------------------
    from qdrant_client import models

    statute_filter = models.Filter(must=[models.FieldCondition(key="source_type", match=models.MatchValue(value=SOURCE_FILTER))])
    hits = client.query_points(
        COLLECTION, query=q_vec.tolist(), using="dense", query_filter=statute_filter, limit=N_STATUTE, with_payload=True,
    ).points

    production_rank = next((i for i, h in enumerate(hits, 1) if h.payload.get("chunk_id") == target_chunk_id), None)
    production_score = next((h.score for h in hits if h.payload.get("chunk_id") == target_chunk_id), None)

    # sanity cross-check: local cosine on the stored vector should match Qdrant's own returned score
    # for this same point (Cosine-distance collections report similarity directly, not distance)
    score_cross_check = None
    if stored_sim is not None and production_score is not None:
        score_cross_check = abs(stored_sim - production_score)

    candidates = [{"chunk_id": h.payload.get("chunk_id"), "point_id": str(h.id), "score": h.score,
                   "context_header": h.payload.get("context_header"), "is_experimental": False} for h in hits]
    candidates.append({
        "chunk_id": f"{target_chunk_id}::ISOLATED_PROVISO_EXPERIMENTAL", "point_id": None,
        "score": isolated_sim, "context_header": chunk["context_header"], "is_experimental": True,
    })
    candidates.sort(key=lambda c: c["score"], reverse=True)
    experimental_rank = next(i for i, c in enumerate(candidates, 1) if c["is_experimental"])

    print(f"[phase1-dilution] PRODUCTION (reproduced): rank={production_rank} score={production_score}")
    print(f"[phase1-dilution] EXPERIMENTAL COMPARATIVE RANK of isolated proviso among "
          f"{len(hits)} real results + 1 experimental candidate: {experimental_rank}")

    top10_comparative = candidates[:10]

    # --- assemble report -----------------------------------------------------------------
    report = {
        "experiment": "phase1_chunk_dilution_diagnostic",
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script_version": 1,
        "target_chunk_id": target_chunk_id,
        "target_point_id": target_point_id,
        "target_identity_source": str(PRIOR_DIAGNOSTIC_PATH),
        "query": Q0,
        "model_name": MODEL_NAME,
        "dense_dim": DENSE_DIM,
        "similarity_metric": "cosine",
        "source_statute_file": str(STATUTE_CHUNK_FILE),
        "original_chunk": {
            "canonical_text": chunk["canonical_text"],
            "embedding_text": chunk["embedding_text"],
            "context_header": chunk.get("context_header"),
            "act_name": chunk.get("act_name"),
            "section_number": chunk.get("section_number"),
            "section_title": chunk.get("section_title"),
        },
        "isolated_passage": {
            "raw_source_span": extraction["raw_source_span"],
            "raw_span_char_offset_start": extraction["raw_span_char_offset_start"],
            "raw_span_char_offset_end": extraction["raw_span_char_offset_end"],
            "isolated_proviso_clean": isolated_proviso_clean,
            "isolated_passage_text_embedded": isolated_passage_text,
            "extraction_rationale": extraction["extraction_rationale"],
            "bracket_stripping_note": (
                "The raw source span retains this project's own internal amendment-footnote bracket "
                "markup (e.g. '[^7.1][...]'), which is not part of the original statutory wording. "
                "The clean/embedded form strips it by taking the exact substring already present in "
                "this chunk's own stored `embedding_text` field (produced by the existing, unmodified "
                "chunking pipeline) -- no new cleaning rule was invented for this experiment."
            ),
        },
        "lengths": lengths,
        "direct_embedding_similarity": direct,
        "local_full_chunk_control": control,
        "production_rank_reproduced": {"rank": production_rank, "score": production_score, "n_candidates_ranked": len(hits)},
        "score_cross_check_stored_vs_qdrant_returned": score_cross_check,
        "experimental_comparative_rank": experimental_rank,
        "experimental_comparative_candidate_pool_size": len(candidates),
        "top10_comparative_candidates": top10_comparative,
        "errors": [],
        "limitations": [],
    }
    if stored_sim is None:
        report["limitations"].append("stored production vector could not be retrieved; direct-similarity comparison against it is unavailable")

    report["run_completed_successfully"] = True

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
        "run_timestamp": report["run_timestamp"],
        "initial_git_status_short": initial_git_status,
        "final_git_status_short": final_git_status,
        "run_completed_successfully": True,
        "note": "No point was inserted/upserted/deleted in the production collection at any point; "
                "the only Qdrant calls were client.retrieve() (read) and client.query_points() (read).",
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n[phase1-dilution] report -> {OUT_JSON}")
    print(f"[phase1-dilution] metadata -> {OUT_META}")


if __name__ == "__main__":
    main()
