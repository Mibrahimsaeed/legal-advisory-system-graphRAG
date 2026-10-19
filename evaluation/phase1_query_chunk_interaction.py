#!/usr/bin/env python3
"""Phase 1, Test 3 -- query-chunk interaction diagnostic.

Computes the full 4-query x 2-document-representation cosine-similarity
matrix (D0 = original target chunk's production embedding text, D1 = the
exact isolated proviso from Test 2) and derives isolation, vocabulary,
authority-name, and interaction effects from it.

Reuses ONLY connect()/load_dense()/encode_dense() from
src.rag_prep.embed_index, unmodified. D0's similarity always uses the
STORED production vector (fetched once via a single read-only
client.retrieve() call); D1's embedding is generated once, locally, from
the exact text already extracted and saved in Test 2's own JSON output --
not re-extracted or re-paraphrased. The only other Qdrant calls are the
same read-only, unfused dense-only query_points() calls (using="dense",
no prefetch/fusion) already used and validated in the three prior Phase 1
scripts, used here only for the optional Section 7 retrieval comparison.

    python evaluation/phase1_query_chunk_interaction.py
"""

from __future__ import annotations

import json
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

TEST1_PATH = Path("evaluation/results/phase1_dense_vocab_diagnostic.json")
TEST2_PATH = Path("evaluation/results/phase1_chunk_dilution_diagnostic.json")
N_STATUTE = 240
SOURCE_FILTER = "statute"

QUERIES = [
    ("Q0", "Baseline", "Can a wife obtain khula without returning dower?"),
    ("Q1", "Vocabulary substitution", "Can a wife obtain khula without returning haq mehr?"),
    ("Q2", "Authority injection", "Can a wife obtain khula without returning dower under Family Courts Act?"),
    ("Q3", "Both modifications", "Can a wife obtain khula without returning haq mehr under Family Courts Act?"),
]

OUT_JSON = Path("evaluation/results/phase1_query_chunk_interaction.json")
OUT_META = Path("evaluation/results/phase1_query_chunk_interaction.meta.json")


def git_status_short() -> str:
    out = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return out.stdout


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_prior_test(path: Path, label: str) -> dict:
    if not path.exists():
        raise SystemExit(f"[phase1-interaction] cannot proceed: {label} artifact not found at {path}. Stopping.")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    initial_git_status = git_status_short()
    print("[phase1-interaction] initial git status --short:")
    print(initial_git_status or "  (clean)")

    for p in (OUT_JSON, OUT_META):
        if p.exists():
            raise SystemExit(f"[phase1-interaction] refusing to overwrite existing path: {p}.")

    test1 = load_prior_test(TEST1_PATH, "Test 1 (vocabulary/authority)")
    test2 = load_prior_test(TEST2_PATH, "Test 2 (chunk dilution)")

    target_chunk_id = test2["target_chunk_id"]
    target_point_id = test2["target_point_id"]
    d0_embedding_text = test2["original_chunk"]["embedding_text"]
    d1_text = test2["isolated_passage"]["isolated_passage_text_embedded"]
    prior_stored_sim_q0 = test2["direct_embedding_similarity"]["stored_vector_cosine_similarity_to_q0"]

    if not d0_embedding_text or not d1_text:
        raise SystemExit("[phase1-interaction] Test 2 artifact does not contain the expected embedding texts. Stopping.")

    print(f"[phase1-interaction] target: chunk_id={target_chunk_id} point_id={target_point_id}")
    print(f"[phase1-interaction] D0 (original) embedding_text: {len(d0_embedding_text)} chars, from {TEST2_PATH}")
    print(f"[phase1-interaction] D1 (isolated) embedding_text: {len(d1_text)} chars, from {TEST2_PATH} "
          f"(verbatim re-use, not re-extracted)")

    client = connect()
    dense_model = load_dense()

    # D0: fetch the stored production vector exactly once (read-only)
    retrieved = client.retrieve(COLLECTION, ids=[target_point_id], with_vectors=True)
    if not retrieved or "dense" not in (retrieved[0].vector or {}):
        raise SystemExit("[phase1-interaction] could not retrieve the stored 'dense' vector for the target point_id. Stopping.")
    d0_vec = np.array(retrieved[0].vector["dense"], dtype=np.float64)

    # D1: generate the experimental embedding exactly once (document encoding is query-independent)
    d1_vec = encode_dense(dense_model, [d1_text], 1)[0].astype(np.float64)

    # Four query embeddings
    query_vecs: dict[str, np.ndarray] = {}
    for qid, _, qtext in QUERIES:
        query_vecs[qid] = encode_dense(dense_model, [qtext], 1)[0].astype(np.float64)

    # 4x2 similarity matrix
    sim: dict[str, dict[str, float]] = {}
    for qid, _, _ in QUERIES:
        sim[qid] = {
            "D0": cosine(query_vecs[qid], d0_vec),
            "D1": cosine(query_vecs[qid], d1_vec),
        }

    # Validation: reproduce Test 2's Q0/D0 figure
    q0_d0 = sim["Q0"]["D0"]
    validation_diff = abs(q0_d0 - prior_stored_sim_q0)
    validation_ok = validation_diff < 1e-4
    print(f"\n[phase1-interaction] VALIDATION: sim(Q0,D0)={q0_d0:.6f} vs. Test 2's recorded {prior_stored_sim_q0:.6f} "
          f"(diff={validation_diff:.2e}) -> {'OK' if validation_ok else 'MISMATCH -- investigate before interpreting'}")

    print("\n[phase1-interaction] SIMILARITY MATRIX")
    print(f"{'Query':<6}{'D0 (original)':>16}{'D1 (isolated)':>16}")
    for qid, _, _ in QUERIES:
        print(f"{qid:<6}{sim[qid]['D0']:>16.6f}{sim[qid]['D1']:>16.6f}")

    # --- A. Isolation effect per query ---
    isolation_effect = {qid: sim[qid]["D1"] - sim[qid]["D0"] for qid, _, _ in QUERIES}

    # --- B. Vocabulary substitution effect per document ---
    vocab_effect = {doc: sim["Q1"][doc] - sim["Q0"][doc] for doc in ("D0", "D1")}
    vocab_effect_with_authority = {doc: sim["Q3"][doc] - sim["Q2"][doc] for doc in ("D0", "D1")}

    # --- C. Authority-name effect per document ---
    authority_effect = {doc: sim["Q2"][doc] - sim["Q0"][doc] for doc in ("D0", "D1")}
    authority_effect_with_haqmehr = {doc: sim["Q3"][doc] - sim["Q1"][doc] for doc in ("D0", "D1")}

    # --- D. Interaction contrast per document ---
    interaction = {
        doc: sim["Q3"][doc] - sim["Q1"][doc] - sim["Q2"][doc] + sim["Q0"][doc]
        for doc in ("D0", "D1")
    }

    # --- E. Difference in isolation effects ---
    isolation_diff_q3_minus_q0 = isolation_effect["Q3"] - isolation_effect["Q0"]

    print("\n[phase1-interaction] EFFECTS")
    print(f"  isolation effect per query: {isolation_effect}")
    print(f"  vocabulary effect (Q1-Q0) per doc: {vocab_effect}")
    print(f"  vocabulary effect with authority present (Q3-Q2) per doc: {vocab_effect_with_authority}")
    print(f"  authority effect (Q2-Q0) per doc: {authority_effect}")
    print(f"  authority effect with haq mehr present (Q3-Q1) per doc: {authority_effect_with_haqmehr}")
    print(f"  interaction contrast per doc: {interaction}")
    print(f"  isolation effect difference (Q3 - Q0): {isolation_diff_q3_minus_q0:+.6f}")

    # --- Section 7 (optional): retrieval comparison per query ---
    from qdrant_client import models

    statute_filter = models.Filter(must=[models.FieldCondition(key="source_type", match=models.MatchValue(value=SOURCE_FILTER))])
    retrieval_comparison = {}
    retrieval_errors = []
    for qid, _, _ in QUERIES:
        try:
            hits = client.query_points(
                COLLECTION, query=query_vecs[qid].tolist(), using="dense", query_filter=statute_filter,
                limit=N_STATUTE, with_payload=True,
            ).points
            target_rank = next((i for i, h in enumerate(hits, 1) if h.payload.get("chunk_id") == target_chunk_id), None)
            target_score = next((h.score for h in hits if h.payload.get("chunk_id") == target_chunk_id), None)
            candidates = [{"chunk_id": h.payload.get("chunk_id"), "score": h.score, "is_experimental": False} for h in hits]
            candidates.append({"chunk_id": f"{target_chunk_id}::ISOLATED_PROVISO_EXPERIMENTAL",
                                "score": sim[qid]["D1"], "is_experimental": True})
            candidates.sort(key=lambda c: c["score"], reverse=True)
            experimental_rank = next(i for i, c in enumerate(candidates, 1) if c["is_experimental"])
            retrieval_comparison[qid] = {
                "production_target_rank": target_rank,
                "production_target_score": target_score,
                "n_candidates_ranked": len(hits),
                "isolated_passage_experimental_comparative_rank": experimental_rank,
                "experimental_candidate_pool_size": len(candidates),
            }
            print(f"  [retrieval, {qid}] production target rank={target_rank} score={target_score}  "
                  f"isolated experimental comparative rank={experimental_rank}")
        except Exception as exc:  # noqa: BLE001
            retrieval_errors.append({"query_id": qid, "error": str(exc)})
            retrieval_comparison[qid] = {"error": str(exc)}

    # --- assemble report ---
    report = {
        "experiment": "phase1_query_chunk_interaction",
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script_version": 1,
        "target_chunk_id": target_chunk_id,
        "target_point_id": target_point_id,
        "sources": {"test1_artifact": str(TEST1_PATH), "test2_artifact": str(TEST2_PATH)},
        "model_name": MODEL_NAME,
        "dense_dim": DENSE_DIM,
        "similarity_metric": "cosine",
        "document_representations": {
            "D0_original": {"source": "stored production vector (client.retrieve)", "embedding_text": d0_embedding_text,
                             "embedding_text_chars": len(d0_embedding_text)},
            "D1_isolated": {"source": f"verbatim from {TEST2_PATH}:isolated_passage.isolated_passage_text_embedded",
                             "embedding_text": d1_text, "embedding_text_chars": len(d1_text)},
        },
        "queries": [{"query_id": qid, "change_from_baseline": change, "query": qtext} for qid, change, qtext in QUERIES],
        "similarity_matrix": sim,
        "validation": {
            "sim_q0_d0_recomputed": q0_d0,
            "test2_recorded_stored_similarity_to_q0": prior_stored_sim_q0,
            "difference": validation_diff,
            "within_tolerance_1e-4": validation_ok,
        },
        "effects": {
            "isolation_effect_per_query": isolation_effect,
            "vocabulary_effect_per_doc_Q1_minus_Q0": vocab_effect,
            "vocabulary_effect_with_authority_per_doc_Q3_minus_Q2": vocab_effect_with_authority,
            "authority_effect_per_doc_Q2_minus_Q0": authority_effect,
            "authority_effect_with_haqmehr_per_doc_Q3_minus_Q1": authority_effect_with_haqmehr,
            "interaction_contrast_per_doc": interaction,
            "isolation_effect_difference_Q3_minus_Q0": isolation_diff_q3_minus_q0,
        },
        "retrieval_comparison_optional": retrieval_comparison,
        "retrieval_comparison_errors": retrieval_errors,
        "deviations": [],
        "limitations": [
            "Single target chunk and single embedding model -- no statistical significance implied by any effect size.",
            "D1's embedding text includes the production context_header prefix, matching Test 2's design; "
            "no header-free variant was tested here either.",
        ],
    }
    if not validation_ok:
        report["deviations"].append(
            f"sim(Q0,D0) recomputed as {q0_d0:.6f} differs from Test 2's recorded {prior_stored_sim_q0:.6f} "
            f"by {validation_diff:.2e}, exceeding the 1e-4 tolerance -- investigate before trusting this matrix."
        )

    report["run_completed_successfully"] = not retrieval_errors

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    final_git_status = git_status_short()
    meta = {
        "script_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "output_json": str(OUT_JSON),
        "output_meta": str(OUT_META),
        "files_read": [str(TEST1_PATH), str(TEST2_PATH)],
        "collection": COLLECTION,
        "model_name": MODEL_NAME,
        "run_timestamp": report["run_timestamp"],
        "initial_git_status_short": initial_git_status,
        "final_git_status_short": final_git_status,
        "run_completed_successfully": report["run_completed_successfully"],
        "no_existing_file_overwritten": True,
        "note": "The only Qdrant operations performed were client.retrieve() (read, one call) and "
                "client.query_points() (read, four calls, one per query, for the optional Section 7 "
                "comparison) -- no upsert/delete/recreate call exists anywhere in this script.",
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n[phase1-interaction] report -> {OUT_JSON}")
    print(f"[phase1-interaction] metadata -> {OUT_META}")


if __name__ == "__main__":
    main()
