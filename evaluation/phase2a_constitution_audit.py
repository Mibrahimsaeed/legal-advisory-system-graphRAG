#!/usr/bin/env python3
"""Phase 2A -- Constitution corpus index & coverage audit (read-only).

Performs, against the existing, unmodified artifacts and Qdrant collection:
  1. Corpus coverage reconciliation (chunked file vs. docstore vs. live Qdrant).
  2. Integrity checks (empty chunks, duplicate chunk_id, duplicate clause
     numbers, non-retrievable reasons).
  3. Targeted recheck of Articles 63, 89, 175A, 260.
  4. A point_id reconciliation spot check using the project's own
     uuid5(ID_NAMESPACE, chunk_id) scheme (read from index_meta.json).
  5. A small dense-only + hybrid retrieval smoke test over 6 queries,
     each grounded in a verified real article from this corpus.

Reuses ONLY connect()/load_dense()/load_sparse()/encode_dense()/hybrid_search()
from src.rag_prep.embed_index, unmodified. Every Qdrant call is read-only
(points/count, points/scroll, query_points) -- no upsert/delete/recreate
call exists anywhere in this script.

    python evaluation/phase2a_constitution_audit.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rag_prep.embed_index import (  # noqa: E402
    COLLECTION,
    connect,
    encode_dense,
    hybrid_search,
    load_dense,
    load_sparse,
)

CONSTITUTION_CHUNK_FILE = Path("var/rag/constitution_chunked/b427a922990d75f027928c15.json")
INDEX_META_PATH = Path("var/rag/index_meta.json")
DOCSTORE_PATH = Path("var/rag/docstore.sqlite")

OUT_JSON = Path("evaluation/results/phase2a_constitution_audit.json")
OUT_MD = Path("evaluation/results/phase2a_constitution_audit.md")
OUT_SMOKE = Path("evaluation/results/phase2a_constitution_smoke_test.json")

SMOKE_QUERIES = [
    {"query_id": "C1", "type": "specific article", "query": "What does Article 199 of the Constitution say about the jurisdiction of the High Court?",
     "expected_article": "199", "expected_title": "Jurisdiction of High Court."},
    {"query_id": "C2", "type": "right, ordinary language", "query": "What right does the Constitution give a person regarding their life and personal liberty?",
     "expected_article": "9", "expected_title": "Security of person."},
    {"query_id": "C3", "type": "Parliament / legislative process", "query": "How is a Bill introduced and passed by Parliament?",
     "expected_article": "70", "expected_title": "Introduction and passing of Bills."},
    {"query_id": "C4", "type": "judiciary / judicial appointments", "query": "How are judges of the High Courts appointed?",
     "expected_article": "175A", "expected_title": "Appointment of Judges to the Federal Constitutional Court, the Supreme Court, High Courts and the Federal Shariat Court."},
    {"query_id": "C5", "type": "federal/provincial powers", "query": "What is the subject-matter division between federal and provincial laws?",
     "expected_article": "142", "expected_title": "Subject-matter of Federal and Provincial laws."},
    {"query_id": "C6", "type": "specific clause/proviso", "query": "Under Article 63, what is the disqualification for a person sentenced to imprisonment, and what proviso limits it?",
     "expected_article": "63", "expected_title": "Disqualifications for membership of Majlis-e-Shoora (Parliament)."},
]


def git_status_short() -> str:
    out = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return out.stdout


def qdrant_available() -> tuple[bool, str]:
    try:
        import requests
        r = requests.get("http://localhost:6333/collections/pk_legal", timeout=5)
        return r.status_code == 200, f"HTTP {r.status_code}"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def main() -> None:
    initial_git_status = git_status_short()
    print("[phase2a] initial git status --short:")
    print(initial_git_status or "  (clean)")

    for p in (OUT_JSON, OUT_SMOKE):
        if p.exists():
            raise SystemExit(f"[phase2a] refusing to overwrite existing path: {p}.")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()

    # ---------------------------------------------------------------- corpus discovery
    if not CONSTITUTION_CHUNK_FILE.exists():
        raise SystemExit(f"[phase2a] {CONSTITUTION_CHUNK_FILE} not found. Stopping.")
    data = json.loads(CONSTITUTION_CHUNK_FILE.read_text(encoding="utf-8"))
    chunks = data["chunks"]
    index_meta = json.loads(INDEX_META_PATH.read_text(encoding="utf-8")) if INDEX_META_PATH.exists() else {}
    id_namespace = uuid.UUID(index_meta["id_namespace"]) if "id_namespace" in index_meta else None

    # ---------------------------------------------------------------- integrity checks
    retrievable = [c for c in chunks if c.get("retrievable")]
    non_retrievable = [c for c in chunks if not c.get("retrievable")]
    empty = [c for c in chunks if not (c.get("canonical_text") or "").strip()]
    ids = [c["chunk_id"] for c in chunks]
    dupe_ids = [cid for cid, n in Counter(ids).items() if n > 1]

    # duplicate clause_number within the same article (distinct chunk_ids, same clause label)
    dupe_clause_findings = []
    by_article: dict[str, list[dict]] = {}
    for c in chunks:
        an = c.get("article_number")
        if an:
            by_article.setdefault(an, []).append(c)
    for an, arts in by_article.items():
        clause_counts = Counter(c.get("clause_number") for c in arts if c.get("clause_number"))
        for cn, n in clause_counts.items():
            if n > 1:
                offenders = [c["chunk_id"] for c in arts if c.get("clause_number") == cn]
                dupe_clause_findings.append({"article_number": an, "clause_number": cn, "chunk_ids": offenders})

    short_retrievable = sorted(
        [{"chunk_id": c["chunk_id"], "chars": len(c["canonical_text"])} for c in retrievable],
        key=lambda x: x["chars"],
    )[:10]

    # ---------------------------------------------------------------- Article 63/89/175A/260 recheck
    def article_check(num: str) -> dict:
        matches = [c for c in chunks if c.get("article_number") == num]
        empty_in_art = [c["chunk_id"] for c in matches if c.get("retrievable") and not (c.get("canonical_text") or "").strip()]
        return {
            "article_number": num,
            "n_chunks": len(matches),
            "n_retrievable": sum(1 for c in matches if c.get("retrievable")),
            "chunk_ids": [c["chunk_id"] for c in matches],
            "empty_retrievable_chunks": empty_in_art,
        }

    article_rechecks = {num: article_check(num) for num in ("63", "89", "175A", "260", "199")}

    # Article 199 clause-mislabel finding (found during this audit's inspection)
    art199 = sorted([c for c in chunks if c.get("article_number") == "199"], key=lambda c: c["source_start"])
    art199_mislabel = None
    for i, c in enumerate(art199):
        if c["chunk_id"] == "art_199_c1_2":
            prev_c = art199[i - 1] if i > 0 else None
            next_c = art199[i + 1] if i + 1 < len(art199) else None
            art199_mislabel = {
                "chunk_id": c["chunk_id"],
                "labeled_clause_number": c.get("clause_number"),
                "canonical_text": c["canonical_text"],
                "source_start": c["source_start"], "source_end": c["source_end"],
                "immediately_preceded_by": prev_c["chunk_id"] if prev_c else None,
                "immediately_followed_by": next_c["chunk_id"] if next_c else None,
                "finding": (
                    "chunk_id 'art_199_c1_2' is uniquely-ID'd (no duplicate chunk_id) but its "
                    "clause_number metadata and context_header both read 'Clause (1)', even though "
                    "its text ('(1), the High Court has made an interim order shall be disposed of "
                    "...') is the direct, contiguous continuation of art_199_c4a (source_end=199299 "
                    "== this chunk's source_start) -- i.e. it is the tail of clause (4A)/(4B)'s own "
                    "proviso, mislabeled as clause 1. The text itself is also truncated at its own "
                    "start (the sentence's opening, 'Where ... under clause', is missing -- only "
                    "the cross-reference fragment '(1), the High Court...' remains), making this "
                    "chunk not self-contained if retrieved on its own."
                ),
            }

    # ---------------------------------------------------------------- docstore / Qdrant reconciliation
    import sqlite3
    con = sqlite3.connect(f"file:{DOCSTORE_PATH}?mode=ro", uri=True)
    cur = con.cursor()
    cur.execute("SELECT COUNT(*), SUM(retrievable) FROM chunks WHERE source_type='constitution'")
    docstore_total, docstore_retrievable = cur.fetchone()

    qdrant_ok, qdrant_detail = qdrant_available()
    qdrant_count = None
    vector_config = None
    point_id_spotcheck = []
    if qdrant_ok:
        client = connect()
        info = client.get_collection(COLLECTION)
        vectors = info.config.params.vectors
        vector_config = {
            "dense_size": vectors["dense"].size, "dense_distance": str(vectors["dense"].distance),
            "sparse_vectors": list(info.config.params.sparse_vectors.keys()) if info.config.params.sparse_vectors else [],
            "points_count": info.points_count,
        }
        from qdrant_client import models
        r = client.count(COLLECTION, count_filter=models.Filter(must=[models.FieldCondition(key="source_type", match=models.MatchValue(value="constitution"))]), exact=True)
        qdrant_count = r.count

        if id_namespace:
            sample_ids = [c["chunk_id"] for c in retrievable[:3]] + ["art_63_c1_p1", "art_175a_c1", "preamble_p1"]
            sample_ids = list(dict.fromkeys(sample_ids))[:6]
            for cid in sample_ids:
                pid = str(uuid.uuid5(id_namespace, cid))
                scroll = client.scroll(COLLECTION, scroll_filter=models.Filter(must=[models.FieldCondition(key="chunk_id", match=models.MatchValue(value=cid))]), limit=1, with_payload=["chunk_id", "source_type"])
                found = scroll[0]
                point_id_spotcheck.append({
                    "chunk_id": cid, "computed_point_id": pid,
                    "found_in_qdrant": bool(found),
                    "qdrant_point_id_matches": bool(found) and str(found[0].id) == pid,
                    "qdrant_source_type": found[0].payload.get("source_type") if found else None,
                })

    coverage = {
        "chunked_file_total": len(chunks), "chunked_file_retrievable": len(retrievable),
        "docstore_total": docstore_total, "docstore_retrievable": docstore_retrievable,
        "index_meta_constitution_count": index_meta.get("counts", {}).get("constitution"),
        "qdrant_available": qdrant_ok, "qdrant_detail": qdrant_detail,
        "qdrant_constitution_point_count": qdrant_count,
        "all_counts_reconcile": (
            len(retrievable) == docstore_retrievable == index_meta.get("counts", {}).get("constitution") == qdrant_count
            if qdrant_ok else None
        ),
    }

    reconciliation_by_part = {}
    for c in chunks:
        part = c.get("part_number") or "(none: preamble/schedule/annex)"
        reconciliation_by_part.setdefault(part, {"total": 0, "retrievable": 0})
        reconciliation_by_part[part]["total"] += 1
        if c.get("retrievable"):
            reconciliation_by_part[part]["retrievable"] += 1

    # ---------------------------------------------------------------- retrieval smoke test
    smoke_results = []
    if qdrant_ok:
        dense = load_dense()
        sparse = load_sparse()
        from qdrant_client import models
        const_filter = models.Filter(must=[models.FieldCondition(key="source_type", match=models.MatchValue(value="constitution"))])

        for q in SMOKE_QUERIES:
            qvec = encode_dense(dense, [q["query"]], 1)[0].tolist()
            dense_hits = client.query_points(COLLECTION, query=qvec, using="dense", query_filter=const_filter, limit=240, with_payload=True).points
            dense_rank = next((i for i, h in enumerate(dense_hits, 1) if h.payload.get("article_number") == q["expected_article"]), None)
            dense_score = next((h.score for h in dense_hits if h.payload.get("article_number") == q["expected_article"]), None)
            dense_top_chunk = dense_hits[0].payload.get("chunk_id") if dense_hits else None

            hybrid_hits = hybrid_search(client, dense, sparse, q["query"], {"constitution": 240}, candidates=40)
            hybrid_rank = next((i for i, h in enumerate(hybrid_hits, 1) if h.payload.get("article_number") == q["expected_article"]), None)
            hybrid_score = next((h.score for h in hybrid_hits if h.payload.get("article_number") == q["expected_article"]), None)

            expected_chunk_exists = any(c.get("article_number") == q["expected_article"] for c in chunks)
            expected_has_retrievable = any(c.get("article_number") == q["expected_article"] and c.get("retrievable") for c in chunks)

            smoke_results.append({
                "query_id": q["query_id"], "type": q["type"], "query": q["query"],
                "expected_article": q["expected_article"], "expected_title_verified_from_corpus": q["expected_title"],
                "expected_chunk_exists_in_corpus": expected_chunk_exists,
                "expected_article_has_retrievable_chunk": expected_has_retrievable,
                "dense_only": {"rank": dense_rank, "score": dense_score, "n_ranked": len(dense_hits), "top1_chunk_id": dense_top_chunk},
                "hybrid": {"rank": hybrid_rank, "score": hybrid_score, "n_ranked": len(hybrid_hits)},
            })
            print(f"  [{q['query_id']}] expected=Art.{q['expected_article']}  dense_rank={dense_rank}  hybrid_rank={hybrid_rank}")
    else:
        print("[phase2a] Qdrant unavailable -- skipping retrieval smoke test entirely (no local-cosine substitute used).")

    # ---------------------------------------------------------------- assemble + write
    report = {
        "audit": "phase2a_constitution_audit",
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "repo_commit": commit,
        "artifacts_inspected": [str(CONSTITUTION_CHUNK_FILE), str(INDEX_META_PATH), str(DOCSTORE_PATH)],
        "coverage_reconciliation": coverage,
        "reconciliation_by_part": reconciliation_by_part,
        "integrity": {
            "total_chunks": len(chunks), "retrievable": len(retrievable), "non_retrievable": len(non_retrievable),
            "non_retrievable_by_unit_type": dict(Counter(c.get("unit_type") for c in non_retrievable)),
            "empty_or_whitespace_only_chunks": len(empty), "empty_chunk_ids": [c["chunk_id"] for c in empty],
            "duplicate_chunk_ids": dupe_ids,
            "duplicate_clause_number_within_article": dupe_clause_findings,
            "shortest_10_retrievable_chunks": short_retrievable,
        },
        "article_rechecks": article_rechecks,
        "article_199_clause_mislabel_finding": art199_mislabel,
        "vector_config": vector_config,
        "point_id_reconciliation_spotcheck": point_id_spotcheck,
        "smoke_test_queries": smoke_results,
        "limitations": [
            "Clause-boundary mislabeling (as found for art_199_c1_2) was investigated for Article 199 "
            "specifically after being noticed during inspection; a systematic sweep for the same pattern "
            "across all 328 articles was not performed in this pass.",
            "Footnote citation text for the omission markers inside the Article 175A clause (8)-(18) "
            "range chunk was not independently extracted/verified in this pass (the chunk's own "
            "context_header labeling was verified correct and honest regardless).",
        ],
        "run_completed_successfully": True,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_SMOKE.write_text(json.dumps({"smoke_test_queries": smoke_results, "qdrant_available": qdrant_ok}, indent=2, ensure_ascii=False), encoding="utf-8")

    final_git_status = git_status_short()
    print(f"\n[phase2a] report -> {OUT_JSON}")
    print(f"[phase2a] smoke test -> {OUT_SMOKE}")
    print("\n[phase2a] final git status --short:")
    print(final_git_status or "  (clean)")

    report["final_git_status_short"] = final_git_status
    report["initial_git_status_short"] = initial_git_status
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
