"""
Phase 2-3: assemble judgment_ledger.jsonl, qrels_expanded.jsonl, and pool_completeness.csv
from (a) the original benchmark's qrels (previously_judged) and (b) this task's new
manual judging pass (manual_judgments_all.json), against the full top-40 pool
(pool_summary_raw.json) for completeness statistics.
"""
import json, sqlite3, csv
from pathlib import Path
from collections import defaultdict

NEW_DIR = Path(__file__).resolve().parent
OLD_DIR = NEW_DIR.parents[0] / "final_family_law_150_query_benchmark_20261019_212630"
REPO_ROOT = NEW_DIR.parents[2]
con = sqlite3.connect(REPO_ROOT / "var/rag/docstore.sqlite")
cur = con.cursor()

queries = {r["id"]: r["query"] for r in (json.loads(l) for l in open(OLD_DIR / "queries_150.jsonl")) if r["id"].startswith("CL")}
pool_summary = json.load(open(NEW_DIR / "pool_summary_raw.json"))          # qid -> case_key -> {ranks_by_method, chunk_ids_by_method, doc_id, citation}
existing_qrels = json.load(open(NEW_DIR / "existing_qrels_by_query.json")) # qid -> case_key -> old qrel row (chunk_id, grade, case_key, citation, rationale, ...)
new_judgments = json.load(open(NEW_DIR / "manual_judgments_all.json"))     # qid -> case_key -> [grade, note]

def chunk_text_meta(chunk_id):
    cur.execute("select canonical_text, json_extract(json,'$.unit_type'), json_extract(json,'$.paragraph_start'), json_extract(json,'$.paragraph_end') from chunks where chunk_id=?", (chunk_id,))
    row = cur.fetchone()
    return row if row else (None, None, None, None)

ledger_rows = []
qrels_expanded_rows = []
completeness_rows = []

for qid in sorted(pool_summary, key=lambda x: int(x[2:])):
    case_pool = dict(pool_summary[qid])
    old = existing_qrels.get(qid, {})
    new = new_judgments.get(qid, {})

    # Preserve previously-judged relevant cases even when they were retrieved by NONE
    # of dense/sparse/hybrid within top-40 -- this must never be silently dropped (it is
    # itself an important finding: a known-relevant case entirely missed at this depth).
    n_orphan_known_relevant = 0
    for case_key, old_row in old.items():
        if case_key not in case_pool:
            case_pool[case_key] = {
                "case_key": case_key, "doc_id": None, "citation": old_row.get("citation"),
                "ranks_by_method": {}, "chunk_ids_by_method": {},
                "scores_by_method": {}, "_orphan_chunk_id": old_row["chunk_id"],
            }
            n_orphan_known_relevant += 1

    n_pooled = len(case_pool)
    n_judged = 0
    n_grade2 = n_grade1 = n_grade0 = n_unjudged = 0
    found_by_method = {"dense": set(), "sparse": set(), "hybrid": set()}

    for case_key, entry in case_pool.items():
        ranks = entry["ranks_by_method"]
        for m in ranks:
            found_by_method[m].add(case_key)

        if case_key in old:
            grade = old[case_key]["grade"]
            status = "previously_judged"
            rationale = old[case_key].get("rationale", "")
            provenance = old[case_key].get("provenance", "researcher_judged")
            evidence_chunk = old[case_key]["chunk_id"]
        elif case_key in new:
            grade, rationale = new[case_key]
            status = "newly_judged_this_task"
            provenance = "researcher_judged"
            evidence_chunk = entry["chunk_ids_by_method"].get(min(ranks, key=lambda m: ranks[m]))
        else:
            grade = None
            status = "unjudged_beyond_top10_pool"
            rationale = "Not pooled within top-10 per method (or already excluded) in this task's judging depth; genuinely unjudged, NOT treated as irrelevant."
            provenance = None
            evidence_chunk = entry["chunk_ids_by_method"].get(min(ranks, key=lambda m: ranks[m])) if ranks else None

        text, unit_type, pstart, pend = chunk_text_meta(evidence_chunk) if evidence_chunk else (None, None, None, None)
        excerpt = (text or "")[:300].replace("\n", " ")

        ledger_rows.append({
            "query_id": qid, "query_text": queries[qid], "case_id": case_key,
            "citation": entry.get("citation"),
            "chunk_ids": entry["chunk_ids_by_method"],
            "retrieval_methods_found": sorted(ranks.keys()),
            "ranks_by_method": ranks,
            "relevance_grade": grade,
            "judgment_status": status,
            "evidence_excerpt": excerpt if grade is not None and grade >= 1 else (excerpt if status != "unjudged_beyond_top10_pool" else None),
            "evidence_location": {"chunk_id": evidence_chunk, "unit_type": unit_type, "paragraph_start": pstart, "paragraph_end": pend},
            "judgment_notes": rationale,
            "provenance": provenance,
        })

        if grade is None:
            n_unjudged += 1
        else:
            n_judged += 1
            if grade == 2: n_grade2 += 1
            elif grade == 1: n_grade1 += 1
            else: n_grade0 += 1
            if grade >= 1:
                qrels_expanded_rows.append({
                    "query_id": qid, "corpus": "case", "case_key": case_key, "citation": entry.get("citation"),
                    "chunk_id": evidence_chunk, "grade": grade, "rationale": rationale,
                    "judgment_status": status, "provenance": provenance,
                    "pooling_status": "pooled_top10_per_method_plus_prior_benchmark" ,
                })

    relevant_keys = {ck for ck, e in case_pool.items()
                      if (ck in old and old[ck]["grade"] >= 1) or (ck in new and new[ck][0] >= 1)}
    overlap = {
        "dense_only": len(found_by_method["dense"] - found_by_method["sparse"] - found_by_method["hybrid"]),
        "sparse_only": len(found_by_method["sparse"] - found_by_method["dense"] - found_by_method["hybrid"]),
        "hybrid_only": len(found_by_method["hybrid"] - found_by_method["dense"] - found_by_method["sparse"]),
        "all_three": len(found_by_method["dense"] & found_by_method["sparse"] & found_by_method["hybrid"]),
    }
    relevant_found_by_method = {m: len(found_by_method[m] & relevant_keys) for m in found_by_method}

    completeness_rows.append({
        "query_id": qid,
        "n_unique_pooled_cases_top40": n_pooled - n_orphan_known_relevant,
        "n_known_relevant_missed_by_all_three_methods_top40": n_orphan_known_relevant,
        "n_judged": n_judged,
        "n_highly_relevant_grade2": n_grade2,
        "n_partially_relevant_grade1": n_grade1,
        "n_nonrelevant_grade0": n_grade0,
        "n_unjudged": n_unjudged,
        "pct_judged": round(100*n_judged/n_pooled, 1) if n_pooled else None,
        "n_relevant_found_by_dense": relevant_found_by_method["dense"],
        "n_relevant_found_by_sparse": relevant_found_by_method["sparse"],
        "n_relevant_found_by_hybrid": relevant_found_by_method["hybrid"],
        "pool_dense_only": overlap["dense_only"], "pool_sparse_only": overlap["sparse_only"],
        "pool_hybrid_only": overlap["hybrid_only"], "pool_all_three_methods": overlap["all_three"],
    })

with open(NEW_DIR / "judgment_ledger.jsonl", "w") as f:
    for r in ledger_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open(NEW_DIR / "qrels_expanded.jsonl", "w") as f:
    for r in qrels_expanded_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open(NEW_DIR / "pool_completeness.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(completeness_rows[0].keys()))
    w.writeheader()
    w.writerows(completeness_rows)

print("ledger rows:", len(ledger_rows))
print("qrels_expanded rows (grade>=1):", len(qrels_expanded_rows))
print("completeness rows:", len(completeness_rows))
totals = defaultdict(int)
for r in completeness_rows:
    for k in ("n_unique_pooled_cases_top40","n_judged","n_highly_relevant_grade2","n_partially_relevant_grade1","n_nonrelevant_grade0","n_unjudged"):
        totals[k]+=r[k]
print(dict(totals))
