"""
Builds the list of NEW candidates needing manual judgment: union of top-10 per method,
minus cases already judged in the original 150-query benchmark's qrels. Fetches a headnote
(or best-available) text snippet per case for grounded judging. Depth-10 is used because
that is what every reported metric (Recall@10, MRR@10, nDCG@10, HitRate@10) actually needs;
ranks 11-40 remain in the pool for completeness statistics (Phase 3) but are explicitly
left UNJUDGED, not silently treated as nonrelevant -- disclosed in pool_completeness.csv
and comparison_report.md.
"""
import json, sqlite3
from pathlib import Path
from collections import defaultdict

NEW_DIR = Path(__file__).resolve().parent
REPO_ROOT = NEW_DIR.parents[2]
con = sqlite3.connect(REPO_ROOT / "var/rag/docstore.sqlite")
cur = con.cursor()

pool_summary = json.load(open(NEW_DIR / "pool_summary_raw.json"))
existing = json.load(open(NEW_DIR / "existing_qrels_by_query.json"))
queries = {r["id"]: r for r in (json.loads(l) for l in open(NEW_DIR.parents[0] / "final_family_law_150_query_benchmark_20261019_212630" / "queries_150.jsonl")) if r["id"].startswith("CL")}

def best_text_snippet(case_key, preferred_chunk_id=None):
    # prefer the given chunk_id (it's what retrieval actually surfaced); else first headnote/headnote_part
    if preferred_chunk_id:
        cur.execute("select canonical_text, unit_type from chunks where chunk_id=?", (preferred_chunk_id,))
        row = cur.fetchone()
        if row:
            return row[0][:500], row[1], preferred_chunk_id
    cur.execute("""select chunk_id, canonical_text, unit_type from chunks
                   where json_extract(json,'$.case_key')=? and unit_type in ('headnote','headnote_part')
                   order by chunk_id limit 1""", (case_key,))
    row = cur.fetchone()
    if row:
        return row[1][:500], row[2], row[0]
    return None, None, None

batches = {}
for qid, case_pool in pool_summary.items():
    judged_already = set(existing.get(qid, {}).keys())
    new_candidates = []
    for case_key, entry in case_pool.items():
        if case_key in judged_already:
            continue
        ranks = entry["ranks_by_method"]
        in_top10 = any(r <= 10 for r in ranks.values())
        if not in_top10:
            continue
        # use whichever method found it at best rank to pick the representative chunk
        best_method = min(ranks, key=lambda m: ranks[m])
        chunk_id = entry["chunk_ids_by_method"][best_method]
        snippet, unit_type, snippet_chunk_id = best_text_snippet(case_key, chunk_id)
        new_candidates.append({
            "case_key": case_key, "citation": entry["citation"], "doc_id": entry["doc_id"],
            "ranks_by_method": ranks, "snippet_chunk_id": snippet_chunk_id, "unit_type": unit_type,
            "snippet": snippet,
        })
    new_candidates.sort(key=lambda c: min(c["ranks_by_method"].values()))
    batches[qid] = {"query": queries[qid]["query"], "candidates": new_candidates}

json.dump(batches, open(NEW_DIR / "judging_batches.json", "w"), indent=2, ensure_ascii=False)
total = sum(len(b["candidates"]) for b in batches.values())
print("total new candidates to judge (top-10 pool):", total)
for qid in list(batches)[:3]:
    print(qid, len(batches[qid]["candidates"]))
