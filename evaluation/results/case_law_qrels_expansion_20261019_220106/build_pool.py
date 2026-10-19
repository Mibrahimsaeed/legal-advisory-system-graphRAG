"""
Phase 2 (structural part): build pool_rankings.jsonl and the candidate pool for judging.

Reuses the EXISTING, already-saved, unmodified top-40 per-method retrieval files from the
150-query benchmark (retrieval_dense_case.jsonl, retrieval_sparse_case.jsonl,
retrieval_hybrid_case.jsonl) rather than re-querying Qdrant. These files already satisfy
Phase 2's requirements: same production implementation, same candidate depth (40), each
method's ranking saved separately and unmerged. Reusing them (vs. re-running retrieval)
guarantees the expanded qrels are evaluated against the EXACT rankings already reported in
the baseline metrics -- a clean, apples-to-apples before/after comparison, and avoids any
embedding/BM25 run-to-run nondeterminism. This choice is documented in README.md.
"""
import json, sqlite3, os
from pathlib import Path
from collections import defaultdict

NEW_DIR = Path(__file__).resolve().parent
OLD_DIR = NEW_DIR.parents[0] / "final_family_law_150_query_benchmark_20261019_212630"
REPO_ROOT = NEW_DIR.parents[2]

con = sqlite3.connect(REPO_ROOT / "var/rag/docstore.sqlite")
cur = con.cursor()

_case_key_cache, _doc_id_cache, _citation_cache, _unit_type_cache = {}, {}, {}, {}
def chunk_meta(chunk_id):
    if chunk_id not in _case_key_cache:
        cur.execute("""select doc_id, json_extract(json,'$.case_key'), json_extract(json,'$.citation'),
                       json_extract(json,'$.unit_type') from chunks where chunk_id=?""", (chunk_id,))
        row = cur.fetchone()
        if row:
            _doc_id_cache[chunk_id], _case_key_cache[chunk_id], _citation_cache[chunk_id], _unit_type_cache[chunk_id] = row
        else:
            _doc_id_cache[chunk_id] = _case_key_cache[chunk_id] = _citation_cache[chunk_id] = _unit_type_cache[chunk_id] = None
    return _doc_id_cache[chunk_id], _case_key_cache[chunk_id], _citation_cache[chunk_id], _unit_type_cache[chunk_id]

def load_method(method):
    out = defaultdict(list)
    for line in open(OLD_DIR / f"retrieval_{method}_case.jsonl"):
        r = json.loads(line)
        out[r["query_id"]].append(r)
    return out

methods = {"dense": load_method("dense"), "sparse": load_method("sparse"), "hybrid": load_method("hybrid")}

# existing qrels (already judged, from the 150-query benchmark)
old_qrels = [json.loads(l) for l in open(OLD_DIR / "qrels.jsonl") if json.loads(l)["corpus"] == "case"]
existing_by_query = defaultdict(dict)  # qid -> {case_key: qrel_row}
for q in old_qrels:
    existing_by_query[q["query_id"]][q["case_key"]] = q

queries = {r["id"]: r for r in (json.loads(l) for l in open(OLD_DIR / "queries_150.jsonl")) if r["id"].startswith("CL")}
assert len(queries) == 50, f"expected 50 CL queries, found {len(queries)}"

pool_out = open(NEW_DIR / "pool_rankings.jsonl", "w")
pool_summary = {}  # qid -> {case_key: {"ranks_by_method": {...}, "chunk_ids_by_method": {...}, "doc_id":..., "citation":...}}

for qid, qrow in queries.items():
    qtext = qrow["query"]
    case_pool = {}
    for method in ("dense", "sparse", "hybrid"):
        for r in methods[method][qid]:
            chunk_id = r["chunk_id"]
            doc_id, case_key, citation, unit_type = chunk_meta(chunk_id)
            if case_key is None:
                continue
            entry = case_pool.setdefault(case_key, {
                "case_key": case_key, "doc_id": doc_id, "citation": citation,
                "ranks_by_method": {}, "chunk_ids_by_method": {}, "scores_by_method": {},
            })
            # keep the BEST (lowest) rank per method if the same case appears via multiple chunks
            if method not in entry["ranks_by_method"] or r["rank"] < entry["ranks_by_method"][method]:
                entry["ranks_by_method"][method] = r["rank"]
                entry["chunk_ids_by_method"][method] = chunk_id
                entry["scores_by_method"][method] = r.get("score", r.get("rrf_score"))
            # record individual (unmerged) retrieval rows for pool_rankings.jsonl
            pool_out.write(json.dumps({
                "query_id": qid, "query": qtext, "method": method, "rank": r["rank"],
                "chunk_id": chunk_id, "case_key": case_key, "citation": citation, "doc_id": doc_id,
                "score": r.get("score", r.get("rrf_score")),
            }, ensure_ascii=False) + "\n")
    pool_summary[qid] = case_pool

pool_out.close()

json.dump(pool_summary, open(NEW_DIR / "pool_summary_raw.json", "w"), indent=2, ensure_ascii=False)

# stats
total_unique = sum(len(v) for v in pool_summary.values())
total_already_judged = sum(len(existing_by_query[qid]) for qid in pool_summary)
print("total pooled rows (all methods, unmerged):", sum(1 for _ in open(NEW_DIR/'pool_rankings.jsonl')))
print("total unique pooled cases across 50 queries (top-40 pool):", total_unique)
print("total already-judged (from original benchmark):", total_already_judged)
json.dump({qid: dict(existing_by_query[qid]) for qid in existing_by_query}, open(NEW_DIR/"existing_qrels_by_query.json","w"), indent=2, ensure_ascii=False)
