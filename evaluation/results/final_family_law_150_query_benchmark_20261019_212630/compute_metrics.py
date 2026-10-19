"""
Metrics computation for the 150-query family-law benchmark.

Grading rules (documented; applied consistently across methods within a corpus):

  Constitution / Statute: relevance is STRUCTURALLY exhaustive -- any retrieved chunk's
  article_number (Constitution) or section identity (Statute) can be compared directly
  against the query's assigned target provision(s) recorded in queries_150.jsonl's
  evidence. A chunk matching a target provision = grade 2 (same as the qrels source
  text). A chunk belonging to ANY OTHER provision is correctly graded 0 -- this is not
  an approximation, because constitutional/statutory relevance for these queries is
  defined by provision identity, not open-ended topical similarity. P@5 and nDCG@10 are
  therefore FULLY computable (no missing labels) for these two corpora.

  Case law: relevance is NOT structurally determinable from metadata alone (two
  different judgments can independently be on-point for the same legal issue). Only
  the pooled qrels.jsonl chunks have been read and judged; every other retrieved chunk
  is UNJUDGED, not irrelevant. P@5/nDCG@10 are therefore computed under the strict
  missing-label-aware rule (calculate_metrics.py's own semantics) and are expected to
  be reported "unavailable" for most queries -- this is disclosed, not hidden, and is
  exactly the scenario the benchmark's Objective section anticipates ("retrieval
  results that cannot be evaluated reliably because relevance judgments are
  incomplete"). Recall@5/10, MRR@10, and HitRate@5/10 do NOT require judging every
  candidate -- only membership against the known-relevant qrels set -- so they ARE
  computed reliably for all three corpora.
"""
import json, sqlite3, os, importlib.util
from pathlib import Path
from collections import defaultdict

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parents[2]

spec = importlib.util.spec_from_file_location("calc_metrics", REPO_ROOT / "evaluation" / "calculate_metrics.py")
calc_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calc_metrics)
precision_at_k, ndcg_at_k, recall_at_k = calc_metrics.precision_at_k, calc_metrics.ndcg_at_k, calc_metrics.recall_at_k

DB = REPO_ROOT / "var/rag/docstore.sqlite"
con = sqlite3.connect(DB)
cur = con.cursor()

queries = {r["id"]: r for r in (json.loads(l) for l in open(AUDIT_DIR / "queries_150.jsonl"))}
qrels_rows = [json.loads(l) for l in open(AUDIT_DIR / "qrels.jsonl")]
qrels_by_query = defaultdict(dict)  # qid -> {chunk_id: grade}
for q in qrels_rows:
    qrels_by_query[q["query_id"]][q["chunk_id"]] = max(q["grade"], qrels_by_query[q["query_id"]].get(q["chunk_id"], 0))

# target provisions for structural grading
const_targets = {qid: [e["article"] for e in r["evidence"]] for qid, r in queries.items() if r["corpus"]=="constitution" and r["evidence"]}
stat_targets = {qid: [e["chunk_id"].split(":",1)[1] for e in r["evidence"]] for qid, r in queries.items() if r["corpus"]=="statute" and r["evidence"]}

def section_root(suffix):
    # "s7_ss1A-2-3-4-5-6" -> "s7" ; "schedule:partI" stays as-is after split on ':' already done upstream
    for sep in ("_ss","_"):
        if sep in suffix:
            return suffix.split(sep)[0]
    return suffix

art_cache = {}
def chunk_article_number(chunk_id):
    if chunk_id not in art_cache:
        cur.execute("select json_extract(json,'$.article_number') from chunks where chunk_id=?", (chunk_id,))
        row = cur.fetchone()
        art_cache[chunk_id] = row[0] if row else None
    return art_cache[chunk_id]

def grade_constitution(qid, chunk_id):
    targets = const_targets.get(qid, [])
    art = chunk_article_number(chunk_id)
    return 2 if art in targets else 0

def grade_statute(qid, chunk_id):
    targets = stat_targets.get(qid, [])
    target_roots = {section_root(t) for t in targets}
    suffix = chunk_id.split(":",1)[1] if ":" in chunk_id else chunk_id
    return 2 if section_root(suffix) in target_roots else 0

def grade_case(qid, chunk_id):
    g = qrels_by_query.get(qid, {}).get(chunk_id)
    return g  # None if unjudged -- strict

def mrr_at_10(ranked_chunk_ids, relevant_chunk_ids):
    if relevant_chunk_ids is None:
        return None, "no relevance reference set"
    for i, cid in enumerate(ranked_chunk_ids[:10], 1):
        if cid in relevant_chunk_ids:
            return 1.0/i, None
    return 0.0, None

def hit_rate_at_k(ranked_chunk_ids, relevant_chunk_ids, k):
    if relevant_chunk_ids is None:
        return None, "no relevance reference set"
    top = set(ranked_chunk_ids[:k])
    return (1.0 if top & relevant_chunk_ids else 0.0), None

chunk_to_case_key = {}
def get_case_key(chunk_id):
    if chunk_id not in chunk_to_case_key:
        cur.execute("select json_extract(json,'$.case_key') from chunks where chunk_id=?", (chunk_id,))
        row = cur.fetchone()
        chunk_to_case_key[chunk_id] = row[0] if row else None
    return chunk_to_case_key[chunk_id]

def load_rankings(path, corpus_filter=None):
    """returns {query_id: [chunk_id, ...]} ordered by rank, restricted to corpus_filter source_type if given"""
    out = defaultdict(list)
    if not os.path.exists(path):
        return out
    for line in open(path):
        r = json.loads(line)
        if corpus_filter and r.get("source_type") != corpus_filter:
            continue
        out[r["query_id"]].append((r["rank"], r.get("chunk_id")))
    for qid in out:
        out[qid] = [cid for _, cid in sorted(out[qid])]
    return out

METHODS = {}
for corpus in ("constitution","statute","case"):
    METHODS[("hybrid",corpus)] = load_rankings(AUDIT_DIR / f"retrieval_hybrid_{corpus}.jsonl", corpus_filter=corpus)
    METHODS[("dense",corpus)]  = load_rankings(AUDIT_DIR / f"retrieval_dense_{corpus}.jsonl")
    METHODS[("sparse",corpus)]= load_rankings(AUDIT_DIR / f"retrieval_sparse_{corpus}.jsonl")

grader = {"constitution": grade_constitution, "statute": grade_statute, "case": grade_case}

per_query_rows = []
for corpus in ("constitution","statute","case"):
    eligible_qids = [qid for qid, r in queries.items() if r["corpus"]==corpus and r["evidence"]]
    for method in ("dense","sparse","hybrid"):
        rankings = METHODS[(method,corpus)]
        for qid in eligible_qids:
            ranked = rankings.get(qid, [])
            relevances = [grader[corpus](qid, cid) for cid in ranked]
            p5, p5note = precision_at_k(relevances, 5)
            ndcg10, ndcgnote = ndcg_at_k(relevances, 10)
            relevant_set = set(qrels_by_query.get(qid, {}).keys()) if qrels_by_query.get(qid) else None
            r5, r5note = recall_at_k(ranked, relevant_set, 5)
            r10, r10note = recall_at_k(ranked, relevant_set, 10)
            mrr, mrrnote = mrr_at_10(ranked, relevant_set)
            hr5, hr5note = hit_rate_at_k(ranked, relevant_set, 5)
            hr10, hr10note = hit_rate_at_k(ranked, relevant_set, 10)
            row = {
                "query_id": qid, "corpus": corpus, "method": method, "n_retrieved": len(ranked),
                "p_at_5": p5, "p_at_5_note": p5note,
                "ndcg_at_10": ndcg10, "ndcg_at_10_note": ndcgnote,
                "recall_at_5": r5, "recall_at_5_note": r5note,
                "recall_at_10": r10, "recall_at_10_note": r10note,
                "mrr_at_10": mrr, "mrr_at_10_note": mrrnote,
                "hit_rate_at_5": hr5, "hit_rate_at_5_note": hr5note,
                "hit_rate_at_10": hr10, "hit_rate_at_10_note": hr10note,
            }
            if corpus == "case":
                # document-level (case_key) view: a case is "found" if ANY of its chunks
                # (not necessarily the exact headnote chunk judged) appears in the ranking.
                ranked_case_keys = [get_case_key(cid) for cid in ranked]
                relevant_case_keys = {get_case_key(cid) for cid in relevant_set} if relevant_set else None
                dr5, dr5note = recall_at_k(ranked_case_keys, relevant_case_keys, 5)
                dr10, dr10note = recall_at_k(ranked_case_keys, relevant_case_keys, 10)
                dmrr, dmrrnote = mrr_at_10(ranked_case_keys, relevant_case_keys)
                dhr5, dhr5note = hit_rate_at_k(ranked_case_keys, relevant_case_keys, 5)
                dhr10, dhr10note = hit_rate_at_k(ranked_case_keys, relevant_case_keys, 10)
                row.update({
                    "doc_recall_at_5": dr5, "doc_recall_at_10": dr10,
                    "doc_mrr_at_10": dmrr, "doc_hit_rate_at_5": dhr5, "doc_hit_rate_at_10": dhr10,
                })
            per_query_rows.append(row)

json.dump(per_query_rows, open(AUDIT_DIR / "metrics_per_query_raw.json","w"), indent=2)
print("per-query rows:", len(per_query_rows))

# quick sanity print
for corpus in ("constitution","statute","case"):
    for method in ("dense","sparse","hybrid"):
        rows = [r for r in per_query_rows if r["corpus"]==corpus and r["method"]==method]
        ndcg_avail = [r["ndcg_at_10"] for r in rows if r["ndcg_at_10"] is not None]
        mrr_avail = [r["mrr_at_10"] for r in rows if r["mrr_at_10"] is not None]
        line = (f"{corpus} {method} n={len(rows)} nDCG@10 avail={len(ndcg_avail)} avg={round(sum(ndcg_avail)/len(ndcg_avail),3) if ndcg_avail else None} "
                f"MRR@10(chunk) avail={len(mrr_avail)} avg={round(sum(mrr_avail)/len(mrr_avail),3) if mrr_avail else None}")
        if corpus == "case":
            dmrr_avail = [r["doc_mrr_at_10"] for r in rows if r["doc_mrr_at_10"] is not None]
            dr10_avail = [r["doc_recall_at_10"] for r in rows if r["doc_recall_at_10"] is not None]
            line += f" | doc-MRR@10 avg={round(sum(dmrr_avail)/len(dmrr_avail),3) if dmrr_avail else None} doc-Recall@10 avg={round(sum(dr10_avail)/len(dr10_avail),3) if dr10_avail else None}"
        print(line)
