"""
Build qrels.jsonl from the evidence already recorded in queries_150.jsonl.

Grading policy (documented, not retrieval-score-derived):
  - Constitution / statute: every identified evidence chunk (the article/section text
    actually answering the query) = grade 2. These corpora have a single canonical
    textual source per topic, so there is no grade-1 tier here.
  - Case law: the FIRST case_key recorded per query (the strongest/most on-point hit
    found during keyword search) = grade 2; remaining pooled case_keys = grade 1
    (supporting/partially relevant authority on the same issue).

Provenance: researcher-judged / machine-assisted keyword search over the indexed
corpus text (docstore.sqlite), NOT lawyer-validated, NOT derived from retrieval ranks.
Pooling status: case-law qrels are POOLED BUT INCOMPLETE (keyword search is not
exhaustive over 25,891 case chunks); Constitution/statute qrels are EXHAUSTIVE for
single-provision queries (the full text of the named article/section was located and
checked) but still a representative, not exhaustive, single-source judgment for the
underlying question.
"""
import json, os, sqlite3

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DB = "var/rag/docstore.sqlite"
con = sqlite3.connect(DB)
cur = con.cursor()

records = [json.loads(l) for l in open(os.path.join(OUT_DIR, "queries_150.jsonl"))]

qrels = []
for r in records:
    qid, corpus, status, evidence = r["id"], r["corpus"], r["status"], r["evidence"]
    if not evidence:
        continue  # C34, C40 -- genuinely unsupported, no judgments possible
    if corpus == "case":
        for i, ev in enumerate(evidence):
            grade = 2 if i == 0 else 1
            qrels.append({
                "query_id": qid, "corpus": corpus, "chunk_id": ev["chunk_id"],
                "case_key": ev.get("case_key"), "citation": ev.get("citation"),
                "grade": grade,
                "rationale": f"Headnote on-point to query topic (rank {i+1} of {len(evidence)} pooled candidates)." if i>0 else "Strongest on-point headnote identified via keyword search.",
                "pooling_status": "pooled_incomplete",
                "provenance": "researcher_judged",
            })
    else:
        for ev in evidence:
            qrels.append({
                "query_id": qid, "corpus": corpus, "chunk_id": ev["chunk_id"],
                "article": ev.get("article"), "grade": 2,
                "rationale": "Direct constitutional/statutory text answering the query (verified by reading full chunk text)." if corpus=="constitution" else "Direct statutory text answering the query (verified by reading full section text).",
                "pooling_status": "representative_not_exhaustive",
                "provenance": "researcher_judged",
            })

with open(os.path.join(OUT_DIR, "qrels.jsonl"), "w") as f:
    for q in qrels:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")

print("qrels rows:", len(qrels))
from collections import Counter
print(Counter(q["corpus"] for q in qrels))
print("queries with >=1 qrel:", len(set(q["query_id"] for q in qrels)))
