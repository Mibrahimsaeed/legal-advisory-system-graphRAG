import json, csv
from pathlib import Path
from collections import defaultdict

AUDIT_DIR = Path(__file__).resolve().parent
rows = json.load(open(AUDIT_DIR / "metrics_per_query_raw.json"))

# --- metrics_per_query.csv ---
fields = ["query_id","corpus","method","n_retrieved","p_at_5","ndcg_at_10","recall_at_5","recall_at_10",
          "mrr_at_10","hit_rate_at_5","hit_rate_at_10","doc_recall_at_5","doc_recall_at_10","doc_mrr_at_10",
          "doc_hit_rate_at_5","doc_hit_rate_at_10","p_at_5_note","ndcg_at_10_note"]
with open(AUDIT_DIR / "metrics_per_query.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)

# --- metrics_summary.json ---
def avg(vals):
    vals = [v for v in vals if v is not None]
    return (sum(vals)/len(vals), len(vals)) if vals else (None, 0)

summary = {"generated_from": "metrics_per_query_raw.json", "corpora": {}}
for corpus in ("constitution","statute","case"):
    summary["corpora"][corpus] = {}
    for method in ("dense","sparse","hybrid"):
        subset = [r for r in rows if r["corpus"]==corpus and r["method"]==method]
        n_total = len(subset)
        entry = {"n_eligible_queries": n_total}
        for metric in ("p_at_5","ndcg_at_10","recall_at_5","recall_at_10","mrr_at_10","hit_rate_at_5","hit_rate_at_10"):
            m, n = avg([r[metric] for r in subset])
            entry[metric] = {"macro_avg": round(m,4) if m is not None else None, "denominator": n, "n_missing_or_unavailable": n_total-n}
        if corpus == "case":
            for metric in ("doc_recall_at_5","doc_recall_at_10","doc_mrr_at_10","doc_hit_rate_at_5","doc_hit_rate_at_10"):
                m, n = avg([r[metric] for r in subset])
                entry[metric] = {"macro_avg": round(m,4) if m is not None else None, "denominator": n}
        summary["corpora"][corpus][method] = entry

summary["limitations"] = {
    "constitution_statute_p5_ndcg10": "Fully computable (no missing labels) because relevance is structurally determined by provision identity (article_number / section root), verified by direct text reading -- not an approximation.",
    "case_p5_ndcg10": "NOT reliably computable -- reported as unavailable for all case-law queries/methods. Only the pooled qrels chunk(s) per query have been read and judged; all other retrieved case-law chunks are UNJUDGED (not proven irrelevant), so P@5/nDCG@10 cannot be honestly computed under the strict missing-label rule. This is a disclosed limitation of pooling depth, not a retrieval failure.",
    "case_recall_mrr_hitrate_chunk_level": "Computed against the exact qrels chunk_id(s) (typically the single headnote chunk read per case). Likely UNDERSTATES true performance because a case has many retrievable chunks (headnote/headnote_part/judgment/judgment_part) and retrieval may correctly surface a different chunk of the SAME relevant case.",
    "case_recall_mrr_hitrate_doc_level": "Computed against the case_key (document identity) regardless of which chunk of that case was retrieved -- a more meaningful view for case law, but still likely understates true performance because the qrels pool (2-3 cases per query, found via one keyword-search pass) is small relative to a topically dense 2,084-document, 25,891-chunk family-law corpus. Spot-check confirmed this directly: for CL01, the dense-retrieval top-1 result (case pld-2024-fsc-9) is a genuine, on-point Khula judgment that was simply not included in this benchmark's qrels pool -- i.e. low doc-level recall here reflects incomplete pooling, not failed retrieval. Do not interpret case-law Recall/MRR/HitRate numbers in this benchmark as a reliable absolute measure of retrieval quality; they are only valid for comparing the THREE METHODS against each other on the SAME (incomplete) reference set, not for judging absolute retrieval adequacy.",
}

json.dump(summary, open(AUDIT_DIR / "metrics_summary.json","w"), indent=2)
print(json.dumps(summary["corpora"], indent=2))
