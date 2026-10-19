# Case-Law Qrels Expansion — Strengthening Retrieval Evaluation Before Reranking

## Scope

This task strengthens the case-law (CL01–CL50) relevance judgments from the earlier
150-query benchmark (`evaluation/results/final_family_law_150_query_benchmark_20261019_212630/`),
which flagged its original case-law qrels (2–3 cases per query, from one keyword-search
pass) as too incomplete to trust for method comparison. It is **evaluation-only**: no
reranker, no fusion-weight change, no GraphRAG work was implemented or attempted.

## Methodology

1. **Pool construction** (`build_pool.py`): reused the EXISTING, already-saved, unmodified
   top-40 dense/sparse/hybrid rankings from the frozen benchmark (rather than re-querying
   Qdrant), guaranteeing the expanded qrels are measured against the exact rankings the
   original metrics were based on. Deduplicated by `case_key` (verified via
   `var/rag/docstore.sqlite`, not assumed from `chunk_id`/`doc_id`), retaining per-method
   rank/score and chunk-id provenance.
2. **Judging depth** (`prepare_judging_batches.py`): judged every NEW case appearing in the
   top-10 of ANY of the three methods, not previously judged in the original benchmark —
   656 cases, read via their actual headnote/judgment text (never titles/metadata alone).
   This depth was chosen because it is exactly what Recall@10/MRR@10/nDCG@10/HitRate@10
   require; ranks 11–40 remain in the pool for completeness statistics but are explicitly
   left **unjudged**, not treated as irrelevant.
3. **Grading** (`manual_judgments_0{1-5}.json`, merged in `build_ledger_and_qrels.py`):
   2 = directly addresses the legal issue; 1 = partially/indirectly relevant; 0 = not
   relevant; every nonzero grade carries a short textual rationale grounded in the actual
   excerpt read. All 125 original judgments were carried forward unchanged and verified
   byte-for-byte preserved — including 80 that are not retrieved by ANY method within the
   top-40 (a real recall-ceiling finding, not a data-loss bug; see `comparison_report.md`).
4. **Metrics recomputation** (`recompute_metrics.py`): case-law relevance is a
   **document-level** property (a judgment has many chunks: caption/headnote/counsel/
   judgment); grading is therefore propagated from the judged case to ANY chunk of that
   case encountered in a ranking. This resolves the "right case, wrong chunk" dilution
   that made the original per-exact-chunk-id grading understate performance.

## Commands

See `reproducibility.md` for the exact order and commands.

## Limitations

- Ranks 11–40 of each method's pool remain genuinely unjudged for 1,506 of 2,207 pooled
  cases (68%) — not an oversight, a disclosed scope boundary. Recall@10/HitRate@10 are
  therefore reliable (they only require knowing a lower-bound relevant set); Recall@40 or
  an "absolute total relevant" claim would not be.
- Judging is researcher-judged / machine-assisted (keyword-search-directed reading of
  actual case text), **not lawyer-validated**.
- 80 known-relevant cases from the original benchmark are not retrieved by any method
  within top-40 — this caps achievable Recall@10 below what a fully-indexed, perfectly
  pooled denominator would allow, and is reported as a real finding, not excluded.
