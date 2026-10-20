# Final Read-Only Retrieval Quality Audit

## Scope

This is an **artifact-only synthesis audit**. No query was issued to Qdrant, no retrieval
or metric-calculation script was executed, no qrels were expanded, and no embeddings were
generated. Every number in this audit's outputs is either (a) quoted directly from an
existing, previously-generated artifact, or (b) a documented, read-only derivation of
numbers already present in those artifacts (e.g. computing a difference between two already-
reported macro-averages). Every such derivation is labeled inline as "derived from reported
values," not independently recomputed from raw rankings.

## Primary source artifacts inspected

1. **`evaluation/results/final_family_law_150_query_benchmark_20261019_212630/`** — the
   150-query (50 Constitution, 50 statute, 50 case-law) dense/sparse/hybrid benchmark.
   Files read: `README.md`, `comparison_report.md`, `metrics_summary.json`,
   `metrics_per_query.csv`, `bootstrap_ci_results.json`, `queries_150.jsonl`,
   `query_coverage.csv`, `qrels.jsonl`, `corpus_index_coverage.md`, `reproducibility.md`,
   `integrity_confirmation.md`.
2. **`evaluation/results/case_law_qrels_expansion_20261019_220106/`** — the expanded
   case-law relevance judgments and recomputed case-law metrics (primary case-law evidence,
   superseding the 150-query benchmark's original case-law numbers per that task's own
   instructions). Files read: `README.md`, `comparison_report.md`, `metrics_summary.json`,
   `metrics_per_query.csv`, `bootstrap_ci_results.json`, `pool_completeness.csv`,
   `judgment_ledger.jsonl`, `qrels_expanded.jsonl`, `integrity_confirmation.md`.
3. **`evaluation/results/case_law_reranker_experiment_20261019_224929/`** — offline
   cross-encoder reranking of the hybrid case-law baseline. Inspected only to understand
   its findings ABOUT the hybrid baseline (candidate quality, qrels-depth limitation it
   exposed); its reranked outputs are explicitly NOT treated as a hybrid retrieval result.
   Files read: `README.md`, `comparison_report.md`, `metrics_summary.json`,
   `significance_analysis.json`.
4. **`evaluation/results/cl11_cl18_fusion_experiment/`** — prior controlled offline
   RRF-fusion experiment diagnosing two specific demotion cases. Files read:
   `cl11_cl18_diagnosis.md`, `final_recommendation.md`, `fusion_comparison.md`,
   `fusion_failure_diagnostics.csv`.
5. **`evaluation/results/constitution_caselaw_40q/`** — an earlier (33-query: 20
   Constitution + 13 case-law) benchmark, superseded in scale by the 150-query benchmark but
   checked for consistency. Files read: `retrieval_comparison.md`, `final_recommendation.md`,
   `retrieval_metrics.json`.

## Secondary artifacts checked for conflicts (not used as primary evidence)

- `evaluation/results/sri_phase2_baseline_metrics.json` and the broader `sri_*` series
  (16-query statute-only benchmark): a single-method (hybrid) baseline measurement with
  **no dense/sparse comparison and no significance testing** — not comparable to this
  audit's question ("is hybrid demonstrably best") and not used as evidence of a method
  ranking. Flagged in `evidence_inventory.csv` as out-of-scope for this comparison.
- `evaluation/results/full_corpus_retrieval_audit/` (6 Constitution + 6 case-law
  micro-benchmark): similarly a single-method baseline run, superseded in scale by both the
  40-query and 150-query benchmarks. Not used as primary evidence.

## What this audit did NOT do

- Did not rerun `compute_metrics.py`, `bootstrap_ci.py`, `run_retrieval.py`, or any other
  evaluation script from any prior task.
- Did not query Qdrant (its point count was checked once, read-only, purely as an
  integrity/unchanged-state confirmation — see `integrity_confirmation.md`).
- Did not alter, delete, or reconcile any conflicting numbers found between the 40-query and
  150-query benchmarks — both are reported, with the discrepancy documented explicitly in
  `corpus_comparison.md`.
