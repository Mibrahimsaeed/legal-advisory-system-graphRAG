# Reproducibility

## Commands (in order)

```bash
export PYTHONDONTWRITEBYTECODE=1
NEW_DIR=evaluation/results/case_law_qrels_expansion_20261019_220106

# 1. Build the structural pool from the EXISTING frozen retrieval files (no re-querying)
python3 $NEW_DIR/build_pool.py

# 2. Compute, per query, the NEW candidates needing judgment (union of top-10 per method,
#    minus cases already judged in the original benchmark)
python3 $NEW_DIR/prepare_judging_batches.py

# 3. Manual judging pass (656 cases, read via docstore.sqlite headnote/judgment text) --
#    recorded in manual_judgments_01.json .. manual_judgments_05.json, each validated
#    against judging_batches.json's pool before merging.

# 4. Merge old + new judgments into the ledger, expanded qrels, and completeness stats
python3 $NEW_DIR/build_ledger_and_qrels.py

# 5. Recompute case-law metrics (document-level grading) against the FROZEN, UNCHANGED
#    dense/sparse/hybrid rankings from the original 150-query benchmark
python3 $NEW_DIR/recompute_metrics.py

# 6. Statistical comparison
python3 $NEW_DIR/bootstrap_ci.py
```

## Key design decisions, stated explicitly

- **Retrieval was NOT re-run.** `retrieval_dense_case.jsonl`, `retrieval_sparse_case.jsonl`,
  and `retrieval_hybrid_case.jsonl` from `evaluation/results/final_family_law_150_query_benchmark_20261019_212630/`
  were read (never modified) and reused exactly as-is. This guarantees the "before vs after"
  comparison in `comparison_report.md` §5 isolates the effect of judgment quality, not any
  retrieval nondeterminism (dense-model/BM25 reruns can differ slightly run to run).
- **Judging depth = top-10 of each method, union, minus already-judged** (656 cases). This
  exactly matches every reported metric's ranking window (P@5, Recall@5/10, MRR@10,
  nDCG@10, HitRate@5/10 all look no deeper than rank 10). Ranks 11–40 are retained in the
  pool (`pool_summary_raw.json`, `pool_rankings.jsonl`) for completeness statistics
  (`pool_completeness.csv`) but intentionally left unjudged.
- **Grading unit = case (`case_key`), not `chunk_id`.** A case-law judgment (headnote) is a
  statement about a judgment/case, which Qdrant indexes as many separate chunks (caption,
  headnote_part ×N, counsel, judgment ×N). Relevance is propagated from the judged case to
  ANY of its chunks encountered in a ranking. `case_key` was verified (not assumed) against
  `doc_id`/`chunk_id` via `var/rag/docstore.sqlite` for every case touched.
- **Previously-judged cases absent from the top-40 pool were preserved, not dropped.** An
  early version of the merge script silently excluded 80 of the original 125 relevant
  judgments because they fell outside all three methods' top-40 retrieval. This was caught
  by a direct before/after diff against the original `qrels.jsonl` and fixed; see the
  comment in `build_ledger_and_qrels.py` and `comparison_report.md` §6.

## Metric definitions

Identical to the original benchmark's `reproducibility.md` (P@5, Recall@K, MRR@10, nDCG@10,
HitRate@K), with one change: the relevant-set membership test and the per-rank relevance
grade both operate on `case_key` (document identity) rather than exact `chunk_id`, per the
design decision above.

## Random seeds

- Bootstrap: `seed=42`, `n_resamples=5000` (same procedure/seed as the original 150-query
  benchmark, for comparability). All other steps (judging, pooling, metric computation) are
  deterministic given the fixed indexed corpus and the already-frozen retrieval rankings.

## Qdrant / environment

No Qdrant query was issued by this task — `build_pool.py` reads only local JSONL files from
the prior benchmark; the only external reads are `sqlite3` `SELECT` statements against
`var/rag/docstore.sqlite` (read-only, for `case_key`/text lookups). See
`integrity_confirmation.md` for the before/after Qdrant point-count and checksum confirmation.
