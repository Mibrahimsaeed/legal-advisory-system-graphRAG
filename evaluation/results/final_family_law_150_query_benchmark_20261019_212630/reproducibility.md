# Reproducibility

## Environment / config verified (read-only inspection, Section 3 of the task)

- Qdrant collection: `pk_legal` (confirmed via `GET /collections/pk_legal`)
- Dense vector: named `dense`, size 1024, distance Cosine, model `BAAI/bge-m3` (per
  `src/rag_prep/embed_index.py` constants `MODEL_NAME`/`DENSE_DIM`, re-read at the start of
  this task, not assumed)
- Sparse vector: named `bm25`, modifier IDF (`Qdrant/bm25` via fastembed)
- Point counts at start and end of this task: case=25,891, constitution=547, statute=240,
  total=26,678 (unchanged)
- Fusion: Qdrant `FusionQuery(fusion=Fusion.RRF)` inside `hybrid_search()`
  (`src/rag_prep/embed_index.py:500-520`), always filtered by `source_type` per corpus —
  **no combined-corpus ranking path exists**, reconfirmed by direct code reading.
- Candidate depth: `candidates=40` (the function's own default; passed explicitly as `40` in
  every retrieval call in this benchmark) is the per-branch (dense/sparse) prefetch depth
  before RRF fusion.

## Commands (in order)

```bash
export PYTHONDONTWRITEBYTECODE=1
AUDIT_DIR=evaluation/results/final_family_law_150_query_benchmark_20261019_212630

# 1. Query validation + qrels (read-only corpus inspection via docstore.sqlite)
python3 $AUDIT_DIR/build_queries_and_qrels.py
python3 $AUDIT_DIR/build_qrels.py

# 2. Production hybrid retrieval (run_retrieval.py, UNMODIFIED), one batch per corpus
python3 evaluation/run_retrieval.py --queries $AUDIT_DIR/queries_constitution_input.json \
  --out $AUDIT_DIR/retrieval_hybrid_constitution.jsonl \
  --k-constitution 40 --k-case 1 --k-statute 1 --candidates 40
python3 evaluation/run_retrieval.py --queries $AUDIT_DIR/queries_statute_input.json \
  --out $AUDIT_DIR/retrieval_hybrid_statute.jsonl \
  --k-statute 40 --k-case 1 --k-constitution 1 --candidates 40
python3 evaluation/run_retrieval.py --queries $AUDIT_DIR/queries_case_input.json \
  --out $AUDIT_DIR/retrieval_hybrid_case.jsonl \
  --k-case 40 --k-statute 1 --k-constitution 1 --candidates 40

# 3. Dense-only / sparse-only retrieval (depth 40), reusing connect/load_dense/load_sparse/
#    encode_dense from embed_index.py UNMODIFIED
python3 $AUDIT_DIR/run_dense_sparse_150.py

# 4. Metrics
python3 $AUDIT_DIR/compute_metrics.py
python3 $AUDIT_DIR/build_summary.py
python3 $AUDIT_DIR/bootstrap_ci.py
```

Note on `--k-case 1`/`--k-statute 1`/`--k-constitution 1` for the two "other" corpora in each
batch: `run_retrieval.py`'s `per_source` dict always queries every configured source_type for
every query regardless of that query's own category, and Qdrant rejects `limit=0`. `k=1` is the
minimal valid value; those throwaway cross-corpus rows are dropped before scoring (filtered out
in `compute_metrics.py`'s `load_rankings(..., corpus_filter=corpus)` and in building
`retrieval_rankings.jsonl`) and documented here rather than silently worked around.

## Metric definitions

- **P@5**: fraction of top-5 results with grade ≥ 1. Denominator is always 5.
- **Recall@5/10**: `|top-k ∩ known-relevant-set| / |known-relevant-set|`, where the
  known-relevant-set is this benchmark's qrels for that query (disclosed as pooled/incomplete
  for case law, structurally exhaustive for Constitution/statute).
- **MRR@10**: reciprocal rank of the first known-relevant chunk within top 10, else 0.
- **nDCG@10**: graded relevance, gain = 2^rel − 1, log2 discount; computed only when every rank
  1–10 has a known grade (else reported unavailable — see `metrics_summary.json.limitations`).
- **HitRate@5/10**: 1 if any known-relevant chunk appears in top-k, else 0.
- **Doc-* (case law only)**: same definitions, but computed over `case_key` (document identity)
  instead of exact `chunk_id`, since a judgment has many retrievable chunks and any one of them
  being retrieved should count as the document being found.
- Grading rule for Constitution/statute: a candidate's relevance is 2 if its `article_number`
  (Constitution) or section root (statute, e.g. `s7` from `s7_ss1A-2-3-4-5-6`) matches one of the
  query's recorded target provisions, else 0 — this is exhaustive/structural, not sampled.
- Grading rule for case law: 2/1 if the candidate's exact `chunk_id` is in `qrels.jsonl` for that
  query, else **unjudged (None)** — never coerced to 0. This is why case-law P@5/nDCG@10 are
  reported unavailable rather than computed with unjudged-as-irrelevant.

## Bootstrap procedure

Paired bootstrap over queries: for each comparison (method A vs. method B, same corpus, same
metric), resample query indices with replacement (5,000 resamples, `random.seed(42)`), recompute
the mean paired difference each resample, report the 2.5th/97.5th percentile as the 95% CI.
Significance = CI excludes zero. Full results: `bootstrap_ci_results.json`.

## Random seeds

- Bootstrap: `seed=42`, `n_resamples=5000` (only source of randomness in this benchmark; all
  retrieval and grading steps are deterministic given the fixed indexed corpus).
