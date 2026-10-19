# Integrity Confirmation

## Git state

**`git status --short`, before and after:** only this experiment's own directory
(`case_law_reranker_experiment_20261019_224929/`) is untracked; no other file changed.

**`git diff --stat`: empty, both before and after.** No tracked file was modified.

## Filesystem-level check (beyond git diff)

```
find . -newer evaluation/results/case_law_reranker_experiment_20261019_224929/_initial_state.txt \
  -not -path "./evaluation/results/case_law_reranker_experiment_20261019_224929/*" \
  -not -path "./.git/*" -not -path "./.venv/*" -type f
```

**Result: zero files.** No project file anywhere in the repository, outside this
experiment's own directory, `.git`, and the Python virtual environment, has a modification
time newer than the moment this task began.

## Qdrant

**Point count: 26,678 before and after** (`GET /collections/pk_legal`). This experiment
issued **zero** Qdrant queries — it reused the already-saved `retrieval_hybrid_case.jsonl`
rankings from the 150-query benchmark exactly as written.

## Frozen input files — re-checksummed, confirmed unchanged

| File | Checksum (unchanged from task start) |
|---|---|
| `final_family_law_150_query_benchmark_.../queries_150.jsonl` | `5a3c8c03698aa8d86b6abf8e9446824d` |
| `final_family_law_150_query_benchmark_.../retrieval_hybrid_case.jsonl` | `b57753cb916c6e843c09733a593069a1` |
| `case_law_qrels_expansion_.../qrels_expanded.jsonl` | `5831bc27a0ac0332e4eb21a66139b258` |
| `case_law_qrels_expansion_.../judgment_ledger.jsonl` | `c83a1d08abf664c1af0d5e3b2589d503` |
| `case_law_qrels_expansion_.../pool_rankings.jsonl` | `6fed0ff0ab866f55c4cc0192d2aa9bbc` |
| `case_law_qrels_expansion_.../metrics_summary.json` | `1020b89532f7631fb4bcd4c9ddd97f78` |
| `case_law_qrels_expansion_.../comparison_report.md` | `ebaa2a6ce25abcd2d5422a47ef0466de` |
| `case_law_qrels_expansion_.../README.md` | `9cfb1cd00abe4eca7adaa50ead441501` |

All eight were read, never written to, throughout this task.

## Model download — outside the repository, not a project file

`BAAI/bge-reranker-v2-m3` was downloaded to the user-level Hugging Face cache
(`~/.cache/huggingface/hub/`), the same mechanism this project's own production code
(`src/rag_prep/embed_index.py`) already uses to cache `BAAI/bge-m3`. This is a machine-level
model cache, not tracked by git and outside the repository; it was not written to any
location inside the repository or this experiment directory. No package was installed
globally and no project dependency file (`pyproject.toml`/`requirements.txt`/lockfile) was
modified — `sentence-transformers`, `transformers`, and `torch` were already present in the
project's existing virtual environment.

## Queries and candidates — verified, not assumed

- All 50 `CL01`–`CL50` query IDs and texts were read directly from the frozen
  `retrieval_hybrid_case.jsonl` (itself traced back to `queries_150.jsonl` in the original
  benchmark) — never retyped or paraphrased.
- Every one of the 2,000 candidates (50 queries × 40) resolved to a non-null `case_key` via
  `var/rag/docstore.sqlite` (read-only `SELECT`) and had non-empty `canonical_text` (0 empty,
  confirmed in `run_reranker_experiment.py`'s own assertion/count and reproduced in
  `latency_report.json`).
- Original hybrid rank and `rrf_score` were copied verbatim into both
  `baseline_hybrid_rankings.jsonl` and `reranked_rankings.jsonl` (as separate fields from
  the new `reranker_score`/`reranked_rank`) — never overwritten.
- The reranker was given only `(query_text, candidate_text)` pairs at inference time; it was
  never given a relevance grade, qrels file, or qrels path. The `judgment_ledger.jsonl`
  frozen qrels were read only by `compute_metrics.py`, a separate script invoked strictly
  after reranking had already completed and been written to disk.

## What could not be fully resolved (disclosed, not hidden)

- See `comparison_report.md` §2 for the qrels-depth gap this experiment surfaced (nDCG@10
  only reliably computable for 7/50 queries after reranking) — not a defect in this
  experiment's own execution, but a limitation of the qrels it was evaluated against.
- Latency/memory figures are specific to this machine's Apple Silicon MPS backend and were
  not cross-validated against CPU or CUDA hardware.
