# Integrity Confirmation

## Git state

**`git status --short`, before and after:** identical — only three untracked directories:
this benchmark's own directory, plus the two pre-existing, untouched directories from prior
sessions (`evaluation/results/constitution_caselaw_40q/`, `evaluation/results/cl11_cl18_fusion_experiment/`).

**`git diff --stat`: empty, both before and after.** No tracked file was modified.

## Filesystem-level check (beyond git diff)

```
find . -newer evaluation/results/final_family_law_150_query_benchmark_20261019_212630/_initial_state.txt \
  -not -path "./evaluation/results/final_family_law_150_query_benchmark_20261019_212630/*" \
  -not -path "./.git/*" -type f
```

**Result: zero files.** No file anywhere in the repository, outside this benchmark's own
directory and `.git`, has a modification time newer than the moment this task began. This
directly confirms every prior evaluation artifact (including the two frozen directories above)
was not touched.

## Qdrant

**Point count: 26,678 before and after** (`GET /collections/pk_legal`, re-verified). Vector
config unchanged: `dense` (1024, Cosine), `bm25` sparse with IDF modifier. Every Qdrant
interaction in this task was a read-only `query_points()` / `points/count` / collection-info
call; no upsert, delete, or recreate was issued at any point.

## Frozen prior-task artifacts

Re-checksummed at the end of this task and confirmed unchanged from their already-established
values: `evaluation/results/constitution_caselaw_40q/queries_scored_for_retrieval.json`
(`745a063ae608f6601d1420dbf509ad47`), `evaluation/results/cl11_cl18_fusion_experiment/audit_manifest.json`
(`1c010238ae5483c072ca055d0cbb6555`). Neither directory was read-from or written-to during this
task beyond this end-of-task checksum re-verification.

## Files created (all 41, exclusively inside this directory)

Required artifacts (13): `README.md`, `audit_manifest.json`, `queries_150.jsonl`,
`query_coverage.csv`, `qrels.jsonl`, `retrieval_rankings.jsonl`, `metrics_per_query.csv`,
`metrics_summary.json`, `comparison_report.md`, `failure_diagnostics.csv`,
`corpus_index_coverage.md`, `reproducibility.md`, `integrity_confirmation.md`.

Supporting scripts and intermediates (28): `build_queries_and_qrels.py`, `build_qrels.py`,
`build_summary.py`, `compute_metrics.py`, `bootstrap_ci.py`, `run_dense_sparse_150.py`,
`search_caselaw.py`, `validate_constitution.py`, three per-corpus query input files, nine
per-corpus/method raw retrieval `.jsonl`/`.meta.json` files, `metrics_per_query_raw.json`,
`bootstrap_ci_results.json`, `_initial_state.txt`.

`PYTHONDONTWRITEBYTECODE=1` was set for every Python invocation in this task; no `__pycache__`
directory was created anywhere, inside or outside this directory (confirmed: `find . -name
__pycache__` returns nothing new).

## Read-only operations confirmed

Every retrieval call (`hybrid_search()` via unmodified `run_retrieval.py`; dense-only/sparse-only
via unmodified `connect()`/`load_dense()`/`load_sparse()`/`encode_dense()` reused by file-path
import) is a `query_points()` call — Qdrant's read path. All corpus inspection was
`sqlite3`/`SELECT` against `var/rag/docstore.sqlite`, never `INSERT`/`UPDATE`/`DELETE`. No
package was installed, no model was downloaded (the existing `BAAI/bge-m3` and `Qdrant/bm25`
models already cached locally from prior sessions were reused via the existing `load_dense()`/
`load_sparse()` functions, unmodified).

## What could not be fully resolved (disclosed, not hidden)

- Case-law qrels are pooled but incomplete (2–3 cases per query, one keyword-search pass);
  directly confirmed understated via a spot-check (see `comparison_report.md` Section 6). This
  is the single largest limitation of this benchmark and is why case-law P@5/nDCG@10 are
  reported unavailable and case-law method rankings are reported as inconclusive rather than
  decisive.
- Exhaustive point-by-point (all 26,678) dense/sparse vector presence and payload-text-integrity
  checks were not performed — only count-matching and targeted sampling (see
  `corpus_index_coverage.md`).
- No reranker, fusion-weight change, or GraphRAG modification was implemented or attempted, per
  the task's explicit instruction.
