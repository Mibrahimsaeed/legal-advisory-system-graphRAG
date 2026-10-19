# Integrity Confirmation

## Git state

**`git status --short`, before and after:** identical — only four untracked directories:
this task's own (`case_law_qrels_expansion_20261019_220106/`) and three pre-existing,
untouched directories from prior sessions (`final_family_law_150_query_benchmark_20261019_212630/`,
`constitution_caselaw_40q/`, `cl11_cl18_fusion_experiment/`).

**`git diff --stat`: empty, both before and after.** No tracked file was modified.

## Filesystem-level check (beyond git diff)

```
find . -newer evaluation/results/case_law_qrels_expansion_20261019_220106/_initial_state.txt \
  -not -path "./evaluation/results/case_law_qrels_expansion_20261019_220106/*" \
  -not -path "./.git/*" -type f
```

**Result: zero files.** No file anywhere in the repository, outside this task's own
directory and `.git`, has a modification time newer than the moment this task began.

## Qdrant

**Point count: 26,678 before and after** (`GET /collections/pk_legal`). This task issued
**zero** Qdrant queries — all retrieval rankings were reused by reading the prior
benchmark's already-saved JSONL files; the only live reads were read-only `sqlite3 SELECT`
statements against `var/rag/docstore.sqlite` for `case_key`/text lookups.

## Frozen prior-benchmark files — re-checksummed, confirmed unchanged

| File | Checksum (unchanged from task start) |
|---|---|
| `retrieval_dense_case.jsonl` | `418d94da21f96bc838d40241c3034d43` |
| `retrieval_sparse_case.jsonl` | `cf38bf97aa997717b4a14c5cd7ed4c92` |
| `retrieval_hybrid_case.jsonl` | `b57753cb916c6e843c09733a593069a1` |
| `qrels.jsonl` (original) | `d40929c00f6a57b2b19e451ea9e22f42` |
| `queries_150.jsonl` | `5a3c8c03698aa8d86b6abf8e9446824d` |

All five were read, never written to, throughout this task.

## Document and chunk identifier mapping — verified, not assumed

- `point_id = uuid5(fixed_namespace, chunk_id)` was independently recomputed for a sample
  chunk and matched the Qdrant-returned `point_id` exactly (see Phase 1 inspection notes).
- `case_key` was confirmed to be a SEPARATE payload field from `doc_id` — for most cases a
  human-readable citation slug, but for a handful of non-standard-citation judgments
  `case_key` literally equals the raw `doc_id` hash. Every judged case_key in this task was
  validated against the actual pool (`judging_batches.json`) before being merged, catching
  and fixing several transcription mismatches during the judging passes (documented
  in-line in the batch-build commands).
- Duplicate cases are not double-counted: `pool_summary_raw.json` and the ledger are both
  keyed by `case_key`, so a case appearing via multiple chunks or in multiple methods'
  rankings is represented once, with per-method rank/chunk-id recorded as sub-fields.

## Queries unchanged

All 50 CL01–CL50 query IDs and texts were parsed directly from the original benchmark's
`queries_150.jsonl` (not retyped) and used verbatim throughout; zero mismatches confirmed
at task start (see Phase 1 inspection).

## Reproducibility of metrics from saved artifacts

`metrics_per_query.csv` / `metrics_summary.json` are fully reproducible by re-running
`recompute_metrics.py` against `judgment_ledger.jsonl` and the three frozen `retrieval_*_case.jsonl`
files alone — no other state is required.

## What could not be fully resolved (disclosed, not hidden)

- 1,506 of 2,207 pooled cases (68%) remain unjudged beyond this task's top-10-per-method
  judging depth; Recall@40 and "total relevant cases in the corpus" claims are explicitly
  NOT supported by this task's evidence.
- No lawyer or independent legal reviewer validated the judgments; they are
  researcher-judged / machine-assisted, consistent with the original benchmark's labeling
  discipline, and must never be described as lawyer-validated.
- An initial version of the qrels-merge script silently dropped 80 previously-judged
  relevant cases that fall outside all three methods' top-40 pools; this was caught via a
  direct set-difference check against the original `qrels.jsonl` before any downstream
  metric was computed, and fixed. Documented transparently here rather than silently
  corrected without mention.
