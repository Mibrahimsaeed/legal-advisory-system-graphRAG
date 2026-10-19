# Integrity Confirmation

## Git state

**Initial and final `git status --short`:** identical — only two untracked directory entries (`evaluation/results/cl11_cl18_fusion_experiment/` and the pre-existing, untouched `evaluation/results/constitution_caselaw_40q/`). **`git diff --stat`: empty, both before and after.** No tracked file was modified.

## Beyond git diff — explicit filesystem verification

Per this task's own caution not to rely on `git diff` alone, a filesystem-level check was run:

```
find . -newer <audit-start-marker> -not -path "./evaluation/results/cl11_cl18_fusion_experiment/*" -not -path "./.git/*" -type f
```

**Result: zero files.** No file anywhere in the repository, outside this audit's own directory and `.git`, has a modification time newer than the moment this audit began. This directly confirms the frozen benchmark directory (`constitution_caselaw_40q/`) was not touched, independent of git tracking status.

## Qdrant

**Point count: 26,678 before and after** (re-verified via a read-only `GET /collections/pk_legal` call). Every Qdrant interaction in this audit was a read-only `query_points()` search call; no insert, upsert, delete, or configuration change was made.

## Frozen benchmark/qrels checksums

All 9 reused files from `evaluation/results/constitution_caselaw_40q/` were checksummed before this task began and re-checksummed at the end — **all 9 are byte-identical, confirmed via direct `diff` of the two checksum captures.** The frozen benchmark was read, never written.

## Files created (exact list, all inside `evaluation/results/cl11_cl18_fusion_experiment/`, nothing elsewhere)

`audit_manifest.json`, `cl11_cl18_diagnosis.md`, `fusion_experiment_config.json`, `fusion_per_query_rankings.jsonl`, `fusion_metrics.json`, `fusion_comparison.md`, `fusion_failure_diagnostics.csv`, `final_recommendation.md`, `integrity_confirmation.md`, plus supporting evidence files: `run_branch_candidates40.py`, `fusion_experiment.py`, `retrieval_dense_c40.jsonl`, `retrieval_sparse_c40.jsonl`, `rrf_validation.json`, `fusion_raw_per_query.json`.

`PYTHONDONTWRITEBYTECODE=1` was set for every Python invocation; the only bytecode caches created were transient, inside this audit directory itself (from explicit `py_compile` syntax checks), and were removed during cleanup — never a violation of the write restriction, since they never existed outside the permitted directory.

## What could not be fully resolved (disclosed, not hidden)

- **2 of 33 queries (C10, C13)** show a genuinely different top-10 candidate *set* under the offline k=1 RRF formula versus production — not merely a reordering. The root cause was not isolated in this task's time budget; flagged as a known limitation of the offline approximation in `fusion_experiment_config.json` and `fusion_failure_diagnostics.csv`, not silently absorbed into the headline results.
- **C13's exact competing chunk(s) under the 0.5-weight regression** were not individually traced (time-bounded).
- **Qdrant's internal RRF constant (k=1, empirically validated here) was not independently confirmed against Qdrant's own source code or official documentation** — it is validated by reproduction accuracy against the frozen production ranking, which is the strongest evidence available from this project's own artifacts, but is disclosed as an empirical fit, not a verified specification value.
