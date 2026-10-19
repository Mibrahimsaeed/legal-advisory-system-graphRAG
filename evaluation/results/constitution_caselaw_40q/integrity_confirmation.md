# Integrity Confirmation

## Git state

**Initial `git status --short`:** only the new audit directory itself appeared (`?? evaluation/results/constitution_caselaw_40q/`), captured before any file inside it existed. **Final `git status --short`:** identical — still only the one new directory entry (git collapses an untracked directory to one line regardless of how many files it contains).

**`git diff --stat`: empty, both before and after.** No tracked file was modified.

## Beyond git diff — explicit filesystem verification

Per the task's own caution not to rely on `git diff` alone (since ignored/untracked files elsewhere would not appear there), this audit additionally ran:

```
find . -newer <audit-start-marker> -not -path "./evaluation/results/constitution_caselaw_40q/*" -not -path "./.git/*" -type f
```

**Result: zero files.** No file anywhere in the repository outside this audit's own directory (and outside `.git`) has a modification time newer than the moment this audit began. A second, targeted check for stray `__pycache__` directories (bytecode caches) anywhere in the repo, newer than the audit start, also returned **zero results** — `PYTHONDONTWRITEBYTECODE=1` was set for every Python invocation in this audit, and no import-time cache was written outside the permitted directory. (Two `.pyc` files were briefly created *inside* the audit directory itself, as a direct result of an explicit `python3 -m py_compile` syntax check on the two new scripts — not a stray side effect, and since they were inside the permitted zone they were simply removed during cleanup, not a violation.)

## Qdrant

**Point count: 26,678 before and after** (re-verified via a read-only `GET /collections/pk_legal` call). No insert, upsert, delete, or collection-configuration call was made — every Qdrant interaction in this audit was `query_points` (read-only search).

## Reused-file checksums

| File | Checksum (unchanged, before = after) |
|---|---|
| `evaluation/results/full_corpus_retrieval_audit/queries_constitution6.json` | `2cd60e5d4815e770c1415b3e3bbdb2fe` |
| `evaluation/results/full_corpus_retrieval_audit/queries_caselaw6.json` | `67e64b64f81420c80cc8bc95f7b376c0` |
| `evaluation/results/full_corpus_retrieval_audit/qrels_caselaw_constitution.json` | `7d45db4376f5356de71037338763e589` |
| `evaluation/queries_statute_benchmark16.json` | `f98c113dc733bd1ea24ac98c3fe11ba9` |
| `evaluation/results/sri_phase1_qrels.json` | `3102c88b927835af449249f303958d7f` |

None of these pre-existing files were read as a basis for retroactive overwriting in this audit — they were consulted only for cross-referencing prior work (e.g. reusing already-verified chunk IDs for C12/C17), and none were written to.

## What could not be evaluated

- **Combined-corpus (Constitution + case law together) retrieval** — not evaluated. Confirmed by direct code reading that `hybrid_search()` has no code path producing an unfiltered, single cross-corpus ranking; fabricating a non-production-equivalent diagnostic for this was out of this task's time budget and is reported as not evaluated, per the task's own instruction, rather than approximated.
- **7 of the original 20 case-law queries (CL03, CL05, CL06, CL09, CL10, CL19, CL20)** — not scored; reported as corpus-unsupported with evidence in `benchmark_coverage.md` and `queries_caselaw.json`, not fabricated.
- **Bootstrap confidence intervals** — not computed; with 20/13 queries per corpus, this audit reports descriptive differences only and explicitly does not claim statistical significance anywhere in `retrieval_comparison.md` or `final_recommendation.md`.

## Files created (exact list, all inside `evaluation/results/constitution_caselaw_40q/`, nothing elsewhere)

`audit_manifest.json`, `queries_constitution.json`, `queries_caselaw.json`, `qrels_constitution.json`, `qrels_caselaw.json`, `qrels_combined_simple.json`, `queries_scored_for_retrieval.json`, `benchmark_coverage.md`, `per_query_rankings.jsonl`, `retrieval_metrics.json`, `retrieval_comparison.md`, `failure_diagnostics.csv`, `final_recommendation.md`, `integrity_confirmation.md`, `run_dense_sparse.py`, `compute_metrics.py`, `retrieval_hybrid.jsonl` (+`.meta.json`), `retrieval_dense.jsonl`, `retrieval_sparse.jsonl`.
