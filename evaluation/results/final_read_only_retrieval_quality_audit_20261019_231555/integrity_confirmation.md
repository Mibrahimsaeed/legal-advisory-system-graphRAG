# Integrity Confirmation

## Git state

**`git status --short`, before and after:** only this audit's own directory
(`final_read_only_retrieval_quality_audit_20261019_231555/`) is untracked; no other file in
the repository changed.

**`git diff --stat`: empty, both before and after.** No tracked file was modified.

## Filesystem-level check (beyond git diff)

```
find . -newer evaluation/results/final_read_only_retrieval_quality_audit_20261019_231555/_initial_state.txt \
  -not -path "./evaluation/results/final_read_only_retrieval_quality_audit_20261019_231555/*" \
  -not -path "./.git/*" -not -path "./.venv/*" -type f
```

**Result: zero files.** No project file anywhere in the repository, outside this audit's own
directory, `.git`, and the Python virtual environment, has a modification time newer than
the moment this audit began. This directly confirms that none of the inspected prior-task
directories were touched during inspection.

## Qdrant

**Point count: 26,678 before and after** (`GET /collections/pk_legal`, a read-only
collection-metadata request, not a retrieval query against any collection data). This audit
issued **no** search, scroll, or retrieval request of any kind against Qdrant — the single
collection-info check above was performed once at the start and once at the end, purely to
confirm the collection's state is unchanged, consistent with the task's explicit prohibition
on issuing queries or running retrieval functions.

## No scripts executed

This audit ran no Python evaluation script. It did not invoke `compute_metrics.py`,
`bootstrap_ci.py`, `run_retrieval.py`, `recompute_metrics.py`, `significance_analysis.py`,
`run_reranker_experiment.py`, or any other script from any previously-generated task
directory. Every number quoted in `corpus_comparison.md`, `known_failure_cases.md`, and
`decision_matrix.csv` was read directly from an existing `.json`/`.csv`/`.md` artifact via
`cat`/`python3 -c "json.load(...)"`-style inspection (no computation beyond trivial
dictionary indexing to print already-stored values), or, where explicitly marked, a verified
read of a CSV via `csv.DictReader` and `collections.Counter` to confirm an already-reported
count — never a re-derivation from raw rankings or qrels.

## Re-checksummed source files — confirmed unchanged from the start of this audit

| File | Checksum |
|---|---|
| `final_family_law_150_query_benchmark_.../metrics_summary.json` | `8b05e3ba803e0a982c1e1186830415ef` |
| `final_family_law_150_query_benchmark_.../bootstrap_ci_results.json` | `4286f4d0a04cd4af680df84c7ce7de2b` |
| `case_law_qrels_expansion_.../metrics_summary.json` | `1020b89532f7631fb4bcd4c9ddd97f78` |
| `case_law_qrels_expansion_.../bootstrap_ci_results.json` | `49a6d3bb642f9179c77e60a665821bb0` |
| `case_law_reranker_experiment_.../comparison_report.md` | `f7bcfc589d2052233e9ffa481ecc5e3b` |
| `cl11_cl18_fusion_experiment/final_recommendation.md` | `44a7c0da23077df519a36730c29b1452` |

All six were read, never written to, throughout this audit. (These checksums were recorded
once, at the end of the audit, as a confirmation that the files were not modified during
inspection; no separate "before" checksum pass was performed for these specific files since
this is a read-only synthesis task with no computation step that could have touched them —
the git-diff and filesystem-sweep checks above are the authoritative before/after proof of
no modification across the whole repository, including these files.)

## Scope confirmation

All seven required output artifacts (`README.md`, `corpus_comparison.md`,
`decision_matrix.csv`, `evidence_inventory.csv`, `known_failure_cases.md`,
`final_recommendation.md`, `integrity_confirmation.md`) were created exclusively inside
`evaluation/results/final_read_only_retrieval_quality_audit_20261019_231555/`. No package
was installed, no model was downloaded or invoked, and no production code, qrels, rankings,
Qdrant data, SQLite data, or Neo4j data was altered.
