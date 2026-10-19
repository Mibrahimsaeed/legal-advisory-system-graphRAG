# Final Family-Law 150-Query Retrieval Benchmark

## Purpose

A read-only, reproducible evaluation of the existing (unmodified) hybrid retrieval pipeline
against 150 frozen, verbatim family-law queries — 50 case-law (CL01–CL50), 50 Constitution
(C01–C50), 50 statute (S01–S50) — comparing dense-only, sparse-only, and production hybrid
retrieval, separately per corpus. It does not evaluate answer generation or legal reasoning.

## Scope

All work in this task is confined to this directory
(`evaluation/results/final_family_law_150_query_benchmark_20261019_212630/`). No production
code, existing benchmarks/qrels/results, source documents, embeddings, Qdrant data, or
environment/config was modified. See `integrity_confirmation.md` for the verification.

## Execution steps

See `reproducibility.md` for exact commands. In order: (1) read-only inspection of the
production retrieval stack and Qdrant config, (2) query validation against the indexed corpus
(`queries_150.jsonl`, `query_coverage.csv`), (3) corpus/index coverage audit
(`corpus_index_coverage.md`), (4) qrels construction (`qrels.jsonl`), (5) dense/sparse/hybrid
retrieval execution (`retrieval_rankings.jsonl`), (6) metrics (`metrics_per_query.csv`,
`metrics_summary.json`), (7) statistical/failure analysis (`comparison_report.md`,
`failure_diagnostics.csv`, `bootstrap_ci_results.json`).

## Headline results

- **150/150 queries preserved verbatim.** 138 supported, 10 partially_supported, 2 unsupported
  (both Constitution — no direct constitutional text on cross-border child removal or foreign
  decree recognition).
- **148/150 queries carried through retrieval and metrics**, all three methods, no failures.
- **Constitution**: dense-only best (nDCG@10 0.879), statistically indistinguishable from hybrid
  (0.870); both significantly beat sparse.
- **Statute**: hybrid best on every metric (small margins), though the dense/hybrid gap is not
  statistically significant.
- **Case law**: **inconclusive** — qrels pooling (2–3 cases per query from one keyword-search
  pass over a 2,084-document corpus) is too incomplete to trust a method ranking; directly
  confirmed by spot-check (a genuinely relevant case for CL01 was retrieved at dense rank 1 but
  was outside the qrels pool). See `comparison_report.md` Section 6 before citing any case-law
  number.

## Key limitations (see `comparison_report.md` and `metrics_summary.json` for full detail)

1. Case-law P@5/nDCG@10 are not reliably computable under this benchmark's pooling depth and are
   reported as unavailable, not approximated.
2. Case-law Recall/MRR/HitRate numbers understate true retrieval performance and are valid only
   for comparing the three methods against each other on the same incomplete reference set.
3. 2 Constitution queries (C24, C38) are only partially answerable from constitutional text alone
   (legislative competence over marriage/divorce/inheritance is a general framework, Art.142 +
   Art.270AA, not a named enumeration) — flagged rather than forced to "supported."
4. A reranker experiment, fusion-weight change, or GraphRAG modification was explicitly out of
   scope for this task and was not performed.

## Not performed (explicitly out of scope)

Reranking, RRF weight changes, GraphRAG modification, embedding/index regeneration, repair of
any discrepancy found during the coverage audit.
