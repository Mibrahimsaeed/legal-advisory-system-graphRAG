# Comparison Report — Case-Law Qrels Expansion

## 1. Pooling and judging completeness

| | Total |
|---|---:|
| Queries processed | 50 / 50 (all CL01–CL50, verbatim, unchanged) |
| Unique pooled cases (top-40 union, 3 methods) | 2,207 |
| Known-relevant cases from the original benchmark NOT retrieved by any method in top-40 | 80 |
| Total judged (combined old + new) | 781 |
| — Highly relevant (grade 2) | 381 |
| — Partially relevant (grade 1) | 293 |
| — Not relevant (grade 0) | 107 |
| Unjudged (ranks 11–40, beyond this task's judging depth) | 1,506 |
| Pool structurally contributed by hybrid alone (not in dense or sparse top-40) | 0 (every one of 50 queries) |

The `pool_hybrid_only = 0` result across all 50 queries is a structural property, not a
coincidence: production hybrid's RRF fusion operates over the same dense/sparse prefetch
sets at the same depth, so it cannot surface a candidate neither branch already retrieved
at that depth. This confirms (again) that a reranker candidate set would need to come from
dense/sparse retrieval depth, not from hybrid fusion itself.

No query's pool was below 20 unique cases, so case-law breadth is not a limiting factor for
this benchmark as a whole (confirmed previously via the corpus's 100%-family-law metadata).

## 2. Was CL01's missing relevant case incorporated correctly?

**Yes.** `pld-2024-fsc-9` ("Dissolution of marriage on the basis of Khula — Principles... a
fundamental right of a woman... cannot be denied") is now in the ledger and qrels as
**grade 2**, found at dense rank 1, hybrid rank 1, and sparse rank 19 — exactly the case
flagged as missing during the original spot-check. See `judgment_ledger.jsonl` (query_id
`CL01`, case_id `pld-2024-fsc-9`).

## 3. Updated case-law metrics (document-level grading, all 50 queries, n=50 for every cell)

| Method | P@5 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | HitRate@5 | HitRate@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense  | 0.904 | 0.297 | 0.546 | 0.946 | 0.890 | 0.960 | 0.980 |
| Sparse | 0.884 | 0.296 | 0.480 | 0.957 | 0.893 | 1.000 | 1.000 |
| Hybrid | 0.900 | 0.292 | 0.540 | 0.970 | 0.900 | 1.000 | 1.000 |

**P@5 and nDCG@10 are now FULLY computable for all 50 queries and all three methods**
(denominator 50, zero missing) — a direct consequence of document-level grading plus
complete top-10 judging coverage, resolving the original benchmark's "unavailable" result
for these two metrics.

### How graded relevance is handled by each metric

- **P@5 / nDCG@10**: a chunk's relevance = its case's judged grade (0/1/2), propagated from
  case-level judgment to every chunk of that case. nDCG uses `2^grade − 1` gain with log2
  discount, exactly as in `calculate_metrics.py`'s existing pure function, reused unmodified.
- **Recall@5/10**: computed against the FULL known-relevant set (674 case-document pairs,
  including the 80 "known relevant, unretrieved" orphans) — this is the honest, conservative
  choice: including the orphans in the denominator means achievable Recall@10 cannot exceed
  the fraction of known-relevant cases that are even retrievable within 40 candidates. **If
  the relevant-set denominator is not fully known (and it is not — 68% of the pool remains
  unjudged beyond rank 10), these Recall figures remain an estimate, not a ceiling-exact
  value**; they are, however, no longer based on a 2–3-case sample and are far more
  trustworthy than the original benchmark's.
- **MRR@10 / HitRate@5/10**: membership-only metrics against the known-relevant set; reliable
  regardless of unjudged depth, since they only ask "is a known-relevant case present/first."
- **Unjudged chunks** (ranks beyond this task's top-10-per-method depth): never coerced to 0;
  where they would have fallen within a metric's ranking window, P@5/nDCG@10 would have been
  marked unavailable exactly as before — in this recomputation that never occurs because the
  judging depth (top-10 of each method) exactly matches the metric windows used.

## 4. Statistical comparison (paired bootstrap, seed=42, 5,000 resamples, n=50)

| Comparison | Metric | Diff | 95% CI | Significant? |
|---|---|---:|---|---|
| Hybrid − Dense | nDCG@10 | +0.010 | [−0.023, 0.049] | No |
| Hybrid − Sparse | nDCG@10 | +0.007 | [−0.022, 0.037] | No |
| Sparse − Dense | nDCG@10 | +0.002 | [−0.052, 0.068] | No |
| Hybrid − Dense | Recall@10 | −0.006 | [−0.044, 0.034] | No |
| Hybrid − Sparse | Recall@10 | **+0.061** | [0.025, 0.096] | **Yes** |
| Sparse − Dense | Recall@10 | **−0.067** | [−0.124, −0.010] | **Yes** |
| Hybrid − Dense | MRR@10 | +0.024 | [−0.030, 0.085] | No |
| Hybrid − Sparse | MRR@10 | +0.013 | [−0.020, 0.053] | No |
| Hybrid − Dense | P@5 | −0.004 | [−0.040, 0.036] | No |
| Hybrid − Sparse | P@5 | +0.016 | [−0.028, 0.060] | No |

**The three methods are statistically indistinguishable on nDCG@10, MRR@10, and P@5** at
this sample size (n=50). The one robust, significant finding: **dense significantly
outperforms sparse on Recall@10** (and by extension hybrid, which inherits dense's recall
advantage, significantly beats sparse too) — sparse alone misses known-relevant cases that
dense retrieval finds, consistent with semantic (dense) retrieval having a broader recall
net than lexical (BM25) retrieval for these fact-pattern-style legal questions.

## 5. Baseline vs. expanded — which changes reflect better judgments, not better retrieval

| Metric | Original (2–3 cases/query, doc-level) | Expanded (674 relevant cases, doc-level) | What changed |
|---|---:|---:|---|
| Dense Recall@10 | 0.097 | 0.546 | **Judgment completeness**, not retrieval. The underlying rankings are byte-identical (same frozen `retrieval_dense_case.jsonl`); only the relevant-set denominator and numerator grew. |
| Sparse Recall@10 | 0.187 | 0.480 | Same — judgment completeness. |
| Hybrid Recall@10 | 0.130 | 0.540 | Same — judgment completeness. |
| Dense/Sparse/Hybrid P@5, nDCG@10 | unavailable | 0.88–0.90 / 0.89–0.90 | **Judgment completeness** (now able to grade every top-5/top-10 candidate) AND a **methodological fix** (document-level propagation instead of exact-chunk-id matching, which previously undercounted "right case, different chunk" hits). |
| Method ranking (best corpus method) | sparse appeared best on doc-Recall@10 (0.187 vs hybrid 0.130, dense 0.097) | dense/hybrid now significantly beat sparse on Recall@10; all three indistinguishable elsewhere | **This is a genuine reversal driven by judgment completeness**, not retrieval change. The original "sparse wins" signal was an artifact of a 2–3-case pool that happened to overlap more with sparse's idiosyncratic hits; with a much larger, still-incomplete-but-far-deeper relevant set, dense's broader semantic recall shows through as the real, statistically significant advantage. |

No retrieval ranking changed between the baseline and this expansion — `retrieval_dense_case.jsonl`,
`retrieval_sparse_case.jsonl`, and `retrieval_hybrid_case.jsonl` were read, never rewritten
(checksums unchanged, see `integrity_confirmation.md`). Every number above moved because the
judgments improved, not because retrieval did.

## 6. Remaining pooling and recall limitations

- 1,506 of 2,207 pooled cases (68%) remain unjudged (ranks 11–40 of each method); Recall@40
  or any claim of "total relevant cases in the corpus" is NOT estimable from this task.
- 80 known-relevant cases are entirely unretrieved by all three methods within top-40 — a
  genuine recall ceiling below 100%, not a data artifact; this is now baked into the
  Recall@10 denominator rather than hidden.
- No manual judging beyond keyword-search-directed reading was performed; a handful of
  queries (e.g. CL49, with only one new candidate found at this pooling depth) have thinner
  support than the rest and should be read with extra caution if used individually.

## 7. Recommendation: is a reranker experiment now justified?

**Better-supported than before, but still not unconditionally.** The dense/sparse/hybrid
comparison on nDCG@10/MRR@10/P@5 is now statistically clean (indistinguishable, n=50,
properly pooled), and the one significant finding (dense > sparse on Recall@10) is a
genuine, not an artifact, result. This evidence base is now strong enough to:
- Justify ranking-quality experiments (reranking a correctly-retrieved candidate set) rather
  than candidate-generation experiments, IF the goal is improving nDCG@10/P@5 specifically —
  but since the three methods are statistically tied on those metrics already, a reranker's
  expected marginal gain on THIS benchmark is unclear and should be tested, not assumed.
- NOT yet justify a firm claim that hybrid (or any one method) is "best" for production —
  only that sparse is weaker specifically on recall. A reranker experiment is a reasonable
  next step, but should be evaluated against this expanded qrels baseline (not the original
  2–3-case one), and ideally after judging the remaining 68% unjudged pool depth if time
  permits, to also validate Recall@40 claims.
