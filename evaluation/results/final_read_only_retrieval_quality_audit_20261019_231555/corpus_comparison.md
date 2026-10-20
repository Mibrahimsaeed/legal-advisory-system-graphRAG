# Corpus-by-Corpus Comparison

All numbers in this section are **quoted verbatim from existing artifacts** (file and
section cited for each), or explicitly marked "derived" where a simple arithmetic
comparison of already-reported numbers was performed. Nothing here was recomputed from raw
rankings or qrels.

## A. Constitution

**Source: `final_family_law_150_query_benchmark_20261019_212630/metrics_summary.json` and
`bootstrap_ci_results.json`, n=48 (of 50; 2 queries — C34, C40 — structurally unsupported by
the Constitution corpus and excluded from scoring, per that benchmark's own
`query_coverage.csv`).**

| Method | P@5 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | HitRate@5 | HitRate@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense  | 0.254 | 0.710 | 0.736 | 0.865 | **0.879** | 0.938 | 0.958 |
| Sparse | 0.238 | 0.697 | 0.709 | 0.795 | 0.821 | 0.896 | 0.896 |
| Hybrid | 0.246 | 0.710 | 0.740 | **0.871** | 0.870 | 0.917 | 0.938 |

**Reported paired bootstrap (seed=42, n_resamples=5000, n=48):**
- Hybrid − Dense, nDCG@10: diff = −0.0092, 95% CI [−0.0446, 0.0264] → **not significant**.
- Hybrid − Sparse, nDCG@10: diff = +0.0483, 95% CI [0.0138, 0.0891] → **significant**.

**Conclusion (reproducing that benchmark's own stated conclusion, not reinterpreted here):**
Hybrid does **not** demonstrably beat dense on Constitution — the difference is not
statistically distinguishable from zero, and dense has the higher point estimate on nDCG@10
and HitRate. Both dense and hybrid clearly and significantly beat sparse. **Hybrid is
statistically tied with dense, not demonstrably best, for Constitution.**

Cross-check against `constitution_caselaw_40q/retrieval_comparison.md` (n=20, earlier,
smaller benchmark): reported dense nDCG@10 = 0.936 > hybrid 0.921 > sparse 0.791 — **the
same ordering** (dense ≥ hybrid > sparse) as the 150-query benchmark. This is a point of
**consistency** between the two independent benchmarks, strengthening confidence in the
Constitution conclusion specifically.

## B. Statutes

**Source: `final_family_law_150_query_benchmark_20261019_212630/metrics_summary.json` and
`bootstrap_ci_results.json`, n=50 (all 50 queries supported and scored).**

| Method | P@5 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | HitRate@5 | HitRate@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense  | 0.284 | 0.877 | 0.933 | 0.779 | 0.832 | 0.920 | 0.940 |
| Sparse | 0.252 | 0.803 | 0.893 | 0.638 | 0.701 | 0.860 | 0.900 |
| Hybrid | **0.288** | **0.897** | **0.940** | **0.791** | **0.833** | 0.920 | 0.940 |

**Reported paired bootstrap (seed=42, n_resamples=5000, n=50):**
- Hybrid − Dense, nDCG@10: diff = +0.0011, 95% CI [−0.0328, 0.0359] → **not significant**.
- Hybrid − Sparse, nDCG@10: diff = +0.1324, 95% CI [0.0672, 0.1995] → **significant**.

**Conclusion:** Hybrid is numerically best on **every** reported metric for statutes, but the
margin over dense is tiny (+0.0011 nDCG@10) and **not statistically distinguishable from
zero**. Both hybrid and dense clearly and significantly beat sparse. **Hybrid is numerically
best but not statistically established as better than dense for statutes** — the highest
score and a statistically supported improvement are two different claims, and only the
former is true here.

## C. Case Law

**Primary evidence: `case_law_qrels_expansion_20261019_220106/metrics_summary.json` and
`bootstrap_ci_results.json`, n=50, document-level grading, 674 relevant case-document
judgments (125 carried over from the original benchmark + 656 newly judged in the
qrels-expansion task).** Per that task's own explicit instruction, this evidence
**supersedes** the original 150-query benchmark's case-law numbers, which are preserved
below only as historical context.

| Method | P@5 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | HitRate@5 | HitRate@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense  | 0.904 | 0.297 | 0.546 | 0.946 | 0.890 | 0.960 | 0.980 |
| Sparse | 0.884 | 0.296 | 0.480 | 0.957 | 0.893 | 1.000 | 1.000 |
| Hybrid | 0.900 | 0.292 | 0.540 | **0.970** | **0.900** | 1.000 | 1.000 |

**Reported paired bootstrap (seed=42, n_resamples=5000, n=50):**
- Hybrid − Dense, nDCG@10: diff = +0.0098, 95% CI [−0.0234, 0.0492] → **not significant**.
- Hybrid − Sparse, nDCG@10: diff = +0.0074, 95% CI [−0.0218, 0.0372] → **not significant**.
- Hybrid − Dense, Recall@10: diff = −0.0061, 95% CI [−0.0437, 0.0340] → **not significant**.
- Hybrid − Sparse, Recall@10: diff = +0.0608, 95% CI [0.0253, 0.0962] → **significant**.
- Sparse − Dense, Recall@10: diff = −0.0669, 95% CI [−0.1235, −0.0102] → **significant**
  (i.e. dense significantly beats sparse on Recall@10).
- Hybrid vs. Dense/Sparse on MRR@10, P@5: **not significant** in either direction.

**Conclusion:** Hybrid is numerically highest on MRR@10 and nDCG@10, and HitRate is already
saturated at 1.0 for both sparse and hybrid. **None of hybrid's apparent advantages over
dense are statistically significant.** The one robust, well-powered, significant finding is
that **dense significantly beats sparse on Recall@10** — hybrid inherits this advantage over
sparse (also significant) but is not distinguishable from dense itself. **Hybrid and dense
are statistically tied for case law; sparse trails both on recall specifically.**

### Essential caveats carried forward from the qrels-expansion task (not reinterpreted, quoted)

- Qrels are **pooled but incomplete**: only the top-10 of each ORIGINAL method's own ranking
  was judged (656 new + 125 original = 781 judgments out of 2,207 unique pooled cases at
  depth 40 — 68% remain unjudged beyond rank 10).
- **80 known-relevant cases are not retrieved by ANY of the three methods within top-40** —
  a genuine recall ceiling baked into these Recall@10 numbers, not an artifact to be
  corrected away.
- Multiple chunks per case are handled via document-level grade propagation (a judged case's
  grade applies to every chunk of that case encountered in a ranking) — this was a
  deliberate methodological fix made in the qrels-expansion task, documented there, and is
  why the original 150-query benchmark's case-law P@5/nDCG@10 (reported as "unavailable")
  differ structurally from the qrels-expansion numbers above (now fully computable).

### Historical context (original 150-query benchmark's case-law numbers, superseded)

`final_family_law_150_query_benchmark_20261019_212630/metrics_summary.json` reported, using
only the original 2–3-case-per-query pool: P@5/nDCG@10 unavailable for all methods;
doc-Recall@10 dense=0.097, sparse=0.187, hybrid=0.130 — on that smaller pool, **sparse
numerically led**. That benchmark's own `comparison_report.md` explicitly warned this was
likely a pooling artifact (confirmed directly by a spot-check showing a genuinely relevant
case at dense rank 1 that was simply outside the 2–3-case pool) and recommended NOT trusting
a method ranking from it. The qrels-expansion task's much larger judgment set reverses the
numerical ordering (dense/hybrid now ahead of sparse on Recall@10) — **this is documented as
an explicit, resolved inconsistency**: the smaller-pool "sparse wins" signal did not survive
deeper judging and should not be cited going forward.

Cross-check against `constitution_caselaw_40q/retrieval_comparison.md` (n=13 case-law
queries, earlier, smaller benchmark): reported hybrid nDCG@10 = 0.407 > dense 0.380 > sparse
0.253 — hybrid numerically ahead there too, but with no significance test performed and an
even smaller, less-exhaustively-pooled qrels set than the later tasks. **This is a second,
independent instance of a small-sample case-law benchmark numerically favoring one non-dense
method, which did not hold up under the larger, more carefully pooled qrels-expansion
evidence.** Documented as a pattern worth noting, not reconciled or treated as resolved in
one direction.
