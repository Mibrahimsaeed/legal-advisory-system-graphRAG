# Comparison Report — 150 Family-Law Query Benchmark

## 1. Query coverage (all 150, preserved verbatim)

| Corpus | supported | partially_supported | unsupported | Total |
|---|---:|---:|---:|---:|
| Constitution (C01–C50) | 45 | 3 | 2 | 50 |
| Case law (CL01–CL50) | 47 | 3 | 0 | 50 |
| Statute (S01–S50) | 46 | 4 | 0 | 50 |
| **All 150** | **138** | **10** | **2** | **150** |

- **Zero queries were rewritten, replaced, or removed.** All 150 IDs and original texts are
  preserved exactly in `queries_150.jsonl`.
- The 2 `unsupported` Constitution queries (C34 — cross-border child removal; C40 — recognition
  of foreign custody/divorce decisions) have **no direct constitutional text** addressing them in
  the indexed Constitution corpus. C34 is answerable at the **statute** level instead (Family
  Courts Act, 1964, Schedule Part I, item 6A, which names the Hague Convention on Child
  Abduction explicitly) — documented as a cross-reference, not invented as constitutional text.
- 10 `partially_supported` queries (C24, C37, C38 in Constitution; S02, S04, S05, S17 in statute;
  CL29, CL37, CL45 in case law) have real but incomplete or indirect evidence — see
  `query_coverage.csv` / `queries_150.jsonl` notes for the specific reasoning per query.
- **148 of 150 queries received a relevance judgment** (qrels.jsonl) and were carried through
  retrieval and metrics. The 2 unsupported Constitution queries were excluded from retrieval
  entirely (nothing to judge against).

## 2. Retrieval execution

All three methods — dense-only (top 40), sparse-only (top 40), and the unmodified production
hybrid (`hybrid_search()`, RRF fusion, `candidates=40`) — completed successfully for all 148
eligible queries across all three corpora, with corpora queried and scored strictly separately
throughout (no combined cross-corpus ranking exists in production — confirmed by direct code
reading of `embed_index.py`, consistent with this project's prior audits). No method failed or
required a substitute implementation.

## 3. Metrics — Constitution (n=48)

| Method | P@5 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | HitRate@5 | HitRate@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense  | 0.254 | 0.710 | 0.736 | 0.865 | **0.879** | 0.938 | 0.958 |
| Sparse | 0.238 | 0.697 | 0.709 | 0.795 | 0.821 | 0.896 | 0.896 |
| Hybrid | 0.246 | 0.710 | 0.740 | **0.871** | 0.870 | 0.917 | 0.938 |

P@5/nDCG@10 are **fully computable** here (no missing labels) — relevance is structurally
determined by article-number identity (verified by direct text reading), so every retrieved
chunk could be graded exactly, not approximated.

**Bootstrap (paired, seed=42, 5,000 resamples):** hybrid vs. dense nDCG@10 difference is **not
significant** (diff −0.009, 95% CI [−0.045, 0.026]); hybrid vs. sparse **is significant** (diff
+0.048, 95% CI [0.014, 0.089]). Dense and hybrid are statistically indistinguishable on nDCG@10;
both clearly beat sparse.

**Total misses (HitRate@10 = 0, all three methods):** C44, C46 (2 of 48). **Genuine ranking
failures** (not corpus absence — see Section 5).

## 4. Metrics — Statute (n=50)

| Method | P@5 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | HitRate@5 | HitRate@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense  | 0.284 | 0.877 | 0.933 | 0.779 | 0.832 | 0.920 | 0.940 |
| Sparse | 0.252 | 0.803 | 0.893 | 0.638 | 0.701 | 0.860 | 0.900 |
| Hybrid | **0.288** | **0.897** | **0.940** | **0.791** | **0.833** | 0.920 | 0.940 |

Also fully computable (section-identity grading). **Hybrid is the best or tied-best method on
every metric in this corpus**, though mostly by a small margin over dense.

**Bootstrap:** hybrid vs. dense nDCG@10 **not significant** (diff +0.001, 95% CI [−0.033,
0.036]); hybrid vs. sparse **significant** (diff +0.132, 95% CI [0.067, 0.200]).

**Total misses (all three methods):** S25 (option of puberty), S41 (guardian's property duties)
(2 of 50) — genuine ranking failures.

## 5. Metrics — Case law (n=50) — **read with the pooling caveat below before drawing conclusions**

| Method | Recall@5 (chunk) | Recall@10 (chunk) | MRR@10 (chunk) | Doc-Recall@5 | Doc-Recall@10 | Doc-MRR@10 | Doc-HitRate@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense  | 0.023 | 0.040 | 0.046 | 0.073 | 0.097 | 0.093 | 0.18 |
| Sparse | 0.037 | 0.090 | 0.084 | 0.087 | **0.187** | **0.118** | **0.30** |
| Hybrid | 0.043 | 0.063 | 0.067 | 0.080 | 0.130 | 0.104 | 0.22 |

**P@5 and nDCG@10 are NOT reliably computable** for case law in this benchmark and are reported
as unavailable for all 50×3 method/query combinations — see Section 6.

**Bootstrap:** sparse vs. dense Doc-Recall@10 **is significant** (diff +0.090, 95% CI [0.013,
0.173]) — sparse outperforms dense on this specific, small, incomplete reference set. Hybrid vs.
dense and hybrid vs. sparse are **both not significant** on Doc-Recall@10 or Doc-MRR@10. Given
the pooling caveat below, this should be read as "distinguishable on this sample" rather than
"sparse is the better case-law method in general."

## 6. The central case-law caveat: incomplete qrels pooling, confirmed by spot-check

Case-law qrels were built from 2–3 case citations per query, found via one targeted
keyword-search pass per query over a 2,084-document, 25,891-chunk, **100% family-law** corpus
(confirmed via the corpus's own `primary_domain` metadata). This is a **small, non-exhaustive
pool** relative to the corpus's real topical density — many family-law topics (khula, dower,
custody, maintenance) are litigated in hundreds of reported judgments, not 2–3.

**Direct evidence this understates true performance:** for CL01 ("khula without husband's
consent"), dense-only retrieval's rank-1 result is case `pld-2024-fsc-9`, headnoted "Dissolution
of marriage on the basis of 'Khula' — Principles relating to the right of Khula stated" — an
unambiguously on-point, genuinely relevant judgment. It was **not** in this benchmark's 3-case
qrels pool for CL01, so it currently scores as "not relevant" under the strict qrels-based
metrics above. **This is a benchmark coverage gap, not a retrieval failure.**

**Conclusion for case law:** the Recall/MRR/HitRate figures above are valid ONLY for relative
comparison between the three methods on this benchmark's specific (incomplete) reference set —
they should NOT be read as an absolute measure of case-law retrieval quality, and the 32/50
"total miss" rate in `failure_diagnostics.csv` for case law overwhelmingly reflects pooling
depth, not a 64% real failure rate. A reliable case-law P@5/nDCG@10/Recall figure would require
either (a) exhaustive exact-topic case research per query (out of scope/time for 50 queries at
this depth) or (b) pooling the top-10 of all three methods and judging every unique candidate —
recommended as the concrete next step before trusting any case-law method ranking numerically.

## 7. Best method per corpus

| Corpus | Best method | Evidence |
|---|---|---|
| Constitution | **Dense** (nDCG@10, HitRate), statistically indistinguishable from hybrid | Hybrid does not beat dense; both clearly beat sparse (significant) |
| Statute | **Hybrid** (every metric, small margins) | Hybrid vs. dense difference not statistically significant on nDCG@10 despite higher point estimate |
| Case law | **Inconclusive** — do not select a method from this pass | Qrels pooling too incomplete for a trustworthy method ranking; sparse's edge over dense is significant on this sample but may not generalize |

This reproduces the Constitution/case-law asymmetry found in this project's prior 33-query and
40-query audits (dense-only best for Constitution; hybrid's advantage specific to case law was
previously observed but is **not confirmed** here once case-law judgments are scrutinized at
this larger scale — the honest read is "unresolved," not "hybrid wins."

## 8. Recommendation

- **Do not change RRF weights or add a reranker based on this pass** (explicitly out of scope
  for this task, and the case-law evidence base is too weak to justify it yet regardless).
- **Before any case-law method-selection decision**: build a properly pooled, exhaustively
  judged case-law qrels set (pool top-10 of all three methods per query, judge every unique
  candidate) — this benchmark's case-law numbers are not strong enough evidence on their own.
- **Constitution C44/C46 and Statute S25/S41** (and the other total-misses) are concrete,
  reliable (non-pooling-artifact) ranking failures worth root-causing before the next stage —
  see `failure_diagnostics.csv`.
- **C24/C38** (legislative competence) and **C34/C40** (cross-border family matters) reveal that
  a purely Constitution-corpus retrieval view cannot answer some real family-law questions —
  any downstream answer-generation stage will need cross-corpus synthesis (Constitution +
  statute + case law together) for these, which the current retrieval architecture does not
  support natively (confirmed: no combined-corpus ranking path exists in production).
