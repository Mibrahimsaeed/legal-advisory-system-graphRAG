# Known Failure Cases and Retrieval Weaknesses (from saved artifacts only)

## 1. Confirmed RRF fusion demotion (CL11, CL18)

**Source: `cl11_cl18_fusion_experiment/cl11_cl18_diagnosis.md`, `fusion_comparison.md`,
`fusion_failure_diagnostics.csv`.**

Both CL11 and CL18 have targets that are present in BOTH the dense and sparse branches
(dense rank 10/6, sparse rank 37/25 at depth 40) — i.e. the candidate generation step
succeeds for both branches. Production hybrid fusion (RRF, empirically validated constant
k=1) demotes these targets to rank 17/10 respectively — reproduced exactly by an offline RRF
recomputation. This is a **confirmed, reproducible fusion-mechanism weakness**: a weak-but-
present sparse rank dilutes a strong dense rank when fused.

**The counter-example, also documented in the same experiment:** CL01's target is entirely
absent from dense and found only via a strong sparse rank (1st). Reducing sparse's fusion
weight to address CL11/CL18 caused CL01's nDCG@10 to drop from 1.000 to 0.500
(`fusion_comparison.md` §"Regressions found"). **This is a genuine, quantified trade-off,
not a free win** — the same experiment's `final_recommendation.md` explicitly recommends
against a global sparse-weight reduction and suggests a reranker or query-aware fusion
instead, which was tested in a later task (see §4 below).

## 2. Case-law queries where ranking quality cannot be reliably assessed due to qrels depth

**Source: `case_law_qrels_expansion_20261019_220106/pool_completeness.csv`,
`comparison_report.md` §6.**

1,506 of 2,207 unique pooled case-law candidates (68%, summed across all 50 queries) remain
unjudged beyond the top-10-per-original-method depth. The highest-unjudged-fraction queries
include CL49 (91.7% of its pool unjudged), CL31 (86.0%), CL30 (75.0%), CL03 (74.2%), CL05
(73.8%) — these specific queries' metrics should be treated with the most caution of all 50.

## 3. Known-relevant cases entirely missed by all three retrieval methods

**Source: `case_law_qrels_expansion_20261019_220106/build_ledger_and_qrels.py` output /
`comparison_report.md` §1.** 80 of the original 125 case-law relevant judgments (from the
pre-expansion qrels) are **not retrieved by dense, sparse, OR hybrid within the top 40** —
a genuine corpus-breadth/candidate-generation ceiling on Recall@10 (and even Recall@40) that
no amount of reranking or fusion-weight tuning can fix, since the candidates are never
generated in the first place at this depth.

## 4. Reranking the hybrid baseline does not reliably help, and measurably hurts Recall@10

**Source: `case_law_reranker_experiment_20261019_224929/comparison_report.md`.**

Reordering hybrid's existing top-40 case-law candidates with `BAAI/bge-reranker-v2-m3`
produced a statistically significant **Recall@10 regression** (−0.108, 95% CI
[−0.138, −0.080], n=50; 39 of 50 queries worsened). A nominally significant nDCG@10
"improvement" was only evaluable on 7 of 50 queries (the other 43 fell outside the
pre-existing qrels' judged depth once reranking reshuffled candidates) and that 7-query
subset was shown to be non-representative (its baseline nDCG@10, 0.812, was well below the
full-50 baseline average, 0.900). **This experiment's own recommendation was "NOT
PROMISING — do not integrate."** It is cited here as evidence that the hybrid baseline's
candidate SET is not obviously the bottleneck for case law (a reranker working on the exact
same candidates did not reliably help), which is relevant context for whether to invest
further in retrieval tuning before GraphRAG — see `final_recommendation.md`.

## 5. Constitution: legislative-competence and cross-border-family-law questions are not fully answerable from constitutional text alone

**Source: `final_family_law_150_query_benchmark_20261019_212630/queries_150.jsonl` (notes
on C24, C34, C38, C40).** Two of 50 Constitution queries (C34, C40) are classified
`unsupported` — no Constitution-corpus text addresses cross-border child removal or foreign
custody/divorce-decree recognition — and two more (C24, C38) are `partially_supported` —
legislative competence over marriage/divorce/inheritance is answerable only via a general
framework (Art.142 + Art.270AA), not a named enumeration. These are corpus-coverage
limitations, not retrieval-ranking failures, and would not be fixed by changing dense,
sparse, or hybrid retrieval — they would require cross-corpus (Constitution+statute+case
law) synthesis at answer-generation time, which is explicitly out of scope for a retrieval
audit but is relevant to deciding what GraphRAG should actually be evaluated for.

## 6. Branch disagreement is common and corpus-dependent

**Source: `final_family_law_150_query_benchmark_20261019_212630/failure_diagnostics.csv`.**
That file records per-query cases where one retrieval method finds the relevant
evidence/chunk in top-10 while another does not, across all three corpora — 3 such
disagreements for Constitution, 4 for statute, and 13 for case law (alongside 2, 2, and 32
"total miss, all three methods" rows respectively for the three corpora in that same file).
The much higher case-law disagreement/miss count is consistent with the qrels-depth and
pooling limitations already documented above, not necessarily a larger true failure rate.
