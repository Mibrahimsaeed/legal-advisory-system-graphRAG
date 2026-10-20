# Final Recommendation

## 1. Is hybrid retrieval demonstrably the best method, per corpus?

| Corpus | Is hybrid demonstrably best? |
|---|---|
| Constitution | **No.** Dense is numerically ahead and the Hybrid–Dense gap is not statistically significant; both significantly beat Sparse. Dense is at least as defensible a default as Hybrid here. |
| Statutes | **No, not established.** Hybrid is numerically best on every metric, but the margin over Dense is tiny and not statistically significant. |
| Case Law | **No.** Hybrid and Dense are statistically indistinguishable on every ranking-quality metric tested (nDCG@10, MRR@10, P@5, Recall@10). Dense significantly beats Sparse on Recall@10; Hybrid inherits that advantage over Sparse but adds nothing significant beyond what Dense already provides. |

**There is no corpus in this evidence base where hybrid is both the numerically best method
AND has a statistically significant margin over the single best alternative.** Statutes come
closest (hybrid leads on every point estimate), but the gap over dense is not significant.

## 2. Does the evidence support claiming hybrid is universally best?

**No.** The evidence instead supports a more precise, corpus-specific picture:
- Dense is the stronger (if statistically tied) choice for Constitution.
- Hybrid and Dense are essentially interchangeable for Statutes and Case Law, with Sparse
  reliably the weakest branch specifically on Case Law Recall.
- Sparse alone is never the best choice for any corpus on any primary metric in this
  evidence base, though it is not negligible — it is the mechanism that uniquely finds some
  individual cases (e.g. CL01 in the fusion experiment) that dense alone misses.

A blanket "hybrid is best" claim would not survive scrutiny against this evidence; a
corpus-aware statement ("dense and hybrid are statistically tied and both clearly outperform
sparse-alone; hybrid's practical value is in rescuing sparse-only-findable cases rather than
in beating dense on aggregate ranking quality") is the claim the evidence actually supports.

## 3. The most important remaining retrieval weakness

**Case-law qrels depth, not a specific retrieval defect.** The single most consequential
limitation across this entire evidence base is that case-law relevance judgments are only
pooled and judged to a depth of 10 per original method — 68% of the candidate pool at depth
40 remains unjudged, and 80 known-relevant cases are never retrieved by any method at any
depth. This limitation:
- Prevented this audit (and the qrels-expansion and reranker-experiment tasks before it)
  from fully resolving whether hybrid's case-law advantage over dense is real or noise.
- Directly caused the reranker experiment's primary metric (nDCG@10) to be evaluable on only
  7 of 50 queries.
- Is the most likely explanation for why two independent smaller benchmarks
  (`constitution_caselaw_40q`, and the original 150-query pass before qrels expansion) each
  numerically favored a DIFFERENT method (hybrid; sparse, respectively) that did not survive
  deeper judging.

The confirmed RRF fusion-demotion mechanism (CL11/CL18, §1 of `known_failure_cases.md`) is a
real, secondary finding, but it is a narrow, well-characterized, already-quantified
trade-off (fixing it via sparse-weight reduction breaks CL01) — not an open question the way
the case-law qrels gap is.

## 4. Should the project proceed to GraphRAG without further retrieval tuning?

**Yes, proceed to GraphRAG evaluation — but do not treat the retrieval-method question as
settled, and do not expect GraphRAG to silently fix it.**

Reasoning:
- Constitution and Statute retrieval are both well-evidenced (n≈48–50, structurally
  exhaustive grading, significant tests run) and already perform strongly (nDCG@10 ≈
  0.83–0.88 across methods); there is no large, actionable retrieval defect here that would
  block a GraphRAG evaluation from being meaningful.
- Case-law retrieval's open question (dense vs. hybrid) is a **statistical tie, not a
  failure** — both methods already achieve nDCG@10 ≈ 0.89–0.90 and perfect/near-perfect
  HitRate@5/10. The qrels-depth gap limits confidence in a method RANKING, but does not
  indicate current retrieval is inadequate for downstream use.
- The reranker experiment already tested the most obvious "quick fix" for case law (reorder
  the existing hybrid candidates) and found it **does not help and measurably hurts
  Recall@10** — so further low-effort retrieval tuning has a documented recent negative
  result to account for, rather than being an untested, obviously-promising next step.
- Deepening the case-law qrels (to resolve the one real open question) is itself a
  non-trivial manual-judging task, not a quick retrieval change — better scheduled as its
  own follow-up than as a blocker before GraphRAG.

## 5. Why GraphRAG should be evaluated as a separate system component, not assumed to improve retrieval automatically

GraphRAG (graph-structured retrieval/reasoning over entities and relationships, e.g. via
Neo4j in this project) answers a **different question** than dense/sparse/hybrid chunk
retrieval: it is designed to support multi-hop reasoning and structured relationship
traversal (e.g., "which statute amends which provision," "which case cites which statute
section"), not to re-rank or re-retrieve the same unstructured text chunks more accurately.
Nothing in this audit's evidence base measures GraphRAG's actual behavior — no GraphRAG
component has been evaluated in any artifact inspected here. It would be a non-sequitur to
infer GraphRAG's quality from chunk-retrieval metrics: a system could have excellent
dense/hybrid chunk retrieval and still have a GraphRAG layer that adds no value (if the
underlying graph is incomplete, the relationships it captures aren't the ones users actually
need, or multi-hop traversal introduces its own new error modes), or conversely a middling
chunk-retrieval setup could still benefit substantially from GraphRAG if the project's real
bottleneck is answering compositional/multi-hop questions that chunk retrieval structurally
cannot address (e.g., the Constitution's legislative-competence gap documented in
`known_failure_cases.md` §5, which requires cross-corpus synthesis, exactly the kind of
question a knowledge graph is built for). GraphRAG therefore needs its OWN benchmark, its
own qrels, and its own dense/sparse/hybrid-style baseline comparison before any claim can be
made about whether it helps — exactly the rigor this series of audits has applied to chunk
retrieval.

## Summary table

| Corpus | Decision |
|---|---|
| Constitution | Another method (Dense) performs better on the available evidence |
| Statutes | Hybrid numerically best, but not statistically established |
| Case Law | Methods statistically tied (Dense ≈ Hybrid; Sparse trails on Recall) |

**Overall path: Option 1 — keep the existing hybrid retrieval baseline and proceed to
GraphRAG evaluation**, while explicitly recording (not fixing) the case-law qrels-depth gap
as a documented follow-up item, not a blocker.
