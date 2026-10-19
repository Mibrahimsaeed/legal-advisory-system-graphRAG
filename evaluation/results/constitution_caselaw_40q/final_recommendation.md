# Final Recommendation

**Which method performs best for Constitution?** Dense-only (nDCG@10 0.936, HitRate@5 1.000) — hybrid is close (0.921) but does not beat it.

**Which method performs best for case law?** Hybrid (nDCG@10 0.407) — beats both dense (0.380) and sparse (0.253) alone.

**Does hybrid improve over both individual branches?** **No, not uniformly.** For Constitution, hybrid underperforms dense alone. For case law, hybrid beats both branches. This asymmetry is itself the most actionable finding in this audit — it suggests the sparse branch's contribution to RRF fusion is net-negative for Constitution queries specifically, while being net-positive (via rescuing dense-weak targets, e.g. CL01) for case law.

**Are failures predominantly candidate-generation or ranking failures?** Mixed, corpus-dependent:
- **Constitution**: almost entirely a ranking matter — every query's target is findable by at least one branch; only C12 shows a real multi-rank gap.
- **Case law**: a genuine mix. CL01 is a clean candidate-generation failure in dense, rescued by fusion. CL04 is weak in both branches, with no rescue. CL11 and CL18 show fusion *actively harming* a dense-findable target because sparse found nothing. CL12/CL14 look like qrels-granularity artifacts (a different, equally-valid case/chunk ranks well). CL15/CL16/CL17 are unresolved — absent from every method's top 20, with no same-document near-miss found in the time available.

**Are there meaningful differences between the two corpora?** Yes, substantial: Constitution's best nDCG@10 (0.936) is more than double case law's best (0.407). Constitution is a single, internally-consistent, recently-amended legal text; this case-law corpus is a large (2084-document), heterogeneous collection of individual judgments, entirely within one legal domain (family law) but with far more lexical/topical diversity per query than a single constitutional article provides.

**Is there enough evidence to justify a reranker experiment?** **For case law, yes — this is the clearest candidate found in this audit.** CL04's target is weak in both branches (a reranker would need new signal, not just reordering); but CL11 and CL18 are cases where a *correct* target already reaches the candidate set via dense, and fusion actively demotes it — a reranker operating after fusion (or replacing RRF with a learned combiner) is a well-motivated next experiment specifically for the "fusion harms a dense-findable target" pattern. For Constitution, the evidence points the other way: dense alone is already near-ceiling, and the gap is sparse dragging hybrid down — the fix there is more likely in how much weight sparse receives in fusion, not a reranker.

**Is query understanding a likely next step?** Not strongly supported by this pass's evidence — no query in either corpus showed a clear "right vocabulary, wrong article/case" pattern analogous to the earlier statute-corpus preamble/title-competition finding. This was not exhaustively re-tested here, so it remains a lower-confidence "not yet justified" rather than a confirmed "no."

**Is there concrete evidence that embeddings or chunking need changes?** No corpus-wide dense weakness was found for Constitution (dense is the best-performing method there). For case law, one instance (CL01) shows a genuine dense weakness fully compensated by sparse+fusion — a single instance, insufficient to implicate the embedding model generally, consistent with this project's standing rule not to over-generalize from one query.

**Is the current retrieval baseline sufficient to proceed with GraphRAG construction?** **Yes for Constitution** (near-ceiling dense performance); **the case-law baseline is weaker and the known integrity gaps (4 missing docs, 3 duplicate pairs) remain unaddressed**, but per the project's own stated principle, graph construction has different requirements than graph-enhanced-retrieval validation and does not need to wait for perfect retrieval.

**Single recommended next action:** **Investigate why RRF fusion demotes dense-findable targets when the sparse branch returns nothing relevant (CL11, CL18 pattern), and separately assess whether reducing sparse's weight in Constitution fusion would close its small gap to dense-only** — both are narrow, well-evidenced, low-risk next experiments directly motivated by this audit's own data, ahead of a general reranker rollout or further corpus-scale query construction.
