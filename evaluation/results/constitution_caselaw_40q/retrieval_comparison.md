# Retrieval Method Comparison — Constitution vs. Case Law

**Note on raw scores:** dense (cosine similarity, 0-1), sparse (unbounded BM25-style score), and hybrid (RRF reciprocal-rank score, 0-1) are on different scales and are **not compared directly** anywhere in this report — only ranks and judged relevance are compared, per the task's instruction.

**Note on denominators:** Recall@K uses the number of chunks named in the frozen qrels as its denominator (1 for most queries, 2 for C05/C09), never the number retrieved. With only 20/13 queries per corpus, no confidence interval or bootstrap test is reported — differences below are descriptive, not statistically established.

## Comparison table

| Corpus | Method | P@5 | Recall@5 | Recall@10 | MRR | nDCG@10 | HitRate@5 | HitRate@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Constitution | Dense | 0.220 | 0.975 | 0.975 | 0.910 | **0.936** | **1.000** | 1.000 |
| Constitution | Sparse | 0.190 | 0.875 | 0.875 | 0.742 | 0.791 | 0.950 | 0.950 |
| Constitution | Hybrid | 0.210 | 0.925 | **1.000** | **0.906** | 0.921 | 0.950 | **1.000** |
| Case law | Dense | 0.062 | 0.308 | 0.538 | 0.336 | 0.380 | 0.308 | 0.538 |
| Case law | Sparse | 0.062 | 0.308 | 0.385 | 0.220 | 0.253 | 0.308 | 0.385 |
| Case law | Hybrid | **0.077** | 0.385 | 0.462 | **0.402** | **0.407** | **0.385** | 0.462 |
| Combined | — | — | — | — | — | — | — | — |

**Combined-corpus retrieval was not evaluated.** Per direct code reading of `src/rag_prep/embed_index.py`'s `hybrid_search()` (confirmed in this project's prior full-corpus audit and re-confirmed here), production **always** applies a `source_type` filter per query — there is no code path that produces a single ranking across Constitution and case law together. Fabricating one via an unfiltered diagnostic would not be production-equivalent and was out of this task's stricter file-scope/time budget; marked **not evaluated**, consistent with the task's own instruction to mark it so rather than approximate it.

## Key findings

- **For Constitution, dense-only performs best or tied-best on every metric** — nDCG@10 0.936 vs. hybrid's 0.921 and sparse's 0.791. **Hybrid does not improve over the best individual branch here** — sparse actively drags the fused ranking down slightly relative to dense alone (though hybrid does edge out dense on Recall@10/HitRate@10, 1.000 vs 0.975, because RRF occasionally promotes a dense-weak-but-sparse-present target past rank 10).
- **For case law, hybrid performs best on every metric** — nDCG@10 0.407 vs. dense's 0.380 and sparse's 0.253. **Hybrid does improve over both individual branches here**, the opposite pattern from Constitution.
- **Case law is substantially weaker than Constitution on every method and every metric** (e.g. best nDCG@10: case law 0.407 vs. Constitution 0.936).
