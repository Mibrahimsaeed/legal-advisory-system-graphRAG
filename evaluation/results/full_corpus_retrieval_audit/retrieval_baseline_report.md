# Retrieval Baseline Report — All Three Corpora

**Configuration (verified from `src/rag_prep/embed_index.py`, unchanged for this run):** `candidates=40`, `k=10` per source, `hybrid_search()` with built-in Qdrant RRF, each source always queried with an explicit `source_type` filter (confirmed by direct code reading — **production never produces a single unfiltered cross-corpus ranking**; "combined-corpus retrieval" in this project means concatenating 3 independently-filtered searches, not true cross-corpus competition).

## Macro metrics by corpus

| Corpus | n | P@5 | nDCG@10 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| Statutes | 16 | 0.275 | **0.896** | 0.896 | **0.969** | 0.922 |
| Constitution | 6 | 0.167 | 0.772 | 0.833 | 0.833 | 0.750 |
| Case law | 6 | 0.167 | 0.670 | 0.833 | 0.833 | 0.617 |

**Statutes score highest on every metric.** Constitution and case law score lower on this pass, but — critically — **2 of the 12 Constitution/case-law queries (C4, CL1) scored a flat 0.0 across every metric**, and Phase 4 found both are very likely **qrels-granularity artifacts** (the production system found closely-related, independently-verifiable-as-relevant content just outside this benchmark's single-chunk target), not confirmed retrieval failures. Recomputing the Constitution/case-law macro excluding these two diagnosed-as-ambiguous queries:

| Corpus (excl. C4/CL1) | n | P@5 | nDCG@10 | Recall@10 |
|---|---:|---:|---:|---:|
| Constitution | 5 | 0.2 | 0.926 | 1.0 |
| Case law | 5 | 0.2 | 0.803 | 1.0 |

With those two removed, Constitution and case law look much closer to the statute benchmark's quality — **this is presented as a sensitivity check, not a replacement headline number**, since excluding queries post-hoc is exactly the kind of score-fitting the task warned against. Both the 6-query and 5-query figures are reported; neither is hidden.

## P@5 vs. nDCG@10/Recall@10 — read jointly, not in isolation

As in the statute-only work, **P@5 is structurally low (0.167-0.275) across all three corpora** because most queries here have a 1-3 chunk qrels set by design (narrow legal lookups rarely have 5 equally relevant passages) — P@5 always divides by 5 regardless. nDCG@10 and Recall@10 are the more informative metrics for this benchmark's query style, exactly as previously established for statutes and now confirmed holding for Constitution and case law too.

## Individual-corpus vs. combined-corpus retrieval

Per the confirmed code-reading above, **the production system has no code path for true combined-corpus ranking**. An isolated, explicitly non-production-equivalent diagnostic (dropping the `source_type` filter entirely) was run for 2 representative queries to test what *would* happen if cross-corpus competition existed:

| Query | Filtered (own-corpus) target rank | Unfiltered (cross-corpus) target rank | Top unfiltered competitors |
|---|---:|---:|---|
| Q06 (statute) | 3 (dense) | **8** | A case-law chunk (`ee317a0759f5eb10a6a0caf8:0003`) and the Dowry Act's own preamble/§7 outrank the target when case law is allowed to compete |
| CL1 (case) | 25 (dense, case-filtered) | **not in top 20** | Other case-law chunks dominate; statute/constitution chunks did not appear in the unfiltered top 5 for this query |

**This confirms per-source filtering is load-bearing**: removing it would measurably hurt Q06 by letting case-law content compete. This is informative but **not a production finding** — it was never going to happen in practice, since `hybrid_search()` always filters. It is included because the task explicitly asked whether cross-corpus competition exists, and the honest answer is "no, by design, and here is what would happen if it didn't."
