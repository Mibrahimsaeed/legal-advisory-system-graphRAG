# Statute Retrieval Improvement — Phase 1: Benchmark Audit

**Scope:** statutes only; case-law and Constitution retrieval are out of scope and were not touched. **Method:** read-only inspection of every existing query file, retrieval output, judging CSV, and the Phase 3A reports — no retrieval was re-run, nothing was modified.

## 1. Query inventory

**20 query entries across 3 files, 18 unique texts, 5 query families (Q01–Q05).** Each family has an original (`queries.json`), two phrasing variants L1/L2 (`queries_expansion_experiment.json`), and one expansion variant (`queries_statute_expanded.json`).

The only two repeated texts are **intentional**, already-disclosed cases: Q02 and Q05 were deliberately left unexpanded in `queries_statute_expanded.json` because no safely-groundable synonym was found for either (see Phase 3A) — these are not accidental duplicates. **No accidental duplicate query text was found anywhere.**

## 2. Judgment inventory and grounding verification

| File | Statute rows | Relevance distribution | Coverage |
|---|---:|---|---|
| `judging_pilot.csv` (baseline Q01–Q05) | 40 | 0: 29 · 1: 7 · 2: 4 | 100% judged |
| `judging_expansion_experiment_manual_review.csv` (L1/L2) | 80 | 0: 53 · 1: 18 · 2: 9 | 100% judged |

These two are the only **independently human-judged** files. The three Phase 3A condition files (`judging_phase3a_cond{2,3,4}_*_prefilled.csv`) are reuse-only — 84/90, 88/90, and 83/90 rows respectively carry a label copied from one of the two files above by exact chunk-identity match; the remainder (6, 2, and 7 rows) are genuinely unjudged, not invented.

**Grounding check:** every one of the 120 judged statute `chunk_id` values in the two independently-judged files was checked against the live `var/rag/statutes_chunked/*.json` corpus. **0 of 120 failed to resolve** — no fabricated or stale chunk reference exists in the judged data. This is a verified check, not an assumption.

## 3. Coverage gaps and ambiguous queries

**3 of 8 indexed statutes have zero queries targeting them:** the Divorce Act 1869, the Child Marriage Restraint Act 1929, and the West Pakistan Muslim Personal Law (Shariat) Application Act 1962. Notably, the Divorce Act's chunks appear *frequently* as top (usually irrelevant) results for the 5 existing queries — meaning its own retrieval behavior has only ever been observed as noise contaminating other queries, never evaluated as a query target in its own right.

**Q01 is an ambiguous query** (restated from an earlier audit, not newly found here): no single statute or provision cleanly answers "can a wife obtain khula without returning dower" in full. Family Courts Act §10's proviso covers one specific route (court-ordered dissolution with mandatory Haq Mehr restoration), but "khula" in the broader Islamic-law sense is not confined to that clause alone.

## 4. Q03 regression — exact root cause

Statute P@5 for Q03 dropped from **0.200 (baseline) to 0.000** after adding "/ support" to the query — the only statute regression in Phase 3A. A direct rank-by-rank diff of the two retrieval runs (both fully judged, no label gaps) shows exactly what happened:

- Baseline rank 5 was `6867f44e86ef03cec19831c6:s9` (MFLO §9 "Maintenance", **relevance=1** — the only positive baseline chunk in the top-5).
- In the expanded run, this chunk is **pushed entirely out of the top-8**.
- Its displacement is directly attributable to `8251b5f4daa92e5a22319fc4:schedule-of-forms:form13_p2` (a Divorce Act procedural *form* fragment, relevance=0) entering at rank 4 — almost certainly matched on the generic English word "support," not on any substantive content overlap.
- A second, arguably *more* relevant chunk — `42769a60dc59c5473affbc32:s24` (Guardians and Wards Act §24, which literally says a guardian "must look to his support, health and education," relevance=1) — **does** enter the ranking because of "support," but only at rank 7, too low to count toward P@5 and only partially offsetting the nDCG@10 drop (0.387 → 0.333).

**Conclusion:** the regression is caused specifically by "support" being lexically generic enough to pull in irrelevant procedural boilerplate, not by candidate depth (held at the default 40 for this comparison) and not by any change to the fusion/ranking logic itself.

## 5. Dense-only / BM25-only / fused availability per query

| Query | Dense-only available? | BM25-only available? | Fused (production) available? |
|---|---|---|---|
| Q01 | **Yes** (multiple dedicated diagnostics) | **Yes** | Yes |
| Q04 | **Yes** (dedicated diagnostic) | **Yes** | Yes |
| Q02 | **No** | **No** | Yes |
| Q03 | **No** | **No** | Yes |
| Q05 | **No** | **No** | Yes |

**This is a real, reported gap, not an assumption.** For Q02, Q03, and Q05, only the fused hybrid result has ever been examined — no claim is made here about whether dense or sparse retrieval individually is responsible for their ranking behavior, because that breakdown was never run for these three queries.

## 6. Metrics sufficiency (statute, per condition)

| Condition | P@5 | nDCG@10 | MRR | Recall@5 / Recall@10 |
|---|---|---|---|---|
| Baseline | sufficient, all 5 queries | sufficient, all 5 | sufficient, all 5 | **insufficient — no qrels exist for any query** |
| Expansion (c=40) | sufficient, all 5 | sufficient, all 5 | sufficient, all 5 | insufficient, same reason |
| Depth only (c=100) | sufficient, all 5 | sufficient for **4 of 5** — Q01 missing a label for a chunk newly surfaced at depth 100 | sufficient, all 5 | insufficient, same reason |
| Combined (c=100 + expansion) | sufficient, all 5 | sufficient, all 5 | sufficient, all 5 | insufficient, same reason |

No Recall@5 or Recall@10 has ever been reported anywhere in this project's history, and none is reported here — no complete relevance reference set (qrels) has ever been constructed for any query, and the existing `calculate_metrics.py` correctly refuses to approximate one from top-k results.

## 7. Recommended benchmark

- **At least 16 queries** (2 per indexed statute × 8 statutes), replacing or supplementing the current 5.
- Per statute: **one direct article/section lookup** (e.g. "What does Section 5 of the Dowry and Bridal Gifts (Restriction) Act say?") **and one natural-language legal question**, both grounded in verified real content — not invented.
- Explicit coverage of the 3 currently-unqueried statutes.
- A **deliberately built, one-time qrels file** (a complete known-relevant-chunk set per query) — this is the only way Recall@5/Recall@10 stop being permanently "unavailable."

## 8. Reproducible baseline protocol

```
Collection: pk_legal · Dense: BAAI/bge-m3 · Sparse: Qdrant/bm25
candidates (prefetch depth) = 40 · k_statute = 8 · k_case = 10

python evaluation/run_retrieval.py --queries evaluation/queries.json --k-case 10 --k-statute 8 --candidates 40
python evaluation/judge_results.py --in <retrieval jsonl> --out <judging csv>
python evaluation/calculate_metrics.py --judged <judging csv> --out <metrics json>
```

This is exactly the configuration every existing report (baseline, Phase 1, Phase 3A) already used — recorded explicitly here so future experiments cite one documented reference point instead of re-deriving it from code each time.

## 9. Highest-priority next action

**Build the 16-query, 8-statute benchmark with a deliberate, one-time qrels file.** This single action unblocks Recall@5/Recall@10 (currently unavailable for every query that has ever been run) and closes the structural gap that 3 of 8 indexed statutes have never been evaluated at all.

## 10. Scope confirmation

No production file, chunk, embedding, Qdrant data, or existing evaluation artifact was modified. No retrieval was re-run. `git diff --stat` is empty before and after this audit; all new artifacts (`evaluation/statute_benchmark_audit.*` if a script were needed — none was, since this is pure inspection — plus `evaluation/results/statute_benchmark_audit.json`/`.md`) use filenames that did not previously exist.
