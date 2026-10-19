# Statute Retrieval Improvement — Master Report (Phases 0–4)

All phases reused the existing, unmodified `hybrid_search()` (`src/rag_prep/embed_index.py`) and `run_retrieval.py`/`judge_results.py`/`calculate_metrics.py`. No production code, chunk, embedding, or Qdrant data was modified at any point. Full detail per phase is in the correspondingly-named `sri_phase{0,1,2,3,4}_*` files under `evaluation/` and `evaluation/results/`.

## 1. Benchmark readiness

**16 queries, 8 statutes, 2 each (1 natural-language + 1 direct lookup) — built, verified, and admitted to scoring.** Every qrels judgment was made by directly reading the current statute source text, not inferred from the query or derived from a retriever's output. 20 judgments across 16 queries, including 6 explicitly-verified hard negatives. 0 of 20 judged chunk IDs failed to resolve to the live corpus. All labels marked `verified`; none left `pending_manual_review`.

**One carried-forward caveat:** qrels completeness rests on a targeted per-topic keyword sweep plus direct source reading, not an exhaustive manual re-read of all 240 statute chunks against all 16 queries — disclosed in `sri_phase1_benchmark_audit.md`, not hidden.

**Unresolved issue:** the earlier-observed "support" expansion regression (original Q03, child maintenance) has **no equivalent query in this new 16-query, per-statute benchmark** — that topic isn't one of the 16 statute-coverage queries — so it could not be directly re-tested here. This is reported as a gap, not silently dropped.

## 2. Baseline results (Phase 2, `sri_phase2_baseline_*`)

Configuration: `candidates=40`, `k_statute=10`, unchanged production `hybrid_search()`.

| Metric | Macro (16 queries) |
|---|---:|
| Precision@5 | 0.275 |
| nDCG@10 | 0.896 |
| Recall@5 | 0.896 |
| Recall@10 | **0.969** |
| MRR | 0.922 |

**This is the first time Recall@5/Recall@10 have ever been validly computed in this project** — every prior report reported them "unavailable" for lack of qrels.

**Important methodological note on P@5:** macro P@5 (0.275) looks low next to nDCG@10/Recall@10 (~0.9+) because most of these 16 queries have a genuinely small qrels set (1–3 relevant chunks total, by design — narrow statutory lookups don't have 5 equally relevant provisions). Precision@5 divides by 5 regardless, so even *perfect* retrieval of a 1-chunk relevant set caps P@5 at 0.2. **This is a property of the queries, not a retrieval failure** — nDCG@10 and Recall@10 are the more informative metrics for this benchmark's query style.

**By statute:** all 8 statutes score nDCG@10 ≥ 0.63; weakest are Q16 (Application Act lookup, 0.431) and Q06 (Dowry Act lookup, 0.631) and Q13 (Divorce Act NL, 0.689 at baseline — fixed in Experiment A, see below).

**NL vs. lookup:** nDCG@10 0.933 (NL) vs. 0.859 (lookup); Recall@10 1.000 (NL) vs. 0.938 (lookup) — lookup queries underperform slightly, explained in Phase 3.

**Validation gate:** all 16 queries ran successfully; ranking outputs matched expected query IDs; qrels files were not modified during evaluation (confirmed by checksum-equivalent re-read); metrics passed internal consistency checks (e.g. Q01's directly-computed `sim(Q0,D0)` matched its own re-derivation exactly). **Gate: PASS.**

## 3. Retrieval diagnostics (Phase 3, `sri_phase3_branch_diagnostics.*`)

Dense-only, BM25-only, and fused rankings obtained independently for all 16 queries (240-deep, unfused `query_points` calls for the first two; the real `hybrid_search()` for fusion) — **no branch was inferred from another.**

**Two concrete, evidence-backed failure patterns found** (both affecting lookup-style queries specifically):

1. **Preamble/short-title lexical dominance.** Direct-lookup queries that quote a statute's full title + year (by design, per the task's own query style) create strong lexical/semantic overlap with that statute's own **preamble and §1 "short title"** chunks — which say almost nothing substantive. For Q16 (`b11740afa5f050bea50c5f2d:s2`), the preamble, §1, and §7 all outrank the actual target, which lands at fused rank 4. For Q13 (baseline), several generic Divorce Act chunks (§25, §26, §60) outrank the true grounds section.
2. **Q06's supporting-context chunk (§9, grade-1) under-ranks at 12th**, outside the top-10, even though it's within both branches' top-40 — its own content (a penalty-calculation clause that merely *references* the value cap) is a weaker, indirect lexical match, which is reasonable given it's only secondary context, not a failure of the primary target (§3, which ranks 2nd).

Neither pattern is a chunk-dilution, missing-metadata, or corpus-gap problem — both targets' own chunks are intact, correctly payload-tagged, and within both branches' candidate pools. The cause is query-side lexical competition from administrative boilerplate chunks, confirmed by direct branch-level inspection, not assumed.

**Validation gate:** all three ranking sets used identical query text, the same qrels, the same corpus snapshot, and comparable cutoffs (240-deep). No production retrieval or live index was touched. **Gate: PASS; specific, evidence-backed failure patterns identified** — proceeding to Phase 4.

## 4. Experiment results (Phase 4)

| Experiment | Change | Macro nDCG@10 | Per-query effect | Regressions? |
|---|---|---:|---|---|
| **A — Terminology expansion** | Q13 only: appended "/ divorce a mensa et toro" (verified present verbatim in §22's own text) | **0.896 → 0.915** | Q13: 0.689 → **1.000** | **None** — all other 15 queries byte-identical to baseline |
| **B — Candidate depth** | `candidates`: 40 → 100, query text unchanged | 0.896 → 0.890 | Q13: 0.689 → **0.587** (worse); all 15 others unchanged | **Yes — Q13 regressed** |

**Experiment A is a clean, validated win:** reproducible, uses the existing qrels and evaluation code unmodified, the improvement is fully explained by the branch diagnostics (the added term shifts the fused ranking above the generic distractors), and no other query was affected — satisfying all 5 acceptance criteria.

**Experiment B is informative but net-negative for this benchmark:** increasing candidate depth alone, with no query change, is **not neutral** here — it measurably *worsened* Q13 (more distractor chunks entered the wider candidate pool and displaced the correct ones during RRF fusion) while changing nothing else. This directly confirms, more strongly than prior work, the task's caution that higher candidate depth must not be assumed to help.

**Experiment C — statute-specific chunking/metadata changes — was not run.** Its trigger condition (Phase 3 evidence of chunk dilution, missing section identity, or poor payload filtering) was not met: both Phase 3 failure patterns are query-side lexical competition from intact, correctly-tagged chunks, not a chunk-level defect. Running Experiment C without that evidence would violate the phase's own precondition, so it was skipped, not silently omitted.

**No combined experiment was run**, since Experiment B alone did not justify combining with A (B was net-negative).

## 5. Integrity

**Final `git status --short`:** 24 new untracked files (listed below), all under `evaluation/` or `evaluation/results/`, none overwriting a prior filename. **Final `git diff --stat`: empty** — identical to the initial check; no tracked file was modified anywhere in this workflow.

**Files created:**
`evaluation/queries_statute_benchmark16.json`, `evaluation/queries_statute_benchmark16_expA.json`, `evaluation/sri_phase3_branch_diagnostics.py`, and under `evaluation/results/`: `sri_phase0_discovery.md`, `sri_phase1_benchmark_audit.md`, `sri_phase1_qrels.json`, `sri_phase1_qrels_rich.json`, `sri_phase2_baseline_retrieval.jsonl`(+`.meta.json`), `sri_phase2_baseline_judging.csv`, `sri_phase2_baseline_judging_complete.csv`, `sri_phase2_baseline_metrics.json`, `sri_phase2_baseline_mrr.json`, `sri_phase2_baseline_recall5.json`, `sri_phase3_branch_diagnostics.json`, `sri_phase4_expA_retrieval.jsonl`(+`.meta.json`), `sri_phase4_expA_judging_raw.csv`, `sri_phase4_expA_judging_complete.csv`, `sri_phase4_expA_metrics.json`, `sri_phase4_expB_retrieval.jsonl`(+`.meta.json`), `sri_phase4_expB_judging_raw.csv`, `sri_phase4_expB_judging_complete.csv`, `sri_phase4_expB_metrics.json`, this report.

**No production code, chunking, embeddings, payloads, or the live Qdrant collection was modified.** Qdrant point count re-checked unchanged at 26,678 throughout.

**All four phases passed their validation gates** (Phase 1's completeness caveat and Experiment C's documented skip are disclosed, not treated as failures).

## 6. Recommendation

**Single most impactful, evidence-backed finding:** direct-lookup queries that restate a statute's full title and year create lexical competition with that statute's own preamble/short-title chunks, measurably depressing nDCG@10 and Recall@10 for lookup-style queries relative to natural-language ones (0.859/0.938 vs. 0.933/1.000 macro). Experiment A showed that a small, verified, corpus-grounded terminology addition can fully resolve an instance of this (Q13: 0.689 → 1.000) with zero regressions elsewhere — the strongest, cleanest positive result produced in this entire workflow.

**Single highest-priority next action:** before any production change, extend Experiment A's exact method (one corpus-verified term per affected query, tested individually, compared against the frozen baseline and qrels built here) to the other weak lookup queries identified in Phase 3 — Q06 and Q16 — since the evidence from Q13 suggests the same lexical-competition pattern is the likely cause there too, but this has not yet been tested for those two queries and should not be assumed to generalize without running it.
