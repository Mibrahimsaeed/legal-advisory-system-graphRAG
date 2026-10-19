# Controlled Experiments — Status and Results

**This audit reuses already-executed experiments where they exist and is explicit about what was NOT re-run for Constitution/case law**, rather than fabricating new experiments under time pressure.

## Experiment A — Candidate depth (already executed, statutes only)

**Result (SRI Phase 4, `evaluation/results/sri_phase4_expB_metrics.json`):** `candidates` 40→100, query text unchanged. Macro nDCG@10 for the 16-query statute benchmark: 0.896 → **0.890** (net negative). Q13 specifically regressed (0.689 → 0.587) — more distractor chunks entered the wider pool and displaced the correct ones during RRF fusion. All other 15 queries unchanged. **Conclusion stands: deeper candidate pools are not assumed to help, and here measurably hurt one query.**

**Not re-run for Constitution or case law in this pass.** Given the statute result was net-negative and the task explicitly warns not to assume depth helps, extending this experiment to the other two corpora was not prioritized — flagged as an open item, not silently skipped.

## Experiment B — Terminology expansion (executed for statutes; outcomes verified, not assumed to generalize)

**Q13 (statute):** adding the verified term "/ divorce a mensa et toro" (confirmed verbatim in the target's own text) → nDCG@10 0.689 → **1.000**, zero regressions elsewhere (SRI Phase 4 Experiment A).

**Q06 (statute):** adding "/ aggregate value limit" (verbatim in §3) → nDCG@10 0.631 → **1.000** (SRI Phase 5).

**Q16 (statute):** adding "/ rule of decision" (verbatim in §2) → nDCG@10 0.431 → **0.500** — a real but incomplete improvement; the Application Act's shorter, more title-dense preamble/§1 chunks still partially resist the fix (SRI Phase 5).

**Combined Q06+Q16:** no interaction effect — each query's result was identical to its own individual experiment (SRI Phase 5).

**Not tested for Constitution or case law in this pass.** CL1 and C4's diagnosed issues (dense weakness, qrels-granularity ambiguity) are different failure *types* than the title/preamble lexical-competition pattern the statute terminology experiments targeted — a new, corpus-appropriate terminology hypothesis for CL1/C4 would need to be separately grounded and verified before testing, which was not done here. This is reported as untested, not assumed to fail or succeed.

## Experiment C — Reranker feasibility

**Not run.** Per its own trigger condition ("if ranking failures are frequent and relevant chunks already enter the candidate set"): this audit found the *statute* corpus's weak queries are exactly this pattern (chunk in candidates, ranking failure) and already has a cheaper, verified fix (terminology expansion) for 2 of 3 instances. For case law (CL1), the dominant issue is a **candidate-generation** weakness in the dense branch (rank 208), which "a reranker cannot recover a relevant chunk that never reaches its candidate set" (per the task's own Phase 4A framing) directly cautions against — CL1's target DOES reach the BM25 candidate set (rank 12) and the fused list (rank 25), so a reranker operating on the already-fused top-N *could* theoretically help here, but this was not tested. **A reranker feasibility assessment is a reasonable next experiment, not run in this pass.**

## Experiment D — Chunking sensitivity

**Not run.** Trigger condition (chunk-content or chunk-boundary defect) was not met for the statute corpus's issues (confirmed query-side lexical competition, not chunk dilution, in the earlier full-corpus integrity audit). For CL1, the "qrels-granularity" finding (the same legal point restated across a headnote chunk and a judgment-body chunk) is arguably a case-law-specific *chunking convention* question (should headnote and judgment-body restatements of the same point be more tightly linked, e.g. via stronger `prev_chunk_id`/`next_chunk_id` context or a combined retrieval unit?) — **this is a plausible hypothesis worth flagging for Experiment D in a future pass, not tested here.**

## Experiment E — Embedding model sensitivity

**Not run**, and per Phase 4G's own instruction, **not justified by current evidence** — only one query (CL1) shows a clearly weak dense rank, and several other case-law/Constitution queries show strong dense ranks (1-3). A single instance does not establish a corpus-wide embedding problem.

## Experiment F — Combined improvements

**Not run** — no two experiments have yet independently demonstrated benefit for the *same* corpus/query pair outside the already-combined Q06+Q16 statute case (which showed no interaction, reported above).

## Summary table

| Experiment | Statutes | Constitution | Case law |
|---|---|---|---|
| A — Candidate depth | Run; net negative | Not run | Not run |
| B — Terminology expansion | Run; 2 clean wins (Q06, Q13), 1 partial (Q16) | Not run | Not run |
| C — Reranker feasibility | Not run (cheaper fix already found) | Not run | Not run (plausible next step) |
| D — Chunking sensitivity | Not run (no trigger) | Not run | Not run (plausible hypothesis for CL1) |
| E — Embedding sensitivity | Not run (no trigger) | Not run | Not run (no trigger — 1 instance insufficient) |
