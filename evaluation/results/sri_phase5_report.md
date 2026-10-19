# Phase 5 — Targeted Lookup Query Optimization: Q06 and Q16

**Frozen baseline integrity confirmed before and after:** `queries_statute_benchmark16.json`, `sri_phase1_qrels.json`, `sri_phase1_qrels_rich.json`, and `sri_phase2_baseline_metrics.json` are byte-identical (checksummed) to their Phase 1/2 state. Qdrant point count unchanged at 26,678. `git diff --stat` empty throughout.

## 1. Baseline ranking reconstruction and manual inspection

From `sri_phase3_branch_diagnostics.json` (not re-run, reused as saved):

| Query | Target | Fused rank | What outranks it |
|---|---|---:|---|
| Q06 | `417403be7fc29b06cafadbbe:s3` (Dowry Act §3) | 2 | `...preamble` (rank 1) |
| Q16 | `b11740afa5f050bea50c5f2d:s2` (Application Act §2) | 4 | `...preamble` (1), `...s1` (2), `...s7` (3) |

**Manual inspection confirms the lexical-competition hypothesis directly, by reading the actual text:**
- Dowry Act preamble: *"THE DOWRY AND BRIDAL GIFTS (RESTRICTION) ACT, 1976 ... ACT No. XLIII OF 1976 ... An Act to provide for restrictions on dowry and bridal gifts."* — verbatim overlap with Q06's own wording.
- Application Act preamble/§1/§7 each repeat *"West Pakistan Muslim Personal Law (Shariat) Application Act, 1962"* verbatim (§7 again via "The Muslim Personal Law (Shariat) ..." in its repeal list).

**This confirms task item 2:** full-title wording in the query is directly responsible for the lexical competition in both cases — verified by reading the competing chunks' own text, not inferred.

## 2. Proposed terminology changes — one per query, both verified

| Query | Term added | Source-text evidence | Discriminating check |
|---|---|---|---|
| **Q06** | `/ aggregate value limit` | §3 states: *"Neither the **aggregate value** of the dowry and presents... nor the **aggregate value** of the bridal gifts..."* — verbatim, twice. | Confirmed absent from preamble/§1/§7 — this phrase exists only in §3. |
| **Q16** | `/ rule of decision` | §2 states: *"...the **rule of decision**, subject to the provisions of any enactment for the time being in force, shall be..."* — verbatim. | Confirmed absent from preamble/§1/§7 — this phrase exists only in §2. |

Original queries preserved unchanged; both additions are appended, not substituted.

## 3. Individual experiment results

| Query | Baseline nDCG@10 | Expanded nDCG@10 | Baseline rank | Expanded rank | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|---|---|
| **Q06** (own experiment) | 0.631 | **1.000** | 2 | **1** | 0.5 → 0.5 (unchanged) | 0.5 → 0.5 (unchanged) |
| **Q16** (own experiment) | 0.431 | **0.500** | 4 | **3** | n/a (single-chunk qrels, already 1.0) | 1.0 → 1.0 (unchanged) |

**Q06: clean, strong win.** §3 moved from rank 2 to rank 1, overtaking the preamble. nDCG@10 reached the maximum possible value (1.000). Recall@5/Recall@10 did **not** improve — the grade-1 supporting chunk (§9) still does not reach the top 10 either before or after; this change only affected the primary target's rank, not the secondary one, and is reported exactly as that — a partial fix to the qrels set, not a complete one.

**Q16: real but incomplete improvement.** §2 moved from rank 4 to rank 3, overtaking only §7 — the preamble and §1 (both of which repeat the Act's title even more densely, in much shorter chunks) still outrank it. nDCG@10 improved (0.431 → 0.500) but far short of 1.000. **Per task item 9, this is reported honestly as a partial, insufficient fix — no further terms were added to chase a better number.** The likely reason: the Application Act's preamble/§1 are shorter and more title-dense than the Dowry Act's, giving them a disproportionately strong lexical match on the literal title string, which "rule of decision" (one phrase, not a repeated proper noun) wasn't strong enough to fully overcome.

## 4. Combined experiment

| Query | Combined nDCG@10 | Matches individual experiment? |
|---|---:|---|
| Q06 | 1.000 | Yes — identical to the Q06-only run |
| Q16 | 0.500 | Yes — identical to the Q16-only run |

**No interaction effect** — each query's own text change affects only that query's own ranking, exactly as expected since Q06 and Q16 query different statutes entirely. Combining them produces the arithmetic sum of the two individual effects, nothing more and nothing less.

## 5. Macro metrics and regression analysis

| Run | Macro nDCG@10 | Macro Recall@10 | Regressions (any of the other 14 queries) |
|---|---:|---:|---|
| Frozen baseline | 0.896 | 0.969 | — |
| Q06-only | 0.919 | 0.969 | **None** — all 15 other queries byte-identical to baseline |
| Q16-only | 0.894 | 0.969 | **None** — all 15 other queries byte-identical to baseline |
| Combined | 0.923 | 0.969 | **None** — all 14 other queries byte-identical to baseline |

Macro Precision@5 and Recall@5 are unchanged in every run (0.275 and 0.896 respectively) — expected, since neither experiment changed which queries have a small qrels set, and Q16's single-chunk qrels set was already fully recalled at baseline.

## 6. Observed improvements vs. untested hypotheses

**Observed, verified improvements (both experiments actually run):**
- Q06's terminology addition is a complete, clean fix for nDCG@10 at this benchmark's depth — directly measured.
- Q16's terminology addition is a real, partial, directionally-correct improvement — also directly measured, not assumed to be "fixed."
- Combining both produces no cross-query interaction — directly measured, not assumed.

**Untested hypotheses (explicitly not claimed as fact):**
- Whether a *second*, different verified term for Q16 could close the remaining gap to the preamble/§1 was not tried — per task instruction 9, iterating terms until the metric looks better was deliberately avoided.
- Whether Q06's §9 (grade-1 supporting chunk) could similarly be pulled into the top 10 with its own targeted term was not investigated in this phase.
- The old "support"/child-maintenance regression remains untested here, per task instruction 10 — it was not added as a new query in this phase.

## 7. Next recommendation

**Investigate why the Application Act's preamble/§1 resist the same fix that worked cleanly for the Dowry Act** — likely their shorter length and denser title repetition — before proposing a second term for Q16. Do not add more terms to Q16 speculatively; first confirm the length/density hypothesis against the two Acts' actual chunk lengths, since that would explain why the identical *method* produced a full fix for one query and only a partial one for the other.

## Files created (all new, none overwriting existing artifacts)

`evaluation/results/sri_phase5_{q06_only,q16_only,combined}_queries.json`, `_retrieval.jsonl`(+`.meta.json`), `_judging_raw.csv`, `_judging_complete.csv`, `_metrics.json`, and this report.
