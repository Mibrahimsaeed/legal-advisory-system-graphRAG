# CL11 / CL18 Diagnosis

## CL11

- **Frozen query text:** "Under what circumstances may a court condone delay in filing a civil proceeding or appeal?"
- **Target chunk:** `8e2059846f6b77bd8ddd9a3a:0002` (case law)
- **Dense-only rank:** 10 (of 40 candidates; score 0.632)
- **Sparse-only rank:** **37 of 40** (score 17.32) — **CORRECTION to the prior audit's report**, which only saved sparse top-20 and reported the target as "absent." Fresh depth-40 retrieval (matching production's actual `candidates=40` prefetch depth) confirms the target **is** present in the sparse candidate pool, just very weakly (rank 37 of 40).
- **Hybrid (production, frozen) rank:** 17
- **Offline-recomputed rank (validated k=1 RRF, equal weights):** 17 — **exact match to production.**
- **Highest-ranking competitors in hybrid (rank 1-5):** not individually traced to their own dense/sparse ranks in this pass (out of time budget); all are judged relevance=0 per the frozen qrels (not in the qrels' relevant set).
- **Category: 3 — present in both branches, demoted by fusion.** Not candidate-generation failure (target reaches both pools); not a qrels limitation (target is genuinely on-topic, directly verified). The demotion is a direct, reproducible mathematical consequence of RRF: a rank-37 sparse contribution is tiny (1/(1+37) ≈ 0.026), so chunks that rank moderately in BOTH branches accumulate a larger combined score and outrank a chunk that is good in one branch and merely present-but-weak in the other.
- **Reproducible from saved rankings?** Only after re-fetching depth-40 branch data (the prior audit's saved top-20 was insufficient — this is disclosed, not hidden). Once depth-40 data was available, the production hybrid rank (17) was reproduced exactly offline.

## CL18

- **Frozen query text:** "When may a constitutional petition be declined because an alternative legal remedy is available?"
- **Target chunk:** `271e7125df86baebc57dc205:0002` (case law)
- **Dense-only rank:** 6 (of 40; score 0.638)
- **Sparse-only rank:** **25 of 40** (score not in top-20, confirmed present at depth 40) — same correction as CL11.
- **Hybrid (production, frozen) rank:** 10
- **Offline-recomputed rank (validated k=1 RRF, equal weights):** 10 — **exact match to production.**
- **Category: 3 — present in both branches, demoted by fusion.** Same mechanism as CL11: sparse rank 25 contributes a small amount (1/(1+25) ≈ 0.038), diluting what would otherwise be a strong dense-rank-6 position.

## Summary: is the demotion pattern reproducible and genuinely caused by fusion?

**Yes to both, now confirmed with corrected evidence.** The original diagnosis's *conclusion* (fusion demotes a dense-findable target) holds, but its *premise* (target "absent" from sparse) was wrong — both targets are present in the sparse candidate pool, just ranked weakly (37, 25 of 40). The demotion is a direct, reproducible, validated mathematical consequence of RRF with weak-but-nonzero sparse contributions, not a missing-candidate issue. This is confirmed, not merely probable, because the offline recomputation — using the exact same fixed candidate lists and a validated fusion constant — reproduces the production rank exactly for both queries.
