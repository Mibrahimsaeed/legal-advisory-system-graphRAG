# Query-Phrasing Experiment — Baseline vs. L1 vs. L2 Comparison

**Source files (read directly, values verified against the JSON — not guessed):**
`evaluation/results/metrics_pilot.json` (baseline, 5 queries) and
`evaluation/results/metrics_expansion_experiment_final.json` (10 variants, now fully judged — every
`precision_at_5`/`ndcg_at_10` cell for all 10 query/source combinations is a real number, not `null`).
`recall_at_10` is `null` for every query in both files (no qrels reference set exists for any query —
consistent with every prior report in this project) and is omitted from the tables below since it was
not requested, but is noted here so it is not mistaken for "not checked."

**Important methodological note:** the `"macro"` block already present in
`metrics_expansion_experiment_final.json` averages **all 10** variants (L1 and L2 combined) together —
it is not usable directly for an "L1 vs. L2" comparison. The macro figures below were computed
separately from the `per_query` data, averaging exactly the 5 `-L1` queries and, separately, the 5
`-L2` queries, each against the same 5 baseline queries, as the task requires.

## Case law

| Query | Source | Baseline P@5 | L1 P@5 | L2 P@5 | Δ L1 | Δ L2 | Baseline nDCG@10 | L1 nDCG@10 | L2 nDCG@10 | Δ L1 | Δ L2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Q01 | case | 1.000 | 1.000 | 0.600 | +0.000 | **−0.400** | 0.9897 | 0.9712 | 0.7319 | −0.0185 | **−0.2578** |
| Q02 | case | 1.000 | 1.000 | 1.000 | +0.000 | +0.000 | 0.8518 | 0.9384 | 0.9855 | +0.0866 | +0.1336 |
| Q03 | case | 1.000 | 1.000 | 0.800 | +0.000 | **−0.200** | 1.0000 | 0.9783 | 0.7518 | −0.0217 | **−0.2482** |
| Q04 | case | 1.000 | 0.800 | 1.000 | **−0.200** | +0.000 | 0.8032 | 0.9446 | 0.9892 | +0.1414 | +0.1860 |
| Q05 | case | 1.000 | 1.000 | 1.000 | +0.000 | +0.000 | 0.7784 | 0.9986 | 0.7990 | +0.2202 | +0.0206 |
| **Macro (5 families)** | case | **1.000** | **0.960** | **0.880** | **−0.040** | **−0.120** | **0.8846** | **0.9662** | **0.8515** | **+0.0816** | **−0.0332** |

## Statutes

| Query | Source | Baseline P@5 | L1 P@5 | L2 P@5 | Δ L1 | Δ L2 | Baseline nDCG@10 | L1 nDCG@10 | L2 nDCG@10 | Δ L1 | Δ L2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Q01 | statute | 0.200 | 0.200 | 0.000 | +0.000 | **−0.200** | 0.5000 | 0.6309 | 0.3333 | +0.1309 | **−0.1667** |
| Q02 | statute | 0.600 | 0.600 | 0.400 | +0.000 | **−0.200** | 0.7581 | 0.6665 | 0.8290 | −0.0916 | +0.0708 |
| Q03 | statute | 0.200 | 0.200 | 0.200 | +0.000 | +0.000 | 0.3869 | 0.4556 | **1.0000** | +0.0687 | **+0.6131** |
| Q04 | statute | **0.000** | 0.200 | 0.200 | **+0.200** | **+0.200** | **0.0000** | 0.8175 | 0.4747 | **+0.8175** | **+0.4747** |
| Q05 | statute | 0.800 | 0.400 | 0.600 | **−0.400** | −0.200 | 0.9925 | 0.9733 | 0.6623 | −0.0191 | **−0.3302** |
| **Macro (5 families)** | statute | **0.360** | **0.320** | **0.280** | **−0.040** | **−0.080** | **0.5275** | **0.7088** | **0.6599** | **+0.1812** | **+0.1324** |

No `n/a` values appear in either table: the judged final metrics file has a real, numeric
`precision_at_5` and `ndcg_at_10` for all 10 query/source pairs, so nothing here is an unavailable
metric being hidden — every number above is a direct read from the file, not an estimate.

## Summary — does L1 or L2 appear to improve retrieval?

**The two metrics tell different stories, and the answer depends on source type:**

- **Statutes — nDCG@10 improves on average for both L1 (+0.181) and L2 (+0.132); Precision@5
  slightly worsens on average for both (−0.04 and −0.08).** The clearest single result is **Q04
  statute**: baseline Precision@5 and nDCG@10 were both 0.000 (zero relevant statute chunks in the top
  5, and none anywhere usably ranked in the top 10); L1 raises nDCG@10 to 0.818 and L2 to 0.475, each
  with Precision@5 = 0.200. This is consistent with the earlier retrieval diagnostic, which showed the
  correct Dowry Act chunks exist and are indexed but were ranked just outside the production cutoff —
  rewording moved them in. Notably, **L1 (vocabulary only) outperformed L2 (named-authority) on Q04's
  nDCG@10** (0.818 vs. 0.475), the opposite of what a simple "naming the Act always helps most"
  expectation would predict.
- **Statutes — rewriting is not uniformly helpful.** Q01 and Q05 statute Precision@5 both *decreased*
  under L2 (Q01: 0.2→0.0; Q05: 0.8→0.6), and Q05 also decreased under L1 (0.8→0.4). Q05 was already
  the best-performing baseline statute query; both rewrites made it worse. Rewriting clearly is not a
  one-directional fix across the board, even within the same source type.
- **Case law — already near-ceiling at baseline (macro P@5 = 1.000), so there is little room to
  improve and real room to regress.** L1's case macro nDCG@10 rose slightly (+0.082) but its P@5
  dipped (−0.04, driven entirely by Q04 case dropping from 1.0 to 0.8). **L2 reduced case-law
  performance on average** (P@5 −0.12, nDCG@10 −0.033) — Q01 and Q03 case-law Precision@5 both dropped
  noticeably (1.0→0.6 and 1.0→0.8) under the named-authority rewrite, even though case-law retrieval
  for those queries was already working well at baseline.

**Net reading:** rewriting shows the most consistent, positive signal specifically for **statute
nDCG@10**, and the single largest, cleanest win is Q04 — the query this whole investigation started
from. But the same rewrites can measurably hurt already-well-performing queries (most visibly Q05
statute and, for case law, Q01/Q03 under L2). There is no evidence here that one rewrite level (L1 vs.
L2) is uniformly superior; which one helps appears to depend on the specific query.

## Required disclosures

- **Exploratory only.** This comparison spans exactly 5 query families (10 variants total). No formal
  significance test was or should be performed on numbers this sparse — a single query's result can
  swing a macro average substantially (e.g. Q04 statute alone contributes +0.82/+0.47 of the entire
  statute nDCG@10 macro gain), and should not be read as evidence of a corpus- or system-wide effect.
- **Q04 is not a blind test.** Its L1/L2 phrasings were informed by an earlier, separate read-only
  retrieval diagnostic that had already probed similar wording for Q04 specifically before this
  experiment was designed. Its large, clean improvement is therefore weaker evidence of a generalizable
  fix than a same-sized effect on Q01, Q02, Q03, or Q05 would be — all of which were tested blind.
- **Topical relevance (what Precision@5/nDCG@10 measure) is not the same as whether a chunk fully
  answers the legal question.** This comparison only reflects the `relevance` labels already recorded;
  it does not draw on the separate `answer_bearing` qualitative dimension prepared earlier in this
  project (`judging_expansion_experiment_manual_review.csv`), which was out of scope for this specific
  numeric comparison request.
- **L2 (named-authority) queries are a ceiling test, not a production rewriting policy.** Real users
  will not typically name a specific Act and year; L2's results show what is achievable with
  near-perfect vocabulary, not what a deployable query-rewriter would reliably produce — and, as shown
  above, even this ceiling test regressed several already-good baseline queries.

## Scope confirmation

No retrieval was re-run, no evaluation script was modified, no judgment was altered, and Qdrant/
embeddings were not touched to produce this comparison — both input files were read as-is and all
numbers above were computed directly from their contents.
