# Comparison Report — Offline Cross-Encoder Reranker Experiment (Case Law)

## 1. Setup summary

- **Model**: `BAAI/bge-reranker-v2-m3` (sentence-transformers `CrossEncoder`), fp32, MPS device, `max_length=512`.
- **Candidates**: the frozen, unmodified hybrid top-40 case-law rankings from the 150-query
  benchmark (`retrieval_hybrid_case.jsonl`) — 50 queries × 40 candidates = 2,000 rows.
- **Qrels**: the frozen expanded case-law judgment ledger (`judgment_ledger.jsonl`,
  674 relevant case-document judgments, document-level grading).
- **Duplicate chunks**: of 2,000 candidate slots, only 1,361 are unique case documents
  (avg 12.8 duplicate-case chunk-slots per query, max 23, min 0) — confirming hybrid
  retrieval frequently returns multiple chunks of the same case within its own top-40.
  Document-level metrics count each case once per query, per the convention already
  established in the qrels-expansion task (grade propagated from the judged case to every
  chunk of that case encountered in a ranking).

## 2. A critical, disclosed limitation discovered by this experiment

The expanded qrels (`case_law_qrels_expansion_20261019_220106`) judged each case-law query
to a depth of **top-10 of each ORIGINAL method's own ranking** (dense/sparse/hybrid),
exactly matching what the original benchmark's metrics needed. **Reranking reorders the
same 40 candidates**, which routinely promotes candidates that were ranked 11–40 in the
original hybrid order into the new top-5/top-10 — and those candidates are frequently
**unjudged**, because they fell outside every original method's top-10 pool.

**Consequence**: nDCG@10 (and, to a lesser extent, P@5/nDCG@5) cannot be honestly computed
for most queries after reranking.

| Metric | Baseline — queries with a usable value | Reranked — queries with a usable value |
|---|---:|---:|
| P@5 | 50 / 50 | 25 / 50 |
| nDCG@5 | 50 / 50 | 25 / 50 |
| nDCG@10 | 50 / 50 | **7 / 50** |
| Recall@5, Recall@10, MRR@5/10, HitRate@5/10 | 50 / 50 | 50 / 50 |

Recall/MRR/HitRate remain fully computable because they are membership tests against the
known-relevant set, not full-ranking-window grading — they do not require every candidate
in the top-k to be judged. **This single fact is the dominant finding of this experiment**:
any reranker evaluation against these qrels is only reliably comparable on nDCG@10 for a
small, non-random 7-query subset, and this must be fixed (re-judge candidates that the
reranker newly promotes) before a future reranker experiment can be trusted on its primary
metric.

## 3. Metrics (macro averages; denominators as shown — see table above for availability)

| Metric | Baseline hybrid (paired subset) | Reranked hybrid | n (paired) | Baseline, full 50 queries |
|---|---:|---:|---:|---:|
| P@5 | 0.864 | 0.896 | 25 | 0.900 |
| nDCG@5 | 0.902 | 0.925 | 25 | 0.921 |
| **nDCG@10** | 0.812 | 0.921 | **7** | 0.900 |
| Recall@5 | 0.292 | 0.270 | 50 | 0.292 |
| **Recall@10** | 0.540 | 0.432 | 50 | 0.540 |
| MRR@5 / MRR@10 | 0.970 | 0.923 | 50 | 0.970 |
| HitRate@5 / HitRate@10 | 1.000 | 1.000 | 50 | 1.000 |

The "baseline, full 50 queries" column is each metric's macro-average over all 50 queries
(reproduced from `metrics_summary.json`'s `baseline_hybrid` entry — identical to the
originally reported case-law baseline). The "paired subset" column is the SAME baseline
metric recomputed only over the queries where the reranked arm also has a usable value —
this is the only fair basis for the significance comparison, since comparing a 50-query
baseline average against a 7-query reranked average would conflate "which queries happen to
stay fully judged after reranking" with "whether reranking helps." Notably, the
**7-query nDCG@10 subset's baseline average (0.812) is noticeably lower than the full-50
average (0.900)** — these 7 queries were NOT a representative, easy subset; if anything they
were baseline-harder-than-average queries, which is a further reason not to generalize the
nDCG@10 "improvement" to the full benchmark.

## 4. Statistical analysis (paired bootstrap, seed=42, 5,000 resamples, query-level resampling)

| Metric | n | Diff (reranked − baseline) | 95% CI | Significant? | Improved / Unchanged / Worsened |
|---|---:|---:|---|---|---|
| P@5 | 25 | +0.032 | [−0.024, 0.088] | No | 6 / 17 / 2 |
| nDCG@5 | 25 | +0.023 | [−0.035, 0.083] | No | 11 / 4 / 10 |
| **nDCG@10** | **7** | **+0.108** | **[0.014, 0.210]** | **Yes** | 6 / 0 / 1 |
| Recall@5 | 50 | −0.022 | [−0.053, 0.010] | No | 10 / 18 / 22 |
| **Recall@10** | **50** | **−0.108** | **[−0.138, −0.080]** | **Yes** | 4 / 7 / 39 |
| MRR@5 / MRR@10 | 50 | −0.047 | [−0.117, 0.017] | No | 3 / 40 / 7 |
| HitRate@5 / HitRate@10 | 50 | 0.000 | [0.000, 0.000] | No (ceiling) | 0 / 50 / 0 |

**Two statistically significant results, pointing in opposite directions:**

1. **nDCG@10 "improves" significantly — but on only 7 of 50 queries (14%)**, a sample far
   too small and too non-random (queries where reranking happened not to disturb the
   already-judged top-10 boundary) to generalize. **This result should NOT be read as
   "reranking improves nDCG@10."**
2. **Recall@10 regresses significantly, on the full 50-query sample** (39 of 50 queries
   worsened, only 4 improved) — a large, consistent, well-powered effect. Reranking
   demonstrably demotes known-relevant cases out of the top-10 more often than it promotes
   new ones in, even though it does not affect whether at least one relevant case is still
   found (HitRate@5/10 stay at a ceiling of 1.0 for both arms).

## 5. Per-query regression inspection

- The single nDCG@10 regression in the 7-query paired subset is **CL49** (0.916 → 0.829,
  Δ=−0.087) — a query with very limited expanded-qrels coverage already (only one newly
  pooled candidate judged in the qrels-expansion task), so this single data point should
  not be over-interpreted either way.
- The Recall@10 regressions are broad and substantial: e.g. CL06 (0.667→0.333), CL11
  (0.625→0.312), CL15 (0.529→0.235), CL47 (0.571→0.286) — drops of 0.23–0.33 in the
  fraction of known-relevant cases retained in the top 10. A small number of queries (CL05,
  CL16, CL22, CL31) improved Recall@10 instead, but these do not offset the broad regression.

## 6. Latency and resource cost

See `latency_report.json` for full detail. Summary: 2,000 candidates scored in 240.8s total
inference (9.0s model init) on Apple Silicon MPS, fp32 — **~4.80s mean / 4.76s median per
query (40 candidates), ~0.12s per candidate**. 43 of 2,000 candidates (2.15%) were truncated
at the 512-token input limit. Peak MPS allocated memory ~2.27GB (driver-reserved ~6.0GB).
These figures are specific to this machine's MPS backend and are not directly comparable to
a CPU or CUDA deployment without separate measurement there.

## 7. Recommendation

**NOT PROMISING — do not integrate this configuration.**

The one metric with adequate statistical power across all 50 queries (Recall@10) shows a
**significant, broad-based regression**. The one metric showing a significant improvement
(nDCG@10, the task's stated primary metric) is evaluable on only 7 of 50 queries due to a
qrels-depth limitation exposed by this very experiment, and cannot be trusted as
representative of the full benchmark. MRR and P@5/nDCG@5 show no significant change in
either direction. HitRate is already at a ceiling for both arms and is uninformative here.

This is not evidence that cross-encoder reranking is fundamentally unsuitable for this
corpus — it is evidence that (a) this specific model/configuration measurably hurts
Recall@10 on this benchmark, and (b) the qrels must be deepened (judge reranked-promoted
candidates, not just each original method's own top-10) before nDCG@10 — the metric that
matters most for a reranker, whose entire purpose is reordering — can be evaluated with
confidence. Both points argue against proceeding to integration testing on the current
evidence, while leaving open that a corrected evaluation (deeper qrels) or a different
reranker configuration could change the picture.
