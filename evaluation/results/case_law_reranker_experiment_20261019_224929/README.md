# Offline Cross-Encoder Reranker Experiment — Case Law

## Scope

Evaluation-only experiment testing whether reordering the existing production hybrid
retrieval's case-law top-40 candidates with a local cross-encoder (`BAAI/bge-reranker-v2-m3`)
improves ranking quality against the frozen expanded qrels. **No production code, fusion
weights, Qdrant data, embeddings, or GraphRAG components were touched.** All new files are
confined to this directory.

## Model selection

The task suggested `BAAI/bge-reranker-v2-m3`. It was not cached locally (would require a
~2.3GB download over the network). Per the task's explicit "stop and ask before downloading"
instruction, I asked for permission before downloading; the user approved downloading this
specific model. It downloaded successfully and ran correctly on this machine's Apple Silicon
MPS backend (fp32, 9.0s load from cache after the one-time download, smoke-tested with a
clearly-relevant query/passage pair scoring 0.999).

## Commands (in order)

```bash
export PYTHONDONTWRITEBYTECODE=1
NEW_DIR=evaluation/results/case_law_reranker_experiment_20261019_224929

# 1. Load frozen hybrid top-40 candidates, download+run the cross-encoder, write baseline
#    and reranked rankings, and record latency/resource stats.
python3 $NEW_DIR/run_reranker_experiment.py

# 2. Compute baseline vs. reranked metrics against the frozen expanded qrels
python3 $NEW_DIR/compute_metrics.py

# 3. Paired bootstrap significance analysis
python3 $NEW_DIR/significance_analysis.py
```

## Headline result

**NOT PROMISING.** Reranking produces a statistically significant, broad-based REGRESSION
in Recall@10 (full 50-query sample, 39/50 queries worsened) and no reliable improvement on
any other metric. A nominally significant nDCG@10 "improvement" is based on only 7 of 50
queries — most queries' reranked top-10 contains candidates that fall outside the original
qrels' judging depth (top-10 of each ORIGINAL method only) and are therefore unjudged, not
irrelevant — so nDCG@10 could not be honestly computed for 43 of 50 queries. See
`comparison_report.md` for full detail, including why this 7-query sample is not
representative (its baseline nDCG@10 is notably below the full-50 average).

## Key limitation this experiment surfaced

The expanded qrels (from the prior `case_law_qrels_expansion_...` task) were judged to a
depth matched to the THREE ORIGINAL retrieval methods' own top-10 rankings. A reranker, by
design, reshuffles candidates — so it routinely promotes previously low-ranked (11–40),
unjudged candidates into the top-5/top-10. This is a structural gap in the existing qrels
for evaluating ANY reranker, not specific to this model, and should be addressed (judge the
union of original-method top-10 AND reranked top-10) before trusting a future nDCG@10/P@5
comparison on this benchmark.

## Reproducibility

All inputs are frozen, already-published files (read-only): `retrieval_hybrid_case.jsonl`
(original hybrid rankings) and `judgment_ledger.jsonl` (expanded qrels). Given the same
model, device, and these frozen inputs, reranker scores are deterministic (no sampling in
cross-encoder inference); the only randomness in this experiment is the bootstrap
resampling (seed=42, 5,000 resamples, recorded in `significance_analysis.json`).

## What was NOT done

- No production integration, no change to `src/rag_prep/embed_index.py` or any fusion
  weight.
- No fine-tuning of the reranker on this benchmark's queries or qrels (it never saw a
  relevance label during inference).
- No re-querying of Qdrant — the candidate set is exactly the frozen hybrid top-40 already
  on disk.
- No repair of the qrels-depth gap identified above — flagged as a recommendation for a
  follow-up task, not fixed here (would itself require new manual judging, out of this
  experiment's scope).
