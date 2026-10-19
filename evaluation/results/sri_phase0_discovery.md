# SRI Phase 0 — Repository and Evaluation Discovery

**Initial `git status --short`:** 50 untracked files (all pre-existing evaluation artifacts from earlier sessions; `git diff --stat` empty — no tracked file modified).

## Confirmed configuration (read-only checks, re-verified fresh)

- **Qdrant collection:** `pk_legal`, `points_count = 26,678`. Dense vector `"dense"`, size 1024, distance `Cosine`. Sparse vector `"bm25"`, modifier `idf`. Statute `source_type` count: **240** (matches every prior report this project).
- **`hybrid_search()`** (`src/rag_prep/embed_index.py:500-520`): one dense `Prefetch(using="dense", limit=candidates)` + one sparse `Prefetch(using="bm25", limit=candidates)`, both filtered by `source_type`, fused via Qdrant's built-in `FusionQuery(fusion=Fusion.RRF)`, final `limit=k` per source. **Default `candidates=40`** (confirmed via `run_retrieval.py --candidates`, default 40). Default final cutoff for statutes in every existing report: `k_statute=8`.
- **No existing code path exposes dense-only or BM25-only rankings directly** — production only ever calls the fused `hybrid_search()`. Branch-level diagnostics in this project (Phase 1 series) were produced by an isolated, evaluation-only script issuing the *same* `client.query_points(..., using="dense", ...)` / `using="bm25"` calls `hybrid_search()` itself makes, just without the `prefetch`/`FusionQuery` wrapper — reusing `connect()`/`load_dense()`/`load_sparse()`/`encode_dense()` unmodified. This experiment reuses that exact, already-validated pattern rather than inventing a new one.
- **Metric scripts:** `evaluation/calculate_metrics.py` computes Precision@5, nDCG@10 (graded 0/1/2, gain `2^rel-1`, log2 discount), and Recall@10 — the last requires a `--qrels` file (`{"Q01": {"statute": ["chunk_id", ...]}}`) and reports "unavailable" without one, never approximating from top-k. `evaluation/phase3a_additional_metrics.py` adds MRR and Recall@5 by calling `calculate_metrics.py`'s own `recall_at_k()` unmodified with `k=5`. **No qrels file has ever existed in this project before now.**
- **Existing filenames inspected to avoid collision:** 60 files under `evaluation/results/` from prior sessions (`judging_pilot.csv`, `metrics_pilot.json`, the `phase1_*`, `phase2a_*`, `phase2b_*`, `phase3a_*`, `statute_benchmark_audit.*` series, etc.). This workflow uses a distinct `sri_phase{0,1,2,3,4}_*` filename prefix throughout to avoid any ambiguity with those unrelated, already-completed investigations.

Proceeding to Phase 1.
