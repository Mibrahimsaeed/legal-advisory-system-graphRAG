# Phase 3A — Statute Retrieval Evaluation: Query Expansion vs. Candidate Depth

**Method:** every retrieval run used the existing, unmodified `evaluation/run_retrieval.py` (which calls the production, unmodified `hybrid_search()` in `src/rag_prep/embed_index.py`) — no code was changed, only `--queries` and `--candidates` CLI arguments were varied. Judgments were reused by exact `(query family, source_type, chunk_id)` match from two existing, already-judged files; no label was invented. `evaluation/results/judging_pilot.csv` (baseline Q01–Q05) was used directly, unmodified, and not re-run.

## 1. Verified implementation (before designing anything)

Read directly from `src/rag_prep/embed_index.py`, lines 500–520: `hybrid_search()` builds exactly one dense `Prefetch(using="dense", limit=candidates)` and one sparse `Prefetch(using="bm25", limit=candidates)`, both filtered by `source_type`, fused via Qdrant's built-in `FusionQuery(fusion=Fusion.RRF)`, with a final `limit=k` per source. **The default `candidates` is confirmed 40** (`run_retrieval.py --candidates`, default `40`). This experiment varies only `candidates` (40 vs 100) and query text — the RRF fusion logic itself was not touched.

## 2. Query expansion design — verified legal equivalents only

| Query | Expansion added | Grounding |
|---|---|---|
| Q01 | `dower` → `dower / Haq Mehr` | Confirmed: Family Courts Act §10's own proviso uses "Haq Mehr" as the operative term (verified in this project's Phase 1 diagnostics). |
| Q02 | **none** | No safe equivalent: "custody" and "guardianship" co-occur often in the corpus but are legally distinct concepts (physical care vs. legal authority) — treating them as synonyms would misrepresent the law. |
| Q03 | `maintenance` → `maintenance / support` | Weaker grounding: "support" co-occurs with "maintenance" in Guardians and Wards Act §24 and Family Courts Act §§7/9 — an ordinary-English near-synonym, not a distinct vernacular legal term (no Islamic-jurisprudence term such as "Nafaqa" was found anywhere in this corpus). |
| Q04 | `dowry` → `dowry / bridal gift` | Confirmed: the Dowry and Bridal Gifts (Restriction) Act, 1976 itself titles and defines dowry and bridal gifts together (7 direct occurrences of "bridal gift" in its chunked text). |
| Q05 | **none** | No safe equivalent: "khula" and "talaq" are specific *species* of dissolution, not general equivalents of "dissolution of marriage" itself — appending either would narrow the question's meaning. |

Q02 and Q05 were run **unexpanded** in the "expansion" conditions — their results there are a same-query stability check, not a true expansion test. This is disclosed, not hidden.

## 3. Conditions

| # | Query file | Candidates | Run |
|---|---|---:|---|
| 1 — Baseline | `queries.json` (original) | 40 | **Reused**, not re-run: `retrieval_20261015T201127.jsonl` + `judging_pilot.csv` |
| 2 — Expansion | `queries_statute_expanded.json` | 40 | New: `retrieval_phase3a_cond2_expansion_c40.jsonl` |
| 3 — Depth only | `queries.json` (original) | 100 | New: `retrieval_phase3a_cond3_original_c100.jsonl` |
| 4 — Combined | `queries_statute_expanded.json` | 100 | New: `retrieval_phase3a_cond4_expansion_c100.jsonl` |

Judgment coverage via exact-match reuse against a combined pool of 211 previously-judged `(family, source_type, chunk_id)` entries: condition 2 — 84/90 rows; condition 3 — 88/90; condition 4 — 83/90. Remaining rows are genuinely unjudged and are reported as unavailable, never as irrelevant.

## 4. Statute metrics

| Condition | P@5 | nDCG@10 | MRR | Recall@5 / Recall@10 |
|---|---:|---:|---:|---|
| 1 — Baseline (c=40, original) | 0.360 | 0.527 | 0.5067 | unavailable (no qrels) |
| 2 — Expansion (c=40) | **0.400** | **0.601** | **0.5286** | unavailable |
| 3 — Depth only (c=100, original) | 0.360 | 0.534 | 0.5067 | unavailable |
| 4 — Combined (c=100, expansion) | **0.400** | **0.601** | **0.5286** | unavailable |

**Case-law macro (for reference, not the focus of this task):** P@5=1.0 and MRR=1.0 in all four conditions — case law is unaffected by either variable, as expected since no query text or candidate depth for case law changed.

### Per-query statute P@5

| Query | Baseline | Expansion (c40) | Depth only (c100) | Combined (c100) |
|---|---:|---:|---:|---:|
| Q01 | 0.2 | 0.2 | 0.2 | 0.2 |
| Q02 | 0.6 | 0.6 | 0.6 | 0.6 |
| Q03 | 0.2 | **0.0** | 0.2 | **0.0** |
| Q04 | **0.0** | **0.4** | 0.0 | **0.4** |
| Q05 | 0.8 | 0.8 | 0.8 | 0.8 |

### Per-query statute nDCG@10

| Query | Baseline | Expansion (c40) | Depth only (c100) | Combined (c100) |
|---|---:|---:|---:|---:|
| Q01 | 0.500 | 0.482 | **unavailable** | 0.482 |
| Q02 | 0.758 | 0.758 | 0.758 | 0.758 |
| Q03 | 0.387 | 0.333 | 0.387 | 0.333 |
| Q04 | **0.000** | **0.440** | 0.000 | **0.440** |
| Q05 | 0.992 | 0.992 | 0.992 | 0.992 |

Condition 3's Q01 nDCG@10 is genuinely unavailable — a chunk newly surfaced at candidates=100 within Q01's top-10 has no existing judgment and was not independently labeled in this pass; it is reported as missing, not estimated.

## 5. Candidate-set-membership diagnostic — does increasing depth or expanding vocabulary actually change what enters the pool before fusion?

Checked directly via genuinely unfused dense-only and BM25-only queries (no prefetch/fusion) for the two best-characterized targets:

| Target | Query variant | Dense rank (of 240) | In dense top40 / top100 | BM25 rank (of scored) | In BM25 top40 / top100 |
|---|---|---:|---|---:|---|
| **Q01** → Family Courts Act §10 | Original | 81 | No / **Yes** | 71/74 | No / **Yes** |
| | Expanded ("Haq Mehr") | 50 | No / **Yes** | **4**/75 | **Yes** / Yes |
| **Q04** → Dowry Act §5 | Original | 54 | No / **Yes** | 25/71 | **Yes** / Yes |
| | Expanded ("bridal gift") | 20 | **Yes** / Yes | **4**/74 | **Yes** / Yes |

**Both targets enter the top-100 window in both branches even at baseline** — meaning candidates=100 *does* admit them into the pre-fusion candidate set that candidates=40 excludes. But entering the pool is not sufficient: a direct re-check of Q01's *true fused RRF rank* at candidates=40 with the expanded query (widening only the reported limit, not the candidate pool, per the same methodology validated in this project's earlier Q04 diagnostic) showed the target lands at **fused rank 9 — one position outside the production `k_statute=8` cutoff.** This is why Q01's P@5 stayed flat at 0.2 across every condition despite the expanded query dramatically improving the target's raw BM25 rank (71→4): the improvement is real and large, but not large enough to clear the exact production cutoff for *this specific* query/target pair.

## 6. Recommendation

**Query expansion with verified legal equivalents (Q01/Q04-style) is the more promising lever; raw candidate-depth increases alone are not.** Condition 3 (depth only) reproduced the baseline almost exactly (P@5 0.360 vs 0.360, nDCG@10 0.534 vs 0.527) — both targets were already within the top-100 window even at baseline, so widening the prefetch pool from 40→100 simply didn't change which chunks got fused, confirming the task's own caution that increasing candidate depth does not necessarily improve results. Expansion produced a real, net-positive macro shift (P@5 +0.04, nDCG@10 +0.074, MRR +0.022), but it is **not uniformly positive**: Q04 improved sharply (P@5 0.0→0.4) while Q03's weaker-grounded "support" synonym caused a regression (P@5 0.2→0.0) — consistent with this project's standing finding that rewording is not a free lunch and must be evaluated per query, not assumed beneficial. **Next step:** investigate Q03's regression specifically (what got pushed out of its top-5) before considering any production query-expansion feature, and do not treat candidate-depth tuning as a priority lever based on this evidence.

## 7. Files created

`evaluation/queries_statute_expanded.json`, `evaluation/phase3a_additional_metrics.py`, retrieval/judging/metrics files for 3 new conditions (`*_cond2_expansion_c40*`, `*_cond3_original_c100*`, `*_cond4_expansion_c100*`), `evaluation/results/metrics_phase3a_baseline_mrr.json`, this report, and its JSON counterpart `evaluation/results/phase3a_statute_retrieval_eval.json`.

## 8. Scope confirmation

No production file, chunk, embedding, Qdrant collection/index, or existing evaluation artifact (`queries.json`, `judging_pilot.csv`, `metrics_pilot.json`, the earlier L1/L2 experiment files) was modified or overwritten — all new artifacts use unique filenames. `git diff --stat` is empty.
