# Query-Phrasing Experiment — Execution Report

**Status:** experiment executed through retrieval and metric calculation; **manual relevance and
answer-bearing judging is still pending**. No production code, index, embeddings, or frozen baseline
artifacts were modified.

## 1. Objective and protocol

Test whether phrasing a query with (a) more domain vocabulary, or (b) an explicitly named governing
Act, changes retrieval quality for the 5 pilot queries, using the existing hybrid retrieval stack
unmodified. Two controlled rewrite levels were applied to each of the 5 original queries:

- **L1 — domain-vocabulary expansion**: adds terms a knowledgeable person would use for the same
  question, without naming a statute.
- **L2 — named-authority expansion**: additionally names the specific Act(s) believed relevant. This
  is a **ceiling test**, not a proposed production query-rewriting policy.

## 2. Exact query variants tested

| query_id | category (verified against `queries.json`, not assumed) | query |
|---|---|---|
| Q01-L1 | `khula_dower` | Under Muslim family law in Pakistan, can a wife obtain khula (judicial divorce) without returning her dower (haq mehr) to her husband? |
| Q01-L2 | `khula_dower` | Under the Family Courts Act, 1964 and the Muslim Family Laws Ordinance, 1961, can a wife obtain khula without returning dower? |
| Q02-L1 | `child_custody` | Which court in Pakistan has jurisdiction to decide disputes over the custody and guardianship of minor children? |
| Q02-L2 | `child_custody` | Under the Guardians and Wards Act, 1890 and the Family Courts Act, 1964, which court has jurisdiction over child custody disputes? |
| Q03-L1 | `child_maintenance` | Under Muslim family law in Pakistan, is a father responsible for maintaining and supporting his minor children? |
| Q03-L2 | `child_maintenance` | Under the Muslim Family Laws Ordinance, 1961 and the Guardians and Wards Act, 1890, is a father legally responsible for the maintenance and support of his children? |
| Q04-L1 | `dowry_recovery` | After a Muslim marriage ends, what is a wife's right to recover her dowry and bridal gifts, and what must be shown to establish ownership? |
| Q04-L2 | `dowry_recovery` | Under the Dowry and Bridal Gifts (Restriction) Act, 1976, what evidence is required to recover dowry and bridal gifts after separation? |
| Q05-L1 | `marriage_dissolution` | What are the legal grounds on which a Muslim wife may seek dissolution of her marriage in Pakistan? |
| Q05-L2 | `marriage_dissolution` | Under the Dissolution of Muslim Marriages Act, 1939, what are the legal grounds for dissolution of a Muslim marriage? |

**Discrepancy found and corrected before proceeding (per Step 1's instruction to verify, not assume):**
the task's draft used `"category": "family_law"` for all 10 rows. Reading `evaluation/queries.json`
directly showed the 5 original queries actually use distinct, specific category values
(`khula_dower`, `child_custody`, `child_maintenance`, `dowry_recovery`, `marriage_dissolution`), not a
shared `family_law` value. Per the task's own fallback instruction ("use the exact existing category
value... if it differs"), each variant was written with its original family's real category instead.
Query IDs and query text were preserved exactly as given, byte for byte.

## 3. Baseline parameters and manifest information (verified, not assumed)

Read directly from `evaluation/results/retrieval_20261015T201127.jsonl.meta.json`:

```json
{
  "collection": "pk_legal", "dense_model": "BAAI/bge-m3", "sparse_model": "Qdrant/bm25",
  "per_source_limit": {"case": 10, "statute": 8}, "candidates_per_source": 40,
  "query_count": 5, "result_counts": {"case": 50, "statute": 40},
  "excluded_from_metrics": ["constitution"]
}
```

CLI flags on `run_retrieval.py`, `judge_results.py`, and `calculate_metrics.py` were read directly from
the current source (`grep add_argument`) immediately before use and matched the planned commands
exactly — no flag name or default differed from what was assumed, so the commands below were run
exactly as specified, with no substitutions.

## 4. Files created and commands executed

| Step | Command | Output |
|---|---|---|
| Variant file | *(written directly, schema-validated against `run_retrieval.py`'s own `load_queries()`)* | `evaluation/queries_expansion_experiment.json` |
| Retrieval | `python3 evaluation/run_retrieval.py --queries evaluation/queries_expansion_experiment.json --out evaluation/results/retrieval_expansion_experiment.jsonl --k-case 10 --k-statute 8 --candidates 40` | `retrieval_expansion_experiment.jsonl` (180 rows) + `.meta.json` |
| Judging sheet (canonical, unmodified script) | `python3 evaluation/judge_results.py --in evaluation/results/retrieval_expansion_experiment.jsonl --out evaluation/results/judging_expansion_experiment.csv` | `judging_expansion_experiment.csv` (180 rows, `relevance` fully blank) |
| Label-reuse working copy | *(separate script, read-only exact-match merge — see §7)* | `judging_expansion_experiment_prefilled.csv` |
| Manual-review copy with `answer_bearing` | *(separate script, additive column only — see §7)* | `judging_expansion_experiment_manual_review.csv` |
| Metrics (canonical, all-blank input) | `python3 evaluation/calculate_metrics.py --judged evaluation/results/judging_expansion_experiment.csv --out evaluation/results/metrics_expansion_experiment.json` | `metrics_expansion_experiment.json` |
| Metrics (supplementary, partially-reused-label input) | `python3 evaluation/calculate_metrics.py --judged evaluation/results/judging_expansion_experiment_prefilled.csv --out evaluation/results/metrics_expansion_experiment_partial.json` | `metrics_expansion_experiment_partial.json` |

**Retrieval completed successfully**: 10/10 queries processed, 180 rows written (100 case + 80 statute,
= 10 queries × 10 case + 8 statute each, matching the baseline's per-source limits exactly).

The original 5 baseline queries were not re-run, and no existing evaluation artifact
(`evaluation/queries.json`, `retrieval_20261015T201127.jsonl`, `judging_pilot.csv`,
`metrics_pilot.json`) was overwritten — verified by checksum before and after this experiment; all four
are byte-identical.

## 5. Per-query, per-source comparison — baseline vs. L1 vs. L2

**Only one cell currently has a complete enough label set to compute a real metric.** The table below
reports it, and marks every other cell `unavailable` rather than inventing a number — this is not a
placeholder choice, it is `calculate_metrics.py`'s own, unmodified behavior given the current judging
state (§6 explains exactly why).

| Query family | Source | Baseline P@5 | L1 P@5 | L2 P@5 | Baseline nDCG@10 | L1 nDCG@10 | L2 nDCG@10 |
|---|---|---|---|---|---|---|---|
| Q01 | case | 1.0 | unavailable | unavailable | 0.990 | unavailable | unavailable |
| Q01 | statute | 0.2 | unavailable | unavailable | 0.500 | unavailable | unavailable |
| Q02 | case | 1.0 | unavailable | unavailable | 0.852 | unavailable | unavailable |
| Q02 | statute | 0.6 | unavailable | unavailable | 0.758 | unavailable | unavailable |
| Q03 | case | 1.0 | unavailable | unavailable | 1.000 | unavailable | unavailable |
| Q03 | statute | 0.2 | unavailable | unavailable | 0.387 | unavailable | unavailable |
| Q04 | case | 1.0 | unavailable | unavailable | 0.803 | unavailable | unavailable |
| Q04 | statute | 0.0 | unavailable | unavailable | 0.000 | unavailable | unavailable |
| Q05 | case | 1.0 | unavailable | unavailable | 0.778 | unavailable | unavailable |
| Q05 | statute | 0.8 | unavailable | **0.600** | 0.992 | unavailable | unavailable |

Baseline values are read directly from the existing, untouched `metrics_pilot.json`. The single L2
value available (Q05-L2, statute, P@5 = 0.600) comes from `metrics_expansion_experiment_partial.json`
and rests entirely on **reused baseline labels** (ranks 1–5 for that query/source all happened to
match chunks the baseline judge already scored for Q05) — it did not require any new human judgment,
but it also means **no new chunk was actually reviewed for this cell**. Its nDCG@10 is still
`unavailable` because ranks 7–8 in that same result set are not yet labeled.

## 6. Missing-judgment / unavailable-metric status

- `judging_expansion_experiment.csv` (the canonical, unmodified `judge_results.py` output): **all 180
  rows have a blank `relevance`**. Running `calculate_metrics.py` against it produces exactly one
  outcome for every query/source cell: `unavailable`, with the precise missing rank(s) named — e.g.
  `precision_at_5: missing relevance label(s) at rank(s) [1, 2, 3, 4, 5]`. This is the script behaving
  correctly, not failing; it is preserved exactly as instructed, and reported here rather than worked
  around.
- `judging_expansion_experiment_prefilled.csv` (separate working copy, reusing baseline labels only
  where `chunk_id` + `source_type` + original query family all matched exactly): **49 of 180 rows
  (27%) were reused from `judging_pilot.csv`; 131 rows (73%) remain blank** and require a human to
  judge them. Breakdown by variant:

  | query_id | reused | still blank | query_id | reused | still blank |
  |---|---|---|---|---|---|
  | Q01-L1 | 9 | 9 | Q04-L1 | 0 | 18 |
  | Q01-L2 | 5 | 13 | Q04-L2 | 0 | 18 |
  | Q02-L1 | 6 | 12 | Q05-L1 | 9 | 9 |
  | Q02-L2 | 2 | 16 | Q05-L2 | 11 | 7 |
  | Q03-L1 | 5 | 13 | | | |
  | Q03-L2 | 2 | 16 | | | |

  **Notable finding on its own:** Q04-L1 and Q04-L2 have **zero** reused labels (0 of 18 rows each) —
  every chunk they retrieved is one the baseline Q04 never surfaced at all. This is independent,
  structural evidence (not a judged relevance claim) that the Q04 rewrites are retrieving a
  genuinely different result set from the baseline's all-Divorce-Act results, consistent with the
  prior diagnostic's finding — but whether that different set is actually *better* still requires the
  pending manual judgment; this fact alone does not establish it.
- Even with reuse, only **one** query/source cell (Q05-L2, statute) ended up with a complete top-5
  label set, and no cell had a complete top-10 set — so `metrics_expansion_experiment_partial.json`
  reports `n/a` for nDCG@10 everywhere and Recall@10 everywhere (no qrels file exists, consistent with
  every prior run in this project).
- **No missing label was treated as irrelevant, and no score was fabricated, anywhere in this
  experiment.** Both metrics runs (`metrics_expansion_experiment.json` and `..._partial.json`) are the
  unmodified script's own direct output.

## 7. How label reuse and the answer-bearing column were produced

Both are read-only, additive artifacts, not changes to any protected script:

- **Label reuse** (`judging_expansion_experiment_prefilled.csv`): a short script read
  `judging_pilot.csv`, built an index of `(query_id, source_type, chunk_id) -> relevance` for every
  baseline row that already has a non-blank label, then — for each variant row — looked up
  `(original_family, source_type, chunk_id)` (e.g. `Q04-L2` maps to family `Q04`) and copied the label
  over only on an exact triple match. No fuzzy matching, no text-similarity matching, and no
  rank/score-based inference was used. `judging_pilot.csv` itself was not opened for writing at any
  point — confirmed unchanged by checksum (§4).
- **Answer-bearing column** (`judging_expansion_experiment_manual_review.csv`): built from the
  prefilled copy by appending one new column, `answer_bearing`, **left blank for every row** — it was
  not inferred, estimated, or defaulted. Rows were reordered (not re-labeled) to put the Q03 and Q04
  families first, per the task's instruction to prioritize them, since the coverage audits already
  found these are the two families most likely to have topically-relevant but non-dispositive chunks.
  `judge_results.py` and `calculate_metrics.py` were not modified to add this column — it exists only
  in this separate manual-review file, and nothing reads it back into any metric.

## 8. Qualitative answer-bearing findings

**None yet** — `answer_bearing` is unfilled pending human review, exactly as required ("leave blank
until a human reviews the chunk"). The file `judging_expansion_experiment_manual_review.csv` is
prepared and ordered (Q03, Q04 first) for that review to happen next.

## 9. Limitations and recommended next steps

Explicitly disclosed, as required:

- **Only 5 query families** underlie this experiment; all comparisons above (and the one in §5
  especially) are exploratory/descriptive. No significance test was run or should be inferred from
  these numbers.
- **Q04 is not a blind test.** Its L1/L2 variants were informed by an earlier read-only retrieval
  diagnostic that had already probed similar phrasings for Q04 specifically. An improvement confirmed
  for Q04 is weaker evidence of a generalizable fix than the same result on Q01, Q02, Q03, or Q05.
- **Better topical relevance is not the same as fully answering the question.** The Q03/Q04 coverage
  audit already found provisions that are clearly on-topic (e.g. Dowry Act §5, Guardians and Wards Act
  §24) without stating the specific fact a question asks for. The `answer_bearing` column exists
  precisely to keep this distinction visible once judging happens, rather than reading a Precision@5
  improvement alone as "the system now answers this question."
- **L2 (named-authority) queries are a ceiling test, not a production query-rewriting policy.** Real
  users will not typically name a specific Act and year. A strong L2 result only shows that the index
  and embeddings can surface the right content when given perfect vocabulary — it does not by itself
  justify shipping a rewriter that guesses statute names.

**Recommended next step:** complete the manual judging pass on
`evaluation/results/judging_expansion_experiment_manual_review.csv` (131 blank `relevance` cells,
all 180 blank `answer_bearing` cells, Q03/Q04 ordered first), then re-run
`python3 evaluation/calculate_metrics.py --judged evaluation/results/judging_expansion_experiment_manual_review.csv --out evaluation/results/metrics_expansion_experiment.json`
(overwriting only this experiment's own metrics file, never the baseline) to get the full L1/L2
comparison this report's §5 table could only partially fill in today.

## 10. Scope confirmation

- `src/rag_prep/embed_index.py`: not modified (confirmed via `git diff --stat`, empty).
- Qdrant collection, embeddings, indexed points: no write operation was performed; every Qdrant call
  made across this and prior diagnostics in this project was a read-only `query_points`/`points/count`/
  `points/scroll` call.
- No ingestion, embedding generation, or reindexing was run.
- `evaluation/queries.json`, `retrieval_20261015T201127.jsonl`, `judging_pilot.csv`, and
  `metrics_pilot.json` are byte-identical to before this experiment (checksummed before and after).
- `run_retrieval.py`, `judge_results.py`, and `calculate_metrics.py` were not modified.
- No relevance or answer-bearing label was fabricated, inferred from rank/score, or defaulted; the 49
  reused labels are exact carry-overs of a human's own prior judgment on the identical chunk for the
  same underlying question.
- **No scope deviation or failure occurred.** The only deviation from the task's literal draft was the
  `category` field correction in §2, which the task itself anticipated and instructed how to handle.
