# Retrieval Evaluation Pipeline

Evaluates the existing Hybrid RAG retrieval stack (`src/rag_prep/embed_index.py`,
collection `pk_legal`) against a small, fixed pilot query set. This pipeline
reuses `hybrid_search()`, `load_dense()`, `load_sparse()`, and `connect()`
unmodified — it does not re-embed, re-index, or alter the Qdrant collection,
schema, or retrieval logic in any way.

**Scope note:** five pilot queries are nowhere near enough to conclude the
retrieval is "good" or "bad." This pipeline exists to produce a reliable,
repeatable way to *measure* retrieval quality — the judgment itself requires
a much larger, more representative query set evaluated the same way.

## Files

- `queries.json` — the fixed pilot query set (`query_id`, `category`, `query`).
- `run_retrieval.py` — runs every query through the live retrieval stack, exports raw results.
- `judge_results.py` — converts a raw export into a CSV for manual relevance labeling.
- `calculate_metrics.py` — computes Precision@5, nDCG@10, Recall@10 from a labeled CSV.
- `results/` — output directory for exports, judging sheets, and metric reports (not committed).

## Prerequisites

- Qdrant reachable (default `http://localhost:6333`) with the `pk_legal` collection already populated.
- The repo's existing Python environment (no new dependencies are required — `qdrant_client`,
  `fastembed`, and `sentence_transformers` are already installed and used by `embed_index.py`).
- Run all commands from the repository root.

## 1. Validate the query file

`run_retrieval.py` validates `queries.json` on every run (non-empty array, required
fields present, no duplicate `query_id`) and fails loudly on any problem. To check it
standalone without hitting Qdrant:

```bash
python3 -c "
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('eval_run_retrieval', Path('evaluation/run_retrieval.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
qs = m.load_queries(Path('evaluation/queries.json'))
print(f'{len(qs)} queries OK:', [q['query_id'] for q in qs])
"
```

(This loads the script by file path rather than `import evaluation.run_retrieval` — the
repo already has an unrelated `src/evaluation/` package on `sys.path`, which would shadow
this directory under a dotted import. `python3 evaluation/run_retrieval.py` itself is
unaffected by this, since it runs as a script rather than importing itself as a package.)

## 2. Run baseline retrieval

```bash
python3 evaluation/run_retrieval.py
```

Defaults: 10 case-law + 8 statute results per query, Constitution not fetched,
40 candidates per source before RRF fusion. Writes:

- `evaluation/results/retrieval_<timestamp>.jsonl` — one JSON record per retrieved
  chunk (`query_id`, `query`, `source_type`, `rank`, `point_id`, `chunk_id`,
  `context_header`, `canonical_text`, `rrf_score`). Text fields are never truncated.
- `evaluation/results/retrieval_<timestamp>.jsonl.meta.json` — run metadata
  (collection, model names, Qdrant URL, per-source limits, timestamp, result counts,
  `excluded_from_metrics: ["constitution"]`).

Useful options:

```bash
python3 evaluation/run_retrieval.py \
  --queries evaluation/queries.json \
  --out evaluation/results/retrieval_pilot.jsonl \
  --qdrant http://localhost:6333 \
  --k-case 10 --k-statute 8 --k-constitution 0 \
  --candidates 40
```

Pass `--k-constitution 5` (or any value > 0) to additionally export Constitution
hits for separate inspection — they are tagged and still excluded from every
metric regardless.

## 3. Build the manual judging sheet

```bash
python3 evaluation/judge_results.py \
  --in evaluation/results/retrieval_pilot.jsonl \
  --out evaluation/results/judging_pilot.csv
```

Produces a CSV with one row per retrieved case-law/statute chunk and a blank
`relevance` column. Pass `--include-constitution` to also carry Constitution
rows into the sheet for reference (they still won't be scored).

## 4. Fill in relevance labels

Open `evaluation/results/judging_pilot.csv` in a spreadsheet editor and fill in
`relevance` for every row using:

| Label | Meaning |
|---|---|
| `0` | Irrelevant |
| `1` | Partially relevant |
| `2` | Highly relevant |

Leave a cell blank only if you genuinely have not judged it yet —
`calculate_metrics.py` treats a blank as "not yet judged" and will report that
query/source's metrics as unavailable rather than guessing. Do not fill labels
from the `rrf_score` column; judge each `canonical_text` on its legal merits.

(Optional, for Recall@10 only) Build a `qrels.json` file listing the complete
known set of relevant chunk IDs per query/source, independent of what this run
happened to retrieve:

```json
{
  "Q01": { "case": ["<chunk_id>", "..."], "statute": ["<chunk_id>", "..."] }
}
```

Without this file, Recall@10 is reported as `unavailable` for every query —
by design, since the top-10 retrieved results can never be assumed to be the
complete relevant set.

## 5. Calculate metrics

```bash
python3 evaluation/calculate_metrics.py \
  --judged evaluation/results/judging_pilot.csv \
  --qrels evaluation/results/qrels.json \
  --out evaluation/results/metrics_pilot.json
```

(`--qrels` is optional; omit it if no reference set exists yet.)

Prints a per-query and macro-average table for `case` and `statute` separately,
and (if `--out` is given) writes the full per-query + macro breakdown as JSON.

## 6. Interpreting the output

- **Precision@5**: fraction of the top 5 results labeled `1` or `2`. Always
  divided by 5, even if fewer than 5 results were retrieved for that query/source.
- **nDCG@10**: graded-relevance ranking quality over the top 10 (gain `2^relevance - 1`,
  log2 discount). `1.0` = ideal ordering for the labels given; `0.0` = all labeled irrelevant.
- **Recall@10**: fraction of the *complete* known-relevant set found in the top 10.
  Reported as `unavailable` whenever no `qrels.json` entry exists for that query/source —
  this is expected and correct, not a bug, until a real reference set is built.
- A metric value of `n/a` in the printed table, or `null` with a `*_note` in the
  JSON output, means the metric could not be computed honestly for that row —
  usually a blank relevance label within the metric's ranking window. It is never
  silently scored as 0.
- Macro averages are computed only over queries where the metric was available;
  excluded query IDs are listed alongside the average (`*_excluded_queries`) so the
  coverage gap stays visible.

## Running the tests

```bash
python3 -m pytest tests/test_evaluation_metrics.py -v
```
