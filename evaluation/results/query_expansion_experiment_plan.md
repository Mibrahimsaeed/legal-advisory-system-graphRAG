# Query-Phrasing Experiment Plan — Design Only, Not Implemented

**Status: design document.** No queries were run, no files other than this plan were created, and
`src/rag_prep/embed_index.py`, the Qdrant index, embeddings, and existing evaluation labels were not
touched or re-read beyond what was already established in prior read-only audits
(`evaluation/results/statute_coverage_audit.md`, `.../statute_ingestion_indexing_audit.md`,
`.../q04_retrieval_diagnostic.md`).

## 1. What this builds on

`evaluation/queries.json` holds the 5 fixed pilot queries (Q01–Q05). The existing scripts —
`evaluation/run_retrieval.py`, `evaluation/judge_results.py`, `evaluation/calculate_metrics.py` — are
all designed to be reusable unmodified against any query file: `run_retrieval.py` takes `--queries
<path>` and `--out <path>` as CLI arguments, so pointing it at a *different* query file produces a
*separate* output, never overwriting the baseline. This experiment is designed entirely around that
existing capability — no script changes are proposed anywhere below.

The `q04_retrieval_diagnostic.md` diagnostic already showed, with live (but code-unmodified, read-only)
queries against the existing index, that rephrasing Q04 to name "dowry and bridal gifts" or the exact
Act title shifts the correct chunks into the retrieved set, and that a middle-ground ownership/return
framing (without naming the Act) helps but ranks §5 lower than the named-Act version. This experiment
plan generalizes that observation into a controlled, repeatable design across all 5 pilot queries, not
just Q04.

## 2. Experimental design

### 2.1 Two rewrite levels, applied uniformly to all 5 queries

To isolate *why* phrasing affects retrieval (pure vocabulary gap vs. something deeper), each original
query gets exactly two controlled rewrites, at two distinct, consistently-applied levels:

- **Level 1 — domain-vocabulary expansion.** Add the general legal/domain terms a knowledgeable
  person would plausibly use when asking the *same* question (e.g. "khula," "dower/haq mehr,"
  "guardianship," "dowry and bridal gifts") without naming any specific statute or section. This
  tests whether the baseline query is simply missing vocabulary that exists in the corpus.
- **Level 2 — named-authority expansion.** Additionally name the specific Act(s) believed relevant.
  This is a deliberate **ceiling test**, not a proposed production rewrite — most real users will
  never name a statute. It tells us the best case achievable by vocabulary alone, and how much of
  the gap (if any) remains even when the "right" statute is named explicitly.

Keeping both levels **identical in structure across all 5 queries** is what makes this a controlled
experiment rather than five ad hoc rewrites: the only thing that varies query-to-query is content, not
rewriting intensity or style.

### 2.2 Rewriting protocol (to keep rewrites "legally faithful" and prevent bias)

1. A rewrite must ask the **same legal question** — same facts, same relief/answer being sought. It
   may never narrow, broaden, or shift the question to one more likely to be answered well.
2. Level 1 may add general domain vocabulary and synonyms only — never a statute name, section
   number, or year.
3. Level 2 may additionally name an Act — but still may not quote or paraphrase a specific section's
   operative text (that would leak the answer into the query rather than testing retrieval).
4. Rewrites are drafted from the query's plain legal meaning and general domain knowledge, not from
   inspecting which chunks the baseline retrieved for that query — except where noted (see §2.4
   limitation for Q04).
5. Each rewrite gets a distinct `query_id` so it is never confused with, or silently merged into, the
   original 5 pilot queries' results.

### 2.3 Proposed variant set (10 new queries + the existing 5 originals = 15 total)

| query_id | Level | Text | Rationale |
|---|---|---|---|
| Q01 | — (baseline, unchanged) | "Can a wife obtain khula without returning dower?" | existing pilot query |
| Q01-L1 | Vocabulary | "Under Muslim family law in Pakistan, can a wife obtain khula (judicial divorce) without returning her dower (haq mehr) to her husband?" | adds the domain synonyms already present in the corpus (Family Courts Act Schedule: "Dissolution of marriage, including Khula"; MFLO: "Dower") without naming either Act |
| Q01-L2 | Named authority | "Under the Family Courts Act, 1964 and the Muslim Family Laws Ordinance, 1961, can a wife obtain khula without returning dower?" | names the two Acts whose provisions the prior coverage audit found most relevant |
| Q02 | — (baseline, unchanged) | "Which court has jurisdiction over child custody disputes?" | existing pilot query |
| Q02-L1 | Vocabulary | "Which court in Pakistan has jurisdiction to decide disputes over the custody and guardianship of minor children?" | adds "guardianship," matching the Guardians and Wards Act's own vocabulary, without naming it |
| Q02-L2 | Named authority | "Under the Guardians and Wards Act, 1890 and the Family Courts Act, 1964, which court has jurisdiction over child custody disputes?" | — |
| Q03 | — (baseline, unchanged) | "Is a father legally responsible for child maintenance?" | existing pilot query; prior audit found this is partly a genuine content gap (no clean "father/child" clause), so this family is also a test of how far rephrasing alone can go when the corpus content itself is thin |
| Q03-L1 | Vocabulary | "Under Muslim family law in Pakistan, is a father responsible for maintaining and supporting his minor children?" | adds "supporting," matching Guardians and Wards Act §24's actual wording ("must look to his support, health and education") without naming it |
| Q03-L2 | Named authority | "Under the Muslim Family Laws Ordinance, 1961 and the Guardians and Wards Act, 1890, is a father legally responsible for the maintenance and support of his children?" | — |
| Q04 | — (baseline, unchanged) | "What evidence is required to recover dowry articles after separation?" | existing pilot query |
| Q04-L1 | Vocabulary | "After a Muslim marriage ends, what is a wife's right to recover her dowry and bridal gifts, and what must be shown to establish ownership?" | corresponds to the "Query C" ownership/return framing already informally tested |
| Q04-L2 | Named authority | "Under the Dowry and Bridal Gifts (Restriction) Act, 1976, what evidence is required to recover dowry and bridal gifts after separation?" | corresponds to the "Query B" named-Act framing already informally tested |
| Q05 | — (baseline, unchanged) | "What are the legal grounds for dissolution of a Muslim marriage?" | existing pilot query; already scores well (statute P@5=0.8) — kept as a **control**: a well-performing baseline should not change much under rewriting, which validates that the experiment isn't simply inflating every score |
| Q05-L1 | Vocabulary | "What are the legal grounds on which a Muslim wife may seek dissolution of her marriage in Pakistan?" | — |
| Q05-L2 | Named authority | "Under the Dissolution of Muslim Marriages Act, 1939, what are the legal grounds for dissolution of a Muslim marriage?" | — |

### 2.4 Known limitation, disclosed up front

Q04's two variants are **not** blind in the same sense as Q01/Q02/Q03/Q05's: the earlier read-only
diagnostic already explored phrasings close to these for Q04 specifically, before this plan was
written. Any improvement confirmed for Q04 is therefore weaker evidence of a generalizable fix than an
improvement confirmed for one of the other four query families, which are being tested for the first
time here. The write-up of results must report this asymmetry explicitly rather than treating all 5
query families as equally "blind" tests.

## 3. Procedure — reusing existing scripts unmodified

All commands below use the project's existing CLI flags exactly as implemented; none require any code
change.

1. **Freeze the baseline.** Do not re-run retrieval for Q01–Q05 — the existing
   `evaluation/results/retrieval_20261015T201127.jsonl` and its judged companion
   `evaluation/results/judging_pilot.csv` / `evaluation/results/metrics_pilot.json` already constitute
   the baseline and should be reused as-is, both to avoid redundant model/Qdrant calls and because
   dense/sparse encoding of identical text is deterministic — a re-run would not change the baseline
   numbers, only cost time.
2. **Write the 10 variant queries** into a *new* file, e.g. `evaluation/queries_expansion_experiment.json`,
   in the exact same schema as `queries.json` (`query_id`, `category`, `query`) — `run_retrieval.py`'s
   `load_queries()` validates only that these three fields are present and that `query_id` is unique
   across the file; it does not require matching the original 5 IDs, so the 10 variants can live in
   their own file without colliding with the originals.
3. **Run retrieval for the variants only**, pointing at the new file and a new output path:
   ```bash
   python3 evaluation/run_retrieval.py \
     --queries evaluation/queries_expansion_experiment.json \
     --out evaluation/results/retrieval_expansion_experiment.jsonl \
     --k-case 10 --k-statute 8 --candidates 40
   ```
   Note `--k-case`, `--k-statute`, and `--candidates` are passed with the **same values as the
   baseline run's defaults** — this is the "keep retrieval parameters unchanged" requirement made
   explicit and auditable, not left to an unstated default that could silently drift.
4. **Build the judging sheet for the variants**, same script, unmodified:
   ```bash
   python3 evaluation/judge_results.py \
     --in evaluation/results/retrieval_expansion_experiment.jsonl \
     --out evaluation/results/judging_expansion_experiment.csv
   ```
5. **Judge the variant results** using the *same rubric* already used for the baseline (0 = irrelevant,
   1 = partially relevant, 2 = highly relevant) — see §4 for how to keep this consistent with the
   baseline's judgments rather than a fresh, incomparable standard.
6. **Calculate metrics for the variants**, same script, unmodified:
   ```bash
   python3 evaluation/calculate_metrics.py \
     --judged evaluation/results/judging_expansion_experiment.csv \
     --out evaluation/results/metrics_expansion_experiment.json
   ```
7. **Compare** the variant metrics against the already-existing `evaluation/results/metrics_pilot.json`
   baseline — see §6. This comparison step is a small, separate analysis (a short read-only script or
   even a manual table, since `calculate_metrics.py` already reports per-query results in both JSON
   files) — not a pipeline change, and not implemented as part of this plan.

No step above modifies `embed_index.py`, re-runs ingestion/indexing, or touches the existing `relevance`
labels already recorded in `judging_pilot.csv`.

## 4. Keeping relevance judgments consistent with the baseline

A rewritten query may retrieve a *different* chunk than the original did, so new judgments are
unavoidably needed for any chunk not already labeled in `judging_pilot.csv`. To keep the comparison
valid rather than introducing a second, incompatible standard:

- **Relevance is judged against the original query family's legal question, not the variant's exact
  wording.** A chunk is relevant to "Q04" as a legal information need regardless of whether Q04,
  Q04-L1, or Q04-L2 happened to retrieve it — the underlying question being asked is the same by
  design (per the rewriting protocol in §2.2, rule 1). The judging sheet should record which `query_id`
  (and therefore which rewrite level) retrieved each row, but the relevance label itself should reflect
  only the chunk's relevance to the shared question, using the exact same 0/1/2 rubric already
  documented in `judge_results.py` and `evaluation/README.md`.
- **Where a chunk already has a baseline label** (i.e. it was also retrieved by the original query and
  already appears in `judging_pilot.csv`), reuse that label rather than re-judging it from scratch —
  this removes one source of inter-session labeling drift and keeps the baseline and variant scores
  on the same scale by construction wherever the two result sets overlap.
- **Blind re-ordering is recommended but not required**: if practical, shuffle the judging sheet's row
  order (or hide the `query_id` column while labeling) so the person judging doesn't know whether a
  given row came from the original query or a "should perform better" rewrite — this reduces the risk
  of unconsciously grading rewrite results more generously. This is a labeling-process recommendation,
  not a change to any script.

## 5. Distinguishing retrieval relevance from whether a chunk actually answers the question

**These are not the same thing, and the experiment should track both separately.** The existing 0/1/2
`relevance` label measures topical/legal relevance — "is this chunk about the right provision/subject
matter" — which is exactly what Precision@5 and nDCG@10 are built to score, and that scope should stay
unchanged (the task explicitly requires `calculate_metrics.py`'s inputs/labels to remain as they are).
But a chunk can be fully on-topic and still not state the specific fact the question asks for. The
Q03/Q04 coverage audit already found concrete examples of this: Dowry Act §5 ("vests absolutely in the
bride") is clearly relevant to Q04 and establishes *why* a wife owns dowry articles, but it never
states an *evidentiary standard* for recovering them — because no statute in this corpus does. Even a
retrieval fix that surfaces §5 reliably would not, by itself, let a downstream answer fully satisfy the
literal question "what evidence is required."

**Recommendation for this experiment:** add one **supplementary, non-scoring** annotation column —
e.g. `answer_bearing` (yes / partial / no) — filled in only for rows already labeled `relevance 1` or
`2`, capturing whether that specific chunk, read on its own, contains enough to actually answer the
question (not just confirm the topic). This column:
- is **additive only** — it does not replace, redefine, or feed into the existing `relevance` column
  or the Precision@5/nDCG@10 formulas, so no evaluation label or metric definition changes;
- should be recorded as a plain extra column added by hand to a copy of the judging CSV during manual
  review, not as a `judge_results.py` code change (adding it to the script's fixed `FIELDNAMES` would
  be a minor, clearly-scoped future enhancement, but is explicitly out of scope for "do not implement
  query expansion yet");
- is reported and discussed in the experiment's write-up as a qualitative finding alongside the P@5/
  nDCG@10 numbers — e.g. "retrieval improved (chunk now in top 5), but the chunk still cannot fully
  answer the question as asked" is a materially different conclusion from "retrieval and answerability
  both improved," and the report should be able to say which one actually happened for each query.

## 6. Comparing Precision@5 and nDCG@10 against the baseline

For each of the 5 original query families and each `source_type` (`case`, `statute` — `constitution`
stays excluded from metrics exactly as it already is by default in `run_retrieval.py`/
`calculate_metrics.py`, consistent with the existing note that Constitution results are noisy for these
family-law questions), build a comparison table with columns:

| query family | source | baseline P@5 | L1 P@5 | L2 P@5 | baseline nDCG@10 | L1 nDCG@10 | L2 nDCG@10 |
|---|---|---|---|---|---|---|---|

- Values come directly from `metrics_pilot.json` (baseline) and `metrics_expansion_experiment.json`
  (variants) — both already produced by the unmodified `calculate_metrics.py`.
- Any cell that `calculate_metrics.py` reports as unavailable (missing labels, or Recall@10 without a
  qrels entry) must be carried into the comparison table as unavailable, not as 0 or blank-as-zero —
  this is the same "never silently treat missing as irrelevant" rule the metrics script already
  enforces, and it applies equally to this comparison layer.
- Report macro deltas (mean baseline vs. mean L1 vs. mean L2, across the 5 query families, per source
  type) alongside the per-query breakdown, so one query's large swing can't be mistaken for a
  corpus-wide effect.
- **Do not run a formal significance test.** Five query families is far too small a sample; the
  comparison is descriptive/exploratory only, exactly as the evaluation pipeline's own README already
  cautions for the baseline numbers themselves. A consistent directional pattern across multiple query
  families (e.g. "L2 ≥ L1 ≥ baseline for 4 of 5 families") is suggestive; a single family's large jump
  is not evidence of a generalizable effect on its own, and the write-up must say so.
- Report Q04's results with the §2.4 caveat attached (its variants were not blind), so a reader does
  not treat Q04's improvement as equally strong evidence as, say, Q02's.

## 7. Scope and safety — explicitly confirmed

- No retrieval parameter, index, embedding, or schema was changed or will be changed by following this
  plan — every command in §3 calls existing, unmodified scripts with existing, unmodified default
  values, only pointed at a new query file and new output paths.
- No ingestion or indexing step is run; the 10 variant queries are encoded at query time only, exactly
  like every other search this project already performs.
- No existing evaluation label in `judging_pilot.csv` is altered — §4 explicitly reuses, never
  overwrites, any baseline label for chunks that reappear under a variant.
- Query expansion itself (i.e. building this into `embed_index.py` or any production rewriting layer)
  is **not** implemented by this plan — this document only specifies the experiment that would tell us
  whether building such a layer is justified.
- The only file created by this task is this plan. `evaluation/queries_expansion_experiment.json` and
  the `retrieval_expansion_experiment.jsonl` / `judging_expansion_experiment.csv` /
  `metrics_expansion_experiment.json` outputs described in §3 are **proposed next steps**, not created
  here.

## 8. Deliverables checklist for whoever runs this experiment next

- [ ] `evaluation/queries_expansion_experiment.json` — the 10 variants from §2.3, in `queries.json`'s schema
- [ ] `evaluation/results/retrieval_expansion_experiment.jsonl` (+ its `.meta.json`) — via unmodified `run_retrieval.py`
- [ ] `evaluation/results/judging_expansion_experiment.csv` — via unmodified `judge_results.py`, labeled per §4, with the optional `answer_bearing` column per §5
- [ ] `evaluation/results/metrics_expansion_experiment.json` — via unmodified `calculate_metrics.py`
- [ ] A short comparison write-up using the §6 table, stating directional findings only, with the Q04 caveat and the "5 families is not statistically significant" caveat both repeated verbatim
