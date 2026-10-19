# Chunking and Embedding Integrity Audit — Read-Only

## 1. Executive Summary

This audit inspected chunking and embedding integrity for the Constitution, Statute, and Case-law
corpora using only read-only file reads, read-only SQLite access, and read-only Qdrant API calls
(`points/count`, `points/scroll`, `points` retrieve, collection info). No data, code, or index state
was modified. Two previously-reported Constitution defects (41 ghost chunks, 45 relabeled duplicate
IDs) and one previously-reported Statute defect (Family Courts Act §18/§17B mislabeling) were
**re-verified against current artifacts and are no longer present** — they appear to have been fixed
since those reports were written. A previously-cited case-law figure ("3,089 chunks ≤5 words") **could
not be reproduced** from any current artifact; direct measurement today found 12 retrievable chunks
meeting that description, and the original 3,089 figure is labeled unverified.

**New, directly-confirmed defects were found that no prior report in this project had surfaced:**

1. **4 case-law documents are completely absent** from `caselaw_chunked/`, the docstore, and Qdrant —
   100% content loss for those 4 cases. One has a live, reproducible root cause: an `AttributeError`
   in `caselaw_chunker.py`'s `_Builder.final_orders()`.
2. **3 case-law documents are full duplicates of 3 other case-law documents** (same case, same
   citation, ingested twice under different `doc_id`s), and **both copies of each are indexed in
   Qdrant as separate points** — confirmed live in the index, not just in the chunk files.
3. **1 case-law document has internally duplicated paragraph content** (the same two paragraphs appear
   twice within one document's own chunk sequence).

Embedding-generation and vector/index integrity checks — cache-to-index vector value comparison,
cache completeness, NaN/Inf/zero-vector scan, dimension/distance/sparse config, token-length headroom
(re-verified against the real BGE-M3 tokenizer, not just the chunkers' proxy tokenizer) — **all
passed** for the artifacts and samples inspected. Semantic retrieval quality is explicitly **not**
assessed here and remains a separate, unproven question (see §10).

## 2. Audit Scope and Read-Only Method

- Repository root: `/Users/ibrahim/Desktop/legal_advisory_system/legal-advisory-system-graphRAG`.
  Branch: `phase-1.2`.
- Baseline `git status --short` (before any work in this audit): 11 untracked files, all under
  `evaluation/` and `evaluation/results/`, all from the prior query-phrasing experiment task. `git diff
  --stat`: empty. These are repeated at the end of §2 bis below, unchanged.
- All inspection in this audit was performed via: direct file reads of chunk/docstore/cache JSON and
  `.npz` artifacts; a read-only SQLite connection (`file:...?mode=ro`); and read-only Qdrant HTTP calls
  (`points/count`, `points/scroll`, `points` retrieve-by-id, `collections/{name}` info). Two in-memory,
  read-only calls to `chunk_document()` (from `src/rag_prep/caselaw_chunker.py`, unmodified) were made
  to reproduce a failure mode on specific documents — nothing was written to disk by these calls,
  mirroring the method explicitly used and accepted in an earlier read-only audit this session. One
  tokenizer (`AutoTokenizer.from_pretrained("BAAI/bge-m3", local_files_only=True)`) was loaded **from
  the local HuggingFace cache only** (confirmed already present at
  `~/.cache/huggingface/hub/models--BAAI--bge-m3` from earlier sessions) — `local_files_only=True` was
  passed explicitly so no network/download could occur; this tokenizer was used only to count tokens
  on already-existing text, never to generate or alter any embedding.
- No chunking, embedding, ingestion, indexing, or repair command was run. No Qdrant write/upsert/
  delete/recreate call was made. No file was created except this report.

## 3. Current Architecture and Artifact Inventory

| Stage | Constitution | Statute | Case law |
|---|---|---|---|
| Source/ingested | `var/rag/constitution_ingested/` (1 doc) | `var/rag/statutes_ingested/` (9 docs, 1 confirmed duplicate pair per the prior ingestion audit) | `var/rag/raw/` (2088), `var/rag/processed/` (2088) |
| Cleaned | `var/rag/cleaned_constitution/` (1) | `var/rag/cleaned_statutes/` (8) | `var/rag/structured_gemini/` (2090 entries, but 2 are run-metadata files `full_run_report.json`/`progress.json`, not documents — 2088 real docs) |
| Parsed/structured | `var/rag/constitution_parsed/` (1) | — | `var/rag/structured_gemini/` (as above) |
| Chunked | `var/rag/constitution_chunked/` (1 doc, 594 chunks, 547 retrievable) | `var/rag/statutes_chunked/` (8 docs, 261 chunks, 240 retrievable) | `var/rag/caselaw_chunked/` (2084 doc files — **4 fewer than the 2088 processed docs; see §6**) |
| Docstore | `var/rag/docstore.sqlite`, table `chunks` | same table, `source_type='statute'` | same table, `source_type='case'` |
| Embedding cache | `var/rag/embeddings/bge-m3/` — 27 shard `.npz` files, shared across all 3 sources | | |
| Index metadata | `var/rag/index_meta.json`: `{"constitution": 547, "statute": 240, "case": 25891}`, `indexed_at: 2026-10-12T00:09:55` | | |
| Vector index | Qdrant collection `pk_legal`, 26,678 points total | | |

Docstore row counts (read-only query): `case` 27,929 total / 25,891 retrievable; `constitution` 594
total / 547 retrievable; `statute` 261 total / 240 retrievable — **all three retrievable counts match
`index_meta.json` and the live Qdrant per-source counts exactly** (re-confirmed via `points/count` with
an exact filter in this audit, independent of the earlier statute-specific audit).

Qdrant collection config (read-only `GET /collections/pk_legal`): `points_count: 26678`,
`indexed_vectors_count: 53356` (= 26,678 × 2, one dense + one sparse vector per point, consistent with
no point missing either vector type), dense vector named `"dense"` (size 1024, distance `Cosine`),
sparse vector named `"bm25"` (modifier `idf`) — matches `DENSE_DIM=1024` and `BM25_MODEL`/IDF
configuration read directly from `src/rag_prep/embed_index.py`, and matches the named-vector usage
(`using="dense"` / `using="bm25"`) in `hybrid_search()`.

## 4. Chunking Integrity — Constitution

**Re-verification of previously reported findings (41 ghost chunks, 45 relabeled duplicate sub-clause
IDs):** scanned all 594 chunks in the current `constitution_chunked/b427a922990d75f027928c15.json`.
**Zero chunks have empty/whitespace-only `canonical_text`. Zero duplicate `chunk_id` values exist
anywhere in the file.** Both previously-reported defects are **not present in current artifacts** —
status: **RESOLVED** (not reproducible today; this session's own history recorded a fix to
`_flatten_clause_children` and a `_filter_monotonic` clause-level fix applied after those reports were
written, which is consistent with this result, though re-verified independently here rather than
assumed).

**Articles 63, 89, 175A, 260** — inspected directly: each has a non-retrievable parent chunk (full
article span) plus retrievable per-clause/per-subclause children, with no duplicate IDs and no
zero-length text. Article 260 (the "Definitions" article, previously the site of a duplicate-lettering
bug) now shows clean disambiguated child IDs (`art_260_c1_p1` … `_p4`, `art_260_c2`, …) with no
collisions. Article 63's oversized clause 1 is split into 4 contiguous child chunks
(`art_63_c1_p1`…`_p4`); a direct boundary check confirmed **zero gap and zero overlap** between every
consecutive pair (`source_end` of part N == `source_start` of part N+1 for all 3 boundaries), and the
text reads continuously across the p1/p2 boundary with no duplicated or dropped text. **Status:
VERIFIED CHECK** for all four named articles.

## 5. Chunking Integrity — Statutes

**Family Courts Act §17/§17A/§17B/§18 (previously reported mislabeling):** read directly from the
current `statutes_chunked/24520c344d2285555aa10988.json`. All four sections are present, distinctly
chunked, correctly numbered, and their `canonical_text` matches their stated section title and number
exactly (§17B's text is about "Power of the Court to issue Commission," §18's is about "Appearance
through agents" — no cross-contamination). **Not reproducible against current artifacts — status:
RESOLVED** (treated as unverified-from-current-evidence rather than confirmed, per the instruction not
to rely blindly on prior claims; see §12).

**`[Omitted]` markers:** found 20 across the 8 statutes (Dowry Act §§7,11; Guardians and Wards Act
§§5,51; Child Marriage Restraint Act §§3,11; Muslim Family Laws Ordinance §§12,13; Divorce Act §§4-9,
16,17,17A,20,56). **Every single one carries an explicit, cited repeal/amendment authority** — either
inline (e.g. "Omitted by the Federal Laws (Revision and Declaration) Ordinance, 1981... s.3") or, for
the two Child Marriage Restraint Act entries that show only a bare `[Omitted.]` inline, in their linked
footnote (verified: footnote 2.3 and 4.1 both cite "Omitted by the Muslim Family Laws Ordinance...
s.12"). **Status: VERIFIED CHECK** — these are faithful extractions of genuine legislative omissions,
not pipeline defects, consistent with the task's own instruction that an `[Omitted]` marker is not
automatically a defect.

**Document-level validation gate:** `embed_index.py`'s `load_chunks()` silently skips an entire
document if `validation.ok` is falsy. Checked all 8 statute documents' `validation.ok` field directly:
all 8 are `true`. **Status: VERIFIED CHECK** — no statute document is being silently dropped at this
gate.

## 6. Chunking Integrity — Case Law

**Document-count discrepancy, confirmed:** `var/rag/processed/` has 2088 source documents;
`var/rag/caselaw_chunked/` has only 2084 output files. The 4 missing `doc_id`s are:
`177dbfa9f6115c8c4bb0cdcf`, `5816efa9297184a599e6e40b`, `6e37d0e6d64fa457d8020333`,
`c303283c9d19d50d623760de`. All 4 confirmed present and readable in `processed/`; all 4 confirmed
**completely absent from the docstore and from Qdrant** (`points/count` filtered by `doc_id` returns 0
for all four). **This is 100% content loss for 4 real cases — Severity: Critical.**

Root-cause investigation (read-only, in-memory `chunk_document()` calls against the real `processed/`
JSON for each, using the current, unmodified `caselaw_chunker.py`):

- `177dbfa9f6115c8c4bb0cdcf`: **reproduces a live exception today**:
  ```
  AttributeError: '_Builder' object has no attribute 'tail_i'
  ```
  at `caselaw_chunker.py:1424`, inside `final_orders()`, which unconditionally reads `self.tail_i`.
  Reading the surrounding code: `self.tail_i` is only assigned at line 1369, inside an earlier
  method (the headnote/body-labeling pass spanning roughly lines 1180–1369). For this specific
  document's structure, that earlier method evidently returns or is bypassed before reaching line
  1369, leaving `tail_i` never set on `self`, and `final_orders()` has no defensive fallback (no
  `getattr(self, "tail_i", None)`). **This is a confirmed, currently-reproducible defect in
  `src/rag_prep/caselaw_chunker.py`.**
- `5816efa9297184a599e6e40b`, `6e37d0e6d64fa457d8020333`, `c303283c9d19d50d623760de`: **do not
  currently raise any exception** — `chunk_document()` succeeds in memory today, producing 38, 27, and
  14 chunks respectively. Their absence from the real `caselaw_chunked/` directory therefore cannot be
  explained by the current code. No persisted failure log exists to explain the historical run: the
  driver script (`var/rag/run_chunking_full.py`) is designed to write a `chunking_report.json` listing
  `failed_doc_ids` (confirmed by reading the script), but **no such file exists** in
  `var/rag/caselaw_chunked/` today. Rerunning the pipeline to find out is explicitly out of scope for
  this audit. **Status: NOT VERIFIABLE** for these 3 — the original cause cannot be established from
  existing artifacts; it is plausible the code has since been fixed and the batch was simply never
  resumed to include them, but this is a hypothesis, not a finding.

**Full-case cross-document duplication, confirmed (new finding, not in any prior report):** scanning
every retrievable chunk's `embedding_text` for byte-identical duplicates across different `doc_id`s
found three duplicate case pairs:

| Case | doc_id A | doc_id B | Retrievable chunks | Identical-text matches |
|---|---|---|---|---|
| PLD 2014 Lah 131 (Azhar Mehmood v. Ms. Hafeez-un-Nisa) | `3ebe1e86de756077fb70c9b7` | `449cb69b290757a36add36c4` | 13 vs 13 | 12 of 13 |
| 2008 YLR 133 | `7bb739706a3cbad3118d3d8b` | `9c99041ec7aef22ba335d463` | 7 vs 7 | 7 of 7 (complete) |
| 2010 MLD 1997 | `43cc3204d9d5b31479aaa94b` | `d83817f52f98ea89a4582eae` | 13 vs 13 | 13 of 13 (complete) |

Each pair's source citation and case content are identical; these are the same case ingested twice
under two different `doc_id`s, with **no deduplication step anywhere in the case-law pipeline**
(unlike the statute pipeline, which has an explicit content-hash dedup during cleaning). Confirmed live
in the index: both `doc_id`s of the PLD 2014 pair were spot-checked by `chunk_id` against Qdrant via
`points/scroll` — both exist as separate points with matching `chunk_id`/`point_id` to the docstore.
**Severity: Medium** — no legal content is lost, but every query that would retrieve one of these
cases has a chance of consuming two of its top-k slots on the identical case instead of a second,
genuinely different relevant case, and the duplication inflates that case's apparent prevalence in the
corpus.

**Internal (within-document) duplicate paragraphs, confirmed (new finding):** document
`9d4ccb7d8a3d9abe50ca3ad6` ("Mst. Balqees Bano v. Shamim Ahmad alias Yasin," 1991 CLC 2057) has chunk
`:0007` byte-identical to `:0010` (an "Evidence" section paragraph) and `:0008` byte-identical to
`:0011` (a "Lower court orders" paragraph) — the same two paragraphs appear twice within this single
document's own chunk sequence. **Severity: Low** — confined to one document, but indicates either a
structuring-stage (paragraph-range) or chunking-stage defect that duplicated a text span; root cause in
the structuring/labeling code was not traced further in this audit (would require inspecting this
specific document's structured-stage intermediate output, which was not done here to keep scope
bounded — flagged as a follow-up, not resolved).

**Previously reported "3,089 chunks with ≤5 words" — not reproducible, labeled UNVERIFIED.** No
current artifact (chunked files, reports, logs) contains this figure or the method used to derive it.
Direct measurement today, scanning `canonical_text` word counts across **all 25,891 retrievable**
case-law chunks, found **12** with ≤5 words, all `unit_type: "headnote"`. All 12 have non-empty
`prev_chunk_id`/`next_chunk_id` (neighbor-chaining present), and **all 12 have an `embedding_text` well
above 5 words** — because `embedding_text` always prepends the full `context_header` (citation, court,
party names, section label) before the short canonical text, so none of them is ever embedded as bare,
context-free text. Two examples (`'1985.'`, `'(b) West'`) look like genuine orphaned
heading/label fragments from headnote sub-topic splitting rather than substantive content on their
own — flagged as a **Low-severity potential quality issue**, not a defect, since nothing is lost
(the fragment's neighbor chunks and context header carry the surrounding meaning).

## 7. Docstore, Chunk IDs, and Qdrant Payload Consistency

Spot-checked across all three corpora (one Constitution chunk, one non-retrievable case chunk, two of
the confirmed-duplicate case chunks, plus the full statute cross-check already on record from an
earlier audit this session): in every case, `docstore.sqlite`'s stored `point_id` for a given
`chunk_id` matches the point actually present in Qdrant at that ID, with matching `chunk_id` and
`source_type` in the payload. The non-retrievable case chunk (`002dcb72b51088b2694c627f:0002`,
`retrievable=0` in the docstore) was confirmed to have **zero** corresponding points in Qdrant —
the `retrievable` filter is being enforced correctly end-to-end, not just recorded. **Status: VERIFIED
CHECK.** Parent/neighbor expansion fields (`prev_chunk_id`/`next_chunk_id` for case law,
`parent_id`/`child_id` for Constitution/statutes) are present and populated in the schema; this audit
did not execute an actual expansion query against the retrieval code (that would be a retrieval-logic
test, out of scope here), so whether expansion can cross document boundaries is **NOT VERIFIABLE**
from static inspection alone — flagged as a question for a future, narrowly-scoped functional test.

## 8. Embedding-Generation Integrity

**Exact embedded text, confirmed by direct code reading** (`src/rag_prep/embed_index.py`, `cmd_embed()`
lines 323–355 and `hybrid_search()` lines 500–520): the text passed to `encode_dense()` is always
`chunk["embedding_text"]` — never `canonical_text` directly, never a separately-constructed string, and
identically on both the indexing side and the query side. **No prefix or instruction string is added
anywhere in this code path** — no `"query: "`/`"passage: "` or similar wrapper. `canonical_text`
(exact source slice, display/verification field) and `embedding_text` (context-header-enriched field,
what's actually embedded) are **intentionally different by design**, consistently across Constitution,
statute, and case-law chunks (confirmed present as two distinct fields in all three schemas).
**Status: VERIFIED CHECK.**

**Document-level validation gate re-confirmed for all 3 corpora** (not just statutes, §5): `validation.
ok` is `true` for the single Constitution document and for all 2084 case-law chunked documents —
**zero** documents are being silently excluded at this gate across the whole corpus.

## 9. Vector and Qdrant Integrity

**Embedding cache contents:** 26,644 unique cached vectors across 27 shard files, **all dimension
1024**, **zero NaN**, **zero Inf**, **zero all-zero vectors** (full scan of every cached vector, not a
sample). **Status: VERIFIED CHECK.**

**Cache-to-chunk alignment:** recomputed the cache key (`sha1(model|max_seq_len|embedding_text)`) for
every one of the 26,678 currently-retrievable chunks across all 3 corpora and checked membership in
the cache. **100% — zero chunks are missing a cached vector.** The cache has 34 fewer unique entries
than there are retrievable chunks (26,644 vs 26,678); this reconciles **exactly** with the 34 chunks
found to have byte-identical `embedding_text` to another chunk (the 3 duplicate-case pairs plus the
1 internal-duplicate document from §6) — the cache is correctly content-deduplicating, not dropping
anything. **Status: VERIFIED CHECK**, and this independently corroborates the §6 duplicate findings
through a completely different mechanism (hash-key collision counting vs. direct text comparison).

**Cached-vector-to-Qdrant-vector consistency:** for one sample chunk per corpus (a Constitution
preamble chunk, a statute preamble chunk, and a case caption chunk), fetched the stored vector directly
from Qdrant (`points` retrieve-by-id, `with_vector: true`, read-only) and compared its first 5
dimensions against the cached float16 vector cast to float32. All three matched within float16
rounding precision (e.g. cached `0.01702880859375` vs. live `0.017023956` — exactly the expected
float16→float32 rounding difference, not a mismatch). **Status: VERIFIED CHECK** for the 3 samples
inspected; this is a sample, not an exhaustive comparison of all 26,678 points, which is noted in §14.

**Stale-cache risk:** the cache key already incorporates `MODEL_NAME` and `MAX_SEQ_LEN`, so a change to
either would produce entirely new, non-colliding keys rather than silently reusing a stale vector — by
construction, not merely by observation. **Status: VERIFIED CHECK** (design-level, confirmed by reading
`cache_key()`).

**Qdrant collection configuration vs. indexing implementation:** dense `size=1024`/`distance=Cosine`
matches `DENSE_DIM` in `embed_index.py`; sparse vector named `bm25` with `modifier: idf` matches
`BM25_MODEL`'s fastembed IDF configuration; both are **named** vectors (`"dense"`, `"bm25"`), matching
`hybrid_search()`'s `using=` parameters exactly — no unnamed/default-vector mismatch. **Status:
VERIFIED CHECK.**

## 10. Semantic Embedding Quality — What Remains Unproven

Per the task's own framing, three separate questions exist, and only the first two were tested here:

1. **Embedding-generation integrity** (were intended texts converted into valid vectors under the
   expected config?) — **substantially checked and passed** (§8–9).
2. **Vector/index integrity** (do cached and indexed vectors correspond to the intended chunks?) —
   **substantially checked and passed** (§9), with the duplicate-case and missing-document findings as
   the exceptions.
3. **Semantic embedding quality** (do the vectors rank legally relevant chunks highly for real legal
   questions?) — **not assessed by this audit, and not assessible from configuration or successful
   indexing alone.** A separate, prior controlled evaluation in this project (the query-phrasing
   experiment, `evaluation/results/query_expansion_comparison.md`) already found real ranking problems
   for several pilot queries that have nothing to do with chunking/embedding *integrity* — e.g. Q04's
   baseline statute retrieval surfaced an entirely unrelated statute despite the correct chunks being
   present, indexed, and integrity-clean by every check in this audit. **This audit does not claim, and
   explicitly should not be read as claiming, that semantic quality is acceptable** — only that the
   mechanical pipeline converting text to indexed vectors is, with the exceptions noted, working
   correctly. One observation worth flagging for a future quality (not integrity) discussion: BGE-M3
   is sometimes used with a retrieval instruction prefix on the query side for improved ranking; this
   pipeline adds no such prefix on either side. Whether that would help is a semantic-quality question,
   not an integrity defect, and is explicitly out of scope for a conclusion here.

## 11. Confirmed Defects

| # | Corpus / stage | Severity | Evidence | Affected IDs / counts | Retrieval impact | Confirming test |
|---|---|---|---|---|---|---|
| 1 | Case law / chunking | **Critical** | `processed/` has 2088 docs, `caselaw_chunked/` has 2084; the 4 missing `doc_id`s have zero Qdrant points | `177dbfa9f6115c8c4bb0cdcf`, `5816efa9297184a599e6e40b`, `6e37d0e6d64fa457d8020333`, `c303283c9d19d50d623760de` (4 of 2088, 0.19%) | These 4 cases can **never** be retrieved, regardless of query — complete, silent loss | `points/count` by `doc_id` = 0 for all 4; file-existence diff `processed/` vs `caselaw_chunked/` |
| 2 | Case law / chunking code | **High** (root cause of #1 for 1 doc) | Reproducible `AttributeError: '_Builder' object has no attribute 'tail_i'` at `caselaw_chunker.py:1424`, inside `final_orders()`; `self.tail_i` only set at line 1369 in an earlier method that doesn't run to completion for this doc's structure | `177dbfa9f6115c8c4bb0cdcf` | Causes #1 for this specific document; may affect other documents with the same structural edge case not yet identified | Live, in-memory `chunk_document()` call against the real source JSON, read-only |
| 3 | Case law / ingestion-dedup (missing) | **Medium** | 3 confirmed cross-`doc_id` full-case duplicate pairs, both copies indexed as separate Qdrant points | PLD 2014 Lah 131 (`3ebe1e86...`/`449cb69b...`, 12/13 chunks identical), 2008 YLR 133 (`7bb73970...`/`9c99041e...`, 7/7), 2010 MLD 1997 (`43cc3204...`/`d83817f5...`, 13/13) — 6 documents, ~33 duplicated chunks | Wastes top-k retrieval slots on a redundant copy of an already-retrieved case; inflates that case's apparent corpus frequency | Direct text-identity scan across all retrievable chunks; corroborated by the cache-key reconciliation in §9; both copies confirmed present in live Qdrant via `points/scroll` |
| 4 | Case law / structuring or chunking | **Low** | 2 pairs of byte-identical chunks within one document's own sequence | `9d4ccb7d8a3d9abe50ca3ad6` chunks `:0007`≡`:0010`, `:0008`≡`:0011` | Minor: wastes 2 of this document's own retrieval slots on repeated content | Direct text-identity scan |

## 12. Potential Defects and Unresolved Questions

| # | Corpus | Issue | Why uncertain | Suggested follow-up test |
|---|---|---|---|---|
| 1 | All 3 | `token_count` is computed with `tiktoken`'s `cl100k_base`, not the real BGE-M3 tokenizer. Direct comparison on the 5 longest chunks per corpus showed the real tokenizer produces 4%–19% more tokens than the stored value. | The discrepancy is confirmed and quantified, but its *impact* is not a defect in this audit's direct finding — every real-tokenizer count observed (max 595, case law) remains well under `MAX_SEQ_LEN=1024`, so no truncation currently occurs. Flagged as potential because a future, longer chunk could cross the line without the proxy tokenizer's count revealing it. | Re-run the existing `cmd_check()` function in `embed_index.py` (reads real token counts via `AutoTokenizer`, already exists, unmodified) periodically or before any future large ingestion, rather than relying on the chunkers' own `token_count` field as a safety gate |
| 2 | Case law | 3 of the 4 missing documents (§6) do not reproduce any exception with current code | Cannot rule out that the historical batch run was simply interrupted and never resumed, vs. a since-fixed bug; no log exists to distinguish these | A supervised, approved re-run of just these 3 `doc_id`s (not the full batch) would resolve this, but is out of scope for this read-only audit |
| 3 | Case law | 12 retrievable headnote chunks ≤5 words, 2 of which look like orphaned heading fragments | Not shown to cause any actual retrieval harm (context header always present in `embedding_text`), so classified as a quality observation rather than a defect | Manual review of whether these fragments should be merged into their neighbor chunk at the structuring stage |
| 4 | Case law | Internal duplicate in `9d4ccb7d8a3d9abe50ca3ad6` (§6, #4) — exact mechanism not traced | Would require inspecting this document's intermediate structured-stage output, out of this audit's bounded scope | Targeted, single-document trace through `structurer_llm_gemini.py`'s paragraph labeling for this `doc_id` |
| 5 | All 3 | Whether parent/neighbor expansion can cross document boundaries | Schema fields exist (`prev_chunk_id`/`next_chunk_id`, `parent_id`) but no expansion query was executed against retrieval code in this audit | A narrow, read-only functional test exercising the actual expansion code path, if one exists in the retrieval layer |

## 13. Verified Checks

- Constitution: 0 empty/ghost chunks (previously reported 41) — resolved, re-verified.
- Constitution: 0 duplicate chunk IDs (previously reported 45) — resolved, re-verified.
- Constitution: Articles 63, 89, 175A, 260 structurally sound; Article 63 clause-1 split is
  gap-free/overlap-free/contiguous.
- Statutes: Family Courts Act §17/§17A/§17B/§18 correctly and distinctly labeled, content matches
  title — previously reported mislabeling not reproducible.
- Statutes: all 20 `[Omitted]` markers carry genuine, cited legislative authority.
- All 3 corpora: `validation.ok` gate has zero silent document exclusions.
- All 3 corpora: retrievable chunk counts match exactly across chunked files, docstore,
  `index_meta.json`, and live Qdrant (547 / 240 / 25,891).
- All 3 corpora: zero chunks exceed `MAX_SEQ_LEN=1024`, confirmed with both the chunkers' own stored
  (proxy) token count and a direct real-tokenizer re-measurement of the longest chunks.
- Embedding cache: 26,644 vectors, all dim 1024, zero NaN/Inf/all-zero.
- Embedding cache ↔ retrievable-chunk alignment: 100% coverage, with the 34-entry gap fully
  reconciled to known duplicate content.
- Cached vector values match live Qdrant-stored vector values (3-sample spot check, one per corpus).
- Cache-key design structurally prevents stale-cache reuse across model/config changes.
- Qdrant dense/sparse vector configuration matches the indexing code's constants exactly; named-vector
  scheme is consistent between Qdrant config and `hybrid_search()`.
- Docstore↔Qdrant `point_id`/`chunk_id` consistency, including correct enforcement of the
  `retrievable` filter (non-retrievable chunks confirmed to have zero Qdrant points).
- Embedded text confirmed, by direct code reading, to be `embedding_text` exactly, with no added
  prefix/instruction, identically for indexing and querying.

## 14. Limitations

- The cached-vector-to-Qdrant comparison (§9) was a 3-point sample, not an exhaustive check of all
  26,678 points — doing so safely read-only is possible (iterate `points/scroll` with `with_vector` and
  compare against every cache entry) but was not run to completion given the audit's time bounds;
  flagged rather than silently assumed to generalize.
- The root cause for 3 of the 4 missing case-law documents could not be established (§6, §12 #2) —
  explicitly labeled NOT VERIFIABLE rather than guessed.
- The `caselaw_chunker.py` `AttributeError` was confirmed for exactly one document; whether other
  documents share the same latent structural trigger was not systematically swept across all 2088
  source documents in this audit (doing so would mean running `chunk_document()` 2088 times in memory —
  judged out of proportion for this pass, but straightforward for a dedicated follow-up).
- Internal duplicate root cause (§6, #4; §12 #4) was identified but not traced to its originating
  pipeline stage.
- Semantic/ranking quality was explicitly not assessed (§10), by design.
- The real-BGE-M3-tokenizer check (§12 #1) was run only on the 5 longest chunks per corpus (by the
  proxy token count), not the full corpus — a reasonable proxy for "is there headroom," not an
  exhaustive guarantee.

## 15. Recommended Next Step

**The pipeline should receive targeted integrity fixes, after separate explicit approval, before any
further retrieval-ranking investigation treats the corpus as a clean baseline.** Specifically:

1. Fix or at minimum add a defensive fallback for `_Builder.tail_i` in `caselaw_chunker.py` (Confirmed
   Defect #2) — this is the one actionable, reproducible root cause found.
2. Decide, with approval, whether to re-run chunking for the 4 missing documents (Confirmed Defect #1)
   and whether to deduplicate the 3 confirmed case pairs (Confirmed Defect #3) — both require a write
   operation this audit was not authorized to perform.
3. Do **not** yet proceed to a new controlled retrieval-quality evaluation claiming a "clean" corpus —
   the 3 duplicate-case pairs and 4 missing documents mean any such evaluation today would be measuring
   against a corpus with known, small but real integrity gaps. Fixing confirmed defect #1–3 first (or
   explicitly accepting them as known, bounded noise) should precede the next quality evaluation.
4. The existing evaluation pipeline and its prior findings (query-phrasing experiment, Q04 diagnostic)
   remain valid on their own terms — they were never about these specific missing/duplicate documents,
   and nothing here contradicts them.

## 16. Final Verdict by Corpus and Component

| Corpus | Chunking integrity | Embedding-generation / index integrity |
|---|---|---|
| Constitution | **PASS FOR INTEGRITY** | **PASS FOR INTEGRITY** |
| Statutes | **PASS FOR INTEGRITY** | **PASS FOR INTEGRITY** |
| Case law | **CONDITIONAL PASS** — 4 documents completely missing (1 with a confirmed, fixable bug), 3 confirmed duplicate case pairs, 1 internal duplicate; all other checks passed | **CONDITIONAL PASS** — vector/cache/index mechanics are clean; the conditional applies because the duplicate and missing-document issues are chunking-stage defects that necessarily propagate into what gets embedded and indexed |

A `PASS FOR INTEGRITY` does not mean semantic retrieval quality is guaranteed — see §10.

---

## Final Git Status Check

```
$ git status --short
```
(repeated at the end of this audit, as required)
