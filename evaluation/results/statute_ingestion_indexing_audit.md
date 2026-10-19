# Statute Ingestion → Indexing Discrepancy Audit — Read-Only

**Question investigated:** 9 statutes were ingested; does only a subset reach embedding/indexing,
and does any such gap explain Q04's poor statute retrieval?

**Headline finding:** the 9-vs-8 gap is real and fully explained — it is a single, correctly-handled
**exact duplicate** (two identical PDFs of the Family Courts Act, 1964), not data loss. All 8 unique
statutes are chunked and indexed; nothing is missing. The "6" figure in the request's premise does
not correspond to any statute-pipeline count found anywhere in this audit — see §6.

## What each count represents (per the request's own instruction not to assume contradiction)

| Stage | Directory / system | Count | What it counts |
|---|---|---|---|
| Ingestion | `var/rag/statutes_ingested/*.json` | **9 files** | One JSON per uploaded/extracted PDF, including exact duplicates |
| Cleaning | `var/rag/cleaned_statutes/*.json` | **8 files** | One JSON per *unique* statute, after content-hash dedup |
| Chunking | `var/rag/statutes_chunked/*.json` | **8 files**, 261 chunks total, 240 marked `retrievable: true` | Per-statute chunk sets; 21 chunks (footnote blocks, schedule-of-forms, and parent sections superseded by per-subsection children) are intentionally non-retrievable |
| Embedding cache | `var/rag/embeddings/bge-m3/*.npz` (20 shard files, generated 2026-10-11 23:10 → 2026-10-12 00:07) | shards cover all sources, not statute-specific | Dense vector cache for the full corpus (case + statute + constitution); not segmented per source in a way this audit can isolate by inspection alone — see §6 caveat |
| Local doc store | `var/rag/docstore.sqlite`, table `chunks` | **261 statute rows total, 240 with `retrievable=1`**, across exactly **8 distinct `doc_id`s** | Durable record of every chunk ever built, retrievable or not |
| Vector index | Qdrant collection `pk_legal`, `source_type="statute"` | **240 points** (`points/count`, exact filter — read-only) | What is actually searchable at query time |
| Index manifest | `var/rag/index_meta.json` | `"statute": 240`, `indexed_at: "2026-10-12T00:09:55"` | Recorded summary from the last indexing run |

Every one of these numbers is mutually consistent once the 9th ingested file is understood as a
duplicate, not a drop. There is no stage at which exactly 6 statutes exist.

## 1–2. All 9 ingested documents, traced through the pipeline

| # | `doc_id` | Source filename | `content_hash` (first 16 hex) | Cleaned? | Chunked? | Retrievable chunks | In docstore? | In Qdrant? | Final status |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `0425160ffedaefd8c79b3241` | `administratorfb32d6...pdf` | `f1c24acf240622d4` | Yes | Yes | 9 | Yes | Yes | **Indexed** |
| 2 | `24520c344d2285555aa10988` | `administratorf08eaec1...pdf` | `b3119259dab4c9aa` | Yes | Yes | 39 | Yes | Yes | **Indexed** (kept copy of the duplicate pair — see below) |
| 3 | `417403be7fc29b06cafadbbe` | `administrator359d2ddc...pdf` | `36cb96c9124a12c9` | Yes | Yes | 13 | Yes | Yes | **Indexed** (Dowry and Bridal Gifts (Restriction) Act, 1976) |
| 4 | `42769a60dc59c5473affbc32` | `administratord265c726...pdf` | `327429bdd862c126` | Yes | Yes | 60 | Yes | Yes | **Indexed** |
| 5 | `470c476c4195cbc08e94a6da` | `administrator0cb12b90...pdf` | `63bb4835448e5972` | Yes | Yes | 13 | Yes | Yes | **Indexed** |
| 6 | `5264ed6989141ef2d10e35df` | `administratorf08eaec1... (1).pdf` | `b3119259dab4c9aa` (**identical to #2**) | **No** | **No** | — | **No** | **No** | **Correctly skipped — exact duplicate of #2** |
| 7 | `6867f44e86ef03cec19831c6` | `MFLO, 1961.pdf` | `bb93798c37911c89` | Yes | Yes | 16 | Yes | Yes | **Indexed** (Muslim Family Laws Ordinance, 1961) |
| 8 | `8251b5f4daa92e5a22319fc4` | `administrator5e93fac2...pdf` | `9cf19e7c5c28d972` | Yes | Yes | 82 | Yes | Yes | **Indexed** (Divorce Act, 1869) |
| 9 | `b11740afa5f050bea50c5f2d` | `administrator43496aa4...pdf` | `a74d8c448e598962` | Yes | Yes | 8 | Yes | Yes | **Indexed** |

**Evidence for the duplicate pair (#2 and #6):** both ingested JSON files' `metadata.content_hash`
are the byte-identical string `b3119259dab4c9aa51c3958135223c385ca4f194a845a41e089c0dc882c5bfbe`,
both `metadata.law_name` read `"FINAL COPY Updated till 17.1.2025"`, and the two `source_file` names
differ only by macOS's standard duplicate-download suffix: `administratorf08eaec19066a65a1b82cb4ad49feb4d.pdf`
vs `administratorf08eaec19066a65a1b82cb4ad49feb4d (1).pdf`. A full pairwise hash comparison across
all 9 ingested files (done directly in this audit, not inferred) found **exactly this one duplicate
pair and no others** — the remaining 7 files all have distinct hashes.

## 3. Summary table (one row per statute, as requested)

| Statute | Unique `doc_id` kept | Total chunks | Retrievable chunks | Embedding cache | Qdrant points | Status |
|---|---|---|---|---|---|---|
| Dissolution of Muslim Marriages Act, 1939 | `0425160ffedaefd8c79b3241` | 11 | 9 | Present (corpus-wide shards) | 9 | OK |
| Family Courts Act, 1964 | `24520c344d2285555aa10988` | 43 | 39 | Present | 39 | OK (duplicate source correctly deduplicated) |
| Dowry and Bridal Gifts (Restriction) Act, 1976 | `417403be7fc29b06cafadbbe` | 14 | 13 | Present | 13 | OK — see §7 |
| Guardians and Wards Act, 1890 | `42769a60dc59c5473affbc32` | 64 | 60 | Present | 60 | OK |
| Child Marriage Restraint Act, 1929 | `470c476c4195cbc08e94a6da` | 14 | 13 | Present | 13 | OK |
| Muslim Family Laws Ordinance, 1961 | `6867f44e86ef03cec19831c6` | 19 | 16 | Present | 16 | OK |
| Divorce Act, 1869 | `8251b5f4daa92e5a22319fc4` | 87 | 82 | Present | 82 | OK |
| West Pakistan Muslim Personal Law (Shariat) Application Act, 1962 | `b11740afa5f050bea50c5f2d` | 9 | 8 | Present | 8 | OK |
| **Total** | **8 unique statutes** | **261** | **240** | — | **240** | **All 8 fully indexed; the 9th ingested file was a verified exact duplicate, not a loss** |

"Embedding cache" above is reported at corpus level (see §6 caveat) rather than per-statute, since the
`.npz` shard files are not individually attributable to one statute by filename alone without loading
and cross-referencing their contents — not done in this pass to keep the audit read-only and fast; the
docstore and Qdrant checks below are the stronger, chunk-level evidence.

## 4–5. The "missing" statute, root cause

Only **one** document was excluded, not three: `5264ed6989141ef2d10e35df`. Root cause, read directly
from `src/rag_prep/statute_cleaning.py`'s `clean_directory()` (lines 1048–1088):

```python
for raw_path in sorted(input_dir.glob("*.json")):
    ...
    h = (raw_doc.get("metadata") or {}).get("content_hash") or hashlib.sha256(...).hexdigest()
    if h in seen_hashes:
        duplicates.append({"source_file": raw_path.name, "duplicate_of": seen_hashes[h]})
        continue
    seen_hashes[h] = raw_doc["doc_id"]
```

`input_dir.glob("*.json")` is sorted alphabetically, so `24520c344d2285555aa10988.json` is processed
before `5264ed6989141ef2d10e35df.json` (`'2' < '5'`). The first occurrence of a given `content_hash`
is kept; later ones with the same hash are recorded as duplicates and `continue`d past — never cleaned,
never chunked, never embedded, never indexed. This is **deliberate, documented, correct behavior**
(the function's own docstring: *"Duplicate documents (same content hash, e.g. a PDF downloaded twice)
are cleaned once; later copies are skipped and reported"*) — not a failure mode. No `_rejected/`
directory exists under `cleaned_statutes/`, confirming zero quality-gate failures either; the only
exclusion in the entire 9-document set is this one correctly-identified duplicate.

I did not find a persisted run log recording this specific skip event (the summary dict is only
printed to stdout by `main()`, not written to disk) — the hash-equality evidence above is direct,
primary evidence from the data itself, which is stronger than a log would be, but it means the
*exact historical run* that produced this skip is not independently logged. This is noted as a minor
process gap, not a finding that changes the conclusion.

## 6. `index_meta.json` and embedding cache vs. the current collection

`var/rag/index_meta.json` reports `"statute": 240`, timestamped `2026-10-12T00:09:55`. This matches:
- The live Qdrant count for `source_type="statute"` (240, via `points/count` with an exact filter — read-only, re-confirmed in this audit).
- `docstore.sqlite`'s `chunks` table: 240 rows with `source_type='statute' AND retrievable=1` (261 total rows, i.e. including non-retrievable chunks, across exactly 8 distinct `doc_id`s — `5264ed6989141ef2d10e35df` is absent from the docstore too, confirming the dedup is consistent end-to-end, not partially applied).

**`index_meta.json` accurately represents the current collection for statutes** — no staleness found.

The caveat from §3 stands: the `.npz` embedding-cache shards under `var/rag/embeddings/bge-m3/`
(20 files, timestamped 2026-10-11 23:10 through 2026-10-12 00:07, i.e. the same run) were confirmed
to exist and to date from the indexing run, but this audit did not open and cross-reference their
binary contents against individual `chunk_id`s — doing so safely read-only is possible but wasn't
necessary once the docstore and Qdrant checks (below) independently confirmed the same 8-statute,
240-chunk set end to end.

## 7. Does this discrepancy explain Q04's poor statute retrieval?

**No — confirmed, not a hypothesis.** The duplicate/skip event affects only the Family Courts Act,
1964 (`doc_id`s `24520c344d2285555aa10988` and `5264ed6989141ef2d10e35df`). It has no connection to
the Dowry and Bridal Gifts (Restriction) Act, 1976 (`417403be7fc29b06cafadbbe`), which was ingested
as a single, unique document (`content_hash` `36cb96c9124a12c9...`, confirmed to have no duplicate
anywhere in the 9-file ingestion set) and passed through every stage normally.

Direct, independent confirmation that Dowry Act §§3 and §5 are indexed, cross-checked against three
separate systems:

| System | §3 (`417403be7fc29b06cafadbbe:s3`) | §5 (`417403be7fc29b06cafadbbe:s5`) |
|---|---|---|
| Local chunk file (`var/rag/statutes_chunked/417403be7fc29b06cafadbbe.json`) | `retrievable: true` | `retrievable: true` |
| `docstore.sqlite` | `point_id = 13943a8f-0605-5753-8214-9c50727f87a7`, `retrievable = 1` | `point_id = 30c415a6-ff92-5954-83a6-c20fa37385a1`, `retrievable = 1` |
| Live Qdrant (`points/scroll` by `chunk_id`, read-only) | point found, same id `13943a8f-...` | point found, same id `30c415a6-...` |

All three systems agree, including the point IDs matching exactly between the local docstore and the
live Qdrant collection — this rules out both a stale docstore and a stale index. **Q04's weak
retrieval (Precision@5 = 0.0, all 8 results drawn from the unrelated Divorce Act, 1869) remains a
retrieval/ranking issue, as diagnosed in the prior read-only audit
(`evaluation/results/statute_coverage_audit.md`), not an ingestion, dedup, or indexing issue.**

## Note on the request's "6 processed" premise

This audit found no stage, system, or file anywhere in the statute pipeline where the count is 6.
The one directory in the repository that does contain exactly 6 files is `var/rag/structured/`
(6 files, e.g. `00ad7248bf77883e6b5975ed.json` — the file open in the editor at the time of this
request) — but that directory holds **case-law** structuring output (fields `case_title`, `citation`,
`court`, `judges`), unrelated to the statute pipeline. It is flagged here as a plausible, concrete
source of the "6" figure, distinguished explicitly as a **hypothesis**, not a verified fact — the
audit cannot know what the person making the observation was actually looking at.

## Verified facts vs. hypotheses

**Verified (direct evidence obtained in this audit):**
- 9 statute documents were ingested; exactly one duplicate pair exists among them (hash-verified).
- 8 unique statutes exist at every downstream stage (cleaning, chunking, docstore, Qdrant).
- 240 retrievable chunks = 240 docstore rows (`retrievable=1`) = 240 Qdrant points = `index_meta.json`'s recorded count.
- Dowry Act §§3 and §5 are retrievable, in the docstore, and in Qdrant, with matching point IDs.
- The skip of `5264ed6989141ef2d10e35df` is caused by `clean_directory()`'s content-hash dedup, by direct code reading.

**Hypotheses (explicitly not confirmed):**
- That the "6 processed" observation originated from `var/rag/structured/`'s file count.
- No hypothesis is offered for Q04's retrieval weakness beyond what the prior audit already flagged (candidate-pool composition / lexical mismatch) — this audit only rules ingestion/indexing *out* as a cause.

## Recommended next action

**Safest next step: none of the pipeline needs to change.** The ingestion→cleaning dedup behaved
correctly and all intended statutes are indexed. The only concrete, low-risk follow-up worth taking
(not implemented here) is to **rename or annotate the two Family Courts Act source filenames**
(`administratorf08eaec19066a65a1b82cb4ad49feb4d.pdf` / `...(1).pdf`) in the raw ingestion source
directory, or add a one-line note to `var/rag/statutes_ingested/` recording that `5264ed6989141ef2d10e35df`
is an intentionally-skipped duplicate — purely to prevent this same question from being re-investigated
from scratch in the future. This is a documentation convenience, not a correctness fix. Continue
treating Q04's retrieval weakness as a ranking/lexical-match problem per the prior audit, not a data
problem.

---
*No source files, chunk files, embeddings, the embedding cache, the SQLite docstore, Qdrant, or
retrieval logic were modified in the course of this audit. The only operations performed were local
file reads, one SQLite read-only query session, and one read-only Qdrant `points/scroll`/`points/count`
call (already used, non-mutating) to re-confirm chunk presence.*
