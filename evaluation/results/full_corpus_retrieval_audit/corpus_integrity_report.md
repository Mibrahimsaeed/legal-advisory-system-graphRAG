# Corpus & Index Integrity Report — All Three Corpora

**Method:** this report reuses the exhaustive (not sampled) integrity sweeps already performed in `evaluation/results/chunking_embedding_integrity_audit.md` and `evaluation/results/phase2b_constitution_clause_audit.md` (both from this project, same repository state), and re-verifies each specific finding fresh in this pass rather than re-deriving every check from zero. The underlying files and the Qdrant collection are confirmed unchanged since those audits (same `indexed_at` timestamp, same point counts) — re-running an identical exhaustive scan would not produce new information. Where this audit found something genuinely new, it is marked **(NEW)**.

## 1. Corpus inventory (reconciled by chunk_id/point_id, not filename)

| | Case law | Constitution | Statutes |
|---|---:|---:|---:|
| Source documents | 2088 | 1 | 9 ingested (1 confirmed duplicate PDF) / 8 unique |
| Total chunks (chunked-file level) | 27,929 (docstore) | 594 | 261 |
| Retrievable chunks | 25,891 | 547 | 240 |
| Non-retrievable chunks | 2,038 | 47 | 21 |
| Docstore records (retrievable) | 25,891 | 547 | 240 |
| Qdrant points (`source_type` filter, exact count, re-verified live this pass) | **25,891** | **547** | **240** |
| Indexed dense vectors | 25,891 | 547 | 240 |
| Indexed sparse vectors | 25,891 | 547 | 240 |
| Duplicate chunk_id (global) | 0 | 0 | 0 |
| Empty/whitespace-only chunks | 0 | 0 | 0 |
| **Orphaned/missing docs** (source exists, chunked output absent) | **4** | 0 | 0 (1 correctly deduplicated before chunking) |

All retrievable-chunk counts reconcile **exactly** across the chunked files, the docstore, `index_meta.json`, and live Qdrant for all three corpora — re-verified via a fresh `points/count` call in this pass, not assumed from the prior audit.

## 2. Confirmed defects — severity-ranked

| Defect | Corpus | Affected count | Evidence | Retrieval impact | Confidence |
|---|---|---:|---|---|---|
| 4 case documents completely absent from `caselaw_chunked/`, docstore, and Qdrant | Case law | 4 of 2088 docs (0.19%) | Re-verified this pass: `ls var/rag/caselaw_chunked/{id}.json` fails for all 4; `points/count` by `doc_id` = 0 for all 4 (prior audit). One has a reproducible root cause: `AttributeError: '_Builder' object has no attribute 'tail_i'` in `caselaw_chunker.py:1424`. | These 4 cases can never be retrieved under any query. | **Confirmed, Critical** |
| 3 full-case duplicate pairs indexed as separate points | Case law | 6 documents (3 pairs), ~33 duplicated chunks | Re-verified this pass: both point_ids of the PLD 2014 Lah 131 pair (`3ebe1e86...`, `449cb69b...`) still return count=1 each in Qdrant. | A query that would retrieve one copy has a chance of consuming 2 of its top-k slots on the identical case instead of a second, distinct relevant case. | **Confirmed, Medium** |
| Clause-number mislabeling from inline cross-references (Articles 59, 185, 199, 224A) | Constitution | 4 confirmed chunks | Re-verified this pass: `art_199_c1_2` and `art_59_c1_2` both still present in the current 594-chunk file with the same mislabeled `clause_number`. | Content is retrievable but mislabeled and syntactically truncated if retrieved alone; does not affect whether the text is findable, only its labeling. | **Confirmed, Medium** |
| Backwards range label (Article 270AA) | Constitution | 1 chunk | Prior audit (`art_270aa_c5`, clause label "(5)-(2)"); chunk's own opening text is genuine, only the declared range end is spurious. | Cosmetic/metadata only. | **Probable, Low** |
| 1 internally self-duplicated document | Case law | 1 doc, 2 duplicate paragraph pairs | Prior audit: `9d4ccb7d8a3d9abe50ca3ad6` chunks `:0007`≡`:0010`, `:0008`≡`:0011`. | Minor — wastes 2 of that one document's own candidate slots. | **Confirmed, Low** |
| Statute ingestion duplicate correctly deduplicated | Statutes | 1 of 9 ingested docs | Prior audit: byte-identical content_hash, correctly skipped by the cleaning pipeline's existing dedup logic. | **None** — this is correct behavior, not a defect, included here only for completeness. | **Verified non-defect** |

## 3. What this audit explicitly did NOT re-derive from scratch

- A full vector-by-vector NaN/Inf/zero-vector scan across all 26,678 points (already performed exhaustively in the prior integrity audit; the collection has not been reindexed since — `indexed_at` unchanged).
- A corpus-wide systematic sweep for the Article 59/185/199/224A-style clause-mislabeling pattern beyond the 5 instances already found (the prior audit's sweep covered all 328 articles; this pass re-confirmed 2 of the 5 specific instances still exist rather than re-running the full 328-article sweep, since nothing in the underlying file changed).

**Net assessment for Phase 4:** the case-law corpus carries the only *critical* integrity defect (4 missing documents) and the only *confirmed, non-cosmetic* duplication issue (3 duplicate case pairs) found anywhere in the three corpora. The Constitution's defects are confirmed but low-impact (mislabeling, not loss). The statute corpus has zero outstanding integrity defects.
