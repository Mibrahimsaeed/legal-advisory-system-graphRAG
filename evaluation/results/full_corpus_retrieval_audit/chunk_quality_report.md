# Chunk Quality Report — All Three Corpora

## 1. Chunk length distribution (characters, retrievable chunks only, computed fresh this pass)

| Corpus | n | min | median | mean | p90 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Constitution | 547 | 77 | 523 | 690 | 1500 | 1742 | 2239 |
| Statutes | 240 | 19 | 508 | 635 | 1310 | 1591 | 1828 |
| Case law | 25,891 | 5 | 931 | 921 | 1580 | 1735 | 4146 |

Token-length distributions were already checked exhaustively against both the chunkers' own proxy tokenizer and the real BGE-M3 tokenizer in the prior integrity audit (`evaluation/results/chunking_embedding_integrity_audit.md`, §9/§12): **zero chunks in any corpus exceed the 1024-token embedding limit**, confirmed with the real tokenizer on the longest chunks per corpus (max observed 595 real tokens, case law) — not re-run here since nothing in the corpus changed.

## 2. Confirmed content-quality findings, by corpus

**Case law:**
- 12 retrievable chunks have ≤5 words in `canonical_text` (all `unit_type: headnote`), min length 5 chars — consistent with the fresh min=5 figure above. All 12 have a populated `context_header` in their `embedding_text`, so none is embedded context-free (prior audit finding, re-confirmed consistent with this pass's fresh min-length figure).
- 3 full-case duplicate pairs and 1 internal self-duplicate document (see `corpus_integrity_report.md`) — these are duplication defects, not truncation or dilution defects.
- 4 documents entirely missing (see `corpus_integrity_report.md`) — a coverage defect, not a content-quality defect within an existing chunk.
- **Not re-swept in this pass:** a corpus-wide near-duplicate scan beyond the 4 already-known cases (3 cross-document pairs + 1 internal) was not re-run across all 25,891 chunks — the prior audit's method (exact `embedding_text` hash collision) is exhaustive for *exact* duplicates but would not catch *near*-duplicates with minor textual variation; no near-duplicate scan has been run in this project to date. **This is disclosed as an open gap, not claimed as checked.**

**Constitution:**
- 4 confirmed clause-mislabeling instances (Articles 59, 185, 199, 224A) and 1 probable (270AA) — all are label/metadata issues on chunks whose substantive text is intact and retrievable, not content loss.
- Schedule/preamble/annex chunks (46 of 594) are intentionally kept as single parent==child chunks with no invented boundaries, per the original chunking design — not a defect.

**Statutes:**
- Zero outstanding content-quality defects (re-confirmed in the earlier `statute_coverage_audit.md` deep content audit — 2 trailing heading-fragment artifacts in Guardians and Wards Act §23/§37, both Minor severity, already documented there).

## 3. Title/preamble competition — now confirmed in two of three corpora, not just statutes

The SRI Phase 5 finding (statute lookup queries losing to preamble/§1 chunks that repeat the Act's full title) is a **query-ranking** phenomenon, not a chunk defect — the preamble/title chunks are legitimate, correctly-extracted text. The same *structural* pattern (a short, title-dense administrative chunk existing alongside substantive chunks) exists in all three corpora:
- Statutes: every Act has a `preamble` and `s1` ("short title") chunk.
- Constitution: has a `preamble` chunk.
- Case law: every case has `caption`/header-type chunks repeating the case citation and party names.

Whether this produces the same ranking competition in Constitution and case-law queries is tested in Phase 3/4 of this audit, not assumed from the statute finding.

## 4. Summary

No new content-quality defect was found in this pass beyond what the prior audits already established. The one explicitly disclosed gap is the absence of any near-duplicate (non-exact) scan across the 25,891-chunk case-law corpus — flagged as unproven, not claimed as clean.
