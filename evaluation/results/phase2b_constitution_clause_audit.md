# Phase 2B — Corpus-Wide Constitution Clause-Boundary Audit

**Source:** `var/rag/constitution_chunked/b427a922990d75f027928c15.json` (read-only). **Method:** a systematic, per-article sweep of every clause-labeled chunk, detecting the exact signature of the two known Phase 2A defects, with every flagged chunk individually classified by reading its actual stored text and source offsets — not by regex pattern-matching alone. Full detection logic in `evaluation/phase2b_constitution_clause_audit.py`.

## 1. Audit scope

- **328 of 328 articles** reported in Phase 2A are covered (confirmed by direct re-count).
- **594 total chunks** examined; **216 chunks carry a `clause_number` label** (the population this audit's checks apply to).
- **46 chunks excluded** from the per-article clause sequence: preamble, schedule, and annex chunks, which carry no `article_number` and sit structurally outside any article's clause sequence.
- Of the 328 articles, **39 have at least one clause-labeled chunk** (the rest are either single-chunk articles with no internal clause subdivision, or articles whose clauses weren't split into separate chunks at all) — **this 39 is the denominator used for every "proportion of articles" figure below**, not 328.
- `src/rag_prep/constitution_parser.py` (the chunk-generation logic) was read, not modified, to understand how `clause_number` and `source_start`/`source_end` are produced, informing the detection method below.

## 2. Detection method

For each of the 39 articles with clause labels, chunks were sorted by `source_start` and scanned in document order, tracking a running maximum of the highest clause value seen (letter suffixes like "4A" ordered between 4 and 5). Two independent checks were applied:

- **(a) Clause-number regression:** a chunk's own clause-number *start* value is lower than the running max, **and** it is textually contiguous with the immediately preceding chunk (`source_start` exactly equals that chunk's `source_end`) — the exact signature of `art_199_c1_2` and `art_224a_c2_2`.
- **(b) Backwards range label:** a chunk's own declared range *end* is lower than its own declared *start* (e.g. `"(5)–(2)"`) — the signature of a cross-reference being misread as marking a range's end.

Every flagged chunk was then individually classified by reading its actual `canonical_text` excerpt: a genuine new clause reads `"(N) <Capitalised operative text>"`; a mislabeled continuation reads `"(N), <lowercase continuation>"` or similar — the latter was required (not just the numeric trigger alone) for a `confirmed_defect` classification.

## 3. Results

| Category | Count | Denominator |
|---|---:|---|
| Confirmed defect | **4** | out of 216 clause-labeled chunks |
| Probable defect | **1** | out of 216 clause-labeled chunks |
| Needs manual review | 0 | — |
| Not a defect (ruled out during triage) | 0 | — |
| **Total flagged** | **5** | **5/216 clause-labeled chunks; 5/39 articles with clause labels (12.8%); 5/328 total articles (1.5%)** |

**Both known Phase 2A examples were correctly re-detected** (`art_199_c1_2`, `art_224a_c2_2` — cross-check passed), confirming the detection method is sound, and **2 previously-unknown confirmed instances were found**: `art_59_c1_2` and `art_185_c2_2`.

### Confirmed defects (4)

| Article | Chunk ID | Label | Preceding chunk | Text excerpt |
|---|---|---|---|---|
| 59 | `art_59_c1_2` | "1" | `art_59_c3` (ends exactly where this starts) | *"(1), seven shall retire after the expiration of the first three years..."* |
| 185 | `art_185_c2_2` | "2" | `art_185_c2` (range "2)–(3", ends exactly where this starts) | *"(2) does not apply shall lie only if the Supreme Court grants leave to appeal..."* |
| 199 | `art_199_c1_2` | "1" | `art_199_c4a` (range "4A)–(4B", ends exactly where this starts) | *"(1), the High Court has made an interim order shall be disposed of..."* |
| 224A | `art_224a_c2_2` | "2" | `art_224a_c4` (range "4)–(5", ends exactly where this starts) | *"(2), if the members of the Opposition are less than five..."* |

All four share the identical mechanism: an inline cross-reference to an earlier clause number (e.g. "...referred to in clause (1)...", "...to which clause (2)...") appearing inside a *later* clause's own text is misread as the start of a *new* clause with that lower number, truncating the real clause's continuation into a separate, mislabeled chunk. **Content status: mislabeled, not lost** — the substantive text is retrievable, just under the wrong clause identifier and in a syntactically truncated (mid-sentence) form.

### Probable defect (1)

| Article | Chunk ID | Label | Note |
|---|---|---|---|
| 270AA | `art_270aa_c5` | "5)–(2" | Chunk's own text *does* open genuinely ("(5) No suit, prosecution or other legal proceedings...") — unlike the 4 confirmed cases, the start is not mislabeled. Only the declared range *end* ("–(2)") is nonsensical, most likely from a "(2)" cross-reference inside clause 5's body being misread as marking where the chunk's range ends. Classified as probable rather than confirmed because the practical content impact (if any) was not independently traced to a specific missing or merged span in this pass. |

## 4. Scope analysis

- **Confirmed-defect articles: 4 of 39 (10.3%) with clause labels; 4 of 328 (1.2%) of all articles.**
- **All 5 flagged items (confirmed + probable) share the same plausible common root cause**: an inline numeric cross-reference to another clause being misinterpreted by the clause-boundary/range-detection logic as a structural marker, rather than being recognized as prose referring elsewhere. This is a single underlying pattern, not several unrelated ones.
- **No clustering by amendment era or chapter was found** — the 5 affected articles (59, 185, 199, 224A, 270AA) span different Parts (The Federation of Pakistan; The Judicature ×2; General Provisions; Miscellaneous/Transitional) and different eras of amendment, suggesting the trigger is the generic textual pattern ("clause (N)" appearing mid-sentence) rather than anything specific to one amendment act or drafting style.
- **This is a narrow, specific, now well-characterized defect — not evidence of broad corpus-wide parsing failure.** 211 of 216 clause-labeled chunks (97.7%) show no evidence of this pattern under the detection method applied.

## 5. Final response

1. **Overall audit status:** read-only sweep completed successfully; defect confirmed as narrow and well-characterized, not corpus-wide.
2. **Total articles and chunks examined:** 328 articles (all of Phase 2A's reported total), 594 total chunks, 216 clause-labeled chunks within 39 articles.
3. **Counts:** 4 confirmed, 1 probable, 0 needing manual review.
4. **Isolated or broader pattern?** Broader than the 2 known cases (2 new confirmed instances found), but still narrow overall — 5 of 328 articles (1.5%), all one shared root cause (clause-number cross-references misread as boundaries).
5. **Top affected article/chunk IDs:** Art. 59 (`art_59_c1_2`), Art. 185 (`art_185_c2_2`), Art. 199 (`art_199_c1_2`), Art. 224A (`art_224a_c2_2`), Art. 270AA (`art_270aa_c5`, probable).
6. **New artifact paths:** `evaluation/phase2b_constitution_clause_audit.py`, `evaluation/results/phase2b_constitution_clause_audit.json`, `evaluation/results/phase2b_constitution_clause_audit.md`, `evaluation/results/phase2b_constitution_clause_audit.csv`.
7. **Confirmation:** no production file, chunk, embedding, or Qdrant data was modified — `git diff --stat` is empty before and after this audit; no Qdrant call of any kind was made (this audit used only the local chunk file).
