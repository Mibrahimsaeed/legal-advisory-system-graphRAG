# Structure Quality Audit Report

Read-only audit of `var/rag/structured_gemini/` against `var/rag/processed/`. No structured/processed files, code, or database data were modified.

## 1. Corpus completeness

- Expected (processed corpus): **2088**
- Found in structured_gemini/: **2088**
- Missing (not yet structured): **0**
- Unexpected (no processed source): **0**
- All expected documents were present at audit time.

structure_status distribution:
- `structured`: 2056
- `non_judgment_text`: 29
- `incomplete_source`: 2

## 2. Legal text preservation

Classification is derived by reconstructing each document's real `Span` objects from the stored JSON and re-running `src.rag_prep.structure_validate.validate_spans` against the processed source's `full_text` -- not by trusting the stored `validation.ok` flag.

- `exact_preservation`: 2087
- `unverifiable`: 1

## 3-4. Structural validity & semantic quality

Total issues found: **3631**
By severity: {'low': 3630, 'high': 1}
By category: {'semantic_quality': 838, 'metadata': 2792, 'completeness': 1}

`heuristic` issues (semantic_quality category, and the provisions_cited variant check) are candidate flags for human review, not confirmed defects. All other categories are deterministic re-checks against the actual schema/validator.

## 5. Known hard cases (from the original structuring pilot)

- `8dd0ca3d14364404d75b76fe` (headnotes+JUDGMENT, quoted lower-court order): **ready**
- `00ad7248bf77883e6b5975ed` (headnotes+ORDER, prose body): **ready_with_caution**
- `a3ec28ba4900855d15a6e870` (no heading, continuous prose, 22K chars): **ready_with_caution**
- `1d0a3afecc805e3d71cb54b9` (no headnotes, numbered-paragraph body): **ready_with_caution**
- `0513f7c1139b729f4cccf83d` (no headnotes, ORDER, continuous prose): **ready_with_caution**
- `04139076ff36792b38b72f8b` (multi-matter, 111K chars, high quote density): **ready_with_caution**
- `177dbfa9f6115c8c4bb0cdcf` (longest in corpus, 193K chars): **ready_with_caution**
- `f7afd3f61f59d8f9798b8742` (incomplete_source): **ready_with_caution**
- `b3b83bb2ceb23d88cf08394a` (incomplete_source): **ready_with_caution**
- `ae4a07835728a54c3dddb4ad` (non_judgment_text (academic article)): **ready_with_caution**
- `2ee9704f426892ca99e2b92e` (shortest in corpus, 1,698 chars): **ready_with_caution**
- `008d928c2926b2a5855bcc36` (headnotes+JUDGMENT+numbered body): **manual_review_required**

## 7. Fallback and LLM performance

- Documents with an LLM fallback (partial or whole): **147** (7.04% of present docs)
- Documents structured without any LLM call: **37**
- Documents that used the LLM: **2050**
- Per-document LLM call counts / retry counts are NOT stored in the structured JSON itself (only `llm_fallback_reason` and `used_llm` are) -- that telemetry exists only in the runtime `progress.json`/`full_run_report.json` files from each run session, which are process logs, not corpus artifacts. This audit reports what is actually in the corpus.

## 9. Readiness for chunking

Rule-based categories (see `audit_structure_quality.py::audit_one` for the exact logic):

- **not_ready**: a high-severity issue was found (unreadable file, missing required field, full_text/metadata mismatch vs. source, or the independently recomputed validation failed).
- **ready_with_caution**: whole-document identity fallback (`fallback_paragraph_groups`), a deliberately short-circuited `incomplete_source`/`non_judgment_text` document, a partial LLM fallback, a multi-matter candidate, or a heuristic semantic-quality flag -- text is safe and chunkable, but semantic granularity is reduced or needs a second look.
- **manual_review_required**: a medium-severity structural issue was found, or more than 20% of the document's paragraphs fell back to `paragraph_group`.
- **ready**: none of the above.

- `ready_with_caution`: 893 (42.77%)
- `ready`: 1060 (50.77%)
- `manual_review_required`: 134 (6.42%)
- `not_ready`: 1 (0.05%)

## 8. Manual review queue

- Priority (flagged) docs queued: **300**
- Stratified random sample of apparently-clean docs: **40** (seed=42, reproducible)
- Full queue: `manual_review_queue.csv`

## Limitations

- Semantic-quality checks are regex-based heuristics over a small set of patterns (counsel-submission phrasing, quotation-marker presence, disposition vocabulary). They surface *candidates* for review; they do not prove a label is wrong, and they will miss many real mislabels that don't match these specific patterns.
- This audit does not re-run the LLM or re-score confidence; `confidence` values in the stored spans are taken as-is from the structuring run.
- If `missing_count` > 0 above, this audit reflects a snapshot of a still-in-progress corpus, not the final 2,088-document result.
