# Final Structure Quality Verification Audit

Read-only verification pass after regenerating the one corrupted file (`0383244195c932e90c4d3520.json`) found by the previous audit. No structured/processed files, code, or database data were modified.

## 1. Corpus completeness

- Expected: **2088**
- Found: **2088**
- Missing: **0**
- Unexpected: **0**

## 2. Exact text preservation

- `exact_preservation`: 2088
- Exact preservation rate: **100.0%**

## 3. Structural validity & semantic quality

Total issues: **3631**
By severity: {'low': 3631}
By category: {'semantic_quality': 838, 'metadata': 2793}

## 4. Regenerated case

- `doc_id`: 0383244195c932e90c4d3520
- `file_size_bytes`: 23506
- `structure_status`: structured
- `used_llm`: True
- `llm_fallback_reason`: 1 batch(es) fell back to paragraph_group
- `validation_ok_recomputed`: True
- `validation_ok_stored`: True
- `preservation`: exact_preservation
- `readiness`: manual_review_required
- `issue_count`: 1

## 5. Fallback verification

- Docs with LLM fallback: **148** (7.09%)
- Docs that used the LLM: **2051**
- Docs needing no LLM call: **37**

## 8. Comparison to previous audit

- `missing_count`: previous=0, current=0
- `preservation_exact`: previous=2087, current=2088
- `preservation_unverifiable`: previous=1, current=0
- `readiness_not_ready`: previous=1, current=0
- `readiness_ready`: previous=1060, current=1060

## 9. Chunking readiness

- `ready_with_caution`: 893 (42.77%)
- `ready`: 1060 (50.77%)
- `manual_review_required`: 135 (6.47%)
