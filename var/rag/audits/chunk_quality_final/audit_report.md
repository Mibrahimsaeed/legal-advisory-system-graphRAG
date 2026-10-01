# Full Chunk Quality Audit Report

Read-only audit of `var/rag/chunked/` against `var/rag/structured_gemini/`. No chunked/structured/processed/raw files, code, or database data were modified.

## 1. Corpus completeness

- Structured cases: **2088**
- Chunked cases: **2088**
- Missing: **0**
- Unexpected: **0**

## Integrity (independently recomputed, not trusting stored validation.ok)

- `exact_text_preservation_cases`: 2088
- `offset_or_integrity_failures`: 0
- `stored_vs_recomputed_disagreements`: 0
- `cases_with_missing_content`: 0
- `cases_with_unintended_duplication`: 0
- `cases_with_intentional_overlap`: 5

## Diagnostics

- `oversized_chunk_count`: 16
- `overlap_chunk_count`: 5

## Size distribution (words/chunk, independently recomputed from chunk text)

- min=1 median=79.0 mean=119.62 p90=275 p95=376 max=2186
- `<250`: 25781 (87.77%)
- `250-499`: 2843 (9.68%)
- `500-800`: 609 (2.07%)
- `801-1000`: 123 (0.42%)
- `>1000`: 16 (0.05%)

## Section distribution

- `headnotes`: 1026 chunks (3.49%), present in 1002 cases
- `procedural_history`: 3184 chunks (10.84%), present in 1769 cases
- `facts`: 1688 chunks (5.75%), present in 1407 cases
- `lower_court_orders`: 830 chunks (2.83%), present in 656 cases
- `arguments`: 2684 chunks (9.14%), present in 1555 cases
- `court_reasoning`: 4849 chunks (16.51%), present in 1713 cases
- `quoted_material`: 2607 chunks (8.88%), present in 965 cases
- `final_order`: 4579 chunks (15.59%), present in 2052 cases
- `issues`: 461 chunks (1.57%), present in 414 cases
- `evidence`: 407 chunks (1.39%), present in 321 cases
- `authorities_cited`: 1207 chunks (4.11%), present in 609 cases
- `applicable_law`: 513 chunks (1.75%), present in 374 cases
- `unclassified`: 4916 chunks (16.74%), present in 243 cases
- `findings`: 421 chunks (1.43%), present in 373 cases

## Readiness

- `ready_with_caution`: 2062 (98.75%)
- `ready`: 26 (1.25%)

## Issues

Total: **0**, by severity: {}, by category: {}

## Overlap chunks (all, individually)

- `48f5291ac2389d56e82e2774` / `48f5291ac2389d56e82e2774:0036` section=court_reasoning overlap_paragraph_count=1 word_count=315
- `68b47bed1b2a0c1f80a96b09` / `68b47bed1b2a0c1f80a96b09:0007` section=quoted_material overlap_paragraph_count=4 word_count=682
- `90c5605b1bcf1ffac717738c` / `90c5605b1bcf1ffac717738c:0012` section=court_reasoning overlap_paragraph_count=1 word_count=239
- `a1d0526a0bba286fe79c062b` / `a1d0526a0bba286fe79c062b:0004` section=procedural_history overlap_paragraph_count=1 word_count=388
- `c9ecc8f245ef961725449102` / `c9ecc8f245ef961725449102:0015` section=quoted_material overlap_paragraph_count=1 word_count=105

## Oversized chunks (all, individually)

- `054d6e159a91066a85d84501` / `054d6e159a91066a85d84501:0006` section=court_reasoning paragraph=30 word_count=1299
- `1545b73b6433b9ebcdbcca05` / `1545b73b6433b9ebcdbcca05:0007` section=quoted_material paragraph=33 word_count=1293
- `1782fdc21be1d4739c9f2cce` / `1782fdc21be1d4739c9f2cce:0005` section=issues paragraph=18 word_count=1135
- `3cc421e9daea9bf08d060bd3` / `3cc421e9daea9bf08d060bd3:0002` section=facts paragraph=17 word_count=1103
- `53da9b198973f615af28245a` / `53da9b198973f615af28245a:0003` section=arguments paragraph=38 word_count=2186
- `7ea304ebd8a9924ed887bebb` / `7ea304ebd8a9924ed887bebb:0015` section=arguments paragraph=82 word_count=1026
- `93dc07ba5826cbd0bb20d580` / `93dc07ba5826cbd0bb20d580:0002` section=unclassified paragraph=15 word_count=1063
- `c6c96c34f36ace3be0b21b9f` / `c6c96c34f36ace3be0b21b9f:0007` section=quoted_material paragraph=26 word_count=1175
- `c883689576ecd494c655dbd5` / `c883689576ecd494c655dbd5:0006` section=court_reasoning paragraph=23 word_count=1045
- `c9e63146079eebe00e470024` / `c9e63146079eebe00e470024:0006` section=court_reasoning paragraph=18 word_count=1137
- `d55362ae8e4bb2e538d5198d` / `d55362ae8e4bb2e538d5198d:0003` section=facts paragraph=30 word_count=1372
- `d80230f67c75dc41495de3b9` / `d80230f67c75dc41495de3b9:0002` section=final_order paragraph=17 word_count=1513
- `e64da0ced58bdcdfc2a63875` / `e64da0ced58bdcdfc2a63875:0003` section=arguments paragraph=19 word_count=1277
- `e9397169d0b054d6fbdb95f7` / `e9397169d0b054d6fbdb95f7:0004` section=court_reasoning paragraph=27 word_count=1185
- `eeade9639a852bcd4d9e795e` / `eeade9639a852bcd4d9e795e:0011` section=court_reasoning paragraph=32 word_count=1126
- `f6103e433d33895f8a6fdbef` / `f6103e433d33895f8a6fdbef:0054` section=unclassified paragraph=54 word_count=1086
