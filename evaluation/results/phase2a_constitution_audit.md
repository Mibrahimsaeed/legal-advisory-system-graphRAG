# Phase 2A — Constitution Corpus Index & Coverage Audit

**Audit date:** 2026-10-18 · **Repo commit:** `58def993f8a8d15e30680011e9fac05ee38a5946`
**Artifacts inspected:** `var/rag/constitution_chunked/b427a922990d75f027928c15.json`, `var/rag/index_meta.json`, `var/rag/docstore.sqlite`, live Qdrant collection `pk_legal`.
**Method:** read-only file reads, a read-only SQLite connection (`file:...?mode=ro`), and read-only Qdrant calls (`points/count`, `points/scroll`, `query_points`, `get_collection`). No insert/upsert/delete/recreate call exists anywhere in the audit script. Full detail in `evaluation/phase2a_constitution_audit.py`.

## 1. Overall status: **CONDITIONAL PASS**

Coverage, docstore/Qdrant reconciliation, point-ID generation, vector configuration, and the retrieval smoke test all passed cleanly — the Constitution corpus is, on the evidence gathered here, in substantially better shape than the statute corpus examined in earlier work, and the prior Family-Courts-Act dense-retrieval failure does **not** generalize to the Constitution (confirmed, not assumed — all 6 smoke-test queries retrieved their expected article at rank 1–3). The CONDITIONAL applies because of one confirmed, reproducible structural defect (clause mislabeling, found in 2 of 328 articles) and one previously-reported pair of concerns (41 "ghost" chunks, 45 duplicate IDs) that are confirmed **resolved**, not merely assumed so.

## 2. Coverage reconciliation

| Check | Value |
|---|---:|
| Chunked file: total chunks | 594 |
| Chunked file: retrievable chunks | 547 |
| Docstore: total / retrievable (constitution) | 594 / 547 |
| `index_meta.json` recorded constitution count | 547 |
| Live Qdrant constitution point count | 547 |
| **All four counts reconcile exactly** | **Yes** |

Non-retrievable breakdown (47 chunks): 41 `article` parents (large articles split into retrievable children), 5 `schedule` parents (same pattern), 1 `preamble` parent. None of these are unexplained gaps — each has confirmed retrievable children covering its content.

No unmatched points were found in either direction: a 6-sample point-ID spot check (computed via the project's own `uuid5(ID_NAMESPACE, chunk_id)` scheme, cross-checked against `index_meta.json`'s recorded namespace) matched the live Qdrant point exactly for every sample, including correct `source_type: constitution` payload tagging.

## 3. Top verified issues, ranked by severity

### Issue 1 — Confirmed, Medium severity: inline clause cross-references mislabeled as new clause boundaries (2 instances found)

**Article 199** (`art_199_c1_2`) and **Article 224A** (`art_224a_c2_2`) each contain a chunk whose `clause_number` metadata and `context_header` claim a clause number that does **not** match its actual position in the source text:

- `art_199_c1_2`: labeled "Clause (1)", but its `source_start` (199299) is exactly `art_199_c4a`'s `source_end` — it is the direct, contiguous tail of clause (4A)/(4B)'s own proviso (the "disposed of... within six months" rule for interim orders), not clause (1). Its own text begins mid-sentence: `"(1), the High Court has made an interim order shall be disposed of..."` — the true sentence opening (`"Where ... under clause"`) is missing, because the parser treated the inline cross-reference `"under clause (1)"` as a new clause-1 boundary and split there.
- `art_224a_c2_2`: labeled "Clause (2)", same signature — directly follows `art_224a_c4`'s end, its text is the tail of clause (4)-(5), beginning `"(2), if the members of the Opposition are less than five..."`.

Both chunk IDs are unique (no duplicate-chunk_id defect — that check passed cleanly, confirming the `_filter_monotonic`-style protection documented earlier in this project does prevent ID collisions), but the clause-number disambiguation logic appears to take a "append a numeric suffix" path instead of correctly rejecting an out-of-sequence, already-seen clause number and merging the text into the preceding clause. **Evidence:** exact chunk IDs, source offsets, and quoted text above — directly reproducible from the current chunk file. **Retrieval impact:** the six-month interim-order rule (Art. 199) and the Opposition-committee-membership rule (Art. 224A) are not lost — they are retrievable, just under a misleading label and in a syntactically truncated form that is not fully self-contained if retrieved alone.

### Issue 2 — Re-checked and confirmed RESOLVED: previously-reported 41 ghost chunks / 45 duplicate IDs

Direct, fresh re-inspection of the current chunk file found **zero** empty/whitespace-only chunks and **zero** duplicate `chunk_id` values anywhere in the corpus. This is not assumed from a prior session's memory — it was independently re-verified against today's artifacts.

### Issue 3 — Verified, no defect: Articles 63, 89, 175A, 260 structurally sound

All four previously-flagged articles were rechecked. Article 175A's clause range `(8)–(18)` chunk (`art_175a_c8`) legitimately bundles clauses 8, 11, and 18 (the only clauses in that span with real text; 9, 10, and 12–17 are genuinely omitted, confirmed via inline `* * * * * * *` omission markers) and is **honestly labeled** with the full range in its `context_header` — this is the same accepted "omitted-range bundling" convention already validated elsewhere in this project, not a defect. Articles 63, 89, and 260 have zero empty chunks and clean, gap-free clause splits.

**Category:** B (chunk structure/metadata problem) for Issue 1; this issue was previously filed as A/B but is now E (resolved, re-verified) for Issue 2; Issue 3 is a verified non-finding.

## 4. Retrieval smoke-test results

Qdrant was available throughout. Dense-only and hybrid were run separately; no formal Recall@K/MRR/nDCG is claimed (no relevance-judged Constitution query set exists in this project yet — this is explicitly a smoke test, not a benchmark).

| ID | Type | Expected article (verified from corpus) | Dense rank | Hybrid rank |
|---|---|---|---:|---:|
| C1 | Specific article | Art. 199 — Jurisdiction of High Court | 1 | 2 |
| C2 | Right, ordinary language | Art. 9 — Security of person | 1 | 1 |
| C3 | Parliament / legislative process | Art. 70 — Introduction and passing of Bills | 1 | 1 |
| C4 | Judiciary / judicial appointments | Art. 175A — Appointment of Judges | 2 | 3 |
| C5 | Federal/provincial powers | Art. 142 — Subject-matter of Federal and Provincial laws | 1 | 1 |
| C6 | Specific clause/proviso | Art. 63 — Disqualifications (imprisonment proviso) | 1 | 1 |

**All six queries retrieved their expected article within the top 3 of both dense and hybrid retrieval.** This stands in clear contrast to the statute corpus's Q01/Q04 findings from earlier sessions — which is exactly why those findings were not generalized here without independent evidence, per this task's own framing.

## 5. What the evidence establishes, and what remains unknown

**Established:** corpus coverage is fully reconciled across all four systems checked; the two previously-reported defects do not reproduce today; two real instances of a clause-mislabeling pattern exist and are precisely located; dense and hybrid retrieval both perform well on 6 diverse, corpus-grounded smoke-test queries.

**Unknown / not checked in this pass:** whether the Article 199/224A mislabeling pattern recurs elsewhere among the 328 articles (only found opportunistically during inspection, not via a systematic sweep); whether the smoke test's strong result holds for a larger, harder, or adversarially-phrased query set; footnote citation text for the Article 175A omission markers was not independently pulled (though the chunk's own labeling was confirmed honest regardless).

## 6. Recommended next diagnostic

A systematic, read-only sweep across all 328 articles checking for the same "inline clause cross-reference mislabeled as a new boundary" signature (chunk whose `source_start` exactly equals a *different*-numbered preceding chunk's `source_end`, and whose own `clause_number` is lower than that preceding chunk's) — to establish whether Issue 1 is confined to these 2 instances or more widespread, before deciding whether a targeted parser fix is warranted.

---
*No production file, chunk, embedding, or Qdrant data was modified in the course of this audit.*
