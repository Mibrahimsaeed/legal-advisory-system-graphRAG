# Statute Corpus Coverage Audit — Read-Only

**Scope:** corpus coverage and retrieval diagnosis for the 5 pilot evaluation queries, focused on
the weak statute results for Q03 and Q04. This is a corpus/retrieval audit, not a legal opinion —
no claim here establishes current law.

**Path note:** the requested path `var/rag/final_Data/statutues_chunked` does not exist. The actual,
live statute chunk directory — confirmed by cross-checking its retrievable-chunk count (240) against
the live Qdrant `source_type="statute"` point count (240, exact match) — is:

```
var/rag/statutes_chunked/
```

## A. Executive summary

- **The 4 statutes named in the task, plus 4 more directly relevant to the 5 queries, are all
  present** in `var/rag/statutes_chunked/` and are indexed in Qdrant. Corpus coverage is not the
  primary problem for Q03/Q04.
- **Q04 is a confirmed retrieval/ranking failure, not a coverage gap.** The two provisions most
  directly on-point for "evidence required to recover dowry articles" — Dowry and Bridal Gifts
  (Restriction) Act, 1976, §3 (restriction on dowry value) and §5 ("Vesting of dowry etc., in the
  bride", the statutory basis for a bride's ownership and hence her right to recover it) — exist in
  the corpus, are marked `retrievable: true`, and are confirmed present in Qdrant by direct point
  lookup. Neither was retrieved in Q04's top-8 statute results; instead, all 8 results are Divorce
  Act, 1869 sections (a Christian-marriage procedural statute, structurally and substantively
  unrelated to Muslim dowry law). All 8 are labeled `relevance=0`, consistent with Precision@5 = 0.0.
- **Q03 is a mixed finding: partly a genuine content gap, partly a retrieval miss.** The corpus has
  no single, clean statutory clause stating a father's maintenance obligation to his children in
  those terms. The closest retrievable provisions — Muslim Family Laws Ordinance, 1961 §9
  ("Maintenance", text speaks only of a husband's obligation to his *wife*) and Guardians and Wards
  Act, 1890 §24 ("Duties of guardian of the person", which does say a guardian "must look to
  [the ward's] support, health and education") — are indexed in Qdrant, but only §9 was retrieved
  (at rank 5, correctly labeled `relevance=1`); §24, arguably the better-worded match, was not
  retrieved at all.
- **No indexing failure was found** for any chunk examined in this audit — every chunk checked
  (6 specific `chunk_id`s, plus the full statute-source-type count) that exists in the local corpus
  as `retrievable: true` was confirmed present in Qdrant via a direct read-only lookup.
- **Highest-priority finding:** statute macro Precision@5 (0.36) and nDCG@10 (0.527) are far below
  case-law's (1.0 / 0.885) across all 5 pilot queries (`evaluation/results/metrics_pilot.json`), and
  the worst queries (Q03, Q04) show relevant, indexed statute chunks sitting outside the fetched
  window rather than being absent — pointing at the statute side of hybrid retrieval/ranking as the
  first place to investigate, not the statute corpus content itself.

## B. Corpus inventory

`var/rag/statutes_chunked/` contains 8 JSON files, one per statute, each with keys
`doc_id, act, metadata, cleaned_text, footnotes, outline, glossary, chunks, validation`. Each entry
in `chunks` carries (among others) `chunk_id, parent_id, unit_type, is_parent, retrievable, act_name,
section_number, section_title, section_status, canonical_text, context_header, embedding_text,
bm25_text, footnote_ids, references`. Non-substantive units (footnotes blocks, schedule-of-forms
appendices) and large sections that are split into per-subsection child chunks are marked
`retrievable: false` on the parent/footnote entry itself — this is a deliberate parent/child design
already used elsewhere in the pipeline (confirmed against the Constitution chunker's identical
`is_parent` convention), not a defect.

| Statute (file) | Found? | Total chunks | Retrievable chunks | Notable quality issues |
|---|---|---|---|---|
| Dissolution of Muslim Marriages Act, 1939 (`0425160f...`) | Confirmed | 11 | 9 | None observed in sampled chunks |
| Family Courts Act, 1964 (`24520c34...`) | Confirmed | 43 | 39 | None observed in sampled chunks |
| Dowry and Bridal Gifts (Restriction) Act, 1976 (`417403be...`) | Confirmed | 14 | 13 | §7 title is literally `"Display of Dowry etc [omitted]"` — correctly flagged `section_status: omitted`, not a bug |
| Guardians and Wards Act, 1890 (`42769a60...`) | Confirmed | 64 | 60 | None observed in sampled chunks |
| Child Marriage Restraint Act, 1929 (`470c476c...`) | Confirmed | 14 | 13 | None observed in sampled chunks |
| Muslim Family Laws Ordinance, 1961 (`6867f44e...`) | Confirmed | 19 | 16 | None observed in sampled chunks |
| Divorce Act, 1869 (`8251b5f4...`) | Confirmed (not named in task, but relevant-by-overretrieval — see §D) | 87 | 82 | Largest single statute in the corpus; governs *Christian* divorce procedure, not Muslim personal law — see retrieval-mismatch finding below |
| Application Act, 1962 (West Pakistan Muslim Personal Law (Shariat) Application Act) | Confirmed (not named in task, plausibly relevant to Q05) | 9 | 8 | None observed in sampled chunks |

**Total: 261 chunks, 240 retrievable.** The 240 retrievable count exactly matches the live Qdrant
`source_type="statute"` point count (`points/count` with an exact filter), confirming the full
retrievable set from this directory is the set actually indexed — not a stale or partial index.

No statute named in the task is "missing under a different filename" — all four were located by
reading each file's `act.name` field directly, not inferred from filenames.

## C. Query-to-provision coverage matrix

| Query | Relevant statute/provision | Present in chunks? | Chunk ID/path | Quality | Likely issue |
|---|---|---|---|---|---|
| Q01 (khula without returning dower) | Family Courts Act §Schedule Part I item 1 ("Dissolution of marriage, including Khula") | Confirmed | `24520c344d2285555aa10988:schedule:partI` | Retrieved (rank 1), but it is a one-line jurisdiction list with no substantive text on dower — judged `relevance=0` by the human labeler, a defensible call | Terminology/granularity mismatch — the chunk confirms khula is justiciable but cannot answer whether dower must be returned |
| Q01 | Dissolution of Muslim Marriages Act §5 ("Rights to dower not to be affected on dissolution... under this Act") | Confirmed | `0425160ffedaefd8c79b3241:s5` | Retrievable, indexed, **not retrieved** in Q01's top 8 | Unverified whether this would have scored higher if retrieved — it addresses dissolution generally, not khula specifically, so relevance is uncertain even if surfaced |
| Q01 | A dedicated statutory clause on dower-return-on-khula specifically | Not found | — | — | Likely a genuine gap: in Pakistani law this point is substantially case-law/jurisprudence-driven rather than a single codified clause in these 8 acts |
| Q02 (child custody jurisdiction) | Family Courts Act §Schedule Part I item 5 ("Custody of children") | Confirmed | `24520c344d2285555aa10988:schedule:partI` | Retrieved (rank 2), judged `relevance=2` | No issue — correctly retrieved and highly relevant |
| Q02 | Guardians and Wards Act §9 ("Court having jurisdiction to entertain application") | Confirmed | `42769a60dc59c5473affbc32:s9` | Retrieved (rank 7), judged `relevance=2` | No issue |
| Q02 | Divorce Act §§41–43 (custody orders in separation/dissolution suits) | Confirmed | `8251b5f4daa92e5a22319fc4:s41/s42/s43` | Retrieved (ranks 1,3,8), judged `relevance=1` each | Terminology note — relevant only insofar as this is a cross-cutting custody mechanism in a Christian-marriage statute; still judged partially relevant, not an error |
| Q03 (father's responsibility for child maintenance) | Muslim Family Laws Ordinance §9 ("Maintenance") | Confirmed | `6867f44e86ef03cec19831c6:s9` | Retrieved (rank 5), judged `relevance=1` | Terminology mismatch — the chunk's own text names only "wife," not "child"/"father" |
| Q03 | Guardians and Wards Act §24 ("Duties of guardian of the person" — "must look to his support, health and education") | Confirmed, retrievable, indexed | `42769a60dc59c5473affbc32:s24` | **Not retrieved** in Q03's top 8 | Retrieval/ranking issue — this wording is arguably a closer match to "legally responsible for... maintenance" than anything that was retrieved |
| Q03 | A single codified "father is responsible for child maintenance" clause | Not found | — | — | Likely a genuine content gap in this 8-act corpus; the real-world basis is distributed across MFLO §9 (wife-framed) and uncodified Muslim personal law/case law |
| Q04 (evidence to recover dowry articles) | Dowry and Bridal Gifts (Restriction) Act §5 ("Vesting of dowry etc., in the bride") | Confirmed, retrievable, indexed | `417403be7fc29b06cafadbbe:s5` | **Not retrieved** in Q04's top 8 | Retrieval/ranking issue — establishes the ownership basis a recovery claim rests on |
| Q04 | Dowry and Bridal Gifts (Restriction) Act §3 (dowry value restriction) | Confirmed, retrievable, indexed | `417403be7fc29b06cafadbbe:s3` | **Not retrieved** in Q04's top 8 | Retrieval/ranking issue |
| Q04 | Dedicated evidentiary-standard provisions ("what evidence is required") | Not found | — | — | Genuine coverage gap — no Evidence Act / Qanun-e-Shahadat Order equivalent exists anywhere in this statute corpus; this specific angle of the query may be structurally unanswerable from statutes alone and would need the case-law corpus or an added evidence-law source |
| Q05 (grounds for dissolution of Muslim marriage) | Dissolution of Muslim Marriages Act §2 (all clauses, grounds for decree) | Confirmed | `0425160ffedaefd8c79b3241:s2_ss...` (2 child chunks) | Retrieved (ranks 1–2), judged `relevance=2` each | No issue — correctly retrieved and highly relevant |

## D. Q03 and Q04 retrieval diagnosis

### Q03 — "Is a father legally responsible for child maintenance?"
Macro result: Precision@5 = 0.2, nDCG@10 ≈ 0.387 (`evaluation/results/metrics_pilot.json`).

Retrieved statute chunks (from `evaluation/results/judging_pilot.csv`), in rank order:

| Rank | Chunk ID | Relevance | Statute |
|---|---|---|---|
| 1 | `42769a60dc59c5473affbc32:s21` | 0 | Guardians and Wards Act §21 (minors as guardians) |
| 2 | `24520c344d2285555aa10988:schedule:partI` | 0 | Family Courts Act Schedule (jurisdiction list) |
| 3 | `470c476c4195cbc08e94a6da:s6` | 0 | Child Marriage Restraint Act §6 (punishment for child-marriage guardians) |
| 4 | `6867f44e86ef03cec19831c6:s4_ss1-2-3-4-5` | 0 | MFLO §4 (succession) |
| 5 | `6867f44e86ef03cec19831c6:s9` | 1 | MFLO §9 (maintenance — wife-framed) |
| 6 | `470c476c4195cbc08e94a6da:preamble` | 0 | Child Marriage Restraint Act preamble |
| 7 | `470c476c4195cbc08e94a6da:s5` | 0 | Child Marriage Restraint Act §5 |
| 8 | `470c476c4195cbc08e94a6da:s4` | 0 | Child Marriage Restraint Act §4 |

A corpus-wide text search for `"maintenance"` co-occurring with `"child"`/`"minor"` surfaced
`Guardians and Wards Act §24` — "A guardian of the person of a ward... must look to his support,
health and education" — which was confirmed `retrievable: true` in the local file and confirmed
present in Qdrant via a direct `chunk_id` scroll lookup, yet is absent from all 8 retrieved rows.

**Diagnosis:** mixed. (1) **Confirmed retrieval miss** — §24 is indexed and arguably the best
semantic match in the corpus, but never surfaced. (2) **Plausible content gap** — even if §24 had
been retrieved, no chunk in this corpus states a father's maintenance duty as cleanly as the query
implies; §9's own wording is wife-specific. Distinguishing "ranking pushed the right chunk out" from
"nothing in the corpus says this well enough to rank highly" would require re-running retrieval with
§24's `chunk_id` forced into the candidate set and inspecting its raw RRF/dense/sparse sub-scores —
not done here, since that is an experiment, not a read of existing data. **This remains a hypothesis,
not a confirmed root cause**, for the ranking-vs-content split specifically; the fact that both
provisions exist in the index is confirmed.

### Q04 — "What evidence is required to recover dowry articles after separation?"
Macro result: Precision@5 = 0.0, nDCG@10 = 0.0.

Retrieved statute chunks, all 8 ranks:

| Rank | Chunk ID | Relevance | Statute |
|---|---|---|---|
| 1 | `8251b5f4daa92e5a22319fc4:s47` | 0 | Divorce Act §47 (stamp duty on petitions) |
| 2 | `8251b5f4daa92e5a22319fc4:s52` | 0 | Divorce Act §52 (spousal evidence re: cruelty/desertion) |
| 3 | `8251b5f4daa92e5a22319fc4:s39` | 0 | Divorce Act §39 (settlement of wife's property) |
| 4 | `8251b5f4daa92e5a22319fc4:s60` | 0 | Divorce Act §60 |
| 5 | `8251b5f4daa92e5a22319fc4:s26` | 0 | Divorce Act §26 |
| 6 | `8251b5f4daa92e5a22319fc4:s25` | 0 | Divorce Act §25 |
| 7 | `8251b5f4daa92e5a22319fc4:s24` | 0 | Divorce Act §24 |
| 8 | `8251b5f4daa92e5a22319fc4:s22` | 0 | Divorce Act §22 |

Every single retrieved statute result for Q04 comes from one statute — the Divorce Act, 1869, a
colonial-era Christian-marriage procedural act — and every one is labeled irrelevant. A corpus-wide
text search for `"dowry"`/`"bridal gift"` found the Dowry and Bridal Gifts (Restriction) Act, 1976 in
full (preamble + §§1,2,3,5,6,8,9 + footnotes), including §5 ("Vesting... in the bride," the statutory
ownership basis for a recovery claim) and §3 (value restriction), both `retrievable: true` and both
confirmed present in Qdrant by direct `chunk_id` lookup — neither appears anywhere in Q04's results.

**Diagnosis:** confirmed retrieval/ranking issue, not a coverage gap, for the "which statute governs
dowry ownership" part of the question. The corpus has the right statute; it was not surfaced at all
in favor of an entire unrelated statute monopolizing all 8 slots — which also suggests the Divorce
Act's sheer size (87 chunks, the largest in the corpus) may be contributing disproportionate
candidate volume relative to the much smaller Dowry Act (14 chunks), though this is a **hypothesis**
about candidate-pool composition, not something this read-only audit measured directly (would
require inspecting per-source candidate counts before RRF fusion, not just final results).
Separately, **the "what evidence is required" framing of the query has no statutory home at all** in
this corpus — no Evidence Act/Qanun-e-Shahadat equivalent exists among the 8 statutes — which is a
genuine, confirmed coverage gap for that specific half of the question, independent of the ranking
issue.

## E. Findings by category

**Missing statutory coverage**
- No evidence-law statute (e.g., Qanun-e-Shahadat Order, 1984) exists anywhere in the corpus —
  confirmed absent. Relevant to the evidentiary half of Q04.
- No single codified clause states a father's maintenance duty to his children in those terms —
  confirmed absent from all 8 acts; the closest analogues are wife-framed (MFLO §9) or
  guardian-framed (Guardians and Wards Act §24).
- No dedicated clause on whether dower must be returned specifically for *khula* (as distinct from
  judicial dissolution under DMMA 1939 §2) was found — confirmed absent; DMMA §5 addresses dower
  rights on dissolution generally but not khula by name.

**Extraction or chunking defects**
- None found in the chunks sampled for this audit. The `retrievable: false` markings on parent
  sections, footnote blocks, and schedule-of-forms entries are deliberate design (parent/child
  chunking, consistent with the same convention used in the Constitution pipeline), not defects.

**Terminology mismatch**
- Q01: query says "khula"/"dower return"; the only directly on-point chunk retrieved is a one-line
  Schedule entry, not substantive text.
- Q03: query says "father... child maintenance"; the retrieved, partially-relevant chunk (MFLO §9)
  textually says "husband... wife," and the better-worded chunk (Guardians and Wards Act §24,
  "support, health and education") was not retrieved at all.

**Potential indexing mismatch**
- None confirmed. Every chunk checked in this audit (6 specific `chunk_id`s spanning 4 statutes,
  plus the full 240-point aggregate count) that is `retrievable: true` in the local corpus was found
  present in Qdrant via direct read-only lookup. This category is ruled out for everything examined
  here, though only a sample of chunks (not the full 240) was individually spot-checked.

**Potential retrieval/ranking issue**
- Q04: Dowry Act §§3 and 5 — indexed, on-topic, not retrieved; all 8 slots instead went to one
  unrelated statute (Divorce Act, 1869). This is the strongest, best-evidenced finding in this audit.
- Q03: Guardians and Wards Act §24 — indexed, arguably on-topic, not retrieved.

**No confirmed issue**
- Q02 and Q05 statute retrieval: the clearly relevant provisions for both (custody jurisdiction;
  dissolution grounds) were retrieved and labeled `relevance=1`/`2`, consistent with their
  comparatively strong macro scores (Q02 P@5=0.6, Q05 P@5=0.8).

## F. Recommended next actions

Ordered by evidence strength, strongest first. No recommendation below has been implemented.

1. **Inspect per-source candidate composition before RRF fusion for Q03 and Q04.**
   Evidence: Dowry Act §§3/5 and Guardians and Wards Act §24 are confirmed indexed but absent from
   final top-8 statute results; Divorce Act chunks dominate Q04's results despite being topically
   unrelated. Component to inspect: `hybrid_search()`'s `candidates` parameter and the dense/sparse
   sub-scores for these specific `chunk_id`s, in `src/rag_prep/embed_index.py` (read-only
   inspection — this audit did not modify or re-run that logic).
2. **Check whether `embedding_text`/`bm25_text` for Dowry Act §5 and Guardians and Wards Act §24
   contain the vocabulary a query-rewriter or BM25 term-match would need.**
   Evidence: §5's `canonical_text` says "vest absolutely in the bride," never "recover" or "evidence";
   §24 says "support, health and education," never "maintenance" or "responsible." Component:
   `embedding_text`/`bm25_text` fields in `var/rag/statutes_chunked/417403be7fc29b06cafadbbe.json`
   and `42769a60dc59c5473affbc32.json` — this is a plausible contributor to low lexical/sparse match,
   not confirmed as causal without re-running sparse scoring in isolation (not done here).
3. **Consider whether a query-expansion or synonym layer is needed for Q01/Q03-style queries**
   before concluding statute coverage itself is insufficient. Evidence: both queries have at least
   one on-topic chunk in the corpus that uses materially different vocabulary from the query. This
   is a design decision for the retrieval team, not a corpus-content fix.
4. (Lower priority, out of this audit's confirmed evidence) **Evaluate whether an evidence-law
   source (e.g. Qanun-e-Shahadat Order, 1984) should be added to the statute corpus** if the Q04-style
   "what evidence is required" framing is expected to recur across future queries. Evidence: confirmed
   absence of any evidence-law statute in the current 8-act corpus. This is a scope/coverage decision,
   not a bug fix.
5. **Re-run the same five pilot queries after any ranking change**, and expand beyond 5 queries,
   before drawing any conclusion about statute retrieval quality generally — per the evaluation
   pipeline's own stated purpose, these 5 queries are not a sufficient sample to judge the system.

---
*No source files, chunk files, embeddings, SQLite databases, Qdrant collections, or retrieval logic
were modified in the course of this audit. The only operations against Qdrant were two read-only
HTTP calls (`points/count`, `points/scroll`), both confirmed non-mutating.*
