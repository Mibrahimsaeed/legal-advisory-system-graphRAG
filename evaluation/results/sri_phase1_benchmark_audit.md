# SRI Phase 1 — Benchmark & Qrels Audit

**Files:** `evaluation/queries_statute_benchmark16.json` (16 queries), `evaluation/results/sri_phase1_qrels_rich.json` (graded, rationale-backed judgments), `evaluation/results/sri_phase1_qrels.json` (derived simple qrels for `calculate_metrics.py --qrels`).

## Per-statute coverage

All 8 indexed statutes represented, 2 queries each (1 natural-language, 1 direct lookup):

| Statute | Queries |
|---|---|
| Dissolution of Muslim Marriages Act, 1939 | Q01, Q02 |
| Family Courts Act, 1964 | Q03, Q04 |
| Dowry and Bridal Gifts (Restriction) Act, 1976 | Q05, Q06 |
| Guardians and Wards Act, 1890 | Q07, Q08 |
| Child Marriage Restraint Act, 1929 | Q09, Q10 |
| Muslim Family Laws Ordinance, 1961 | Q11, Q12 |
| Divorce Act, 1869 | Q13, Q14 |
| West Pakistan Muslim Personal Law (Shariat) Application Act, 1962 | Q15, Q16 |

## Grounding corrections documented (none silently rewritten)

- **Q03/Q04:** Family Courts Act §10 covers pre-trial/conciliation and the dissolution-decree proviso, but its text never uses the word "khula" — it says "a suit for dissolution of marriage" generally. Noted as an interpretive label, not a factual error; original wording retained.
- **Q13/Q14:** §22's title ("Bar to decree for divorce a mensa et toro...") is archaic/indirect and doesn't itself signal "grounds for judicial separation," but its operative text states the three grounds (adultery, cruelty, 2-year desertion) explicitly. §23 is a necessary procedural companion. No misrepresentation found; no correction needed.

## Qrels construction

Every judgment was made by directly reading the current `var/rag/statutes_chunked/*.json` source text — not inferred from the query or filename, and not derived from any retriever's top-k output. A targeted keyword sweep across the full corpus (240 retrievable statute chunks) was run per query topic as a completeness check (not an exhaustive manual re-read of every chunk) and surfaced one genuine addition: `0425160ffedaefd8c79b3241:s2_ssb-c` (a proviso qualifying DMMA §2's grounds (i) and (v)), now included as grade-1 context for Q01/Q02.

**Hard negatives included**, each verified by reading its actual text and confirming it is a different legal concept, not merely a different chunk:

| Query | Hard negative | Why it's a plausible distractor |
|---|---|---|
| Q01/Q02 | Divorce Act §19 "Grounds of decrees" | Same "grounds for X decree" structure, but governs nullity of a Christian marriage, not dissolution under Muslim law |
| Q03/Q04 | Divorce Act §22 | Same marital-relief domain, wrong Act and wrong relief (judicial separation vs. khula/dissolution) |
| Q07/Q08 | Family Courts Act §5 | Matches on "jurisdiction" generically; governs a different court's subject-matter scope |
| Q09/Q10 | CMRA §§4, 5 | Same penalty scheme, but punish the adult male party / solemniser, not the parent/guardian |
| Q11/Q12 | MFLO §9 | Same institution (Arbitration Council), unrelated purpose (maintenance, not polygamy permission) |
| Q13/Q14 | Divorce Act §19 | Same as Q01/Q02's negative — nullity grounds, not separation grounds |

## Validation gate checklist

- [x] All 8 indexed statutes represented.
- [x] 16 queries exist; 2 corrections documented (Q03/Q04, Q13/Q14), neither required a wording change.
- [x] No accidental duplicate query texts (16/16 unique, programmatically verified).
- [x] Every qrels chunk ID resolves to a current corpus chunk (0 of 20 judged chunk IDs failed; programmatically verified).
- [x] Every query has at least one independently-verified grade-2 relevant chunk.
- [x] All relevance labels marked `verified` (none left `pending_manual_review`).
- [x] No unexplained conflicts.
- [~] **Qrels denominator completeness for Recall@5/Recall@10:** reasonably established via a targeted per-topic keyword sweep across the full corpus plus direct source reading, **not** an exhaustive manual line-by-line review of all 240 statute chunks against all 16 queries. This is the one caveat carried forward into Phase 2 — flagged explicitly, not hidden.

**Gate result: PASS, with the completeness caveat documented.** Proceeding to Phase 2.
