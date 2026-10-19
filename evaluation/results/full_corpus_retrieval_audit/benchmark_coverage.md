# Benchmark Coverage Report

**28 queries total**, built from one frozen benchmark plus two new additions:

| Corpus | Queries | Source | Query types | Judgment completeness |
|---|---:|---|---|---|
| Statutes | 16 (Q01-Q16) | `evaluation/queries_statute_benchmark16.json` — **frozen, reused unmodified**, checksum-verified | 8 natural-language + 8 direct lookup, across all 8 indexed statutes | Verified by direct source reading; includes 6 hard negatives (SRI Phase 1) |
| Constitution | 6 (C1-C6) | `evaluation/results/phase2a_constitution_audit.json` — **reused verbatim**, promoted from smoke test to scored benchmark with new qrels | 2 direct lookup + 4 natural-language | **Representative, not exhaustive** — 1 primary chunk per query; Articles 175A and 199 have additional clauses not covered |
| Case law | 6 (CL1-CL6) | New, built in this audit (`queries_caselaw6.json`) | 6 natural-language | **Representative, not exhaustive** — 1 headnote chunk per query, verified by direct text reading; no hard negatives added for case law in this pass |

**Per-statute distribution (Q01-Q16):** 2 queries per statute × 8 statutes — fully stratified, no statute dominates.

**Case-law query topics:** khula entitlement standard, meaning of "maintenance," dowry-article evidentiary rules, custody jurisdiction (HC vs. Guardian Judge), khula+dower interaction, factum-of-hatred standard — all drawn from real, verified case headnotes, deliberately overlapping with statute benchmark topics (khula, maintenance, dowry, custody) to support cross-corpus comparison per task item 6.

**Constitution query topics:** unchanged from the existing smoke test (specific article lookup, a fundamental right, legislative process, judicial appointments, federal/provincial powers, a specific clause/proviso).

## Judgment completeness caveats (explicit, not hidden)

- **Case law and Constitution qrels are representative, not exhaustive.** Each has exactly one verified grade-2 chunk per query; several target articles/cases plausibly have additional genuinely-relevant passages not included (e.g. Article 175A's other clauses; the companion judgment-body paragraph for CL1's case). This was discovered during Phase 4 (see `retrieval_failure_taxonomy.csv`) and is reported as a benchmark limitation, not retroactively patched into the qrels (to avoid fitting qrels to retrieval results).
- **Statute qrels remain the most complete** of the three (SRI Phase 1's targeted completeness sweep plus hard negatives).
- No stratified sampling across all 2088 case-law source documents or all 328 constitutional articles was attempted — 6 queries each is a deliberately small, well-judged subset, not a claim of corpus-wide representativeness. Per the task's own instruction, a smaller well-judged benchmark was preferred over a larger unreliable one.

## What this benchmark does NOT cover

- Case-law queries requiring cross-referencing a specific statutory provision that a case interprets (task item 6) were approximated by topic overlap (e.g. CL5 covers the same khula/dower relationship as statute Q01), not by a query explicitly demanding both a case AND its statutory authority in one combined relevance judgment.
- Queries deliberately designed to have low lexical overlap with the target (task item 8) were not separately constructed — the existing SRI Phase 1 "mensa et toro" and "aggregate value" terminology-gap findings already demonstrate this phenomenon for statutes; it was not independently re-tested for case law or Constitution in this pass.
