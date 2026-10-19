# Benchmark Coverage Report

## Constitution: 20 of 20 queries supported

All 20 task-specified queries (C01-C20) are supported by the indexed Constitution corpus, with exact query text preserved. **One notable corpus-validation finding:** C16 ("Under what constitutional circumstances can the Supreme Court exercise original jurisdiction over matters involving public importance and fundamental rights?") evokes **Article 184**, which direct text reading confirmed was **entirely omitted by the Constitution (Twenty-Seventh Amendment) Act, 2025, s.29** — a real, very recent legislative change reflected in this corpus (`var/rag/constitution_chunked/`). A targeted search found the substance of this jurisdiction now resides in **Article 175E(3)**, under a new Chapter 1A "THE FEDERAL CONSTITUTIONAL COURT" (created by the same amendment), whose text closely tracks the query's own wording. The query text was **not** altered; only the verified target was corrected from the no-longer-existent Article 184 to the current Article 175E.

Two queries (C05, C09) span two distinct articles each (assembly/association; property acquisition/deprivation) — both are recorded as valid grade-2 targets rather than arbitrarily picking one.

## Case law: 13 of 20 queries supported, 7 unsupported and reported (not fabricated)

**Decisive corpus-validation finding:** this case-law corpus's own payload metadata confirms **100% of 2084 documents are tagged `primary_domain: family_law`** — it is a curated family-law corpus, not a general law reporter. Exhaustive keyword search (exact-phrase and plausible-variant) across all 2084 documents found:

| Original query topic | Documents containing the key term(s) | Verdict |
|---|---:|---|
| Pre-arrest / anticipatory bail (CL01) | 0 | Unsupported |
| Post-arrest bail / further inquiry (CL02) | 0 | Unsupported |
| FIR registration delay (CL03) | 17 (none at headnote level on this doctrine) | Unsupported, not replaced |
| Circumstantial evidence / benefit of doubt (CL04) | 4 (none on-point) | Unsupported |
| Identification parade (CL05) | 0 | Unsupported, not replaced |
| Weapon recovery (CL06) | 0 | Unsupported, not replaced |
| Dying declaration (CL07) | 0 | Unsupported |
| Extra-judicial confession (CL08) | 0 | Unsupported |
| Material witness non-examination (CL09) | 0 | Unsupported, not replaced |
| Ocular vs. medical evidence (CL10) | 1-6 (none on-point) | Unsupported, not replaced |
| Condonation of delay (CL11) | real headnote hits found | **Supported as-is** |
| Specific performance (CL12) | 18 (none at clean headnote level) | Unsupported |
| Temporary injunction (CL13) | 3 (none on-point) | Unsupported |
| Inheritance / legal heir (CL14) | 68 / 98 | **Supported as-is** |
| Khula + dower (CL15) | core family-law topic | **Supported as-is** |
| Custody / welfare of minor (CL16) | core family-law topic | **Supported as-is** |
| Maintenance factors (CL17) | core family-law topic | **Supported as-is** |
| Alternate remedy / Art.199 (CL18) | 32 (clean hits found) | **Supported as-is** |
| Disciplinary proceedings (CL19) | 1 | Unsupported, not replaced |
| Landlord-tenant eviction (CL20) | 0 | Unsupported, not replaced |

**7 of the 14 unsupported queries (CL01, CL02, CL04, CL07, CL08, CL12, CL13) were replaced** with new, real-corpus-grounded family-law questions (verified against actual headnote text — nikahnama proof, restitution of conjugal rights, legitimacy presumption, talaq as an absolute right, iddat commencement, jactitation of marriage, polygamy restraints). Each replacement's original query, evidence, and reason are recorded in `queries_caselaw.json`.

**7 of the 14 unsupported queries (CL03, CL05, CL06, CL09, CL10, CL19, CL20) were left unreplaced and are reported as genuinely unsupported** — these are specifically criminal-trial-evidence doctrines (FIR, identification parade, weapon recovery, witness examination, ocular/medical contradiction) or general administrative/tenancy law (disciplinary proceedings, eviction), for which no confident, clean, on-point real-headnote match could be found after a dedicated search. **Per the task's own instruction, this count (13 of 20) is reported as the actual achieved coverage, not padded to 20 with weaker or redundant substitutes.**

## Combined totals actually evaluated

**33 scored queries: 20 Constitution + 13 case law.** All three retrieval methods (dense, sparse, hybrid) were run against all 33.
