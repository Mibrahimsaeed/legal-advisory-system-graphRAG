# Final Recommendation

## What is demonstrably working

- Corpus/index integrity: all retrievable-chunk counts reconcile exactly (chunked files = docstore = `index_meta.json` = live Qdrant) across all three corpora. Vector configuration (dense 1024/Cosine, sparse BM25/IDF) matches the implementation exactly.
- Statute retrieval is strong: macro nDCG@10 0.896, Recall@10 0.969 on a 16-query, 8-statute, well-judged benchmark.
- Two of three diagnosed statute ranking weaknesses (Q06, Q13) were **fixed** with a verified, zero-regression terminology addition — a cheap, low-risk lever that works.
- Per-source filtering is confirmed load-bearing and already correctly implemented — an isolated check showed cross-corpus competition would hurt statute retrieval if it existed, and it does not exist in production.

## What is demonstrably failing

- **4 case-law documents are completely absent from the index** (0.19% of 2088, but 100% loss for those 4) — a confirmed, pre-existing integrity defect, unrelated to this benchmark.
- **3 full-case duplicate pairs** are indexed as separate points in case law — confirmed, re-verified live this pass.
- Statute Q16's title/preamble lexical-competition issue is only **partially** fixed (nDCG@10 0.431→0.500) — the Application Act's preamble/§1 chunks still partially win.

## What is merely suspected (not confirmed)

- That Constitution query C4 and case-law query CL1's "0.0" scores reflect genuine retrieval failures. **Phase 4 found both are more consistent with benchmark qrels-granularity limitations** than confirmed defects: C4's target article has 5 other clauses ranking in the real top-10; CL1's target case has a directly on-point judgment-body paragraph ranking 8th. Neither qrels set was retroactively expanded to confirm this numerically, specifically to avoid fitting qrels to results — so this remains a well-evidenced hypothesis, not a demonstrated fact.
- That the embedding model is a bottleneck for case law. Only one query (CL1, dense rank 208) shows this pattern; five other case-law/Constitution queries show dense ranks of 1-3. Insufficient evidence per the task's own Phase 4G standard.
- That a case-law-specific chunking convention issue (headnote vs. judgment-body restating the same point as separate chunks) affects retrieval broadly — one instance observed (CL1), not systematically swept.

## Which issues affect one corpus vs. the whole system

| Issue | Scope |
|---|---|
| 4 missing case docs, 3 duplicate case pairs | Case law only |
| Title/preamble lexical competition (Q06, Q13, Q16) | Statutes (confirmed); not yet tested in Constitution/case law, though the same structural pattern (preamble/short-title chunks) exists in all three |
| No production code path for cross-corpus ranking | Architectural, affects nothing currently (no corpus suffers from it, since it's never invoked) but also means it can't be leveraged either |
| Possible qrels-granularity issue (C4, CL1) | A benchmark-construction limitation, not a corpus or retrieval-pipeline issue |

## Are relevant passages missing from candidates, or merely ranked poorly?

**Mostly ranked poorly, not missing** — for every diagnosed weak query except the 4 missing case documents, the relevant content is confirmed present in at least one branch's candidate pool (often both). The one partial exception is CL1's dense branch specifically (rank 208, effectively absent from any reasonable dense-only candidate depth), compensated only partially by BM25 (rank 12).

## Does current evidence support a reranker?

**Not yet, but it is the most plausible next experiment for case law specifically** — CL1's target does reach the fused candidate list (rank 25) via BM25, which is exactly the precondition Phase 4A sets for a reranker to be useful (a reranker cannot recover a candidate that never arrives, but CL1's does arrive, just ranked low). This was not tested in this pass. For statutes, a reranker is lower priority since 2 of 3 diagnosed weaknesses already have a cheaper, verified fix (terminology expansion).

## Does chunking or embeddings need further investigation?

**Chunking:** no confirmed chunk-content defect was found driving any ranking weakness in this pass (the earlier case-law integrity defects — missing docs, duplicates — are a separate, already-documented issue, not newly implicated by this benchmark). The CL1 headnote/judgment-body granularity question is a plausible hypothesis worth a dedicated, narrow follow-up, not yet confirmed.

**Embeddings:** not implicated by current evidence beyond one instance (CL1). Do not replace or re-evaluate the embedding model based on this audit alone.

## Is retrieval quality sufficient to proceed to GraphRAG?

**Yes, with the explicit caveat that GraphRAG construction and validation have different requirements than graph-enhanced retrieval quality.** Per the task's own framing: building the knowledge graph (entity/relation extraction, citation linking) does not require retrieval to be perfect first — it can proceed on the existing, demonstrably-reconciled, integrity-checked corpus. What retrieval quality *does* need to be adequate for is validating that graph-enhanced retrieval actually improves over the current hybrid baseline later — and this audit's statute benchmark (and, with the stated caveats, Constitution/case-law benchmarks) now exist as that comparison point for when graph-enhanced retrieval is ready to be measured against it.

## Single next action, and why

**Fix the 4 missing case-law documents and the 3 duplicate case pairs before anything else.** This is the only finding in the entire audit that is (a) a confirmed defect, (b) unambiguous (no qrels-interpretation question), (c) affects the corpus itself rather than a ranking parameter, and (d) is a precondition for GraphRAG to build correctly over case law — a knowledge graph built from a corpus with 4 silently-missing documents and 3 duplicated cases will propagate both problems into the graph (missing nodes; duplicate/redundant nodes for the same case). Fixing this is independent of every other open question in this audit (reranker, terminology expansion, embedding model) and should happen regardless of which of those is pursued next.

**Second priority, in parallel:** extend the verified terminology-expansion method (Experiment B) to Q16's remaining gap and, separately, investigate a reranker for CL1 specifically — both are cheap, targeted, low-risk experiments with a clear hypothesis already established by this audit's own evidence.
