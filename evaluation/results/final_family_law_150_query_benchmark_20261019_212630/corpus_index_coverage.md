# Corpus / Index Coverage Audit (read-only)

This section is separate from the relevance-based retrieval metrics in `metrics_summary.json`.
It verifies whether the indexed corpus represents the expected retrievable chunks, not whether
retrieval ranks them well.

## Method

All counts below were obtained read-only from two independent sources and cross-checked:

1. `var/rag/docstore.sqlite` (`chunks` table) — the full chunk manifest (28,784 rows total,
   including non-retrievable parents/footnotes), queried via `sqlite3`/`SELECT`, never written to.
2. Qdrant `pk_legal` collection (`GET /collections/pk_legal`, `POST /points/count`) — read-only
   HTTP calls only.

## Expected vs. indexed retrievable chunks

| Source type | Docstore `retrievable=1` rows | Qdrant indexed points (`source_type` filter) | Match |
|---|---:|---:|---|
| case | 25,891 | 25,891 | exact |
| constitution | 547 | 547 | exact |
| statute | 240 | 240 | exact |
| **Total** | **26,678** | **26,678** | exact |

Qdrant's total collection point count (`26,678`) equals the sum of the three filtered counts
exactly, confirming no stray/orphaned points outside the three known `source_type` values.

## Non-retrievable chunks (expected, not a gap)

| Source type | `retrievable=0` rows |
|---|---:|
| case | 2,038 |
| constitution | 47 |
| statute | 21 |

These are parent/footnote/counsel-line units intentionally excluded from the retrievable index
(per `src/rag_prep/embed_index.py`'s design: "Only `retrievable` chunks become points. Every
chunk ... goes to the SQLite docstore for parent/neighbour expansion"). Their presence in the
docstore but absence from Qdrant is correct, expected behaviour, not a coverage gap.

## Document-level counts

| Source type | Distinct `doc_id` (docstore) |
|---|---:|
| case | 2,084 |
| constitution | 1 |
| statute | 8 |

The 8 statute `doc_id`s were independently enumerated and confirmed to be exactly: Dissolution
of Muslim Marriages Act 1939, Family Courts Act 1964, Dowry and Bridal Gifts (Restriction) Act
1976, Guardians and Wards Act 1890, Child Marriage Restraint Act 1929, Muslim Family Laws
Ordinance 1961, Divorce Act 1869 (Christian marriages — present in the corpus but not targeted
by any of the 50 family-law statute queries, since it is a different personal-law regime), and
the West Pakistan Muslim Personal Law (Shariat) Application Act 1962.

## Sample-check: payload vs. docstore text integrity

A sample of chunks referenced in this benchmark's qrels (Constitution articles, statute
sections, case headnotes) were read directly from `docstore.sqlite`'s `canonical_text`/`json`
columns during query validation (see `queries_150.jsonl`'s `evidence[].snippet` fields, all of
which are verbatim excerpts of that column). No mismatch was found between the stored
`canonical_text` and the quoted statutory/constitutional/headnote text. This was a targeted
sample (every chunk cited as evidence for a `supported`/`partially_supported` query, ~195
distinct chunks), not an exhaustive scan of all 26,678 points — exhaustive text-integrity
diffing against original source PDFs/documents was out of scope and is recorded as a limitation.

## Missing / duplicate identifier checks

- `chunk_id` is the SQLite primary key (`PRIMARY KEY` constraint) — duplicates are structurally
  impossible in the docstore by construction; not separately re-verified here.
- Every `chunk_id` referenced in `qrels.jsonl` and in `retrieval_rankings.jsonl` was confirmed to
  resolve to a non-empty `canonical_text` row in `docstore.sqlite` (the `build_qrels.py` and
  `compute_metrics.py` scripts would have raised/produced empty snippets otherwise — none did).
- Point-ID to chunk-ID mapping: `src/rag_prep/embed_index.py`'s `point_id()` function derives
  Qdrant point IDs as `uuid5(chunk_id)` — a deterministic, collision-resistant mapping; this was
  not independently re-derived/re-verified bit-for-bit in this read-only audit (would require
  recomputing uuid5 for all 26,678 points), but the exact point-count match above is strong
  indirect evidence the mapping is intact.

## Dense/sparse vector availability

Qdrant's collection config (`GET /collections/pk_legal`) confirms both named vectors are
configured collection-wide: `dense` (size 1024, Cosine) and `bm25` (sparse, IDF modifier) — see
`audit_manifest.json`. Every retrieval call in this benchmark (dense-only, sparse-only, hybrid)
executed without error across all 148 eligible queries and all three corpora, which is itself
read-only evidence that both vector types are populated and queryable for points actually
returned (a point with a missing vector of the queried type would not appear in that type's
`query_points()` results at all). No point-by-point dense/sparse presence audit beyond this was
performed (would require iterating all 26,678 points' raw vectors — out of scope for a read-only
evaluation pass).

## Limitations disclosed

- Coverage was verified by count-matching and targeted sampling, not an exhaustive
  point-by-point reconciliation of all 26,678 Qdrant points against all 26,678 docstore rows.
- No regeneration, repair, or re-indexing was performed or considered, per the task's explicit
  read-only scope.
