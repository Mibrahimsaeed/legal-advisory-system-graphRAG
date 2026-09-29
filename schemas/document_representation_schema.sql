-- Corpus-neutral classification input: one row per document representation.
--
-- Replaces `document_signatures` as the corpus the classification path
-- (Stage 1.2 domain discovery: embed -> UMAP -> HDBSCAN -> label) reads
-- from. `document_signatures` (schemas/signature_schema.sql) is NOT
-- dropped -- it stays as the legacy PDF/book path's table and is still
-- readable via discovery.corpus_source='signatures'.
--
-- Deliberately keeps the same "lightweight, bounded" posture as the
-- signature table (see docs/data_retention_policy.md): no full document
-- text, only a title, a bounded heading list, and a char-capped body
-- preview. Full case text stays in the source case.html on disk; the
-- later RAG pipeline reads it from there, not from this table.
--
-- No FK to document_manifest: case folders are scanned directly by
-- orchestration/dags/case_ingest_flow.py, which does not go through the
-- Stage 0 claim/pull manifest (that path is PDF/file-extension shaped).

CREATE TABLE IF NOT EXISTS document_representations (
    doc_id          TEXT PRIMARY KEY,

    -- Provenance: enough to reopen the exact source. source_uri is the
    -- case folder, source_file the parsed case.html, source_relpath the
    -- folder path relative to caselaw.corpus_root (stable across
    -- remounts -- doc_id is its hash), content_hash a sha256 of the
    -- extracted text for change detection between scans.
    source_uri      TEXT NOT NULL,
    source_file     TEXT,
    source_relpath  TEXT,
    content_hash    TEXT,
    source_type     TEXT NOT NULL DEFAULT 'case_html',
    -- 'case_html' | (future source families)

    status          TEXT NOT NULL DEFAULT 'ok',
    -- 'ok' | 'failed'

    -- Classification input.
    title           TEXT,
    headings_json   TEXT NOT NULL DEFAULT '[]',
    body_preview    TEXT,
    char_count      INTEGER NOT NULL DEFAULT 0,
    -- NOTE: full extracted text is deliberately NOT stored (see
    -- docs/data_retention_policy.md). char_count is the length of the
    -- full text; body_preview is capped at caselaw.body_preview_char_limit.
    -- Re-derive full text from source_file when a later stage needs it.

    -- Case-law facts, normalized from metadata.json by
    -- src/extraction/case_metadata.py. Nullable by design: a case with
    -- no metadata.json is still a usable classification input.
    court           TEXT,
    decision_date   TEXT,   -- ISO-8601 YYYY-MM-DD when parseable, else NULL
    citation        TEXT,
    judges_json     TEXT NOT NULL DEFAULT '[]',
    case_number     TEXT,

    -- Verbatim metadata.json for the case, kept for provenance and for
    -- later phases (filtering, citation graph). Nothing in the
    -- classification path reads it.
    metadata_json   TEXT NOT NULL DEFAULT '{}',

    -- Cleaned case text produced by Phase 2 structural cleaning.
    -- NULL until then: Phase 1 only provides the column. See the
    -- "Case law" section of docs/data_retention_policy.md for the
    -- explicit amendment that permits durable cleaned text here, and why
    -- body_preview (bounded) remains the classification input.
    cleaned_text    TEXT,

    -- Legal metadata storage for later feature extraction. Phase 1
    -- provides the columns and their empty defaults only -- no statute or
    -- court extraction is implemented. SQLite has no JSONB; these are
    -- JSON documents held in TEXT, matching headings_json/metadata_json.
    statute_citations_json TEXT NOT NULL DEFAULT '[]',
    court_metadata_json    TEXT NOT NULL DEFAULT '{}',

    -- -- Current classification state -------------------------------
    -- Denormalized onto the document because a case must be able to
    -- exist in a 'pending' state *before* any classification runs, which
    -- document_classifications cannot express (its rows are written only
    -- after a verdict). That table remains the append-only per-run
    -- history; these columns are the current answer for this document.
    -- secondary_domain holds the single most relevant secondary domain;
    -- the full multi-domain list stays in
    -- document_classifications.secondary_domains_json.
    primary_domain      TEXT,
    secondary_domain    TEXT,
    domain_confidence   REAL,
    classification_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (classification_status IN (
            'pending',
            'auto_accepted',
            'needs_review',
            'dropped_procedural',
            'dropped_off_domain'
        )),
    -- Why a document was dropped, when classification_status is one of
    -- the dropped_* values: 'procedural', 'too_short', 'incomplete_scrape',
    -- 'cause_list', 'office_report', 'off_domain'. Deliberately NOT a
    -- CHECK constraint -- the drop policy is Phase 2+ work and its
    -- vocabulary is expected to grow.
    drop_reason         TEXT,

    -- Outcome. error is set only when status='failed' and is prefixed by
    -- an ERROR_* code from case_loader.py; warnings_json holds WARNING_*
    -- codes for non-fatal problems on an otherwise usable record.
    warnings_json   TEXT NOT NULL DEFAULT '[]',
    error           TEXT,

    batch_id        TEXT,

    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_representations_status
    ON document_representations (status);

CREATE INDEX IF NOT EXISTS idx_representations_batch
    ON document_representations (batch_id);

CREATE INDEX IF NOT EXISTS idx_representations_source_type
    ON document_representations (source_type);

CREATE INDEX IF NOT EXISTS idx_representations_court
    ON document_representations (court);

CREATE INDEX IF NOT EXISTS idx_representations_decision_date
    ON document_representations (decision_date);

CREATE INDEX IF NOT EXISTS idx_representations_content_hash
    ON document_representations (content_hash);

CREATE INDEX IF NOT EXISTS idx_representations_classification_status
    ON document_representations (classification_status);

CREATE INDEX IF NOT EXISTS idx_representations_primary_domain
    ON document_representations (primary_domain);
