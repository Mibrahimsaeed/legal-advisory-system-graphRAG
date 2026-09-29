-- The durable Phase 3 case signature: one row per document.
--
-- A "signature" here is the COMPACT CASE REPRESENTATION -- the reduced
-- title + headings + bounded body that Phase 3 reasons about
-- (src/classification/case_representation.py). Until now it existed only
-- in memory for the duration of a run, which meant two runs could not be
-- compared and nothing downstream could answer "what text was this
-- verdict actually derived from?". This table makes it an artifact.
--
-- WHY NOT `document_signatures`: that name is already taken by the legacy
-- PDF/book pipeline (schemas/signature_schema.sql), whose table has a
-- different shape, a FOREIGN KEY to document_manifest that case law never
-- populates, and overlapping column names (doc_id, signature_hash).
-- Because every schema file here uses CREATE TABLE IF NOT EXISTS, reusing
-- the name would make this DDL a silent no-op on any database that had
-- ever run `ingest`, and every INSERT below would then fail on a missing
-- column. A distinct name keeps the two eras of this project from
-- colliding in one table.
--
-- The full judgment is NOT duplicated here: it stays in
-- document_representations.cleaned_text. What this table holds is the
-- bounded derivative, which is a different (smaller) thing.

CREATE TABLE IF NOT EXISTS case_signatures (
    doc_id              TEXT PRIMARY KEY
                            REFERENCES document_representations(doc_id)
                            ON DELETE CASCADE,

    -- The signature itself: title + headings + bounded body, flattened
    -- exactly as CaseRepresentation.signal_text produces it. This is what
    -- keyword scanning consumes and what the embedding is pooled from.
    signature_text      TEXT NOT NULL,
    -- Kept alongside so the weighted embedding input (title x2.0,
    -- headings x1.5, body x1.0) can be reconstructed from this row alone
    -- -- a flattened string cannot be split back into its parts.
    title               TEXT,
    headings_json       TEXT NOT NULL DEFAULT '[]',

    signature_version   TEXT NOT NULL,
    -- sha256 of the analysed text: changes iff the signature changed.
    signature_hash      TEXT NOT NULL,
    -- document_representations.content_hash at build time. Together with
    -- signature_version this is the reuse key: same source text + same
    -- builder version means the stored signature is still correct and is
    -- reused instead of regenerated.
    source_content_hash TEXT,

    -- 'cleaned_text' (Phase 2 populated it) | 'body_preview' (fallback).
    text_source         TEXT NOT NULL DEFAULT 'cleaned_text',
    char_count          INTEGER NOT NULL DEFAULT 0,
    word_count          INTEGER NOT NULL DEFAULT 0,

    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_case_signatures_version
    ON case_signatures (signature_version);

CREATE INDEX IF NOT EXISTS idx_case_signatures_hash
    ON case_signatures (signature_hash);

CREATE INDEX IF NOT EXISTS idx_case_signatures_source_hash
    ON case_signatures (source_content_hash);
