-- Stage 1 (Feature Extraction) output: one row per document signature.
-- Deliberately does NOT store full document text -- title, TOC/headings,
-- and a char-capped body preview only (see config/pipeline.yaml ->
-- extraction.body_preview_char_limit). Raw PDFs never leave ephemeral
-- scratch (see docs/data_retention_policy.md).

CREATE TABLE IF NOT EXISTS document_signatures (
    doc_id              TEXT PRIMARY KEY
                            REFERENCES document_manifest(doc_id)
                            ON DELETE CASCADE,
    source_uri          TEXT NOT NULL,
    signature_hash      TEXT NOT NULL,
    is_scanned          INTEGER NOT NULL DEFAULT 0,

    extraction_status   TEXT NOT NULL DEFAULT 'pending',
    -- 'pending' | 'ok' | 'ok_ocr' | 'ok_partial_ocr' | 'failed'
    extractor_used      TEXT,
    -- 'fitz' | 'pdfplumber' | 'pypdf' | 'ocr' | 'mixed' | 'none'

    title               TEXT,
    toc_json            TEXT NOT NULL DEFAULT '[]',
    body_preview        TEXT,
    pages_used          INTEGER NOT NULL DEFAULT 0,
    char_count          INTEGER NOT NULL DEFAULT 0,
    quality_score       REAL NOT NULL DEFAULT 0,

    batch_id            TEXT,
    error               TEXT,

    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_signatures_status
    ON document_signatures (extraction_status);

CREATE INDEX IF NOT EXISTS idx_signatures_batch
    ON document_signatures (batch_id);

CREATE INDEX IF NOT EXISTS idx_signatures_hash
    ON document_signatures (signature_hash);