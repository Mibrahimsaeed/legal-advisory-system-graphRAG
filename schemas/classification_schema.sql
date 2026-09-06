-- Phase 6 output: one row per (classification run, document).
--
-- APPEND-ONLY ACROSS RUNS. A row is keyed by (run_id, doc_id): re-running
-- the same run_id re-writes that run's row (which is what makes a resumed
-- or retried batch idempotent), but a new run_id -- a new taxonomy
-- version, a new classifier version, a re-classification months later --
-- writes NEW rows and leaves every earlier verdict intact. Nothing in the
-- pipeline deletes or updates another run's rows.
--
-- "What is this document's domain today?" is therefore a query over the
-- latest run, not a mutable column: see
-- src/classification/classification_store.get_current_classifications().

CREATE TABLE IF NOT EXISTS document_classifications (
    classification_id   TEXT PRIMARY KEY,          -- "<run_id>:<doc_id>"
    run_id              TEXT NOT NULL,
    doc_id              TEXT NOT NULL,

    -- Which cluster this document sat in (from cluster_assignments of the
    -- discovery run the taxonomy came from). NULL when the document was
    -- never clustered -- classification does not require it.
    cluster_id          INTEGER,

    primary_domain      TEXT,                       -- frozen domain id, or 'other_uncertain'
    secondary_domains_json TEXT NOT NULL DEFAULT '[]',
    confidence          REAL,                       -- 0.0 - 1.0, as reported by the classifier
    justification       TEXT,

    status              TEXT NOT NULL DEFAULT 'classified',
    -- 'classified'   -> accepted automatically
    -- 'needs_review' -> low confidence / ambiguous / mixed: a label is
    --                   proposed but must not be treated as final
    -- 'failed'       -> classifier or validation error; NO label is implied
    review_reason       TEXT,
    error               TEXT,

    -- Provenance: every row says exactly what produced it.
    taxonomy_version    TEXT NOT NULL,
    classifier_version  TEXT NOT NULL,
    model_name          TEXT,
    batch_id            TEXT,

    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (run_id, doc_id)
);

CREATE INDEX IF NOT EXISTS idx_classifications_run
    ON document_classifications (run_id);

CREATE INDEX IF NOT EXISTS idx_classifications_doc
    ON document_classifications (doc_id, created_at);

CREATE INDEX IF NOT EXISTS idx_classifications_domain
    ON document_classifications (run_id, primary_domain);

CREATE INDEX IF NOT EXISTS idx_classifications_status
    ON document_classifications (run_id, status);

CREATE INDEX IF NOT EXISTS idx_classifications_taxonomy
    ON document_classifications (taxonomy_version);
