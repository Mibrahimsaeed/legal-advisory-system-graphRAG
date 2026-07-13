 CREATE TABLE IF NOT EXISTS document_manifest (
    doc_id          TEXT PRIMARY KEY,
    source_uri      TEXT NOT NULL,
    checksum        TEXT NOT NULL,
    checksum_algo   TEXT NOT NULL DEFAULT 'sha256',
    byte_size       INTEGER,

    ingest_status   TEXT NOT NULL DEFAULT 'pending',

    batch_id        TEXT,
    attempts        INTEGER NOT NULL DEFAULT 0,
    last_error      TEXT,

    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_manifest_status_created
    ON document_manifest (ingest_status, created_at);

CREATE INDEX IF NOT EXISTS idx_manifest_batch
    ON document_manifest (batch_id);

CREATE TABLE IF NOT EXISTS batch_runs (
    batch_id        TEXT PRIMARY KEY,
    requested_size  INTEGER NOT NULL,
    actual_size     INTEGER,
    status          TEXT NOT NULL DEFAULT 'open',
    started_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at     TIMESTAMP,
    notes           TEXT
);