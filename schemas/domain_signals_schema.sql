-- Phase 3 output: the EVIDENCE gathered about one document's domain.
--
-- This is not a classification. Phase 3 collects three independent
-- signals -- an unsupervised cluster, deterministic keyword matches, and
-- a broad LLM reading -- and stores them side by side. Phase 4 weighs
-- them and writes the actual verdict to document_classifications.
--
-- Why a separate table rather than more columns on
-- document_representations or a row in document_classifications:
--
--   * document_representations holds one current row per document; this
--     is per (run, document), because signals are regenerated whenever
--     the embedding model, keyword profiles or prompt change.
--   * document_classifications holds decisions. Putting evidence there
--     would make "what did we conclude" and "what did we observe"
--     indistinguishable, and Phase 4 needs to read the second to produce
--     the first.
--
-- APPEND-ONLY ACROSS RUNS, same contract as document_classifications:
-- re-running the same run_id refreshes that run's rows (which is what
-- makes a resumed batch idempotent), while a new run_id adds rows and
-- leaves earlier evidence intact.

CREATE TABLE IF NOT EXISTS document_domain_signals (
    signal_id           TEXT PRIMARY KEY,          -- "<run_id>:<doc_id>"
    run_id              TEXT NOT NULL,
    doc_id              TEXT NOT NULL,

    -- What was actually analysed. representation_hash covers exactly the
    -- text that was embedded and scanned, so an unchanged hash means the
    -- deterministic signals must be reproducible.
    representation_hash TEXT,
    text_source         TEXT,                      -- 'cleaned_text' | 'body_preview'
    char_count          INTEGER NOT NULL DEFAULT 0,
    word_count          INTEGER NOT NULL DEFAULT 0,

    -- Signal 1: unsupervised clustering. A cluster id is a discovery
    -- signal, NOT a domain -- nothing here maps cluster N to family_law.
    -- The full per-document mapping also lives in cluster_assignments.
    cluster_id          INTEGER,
    cluster_confidence  REAL,
    embedding_model     TEXT,

    -- Signal 2: deterministic keyword/concept profiles. The JSON holds
    -- per-domain scores and the matched terms behind them.
    keyword_signals_json TEXT NOT NULL DEFAULT '{}',
    keyword_top_domain  TEXT,
    keyword_margin      REAL,

    -- Signal 3: the LLM's broad reading. llm_status='failed' means no
    -- domain was recorded, never a guessed one.
    llm_domain          TEXT,
    llm_confidence      REAL,
    llm_reason          TEXT,
    llm_model           TEXT,
    llm_status          TEXT NOT NULL DEFAULT 'ok', -- 'ok' | 'failed' | 'skipped'
    llm_error           TEXT,

    -- Provenance.
    signal_version      TEXT NOT NULL,
    batch_id            TEXT,
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE (run_id, doc_id)
);

CREATE INDEX IF NOT EXISTS idx_domain_signals_run
    ON document_domain_signals (run_id);

CREATE INDEX IF NOT EXISTS idx_domain_signals_doc
    ON document_domain_signals (doc_id, created_at);

CREATE INDEX IF NOT EXISTS idx_domain_signals_cluster
    ON document_domain_signals (run_id, cluster_id);

CREATE INDEX IF NOT EXISTS idx_domain_signals_llm_domain
    ON document_domain_signals (run_id, llm_domain);

CREATE INDEX IF NOT EXISTS idx_domain_signals_keyword_domain
    ON document_domain_signals (run_id, keyword_top_domain);
