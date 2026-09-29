-- Phase 7: the frozen corpus handed to GraphRAG.
--
-- A freeze is a SNAPSHOT, not a move: it records which doc_ids were
-- accepted at a moment in time, under which evaluation. The documents
-- themselves stay exactly where they are in document_representations.
-- Nothing is copied, nothing is deleted, and a later freeze does not
-- disturb an earlier one -- so a GraphRAG index built on freeze A stays
-- reproducible after freeze B exists.
--
-- Why the evaluation is recorded on the freeze row: "this corpus is
-- ready" is a claim, and a claim needs its evidence attached. The verdict,
-- the classification run and the validation-set size travel with the
-- snapshot so that months later it is still possible to ask what was
-- known when the corpus was frozen.

CREATE TABLE IF NOT EXISTS corpus_freezes (
    freeze_id           TEXT PRIMARY KEY,
    decision_run_id     TEXT NOT NULL,      -- the Phase 5 run frozen
    signal_run_id       TEXT,               -- the Phase 3 evidence behind it
    taxonomy_version    TEXT NOT NULL,

    -- The Phase 7 verdict that permitted this freeze.
    readiness_verdict   TEXT NOT NULL,
    readiness_json      TEXT NOT NULL DEFAULT '{}',
    validation_size     INTEGER NOT NULL DEFAULT 0,
    macro_f1            REAL,

    document_count      INTEGER NOT NULL DEFAULT 0,
    domain_counts_json  TEXT NOT NULL DEFAULT '{}',
    notes               TEXT,
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS corpus_freeze_members (
    freeze_id       TEXT NOT NULL,
    doc_id          TEXT NOT NULL,
    domain          TEXT NOT NULL,
    confidence      REAL,
    -- Whether a human confirmed this document, as opposed to the
    -- classifier accepting it unreviewed. GraphRAG may want to weight
    -- these differently, and it cannot if the distinction is lost here.
    human_reviewed  INTEGER NOT NULL DEFAULT 0,

    PRIMARY KEY (freeze_id, doc_id),
    FOREIGN KEY (freeze_id) REFERENCES corpus_freezes (freeze_id)
);

CREATE INDEX IF NOT EXISTS idx_freeze_members_domain
    ON corpus_freeze_members (freeze_id, domain);

CREATE INDEX IF NOT EXISTS idx_freeze_members_doc
    ON corpus_freeze_members (doc_id);
