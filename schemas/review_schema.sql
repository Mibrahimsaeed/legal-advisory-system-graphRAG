-- Phase 6: the human review ledger. One row per review event.
--
-- APPEND-ONLY. A re-review of the same document writes a NEW row; the
-- newest row per doc_id is the current human decision (see
-- src/classification/review_store.get_current_reviews()). Nothing in the
-- pipeline updates or deletes a review: a person's decision is evidence
-- about the corpus, and evidence is not edited in place.
--
-- This table is the authority on "was this document reviewed by a human?"
-- The document_representations row records the resulting STATE in Phase
-- 1's vocabulary ('auto_accepted' / 'needs_review' /
-- 'dropped_off_domain'); this records WHO decided it, WHEN, and what the
-- machine had said at the time -- which is also what Phase 7 needs to
-- measure the classifier against human labels.

CREATE TABLE IF NOT EXISTS document_review_decisions (
    review_id       TEXT PRIMARY KEY,        -- "<doc_id>:<created_at>"
    doc_id          TEXT NOT NULL,

    -- The human verdict.
    decision        TEXT NOT NULL,
    -- 'human_accepted'  -> the machine verdict is right
    -- 'human_corrected' -> wrong domain; decision_domain holds the right one
    -- 'human_rejected'  -> not in any target domain; withheld, NOT deleted
    -- 'human_uncertain' -> the reviewer could not decide; stays in the queue
    decision_domain TEXT,                    -- the domain the human settled on
    reviewer        TEXT NOT NULL,           -- who; required, never inferred
    notes           TEXT,                    -- why, in their words

    -- What the pipeline said when it was reviewed. Kept on the row rather
    -- than joined later, so a review stays interpretable even after the
    -- classifier is re-run with new weights.
    machine_domain      TEXT,
    machine_confidence  REAL,
    machine_status      TEXT,
    machine_band        TEXT,
    decision_run_id     TEXT,                -- the Phase 5 run reviewed
    signal_run_id       TEXT,                -- the Phase 3 evidence behind it

    -- How the document reached a reviewer: 'queue' (routed by policy),
    -- 'audit_sample' (Phase 7 spot-check of an auto-accepted document).
    source          TEXT NOT NULL DEFAULT 'queue',

    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_reviews_doc
    ON document_review_decisions (doc_id, created_at);

CREATE INDEX IF NOT EXISTS idx_reviews_decision
    ON document_review_decisions (decision);

CREATE INDEX IF NOT EXISTS idx_reviews_run
    ON document_review_decisions (decision_run_id);

CREATE INDEX IF NOT EXISTS idx_reviews_reviewer
    ON document_review_decisions (reviewer, created_at);
