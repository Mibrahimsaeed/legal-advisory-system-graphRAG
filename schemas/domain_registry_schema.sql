-- Domain registry: Stage 1.2 (Domain Discovery) writes DRAFT rows here.
--
-- This is deliberately NOT the frozen domain registry described in
-- config/domains.yaml (id, label, centroid_ref) -- that file is the
-- source of truth for domains an operator has reviewed and accepted.
-- domain_candidates is scratch/staging: every run of
-- orchestration/dags/domain_discovery_flow.py inserts one row per
-- discovered cluster (top-N + the "other/uncertain" bucket), all with
-- status='draft'. Promoting a candidate into config/domains.yaml is a
-- manual, human-reviewed step -- no code path in this stage does it
-- automatically.

CREATE TABLE IF NOT EXISTS domain_candidates (
    candidate_id        TEXT PRIMARY KEY,
    run_id               TEXT NOT NULL,

    cluster_id           INTEGER,
    -- NULL/-1 cluster_id means this row is the "other / uncertain" bucket,
    -- not a real discovered cluster.
    is_other_bucket       INTEGER NOT NULL DEFAULT 0,

    name                 TEXT,
    description          TEXT,
    inclusion_criteria    TEXT,
    exclusion_criteria    TEXT,
    keywords_json         TEXT NOT NULL DEFAULT '[]',
    representative_doc_ids_json TEXT NOT NULL DEFAULT '[]',

    doc_count             INTEGER NOT NULL DEFAULT 0,
    sample_size           INTEGER NOT NULL DEFAULT 0,

    status                TEXT NOT NULL DEFAULT 'draft',
    -- 'draft' | 'accepted' | 'rejected' | 'merged' (only 'draft' is ever
    -- written by this stage; the rest are for a future human-review step)

    created_at            TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_domain_candidates_run
    ON domain_candidates (run_id);

CREATE INDEX IF NOT EXISTS idx_domain_candidates_status
    ON domain_candidates (status);