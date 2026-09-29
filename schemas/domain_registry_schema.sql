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

    -- Stable slug for the domain ("criminal_appeals"). cluster_id is NOT
    -- an identity: HDBSCAN renumbers clusters on every run, so anything
    -- downstream must key on domain_id.
    domain_id             TEXT,

    -- Review state written by src/clustering/taxonomy_draft.py.
    -- confidence='low' + review_required=1 marks a domain whose cluster
    -- was flagged as mixed/weak in the Phase 4 review: it is drafted so
    -- the real domain isn't lost, but must not be accepted unexamined.
    confidence            TEXT NOT NULL DEFAULT 'high',   -- 'high' | 'low'
    review_required       INTEGER NOT NULL DEFAULT 0,
    flags_json            TEXT NOT NULL DEFAULT '[]',
    notes_json            TEXT NOT NULL DEFAULT '[]',

    created_at            TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- NOTE: `CREATE TABLE IF NOT EXISTS` cannot add columns to a table that
-- already exists, so the four review columns above are also applied via
-- ALTER TABLE by taxonomy_card.ensure_domain_candidate_columns().


CREATE INDEX IF NOT EXISTS idx_domain_candidates_run
    ON domain_candidates (run_id);

CREATE INDEX IF NOT EXISTS idx_domain_candidates_status
    ON domain_candidates (status);

-- Raw per-document HDBSCAN cluster assignments for a Stage 1.2 run --
-- every sampled doc_id gets a row here (including noise, label -1), not
-- just the top-N clusters that get promoted to domain_candidates. This is
-- what lets a cluster's full membership be inspected later without
-- re-running discovery.
CREATE TABLE IF NOT EXISTS cluster_assignments (
    run_id      TEXT NOT NULL,
    doc_id      TEXT NOT NULL,
    cluster_id  INTEGER NOT NULL,
    confidence  REAL,
    -- HDBSCAN's probabilities_ (per-point membership strength); NULL when
    -- the clusterer backend didn't provide one.
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (run_id, doc_id)
);

CREATE INDEX IF NOT EXISTS idx_cluster_assignments_run
    ON cluster_assignments (run_id);

CREATE INDEX IF NOT EXISTS idx_cluster_assignments_run_cluster
    ON cluster_assignments (run_id, cluster_id);