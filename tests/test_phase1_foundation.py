"""Phase 1: the case-law schema and configuration foundation.

Phase 1 adds *storage and configuration only* -- no extraction, no
scoring, no classifier. These tests therefore assert the contract a later
phase will build on: that a case can exist in the database from the
moment it is ingested, in a ``pending`` state with no domain assigned;
that the status vocabulary is enforced; that the legal-metadata columns
exist with empty defaults; and that the configuration actually resolves
to the intended values.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import yaml

from src.classification.taxonomy_registry import (
    OTHER_DOMAIN_ID,
    load_frozen_taxonomy,
)
from src.common.config import (
    ClassificationSignalWeights,
    DocumentSettings,
    get_settings,
    load_settings,
)
from src.common.db import connection_scope, init_schema
from src.extraction.doc_representation import (
    CLASSIFICATION_STATUS_AUTO_ACCEPTED,
    CLASSIFICATION_STATUS_DROPPED_OFF_DOMAIN,
    CLASSIFICATION_STATUS_DROPPED_PROCEDURAL,
    CLASSIFICATION_STATUS_NEEDS_REVIEW,
    CLASSIFICATION_STATUS_PENDING,
    CLASSIFICATION_STATUSES,
    DocumentRepresentation,
)
from src.extraction.representation_store import (
    DEFAULT_REPRESENTATION_SCHEMA_FILE,
    ensure_representation_columns,
    get_representation,
    upsert_representations,
)

REPRESENTATION_SCHEMA_FILE = str(DEFAULT_REPRESENTATION_SCHEMA_FILE)


@pytest.fixture()
def db(tmp_path) -> Path:
    """A fresh database with only the case-law schema applied."""

    db_path = tmp_path / "phase1.db"
    init_schema(db_path=db_path, schema_file=REPRESENTATION_SCHEMA_FILE)
    return db_path


def _case(doc_id: str = "case_1", **overrides) -> DocumentRepresentation:
    base = dict(
        doc_id=doc_id,
        source_uri=f"/corpus/{doc_id}",
        source_file=f"/corpus/{doc_id}/case.html",
        source_relpath=f"lahore_high_court/{doc_id}",
        source_type="case_html",
        title="Zainab Bibi v. The State",
        citation="2021 PLJ 88",
        court="Lahore High Court",
        decision_date="2021-04-11",
        judges=["Mr. Justice Shahid Karim"],
        case_number="Criminal Appeal No. 1 of 2020",
        body_preview="The appellant was convicted under section 302 PPC.",
        char_count=52,
    )
    base.update(overrides)
    return DocumentRepresentation(**base)


# ---------------------------------------------------------------------------
# Case-law document representation
# ---------------------------------------------------------------------------


def test_case_law_document_can_be_represented_and_stored(db):
    """Every field the case-law model must carry, round-tripped."""

    upsert_representations([_case()], db_path=db)
    stored = get_representation("case_1", db_path=db)

    assert stored.doc_id == "case_1"                      # case_id
    assert stored.title == "Zainab Bibi v. The State"     # title
    assert stored.citation == "2021 PLJ 88"               # case_citation
    assert stored.court == "Lahore High Court"            # court
    assert stored.judges == ["Mr. Justice Shahid Karim"]  # bench
    assert stored.decision_date == "2021-04-11"           # judgment_date
    assert stored.source_type == "case_html"              # source
    assert stored.source_relpath == "lahore_high_court/case_1"  # source_folder
    assert stored.source_file.endswith("case.html")       # provenance
    assert stored.case_number == "Criminal Appeal No. 1 of 2020"


def test_cleaned_text_column_exists_and_defaults_to_null(db):
    """Phase 1 provides the column; Phase 2 structural cleaning fills it."""

    upsert_representations([_case()], db_path=db)
    assert get_representation("case_1", db_path=db).cleaned_text is None

    upsert_representations(
        [_case(cleaned_text="IN THE LAHORE HIGH COURT ...")], db_path=db
    )
    assert get_representation("case_1", db_path=db).cleaned_text.startswith("IN THE")


def test_body_preview_remains_the_bounded_classification_input(db):
    """cleaned_text must not displace the bounded preview the embedder reads."""

    upsert_representations(
        [_case(body_preview="short preview", cleaned_text="x" * 50_000)], db_path=db
    )
    stored = get_representation("case_1", db_path=db)

    assert stored.body_preview == "short preview"
    assert len(stored.cleaned_text) == 50_000


# ---------------------------------------------------------------------------
# Classification state
# ---------------------------------------------------------------------------


def test_new_case_is_pending_with_no_domain_assigned(db):
    """The required initial state, straight out of the database."""

    upsert_representations([_case()], db_path=db)

    with connection_scope(db) as conn:
        row = conn.execute(
            "SELECT primary_domain, secondary_domain, domain_confidence, "
            "classification_status FROM document_representations WHERE doc_id = ?",
            ("case_1",),
        ).fetchone()

    assert row["primary_domain"] is None
    assert row["secondary_domain"] is None
    assert row["domain_confidence"] is None
    assert row["classification_status"] == "pending"


def test_pending_is_the_column_default_even_for_a_bare_insert(db):
    """A row written by any path -- not just the store -- starts pending."""

    with connection_scope(db) as conn:
        conn.execute(
            "INSERT INTO document_representations (doc_id, source_uri) VALUES (?, ?)",
            ("bare", "/corpus/bare"),
        )
        status = conn.execute(
            "SELECT classification_status FROM document_representations WHERE doc_id='bare'"
        ).fetchone()["classification_status"]

    assert status == CLASSIFICATION_STATUS_PENDING


@pytest.mark.parametrize("status", CLASSIFICATION_STATUSES)
def test_every_status_in_the_vocabulary_is_accepted(db, status):
    upsert_representations([_case(classification_status=status)], db_path=db)

    assert get_representation("case_1", db_path=db).classification_status == status


def test_the_vocabulary_is_exactly_the_five_required_states():
    assert set(CLASSIFICATION_STATUSES) == {
        "pending",
        "auto_accepted",
        "needs_review",
        "dropped_procedural",
        "dropped_off_domain",
    }


@pytest.mark.parametrize(
    "bad_status", ["classified", "accepted", "PENDING", "dropped", "", "reviewed"]
)
def test_invalid_classification_status_is_rejected_by_the_database(db, bad_status):
    """The CHECK constraint, not application code, is the guarantee."""

    with pytest.raises(sqlite3.IntegrityError):
        with connection_scope(db) as conn:
            conn.execute(
                "INSERT INTO document_representations "
                "(doc_id, source_uri, classification_status) VALUES (?, ?, ?)",
                ("bad", "/corpus/bad", bad_status),
            )


def test_domain_values_are_not_constrained_by_the_database(db):
    """Domains come from the taxonomy registry, so SQL must not fight it.

    A new domain must be addable by editing config/domains.yaml, without a
    schema migration.
    """

    upsert_representations(
        [_case(primary_domain="a_future_domain", secondary_domain="another")],
        db_path=db,
    )

    stored = get_representation("case_1", db_path=db)
    assert stored.primary_domain == "a_future_domain"


def test_classified_state_round_trips(db):
    upsert_representations(
        [
            _case(
                primary_domain="criminal_law",
                secondary_domain="family_law",
                domain_confidence=0.87,
                classification_status=CLASSIFICATION_STATUS_AUTO_ACCEPTED,
            )
        ],
        db_path=db,
    )

    stored = get_representation("case_1", db_path=db)
    assert stored.primary_domain == "criminal_law"
    assert stored.secondary_domain == "family_law"
    assert stored.domain_confidence == pytest.approx(0.87)
    assert stored.classification_status == "auto_accepted"


# ---------------------------------------------------------------------------
# Legal metadata storage + auditability
# ---------------------------------------------------------------------------


def test_statute_citations_default_to_an_empty_list_and_round_trip(db):
    upsert_representations([_case()], db_path=db)
    assert get_representation("case_1", db_path=db).statute_citations == []

    citations = [
        {"statute": "Pakistan Penal Code, 1860", "section": "302"},
        {"statute": "Criminal Procedure Code, 1898", "section": "497"},
    ]
    upsert_representations([_case(statute_citations=citations)], db_path=db)

    assert get_representation("case_1", db_path=db).statute_citations == citations


def test_court_metadata_defaults_to_an_empty_object_and_round_trips(db):
    upsert_representations([_case()], db_path=db)
    assert get_representation("case_1", db_path=db).court_metadata == {}

    metadata = {"bench_size": 2, "seat": "Lahore", "jurisdiction": "appellate"}
    upsert_representations([_case(court_metadata=metadata)], db_path=db)

    assert get_representation("case_1", db_path=db).court_metadata == metadata


def test_json_columns_are_stored_as_text_not_jsonb(db):
    """SQLite has no JSONB; these must be plain TEXT holding JSON."""

    upsert_representations([_case()], db_path=db)

    with connection_scope(db) as conn:
        types = {
            r["name"]: r["type"]
            for r in conn.execute("PRAGMA table_info(document_representations)")
        }

    assert types["statute_citations_json"] == "TEXT"
    assert types["court_metadata_json"] == "TEXT"


@pytest.mark.parametrize(
    "reason",
    ["procedural", "too_short", "incomplete_scrape", "cause_list",
     "office_report", "off_domain"],
)
def test_drop_reason_can_be_stored_for_each_anticipated_reason(db, reason):
    status = (
        CLASSIFICATION_STATUS_DROPPED_OFF_DOMAIN
        if reason == "off_domain"
        else CLASSIFICATION_STATUS_DROPPED_PROCEDURAL
    )
    upsert_representations(
        [_case(classification_status=status, drop_reason=reason)], db_path=db
    )

    stored = get_representation("case_1", db_path=db)
    assert stored.drop_reason == reason
    assert stored.classification_status.startswith("dropped_")


def test_a_case_remains_traceable_to_its_source_after_being_dropped(db):
    upsert_representations(
        [
            _case(
                classification_status=CLASSIFICATION_STATUS_DROPPED_PROCEDURAL,
                drop_reason="cause_list",
            )
        ],
        db_path=db,
    )

    stored = get_representation("case_1", db_path=db)
    assert stored.source_file and stored.source_relpath  # provenance kept
    with connection_scope(db) as conn:
        row = conn.execute(
            "SELECT created_at, updated_at FROM document_representations WHERE doc_id='case_1'"
        ).fetchone()
    assert row["created_at"] and row["updated_at"]  # timestamps kept


def test_review_queue_is_queryable_from_status(db):
    """The state model has to answer 'what needs a human?' in SQL."""

    upsert_representations(
        [
            _case("pending_1"),
            _case("review_1", classification_status=CLASSIFICATION_STATUS_NEEDS_REVIEW),
            _case("accepted_1", classification_status=CLASSIFICATION_STATUS_AUTO_ACCEPTED),
            _case("dropped_1", classification_status=CLASSIFICATION_STATUS_DROPPED_OFF_DOMAIN),
        ],
        db_path=db,
    )

    with connection_scope(db) as conn:
        counts = {
            r["classification_status"]: r["n"]
            for r in conn.execute(
                "SELECT classification_status, COUNT(*) AS n "
                "FROM document_representations GROUP BY classification_status"
            )
        }

    assert counts == {
        "pending": 1, "needs_review": 1, "auto_accepted": 1, "dropped_off_domain": 1
    }


# ---------------------------------------------------------------------------
# Migration of a pre-Phase-1 database
# ---------------------------------------------------------------------------


def test_existing_database_gains_the_new_columns(tmp_path):
    """A database created before Phase 1 must not break on upsert."""

    db_path = tmp_path / "legacy.db"
    with connection_scope(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE document_representations (
                doc_id TEXT PRIMARY KEY, source_uri TEXT NOT NULL,
                source_file TEXT, source_relpath TEXT, content_hash TEXT,
                source_type TEXT NOT NULL DEFAULT 'case_html',
                status TEXT NOT NULL DEFAULT 'ok', title TEXT,
                headings_json TEXT NOT NULL DEFAULT '[]', body_preview TEXT,
                char_count INTEGER NOT NULL DEFAULT 0, court TEXT,
                decision_date TEXT, citation TEXT,
                judges_json TEXT NOT NULL DEFAULT '[]', case_number TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                warnings_json TEXT NOT NULL DEFAULT '[]', error TEXT,
                batch_id TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            INSERT INTO document_representations (doc_id, source_uri)
            VALUES ('old_case', '/corpus/old_case');
            """
        )

    upsert_representations([_case("new_case")], db_path=db_path)

    assert get_representation("old_case", db_path=db_path).classification_status == "pending"
    assert get_representation("new_case", db_path=db_path).classification_status == "pending"
    assert get_representation("old_case", db_path=db_path).statute_citations == []


def test_column_migration_is_idempotent(db):
    with connection_scope(db) as conn:
        assert ensure_representation_columns(conn) == []  # schema already current


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_document_thresholds_resolve_to_the_configured_values():
    document = get_settings().document

    assert document.min_characters == 1200
    assert document.min_words == 250


def test_classification_thresholds_resolve_to_the_configured_values():
    classification = get_settings().classification

    assert classification.auto_accept_threshold == 0.80
    assert classification.review_threshold == 0.50
    assert classification.review_threshold < classification.auto_accept_threshold


def test_signal_weights_resolve_and_sum_to_one():
    signals = get_settings().classification.signals

    assert signals.statute == 0.45
    assert signals.cluster == 0.25
    assert signals.court == 0.15
    assert signals.source_folder == 0.10
    assert signals.title == 0.05
    assert signals.total == pytest.approx(1.00, abs=1e-9)


def test_signal_weights_that_do_not_sum_to_one_are_rejected():
    with pytest.raises(ValueError, match="sum to 1.0"):
        ClassificationSignalWeights(
            statute=0.5, cluster=0.25, court=0.15, source_folder=0.10, title=0.05
        )


def test_unknown_signal_is_rejected():
    with pytest.raises(ValueError):
        ClassificationSignalWeights(
            statute=0.45, cluster=0.25, court=0.15, source_folder=0.10,
            title=0.05, headnote=0.0,
        )


def test_document_threshold_drives_the_short_text_warning(tmp_path):
    """The formerly hardcoded threshold is now the configured one."""

    from src.extraction.case_loader import DEFAULT_MIN_CHARACTERS, load_case_folder

    assert DEFAULT_MIN_CHARACTERS == get_settings().document.min_characters

    folder = tmp_path / "case_x"
    folder.mkdir()
    folder.joinpath("case.html").write_text(
        "<html><body><h1>Order</h1><p>" + ("word " * 100) + "</p></body></html>",
        encoding="utf-8",
    )

    lenient = load_case_folder(folder, root=tmp_path, min_characters=100)
    strict = load_case_folder(folder, root=tmp_path, min_characters=5_000)

    assert "short_text" not in lenient.warnings
    assert "short_text" in strict.warnings


def test_full_configuration_still_loads(tmp_path):
    """base.yaml must remain valid YAML and a valid Settings document."""

    settings = load_settings(config_dir="config")

    assert settings.document == DocumentSettings(min_characters=1200, min_words=250)
    assert settings.caselaw.representation_schema_file.name.endswith(".sql")
    assert settings.classification.taxonomy_file == Path("config/domains.yaml")


def test_base_yaml_parses_as_yaml():
    payload = yaml.safe_load(Path("config/base.yaml").read_text())

    assert payload["document"]["min_characters"] == 1200
    assert payload["classification"]["signals"]["statute"] == 0.45


# ---------------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------------


def test_frozen_registry_defines_the_three_initial_domains():
    taxonomy = load_frozen_taxonomy()

    assert taxonomy.domain_ids == {"family_law", "criminal_law", OTHER_DOMAIN_ID}
    assert taxonomy.assignable_ids == {"family_law", "criminal_law", OTHER_DOMAIN_ID}
    assert taxonomy.version


def test_each_domain_carries_definitions_the_registry_can_validate():
    taxonomy = load_frozen_taxonomy()

    for domain in taxonomy.domains:
        assert domain.name
        assert domain.description
        assert domain.inclusion_criteria
        if not domain.is_other:
            assert domain.exclusion_criteria
            assert domain.keywords


def test_domains_are_not_hardcoded_in_python():
    """The registry is the source of truth, not a Python constant."""

    import subprocess

    hits = subprocess.run(
        ["grep", "-rn", "family_law", "--include=*.py", "src", "orchestration"],
        capture_output=True, text=True,
    ).stdout.strip()

    assert hits == "", f"domain ids hardcoded in Python: {hits}"
