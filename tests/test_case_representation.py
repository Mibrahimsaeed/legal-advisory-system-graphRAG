"""Phase 2: case.html + metadata.json -> case-law document representation.

Driven off the representative sample in ``tests/fixtures/caselaw/``, which
covers the shapes a real Pakistani case-law corpus actually contains:

    supreme_court/2019_scmr_123               complete metadata, clean markup
    lahore_high_court/2021_plj_88             partial metadata, day-first date,
                                              bench as one string, layout table,
                                              unclosed <p> tags
    sindh_high_court/no_metadata_case         no metadata.json at all
    federal_shariat_court/bad_metadata_case   metadata.json is a JSON array
    islamabad_high_court/legacy_encoding_case cp1252 bytes, no declared charset
    tribunal/malformed_case                   unclosed <h1>, stray "<", bad entity
    tribunal/empty_case                       portal stub with no judgment text
    orphans/metadata_only_case                metadata.json but no case.html
"""

from __future__ import annotations

import json
import shutil
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchestration.dags.case_ingest_flow import run_case_ingest
from src.common.db import init_schema
from src.embedding.doc_pooling import embed_documents
from src.embedding.embed_model import DeterministicHashEmbedder
from src.extraction.case_loader import (
    ERROR_EMPTY_DOCUMENT,
    ERROR_MISSING_HTML,
    WARNING_FALLBACK_ENCODING,
    WARNING_INVALID_METADATA,
    WARNING_MISSING_METADATA,
    WARNING_NO_HEADINGS,
    WARNING_SHORT_TEXT,
    decode_html_bytes,
    doc_id_for_case,
    iter_case_folders,
    load_case_folder,
    parse_case_html,
)
from src.extraction.case_metadata import (
    WARNING_UNEXPECTED_TYPE,
    WARNING_UNPARSEABLE_DATE,
    extract_case_metadata,
    parse_citation,
    parse_date,
    parse_judges,
)
from src.extraction.representation_store import (
    DEFAULT_REPRESENTATION_SCHEMA_FILE,
    get_representation,
    list_representations,
    upsert_representations,
)

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "caselaw"

SUPREME_COURT = "supreme_court/2019_scmr_123"
LAHORE = "lahore_high_court/2021_plj_88"
NO_METADATA = "sindh_high_court/no_metadata_case"
BAD_METADATA = "federal_shariat_court/bad_metadata_case"
LEGACY_ENCODING = "islamabad_high_court/legacy_encoding_case"
MALFORMED = "tribunal/malformed_case"
EMPTY = "tribunal/empty_case"
ORPHAN = "orphans/metadata_only_case"


# The fixture cases are deliberately short excerpts; measuring them against
# the configured production threshold (document.min_characters = 1200) would
# flag every one as short_text and turn these metadata/encoding tests into
# length tests. The configured value is exercised in
# tests/test_phase1_foundation.py instead.
FIXTURE_MIN_CHARACTERS = 200


def _load(relpath: str, **kwargs):
    kwargs.setdefault("min_characters", FIXTURE_MIN_CHARACTERS)
    return load_case_folder(FIXTURE_ROOT / relpath, root=FIXTURE_ROOT, **kwargs)


@pytest.fixture()
def caselaw_db(tmp_path) -> Path:
    db_path = tmp_path / "metadata.db"
    init_schema(db_path=db_path, schema_file=str(DEFAULT_REPRESENTATION_SCHEMA_FILE))
    return db_path


# ---------------------------------------------------------------------------
# The representative sample: full extraction schema
# ---------------------------------------------------------------------------


def test_complete_case_extracts_every_schema_field():
    rep = _load(SUPREME_COURT)

    # identity + provenance
    assert rep.doc_id == doc_id_for_case(FIXTURE_ROOT, FIXTURE_ROOT / SUPREME_COURT)
    assert rep.source_relpath == SUPREME_COURT
    assert Path(rep.source_file).name == "case.html"
    assert Path(rep.source_uri) == FIXTURE_ROOT / SUPREME_COURT
    assert rep.source_type == "case_html"
    assert len(rep.content_hash) == 64

    # classification input
    assert rep.title == "Muhammad Aslam v. The State"
    assert rep.headings == ["Facts", "Arguments of Counsel", "Findings", "Order"]
    assert "section 302(b) of the Pakistan Penal Code" in rep.body_preview
    assert rep.char_count == len(rep.full_text)

    # case-law facts
    assert rep.court == "Supreme Court of Pakistan"
    assert rep.decision_date == "2019-04-11"
    assert rep.citation == "2019 SCMR 123"
    assert rep.case_number == "Criminal Appeal No. 45 of 2018"
    assert rep.judges == [
        "Mr. Justice Asif Saeed Khan Khosa, CJ",
        "Mr. Justice Mazhar Alam Khan Miankhel",
    ]

    # outcome + verbatim metadata
    assert rep.status == "ok"
    assert rep.warnings == []
    assert rep.error is None
    assert rep.metadata["result"] == "allowed"  # unmapped keys survive verbatim


def test_html_noise_is_excluded_from_extracted_text():
    rep = _load(SUPREME_COURT)

    assert "window.analytics" not in rep.full_text  # <script>
    assert "background: #123" not in rep.full_text  # <style>
    assert "Home" not in rep.full_text.split("\n")[0]  # <nav> chrome
    assert "&amp;" not in rep.full_text and "&gt;" not in rep.full_text  # entities decoded
    assert "defended the concurrent findings &" in rep.full_text


def test_paragraph_structure_survives_but_is_not_chunked():
    rep = _load(SUPREME_COURT)

    assert "\n\n" in rep.body_preview  # block structure preserved
    assert isinstance(rep.body_preview, str)  # one blob, no chunk list
    assert not hasattr(rep, "chunks")


def test_partial_metadata_case_normalizes_bench_dates_and_citations():
    rep = _load(LAHORE)

    assert rep.court == "Lahore High Court"  # via the "court_name" alias
    assert rep.decision_date == "2021-04-11"  # "11/04/2021" read day-first
    assert rep.citation == "2021 PLJ 88"  # first of a citations list
    assert rep.judges == ["Mr. Justice Shahid Karim", "Mr. Justice Asim Hafeez"]
    assert rep.case_number is None  # absent, not invented
    assert rep.headings == ["Question of Law", "Discussion", "Conclusion"]
    assert rep.warnings == []


def test_case_without_metadata_json_is_usable_and_warned():
    rep = _load(NO_METADATA)

    assert rep.status == "ok"
    assert rep.warnings == [WARNING_MISSING_METADATA]
    assert rep.metadata == {}
    assert rep.court is None and rep.decision_date is None and rep.citation is None
    # Title falls back to the <h1>; the text is fully extracted.
    assert rep.title == "Abdul Rehman v. Province of Sindh"
    assert "Article 199" in rep.body_preview


def test_non_object_metadata_json_is_warned_not_fatal():
    rep = _load(BAD_METADATA)

    assert rep.status == "ok"
    assert rep.warnings == [f"{WARNING_INVALID_METADATA}:not_an_object"]
    assert rep.metadata == {}
    assert rep.title == "Shariat Petition No. 3 of 2018"


def test_unreadable_metadata_json_is_warned_not_fatal(tmp_path):
    folder = tmp_path / "case"
    shutil.copytree(FIXTURE_ROOT / NO_METADATA, folder)
    (folder / "metadata.json").write_text("{'court': not json,", encoding="utf-8")

    rep = load_case_folder(folder, root=tmp_path)

    assert rep.status == "ok"
    assert any(w.startswith(WARNING_INVALID_METADATA) for w in rep.warnings)


def test_legacy_cp1252_export_decodes_without_mojibake():
    rep = _load(LEGACY_ENCODING)

    assert rep.status == "ok"
    assert rep.warnings == [f"{WARNING_FALLBACK_ENCODING}:cp1252"]
    assert "�" not in rep.full_text  # no replacement characters
    assert "Muhammad Ashraf – Petitioner" in rep.title  # en dash intact
    assert "petitioner’s grievance" in rep.full_text  # curly apostrophe
    assert "“fitness”" in rep.full_text  # curly quotes
    assert rep.decision_date == "2015-01-01"  # bare year normalized
    assert rep.judges == ["Mr. Justice Athar Minallah"]


def test_malformed_markup_still_yields_text_and_a_title():
    rep = _load(MALFORMED)

    assert rep.status == "ok"
    # <h1> is never closed; the following block implies its end, so the
    # case name is recovered as the title instead of the folder name.
    assert rep.title == "Nasreen Akhtar v. Secretary Education"
    assert WARNING_NO_HEADINGS in rep.warnings
    assert "seniority list issued on 5 < 6 grounds" in rep.full_text  # stray "<"
    # "&amp" without its semicolon still decodes (HTML5 rule), so the text
    # reads correctly rather than leaking raw entity syntax.
    assert "applicant & errors" in rep.full_text
    assert rep.court == "Punjab Service Tribunal"
    assert rep.decision_date == "2020-09-23"  # "23 September 2020"


def test_portal_stub_with_no_judgment_text_fails_explicitly():
    rep = _load(EMPTY)

    assert rep.status == "failed"
    assert rep.error.startswith(ERROR_EMPTY_DOCUMENT)
    assert rep.body_preview == "" and rep.char_count == 0
    # Provenance is still recorded, so the failure is triageable.
    assert rep.source_relpath == EMPTY
    assert Path(rep.source_file).exists()


def test_folder_without_case_html_fails_explicitly():
    rep = _load(ORPHAN)

    assert rep.status == "failed"
    assert rep.error.startswith(ERROR_MISSING_HTML)


def test_scanner_only_yields_folders_that_contain_case_html():
    relpaths = {
        f.relative_to(FIXTURE_ROOT).as_posix() for f in iter_case_folders(FIXTURE_ROOT)
    }

    assert ORPHAN not in relpaths
    assert relpaths == {
        SUPREME_COURT, LAHORE, NO_METADATA, BAD_METADATA,
        LEGACY_ENCODING, MALFORMED, EMPTY,
    }


def test_no_case_in_the_sample_raises():
    """The contract the 10k-case scan depends on: loading never throws."""

    for folder in list(iter_case_folders(FIXTURE_ROOT)) + [FIXTURE_ROOT / ORPHAN]:
        rep = load_case_folder(folder, root=FIXTURE_ROOT)
        assert rep.status in {"ok", "failed"}
        assert (rep.error is None) == (rep.status == "ok")


def test_short_text_is_flagged_but_still_stored(tmp_path):
    folder = tmp_path / "short_case"
    folder.mkdir()
    (folder / "case.html").write_text(
        "<html><body><h1>Order</h1><p>Dismissed as withdrawn.</p></body></html>",
        encoding="utf-8",
    )

    rep = load_case_folder(folder, root=tmp_path)

    assert rep.status == "ok"
    assert WARNING_SHORT_TEXT in rep.warnings


def test_body_preview_is_capped_while_char_count_reports_full_length(tmp_path):
    folder = tmp_path / "long_case"
    folder.mkdir()
    (folder / "case.html").write_text(
        "<html><body><h1>Long Judgment</h1><p>" + ("evidence " * 5_000) + "</p></body></html>",
        encoding="utf-8",
    )

    rep = load_case_folder(folder, root=tmp_path, body_preview_char_limit=1_000)

    assert len(rep.body_preview) == 1_000
    assert rep.char_count > 40_000
    assert len(rep.full_text) == rep.char_count


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_provenance_round_trips_back_to_the_source_file():
    rep = _load(SUPREME_COURT)

    reloaded = load_case_folder(Path(rep.source_uri), root=FIXTURE_ROOT)

    assert Path(rep.source_file).read_bytes()  # the recorded file still opens
    assert reloaded.content_hash == rep.content_hash  # extraction is deterministic
    assert reloaded.doc_id == rep.doc_id


def test_content_hash_detects_changed_case_text(tmp_path):
    folder = tmp_path / "case"
    shutil.copytree(FIXTURE_ROOT / SUPREME_COURT, folder)
    before = load_case_folder(folder, root=tmp_path)

    html = (folder / "case.html").read_text(encoding="utf-8")
    (folder / "case.html").write_text(
        html.replace("The appeal is allowed", "The appeal is dismissed"), encoding="utf-8"
    )
    after = load_case_folder(folder, root=tmp_path)

    assert after.doc_id == before.doc_id  # same case
    assert after.content_hash != before.content_hash  # changed content


def test_doc_id_depends_on_relpath_not_on_the_mount_point(tmp_path):
    """A corpus copied to another path keeps every doc_id."""

    copied_root = tmp_path / "elsewhere"
    shutil.copytree(FIXTURE_ROOT / SUPREME_COURT, copied_root / SUPREME_COURT)

    original = _load(SUPREME_COURT)
    moved = load_case_folder(copied_root / SUPREME_COURT, root=copied_root)

    assert moved.doc_id == original.doc_id
    assert moved.source_relpath == original.source_relpath
    assert moved.source_uri != original.source_uri


# ---------------------------------------------------------------------------
# metadata.json normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2019-04-11", "2019-04-11"),
        ("2019-04-11T09:30:00Z", "2019-04-11"),
        ("11/04/2021", "2021-04-11"),  # day-first
        ("11-04-2021", "2021-04-11"),
        ("14.03.2016", "2016-03-14"),
        ("23 September 2020", "2020-09-23"),
        ("April 11, 2019", "2019-04-11"),
        ("2015", "2015-01-01"),
        (2015, "2015-01-01"),
        (date(2019, 4, 11), "2019-04-11"),
        (datetime(2019, 4, 11, 9, 30), "2019-04-11"),
        ("not available", None),
        ("", None),
        (None, None),
        ({"year": 2019}, None),
    ],
)
def test_parse_date(raw, expected):
    assert parse_date(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (["A. Khan", "B. Ahmed"], ["A. Khan", "B. Ahmed"]),
        ([{"name": "A. Khan"}, {"name": "B. Ahmed"}], ["A. Khan", "B. Ahmed"]),
        ("A. Khan and B. Ahmed", ["A. Khan", "B. Ahmed"]),
        ("A. Khan; B. Ahmed", ["A. Khan", "B. Ahmed"]),
        ("A. Khan & B. Ahmed", ["A. Khan", "B. Ahmed"]),
        # Commas are NOT split points: judicial suffixes contain them.
        ("Mr. Justice Asif Saeed Khan Khosa, CJ", ["Mr. Justice Asif Saeed Khan Khosa, CJ"]),
        ("  ", []),
        (42, []),
    ],
)
def test_parse_judges(raw, expected):
    assert parse_judges(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2019 SCMR 123", "2019 SCMR 123"),
        (["2021 PLJ 88", "2021 PTD 456"], "2021 PLJ 88"),
        ([], None),
        ([None, "2020 CLC 1"], "2020 CLC 1"),
        ({"reported": "x"}, None),
    ],
)
def test_parse_citation(raw, expected):
    assert parse_citation(raw) == expected


def test_metadata_key_aliases_are_case_and_separator_insensitive():
    meta = extract_case_metadata(
        {
            "caseTitle": "A v. B",
            "Court Name": "Peshawar High Court",
            "Decided On": "2022-01-05",
            "Case No": "W.P. 1 of 2022",
        }
    )

    assert meta.title == "A v. B"
    assert meta.court == "Peshawar High Court"
    assert meta.decision_date == "2022-01-05"
    assert meta.case_number == "W.P. 1 of 2022"


def test_metadata_records_warnings_for_unusable_values():
    meta = extract_case_metadata(
        {"court": {"name": "SC"}, "date": "sometime in 2019", "judges": 7}
    )

    assert meta.court is None and meta.decision_date is None and meta.judges == []
    assert f"{WARNING_UNEXPECTED_TYPE}:court" in meta.warnings
    assert f"{WARNING_UNPARSEABLE_DATE}:date" in meta.warnings
    assert f"{WARNING_UNEXPECTED_TYPE}:judges" in meta.warnings


def test_empty_metadata_yields_empty_fields_without_warnings():
    meta = extract_case_metadata({})

    assert meta == extract_case_metadata({"unrelated_key": None})
    assert meta.warnings == []
    assert meta.judges == []


# ---------------------------------------------------------------------------
# HTML decoding / parsing units
# ---------------------------------------------------------------------------


def test_decode_html_bytes_prefers_declared_charset():
    raw = '<html><head><meta charset="cp1252"></head><body>café</body></html>'.encode("cp1252")

    html, encoding, warnings = decode_html_bytes(raw)

    assert "café" in html
    assert encoding == "cp1252"
    assert warnings == []  # declared, not a fallback


def test_decode_html_bytes_handles_utf8_bom():
    html, encoding, warnings = decode_html_bytes("<html>café</html>".encode("utf-8-sig"))

    assert html.startswith("<html>")
    assert encoding == "utf-8-sig" and warnings == []


def test_parse_case_html_on_empty_and_whitespace_input():
    assert parse_case_html("") == ("", [], None)
    assert parse_case_html("   \n  ") == ("", [], None)


def test_parse_case_html_survives_truncated_markup():
    text, headings, _ = parse_case_html("<html><body><h2>Order</h2><p>Cut off mid-sent")

    assert headings == ["Order"]
    assert "Cut off mid-sent" in text


# ---------------------------------------------------------------------------
# SQL storage of the new schema
# ---------------------------------------------------------------------------


def test_store_round_trip_preserves_every_case_field(caselaw_db):
    rep = _load(SUPREME_COURT)

    upsert_representations([rep], db_path=caselaw_db)
    fetched = get_representation(rep.doc_id, db_path=caselaw_db)

    for field in (
        "doc_id", "source_uri", "source_file", "source_relpath", "content_hash",
        "source_type", "title", "headings", "body_preview", "char_count",
        "court", "decision_date", "citation", "judges", "case_number",
        "status", "warnings", "error", "metadata",
    ):
        assert getattr(fetched, field) == getattr(rep, field), field


def test_full_text_is_not_persisted_but_is_re_derivable(caselaw_db):
    rep = _load(SUPREME_COURT)
    assert rep.full_text  # populated in memory by the loader

    upsert_representations([rep], db_path=caselaw_db)
    fetched = get_representation(rep.doc_id, db_path=caselaw_db)

    assert fetched.full_text == ""  # deliberately not a column
    # ... but provenance gets it back verbatim.
    re_derived = load_case_folder(Path(fetched.source_uri), root=FIXTURE_ROOT)
    assert re_derived.full_text == rep.full_text
    assert re_derived.content_hash == fetched.content_hash


def test_failed_records_are_stored_but_kept_out_of_the_corpus(caselaw_db):
    ok = _load(SUPREME_COURT)
    failed = _load(EMPTY)

    upsert_representations([ok, failed], db_path=caselaw_db)

    assert [r.doc_id for r in list_representations(db_path=caselaw_db)] == [ok.doc_id]
    stored_failure = get_representation(failed.doc_id, db_path=caselaw_db)
    assert stored_failure.status == "failed"
    assert stored_failure.error.startswith(ERROR_EMPTY_DOCUMENT)


# ---------------------------------------------------------------------------
# The stage end to end, over the whole sample
# ---------------------------------------------------------------------------


@pytest.fixture()
def caselaw_settings(monkeypatch):
    import orchestration.dags.case_ingest_flow as flow

    monkeypatch.setattr(
        flow,
        "get_settings",
        lambda: SimpleNamespace(
            document=SimpleNamespace(min_characters=200, min_words=250),
            # The fixture cases are short hand-written excerpts, not full
            # judgments; they are measured against the pre-Phase-1 threshold so
            # these tests stay about metadata/encoding rather than length.
            # The configured production value (1200) is covered by
            # tests/test_phase1_foundation.py.
            caselaw=SimpleNamespace(
                corpus_root=FIXTURE_ROOT,
                case_html_filename="case.html",
                metadata_filename="metadata.json",
                body_preview_char_limit=20_000,
                max_headings=50,
                representation_schema_file=DEFAULT_REPRESENTATION_SCHEMA_FILE,
            )
        ),
    )
    return flow


def test_ingest_over_the_sample_corpus_reports_every_outcome(caselaw_db, caselaw_settings):
    result = run_case_ingest(root=FIXTURE_ROOT, db_path=caselaw_db, batch_id="fixtures")

    assert len(result.succeeded_doc_ids) == 6
    assert len(result.failed_doc_ids) == 1
    assert result.failures_by_code == {ERROR_EMPTY_DOCUMENT: 1}
    assert result.warnings_by_code == {
        WARNING_MISSING_METADATA: 1,
        WARNING_INVALID_METADATA: 1,
        WARNING_FALLBACK_ENCODING: 1,
        WARNING_NO_HEADINGS: 1,
    }

    stored = list_representations(db_path=caselaw_db)
    assert len(stored) == 6
    assert all(r.batch_id == "fixtures" for r in stored)
    assert all(r.source_relpath and r.source_file for r in stored)
    # Every stored case carries usable classification text.
    assert all(r.title and r.body_preview and r.char_count > 0 for r in stored)


def test_ingest_is_idempotent_across_rescans(caselaw_db, caselaw_settings):
    first = run_case_ingest(root=FIXTURE_ROOT, db_path=caselaw_db)
    second = run_case_ingest(root=FIXTURE_ROOT, db_path=caselaw_db)

    assert first.succeeded_doc_ids == second.succeeded_doc_ids
    assert len(list_representations(db_path=caselaw_db)) == 6


def test_representations_from_the_sample_feed_classification_embeddings(
    caselaw_db, caselaw_settings
):
    """Phase 2 output -> Phase 1 embedding path, unchanged."""

    run_case_ingest(root=FIXTURE_ROOT, db_path=caselaw_db)
    stored = list_representations(db_path=caselaw_db)

    vectors = embed_documents(stored, DeterministicHashEmbedder(dimension=16))

    assert set(vectors) == {r.doc_id for r in stored}
    assert all(v.shape == (16,) for v in vectors.values())


def test_metadata_json_is_stored_verbatim(caselaw_db, caselaw_settings):
    run_case_ingest(root=FIXTURE_ROOT, db_path=caselaw_db)

    rep = get_representation(
        doc_id_for_case(FIXTURE_ROOT, FIXTURE_ROOT / LAHORE), db_path=caselaw_db
    )

    on_disk = json.loads((FIXTURE_ROOT / LAHORE / "metadata.json").read_text())
    assert rep.metadata == on_disk  # nothing dropped by normalization
    assert rep.metadata["citations"] == ["2021 PLJ 88", "2021 PTD 456"]
