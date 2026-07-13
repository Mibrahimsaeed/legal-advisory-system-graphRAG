from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.common.db import connection_scope, init_schema
from src.ingestion import manifest as manifest_db
from src.ingestion.batch_puller import StorageBackend, pull_batch
from src.ingestion.scratch_manager import scratch_workspace, purge_stale_workspaces
from orchestration.dags.batch_ingest_flow import run_stage0


SCHEMA_FILE = "schemas/manifest_schema.sql"


@pytest.fixture()
def db_path(tmp_path) -> Path:
    p = tmp_path / "metadata.db"
    init_schema(db_path=p, schema_file=SCHEMA_FILE)
    return p


@pytest.fixture()
def scratch_root(tmp_path) -> Path:
    return tmp_path / "scratch"


class FakeBackend(StorageBackend):
    def __init__(self, contents: dict[str, bytes], fail_uris: set[str] | None = None):
        self.contents = contents
        self.fail_uris = fail_uris or set()

    def download(self, source_uri: str, local_path: Path) -> None:
        if source_uri in self.fail_uris:
            raise ConnectionError(f"simulated download failure for {source_uri}")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(self.contents[source_uri])


def _checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _register_fake_docs(db_path: Path, n: int, prefix: str = "doc") -> dict[str, bytes]:
    contents: dict[str, bytes] = {}
    entries = []

    for i in range(n):
        doc_id = f"{prefix}_{i}"
        uri = f"s3://landing/{doc_id}.txt"
        data = f"content for {doc_id}".encode()

        contents[uri] = data

        entries.append(
            {
                "doc_id": doc_id,
                "source_uri": uri,
                "checksum": _checksum(data),
            }
        )

    manifest_db.register_documents(entries, db_path=db_path)

    return contents


def test_register_documents_is_idempotent_on_doc_id(db_path):
    entries = [
        {
            "doc_id": "d1",
            "source_uri": "s3://b/d1.pdf",
            "checksum": "abc",
        }
    ]

    first = manifest_db.register_documents(entries, db_path=db_path)
    second = manifest_db.register_documents(entries, db_path=db_path)

    assert first == 1
    assert second == 0


def test_claim_batch_marks_docs_claimed_and_sets_batch_id(db_path):
    _register_fake_docs(db_path, 5)

    batch_id, entries = manifest_db.claim_batch(
        batch_size=3,
        db_path=db_path,
    )

    assert len(entries) == 3
    assert all(e.ingest_status == "claimed" for e in entries)
    assert all(e.batch_id == batch_id for e in entries)

    with connection_scope(db_path) as conn:
        remaining = conn.execute(
            """
            SELECT COUNT(*) AS c
            FROM document_manifest
            WHERE ingest_status = 'pending'
            """
        ).fetchone()["c"]

    assert remaining == 2


def test_claim_batch_does_not_double_claim_across_calls(db_path):
    _register_fake_docs(db_path, 4)

    _, first_batch = manifest_db.claim_batch(
        batch_size=3,
        db_path=db_path,
    )

    _, second_batch = manifest_db.claim_batch(
        batch_size=3,
        db_path=db_path,
    )

    first_ids = {e.doc_id for e in first_batch}
    second_ids = {e.doc_id for e in second_batch}

    assert first_ids.isdisjoint(second_ids)
    assert len(second_ids) == 1


def test_claim_batch_with_no_pending_docs_returns_empty(db_path):
    batch_id, entries = manifest_db.claim_batch(
        batch_size=10,
        db_path=db_path,
    )

    assert entries == []
    assert batch_id


def test_mark_status_records_error_and_increments_attempts(db_path):
    _register_fake_docs(db_path, 1)

    _, entries = manifest_db.claim_batch(
        batch_size=1,
        db_path=db_path,
    )

    doc_id = entries[0].doc_id

    manifest_db.mark_status(
        [doc_id],
        "failed",
        db_path=db_path,
        error="boom",
        increment_attempts=True,
    )

    with connection_scope(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM document_manifest WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()

    assert row["ingest_status"] == "failed"
    assert row["last_error"] == "boom"
    assert row["attempts"] == 1


def test_requeue_failed_respects_max_attempts(db_path):
    _register_fake_docs(db_path, 1)

    batch_id, entries = manifest_db.claim_batch(
        batch_size=1,
        db_path=db_path,
    )

    doc_id = entries[0].doc_id

    for _ in range(3):
        manifest_db.mark_status(
            [doc_id],
            "failed",
            db_path=db_path,
            error="x",
            increment_attempts=True,
        )

    requeued = manifest_db.requeue_failed(
        batch_id,
        max_attempts=3,
        db_path=db_path,
    )

    assert requeued == 0

    with connection_scope(db_path) as conn:
        row = conn.execute(
            "SELECT ingest_status FROM document_manifest WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()

    assert row["ingest_status"] == "failed"


def test_scratch_workspace_purged_on_success(scratch_root):
    with scratch_workspace("batch_ok", root=scratch_root) as ws:
        f = ws.path_for_doc("d1", ".txt")
        f.write_text("hello")
        assert ws.batch_dir.exists()

    assert not ws.batch_dir.exists()


def test_scratch_workspace_purged_even_when_body_raises(scratch_root):
    ws_ref = {}

    with pytest.raises(RuntimeError):
        with scratch_workspace("batch_crash", root=scratch_root) as ws:
            ws_ref["ws"] = ws
            (ws.raw_dir / "partial.txt").write_text("oops")
            raise RuntimeError("simulated crash mid-batch")

    assert not ws_ref["ws"].batch_dir.exists()


def test_purge_stale_workspaces_skips_active_batches(scratch_root):
    scratch_root.mkdir(parents=True, exist_ok=True)

    (scratch_root / "batch_active" / "raw").mkdir(parents=True)
    (scratch_root / "batch_orphaned" / "raw").mkdir(parents=True)

    removed = purge_stale_workspaces(
        root=scratch_root,
        known_active_batch_ids={"batch_active"},
    )

    assert removed == 1
    assert (scratch_root / "batch_active").exists()
    assert not (scratch_root / "batch_orphaned").exists()


def test_pull_batch_verifies_checksum_and_marks_pulled(
    db_path,
    scratch_root,
):
    contents = _register_fake_docs(db_path, 2)

    batch_id, entries = manifest_db.claim_batch(
        batch_size=2,
        db_path=db_path,
    )

    backend = FakeBackend(contents)

    with scratch_workspace(batch_id, root=scratch_root) as ws:
        results = pull_batch(
            entries,
            ws,
            backend=backend,
            db_path=db_path,
        )

        assert all(r.success for r in results)

        for r in results:
            assert r.local_path.exists()

    with connection_scope(db_path) as conn:
        statuses = {
            row["doc_id"]: row["ingest_status"]
            for row in conn.execute(
                "SELECT doc_id, ingest_status FROM document_manifest"
            )
        }

    assert all(s == "pulled" for s in statuses.values())


def test_pull_batch_detects_checksum_mismatch_and_does_not_mark_pulled(
    db_path,
    scratch_root,
):
    entries_in = [
        {
            "doc_id": "bad1",
            "source_uri": "s3://landing/bad1.txt",
            "checksum": "deadbeef" * 8,
        }
    ]

    manifest_db.register_documents(entries_in, db_path=db_path)

    batch_id, entries = manifest_db.claim_batch(
        batch_size=1,
        db_path=db_path,
    )

    backend = FakeBackend(
        {
            "s3://landing/bad1.txt": b"actual content, wrong checksum"
        }
    )

    with scratch_workspace(batch_id, root=scratch_root) as ws:
        results = pull_batch(
            entries,
            ws,
            backend=backend,
            db_path=db_path,
        )

        assert results[0].success is False
        assert "Checksum mismatch" in results[0].error
        assert not ws.path_for_doc("bad1", ".txt").exists()

    with connection_scope(db_path) as conn:
        row = conn.execute(
            """
            SELECT ingest_status, attempts
            FROM document_manifest
            WHERE doc_id = 'bad1'
            """
        ).fetchone()

    assert row["ingest_status"] == "failed"
    assert row["attempts"] == 1