"""The Stage 1.2 output record: a draft taxonomy card.

Explicitly a *draft* -- nothing here writes to ``config/domains.yaml``
(the frozen domain registry) or promotes a candidate automatically.
A :class:`TaxonomyCard` is one full run's output: the top-N discovered
domains (as :class:`DomainDraft`), the "Other / Uncertain" bucket, and
enough run metadata (sample size, embedding model, timestamp) to make
sense of it later. Persisted both as JSON (for human review) and as
``domain_candidates`` rows (``schemas/domain_registry_schema.sql``, all
``status='draft'``) for programmatic access from later stages.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.common.db import DEFAULT_DB_PATH, connection_scope
from src.common.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_DOMAIN_REGISTRY_SCHEMA_FILE = Path("schemas/domain_registry_schema.sql")


@dataclass
class DomainDraft:
    """One discovered cluster, promoted to a draft domain definition."""

    cluster_id: int
    name: str
    description: str
    inclusion_criteria: list[str] = field(default_factory=list)
    exclusion_criteria: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    representative_doc_ids: list[str] = field(default_factory=list)
    doc_count: int = 0
    sample_size: int = 0
    error: str | None = None
    # Present only when this row IS the "Other / Uncertain" bucket rather
    # than a real discovered cluster (see build_other_bucket_draft below).
    is_other_bucket: bool = False


def build_other_bucket_draft(
    doc_ids: list[str], sample_size: int
) -> DomainDraft:
    """The "Other / Uncertain" bucket, packaged the same shape as a real
    :class:`DomainDraft` so downstream persistence doesn't need a special case."""

    return DomainDraft(
        cluster_id=-1,
        name="Other / Uncertain",
        description=(
            "Documents that did not clearly belong to any of the top "
            "discovered domains -- either HDBSCAN noise (no dense enough "
            "neighborhood) or members of a smaller cluster below the "
            "top-N cutoff. Needs human review before being split out into "
            "real domains or left as a permanent catch-all."
        ),
        inclusion_criteria=[],
        exclusion_criteria=[],
        keywords=[],
        representative_doc_ids=doc_ids[:20],
        doc_count=len(doc_ids),
        sample_size=sample_size,
        is_other_bucket=True,
    )


@dataclass
class TaxonomyCard:
    """One full Stage 1.2 run's output."""

    run_id: str
    generated_at: str
    sample_size: int
    total_signatures: int
    embedding_model: str
    domains: list[DomainDraft] = field(default_factory=list)
    other_bucket: DomainDraft | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def new_run_id() -> str:
    return f"discovery_{uuid.uuid4().hex[:12]}"


def build_taxonomy_card(
    run_id: str,
    sample_size: int,
    total_signatures: int,
    embedding_model: str,
    domains: list[DomainDraft],
    other_bucket: DomainDraft,
    notes: list[str] | None = None,
) -> TaxonomyCard:
    return TaxonomyCard(
        run_id=run_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        sample_size=sample_size,
        total_signatures=total_signatures,
        embedding_model=embedding_model,
        domains=domains,
        other_bucket=other_bucket,
        notes=notes or [],
    )


def write_taxonomy_card_json(
    card: TaxonomyCard, output_dir: str | Path
) -> Path:
    """Persist the full card as human-reviewable JSON. Returns the file path."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{card.run_id}.json"

    try:
        path.write_text(json.dumps(card.to_dict(), indent=2, ensure_ascii=False))
    except OSError as exc:
        logger.error("Failed to write taxonomy card to %s: %s", path, exc)
        raise

    logger.info("Wrote draft taxonomy card to %s", path)
    return path


def _row_for(draft: DomainDraft, run_id: str) -> tuple:
    candidate_id = f"{run_id}_{draft.cluster_id if not draft.is_other_bucket else 'other'}"
    return (
        candidate_id,
        run_id,
        None if draft.is_other_bucket else draft.cluster_id,
        int(draft.is_other_bucket),
        draft.name,
        draft.description,
        json.dumps(draft.inclusion_criteria, ensure_ascii=False),
        json.dumps(draft.exclusion_criteria, ensure_ascii=False),
        json.dumps(draft.keywords, ensure_ascii=False),
        json.dumps(draft.representative_doc_ids, ensure_ascii=False),
        draft.doc_count,
        draft.sample_size,
        "draft",
    )


def persist_taxonomy_card(
    card: TaxonomyCard,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Insert one ``domain_candidates`` row per domain (+ the other bucket).

    Idempotent per ``run_id``: re-running with the same run_id (e.g. a
    checkpoint resume that redoes the ``persist`` phase) replaces rows
    rather than duplicating them.
    """

    rows = [_row_for(d, card.run_id) for d in card.domains]
    if card.other_bucket is not None:
        rows.append(_row_for(card.other_bucket, card.run_id))

    if not rows:
        return 0

    with connection_scope(db_path) as conn:
        conn.execute("DELETE FROM domain_candidates WHERE run_id = ?", (card.run_id,))
        conn.executemany(
            """
            INSERT INTO domain_candidates (
                candidate_id, run_id, cluster_id, is_other_bucket,
                name, description, inclusion_criteria, exclusion_criteria,
                keywords_json, representative_doc_ids_json,
                doc_count, sample_size, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    logger.info(
        "Persisted %d draft domain candidate(s) for run %s", len(rows), card.run_id
    )
    return len(rows)


def get_candidates_for_run(
    run_id: str, db_path: str | Path = DEFAULT_DB_PATH
) -> list[sqlite3.Row]:
    with connection_scope(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM domain_candidates WHERE run_id = ? ORDER BY doc_count DESC",
            (run_id,),
        ).fetchall()
    return rows