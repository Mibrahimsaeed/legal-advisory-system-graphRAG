"""Promote reviewed draft domains into the frozen registry.

This is the human step Phase 5 stops before: it takes ``domain_candidates``
rows a person has actually read, writes them to ``config/domains.yaml``,
and marks them ``status='accepted'``. Nothing calls it automatically --
it exists so that "freeze the taxonomy" is one auditable action rather
than hand-edited YAML.

Two safety rules:

* **A domain flagged for review is never frozen implicitly.** Candidates
  with ``review_required=1`` (Phase 4 flagged their cluster as possibly
  mixed) must be named explicitly in ``accepted_domain_ids``, or they are
  left behind with a warning.
* **An existing frozen taxonomy is never clobbered.** Freezing over a
  non-empty ``domains.yaml`` requires ``overwrite=True``, because
  classifications already written reference the version it holds.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.common.db import DEFAULT_DB_PATH, connection_scope
from src.common.exceptions import ConfigurationError
from src.common.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_TAXONOMY_FILE = Path("config/domains.yaml")


def freeze_taxonomy(
    run_id: str,
    accepted_domain_ids: list[str] | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    output_path: str | Path = DEFAULT_TAXONOMY_FILE,
    version: str | None = None,
    notes: list[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write the accepted domains of ``run_id`` to the frozen registry.

    Args:
        accepted_domain_ids: The domains a human accepted. ``None`` means
            "every candidate that is not flagged for review and is not the
            Other/Uncertain bucket".
        version: Taxonomy version string. Defaults to ``<run_id>@<date>``.
        overwrite: Required to replace an existing non-empty registry.

    Raises:
        ConfigurationError: if the run has no candidates, if a requested
            domain id is not among them, or if the registry already holds
            a taxonomy and ``overwrite`` is False.
    """

    output_path = Path(output_path)
    if output_path.exists() and output_path.read_text(encoding="utf-8").strip() and not overwrite:
        raise ConfigurationError(
            f"{output_path} already holds a frozen taxonomy. Classifications "
            "reference its version; pass overwrite=True only if you intend to "
            "supersede it (and re-run classification under a new run_id)."
        )

    with connection_scope(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM domain_candidates WHERE run_id = ? ORDER BY doc_count DESC",
            (run_id,),
        ).fetchall()

    if not rows:
        raise ConfigurationError(f"No domain candidates found for run {run_id!r}")

    by_id = {r["domain_id"]: r for r in rows if r["domain_id"]}

    if accepted_domain_ids is None:
        accepted = [
            r for r in rows
            if r["domain_id"]
            and not r["is_other_bucket"]
            and not r["review_required"]
        ]
        skipped = [
            r["domain_id"] for r in rows
            if r["domain_id"] and not r["is_other_bucket"] and r["review_required"]
        ]
        if skipped:
            logger.warning(
                "Not freezing flagged domain(s) %s: they were marked "
                "review_required by the cluster review. Name them explicitly "
                "in accepted_domain_ids if a human has cleared them.",
                ", ".join(skipped),
            )
    else:
        unknown = [d for d in accepted_domain_ids if d not in by_id]
        if unknown:
            raise ConfigurationError(
                f"Unknown domain id(s) for run {run_id!r}: {', '.join(unknown)}"
            )
        accepted = [by_id[d] for d in accepted_domain_ids]

    if not accepted:
        raise ConfigurationError(
            f"Nothing to freeze for run {run_id!r}: no accepted domains "
            "(every candidate was flagged for review or excluded)."
        )

    payload = {
        "version": version or f"{run_id}@{datetime.now(timezone.utc).date().isoformat()}",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "source_run_id": run_id,
        "notes": notes or [
            "Frozen from reviewed draft candidates. Classification rows stamp "
            "this version; changing it means re-classifying under a new run_id.",
        ],
        "domains": [
            {
                "id": r["domain_id"],
                "name": r["name"],
                "description": r["description"] or "",
                "inclusion_criteria": json.loads(r["inclusion_criteria"] or "[]"),
                "exclusion_criteria": json.loads(r["exclusion_criteria"] or "[]"),
                "keywords": json.loads(r["keywords_json"] or "[]"),
            }
            for r in accepted
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    with connection_scope(db_path) as conn:
        conn.executemany(
            "UPDATE domain_candidates SET status = 'accepted' WHERE run_id = ? AND domain_id = ?",
            [(run_id, r["domain_id"]) for r in accepted],
        )

    logger.info(
        "Froze %d domain(s) from run %s into %s (version=%s)",
        len(accepted), run_id, output_path, payload["version"],
    )
    return output_path
