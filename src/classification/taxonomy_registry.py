"""Load and validate the *frozen* domain taxonomy.

``config/domains.yaml`` is the source of truth for domains a human has
reviewed and accepted (see ``src/clustering/taxonomy_draft.py`` for where
draft candidates come from, and ``docs/domain_taxonomy.md``). Phase 6
classification reads it and nothing else: draft ``domain_candidates``
rows are explicitly *not* a classification target, because a draft can
still change under a running classification and leave half the corpus
labeled against a taxonomy that no longer exists.

Two invariants this module enforces, loudly:

* **A frozen taxonomy must exist and be complete.** An empty or missing
  ``domains.yaml``, a domain without an id/name, or duplicate ids raise
  :class:`~src.common.exceptions.ConfigurationError` rather than
  degrading into "classify against nothing".
* **Every classification is traceable to an exact taxonomy text.**
  :attr:`FrozenTaxonomy.version` is the declared ``version`` when the
  file sets one, otherwise a content hash. Either way it is stamped on
  every row the classifier writes, so a label can always be read back
  against the domain definitions that produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.common.exceptions import ConfigurationError
from src.common.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_TAXONOMY_FILE = Path("config/domains.yaml")

# The catch-all every taxonomy carries implicitly: the classifier may
# always answer "none of these" instead of forcing a domain.
OTHER_DOMAIN_ID = "other_uncertain"
OTHER_DOMAIN_NAME = "Other / Uncertain"


@dataclass(frozen=True)
class FrozenDomain:
    domain_id: str
    name: str
    description: str = ""
    inclusion_criteria: list[str] = field(default_factory=list)
    exclusion_criteria: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    @property
    def is_other(self) -> bool:
        return self.domain_id == OTHER_DOMAIN_ID


@dataclass(frozen=True)
class FrozenTaxonomy:
    version: str
    domains: list[FrozenDomain]
    source_path: str | None = None
    frozen_at: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def domain_ids(self) -> set[str]:
        return {d.domain_id for d in self.domains}

    @property
    def assignable_ids(self) -> set[str]:
        """Ids a classifier may return: the real domains plus the catch-all."""

        return self.domain_ids | {OTHER_DOMAIN_ID}

    def get(self, domain_id: str) -> FrozenDomain | None:
        return next((d for d in self.domains if d.domain_id == domain_id), None)


def _content_version(payload: dict[str, Any]) -> str:
    """Stable hash of the taxonomy's substantive content.

    Keyed on the domain definitions only, so re-serializing the file or
    editing a comment does not invent a new taxonomy version, while
    changing any criterion does.
    """

    blob = json.dumps(payload.get("domains", []), sort_keys=True, ensure_ascii=False)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()]


def load_frozen_taxonomy(
    path: str | Path = DEFAULT_TAXONOMY_FILE,
) -> FrozenTaxonomy:
    """Read, validate and version the frozen taxonomy.

    Raises:
        ConfigurationError: if the file is missing, empty, malformed, or
            defines no usable domain. Classification cannot proceed
            without a frozen taxonomy, and guessing one would silently
            corrupt the corpus's labels.
    """

    path = Path(path)
    if not path.exists():
        raise ConfigurationError(
            f"Frozen taxonomy not found at {path}. Review a draft taxonomy "
            "card and freeze it first (see src/clustering/taxonomy_freeze.py)."
        )

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ConfigurationError(
            f"Frozen taxonomy at {path} is empty. Phase 5 produces DRAFT "
            "candidates only; freezing accepted domains into this file is a "
            "deliberate human step. Nothing can be classified until it is done."
        )

    try:
        payload = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Failed to parse taxonomy YAML at {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ConfigurationError(f"Taxonomy file {path} must contain a mapping at the top level")

    entries = payload.get("domains")
    if not isinstance(entries, list) or not entries:
        raise ConfigurationError(
            f"Taxonomy file {path} defines no domains (expected a non-empty 'domains' list)"
        )

    domains: list[FrozenDomain] = []
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ConfigurationError(f"Taxonomy domain #{i + 1} in {path} is not a mapping")

        domain_id = str(entry.get("id") or entry.get("domain_id") or "").strip()
        name = str(entry.get("name") or entry.get("label") or "").strip()
        if not domain_id:
            raise ConfigurationError(f"Taxonomy domain #{i + 1} in {path} has no id")
        if not name:
            raise ConfigurationError(f"Taxonomy domain {domain_id!r} in {path} has no name")
        if domain_id in seen:
            raise ConfigurationError(f"Duplicate taxonomy domain id {domain_id!r} in {path}")
        seen.add(domain_id)

        domains.append(
            FrozenDomain(
                domain_id=domain_id,
                name=name,
                description=str(entry.get("description") or "").strip(),
                inclusion_criteria=_as_list(entry.get("inclusion_criteria")),
                exclusion_criteria=_as_list(entry.get("exclusion_criteria")),
                keywords=_as_list(entry.get("keywords")),
            )
        )

    # The catch-all is always assignable; it only needs to be *listed* if
    # the operator wants to give it their own wording.
    real_domains = [d for d in domains if not d.is_other]
    if not real_domains:
        raise ConfigurationError(
            f"Taxonomy file {path} defines only the '{OTHER_DOMAIN_ID}' bucket; "
            "at least one real domain is required to classify against"
        )

    version = str(payload.get("version") or "").strip() or _content_version(payload)

    taxonomy = FrozenTaxonomy(
        version=version,
        domains=domains,
        source_path=str(path),
        frozen_at=str(payload.get("frozen_at") or "") or None,
        notes=_as_list(payload.get("notes")),
    )
    logger.info(
        "Loaded frozen taxonomy version=%s with %d domain(s) from %s",
        taxonomy.version,
        len(taxonomy.domains),
        path,
    )
    return taxonomy
