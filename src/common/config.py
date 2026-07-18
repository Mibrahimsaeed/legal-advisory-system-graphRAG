"""Typed, layered configuration for the Legal GraphRAG pipeline.

Configuration is resolved in three layers, each overriding the previous:

    1. ``config/base.yaml``        — defaults shared by every environment.
    2. ``config/{env}.yaml``       — environment-specific overrides
                                      (``env`` comes from ``APP_ENV``,
                                      defaults to ``dev``).
    3. Environment variables        — ``APP__SECTION__FIELD=value``
                                      (double underscore separated),
                                      e.g. ``APP__PIPELINE__BATCH_SIZE=100``.

The merged mapping is validated against :class:`Settings`, a Pydantic model,
so bad or missing configuration fails fast at startup rather than as a
``KeyError`` deep in the pipeline.

Usage::

    from src.common.config import get_settings

    settings = get_settings()
    db_path = settings.database.path
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from src.common.exceptions import ConfigurationError

DEFAULT_CONFIG_DIR = Path("config")
ENV_PREFIX = "APP"
ENV_VAR_NAME = "APP_ENV"


class DatabaseSettings(BaseModel):
    """SQLite manifest database configuration."""

    path: Path = Path("var/metadata.db")
    schema_file: Path = Path("schemas/manifest_schema.sql")


class ScratchSettings(BaseModel):
    """Local scratch workspace used to stage documents during ingestion."""

    root: Path = Path("var/scratch")
    purge_on_exit: bool = True


class StorageSettings(BaseModel):
    """Source document storage backend."""

    backend: Literal["s3", "local"] = "s3"
    bucket: str | None = None
    local_root: Path = Path("var/local_storage")


class PipelineSettings(BaseModel):
    """General pipeline execution settings."""

    batch_size: int = Field(default=500, gt=0)
    checkpoint_dir: Path = Path("var/checkpoints")
    phases: list[str] = Field(
        default_factory=lambda: ["claim", "pull", "extract", "discover_domains"]
    )

class ExtractionSettings(BaseModel):
    """Stage 1 (Feature Extraction / document-signature) settings.

    See ``docs/data_retention_policy.md`` for why this stage only ever
    reads the first ``max_pages`` pages, and why nothing beyond the
    bounded fields here (title, TOC, a char-capped body preview) is ever
    persisted.
    """

    max_pages: int = Field(
        default=15, gt=0, description="Pages sampled from the front of each PDF."
    )
    min_chars_per_page: float = Field(
        default=40.0,
        ge=0,
        description="Below this many extracted chars, a page is flagged low-quality.",
    )
    min_alpha_ratio: float = Field(
        default=0.6,
        ge=0,
        le=1,
        description="Below this alphabetic-character ratio, a page is flagged low-quality.",
    )
    flagged_page_ratio: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Fraction of sampled pages that must be flagged before OCR is attempted.",
    )
    ocr_dpi: int = Field(default=300, gt=0)
    ocr_lang: str = "eng"
    body_preview_char_limit: int = Field(default=20_000, gt=0)
    signature_schema_file: Path = Path("schemas/signature_schema.sql")
    
class ExtractionSettings(BaseModel):
    """Stage 1 (Feature Extraction / document-signature) settings.

    See ``docs/data_retention_policy.md`` for why this stage only ever
    reads the first ``max_pages`` pages, and why nothing beyond the
    bounded fields here (title, TOC, a char-capped body preview) is ever
    persisted.
    """

    max_pages: int = Field(
        default=15, gt=0, description="Pages sampled from the front of each PDF."
    )
    min_chars_per_page: float = Field(
        default=40.0,
        ge=0,
        description="Below this many extracted chars, a page is flagged low-quality.",
    )
    min_alpha_ratio: float = Field(
        default=0.6,
        ge=0,
        le=1,
        description="Below this alphabetic-character ratio, a page is flagged low-quality.",
    )
    flagged_page_ratio: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Fraction of sampled pages that must be flagged before OCR is attempted.",
    )
    ocr_dpi: int = Field(default=300, gt=0)
    ocr_lang: str = "eng"
    body_preview_char_limit: int = Field(default=20_000, gt=0)
    signature_schema_file: Path = Path("schemas/signature_schema.sql")


class DiscoverySettings(BaseModel):
    """Stage 1.2 (Domain Discovery: sample -> embed -> cluster -> label) settings.

    This stage runs once against a *sample* of Stage 1 signatures, not the
    full corpus -- see ``sample_min``/``sample_max``. Nothing here writes
    to the frozen domain registry (``config/domains.yaml``); output is a
    draft taxonomy card + ``domain_candidates`` rows with ``status='draft'``.
    """

    # -- sampling --------------------------------------------------------
    sample_min: int = Field(default=1500, gt=0)
    sample_max: int = Field(default=2000, gt=0)
    sample_seed: int = 42

    # -- embedding ---------------------------------------------------------
    # Swap in a legal-tuned model (e.g. an InLegalBERT/Legal-BERT sentence
    # embedding checkpoint) via config if it outperforms the general-purpose
    # default for this corpus -- nothing downstream assumes a specific model.
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_batch_size: int = Field(default=32, gt=0)
    title_weight: float = Field(default=2.0, ge=0)
    toc_weight: float = Field(default=1.5, ge=0)
    body_weight: float = Field(default=1.0, ge=0)
    body_chunk_chars: int = Field(default=2000, gt=0)
    max_body_chunks: int = Field(default=4, gt=0)

    # -- dimensionality reduction (UMAP) -------------------------------
    umap_n_components: int = Field(default=50, gt=0)
    umap_n_neighbors: int = Field(default=15, gt=1)
    umap_min_dist: float = Field(default=0.0, ge=0)
    umap_metric: str = "cosine"
    umap_min_docs: int = Field(
        default=50,
        gt=0,
        description="Below this many sampled docs, skip UMAP and cluster on raw embeddings.",
    )

    # -- clustering (HDBSCAN) -------------------------------------------
    hdbscan_min_cluster_size: int = Field(default=15, gt=1)
    hdbscan_min_samples: int | None = None
    hdbscan_metric: str = "euclidean"

    # -- ranking / labeling ----------------------------------------------
    top_n_domains: int = Field(default=3, gt=0)
    representative_docs_per_cluster: int = Field(default=8, gt=0)
    keywords_per_cluster: int = Field(default=15, gt=0)

    # -- LLM labeling ------------------------------------------------------
    llm_model: str = "claude-sonnet-5"
    llm_max_tokens: int = Field(default=1024, gt=0)

    # -- output -------------------------------------------------------------
    taxonomy_output_dir: Path = Path("var/taxonomy")
    domain_registry_schema_file: Path = Path("schemas/domain_registry_schema.sql")


class RetrySettings(BaseModel):
    """Exponential-backoff retry policy for retryable errors."""

    max_attempts: int = Field(default=3, ge=1)
    base_delay_seconds: float = Field(default=1.0, ge=0)
    max_delay_seconds: float = Field(default=30.0, ge=0)
    multiplier: float = Field(default=2.0, ge=1.0)
    jitter: bool = True

    @field_validator("max_delay_seconds")
    @classmethod
    def _max_gte_base(cls, v: float, info: Any) -> float:
        base = info.data.get("base_delay_seconds", 0.0)
        if v < base:
            raise ValueError("max_delay_seconds must be >= base_delay_seconds")
        return v


class LoggingSettings(BaseModel):
    """Centralized logging configuration."""

    level: str = "INFO"
    json_format: bool = False

    @field_validator("level")
    @classmethod
    def _valid_level(cls, v: str) -> str:
        import logging as _logging

        if not hasattr(_logging, v.upper()):
            raise ValueError(f"Invalid log level: {v}")
        return v.upper()


class MetricsSettings(BaseModel):
    """SQLite metrics store configuration."""

    db_path: Path = Path("var/metrics.db")


class Settings(BaseModel):
    """Root, fully-validated application configuration."""

    env: str = "dev"
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    scratch: ScratchSettings = Field(default_factory=ScratchSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    discovery: DiscoverySettings = Field(default_factory=DiscoverySettings)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    metrics: MetricsSettings = Field(default_factory=MetricsSettings)

    model_config = {"extra": "forbid", "frozen": True}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` on top of ``base``, returning a new dict."""

    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Failed to parse YAML config at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"Config file {path} must contain a mapping at the top level")
    return data


def _coerce_scalar(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _apply_env_overrides(merged: dict[str, Any], prefix: str = ENV_PREFIX) -> dict[str, Any]:
    """Apply ``APP__SECTION__FIELD=value`` environment variable overrides."""

    result = dict(merged)
    env_marker = f"{prefix}__"
    for key, raw_value in os.environ.items():
        if not key.startswith(env_marker):
            continue
        path = key[len(env_marker) :].lower().split("__")
        if not path:
            continue
        cursor = result
        for part in path[:-1]:
            existing = cursor.get(part)
            if not isinstance(existing, dict):
                existing = {}
                cursor[part] = existing
            cursor = existing
        cursor[path[-1]] = _coerce_scalar(raw_value)
    return result


def load_settings(
    env: str | None = None,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
) -> Settings:
    """Load and validate layered configuration.

    Args:
        env: Environment name (``dev``/``prod``/...). Falls back to the
            ``APP_ENV`` environment variable, then ``"dev"``.
        config_dir: Directory containing ``base.yaml`` and ``{env}.yaml``.

    Raises:
        ConfigurationError: if any YAML file is malformed or the merged
            configuration fails Pydantic validation.
    """

    resolved_env = env or os.environ.get(ENV_VAR_NAME, "dev")
    config_dir = Path(config_dir)

    base_config = _load_yaml(config_dir / "base.yaml")
    env_config = _load_yaml(config_dir / f"{resolved_env}.yaml")

    merged = _deep_merge(base_config, env_config)
    merged.setdefault("env", resolved_env)
    merged["env"] = resolved_env
    merged = _apply_env_overrides(merged)

    try:
        return Settings(**merged)
    except Exception as exc:  # pydantic.ValidationError and friends
        raise ConfigurationError(f"Invalid configuration for env={resolved_env!r}: {exc}") from exc


@lru_cache(maxsize=None)
def _cached_settings(env: str | None, config_dir: str) -> Settings:
    return load_settings(env=env, config_dir=config_dir)


def get_settings(
    env: str | None = None,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
) -> Settings:
    """Return process-cached, validated settings.

    Subsequent calls with the same arguments return the same instance.
    Use :func:`clear_settings_cache` (mainly in tests) to force a reload.
    """

    return _cached_settings(env, str(config_dir))


def clear_settings_cache() -> None:
    """Clear the memoized settings cache (primarily useful in tests)."""

    _cached_settings.cache_clear()