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
from pydantic import BaseModel, Field, field_validator, model_validator

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

    source_root: Path | None = Field(
        default=None,
        description=(
            "Root directory containing source documents when backend='local'. "
            "Can be an external drive or mounted filesystem."
        ),
    )

    checksum_mode: Literal["fast", "full"] = "fast"

    file_extensions: tuple[str, ...] = (".pdf",)
    
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


class CaselawSettings(BaseModel):
    """Stage 1 (case law) settings: case folders -> document representations.

    The active corpus is Pakistani case law, one folder per case holding
    ``case.html`` + ``metadata.json``. This stage replaces the PDF/book
    signature stage (:class:`ExtractionSettings`, which is kept for the
    legacy PDF path only) -- see
    ``orchestration/dags/case_ingest_flow.py``.
    """

    corpus_root: Path | None = Field(
        default=None,
        description=(
            "Directory containing the case folders (scanned recursively). "
            "Required before orchestration.dags.case_ingest_flow.run_case_ingest "
            "can run."
        ),
    )
    case_html_filename: str = "case.html"
    metadata_filename: str = "metadata.json"
    body_preview_char_limit: int = Field(default=20_000, gt=0)
    max_headings: int = Field(
        default=50,
        gt=0,
        description="Cap on <hN> headings kept per case for the embedding input.",
    )
    representation_schema_file: Path = Path(
        "schemas/document_representation_schema.sql"
    )


class DiscoverySettings(BaseModel):
    """Stage 1.2 (Domain Discovery: load -> embed -> cluster -> label) settings.

    This stage runs once against the *entire corpus* of Stage 1 signatures
    (no sampling) -- it is intended as a single, one-time pass: the
    resulting draft taxonomy is manually reviewed and frozen afterward, and
    subsequent pipeline runs do not invoke domain discovery again. Nothing
    here writes to the frozen domain registry (``config/domains.yaml``);
    output is a draft taxonomy card + ``domain_candidates`` rows with
    ``status='draft'``.
    """

    # -- corpus source -----------------------------------------------------
    # Which SQLite table this stage loads its corpus from.
    #   "representations" -> document_representations (case law; the active
    #                        path, populated by case_ingest_flow)
    #   "signatures"      -> document_signatures (legacy PDF/book path,
    #                        populated by feature_extraction_flow)
    # Nothing else about the stage changes between the two: both feed the
    # same embedding -> UMAP -> HDBSCAN -> label pipeline.
    corpus_source: Literal["representations", "signatures"] = "representations"

    # -- embedding ---------------------------------------------------------
    # Swap in a legal-tuned model (e.g. an InLegalBERT/Legal-BERT sentence
    # embedding checkpoint) via config if it outperforms the general-purpose
    # default for this corpus -- nothing downstream assumes a specific model.
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_batch_size: int = Field(
        default=32, gt=0, description="Texts per forward pass inside the model."
    )
    embedding_doc_batch_size: int = Field(
        default=256,
        gt=0,
        description=(
            "Documents per flattened encode() call. Bounds peak memory on a "
            "10k+ document corpus; does not change the resulting vectors, "
            "since one document's inputs never straddle two batches."
        ),
    )
    # Which fields of a document actually get embedded (title, headings,
    # body preview) and why court/date/citation/judges do not is documented
    # in src/embedding/doc_pooling.py.
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

    # -- taxonomy drafting (Stage 1.2b) --------------------------------------
    # Gates that stop a cluster from silently becoming a legal domain. A
    # cluster must clear BOTH floors; see src/clustering/taxonomy_draft.py.
    taxonomy_min_domain_docs: int = Field(
        default=15,
        gt=0,
        description="A cluster smaller than this is never drafted as a domain.",
    )
    taxonomy_min_domain_share: float = Field(
        default=0.02,
        ge=0,
        le=1,
        description="A cluster below this share of the reviewed set is never drafted.",
    )
    taxonomy_uncertain_membership_probability: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description=(
            "Membership probability below which a document counts as a "
            "possible mixed-domain case in the taxonomy audit."
        ),
    )

    # -- output -------------------------------------------------------------
    taxonomy_output_dir: Path = Path("var/taxonomy")
    domain_registry_schema_file: Path = Path("schemas/domain_registry_schema.sql")

    # -- review mode ---------------------------------------------------------
    # orchestration/dags/cluster_review_flow.py writes its read-only,
    # no-LLM cluster report here -- separate from taxonomy_output_dir since
    # a review report is not a taxonomy artifact and nothing here is ever
    # written to domain_candidates.
    review_output_dir: Path = Path("var/cluster_review")
    review_representative_docs_per_cluster: int = Field(default=25, gt=0)
    review_max_sample_doc_ids: int = Field(default=50, gt=0)

    # Cluster a stratified sample of this many documents instead of the
    # whole corpus. null/0 = review everything.
    review_sample_size: int | None = Field(default=None, ge=0)

    # Cluster triage thresholds (src/clustering/cluster_summary.py). These
    # decide how a cluster is *described*, never what it is named.
    review_major_min_share: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description="Share of the reviewed set at/above which a cluster is 'major'.",
    )
    review_mixed_max_mean_probability: float = Field(
        default=0.6,
        ge=0,
        le=1,
        description="Below this mean HDBSCAN membership probability, flag the cluster.",
    )
    review_mixed_max_cohesion: float = Field(
        default=0.35,
        ge=0,
        le=1,
        description=(
            "Below this mean cosine similarity to the cluster centroid (in "
            "embedding space), the cluster is probably holding more than one topic."
        ),
    )
    review_mixed_max_share: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Above this share of the reviewed set, a cluster looks like a catch-all.",
    )
    review_relative_confidence_ratio: float = Field(
        default=0.8,
        ge=0,
        le=1,
        description=(
            "Flag a cluster whose mean membership probability falls below this "
            "fraction of the median across clusters -- catches a contaminated "
            "cluster that still clears the absolute threshold."
        ),
    )
    review_court_dominance_threshold: float = Field(
        default=0.9,
        ge=0,
        le=1,
        description=(
            "If this share of a cluster comes from one court (and the corpus "
            "spans several), the cluster may be grouping by forum, not subject."
        ),
    )
    
class DocumentSettings(BaseModel):
    """Document-quality thresholds for the case-law corpus.

    Initial, deliberately tunable values -- not scientifically derived.
    ``min_characters`` is the active threshold behind
    :data:`src.extraction.case_loader.WARNING_SHORT_TEXT`;
    ``min_words`` is provided for Phase 2 structural pre-filtering and is
    not consumed yet.
    """

    min_characters: int = Field(
        default=1200,
        gt=0,
        description="Below this many extracted characters a case is flagged short.",
    )
    min_words: int = Field(
        default=250,
        gt=0,
        description="Word-count floor below which a document is structurally short.",
    )

    # -- Phase 2 structural pre-filter (src/extraction/structural_filter.py) --
    # All thresholds of the filter live here so the rules can be tuned
    # against a real corpus without touching code.
    incomplete_scrape_max_characters: int = Field(
        default=200,
        gt=0,
        description="At/below this length a document is treated as an incomplete scrape.",
    )
    procedural_max_characters: int = Field(
        default=3_000,
        gt=0,
        description=(
            "A procedural or office-report phrase can only drop a document "
            "shorter than this; longer documents need other evidence."
        ),
    )
    cause_list_min_case_numbers: int = Field(
        default=8,
        gt=0,
        description="Case numbers required before a document may be judged a cause list.",
    )
    cause_list_min_list_ratio: float = Field(
        default=0.30,
        ge=0,
        le=1,
        description="Share of lines that must look like numbered list entries.",
    )
    substantive_min_markers: int = Field(
        default=2,
        gt=0,
        description=(
            "Judgment markers that, with sufficient length and reasoning, make "
            "a document immune to the phrase-based drop rules."
        ),
    )


class ClassificationSignalWeights(BaseModel):
    """SUPERSEDED -- the Phase 1 draft signal weights.

    Kept only so an existing ``config/classification.signals`` block does
    not fail validation. **No code reads these.** The Phase 5 engine uses
    :class:`DomainDecisionWeights` instead: there is no statute extraction
    to weight at 0.45, court is metadata rather than subject evidence, and
    source-folder labels are validation ground truth that must never feed
    a classification decision. The weights must sum to 1.0 -- a silently
    unnormalized set would skew every score -- so it is validated here
    rather than discovered downstream.
    """

    statute: float = Field(default=0.45, ge=0, le=1)
    cluster: float = Field(default=0.25, ge=0, le=1)
    court: float = Field(default=0.15, ge=0, le=1)
    source_folder: float = Field(default=0.10, ge=0, le=1)
    title: float = Field(default=0.05, ge=0, le=1)

    model_config = {"extra": "forbid"}

    @property
    def total(self) -> float:
        return self.statute + self.cluster + self.court + self.source_folder + self.title

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> "ClassificationSignalWeights":
        # Tolerance covers float representation only (0.45 + 0.25 + 0.15 +
        # 0.10 + 0.05 is not exactly 1.0 in binary floating point).
        if abs(self.total - 1.0) > 1e-9:
            raise ValueError(
                f"classification.signals must sum to 1.0, got {self.total!r}"
            )
        return self


class DomainSignalSettings(BaseModel):
    """Phase 3 (case representation + domain signal generation) settings.

    Phase 3 gathers evidence -- an unsupervised cluster, keyword matches
    and a broad LLM reading -- and stores it. It does not classify;
    Phase 4 weighs these signals. Everything tunable lives here so the
    profiles and thresholds can be adjusted against a real corpus without
    a code change.
    """

    schema_file: Path = Path("schemas/domain_signals_schema.sql")
    signature_schema_file: Path = Path("schemas/case_signature_schema.sql")

    # -- representation ----------------------------------------------------
    max_text_chars: int = Field(
        default=20_000,
        gt=0,
        description="Substantive text carried per case into embedding/scanning.",
    )
    max_headings: int = Field(default=20, gt=0)

    # -- batching ----------------------------------------------------------
    batch_size: int = Field(
        default=25,
        gt=0,
        description="Cases per persisted batch; a crash costs at most one batch.",
    )

    # -- keyword signals ---------------------------------------------------
    min_keyword_matches: int = Field(
        default=2,
        gt=0,
        description=(
            "Total matches a domain profile needs before its keyword signal "
            "counts at all -- one passing mention should not register."
        ),
    )
    profiles: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Broad concept profiles per domain id, validated against the "
            "frozen taxonomy. High-level concepts only -- Phase 3 must not "
            "become statute-level extraction."
        ),
    )

    # -- LLM broad assessment ---------------------------------------------
    llm_enabled: bool = Field(
        default=True,
        description="Set false to gather only the deterministic signals.",
    )
    llm_model: str = "qwen3:14b"
    llm_max_tokens: int = Field(
        default=256,
        gt=0,
        description=(
            "256. The historical 512 truncation happened while qwen3's "
            "reasoning block was still enabled and consumed the budget "
            "before any JSON was emitted; `think=False` removed that. "
            "Measured over the 187 assessments of the sig_pilot run, the "
            "largest response the model actually produced was 284 chars "
            "(~95 tokens at a pessimistic 3 chars/token), so 256 leaves "
            "roughly 160 tokens of headroom on the worst observed case."
        ),
    )
    llm_prompt_chars: int = Field(
        default=3_000,
        gt=0,
        description="Case text shown to the assessor.",
    )


class ClusterValidationSettings(BaseModel):
    """Phase 4 (blind cluster validation) settings.

    The clustering itself is configured by :class:`DiscoverySettings` --
    this section only controls how the result is *scored* and swept. The
    verdict thresholds are tunable defaults, deliberately not presented as
    standards: the report always carries the raw measurements so a human
    can disagree with the verdict.
    """

    output_dir: Path = Path("var/cluster_validation")

    # Ground-truth labels for evaluation only, taken from this many
    # leading segments of source_relpath (1 = the top-level corpus folder).
    label_folder_depth: int = Field(default=1, gt=0)

    # Parameter sweep. "HDBSCAN carries no domain signal" and "these
    # hyperparameters do not" are different findings; the sweep separates
    # them. 0 in sweep_min_samples means "default to min_cluster_size".
    sweep_min_cluster_sizes: list[int] = Field(default_factory=lambda: [5, 10, 15, 25])
    sweep_min_samples: list[int] = Field(default_factory=lambda: [0, 5])

    # -- verdict thresholds (tunable, not standards) --------------------
    min_coverage: float = Field(
        default=0.50,
        ge=0,
        le=1,
        description="Share of documents that must land in a real cluster.",
    )
    min_purity_lift: float = Field(
        default=0.15,
        ge=0,
        le=1,
        description=(
            "How far weighted purity must exceed the majority-label baseline. "
            "Raw purity alone is meaningless on an unbalanced corpus."
        ),
    )
    min_adjusted_rand: float = Field(
        default=0.10,
        ge=-1,
        le=1,
        description="Cluster/label agreement beyond chance (~0 means independent).",
    )
    min_label_recall: float = Field(
        default=0.50,
        ge=0,
        le=1,
        description=(
            "Weakest domain's recall into clusters dominated by that domain -- "
            "catches a run that separates one domain well and loses the other."
        ),
    )


class DomainDecisionWeights(BaseModel):
    """Phase 5 signal weights. Validated to sum to 1.0.

    Four signals only, and deliberately not the superseded
    :class:`ClassificationSignalWeights` set:

    * ``keyword``/``llm`` carry most of the weight -- they are the two
      independent readings of the document's own text.
    * ``cluster`` is corroboration, not evidence: a cluster id means
      nothing until its *other* members are polled. Set it to 0 when
      Phase 4 reports that clustering carries no domain signal, and the
      signal drops out of the decision entirely.
    * ``title`` is a hint. "X v. The State" is suggestive; it is not a
      finding, so it is weighted accordingly.

    There is no ``court`` weight (court predicts forum, not subject) and
    no ``source_folder`` weight (validation ground truth only).
    """

    keyword: float = Field(default=0.40, ge=0, le=1)
    llm: float = Field(default=0.40, ge=0, le=1)
    cluster: float = Field(default=0.15, ge=0, le=1)
    title: float = Field(default=0.05, ge=0, le=1)

    model_config = {"extra": "forbid"}

    @property
    def total(self) -> float:
        return self.keyword + self.llm + self.cluster + self.title

    def as_mapping(self) -> dict[str, float]:
        """The shape :func:`src.classification.domain_decision.decide_domain` wants."""

        return {
            "keyword": self.keyword,
            "llm": self.llm,
            "cluster": self.cluster,
            "title": self.title,
        }

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> "DomainDecisionWeights":
        if abs(self.total - 1.0) > 1e-9:
            raise ValueError(
                f"domain_decision.weights must sum to 1.0, got {self.total!r}"
            )
        return self


class DomainDecisionSettings(BaseModel):
    """Phase 5 (multi-signal broad domain decision) settings.

    Phase 5 weighs the evidence Phase 3 stored and writes one broad
    verdict per document: a domain id from the frozen registry in
    ``config/domains.yaml``, or its catch-all. It makes no model calls, so
    every threshold here fully determines the outcome for a given set of
    signals -- change one and you must start a new ``run_id``.
    """

    # Every score below is a weighted sum of signal scores in [0, 1], not
    # a probability. Do not read them as calibrated likelihoods.
    weights: DomainDecisionWeights = Field(default_factory=DomainDecisionWeights)

    min_domain_score: float = Field(
        default=0.35,
        ge=0,
        le=1,
        description=(
            "Below this, the winning domain is not supported well enough to "
            "assign at all: the document becomes other_uncertain."
        ),
    )
    min_margin: float = Field(
        default=0.10,
        ge=0,
        le=1,
        description=(
            "Minimum lead over the runner-up. Inside this band the signals "
            "disagree, and a human resolves it rather than a rounding rule."
        ),
    )
    auto_accept_threshold: float = Field(
        default=0.80,
        ge=0,
        le=1,
        description="At or above this score a verdict is auto_accepted.",
    )
    review_threshold: float = Field(
        default=0.50,
        ge=0,
        le=1,
        description=(
            "Below this a needs_review verdict is additionally flagged as "
            "uncertain in the audit trail. It never forces a label."
        ),
    )
    off_domain_drop_confidence: float = Field(
        default=0.70,
        ge=0,
        le=1,
        description=(
            "A document is dropped as off-domain only when the LLM said "
            "'neither' at or above this confidence AND no keyword evidence "
            "contradicts it. Otherwise an uncertain document is reviewed, "
            "not discarded."
        ),
    )
    min_coverage: float = Field(
        default=0.50,
        ge=0,
        le=1,
        description=(
            "Minimum weight that must have fired for a verdict to be "
            "auto-accepted. At the default weights this requires at least two "
            "substantive signals (keyword+llm=0.80, keyword+cluster=0.55): "
            "unanimity among two weak signals is not the same evidence as "
            "unanimity among four, and one signal alone should never "
            "auto-accept a legal classification. Below it the verdict still "
            "stands but is routed to review as low_coverage."
        ),
    )
    min_cluster_members: int = Field(
        default=3,
        ge=2,
        description=(
            "Clusters smaller than this are not profiled at all: a two-member "
            "cluster voting on itself is noise dressed up as corroboration."
        ),
    )
    title_min_matches: int = Field(
        default=1,
        gt=0,
        description=(
            "Concept matches needed in the cause title. 1, unlike the body's "
            "threshold -- a title is a few words, so requiring two is "
            "effectively disabling the signal."
        ),
    )
    batch_size: int = Field(
        default=100,
        gt=0,
        description="Decisions per persisted batch; a crash costs at most one batch.",
    )

    @model_validator(mode="after")
    def _thresholds_are_ordered(self) -> "DomainDecisionSettings":
        if self.review_threshold > self.auto_accept_threshold:
            raise ValueError(
                "domain_decision.review_threshold must not exceed "
                f"auto_accept_threshold ({self.review_threshold} > "
                f"{self.auto_accept_threshold})"
            )
        return self


class ReviewSettings(BaseModel):
    """Phase 6 (review routing and decision policy) settings.

    The confidence bands themselves live in ``domain_decision`` -- one set
    of thresholds decides and routes, so they cannot drift apart. What is
    here is everything about the *review process*: where the queue goes,
    and what a rerun is allowed to overwrite.
    """

    schema_file: Path = Path("schemas/review_schema.sql")
    queue_output_dir: Path = Path("var/review_queue")

    protect_reviewed: bool = Field(
        default=True,
        description=(
            "Refuse to reset a document a human has reviewed when the "
            "pipeline is re-run. The audit row is still written, so the "
            "machine verdict is recorded either way -- it just does not "
            "take effect. Turning this off makes review a treadmill."
        ),
    )
    protect_statuses: list[str] = Field(
        default_factory=list,
        description=(
            "Machine classification_status values that a rerun must not "
            "overwrite. Empty by default: machine verdicts are meant to be "
            "recomputable. Set e.g. ['auto_accepted'] to pin a slice of the "
            "corpus while iterating on thresholds."
        ),
    )

    @field_validator("protect_statuses")
    @classmethod
    def _statuses_are_known(cls, value: list[str]) -> list[str]:
        # Imported here rather than at module scope: config must not
        # depend on the extraction package at import time.
        from src.extraction.doc_representation import CLASSIFICATION_STATUSES

        unknown = [v for v in value if v not in CLASSIFICATION_STATUSES]
        if unknown:
            raise ValueError(
                f"review.protect_statuses has unknown status(es) {unknown}; "
                f"expected from {list(CLASSIFICATION_STATUSES)}"
            )
        return value


class EvaluationSettings(BaseModel):
    """Phase 7 (evaluation and dataset freeze) settings.

    The thresholds here decide when *this project* is willing to call a
    domain usable. None of them is a standard, and every raw measurement
    travels with the verdict so a reader can disagree with a threshold and
    still use the report.

    Note what is absent: there is no cluster purity or noise threshold,
    because clustering does not gate corpus acceptance. Those numbers are
    reported as diagnostics only -- see
    :mod:`src.evaluation.readiness`.
    """

    output_dir: Path = Path("var/evaluation")
    freeze_schema_file: Path = Path("schemas/corpus_freeze_schema.sql")
    labels_file: Path | None = Field(
        default=None,
        description=(
            "Optional external gold-label file (CSV/JSON). Left unset, the "
            "validation set is built from the Phase 6 review ledger, which "
            "also records who decided each label."
        ),
    )

    # -- how much human review is needed before a claim is supportable --
    min_validation_documents: int = Field(
        default=100,
        gt=0,
        description="Below this, the evaluation reports not_evaluable rather than a score.",
    )
    min_validation_per_domain: int = Field(
        default=25,
        gt=0,
        description="Gold examples needed per domain before that domain can be scored.",
    )
    comfortable_validation_documents: int = Field(
        default=200,
        gt=0,
        description="Above this the validation set stops being a stated reservation.",
    )

    # -- what counts as good enough ------------------------------------
    min_domain_f1: float = Field(
        default=0.75,
        ge=0,
        le=1,
        description="Per-domain F1 required against human labels. A choice, not a standard.",
    )
    min_accepted_per_domain: int = Field(
        default=200,
        ge=0,
        description="Accepted documents per domain below which a corpus is not worth indexing.",
    )
    min_llm_agreement: float = Field(
        default=0.70,
        ge=0,
        le=1,
        description=(
            "LLM/human agreement below this is a stated reservation, never a "
            "blocker: the classifier is judged on its output, not on one input."
        ),
    )

    # -- the manual audit ----------------------------------------------
    # 100% of needs_review is an obligation and has no setting. These are
    # the spot-check sizes for documents that would otherwise never be
    # seen -- audit effort, not statistical power.
    audit_sample_per_domain: int = Field(default=30, ge=0)
    audit_sample_other_uncertain: int = Field(default=20, ge=0)
    audit_sample_dropped: int = Field(default=10, ge=0)
    audit_seed: int = Field(
        default=42,
        description="Sampling seed; fixed so an audit can be re-run or handed over.",
    )

    @model_validator(mode="after")
    def _validation_sizes_are_ordered(self) -> "EvaluationSettings":
        if self.comfortable_validation_documents < self.min_validation_documents:
            raise ValueError(
                "evaluation.comfortable_validation_documents must not be below "
                f"min_validation_documents ({self.comfortable_validation_documents} "
                f"< {self.min_validation_documents})"
            )
        return self


class LegalBertSettings(BaseModel):
    """Legal-BERT encoder settings.

    This configures an **encoder**, not a classifier. The base model has no
    task head, so it cannot map text to a domain; see
    :mod:`src.classification.legal_bert`. ``classifier_checkpoint`` stays
    None until a head has been fine-tuned on human-reviewed labels, and the
    loader refuses while it is None rather than emitting untrained scores.
    """

    model_name: str = Field(
        default="nlpaueb/legal-bert-base-uncased",
        description="Any HuggingFace encoder; the domain vocabulary is never from here.",
    )
    max_length: int = Field(
        default=512,
        gt=0,
        le=512,
        description=(
            "Hard-capped at BERT's 512 positional embeddings. At this limit "
            "the encoder sees roughly the first 2,000 characters of a case "
            "signature."
        ),
    )
    batch_size: int = Field(default=8, gt=0)
    device: str = Field(
        default="cpu",
        description="'cpu', 'mps' or 'cuda'. cpu by default so a run is portable.",
    )
    classifier_checkpoint: Path | None = Field(
        default=None,
        description=(
            "A fine-tuned Family/Criminal/Other head. Unset: no trained head "
            "exists, and load_domain_classifier() refuses rather than "
            "fabricating predictions."
        ),
    )


class ClassificationSettings(BaseModel):
    """Stage 2 (full-corpus domain classification) settings.

    Classification reads the FROZEN taxonomy only (``taxonomy_file``);
    draft candidates are never a classification target. Changing
    ``llm_model``, the thresholds here, or the taxonomy means starting a
    new ``run_id`` -- rows carry the versions that produced them and are
    never overwritten across runs.
    """

    taxonomy_file: Path = Path("config/domains.yaml")
    schema_file: Path = Path("schemas/classification_schema.sql")

    batch_size: int = Field(
        default=25,
        gt=0,
        description="Documents per persisted batch; a crash costs at most one batch.",
    )
    pilot_size: int = Field(
        default=25,
        gt=0,
        description="Default size of the pilot slice run before a full pass.",
    )

    llm_model: str = "qwen3:14b"
    # 1024, not 512: the pilot showed a 512-token budget truncating the
    # JSON mid-justification, which surfaces as a "could not find a JSON
    # object" failure rather than a bad label. Cheap insurance.
    llm_max_tokens: int = Field(default=1024, gt=0)
    body_chars: int = Field(
        default=3_000,
        gt=0,
        description="Characters of the body preview shown to the classifier.",
    )

    # -- multi-signal scoring thresholds: SUPERSEDED by Phase 5 --------
    # These were the Phase 1 draft. The scoring engine that was "a later
    # phase" now exists and reads `domain_decision` instead; these remain
    # so existing config validates. The single-call LLM classifier below
    # (min_confidence and the review_* flags) does still use its own.
    auto_accept_threshold: float = Field(default=0.80, ge=0, le=1)
    review_threshold: float = Field(default=0.50, ge=0, le=1)
    signals: ClassificationSignalWeights = Field(
        default_factory=ClassificationSignalWeights
    )

    # Review routing for the existing single-call LLM classifier: below
    # this confidence a label is proposed but never treated as final (see
    # src/classification/domain_classifier.py). Distinct from the
    # thresholds above, which belong to the future scoring engine.
    min_confidence: float = Field(default=0.6, ge=0, le=1)
    review_multi_domain: bool = Field(
        default=True,
        description="Route documents with secondary domains to review.",
    )
    review_other_bucket: bool = Field(
        default=True,
        description="Route documents classified as other_uncertain to review.",
    )


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
    # extraction = legacy PDF/book signature stage; caselaw = active stage.
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    caselaw: CaselawSettings = Field(default_factory=CaselawSettings)
    document: DocumentSettings = Field(default_factory=DocumentSettings)
    discovery: DiscoverySettings = Field(default_factory=DiscoverySettings)
    classification: ClassificationSettings = Field(default_factory=ClassificationSettings)
    domain_signals: DomainSignalSettings = Field(default_factory=DomainSignalSettings)
    domain_decision: DomainDecisionSettings = Field(
        default_factory=DomainDecisionSettings
    )
    review: ReviewSettings = Field(default_factory=ReviewSettings)
    legal_bert: LegalBertSettings = Field(default_factory=LegalBertSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    cluster_validation: ClusterValidationSettings = Field(
        default_factory=ClusterValidationSettings
    )
    retry: RetrySettings = Field(default_factory=RetrySettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    metrics: MetricsSettings = Field(default_factory=MetricsSettings)

    model_config = {"extra": "forbid", "frozen": True}
class StorageSettings(BaseModel):
    """Source document storage backend.

    ``backend="local"`` covers *any* mounted filesystem source -- an
    internal drive, an external/USB drive, or a network share -- not just
    a project-local folder. ``source_root`` is where
    :func:`src.ingestion.discovery.discover_local_documents` scans for
    documents; ``local_root`` is kept separate/legacy for anything that
    still wants a project-relative staging root. For a "read the corpus
    directly off an external disk, never copy it into the project" setup,
    set ``backend: local`` and point ``source_root`` at the disk's mount
    path -- Stage 0 (`orchestration/dags/batch_ingest_flow.py`) then never
    downloads/copies bytes for local sources; Stage 1 extraction opens
    ``source_uri`` (that same on-disk path) directly.
    """

    backend: Literal["s3", "local"] = "s3"
    bucket: str | None = None
    local_root: Path = Path("var/local_storage")
    source_root: Path | None = Field(
        default=None,
        description=(
            "Mount path to scan for source documents when backend='local' "
            "(e.g. an external drive's mount point). Required for "
            "src.ingestion.discovery.discover_local_documents; not used "
            "for backend='s3'."
        ),
    )
    checksum_mode: Literal["fast", "full"] = Field(
        default="fast",
        description=(
            "'fast' hashes size+mtime only (no file content read) -- safe "
            "for large/slow external disks. 'full' reads and sha256-hashes "
            "every file's full content; slower but detects in-place edits "
            "a fast checksum would miss."
        ),
    )
    file_extensions: tuple[str, ...] = (".pdf",)

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