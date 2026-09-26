"""Phase 5: turn the Phase 3 evidence into one broad domain verdict per case.

    document_domain_signals (one signal run)
        -> cluster profiles      (leave-one-out, corpus-level)
        -> weighted decision     (deterministic, per case)
        -> document_representations   (current state: Phase 1's fields)
        -> document_classifications   (audit row: scores + every signal)

The whole phase is **deterministic**: the only stochastic step in the
chain was Phase 3's LLM reading, which is already recorded. Re-running
this flow over the same ``signal_run_id`` produces identical verdicts, so
a disagreement between two runs always means a threshold or weight
changed -- never model drift.

Two writes per document, by design:

* ``document_representations`` carries the **current answer**. Phase 5
  writes only the five classification fields Phase 1 defined
  (``classification_status``, ``primary_domain``, ``secondary_domain``,
  ``domain_confidence``, ``drop_reason``) and touches nothing else. No new
  table was created for it.
* ``document_classifications`` carries the **audit trail** -- the scores,
  each signal's contribution and the justification -- append-only and
  keyed by ``(run_id, doc_id)``, with ``signal_run_id`` naming the
  evidence the verdict rests on. Re-deciding later leaves the old row
  intact.

Reruns are conservative (Phase 6). A document a human has reviewed is
protected: the audit row is still written -- so what the pipeline *would*
have said is always on record -- but the current state is left as the
reviewer set it. A write that would change nothing is skipped outright
rather than re-stamping ``updated_at``. Nothing is ever deleted; an
off-domain document is withheld by status, and that is reversible.

What this flow deliberately does not read: source-folder labels. They are
Phase 4's validation ground truth. Using them here would make the
evaluation circular and the classifier useless on an unlabelled corpus,
so nothing in this module imports :mod:`src.clustering.cluster_validation`.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from src.classification.classification_store import (
    classification_stats,
    get_classified_doc_ids,
    get_latest_signal_run_ids,
    persist_classifications,
)
from src.classification.domain_classifier import ClassificationResult
from src.classification.domain_decision import (
    DECISION_VERSION,
    STATUS_AUTO_ACCEPTED,
    STATUS_DROPPED_OFF_DOMAIN,
    STATUS_NEEDS_REVIEW,
    DomainDecision,
    build_cluster_profiles,
    decide_domain,
)
from src.classification import review_policy as policy
from src.classification.keyword_signals import compile_profiles
from src.classification.review_store import get_reviewed_doc_ids
from src.classification.signal_store import get_signals_for_run
from src.classification.taxonomy_registry import FrozenTaxonomy, load_frozen_taxonomy
from src.common.checkpoint import CheckpointManager
from src.common.config import get_settings
from src.common.db import DEFAULT_DB_PATH, init_schema
from src.common.logging_utils import current_run_id, get_logger, log_context
from src.common.metrics import MetricsStore
from src.extraction.representation_store import (
    WRITE_INELIGIBLE,
    WRITE_PROTECTED,
    WRITE_STALE,
    get_classification_states,
    list_representations,
    update_classification_state,
)

# Re-exported: both refusals are decided by the store, inside the same
# transaction as the write. The audit row is still written either way, so a
# skip is visible rather than silent.

logger = get_logger(__name__)


@dataclass(frozen=True)
class DomainClassificationResult:
    """What one Phase 5 run did, and how confident it was."""

    run_id: str
    signal_run_id: str
    decision_version: str = DECISION_VERSION
    signals_available: int = 0
    decided: int = 0
    skipped_already_done: int = 0
    auto_accepted: int = 0
    needs_review: int = 0
    dropped_off_domain: int = 0
    by_domain: dict[str, int] = field(default_factory=dict)
    by_review_reason: dict[str, int] = field(default_factory=dict)
    clusters_profiled: int = 0
    cluster_signal_enabled: bool = True
    # Current-state write outcomes: updated / unchanged / protected /
    # missing. A rerun over settled work should be mostly "unchanged".
    write_outcomes: dict[str, int] = field(default_factory=dict)
    protected_by_review: int = 0
    ineligible_for_decision: int = 0
    skipped_stale_evidence: int = 0
    decisions: list[DomainDecision] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def auto_accept_rate(self) -> float:
        return self.auto_accepted / self.decided if self.decided else 0.0

    @property
    def review_rate(self) -> float:
        return self.needs_review / self.decided if self.decided else 0.0


def cluster_signal_is_approved(
    signal_run_id: str,
    output_dir: str | Path | None = None,
    signal_rows: list[dict] | None = None,
) -> tuple[bool, str]:
    """Has Phase 4 approved cluster membership as a domain signal, and does
    that approval still apply to the current configuration?

    Returns ``(approved, reason)``. **Fails closed** on every negative
    answer. Trusting HDBSCAN because nobody checked it is the one outcome
    Phase 4 exists to prevent, and it is invisible once the cluster
    contribution is blended into a weighted score.

    Four conditions, all required:

    1. **the report exists** -- Phase 4 ran for this signal run;
    2. **it is readable** -- a truncated file is not an approval;
    3. **the verdict approves** -- ``cluster_is_a_useful_signal``. Phase 4
       sets this False for ``weak``, ``not_useful`` and ``not_evaluable``
       alike, so "marginal" does not count as yes;
    4. **it is still compatible** with the evidence being decided, checked
       against ``signal_rows``:

       * the **embedding model** recorded on the signal run must match the
         configured one. Different vectors mean different clusters, so a
         validation computed on other embeddings says nothing about these.
       * the **document count** must match the report's ``n_documents``.
         Validating 200 documents does not approve a 14,000-document
         clustering, and a corpus that has grown since is a different
         dataset.

    Condition 4 uses state Phase 3 already persists, so Phase 4 needs no
    change and no parallel validation system is introduced.
    """

    settings = get_settings()
    directory = Path(output_dir or settings.cluster_validation.output_dir)
    report_path = directory / f"{signal_run_id}.json"

    if not report_path.exists():
        return False, (
            f"no Phase 4 validation report at {report_path} -- run "
            f"`validate-clusters --run-id {signal_run_id}` first"
        )
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"Phase 4 report at {report_path} is unreadable: {exc}"

    if payload.get("cluster_is_a_useful_signal") is not True:
        return False, (
            f"Phase 4 verdict {payload.get('verdict')!r} does not approve "
            "cluster membership as a domain signal"
        )

    if signal_rows is not None:
        compatible, why = _validation_is_compatible(payload, signal_rows, settings)
        if not compatible:
            return False, f"Phase 4 validation is no longer applicable: {why}"

    return True, f"Phase 4 verdict {payload.get('verdict')!r} approves the signal"


def _validation_is_compatible(
    payload: dict, signal_rows: list[dict], settings
) -> tuple[bool, str]:
    """Does a passing validation still describe the clustering being used?"""

    configured_model = settings.discovery.embedding_model_name
    recorded = {
        row.get("embedding_model") for row in signal_rows if row.get("embedding_model")
    }
    if recorded and recorded != {configured_model}:
        return False, (
            f"it was computed on embedding model(s) {sorted(recorded)} but "
            f"{configured_model!r} is configured now"
        )

    validated_count = payload.get("n_documents")
    if isinstance(validated_count, int) and validated_count != len(signal_rows):
        return False, (
            f"it validated {validated_count} document(s) but this run has "
            f"{len(signal_rows)}"
        )

    return True, "embedding model and document count match"


def _to_classification_result(decision: DomainDecision) -> ClassificationResult:
    """Render a decision as the audit row the classification store persists.

    ``justification`` carries the human-readable trace and ``review_reason``
    the machine-readable one; the full per-signal breakdown rides along as
    JSON so a verdict can be re-examined without recomputing it.
    """

    return ClassificationResult(
        doc_id=decision.doc_id,
        primary_domain=decision.primary_domain,
        secondary_domains=[],  # broad decision only: no secondary domain
        confidence=decision.confidence,
        justification=(
            decision.justification()
            + " evidence="
            + json.dumps(decision.evidence(), ensure_ascii=False, sort_keys=True)
        ),
        status=decision.status,
        review_reason=",".join(decision.review_reasons) or None,
        cluster_id=decision.cluster_id,
    )


def _batches(items: list, size: int):
    for start in range(0, len(items), max(size, 1)):
        yield start // max(size, 1), items[start : start + max(size, 1)]


def run_domain_classification(
    run_id: str,
    signal_run_id: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    limit: int | None = None,
    batch_size: int | None = None,
    checkpoint_dir: str | Path | None = None,
    metrics_db_path: str | Path | None = None,
    taxonomy: FrozenTaxonomy | None = None,
    cluster_signal_enabled: bool | None = None,
    write_current_state: bool = True,
    protect_reviewed: bool | None = None,
    protect_statuses: tuple[str, ...] | None = None,
) -> DomainClassificationResult:
    """Decide a broad domain for every case with Phase 3 evidence.

    ``run_id`` is the decision version boundary and ``signal_run_id`` names
    the evidence: reuse ``run_id`` to resume a partial pass, choose a new
    one whenever weights or thresholds change so earlier verdicts stay
    intact and comparable.

    ``protect_reviewed`` (default from configuration) keeps documents a
    human has decided from being reset; ``protect_statuses`` additionally
    freezes documents already in the given machine states. Protection
    applies to the current state only -- the audit row is written either
    way.

    Set ``cluster_signal_enabled=False`` when Phase 4 reported that
    clustering carries no domain signal -- the cluster weight is forced to
    zero and the remaining signals are *not* renormalized, so a document
    that relied on cluster corroboration correctly loses confidence and
    falls to review rather than being propped up. Pass
    ``write_current_state=False`` for a dry run that records the audit
    rows without changing any document's current state.
    """

    settings = get_settings()
    config = settings.domain_decision
    batch_size = batch_size or config.batch_size
    checkpoint_dir = checkpoint_dir or settings.pipeline.checkpoint_dir
    metrics_db_path = metrics_db_path or settings.metrics.db_path

    taxonomy = taxonomy or load_frozen_taxonomy(settings.classification.taxonomy_file)
    compiled_profiles = compile_profiles(settings.domain_signals.profiles, taxonomy)

    # CI-6: resolved after the evidence loads, so the approval can be
    # checked for compatibility with it. See _resolve_cluster_signal below.
    requested_cluster_signal = cluster_signal_enabled

    review_config = settings.review
    if protect_reviewed is None:
        protect_reviewed = review_config.protect_reviewed
    if protect_statuses is None:
        protect_statuses = tuple(review_config.protect_statuses)
    # The invariant is unioned in rather than left to configuration: an
    # operator may choose to freeze extra statuses, but never to un-freeze
    # a Phase 2 structural drop.
    protect_statuses = tuple(
        set(protect_statuses) | policy.PHASE_5_INELIGIBLE_STATUSES
    )

    init_schema(db_path=db_path, schema_file=settings.caselaw.representation_schema_file)
    init_schema(db_path=db_path, schema_file=settings.classification.schema_file)
    init_schema(db_path=db_path, schema_file=review_config.schema_file)

    metrics = MetricsStore(metrics_db_path)
    metrics.init_schema()
    metrics_run_id = current_run_id()

    checkpoint = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        run_key=f"decide_{run_id}",
        phases=["profile", "decide"],
    )

    with log_context(batch_id=run_id, phase="domain_classification"):
        signal_rows = sorted(
            get_signals_for_run(signal_run_id, db_path=db_path),
            key=lambda r: r["doc_id"],
        )
        if not signal_rows:
            logger.warning(
                "No Phase 3 evidence for signal run %s -- nothing to decide. "
                "Run the domain-signal flow first.",
                signal_run_id,
            )
            return DomainClassificationResult(
                run_id=run_id,
                signal_run_id=signal_run_id,
                cluster_signal_enabled=cluster_signal_enabled,
                stats=classification_stats(run_id, db_path=db_path),
            )

        # Titles come from the representation, not the signal row: Phase 3
        # deliberately excluded court/date/citation from its text, and the
        # cause title is the one metadata field Phase 5 is allowed to read.
        titles = {
            rep.doc_id: rep.title
            for rep in list_representations(db_path=db_path, exclude_dropped=False)
        }

        # CI-6: the cluster signal is used only on Phase 4's approval, and
        # only while that approval still applies to this evidence. None
        # (the default) consults the stored verdict and fails closed; an
        # explicit True/False overrides for a deliberate experiment.
        if requested_cluster_signal is None:
            cluster_signal_enabled, cluster_reason = cluster_signal_is_approved(
                signal_run_id, signal_rows=signal_rows
            )
        else:
            cluster_signal_enabled = requested_cluster_signal
            cluster_reason = (
                f"caller forced cluster_signal_enabled={cluster_signal_enabled}"
            )

        weights = config.weights.as_mapping()
        if not cluster_signal_enabled:
            # Neutralised, not renormalised: the remaining signals keep
            # their own weights and the document's coverage drops
            # accordingly, so a verdict that had leaned on corroboration
            # correctly loses confidence instead of being propped up.
            weights = {**weights, "cluster": 0.0}
        logger.info("Cluster signal %s -- %s",
                    "ENABLED" if cluster_signal_enabled else "disabled", cluster_reason)

        # When each document's evidence was gathered, for the staleness check.
        signal_created_at = {
            row["doc_id"]: row.get("created_at") for row in signal_rows
        }

        domains = [d.domain_id for d in taxonomy.domains if not d.is_other]

        # Profiles are built from the WHOLE signal run, not the pending
        # slice: a resumed run must see the same cluster composition as
        # the first pass, or its verdicts would not be reproducible.
        cluster_profiles: dict[int, Counter] = (
            build_cluster_profiles(
                signal_rows, domains, min_cluster_members=config.min_cluster_members
            )
            if cluster_signal_enabled
            else {}
        )
        logger.info(
            "Decision run %s over signal run %s: %d document(s) with evidence, "
            "%d cluster(s) profiled (cluster signal %s)",
            run_id,
            signal_run_id,
            len(signal_rows),
            len(cluster_profiles),
            "enabled" if cluster_signal_enabled else "DISABLED by caller",
        )
        checkpoint.complete_phase(
            "profile",
            state={"signals": len(signal_rows), "clusters": len(cluster_profiles)},
        )

        already_done = get_classified_doc_ids(run_id, db_path=db_path)
        pending = [r for r in signal_rows if r["doc_id"] not in already_done]
        if limit is not None:
            pending = pending[:limit]

        # CI-4: the CURRENT state of every document -- status AND when it
        # last moved -- read now rather than inferred from the (possibly
        # older) signal run. Dropped documents are included on purpose: a
        # dropped document's status is exactly what has to be seen here.
        # Loaded for the run summary only. The authority for both refusals
        # is the row read inside update_classification_state's transaction.
        states = get_classification_states(db_path=db_path)
        current_status = {doc_id: status for doc_id, (status, _) in states.items()}
        # Which evidence produced each document's newest decision, so this
        # run can tell its own earlier write from someone else's newer one.
        state_signal_run = get_latest_signal_run_ids(db_path=db_path)

        reviewed = get_reviewed_doc_ids(db_path=db_path) if protect_reviewed else set()
        if reviewed:
            logger.info(
                "%d document(s) carry a human review decision and are protected "
                "from this rerun", len(reviewed),
            )

        decisions: list[DomainDecision] = []
        write_outcomes: dict[str, int] = {}
        by_domain: Counter = Counter()
        by_review_reason: Counter = Counter()
        status_counts: Counter = Counter()

        with metrics.record_phase(
            run_id=metrics_run_id, phase="decide", batch_id=run_id
        ) as metric:
            for index, batch in _batches(pending, batch_size):
                batch_id = f"{run_id}_b{index:04d}"
                batch_decisions = [
                    decide_domain(
                        row,
                        taxonomy,
                        weights,
                        cluster_profiles=cluster_profiles,
                        compiled_profiles=compiled_profiles,
                        title=titles.get(row["doc_id"]),
                        min_domain_score=config.min_domain_score,
                        min_margin=config.min_margin,
                        auto_accept_threshold=config.auto_accept_threshold,
                        review_threshold=config.review_threshold,
                        off_domain_drop_confidence=config.off_domain_drop_confidence,
                        min_coverage=config.min_coverage,
                        title_min_matches=config.title_min_matches,
                        signal_run_id=signal_run_id,
                    )
                    for row in batch
                ]

                # Audit row first, then current state: if the run dies in
                # between, a document's live status is still backed by a
                # persisted explanation rather than the reverse.
                persist_classifications(
                    run_id,
                    [_to_classification_result(d) for d in batch_decisions],
                    taxonomy_version=taxonomy.version,
                    classifier_version=DECISION_VERSION,
                    model_name=None,  # no model call in Phase 5
                    batch_id=batch_id,
                    signal_run_id=signal_run_id,
                    db_path=db_path,
                )

                if write_current_state:
                    for decision in batch_decisions:
                        if policy.is_protected(
                            decision.doc_id,
                            current_status=None,
                            reviewed_doc_ids=reviewed,
                            protect_reviewed=protect_reviewed,
                        ):
                            # A person already decided this one. The audit
                            # row above still records what the pipeline
                            # would have said.
                            write_outcomes[WRITE_PROTECTED] = (
                                write_outcomes.get(WRITE_PROTECTED, 0) + 1
                            )
                            continue

                        # Eligibility and staleness are decided inside the
                        # store's own transaction, against the row as it is
                        # at write time -- not against the snapshot taken at
                        # the top of this run. A document dropped or
                        # re-decided while this run was working is therefore
                        # seen, and the write is refused rather than racing.
                        outcome = update_classification_state(
                            decision.doc_id,
                            classification_status=decision.status,
                            primary_domain=(
                                None if decision.is_uncertain else decision.primary_domain
                            ),
                            secondary_domain=None,
                            domain_confidence=decision.confidence,
                            drop_reason=decision.drop_reason,
                            protect_statuses=protect_statuses,
                            ineligible_statuses=tuple(
                                policy.PHASE_5_INELIGIBLE_STATUSES
                            ),
                            evidence_created_at=signal_created_at.get(decision.doc_id),
                            evidence_run_id=signal_run_id,
                            state_evidence_run_id=state_signal_run.get(decision.doc_id),
                            db_path=db_path,
                        )
                        write_outcomes[outcome] = write_outcomes.get(outcome, 0) + 1

                for decision in batch_decisions:
                    by_domain[decision.primary_domain] += 1
                    status_counts[decision.status] += 1
                    for reason in decision.review_reasons:
                        by_review_reason[reason] += 1
                decisions.extend(batch_decisions)

                checkpoint.set_state(
                    "decide", {"last_batch_index": index, "decided": len(decisions)}
                )
                logger.info(
                    "Batch %s: %d decision(s) (%d/%d this run)",
                    batch_id, len(batch_decisions), len(decisions), len(pending),
                )

            metric.processed = len(decisions)

        checkpoint.complete_phase("decide", state={"decided": len(decisions)})
        stats = classification_stats(run_id, db_path=db_path)

        result = DomainClassificationResult(
            run_id=run_id,
            signal_run_id=signal_run_id,
            signals_available=len(signal_rows),
            decided=len(decisions),
            skipped_already_done=len(already_done),
            auto_accepted=status_counts[STATUS_AUTO_ACCEPTED],
            needs_review=status_counts[STATUS_NEEDS_REVIEW],
            dropped_off_domain=status_counts[STATUS_DROPPED_OFF_DOMAIN],
            by_domain=dict(by_domain),
            by_review_reason=dict(by_review_reason),
            clusters_profiled=len(cluster_profiles),
            cluster_signal_enabled=cluster_signal_enabled,
            write_outcomes=write_outcomes,
            protected_by_review=write_outcomes.get(WRITE_PROTECTED, 0),
            ineligible_for_decision=write_outcomes.get(WRITE_INELIGIBLE, 0),
            skipped_stale_evidence=write_outcomes.get(WRITE_STALE, 0),
            decisions=decisions,
            stats=stats,
        )
        logger.info(
            "Decision run %s complete: %d decided (%.1f%% auto-accepted, "
            "%.1f%% review, %d dropped off-domain), %d already done. "
            "writes: %s",
            run_id,
            result.decided,
            result.auto_accept_rate * 100,
            result.review_rate * 100,
            result.dropped_off_domain,
            result.skipped_already_done,
            write_outcomes or "none",
        )

    return result
