"""Phase 7: evaluate the classified corpus and answer the corpus question.

    human review ledger (or a labels file)   -> the validation set
    document_representations                 -> what the pipeline decided
    document_domain_signals                  -> the LLM's own answers
    cluster assignments                      -> clustering diagnostics
        -> classification metrics (P/R/F1, confusion matrix)
        -> audit plan (what a human still has to look at)
        -> readiness verdict
        -> var/evaluation/<run>.json + a readable summary

Read-only with respect to the corpus: this flow measures, it does not
reclassify, and it never writes to ``document_representations``. The one
thing it can write is the report.

The predictions it scores are the **current state** of each document --
after any human corrections -- because that is what would actually be
handed to GraphRAG. Scoring the raw machine output instead would flatter
the pipeline by ignoring the review step that Phase 6 exists to provide.
For that reason a document whose gold label came from a *correction* is
still scored against the machine's verdict, not the human's: otherwise
every corrected document would count as a success, and the classifier
would score 100% by construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.classification.review_store import get_current_reviews
from src.classification.signal_store import get_signals_for_run
from src.classification.taxonomy_registry import (
    OTHER_DOMAIN_ID,
    FrozenTaxonomy,
    load_frozen_taxonomy,
)
from src.clustering.cluster_validation import evaluate_clustering
from src.common.config import get_settings
from src.common.db import DEFAULT_DB_PATH, connection_scope, init_schema
from src.common.logging_utils import get_logger, log_context
from src.evaluation.audit_sampling import build_audit_plan
from src.evaluation.classification_metrics import (
    ClassificationReport,
    evaluate_classification,
    render_confusion_matrix,
)
from src.evaluation.readiness import (
    ReadinessReport,
    ReadinessThresholds,
    assess_readiness,
)
from src.evaluation.validation_set import ValidationSet, load_validation_set

logger = get_logger(__name__)


@dataclass(frozen=True)
class EvaluationResult:
    """Everything Phase 7 measured, and what it concluded."""

    run_id: str
    decision_run_id: str
    validation: ValidationSet
    classification: ClassificationReport
    readiness: ReadinessReport
    audit_plan: object = None
    clustering: object = None
    llm_agreement: float | None = None
    llm_comparable: int = 0
    accepted_by_domain: dict[str, int] = field(default_factory=dict)
    report_paths: dict[str, str] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        return self.readiness.verdict

    @property
    def can_freeze(self) -> bool:
        return self.readiness.can_freeze


def _predictions(
    domains: list[str], db_path: str | Path
) -> dict[str, str | None]:
    """What the pipeline currently says each document is.

    A document withheld as off-domain, or still awaiting review, predicts
    the catch-all rather than nothing: the pipeline *has* made a call
    about it, and that call belongs in the matrix.
    """

    with connection_scope(db_path) as conn:
        rows = conn.execute(
            "SELECT doc_id, primary_domain, classification_status "
            "FROM document_representations"
        ).fetchall()

    predictions: dict[str, str | None] = {}
    for row in rows:
        domain = row["primary_domain"]
        predictions[row["doc_id"]] = domain if domain in domains else OTHER_DOMAIN_ID
    return predictions


def _machine_predictions_for_gold(
    gold: dict[str, str], domains: list[str], db_path: str | Path
) -> dict[str, str | None]:
    """The machine's verdict, recovered where a human has since overwritten it.

    Without this, a corrected document would be scored against the label
    the correction wrote -- guaranteeing agreement and turning the report
    into a tautology.
    """

    predictions = _predictions(domains, db_path=db_path)
    for doc_id, review in get_current_reviews(db_path=db_path).items():
        if doc_id not in gold:
            continue
        machine = review.get("machine_domain")
        if machine is not None:
            predictions[doc_id] = machine if machine in domains else OTHER_DOMAIN_ID
    return predictions


def _llm_agreement(
    gold: dict[str, str], signal_run_id: str, db_path: str | Path
) -> tuple[float | None, int]:
    """How often Phase 3's broad LLM reading matched the human label."""

    if not signal_run_id:
        return None, 0

    rows = {r["doc_id"]: r for r in get_signals_for_run(signal_run_id, db_path=db_path)}
    comparable = agreed = 0
    for doc_id, gold_domain in gold.items():
        row = rows.get(doc_id)
        if row is None or row.get("llm_status") != "ok" or not row.get("llm_domain"):
            continue
        comparable += 1
        agreed += int(row["llm_domain"] == gold_domain)

    return ((agreed / comparable) if comparable else None), comparable


def _clustering_diagnostics(
    gold: dict[str, str], signal_run_id: str, db_path: str | Path
):
    """Score the Phase 3 clustering against the HUMAN labels this time.

    Phase 4 used source-folder names because nothing better existed. Here
    there are real labels, so the same evaluator gets better ground truth
    -- and the result stays a diagnostic either way: nothing in
    :func:`src.evaluation.readiness.assess_readiness` reads it.
    """

    if not signal_run_id or not gold:
        return None

    rows = [
        r for r in get_signals_for_run(signal_run_id, db_path=db_path)
        if r.get("cluster_id") is not None and r["doc_id"] in gold
    ]
    if not rows:
        return None

    doc_ids = [r["doc_id"] for r in rows]
    labels = np.array([int(r["cluster_id"]) for r in rows], dtype=int)
    probabilities = np.array(
        [float(r.get("cluster_confidence") or 0.0) for r in rows]
    )
    return evaluate_clustering(
        run_id=signal_run_id,
        doc_ids=doc_ids,
        labels=labels,
        ground_truth={d: gold[d] for d in doc_ids},
        probabilities=probabilities,
        label_source="human_review",
    )


def _accepted_counts(domains: list[str], db_path: str | Path) -> dict[str, int]:
    from src.evaluation.dataset_freeze import accepted_documents

    counts = {d: 0 for d in domains}
    for document in accepted_documents(domains, db_path=db_path):
        counts[document["primary_domain"]] = counts.get(document["primary_domain"], 0) + 1
    return counts


def run_evaluation(
    run_id: str,
    decision_run_id: str,
    signal_run_id: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    labels_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    taxonomy: FrozenTaxonomy | None = None,
    thresholds: ReadinessThresholds | None = None,
    seed: int | None = None,
) -> EvaluationResult:
    """Measure the classified corpus and decide whether it can be frozen."""

    settings = get_settings()
    config = settings.evaluation
    output_dir = Path(output_dir or config.output_dir)
    thresholds = thresholds or ReadinessThresholds.from_settings(config)
    seed = config.audit_seed if seed is None else seed

    taxonomy = taxonomy or load_frozen_taxonomy(settings.classification.taxonomy_file)
    target_domains = [d.domain_id for d in taxonomy.domains if not d.is_other]
    # The catch-all is scored as a class: see classification_metrics.
    scored_domains = [*target_domains, OTHER_DOMAIN_ID]

    init_schema(db_path=db_path, schema_file=settings.review.schema_file)
    init_schema(db_path=db_path, schema_file=config.freeze_schema_file)

    with log_context(batch_id=run_id, phase="evaluation"):
        validation = load_validation_set(
            db_path=db_path, labels_file=labels_file, domains=scored_domains
        )
        logger.info(
            "Validation set: %d label(s) from %s across %d reviewer(s) -- %s",
            len(validation), validation.source, validation.reviewer_count,
            validation.by_domain or "empty",
        )

        predicted = _machine_predictions_for_gold(
            validation.labels, scored_domains, db_path=db_path
        )
        classification = evaluate_classification(
            gold=validation.labels, predicted=predicted, domains=scored_domains
        )

        audit_plan = build_audit_plan(
            decision_run_id,
            domains=target_domains,
            db_path=db_path,
            sample_per_domain=config.audit_sample_per_domain,
            sample_other_uncertain=config.audit_sample_other_uncertain,
            sample_dropped=config.audit_sample_dropped,
            seed=seed,
        )

        llm_agreement, llm_comparable = _llm_agreement(
            validation.labels, signal_run_id or "", db_path=db_path
        )
        clustering = _clustering_diagnostics(
            validation.labels, signal_run_id or "", db_path=db_path
        )
        accepted_by_domain = _accepted_counts(target_domains, db_path=db_path)

        readiness = assess_readiness(
            validation=validation,
            classification=classification,
            accepted_by_domain=accepted_by_domain,
            domains=target_domains,
            thresholds=thresholds,
            audit_plan=audit_plan,
            clustering_report=clustering,
            llm_agreement=llm_agreement,
        )

        result = EvaluationResult(
            run_id=run_id,
            decision_run_id=decision_run_id,
            validation=validation,
            classification=classification,
            readiness=readiness,
            audit_plan=audit_plan,
            clustering=clustering,
            llm_agreement=llm_agreement,
            llm_comparable=llm_comparable,
            accepted_by_domain=accepted_by_domain,
        )
        paths = write_evaluation_report(result, output_dir)

    return EvaluationResult(
        run_id=result.run_id,
        decision_run_id=result.decision_run_id,
        validation=result.validation,
        classification=result.classification,
        readiness=result.readiness,
        audit_plan=result.audit_plan,
        clustering=result.clustering,
        llm_agreement=result.llm_agreement,
        llm_comparable=result.llm_comparable,
        accepted_by_domain=result.accepted_by_domain,
        report_paths=paths,
    )


def render_evaluation_summary(result: EvaluationResult) -> str:
    """The report a person reads, in the order they need it."""

    readiness = result.readiness
    lines = [
        "=" * 72,
        f"CORPUS EVALUATION -- run {result.run_id} over decision run "
        f"{result.decision_run_id}",
        "=" * 72,
        "",
        readiness.question,
        "",
        f"VERDICT: {readiness.verdict.upper()}",
        readiness.answer,
        "",
    ]

    if readiness.blockers:
        lines.append("Blockers:")
        lines += [f"  - {b}" for b in readiness.blockers]
        lines.append("")
    if readiness.reservations:
        lines.append("Reservations:")
        lines += [f"  - {r}" for r in readiness.reservations]
        lines.append("")

    validation = result.validation
    lines += [
        "-" * 72,
        "VALIDATION SET",
        f"  {len(validation)} human label(s) from {validation.source}, "
        f"{validation.reviewer_count} reviewer(s)",
        f"  by domain: {validation.by_domain or '(none)'}",
        f"  undecided reviews (no label): {len(validation.unresolved)}",
        "",
        "-" * 72,
        "CLASSIFICATION",
    ]

    report = result.classification
    if report.n_documents:
        lines.append(
            f"  accuracy {_pct(report.accuracy)}  macro-F1 {_num(report.macro_f1)}  "
            f"micro-F1 {_num(report.micro_f1)}  weighted-F1 {_num(report.weighted_f1)}"
        )
        lines.append("")
        header = f"  {'domain':<18}{'P':>8}{'R':>8}{'F1':>8}{'support':>10}{'predicted':>11}"
        lines += [header, "  " + "-" * (len(header) - 2)]
        for domain in report.domains:
            metrics = report.per_domain[domain]
            lines.append(
                f"  {domain:<18}{_num(metrics.precision):>8}{_num(metrics.recall):>8}"
                f"{_num(metrics.f1):>8}{metrics.support:>10}{metrics.predicted:>11}"
            )
        lines += ["", "  Confusion matrix (gold down, predicted across):", ""]
        lines += [
            "    " + line for line in render_confusion_matrix(report).splitlines()
        ]
    else:
        lines.append("  not measured -- no validation set")
    lines.append("")

    lines += ["-" * 72, "LLM (Phase 3 broad assessment)"]
    if result.llm_agreement is not None:
        lines.append(
            f"  agreement with human labels: {_pct(result.llm_agreement)} "
            f"over {result.llm_comparable} comparable document(s)"
        )
    else:
        lines.append("  not measured -- no signal run given, or no successful assessments")
    lines.append("")

    lines += ["-" * 72, "HDBSCAN (diagnostic only -- does not gate acceptance)"]
    clustering = result.clustering
    if clustering is not None:
        sizes = sorted(
            (c.size for c in clustering.per_cluster if not c.is_noise), reverse=True
        )
        lines += [
            f"  clusters: {clustering.n_clusters}   noise: "
            f"{clustering.noise_documents} ({_pct(clustering.noise_share)})   "
            f"coverage: {_pct(clustering.coverage)}",
            f"  cluster sizes: {sizes or '(none)'}",
            f"  weighted purity {_num(clustering.weighted_purity)} "
            f"(lift {_num(clustering.purity_lift)} over a majority baseline)",
            f"  ARI {_num(clustering.adjusted_rand_index)}   "
            f"NMI {_num(clustering.normalized_mutual_info)}",
            f"  per-label recall: "
            + ", ".join(f"{k} {_pct(v)}" for k, v in clustering.per_label_recall.items()),
            f"  Phase 4 verdict on the cluster signal: {clustering.verdict}",
        ]
    else:
        lines.append("  not measured -- no clustered signal run given")
    lines.append("")

    lines += ["-" * 72, "MANUAL AUDIT"]
    plan = result.audit_plan
    if plan is not None:
        for name, stratum in plan.strata.items():
            kind = "100% census" if stratum.is_census else f"sample of {stratum.required}"
            lines.append(
                f"  {name:<32} population {stratum.population:>6}   {kind:<18}"
                f"reviewed {stratum.already_reviewed:>5}   outstanding "
                f"{stratum.outstanding:>5}"
            )
        lines.append(
            f"  review backlog cleared: "
            f"{'yes' if plan.review_backlog_cleared else 'NO'}   "
            f"(sample seed {plan.seed}, reproducible)"
        )
    lines.append("")

    lines += ["-" * 72, "ACCEPTED CORPUS"]
    for domain, count in sorted(result.accepted_by_domain.items()):
        lines.append(f"  {domain:<18}{count:>8} document(s)")
    lines += [
        "",
        (
            "Freeze permitted: yes -- run dataset_freeze.freeze_corpus()"
            if readiness.can_freeze
            else "Freeze permitted: NO -- the freeze is blocked until the above is resolved"
        ),
        "=" * 72,
    ]
    return "\n".join(lines)


def write_evaluation_report(
    result: EvaluationResult, output_dir: str | Path
) -> dict[str, str]:
    """Write the machine-readable report and the human-readable summary."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"evaluation_{result.run_id}.json"
    text_path = output_dir / f"evaluation_{result.run_id}.txt"

    payload = {
        "run_id": result.run_id,
        "decision_run_id": result.decision_run_id,
        "readiness": result.readiness.as_dict(),
        "classification": result.classification.as_dict(),
        "validation": result.validation.as_dict(),
        "llm": {
            "agreement_with_human_labels": result.llm_agreement,
            "comparable_documents": result.llm_comparable,
        },
        "audit": result.audit_plan.as_dict() if result.audit_plan else {},
        "accepted_by_domain": result.accepted_by_domain,
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    text_path.write_text(render_evaluation_summary(result), encoding="utf-8")

    logger.info("Evaluation report written to %s", text_path)
    return {"json": str(json_path), "text": str(text_path)}


def _num(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def _pct(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "n/a"
