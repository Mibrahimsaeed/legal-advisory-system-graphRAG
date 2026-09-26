#!/usr/bin/env python3
"""Command-line entry point for running the pipeline against a real corpus.

This is the file you actually run. It ties together the pieces that were
previously only importable as functions:

    cases             -> case_ingest_flow.run_case_ingest          (P1 + P2)
    signals           -> domain_signal_flow.run_domain_signals     (P3)
    validate-clusters -> cluster_validation_flow.run_cluster_validation (P4)
    decide            -> domain_classification_flow.run_domain_classification
                                                                   (P5 + P6)
    review-queue      -> review_store.export_review_queue          (P6)
    apply-reviews     -> review_store.apply_review_decisions        (P6)
    evaluate          -> evaluation_flow.run_evaluation             (P7)
    freeze            -> evaluation_flow.run_evaluation
                         + dataset_freeze.freeze_corpus             (P7)

Every command is a thin wrapper: argument parsing, one call into the
phase, and a printed summary. No phase logic lives in this file.

and, for the legacy PDF/book corpus only:

    discover  -> src.ingestion.discovery.discover_local_documents
    ingest    -> orchestration.dags.batch_ingest_flow.run_stage0
                 + orchestration.dags.feature_extraction_flow.run_stage1
                 (looped batch-by-batch until the manifest has nothing
                 left in 'pending'/'claimed')

`classify` is RETIRED -- it ran the superseded single-call LLM classifier,
which never wrote a document's current classification state. Use `decide`.

Usage (case law -- the active corpus)::

    # 1. scan caselaw.corpus_root for case folders (case.html +
    #    metadata.json), filter structural noise, store one row each.
    python scripts/run_pipeline.py cases

    # 2. gather Phase 3 evidence: embeddings, clusters, keywords, LLM.
    python scripts/run_pipeline.py signals --run-id sig1

    # 3. ask whether the clustering is a useful domain signal at all.
    python scripts/run_pipeline.py validate-clusters --run-id sig1

    # 4. weigh the evidence into family_law / criminal_law /
    #    other_uncertain and route each verdict.
    python scripts/run_pipeline.py decide --run-id dec1 --signal-run-id sig1

    # 5. hand the uncertain cases to a human, then record their answers.
    python scripts/run_pipeline.py review-queue --run-id dec1
    python scripts/run_pipeline.py apply-reviews \
        --csv var/review_queue/review_queue_dec1.csv --run-id dec1

    # 6. measure against the reviewed set; only then freeze.
    python scripts/run_pipeline.py evaluate --run-id eval1 \
        --decision-run-id dec1 --signal-run-id sig1
    python scripts/run_pipeline.py freeze --freeze-id corpus_v1 \
        --decision-run-id dec1 --signal-run-id sig1

Legacy taxonomy drafting (`domains`, `freeze-taxonomy`) is kept for
reference; config/domains.yaml is already frozen at
caselaw_family_criminal@v1.

Usage (legacy PDF/book corpus -- requires
``discovery.corpus_source: signatures`` in config)::

    python scripts/run_pipeline.py discover
    python scripts/run_pipeline.py ingest
    python scripts/run_pipeline.py domains

    # or discover -> ingest -> domains in sequence:
    python scripts/run_pipeline.py all

Every step reads its configuration from ``config/base.yaml`` (+
``config/{env}.yaml``, + ``APP__...`` env var overrides) via
``src.common.config.get_settings()`` -- there are no required CLI flags
for paths/settings; set ``storage.source_root`` in config first (see
``config/base.yaml``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Running a script *by path* puts scripts/ on sys.path, not the repo root,
# so `import src.…` fails even though the package is installed editable
# (that install exposes <repo>/src, i.e. `common.…`, not `src.common.…`).
# Without this, every command below is unreachable via the documented
# `python scripts/run_pipeline.py …` invocation.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.common.config import get_settings  # noqa: E402
from src.common.logging_utils import get_logger

logger = get_logger(__name__)

# If this many consecutive batches claim documents but fail to pull every
# single one of them, stop and say so instead of grinding through the rest
# of a (possibly 14k-document) manifest marking everything pull_failed --
# that pattern almost always means the external disk got unmounted, lost
# power, or a permissions/path problem hit mid-run, not "these particular
# files are bad".
CONSECUTIVE_FULL_FAILURE_LIMIT = 3


def cmd_discover(args: argparse.Namespace) -> None:
    from src.ingestion.discovery import discover_local_documents

    settings = get_settings()
    if settings.storage.backend != "local":
        print(
            f"storage.backend='{settings.storage.backend}', not 'local' -- "
            "discovery is only for locally-mounted sources (see config/base.yaml)."
        )
        sys.exit(1)

    root = settings.storage.source_root
    if root is None:
        print(
            "storage.source_root is not set. Edit config/base.yaml (or "
            "config/<env>.yaml) and set storage.source_root to your "
            "external disk's mount path, e.g.:\n\n"
            "  storage:\n"
            "    backend: local\n"
            "    source_root: /Volumes/LegalDocs   # macOS example\n"
            "    # source_root: /mnt/legal_docs    # Linux example\n"
            "    # source_root: D:\\LegalDocs        # Windows example\n"
        )
        sys.exit(1)

    print(f"Scanning {root} for {settings.storage.file_extensions} files ...")
    n = discover_local_documents(
        root=root,
        db_path=settings.database.path,
        checksum_mode=settings.storage.checksum_mode,
        extensions=settings.storage.file_extensions,
    )
    print(f"Registered {n} new document(s) into the manifest.")


def cmd_ingest(args: argparse.Namespace) -> None:
    from orchestration.dags.batch_ingest_flow import run_stage0
    from orchestration.dags.feature_extraction_flow import run_stage1

    settings = get_settings()
    batch_size = args.batch_size or settings.pipeline.batch_size
    # Passed explicitly rather than relying on run_stage0()/run_stage1()'s
    # own (hardcoded, settings-unaware) defaults for db_path/scratch_root --
    # otherwise a custom database.path or scratch.root in config would be
    # silently ignored by this CLI.
    db_path = settings.database.path
    scratch_root = settings.scratch.root

    total_batches = 0
    total_succeeded = 0
    total_failed = 0
    consecutive_full_failures = 0

    while True:
        stage0 = run_stage0(batch_size=batch_size, db_path=db_path, scratch_root=scratch_root)

        if stage0.claimed_count == 0:
            # Nothing left pending in the manifest -- genuinely done.
            break

        if not stage0.entries:
            # Claimed a batch, but every single pull in it failed.
            consecutive_full_failures += 1
            print(
                f"Batch {stage0.batch_id}: claimed {stage0.claimed_count} "
                f"document(s), 0 pulled successfully (attempt "
                f"{consecutive_full_failures}/{CONSECUTIVE_FULL_FAILURE_LIMIT})"
            )
            if consecutive_full_failures >= CONSECUTIVE_FULL_FAILURE_LIMIT:
                print(
                    "\nStopping: the last "
                    f"{CONSECUTIVE_FULL_FAILURE_LIMIT} batches each failed to "
                    "pull every document they claimed. This usually means "
                    "the external disk got unmounted, lost power, or "
                    "storage.source_root no longer resolves correctly -- "
                    "check the disk and re-run `ingest` once it's fixed; "
                    "already-processed documents will not be redone."
                )
                sys.exit(1)
            continue

        consecutive_full_failures = 0
        stage1 = run_stage1(stage0.batch_id, db_path=db_path, scratch_root=scratch_root)
        total_batches += 1
        total_succeeded += len(stage1.succeeded_doc_ids)
        total_failed += len(stage1.failed_doc_ids)

        print(
            f"Batch {stage0.batch_id}: {len(stage1.succeeded_doc_ids)} "
            f"succeeded, {len(stage1.failed_doc_ids)} failed"
        )

        if args.max_batches and total_batches >= args.max_batches:
            print(f"Reached --max-batches={args.max_batches}; stopping.")
            break

    print(
        f"\nDone. {total_batches} batch(es) processed: "
        f"{total_succeeded} signature(s) succeeded, {total_failed} failed."
    )


def cmd_cases(args: argparse.Namespace) -> None:
    from orchestration.dags.case_ingest_flow import run_case_ingest

    settings = get_settings()
    root = args.root or settings.caselaw.corpus_root
    if root is None:
        print(
            "caselaw.corpus_root is not set. Edit config/base.yaml (or "
            "config/<env>.yaml) and point it at the directory holding the "
            "case folders, e.g.:\n\n"
            "  caselaw:\n"
            "    corpus_root: /Volumes/LegalDocs/pakistani_case_law\n\n"
            "or pass --root."
        )
        sys.exit(1)

    print(f"Scanning {root} for case folders ...")
    result = run_case_ingest(
        root=root,
        db_path=settings.database.path,
        batch_id=args.batch_id,
        limit=args.limit,
    )
    print(
        f"Stored {len(result.succeeded_doc_ids)} case representation(s), "
        f"{len(result.failed_doc_ids)} failed."
    )


def cmd_domains(args: argparse.Namespace) -> None:
    from orchestration.dags.domain_discovery_flow import run_stage1_2

    result = run_stage1_2(run_id=args.run_id)

    print(f"\nRun {result.run_id}: sample {result.sample_size}/{result.total_signatures}")
    for d in result.domains:
        print(f"  [{d.doc_count} docs] {d.name} -- {d.description}")
    if result.other_bucket:
        print(f"  [{result.other_bucket.doc_count} docs] Other / Uncertain")
    if result.taxonomy_card_path:
        print(f"\nDraft taxonomy card written to: {result.taxonomy_card_path}")


def cmd_freeze_taxonomy(args: argparse.Namespace) -> None:
    from src.clustering.taxonomy_freeze import freeze_taxonomy

    settings = get_settings()
    path = freeze_taxonomy(
        run_id=args.run_id,
        accepted_domain_ids=args.accept or None,
        db_path=settings.database.path,
        output_path=args.output or settings.classification.taxonomy_file,
        version=args.version,
        overwrite=args.overwrite,
    )
    print(f"Froze taxonomy from run {args.run_id} into {path}")
    print(
        "Review it, then gather evidence and decide with:\n"
        "  python scripts/run_pipeline.py signals --run-id sig1\n"
        "  python scripts/run_pipeline.py decide --run-id dec1 --signal-run-id sig1"
    )


# The superseded single-call LLM classifier
# (orchestration/dags/classification_flow.py) is deliberately NOT reachable
# from this CLI any more. It wrote only document_classifications rows and
# never set document_representations' current-state fields, so a corpus
# classified through it looked classified in the audit table while every
# document stayed 'pending' -- and the Phase 7 freeze, which reads the
# current state, then found nothing to freeze. It also used a different
# status vocabulary ('classified'). `decide` is the one classification
# path; this stub exists so an old command line fails loudly instead of
# silently running the wrong classifier.
RETIRED_CLASSIFY_MESSAGE = """\
`classify` has been retired: it ran the superseded single-call LLM
classifier, which never wrote a document's current classification state
and used the old 'classified' status vocabulary. A corpus classified that
way cannot be evaluated or frozen.

The active classification path is the Phase 5 multi-signal decision
engine. Run, in order:

  python scripts/run_pipeline.py cases
  python scripts/run_pipeline.py signals --run-id sig1
  python scripts/run_pipeline.py validate-clusters --run-id sig1
  python scripts/run_pipeline.py decide --run-id dec1 --signal-run-id sig1

The old flow is still importable as
orchestration.dags.classification_flow.run_classification for reference,
but it is not part of the case-law pipeline."""


def cmd_classify(args: argparse.Namespace) -> None:
    print(RETIRED_CLASSIFY_MESSAGE)
    sys.exit(2)


def cmd_signals(args: argparse.Namespace) -> None:
    from orchestration.dags.domain_signal_flow import run_domain_signals

    settings = get_settings()
    result = run_domain_signals(
        run_id=args.run_id,
        db_path=settings.database.path,
        limit=args.limit,
        batch_size=args.batch_size,
        # None keeps the configured default; --no-llm forces it off for a
        # deterministic-signals-only run.
        llm_enabled=False if args.no_llm else None,
    )

    stats = result.stats
    print(f"\nSignal run {result.run_id} ({result.signal_version})")
    print(f"  corpus     : {result.corpus_size} document(s) after Phase 2")
    print(f"  processed  : {result.processed} (skipped {result.skipped_already_done} already done)")
    print(f"  clusters   : {result.n_clusters}, noise {result.noise_documents} ({result.noise_share * 100:.1f}%)")
    if stats.get("keyword_llm_agreement_rate") is not None:
        print(f"  keyword/LLM agreement: {stats['keyword_llm_agreement_rate'] * 100:.1f}%")
    if stats.get("by_llm_status"):
        print(f"  LLM status : {stats['by_llm_status']}")
    print(f"\nNext: python scripts/run_pipeline.py validate-clusters --run-id {result.run_id}")


def cmd_validate_clusters(args: argparse.Namespace) -> None:
    from orchestration.dags.cluster_validation_flow import run_cluster_validation

    settings = get_settings()
    result = run_cluster_validation(
        run_id=args.run_id,
        db_path=settings.database.path,
        output_dir=args.output_dir,
        limit=args.limit,
    )

    print(f"\nCluster validation for signal run {result.run_id}")
    print(f"  documents  : {result.n_documents}")
    print(f"  verdict    : {result.verdict}")
    best = result.best
    if best is not None:
        print(f"  clusters   : {best.n_clusters}, noise {best.noise_share * 100:.1f}%")
        print(f"  purity     : {best.weighted_purity:.3f} (lift {best.purity_lift:+.3f} over baseline)")
        if best.adjusted_rand_index is not None:
            print(f"  ARI / NMI  : {best.adjusted_rand_index:.3f} / {best.normalized_mutual_info:.3f}")
        for reason in best.verdict_reasons:
            print(f"    - {reason}")
    if result.report_path:
        print(f"  report     : {result.report_path}")

    # The gate is manual: `decide` defaults to using the cluster signal.
    if result.cluster_is_a_useful_signal:
        print("\nCluster membership is a useful domain signal; `decide` may use it (default).")
    else:
        print(
            "\nCluster membership is NOT a useful domain signal here. Pass "
            "--no-cluster-signal to `decide` so it carries no weight."
        )


def cmd_decide(args: argparse.Namespace) -> None:
    from orchestration.dags.domain_classification_flow import run_domain_classification

    settings = get_settings()
    result = run_domain_classification(
        run_id=args.run_id,
        signal_run_id=args.signal_run_id,
        db_path=settings.database.path,
        limit=args.limit,
        batch_size=args.batch_size,
        # None = consult Phase 4's stored verdict and fail closed. The flag
        # only ever forces the signal OFF; it cannot force it on.
        cluster_signal_enabled=False if args.no_cluster_signal else None,
        write_current_state=not args.dry_run,
    )

    print(f"\nDecision run {result.run_id} over signal run {result.signal_run_id}")
    print(f"  evidence   : {result.signals_available} document(s) with Phase 3 signals")
    print(f"  decided    : {result.decided} (skipped {result.skipped_already_done} already done)")
    print(f"  cluster    : {'enabled' if result.cluster_signal_enabled else 'DISABLED'}"
          f" ({result.clusters_profiled} cluster(s) profiled)")
    print(f"  accepted   : {result.auto_accepted} ({result.auto_accept_rate * 100:.1f}%)")
    print(f"  review     : {result.needs_review} ({result.review_rate * 100:.1f}%)")
    print(f"  off-domain : {result.dropped_off_domain} (withheld, not deleted)")
    print(f"  by domain  : {result.by_domain}")
    if result.by_review_reason:
        print(f"  reasons    : {result.by_review_reason}")
    if result.write_outcomes:
        print(f"  writes     : {result.write_outcomes}")
    if result.protected_by_review:
        print(f"  protected  : {result.protected_by_review} document(s) already reviewed by a human")
    if args.dry_run:
        print("\nDry run: audit rows written, no document's current state changed.")
    else:
        print(f"\nNext: python scripts/run_pipeline.py review-queue --run-id {result.run_id}")


def cmd_review_queue(args: argparse.Namespace) -> None:
    from src.classification.review_store import export_review_queue

    settings = get_settings()
    exported = export_review_queue(
        args.run_id,
        output_dir=args.output_dir or settings.review.queue_output_dir,
        db_path=settings.database.path,
        limit=args.limit,
    )

    print(f"\nReview queue for decision run {args.run_id}")
    print(f"  queued     : {exported['queued']} document(s) awaiting a human")
    print(f"  worksheet  : {exported['csv']}")
    print(f"  evidence   : {exported['jsonl']}")
    if exported["queued"]:
        print(
            "\nFill in the decision / decision_domain / reviewer / notes columns "
            "of the CSV, then:\n"
            f"  python scripts/run_pipeline.py apply-reviews --csv {exported['csv']} "
            f"--run-id {args.run_id}"
        )


def cmd_apply_reviews(args: argparse.Namespace) -> None:
    from src.classification.review_store import (
        apply_review_decisions,
        load_review_decisions_csv,
    )

    settings = get_settings()
    try:
        decisions = load_review_decisions_csv(args.csv)
    except ValueError as exc:
        # An unknown decision value: never silently discard a reviewer's work.
        print(f"Could not read {args.csv}: {exc}")
        sys.exit(1)

    if not decisions:
        print(f"No completed rows in {args.csv} -- nothing to apply.")
        return

    result = apply_review_decisions(
        decisions,
        db_path=settings.database.path,
        decision_run_id=args.run_id,
    )
    print(f"\nRecorded {result['recorded']} review decision(s) from {args.csv}")
    print(f"  state writes: {result['outcomes']}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from orchestration.dags.evaluation_flow import (
        render_evaluation_summary,
        run_evaluation,
    )

    settings = get_settings()
    result = run_evaluation(
        run_id=args.run_id,
        decision_run_id=args.decision_run_id,
        signal_run_id=args.signal_run_id,
        db_path=settings.database.path,
        labels_file=args.labels_file,
        output_dir=args.output_dir,
    )

    # The summary is the report Phase 7 already renders -- don't rebuild it here.
    print()
    print(render_evaluation_summary(result))
    if result.report_paths:
        print(f"\nReport: {result.report_paths['text']}")
        print(f"JSON  : {result.report_paths['json']}")


def cmd_freeze(args: argparse.Namespace) -> None:
    from orchestration.dags.evaluation_flow import run_evaluation
    from src.classification.taxonomy_registry import load_frozen_taxonomy
    from src.common.exceptions import ConfigurationError
    from src.evaluation.dataset_freeze import freeze_corpus

    settings = get_settings()
    taxonomy = load_frozen_taxonomy(settings.classification.taxonomy_file)
    domains = [d.domain_id for d in taxonomy.domains if not d.is_other]

    # A freeze must carry the evaluation that permitted it, and
    # freeze_corpus() refuses without one -- so the verdict is produced
    # here rather than trusted from an earlier run.
    eval_run_id = args.eval_run_id or f"eval_{args.freeze_id}"
    evaluation = run_evaluation(
        run_id=eval_run_id,
        decision_run_id=args.decision_run_id,
        signal_run_id=args.signal_run_id,
        db_path=settings.database.path,
        labels_file=args.labels_file,
    )

    print(f"\nReadiness verdict: {evaluation.verdict}")
    print(evaluation.readiness.answer)
    for blocker in evaluation.readiness.blockers:
        print(f"  blocker    : {blocker}")
    for reservation in evaluation.readiness.reservations:
        print(f"  reservation: {reservation}")

    try:
        frozen = freeze_corpus(
            args.freeze_id,
            args.decision_run_id,
            domains,
            taxonomy_version=taxonomy.version,
            readiness=evaluation.readiness,
            db_path=settings.database.path,
            signal_run_id=args.signal_run_id,
            notes=args.notes,
            allow_reservations=not args.no_reservations,
        )
    except ConfigurationError as exc:
        print(f"\nFreeze refused: {exc}")
        sys.exit(1)

    print(f"\nFroze corpus {frozen.freeze_id}")
    print(f"  documents  : {frozen.document_count}")
    print(f"  by domain  : {frozen.domain_counts}")
    print(f"  human-reviewed: {frozen.human_reviewed_count} ({frozen.human_reviewed_share * 100:.1f}%)")
    print(f"  taxonomy   : {frozen.taxonomy_version}")
    print(f"  verdict    : {frozen.readiness_verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_discover = sub.add_parser("discover", help="Scan storage.source_root and register documents")
    p_discover.set_defaults(func=cmd_discover)

    p_ingest = sub.add_parser("ingest", help="Pull + extract signatures, batch by batch, until drained")
    p_ingest.add_argument("--batch-size", type=int, default=None)
    p_ingest.add_argument("--max-batches", type=int, default=None, help="Stop after N batches (default: run until drained)")
    p_ingest.set_defaults(func=cmd_ingest)

    p_cases = sub.add_parser(
        "cases",
        help="Scan caselaw.corpus_root for case folders and store one representation each",
    )
    p_cases.add_argument("--root", type=str, default=None, help="Override caselaw.corpus_root")
    p_cases.add_argument("--batch-id", type=str, default=None)
    p_cases.add_argument("--limit", type=int, default=None, help="Stop after N case folders")
    p_cases.set_defaults(func=cmd_cases)

    p_domains = sub.add_parser("domains", help="Run Stage 1.2 domain discovery over the configured corpus")
    p_domains.add_argument("--run-id", type=str, default=None)
    p_domains.set_defaults(func=cmd_domains)

    p_freeze = sub.add_parser(
        "freeze-taxonomy",
        help="Promote reviewed draft domains of a run into config/domains.yaml",
    )
    p_freeze.add_argument("--run-id", type=str, required=True)
    p_freeze.add_argument("--accept", type=str, nargs="*", help="Domain ids to freeze (default: all not flagged for review)")
    p_freeze.add_argument("--output", type=str, default=None)
    p_freeze.add_argument("--version", type=str, default=None)
    p_freeze.add_argument("--overwrite", action="store_true", help="Replace an existing frozen taxonomy")
    p_freeze.set_defaults(func=cmd_freeze_taxonomy)

    # -- active case-law pipeline: Phases 3-7 -------------------------------
    p_signals = sub.add_parser(
        "signals",
        help="Phase 3: embed + cluster + keyword + LLM evidence for every valid case",
    )
    p_signals.add_argument("--run-id", type=str, required=True, help="Evidence version; reuse to resume")
    p_signals.add_argument("--limit", type=int, default=None, help="Stop after N documents")
    p_signals.add_argument("--batch-size", type=int, default=None)
    p_signals.add_argument("--no-llm", action="store_true", help="Deterministic signals only (no LLM calls)")
    p_signals.set_defaults(func=cmd_signals)

    p_validate = sub.add_parser(
        "validate-clusters",
        help="Phase 4: score the clustering blind and report whether it is a useful signal",
    )
    p_validate.add_argument("--run-id", type=str, required=True, help="The signal run to validate")
    p_validate.add_argument("--limit", type=int, default=None)
    p_validate.add_argument("--output-dir", type=str, default=None)
    p_validate.set_defaults(func=cmd_validate_clusters)

    p_decide = sub.add_parser(
        "decide",
        help="Phases 5+6: weigh the evidence into a domain decision and route it",
    )
    p_decide.add_argument("--run-id", type=str, required=True, help="Decision version; reuse to resume")
    p_decide.add_argument("--signal-run-id", type=str, required=True, help="The Phase 3 evidence to decide on")
    p_decide.add_argument("--limit", type=int, default=None)
    p_decide.add_argument("--batch-size", type=int, default=None)
    p_decide.add_argument(
        "--no-cluster-signal",
        action="store_true",
        help=(
            "Force the cluster signal off. Without this, the signal is used "
            "only if Phase 4's validation report approves it."
        ),
    )
    p_decide.add_argument(
        "--dry-run",
        action="store_true",
        help="Write audit rows only; leave every document's current state untouched",
    )
    p_decide.set_defaults(func=cmd_decide)

    p_queue = sub.add_parser(
        "review-queue",
        help="Phase 6: export the needs_review documents as a reviewer worksheet",
    )
    p_queue.add_argument("--run-id", type=str, required=True, help="The decision run to queue")
    p_queue.add_argument("--output-dir", type=str, default=None)
    p_queue.add_argument("--limit", type=int, default=None)
    p_queue.set_defaults(func=cmd_review_queue)

    p_apply = sub.add_parser(
        "apply-reviews",
        help="Phase 6: record a completed reviewer worksheet and apply its decisions",
    )
    p_apply.add_argument("--csv", type=str, required=True, help="The completed review_queue_*.csv")
    p_apply.add_argument("--run-id", type=str, default=None, help="Decision run the reviews belong to")
    p_apply.set_defaults(func=cmd_apply_reviews)

    p_evaluate = sub.add_parser(
        "evaluate",
        help="Phase 7: score the corpus against human labels and give a readiness verdict",
    )
    p_evaluate.add_argument("--run-id", type=str, required=True, help="Evaluation run id")
    p_evaluate.add_argument("--decision-run-id", type=str, required=True)
    p_evaluate.add_argument("--signal-run-id", type=str, default=None, help="Enables LLM/cluster diagnostics")
    p_evaluate.add_argument("--labels-file", type=str, default=None, help="External gold labels (CSV/JSON)")
    p_evaluate.add_argument("--output-dir", type=str, default=None)
    p_evaluate.set_defaults(func=cmd_evaluate)

    p_freeze_corpus = sub.add_parser(
        "freeze",
        help="Phase 7: snapshot the accepted corpus, if the evaluation permits it",
    )
    p_freeze_corpus.add_argument("--freeze-id", type=str, required=True, help="Immutable snapshot id")
    p_freeze_corpus.add_argument("--decision-run-id", type=str, required=True)
    p_freeze_corpus.add_argument("--signal-run-id", type=str, default=None)
    p_freeze_corpus.add_argument("--eval-run-id", type=str, default=None, help="Default: eval_<freeze-id>")
    p_freeze_corpus.add_argument("--labels-file", type=str, default=None)
    p_freeze_corpus.add_argument("--notes", type=str, default=None)
    p_freeze_corpus.add_argument(
        "--no-reservations",
        action="store_true",
        help="Refuse to freeze when the evaluation passed only with reservations",
    )
    p_freeze_corpus.set_defaults(func=cmd_freeze)

    # Retired: ran the superseded classifier. See RETIRED_CLASSIFY_MESSAGE.
    p_classify = sub.add_parser(
        "classify",
        help="RETIRED -- use `decide` (Phase 5 multi-signal decision engine)",
    )
    p_classify.set_defaults(func=cmd_classify)

    p_all = sub.add_parser("all", help="discover -> ingest -> domains, in sequence")
    p_all.add_argument("--batch-size", type=int, default=None)
    p_all.add_argument("--max-batches", type=int, default=None)
    p_all.add_argument("--run-id", type=str, default=None)

    def cmd_all(args: argparse.Namespace) -> None:
        cmd_discover(args)
        cmd_ingest(args)
        cmd_domains(args)

    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()