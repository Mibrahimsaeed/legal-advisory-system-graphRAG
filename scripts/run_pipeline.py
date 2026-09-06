#!/usr/bin/env python3
"""Command-line entry point for running the pipeline against a real corpus.

This is the file you actually run. It ties together the pieces that were
previously only importable as functions:

    cases     -> orchestration.dags.case_ingest_flow.run_case_ingest
    domains   -> orchestration.dags.domain_discovery_flow.run_stage1_2

and, for the legacy PDF/book corpus only:

    discover  -> src.ingestion.discovery.discover_local_documents
    ingest    -> orchestration.dags.batch_ingest_flow.run_stage0
                 + orchestration.dags.feature_extraction_flow.run_stage1
                 (looped batch-by-batch until the manifest has nothing
                 left in 'pending'/'claimed')

Usage (case law -- the active corpus)::

    # 1. scan caselaw.corpus_root for case folders (case.html +
    #    metadata.json) and store one lightweight representation each.
    python scripts/run_pipeline.py cases

    # 2. run domain discovery (Stage 1.2) over those representations.
    python scripts/run_pipeline.py domains

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

from src.common.config import get_settings
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
    print("Review it, then classify with: python scripts/run_pipeline.py classify --pilot 25")


def cmd_classify(args: argparse.Namespace) -> None:
    from orchestration.dags.classification_flow import run_classification

    settings = get_settings()
    result = run_classification(
        run_id=args.run_id,
        db_path=settings.database.path,
        cluster_run_id=args.cluster_run_id,
        batch_size=args.batch_size,
        pilot_size=args.pilot,
    )

    stats = result.stats
    label = "PILOT" if result.is_pilot else "FULL"
    print(f"\n{label} classification run {result.run_id}")
    print(f"  taxonomy   : {result.taxonomy_version}")
    print(f"  classifier : {result.classifier_version} ({result.model_name})")
    print(f"  processed  : {result.processed} (skipped {result.skipped_already_done} already done)")
    print(f"  by status  : {stats.get('by_status')}")
    print(f"  by domain  : {stats.get('by_primary_domain')}")
    if stats.get("by_review_reason"):
        print(f"  review     : {stats['by_review_reason']}")
    if stats.get("by_error"):
        print(f"  errors     : {stats['by_error']}")
    mean = stats.get("mean_confidence")
    if mean is not None:
        print(f"  confidence : mean={mean:.2f} min={stats['min_confidence']:.2f} max={stats['max_confidence']:.2f}")
    print(f"  needs review: {stats.get('needs_review_rate', 0) * 100:.1f}%  failures: {stats.get('failure_rate', 0) * 100:.1f}%")
    if result.is_pilot:
        print("\nPilot only. Re-run without --pilot (same --run-id) to continue the full corpus.")


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

    p_classify = sub.add_parser(
        "classify",
        help="Classify the corpus against the frozen taxonomy (resumable, batched)",
    )
    p_classify.add_argument("--run-id", type=str, required=True, help="Reuse to resume; change when taxonomy/classifier changes")
    p_classify.add_argument("--cluster-run-id", type=str, default=None, help="Discovery/review run to read cluster_id from")
    p_classify.add_argument("--batch-size", type=int, default=None)
    p_classify.add_argument("--pilot", type=int, default=None, help="Classify only N documents (pilot batch)")
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