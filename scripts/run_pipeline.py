#!/usr/bin/env python3
"""Command-line entry point for running the pipeline against a real corpus.

This is the file you actually run. It ties together the pieces that were
previously only importable as functions:

    discover  -> src.ingestion.discovery.discover_local_documents
    ingest    -> orchestration.dags.batch_ingest_flow.run_stage0
                 + orchestration.dags.feature_extraction_flow.run_stage1
                 (looped batch-by-batch until the manifest has nothing
                 left in 'pending'/'claimed')
    domains   -> orchestration.dags.domain_discovery_flow.run_stage1_2

Usage::

    # 1. one-time (or whenever new files land on the disk): register
    #    every PDF under storage.source_root into the manifest.
    python scripts/run_pipeline.py discover

    # 2. pull + extract signatures, batch by batch, until the manifest
    #    is drained.
    python scripts/run_pipeline.py ingest

    # 3. once enough signatures exist (a few thousand), run domain
    #    discovery (Stage 1.2) on a sample of them.
    python scripts/run_pipeline.py domains

    # or all three in sequence:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_discover = sub.add_parser("discover", help="Scan storage.source_root and register documents")
    p_discover.set_defaults(func=cmd_discover)

    p_ingest = sub.add_parser("ingest", help="Pull + extract signatures, batch by batch, until drained")
    p_ingest.add_argument("--batch-size", type=int, default=None)
    p_ingest.add_argument("--max-batches", type=int, default=None, help="Stop after N batches (default: run until drained)")
    p_ingest.set_defaults(func=cmd_ingest)

    p_domains = sub.add_parser("domains", help="Run Stage 1.2 domain discovery on a sample of signatures")
    p_domains.add_argument("--run-id", type=str, default=None)
    p_domains.set_defaults(func=cmd_domains)

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