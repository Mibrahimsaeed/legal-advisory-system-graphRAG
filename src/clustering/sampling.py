"""Stratified sampling of document records for a controlled clustering run.

Stage 1.2 proper clusters the whole corpus. A *review* run
(``orchestration/dags/cluster_review_flow.py``) deliberately starts with
a sample instead: it is faster to iterate on, and if the clustering is
wrong the sample shows it just as clearly as 10k documents would.

There's a chicken-and-egg problem with stratifying by "domain": that's
the thing this stage discovers, so it can't also be the sampling key.
Instead this stratifies by proxies that are (a) already on the document
record and (b) plausibly correlated with subject matter before any
domain is known: a coarse text-length bucket (a two-page order and a
hundred-page judgment are different kinds of document) and, for case
law, the court. Records that carry neither -- e.g. the legacy
:class:`~src.extraction.signature.DocumentSignature` -- simply fall into
the "unknown" court stratum, so this stays usable for both corpora.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Callable

from typing import Any

from src.common.logging_utils import get_logger

logger = get_logger(__name__)

StrataKeyFn = Callable[[Any], tuple]

_SHORT_MAX_CHARS = 2_000
_MEDIUM_MAX_CHARS = 8_000


def _length_bucket(char_count: int) -> str:
    if char_count < _SHORT_MAX_CHARS:
        return "short"
    if char_count < _MEDIUM_MAX_CHARS:
        return "medium"
    return "long"


def default_strata_key(document: Any) -> tuple:
    """Default stratification key: ``(length_bucket, court)``.

    Read with ``getattr`` rather than typed against one record class: the
    classification path is corpus-neutral (see
    :class:`src.extraction.doc_representation.EmbeddableDocument`), and a
    record without a court still stratifies usefully by length.
    """

    return (
        _length_bucket(getattr(document, "char_count", 0) or 0),
        getattr(document, "court", None) or "unknown",
    )


def stratified_sample(
    documents: list[Any],
    sample_min: int,
    sample_max: int,
    seed: int = 42,
    key_fn: StrataKeyFn = default_strata_key,
) -> list[Any]:
    """Draw a stratified sample sized between ``sample_min`` and ``sample_max``.

    Allocation per stratum is proportional to that stratum's share of the
    population, rounded via the largest-remainder method so the sample
    size lands exactly on the target rather than drifting from repeated
    ``floor()`` rounding. If the population is smaller than
    ``sample_min``, every document is returned (nothing to sample) and a
    warning is logged -- Stage 1.2 can still run, just on less data than
    intended.
    """

    population = len(documents)
    if population == 0:
        return []

    if population <= sample_min:
        logger.warning(
            "Population (%d) is at or below sample_min (%d); using the "
            "full population instead of sampling",
            population,
            sample_min,
        )
        return list(documents)

    target = min(sample_max, population)

    strata: dict[tuple, list[Any]] = defaultdict(list)
    for document in documents:
        strata[key_fn(document)].append(document)

    exact_allocations: dict[tuple, float] = {
        key: target * len(items) / population for key, items in strata.items()
    }
    floor_allocations: dict[tuple, int] = {
        key: math.floor(exact) for key, exact in exact_allocations.items()
    }

    remainder = target - sum(floor_allocations.values())
    fractional_order = sorted(
        strata.keys(),
        key=lambda k: exact_allocations[k] - floor_allocations[k],
        reverse=True,
    )
    for key in fractional_order[: max(remainder, 0)]:
        floor_allocations[key] += 1

    rng = random.Random(seed)
    sampled: list[Any] = []
    for key, items in strata.items():
        n = min(floor_allocations.get(key, 0), len(items))
        sampled.extend(rng.sample(items, n))

    rng.shuffle(sampled)

    logger.info(
        "Stratified sample: %d/%d documents across %d strata (target=%d)",
        len(sampled),
        population,
        len(strata),
        target,
    )
    return sampled