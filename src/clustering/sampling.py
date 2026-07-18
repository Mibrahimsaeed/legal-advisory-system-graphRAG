"""Stratified sampling of Stage 1 document signatures for Stage 1.2.

Domain Discovery clusters a *sample* (~1,500-2,000 documents), not the
full corpus -- 14k documents through UMAP/HDBSCAN is unnecessary cost for
a one-time exploratory clustering pass, and a good stratified sample is
representative enough to find the dominant domains.

There's a chicken-and-egg problem with stratifying by "domain": that's
the thing this stage discovers, so it can't also be the sampling key.
Instead this stratifies by proxies that are (a) already on the signature
record and (b) plausibly correlated with document *type* even before
domain is known: a coarse body-length bucket, whether the document
needed OCR, and which extractor produced it. This keeps short contracts,
long scanned filings, etc. all represented rather than the sample
accidentally skewing toward whichever type happens to be most common.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Callable

from src.common.logging_utils import get_logger
from src.extraction.signature import DocumentSignature

logger = get_logger(__name__)

StrataKeyFn = Callable[[DocumentSignature], tuple]

_SHORT_MAX_CHARS = 2_000
_MEDIUM_MAX_CHARS = 8_000


def _length_bucket(char_count: int) -> str:
    if char_count < _SHORT_MAX_CHARS:
        return "short"
    if char_count < _MEDIUM_MAX_CHARS:
        return "medium"
    return "long"


def default_strata_key(signature: DocumentSignature) -> tuple:
    """Default stratification key: (length_bucket, is_scanned, extractor_used)."""

    return (
        _length_bucket(signature.char_count),
        signature.is_scanned,
        signature.extractor_used or "unknown",
    )


def stratified_sample(
    signatures: list[DocumentSignature],
    sample_min: int,
    sample_max: int,
    seed: int = 42,
    key_fn: StrataKeyFn = default_strata_key,
) -> list[DocumentSignature]:
    """Draw a stratified sample sized between ``sample_min`` and ``sample_max``.

    Allocation per stratum is proportional to that stratum's share of the
    population, rounded via the largest-remainder method so the sample
    size lands exactly on the target rather than drifting from repeated
    ``floor()`` rounding. If the population is smaller than
    ``sample_min``, every signature is returned (nothing to sample) and a
    warning is logged -- Stage 1.2 can still run, just on less data than
    intended.
    """

    population = len(signatures)
    if population == 0:
        return []

    if population <= sample_min:
        logger.warning(
            "Population (%d) is at or below sample_min (%d); using the "
            "full population instead of sampling",
            population,
            sample_min,
        )
        return list(signatures)

    target = min(sample_max, population)

    strata: dict[tuple, list[DocumentSignature]] = defaultdict(list)
    for sig in signatures:
        strata[key_fn(sig)].append(sig)

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
    sampled: list[DocumentSignature] = []
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