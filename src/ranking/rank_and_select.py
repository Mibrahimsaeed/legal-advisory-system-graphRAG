"""Rank discovered clusters by size and split top-N vs "Other / Uncertain".

Stage 1.2 spec points 6 and 8: rank clusters by document count, take the
top 3 largest as domain candidates, and route everything else --
smaller clusters *and* HDBSCAN noise alike -- into a single
"Other / Uncertain" bucket rather than forcing every document into a
named domain it doesn't clearly belong to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.ranking.volume_aggregator import ClusterVolume


@dataclass
class SelectionResult:
    top_clusters: list[ClusterVolume]
    other_doc_ids: list[str] = field(default_factory=list)

    @property
    def other_doc_count(self) -> int:
        return len(self.other_doc_ids)


def rank_clusters(volumes: list[ClusterVolume]) -> list[ClusterVolume]:
    """Non-noise clusters sorted by document count, largest first.

    Ties are broken by ``cluster_id`` purely for deterministic output
    ordering (label IDs are otherwise arbitrary/unstable across runs).
    """

    real_clusters = [v for v in volumes if not v.is_noise]
    return sorted(real_clusters, key=lambda v: (-v.doc_count, v.cluster_id))


def select_top_n(volumes: list[ClusterVolume], top_n: int = 3) -> SelectionResult:
    """Top ``top_n`` clusters by size, plus every remaining document
    (smaller clusters + noise) folded into the "Other / Uncertain" bucket."""

    ranked = rank_clusters(volumes)
    top = ranked[:top_n]
    top_ids = {v.cluster_id for v in top}

    other_doc_ids: list[str] = []
    for volume in volumes:
        if volume.cluster_id not in top_ids:
            other_doc_ids.extend(volume.doc_ids)

    return SelectionResult(top_clusters=top, other_doc_ids=other_doc_ids)