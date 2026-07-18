"""Aggregate document counts per HDBSCAN cluster label.

Turns the flat ``(doc_id, label)`` pairing HDBSCAN produces into
per-cluster volumes -- what :mod:`~src.ranking.rank_and_select` sorts on
to find the top-N largest discovered domains (Stage 1.2 spec, point 6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.clustering.cluster import NOISE_LABEL


@dataclass
class ClusterVolume:
    cluster_id: int
    doc_ids: list[str] = field(default_factory=list)
    probabilities: list[float] = field(default_factory=list)

    @property
    def doc_count(self) -> int:
        return len(self.doc_ids)

    @property
    def is_noise(self) -> bool:
        return self.cluster_id == NOISE_LABEL


def aggregate_cluster_volumes(
    doc_ids: list[str],
    labels: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> list[ClusterVolume]:
    """Group ``doc_ids`` by their parallel ``labels`` into one :class:`ClusterVolume`
    per distinct label (including the ``NOISE_LABEL`` bucket, if present)."""

    if len(doc_ids) != len(labels):
        raise ValueError(
            f"doc_ids ({len(doc_ids)}) and labels ({len(labels)}) must be the same length"
        )

    by_cluster: dict[int, ClusterVolume] = {}
    for i, (doc_id, label) in enumerate(zip(doc_ids, labels)):
        cluster_id = int(label)
        volume = by_cluster.setdefault(cluster_id, ClusterVolume(cluster_id=cluster_id))
        volume.doc_ids.append(doc_id)
        if probabilities is not None:
            volume.probabilities.append(float(probabilities[i]))

    return list(by_cluster.values())