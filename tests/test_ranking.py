from __future__ import annotations

import numpy as np
import pytest

from src.clustering.cluster import NOISE_LABEL
from src.ranking.rank_and_select import rank_clusters, select_top_n
from src.ranking.volume_aggregator import ClusterVolume, aggregate_cluster_volumes


def test_aggregate_cluster_volumes_groups_by_label():
    doc_ids = [f"d{i}" for i in range(10)]
    labels = np.array([0, 0, 0, 1, 1, -1, -1, -1, 2, 2])
    volumes = aggregate_cluster_volumes(doc_ids, labels)
    by_id = {v.cluster_id: v for v in volumes}
    assert by_id[0].doc_count == 3
    assert by_id[1].doc_count == 2
    assert by_id[2].doc_count == 2
    assert by_id[NOISE_LABEL].doc_count == 3


def test_aggregate_cluster_volumes_carries_probabilities():
    doc_ids = ["a", "b", "c"]
    labels = np.array([0, 0, 1])
    probs = np.array([0.9, 0.8, 0.5])
    volumes = aggregate_cluster_volumes(doc_ids, labels, probs)
    by_id = {v.cluster_id: v for v in volumes}
    assert by_id[0].probabilities == [0.9, 0.8]
    assert by_id[1].probabilities == [0.5]


def test_aggregate_cluster_volumes_rejects_length_mismatch():
    with pytest.raises(ValueError):
        aggregate_cluster_volumes(["a", "b"], np.array([0]))


def test_cluster_volume_is_noise_property():
    assert ClusterVolume(cluster_id=NOISE_LABEL).is_noise is True
    assert ClusterVolume(cluster_id=0).is_noise is False


def test_rank_clusters_excludes_noise_and_sorts_by_size():
    volumes = [
        ClusterVolume(cluster_id=0, doc_ids=["a"] * 5),
        ClusterVolume(cluster_id=1, doc_ids=["b"] * 20),
        ClusterVolume(cluster_id=NOISE_LABEL, doc_ids=["c"] * 100),
        ClusterVolume(cluster_id=2, doc_ids=["d"] * 10),
    ]
    ranked = rank_clusters(volumes)
    assert [v.cluster_id for v in ranked] == [1, 2, 0]


def test_rank_clusters_breaks_ties_by_cluster_id():
    volumes = [
        ClusterVolume(cluster_id=5, doc_ids=["a"] * 10),
        ClusterVolume(cluster_id=1, doc_ids=["b"] * 10),
    ]
    ranked = rank_clusters(volumes)
    assert [v.cluster_id for v in ranked] == [1, 5]


def test_select_top_n_buckets_everything_else_as_other():
    volumes = [
        ClusterVolume(cluster_id=0, doc_ids=[f"a{i}" for i in range(8)]),
        ClusterVolume(cluster_id=1, doc_ids=[f"b{i}" for i in range(5)]),
        ClusterVolume(cluster_id=2, doc_ids=[f"c{i}" for i in range(3)]),
        ClusterVolume(cluster_id=NOISE_LABEL, doc_ids=[f"n{i}" for i in range(4)]),
    ]
    selection = select_top_n(volumes, top_n=2)
    assert [v.cluster_id for v in selection.top_clusters] == [0, 1]
    assert selection.other_doc_count == 3 + 4
    assert set(selection.other_doc_ids) == {f"c{i}" for i in range(3)} | {f"n{i}" for i in range(4)}


def test_select_top_n_with_fewer_clusters_than_top_n():
    volumes = [ClusterVolume(cluster_id=0, doc_ids=["a", "b"])]
    selection = select_top_n(volumes, top_n=3)
    assert len(selection.top_clusters) == 1
    assert selection.other_doc_count == 0


def test_select_top_n_all_noise_yields_no_top_clusters():
    volumes = [ClusterVolume(cluster_id=NOISE_LABEL, doc_ids=["a", "b", "c"])]
    selection = select_top_n(volumes, top_n=3)
    assert selection.top_clusters == []
    assert selection.other_doc_count == 3