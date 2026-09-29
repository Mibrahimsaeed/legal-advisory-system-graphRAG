"""Phase 4: blind cluster validation.

The tests are organised around the two things that make this phase
trustworthy:

* **Blindness.** The clusterer can only ever receive vectors, and labels
  are read after the fact. If that ever stopped being true the whole
  evaluation would be circular.
* **A verdict that can say no.** Purity is scored against the majority
  baseline, so a clustering that merely reflects corpus imbalance is
  reported as useless rather than as 85% pure. There are tests for a
  clustering that genuinely separates domains, one that is pure noise,
  and one that looks good only because the corpus is unbalanced.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from orchestration.dags.cluster_validation_flow import (
    _sweep_settings,
    run_cluster_validation,
)
from src.classification.case_representation import build_case_representation
from src.clustering.cluster import ClusterResult, NOISE_LABEL
from src.clustering.cluster_validation import (
    VERDICT_NOT_EVALUABLE,
    VERDICT_NOT_USEFUL,
    VERDICT_USEFUL,
    VERDICT_WEAK,
    assess_usefulness,
    evaluate_clustering,
    labels_from_source_folder,
)
from src.common.config import get_settings
from src.common.db import init_schema
from src.embedding.embed_model import DeterministicHashEmbedder
from src.extraction.doc_representation import (
    CLASSIFICATION_STATUS_DROPPED_PROCEDURAL,
    CLASSIFICATION_STATUS_PENDING,
    DocumentRepresentation,
)
from src.extraction.representation_store import (
    DEFAULT_REPRESENTATION_SCHEMA_FILE,
    upsert_representations,
)

FAMILY_TEXT = (
    "The respondent instituted a suit for recovery of dower, dowry articles and "
    "maintenance allowance before the learned Judge Family Court. The wife seeks "
    "dissolution of marriage on the basis of khula. Custody of the minor is decided "
    "with reference to the welfare of the minor under the Guardians and Wards Act. "
    "The nikahnama was exhibited and the matrimonial dispute is disposed of. "
)
CRIMINAL_TEXT = (
    "The accused was convicted under the Penal Code and sentenced by the trial "
    "court. Learned counsel challenges the conviction, arguing the ocular account "
    "of the complainant is doubtful. The investigating officer joined no private "
    "witness. The prosecution failed to prove the charge and bail was declined "
    "below; the appeal against conviction is allowed. "
)


def _doc(doc_id: str, folder: str, text: str, **overrides) -> DocumentRepresentation:
    base = dict(
        doc_id=doc_id,
        source_uri=f"/corpus/{folder}/{doc_id}",
        source_relpath=f"{folder}/{doc_id}",
        source_type="case_html",
        title=f"{doc_id} cause title",
        headings=["Facts", "Order"],
        body_preview=text[:400],
        cleaned_text=text * 4,
        char_count=len(text) * 4,
        court="Lahore High Court",
        classification_status=CLASSIFICATION_STATUS_PENDING,
    )
    base.update(overrides)
    return DocumentRepresentation(**base)


def _labelled(n_family: int = 10, n_criminal: int = 10) -> list[DocumentRepresentation]:
    docs = [_doc(f"fam_{i:03d}", "family", FAMILY_TEXT) for i in range(n_family)]
    docs += [_doc(f"crim_{i:03d}", "criminal", CRIMINAL_TEXT) for i in range(n_criminal)]
    return docs


def _truth(docs) -> dict[str, str]:
    return labels_from_source_folder([build_case_representation(d) for d in docs])


# ---------------------------------------------------------------------------
# Blindness
# ---------------------------------------------------------------------------


def test_the_clusterer_contract_cannot_accept_labels():
    """Structural guarantee: Clusterer is (vectors) -> ClusterResult."""

    import inspect

    from src.clustering.cluster import Clusterer

    params = list(inspect.signature(Clusterer.__call__).parameters)
    assert params == ["self", "vectors"]


def test_clustering_receives_only_vectors(tmp_path):
    """Instrument the clusterer: it must see an array and nothing else."""

    db = tmp_path / "v.db"
    init_schema(db_path=db, schema_file=str(DEFAULT_REPRESENTATION_SCHEMA_FILE))
    upsert_representations(_labelled(4, 4), db_path=db)

    seen: list[object] = []

    def _recording_clusterer(vectors):
        seen.append(vectors)
        return ClusterResult(
            labels=np.zeros(vectors.shape[0], dtype=int),
            probabilities=None,
            n_clusters=1,
        )

    with _validation_settings(tmp_path):
        run_cluster_validation(
            run_id="v1", db_path=db,
            embedder=DeterministicHashEmbedder(dimension=16),
            clusterer=_recording_clusterer,
        )

    assert seen, "clusterer was never called"
    for arg in seen:
        assert isinstance(arg, np.ndarray)
        assert arg.dtype.kind == "f"  # vectors, never labels or ids


# ---------------------------------------------------------------------------
# Label extraction (evaluation only)
# ---------------------------------------------------------------------------


def test_labels_come_from_the_source_folder():
    reps = [build_case_representation(d) for d in _labelled(2, 2)]

    labels = labels_from_source_folder(reps)

    assert labels == {
        "fam_000": "family", "fam_001": "family",
        "crim_000": "criminal", "crim_001": "criminal",
    }


def test_label_depth_is_configurable():
    rep = build_case_representation(
        _doc("c1", "criminal/narcotics", CRIMINAL_TEXT)
    )

    assert labels_from_source_folder([rep], depth=1) == {"c1": "criminal"}
    assert labels_from_source_folder([rep], depth=2) == {"c1": "criminal/narcotics"}


def test_documents_with_no_parent_folder_are_unlabelled():
    """A flat corpus carries no labels -- it must not invent one."""

    rep = build_case_representation(_doc("c1", "", CRIMINAL_TEXT, source_relpath="c1"))

    assert labels_from_source_folder([rep]) == {}


# ---------------------------------------------------------------------------
# Evaluation: the measurements
# ---------------------------------------------------------------------------


def test_perfect_separation_is_reported_as_useful():
    docs = _labelled(10, 10)
    doc_ids = [d.doc_id for d in docs]
    labels = np.array([0] * 10 + [1] * 10)

    report = evaluate_clustering("r1", doc_ids, labels, _truth(docs))

    assert report.n_clusters == 2
    assert report.noise_documents == 0
    assert report.coverage == 1.0
    assert report.weighted_purity == 1.0
    assert report.baseline_majority_share == pytest.approx(0.5)
    assert report.purity_lift == pytest.approx(0.5)
    assert report.adjusted_rand_index == pytest.approx(1.0)
    assert report.per_label_recall == {"family": 1.0, "criminal": 1.0}
    assert report.verdict == VERDICT_USEFUL
    assert report.cluster_is_a_useful_signal


def test_random_clustering_is_reported_as_not_useful():
    """Clusters independent of domain must not pass, whatever the purity."""

    docs = _labelled(20, 20)
    doc_ids = [d.doc_id for d in docs]
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=len(doc_ids))

    report = evaluate_clustering("r1", doc_ids, labels, _truth(docs))

    assert report.purity_lift < 0.15
    assert report.adjusted_rand_index is not None and report.adjusted_rand_index < 0.1
    assert report.verdict == VERDICT_NOT_USEFUL
    assert not report.cluster_is_a_useful_signal
    assert any("majority baseline" in r for r in report.verdict_reasons)


def test_high_purity_from_an_unbalanced_corpus_is_not_mistaken_for_signal():
    """The point of the baseline: 95% purity is worthless when 95% of the
    corpus is one domain and the clustering just reflects that."""

    docs = _labelled(n_family=95, n_criminal=5)
    doc_ids = [d.doc_id for d in docs]
    # Two clusters that carve the corpus without regard to domain: the five
    # criminal cases are split across both, so neither cluster is anything
    # but family-dominated.
    labels = np.array([0] * 47 + [1] * 48 + [0, 1, 0, 1, 0])

    report = evaluate_clustering("r1", doc_ids, labels, _truth(docs))

    assert report.weighted_purity > 0.85           # looks impressive
    assert report.baseline_majority_share > 0.90   # but the baseline is higher
    assert report.purity_lift < 0.15               # so the lift is nil
    assert report.per_label_recall["criminal"] == 0.0  # criminal is unrecoverable
    assert report.verdict == VERDICT_NOT_USEFUL


def test_one_domain_separated_and_the_other_lost_is_not_useful():
    """Per-label recall catches a run that only finds one domain."""

    docs = _labelled(20, 20)
    doc_ids = [d.doc_id for d in docs]
    # Family forms a clean cluster; criminal is scattered as noise.
    labels = np.array([0] * 20 + [NOISE_LABEL] * 20)

    report = evaluate_clustering("r1", doc_ids, labels, _truth(docs))

    assert report.per_label_recall["family"] == 1.0
    assert report.per_label_recall["criminal"] == 0.0
    assert report.coverage == pytest.approx(0.5)
    assert report.verdict in {VERDICT_WEAK, VERDICT_NOT_USEFUL}
    assert not report.cluster_is_a_useful_signal


def test_noise_is_excluded_from_purity_but_reported():
    docs = _labelled(10, 10)
    doc_ids = [d.doc_id for d in docs]
    labels = np.array([0] * 8 + [NOISE_LABEL] * 2 + [1] * 8 + [NOISE_LABEL] * 2)

    report = evaluate_clustering("r1", doc_ids, labels, _truth(docs))

    assert report.noise_documents == 4
    assert report.noise_share == pytest.approx(0.2)
    assert report.coverage == pytest.approx(0.8)
    assert report.weighted_purity == 1.0  # the clustered ones are pure
    assert any("noise" in n for n in report.notes)


def test_mixed_clusters_are_identified():
    docs = _labelled(10, 10)
    doc_ids = [d.doc_id for d in docs]
    # Cluster 0 is pure family; cluster 1 mixes 6 criminal with 4 family.
    labels = np.array([0] * 6 + [1] * 4 + [1] * 10)

    report = evaluate_clustering("r1", doc_ids, labels, _truth(docs))

    mixed = report.mixed_clusters
    assert [c.cluster_id for c in mixed] == [1]
    assert mixed[0].dominant_label == "criminal"
    assert mixed[0].purity == pytest.approx(10 / 14)
    assert mixed[0].margin == pytest.approx((10 - 4) / 14)


def test_cluster_sizes_and_shares_are_reported():
    docs = _labelled(6, 4)
    doc_ids = [d.doc_id for d in docs]
    labels = np.array([0] * 6 + [1] * 4)

    report = evaluate_clustering("r1", doc_ids, labels, _truth(docs))

    by_id = {c.cluster_id: c for c in report.per_cluster}
    assert by_id[0].size == 6 and by_id[0].share == pytest.approx(0.6)
    assert by_id[1].size == 4 and by_id[1].share == pytest.approx(0.4)
    assert report.contingency["0"] == {"family": 6}
    assert report.label_distribution == {"family": 6, "criminal": 4}


def test_a_single_cluster_cannot_discriminate():
    docs = _labelled(10, 10)
    labels = np.zeros(len(docs), dtype=int)

    report = evaluate_clustering("r1", [d.doc_id for d in docs], labels, _truth(docs))

    assert report.n_clusters == 1
    assert report.verdict == VERDICT_NOT_USEFUL
    assert any("cannot discriminate" in r for r in report.verdict_reasons)


def test_unlabelled_corpus_is_reported_as_not_evaluable():
    """No labels means no evaluation -- never a fabricated verdict."""

    docs = [_doc(f"d{i}", "", CRIMINAL_TEXT, source_relpath=f"d{i}") for i in range(10)]
    labels = np.array([0] * 5 + [1] * 5)

    report = evaluate_clustering("r1", [d.doc_id for d in docs], labels, {})

    assert report.verdict == VERDICT_NOT_EVALUABLE
    assert report.weighted_purity == 0.0
    assert any("no source-folder labels" in r for r in report.verdict_reasons)


def test_single_label_corpus_is_not_evaluable():
    docs = _labelled(10, 0)
    labels = np.array([0] * 5 + [1] * 5)

    report = evaluate_clustering("r1", [d.doc_id for d in docs], labels, _truth(docs))

    assert report.verdict == VERDICT_NOT_EVALUABLE
    assert any("one distinct label" in r for r in report.verdict_reasons)


def test_unlabelled_documents_are_excluded_and_counted():
    docs = _labelled(5, 5) + [
        _doc(f"orphan{i}", "", CRIMINAL_TEXT, source_relpath=f"orphan{i}") for i in range(3)
    ]
    labels = np.array([0] * 5 + [1] * 5 + [1] * 3)

    report = evaluate_clustering("r1", [d.doc_id for d in docs], labels, _truth(docs))

    assert report.n_documents == 13
    assert any("no source-folder label" in n for n in report.notes)
    assert sum(report.label_distribution.values()) == 10


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        evaluate_clustering("r1", ["a", "b"], np.array([0]), {"a": "x", "b": "y"})


def test_thresholds_are_not_hardcoded_in_the_verdict():
    """The same measurements can yield different verdicts under different
    configured thresholds -- the rule is policy, not physics."""

    values = {
        "n_clusters": 2, "coverage": 0.6, "weighted_purity": 0.70,
        "baseline_majority_share": 0.55, "purity_lift": 0.15,
        "adjusted_rand_index": 0.12, "per_label_recall": {"a": 0.6, "b": 0.55},
    }

    lenient, _ = assess_usefulness(values)
    strict, _ = assess_usefulness(
        values, min_coverage=0.9, min_purity_lift=0.4, min_ari=0.5, min_label_recall=0.9
    )

    assert lenient == VERDICT_USEFUL
    assert strict in {VERDICT_WEAK, VERDICT_NOT_USEFUL}


def test_verdict_reasons_always_carry_the_numbers():
    docs = _labelled(10, 10)
    labels = np.array([0] * 10 + [1] * 10)

    report = evaluate_clustering("r1", [d.doc_id for d in docs], labels, _truth(docs))

    joined = " ".join(report.verdict_reasons)
    assert "coverage" in joined and "baseline" in joined and "lift" in joined


# ---------------------------------------------------------------------------
# The flow
# ---------------------------------------------------------------------------


def _validation_settings(tmp_path, **overrides):
    """Patch the flow's settings; returns a context manager."""

    import contextlib

    import orchestration.dags.cluster_validation_flow as flow

    real = get_settings()
    validation = real.cluster_validation.model_copy(
        update={
            "output_dir": tmp_path / "validation",
            "sweep_min_cluster_sizes": [2],
            "sweep_min_samples": [0],
            **overrides,
        }
    )
    fake = SimpleNamespace(
        cluster_validation=validation,
        discovery=real.discovery.model_copy(update={"umap_min_docs": 10_000}),
        domain_signals=real.domain_signals,
        caselaw=real.caselaw,
        metrics=SimpleNamespace(db_path=tmp_path / "metrics.db"),
    )

    @contextlib.contextmanager
    def _patched():
        original = flow.get_settings
        flow.get_settings = lambda: fake
        try:
            yield fake
        finally:
            flow.get_settings = original

    return _patched()


@pytest.fixture()
def db(tmp_path) -> Path:
    db_path = tmp_path / "validation.db"
    init_schema(db_path=db_path, schema_file=str(DEFAULT_REPRESENTATION_SCHEMA_FILE))
    return db_path


def _separating_clusterer(vectors: np.ndarray) -> ClusterResult:
    """Splits the batch in half -- the corpus is seeded family-then-criminal."""

    n = vectors.shape[0]
    labels = np.array([0 if i < n // 2 else 1 for i in range(n)])
    return ClusterResult(
        labels=labels, probabilities=np.full(n, 0.9), n_clusters=2
    )


def test_flow_scores_the_clustering_and_writes_a_report(db, tmp_path):
    upsert_representations(_labelled(6, 6), db_path=db)

    with _validation_settings(tmp_path) as settings:
        result = run_cluster_validation(
            run_id="v1", db_path=db,
            embedder=DeterministicHashEmbedder(dimension=16),
            clusterer=_separating_clusterer,
        )

    assert result.n_documents == 12
    assert result.verdict == VERDICT_USEFUL
    assert result.cluster_is_a_useful_signal

    payload = json.loads(Path(result.report_path).read_text())
    assert payload["run_id"] == "v1"
    assert payload["verdict"] == VERDICT_USEFUL
    assert payload["best"]["weighted_purity"] == 1.0
    assert payload["best"]["parameters"]["embedding_model"]
    assert payload["sweep"]


def test_flow_reports_an_unusable_signal_rather_than_forcing_it(db, tmp_path):
    upsert_representations(_labelled(10, 10), db_path=db)

    def _noise_clusterer(vectors):
        n = vectors.shape[0]
        return ClusterResult(
            labels=np.full(n, NOISE_LABEL), probabilities=np.zeros(n), n_clusters=0
        )

    with _validation_settings(tmp_path):
        result = run_cluster_validation(
            run_id="v1", db_path=db,
            embedder=DeterministicHashEmbedder(dimension=16),
            clusterer=_noise_clusterer,
        )

    assert result.verdict == VERDICT_NOT_USEFUL
    assert result.cluster_is_a_useful_signal is False
    assert result.best.noise_share == 1.0


def test_flow_excludes_phase_2_drops(db, tmp_path):
    docs = _labelled(5, 5) + [
        _doc(
            f"dropped_{i}", "family", "Case called. None present. Adjourned.",
            classification_status=CLASSIFICATION_STATUS_DROPPED_PROCEDURAL,
            drop_reason="procedural_adjournment", cleaned_text=None,
        )
        for i in range(4)
    ]
    upsert_representations(docs, db_path=db)

    with _validation_settings(tmp_path):
        result = run_cluster_validation(
            run_id="v1", db_path=db,
            embedder=DeterministicHashEmbedder(dimension=16),
            clusterer=_separating_clusterer,
        )

    assert result.n_documents == 10  # the 4 dropped never reach validation


def test_flow_sweeps_every_configured_parameter_set(db, tmp_path):
    upsert_representations(_labelled(6, 6), db_path=db)
    calls: list[int] = []

    def _counting_clusterer(vectors):
        calls.append(vectors.shape[0])
        return _separating_clusterer(vectors)

    with _validation_settings(
        tmp_path, sweep_min_cluster_sizes=[2, 3], sweep_min_samples=[0, 2]
    ):
        result = run_cluster_validation(
            run_id="v1", db_path=db,
            embedder=DeterministicHashEmbedder(dimension=16),
            clusterer=_counting_clusterer,
        )

    assert len(calls) == 4          # 2 sizes x 2 min_samples
    assert len(result.reports) == 4
    assert len(set(calls)) == 1     # embedded once, clustered four times


def test_sweep_expands_zero_min_samples_to_none():
    validation = SimpleNamespace(
        sweep_min_cluster_sizes=[5, 10], sweep_min_samples=[0, 3]
    )

    combos = _sweep_settings(validation)

    assert {c["hdbscan_min_samples"] for c in combos} == {None, 3}
    assert {c["hdbscan_min_cluster_size"] for c in combos} == {5, 10}


def test_flow_handles_an_empty_corpus(db, tmp_path):
    with _validation_settings(tmp_path):
        result = run_cluster_validation(
            run_id="v1", db_path=db,
            embedder=DeterministicHashEmbedder(dimension=16),
            clusterer=_separating_clusterer,
        )

    assert result.n_documents == 0
    assert result.verdict == VERDICT_NOT_EVALUABLE
    assert Path(result.report_path).exists()


def test_flow_writes_no_classification_or_signal_rows(db, tmp_path):
    """Phase 4 evaluates; it must not decide anything about a document."""

    upsert_representations(_labelled(6, 6), db_path=db)

    with _validation_settings(tmp_path):
        run_cluster_validation(
            run_id="v1", db_path=db,
            embedder=DeterministicHashEmbedder(dimension=16),
            clusterer=_separating_clusterer,
        )

    from src.common.db import connection_scope

    with connection_scope(db) as conn:
        tables = {
            r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert "document_classifications" not in tables
    assert "document_domain_signals" not in tables


def test_court_metadata_does_not_affect_validation(db, tmp_path):
    """Court is metadata: changing it must not change the verdict."""

    verdicts = []
    for court in ("Supreme Court of Pakistan", "Lahore High Court", None):
        docs = [
            _doc(f"fam_{i}", "family", FAMILY_TEXT, court=court) for i in range(6)
        ] + [
            _doc(f"crim_{i}", "criminal", CRIMINAL_TEXT, court=court) for i in range(6)
        ]
        local_db = tmp_path / f"court_{court or 'none'}.db".replace(" ", "_")
        init_schema(db_path=local_db, schema_file=str(DEFAULT_REPRESENTATION_SCHEMA_FILE))
        upsert_representations(docs, db_path=local_db)

        with _validation_settings(tmp_path):
            result = run_cluster_validation(
                run_id="v1", db_path=local_db,
                embedder=DeterministicHashEmbedder(dimension=16),
                clusterer=_separating_clusterer,
            )
        verdicts.append((result.verdict, result.best.weighted_purity))

    assert len(set(verdicts)) == 1
