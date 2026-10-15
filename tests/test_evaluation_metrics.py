"""Tests for evaluation/calculate_metrics.py's pure metric functions."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

# Loaded by file path, not by package name: the repo's `src/` is on
# sys.path and already exposes an unrelated top-level `evaluation` package
# (src/evaluation/), which would shadow evaluation/calculate_metrics.py at
# the repo root if imported as `evaluation.calculate_metrics`.
_MODULE_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "calculate_metrics.py"
_spec = importlib.util.spec_from_file_location("eval_calculate_metrics", _MODULE_PATH)
_calculate_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_calculate_metrics)

evaluate = _calculate_metrics.evaluate
load_judged_csv = _calculate_metrics.load_judged_csv
load_qrels = _calculate_metrics.load_qrels
ndcg_at_k = _calculate_metrics.ndcg_at_k
precision_at_k = _calculate_metrics.precision_at_k
recall_at_k = _calculate_metrics.recall_at_k


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------

def test_precision_perfect_ranking():
    p, note = precision_at_k([2, 2, 1, 2, 1], k=5)
    assert p == 1.0
    assert note is None


def test_precision_completely_irrelevant():
    p, note = precision_at_k([0, 0, 0, 0, 0], k=5)
    assert p == 0.0
    assert note is None


def test_precision_relevant_results_below_rank_5_not_counted():
    # relevant items only at ranks 6-7, outside the @5 window -- precision@5 must be 0
    p, note = precision_at_k([0, 0, 0, 0, 0, 2, 2], k=5)
    assert p == 0.0
    assert note is None


def test_precision_always_divides_by_k_even_with_fewer_results():
    # only 3 results exist at all; @5 must still divide by 5, not by 3
    p, note = precision_at_k([2, 1, 0], k=5)
    assert p == pytest.approx(2 / 5)
    assert note is None


def test_precision_missing_label_reports_unavailable_not_irrelevant():
    p, note = precision_at_k([2, None, 1, 0, 2], k=5)
    assert p is None
    assert note is not None
    assert "rank" in note and "2" in note


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------

def test_ndcg_perfect_ranking_is_one():
    n, note = ndcg_at_k([2, 2, 1, 1, 0, 0, 0, 0, 0, 0], k=10)
    assert n == pytest.approx(1.0)
    assert note is None


def test_ndcg_completely_irrelevant_is_zero():
    n, note = ndcg_at_k([0] * 10, k=10)
    assert n == 0.0
    assert note is None


def test_ndcg_graded_relevance_worse_ordering_scores_lower():
    ideal, _ = ndcg_at_k([2, 1, 0], k=3)
    worse, _ = ndcg_at_k([0, 1, 2], k=3)
    assert ideal == pytest.approx(1.0)
    assert worse < ideal


def test_ndcg_missing_label_reports_unavailable():
    n, note = ndcg_at_k([2, 1, None, 0], k=10)
    assert n is None
    assert "rank" in note and "3" in note


def test_ndcg_no_results():
    n, note = ndcg_at_k([], k=10)
    assert n is None
    assert note == "no results retrieved"


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

def test_recall_without_reference_set_is_unavailable():
    r, note = recall_at_k(["a", "b", "c"], None, k=10)
    assert r is None
    assert "reference set" in note


def test_recall_with_complete_reference_set():
    ranked = ["a", "b", "c", "d"]
    relevant = {"a", "c", "z"}  # z is relevant but never retrieved
    r, note = recall_at_k(ranked, relevant, k=10)
    assert r == pytest.approx(2 / 3)
    assert note is None


def test_recall_does_not_treat_top_k_as_the_complete_relevant_set():
    # if the reference set were (wrongly) taken from the top-k itself, recall would trivially be 1.0
    ranked = ["a", "b", "c"]
    relevant = {"a", "b", "c", "d", "e"}
    r, _ = recall_at_k(ranked, relevant, k=10)
    assert r == pytest.approx(3 / 5)


def test_recall_empty_reference_set_is_undefined():
    r, note = recall_at_k(["a", "b"], set(), k=10)
    assert r is None
    assert "undefined" in note


# ---------------------------------------------------------------------------
# evaluate() aggregation, via a small judged CSV + optional qrels
# ---------------------------------------------------------------------------

def _write_judged_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["query_id", "source_type", "rank", "point_id", "chunk_id",
                  "context_header", "canonical_text", "relevance"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            full = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(full)


def test_evaluate_macro_excludes_queries_missing_labels(tmp_path):
    rows = []
    # Q01/case: fully judged, perfect ranking
    for rank, rel in enumerate([2, 2, 1, 1, 0], start=1):
        rows.append({"query_id": "Q01", "source_type": "case", "rank": rank,
                      "chunk_id": f"q1c{rank}", "relevance": rel})
    # Q02/case: one missing label within the @5 window
    for rank, rel in enumerate(["2", "", "1", "0", "0"], start=1):
        rows.append({"query_id": "Q02", "source_type": "case", "rank": rank,
                      "chunk_id": f"q2c{rank}", "relevance": rel})
    csv_path = tmp_path / "judged.csv"
    _write_judged_csv(csv_path, rows)

    judged = load_judged_csv(csv_path)
    result = evaluate(judged, qrels={})

    assert result["per_query"]["Q01"]["case"]["precision_at_5"] == pytest.approx(0.8)
    assert result["per_query"]["Q02"]["case"]["precision_at_5"] is None
    assert result["per_query"]["Q02"]["case"]["precision_at_5_note"] is not None

    macro = result["macro"]["case"]
    # only Q01 contributes to the macro average -- Q02 is excluded, not counted as 0
    assert macro["precision_at_5_n_queries"] == 1
    assert macro["mean_precision_at_5"] == pytest.approx(0.8)
    assert "Q02" in macro["precision_at_5_excluded_queries"]


def test_evaluate_recall_with_and_without_qrels(tmp_path):
    rows = [
        {"query_id": "Q01", "source_type": "statute", "rank": 1, "chunk_id": "s1", "relevance": "2"},
        {"query_id": "Q01", "source_type": "statute", "rank": 2, "chunk_id": "s2", "relevance": "0"},
        {"query_id": "Q02", "source_type": "statute", "rank": 1, "chunk_id": "t1", "relevance": "1"},
    ]
    csv_path = tmp_path / "judged.csv"
    _write_judged_csv(csv_path, rows)
    judged = load_judged_csv(csv_path)

    qrels_path = tmp_path / "qrels.json"
    qrels_path.write_text(json.dumps({"Q01": {"statute": ["s1", "s_outside_topk"]}}), encoding="utf-8")
    qrels = load_qrels(qrels_path)

    result = evaluate(judged, qrels)
    assert result["per_query"]["Q01"]["statute"]["recall_at_10"] == pytest.approx(0.5)
    # Q02 has no qrels entry at all -- must be reported unavailable, never computed from its top-k
    assert result["per_query"]["Q02"]["statute"]["recall_at_10"] is None
    assert result["per_query"]["Q02"]["statute"]["recall_at_10_note"] is not None


def test_evaluate_constitution_rows_never_scored(tmp_path):
    rows = [
        {"query_id": "Q01", "source_type": "constitution", "rank": 1, "chunk_id": "c1", "relevance": "2"},
        {"query_id": "Q01", "source_type": "case", "rank": 1, "chunk_id": "x1", "relevance": "2"},
    ]
    csv_path = tmp_path / "judged.csv"
    _write_judged_csv(csv_path, rows)
    judged = load_judged_csv(csv_path)
    # load_judged_csv already filters constitution out at load time
    assert ("Q01", "constitution") not in judged
    assert ("Q01", "case") in judged


def test_load_judged_csv_rejects_invalid_relevance_value(tmp_path):
    rows = [{"query_id": "Q01", "source_type": "case", "rank": 1, "chunk_id": "x1", "relevance": "maybe"}]
    csv_path = tmp_path / "judged.csv"
    _write_judged_csv(csv_path, rows)
    with pytest.raises(ValueError, match="invalid relevance label"):
        load_judged_csv(csv_path)
