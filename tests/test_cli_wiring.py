"""The production CLI: does each command reach the phase it claims to?

These tests are about *wiring only*. Every phase function is stubbed, so
nothing embeds, calls an LLM, or touches a database here -- what is being
checked is that:

* all eight case-law commands are exposed and parse their arguments,
* each handler calls the corresponding phase function with the arguments
  the user gave, and no phase logic is reimplemented in the CLI,
* the retired ``classify`` command cannot reach the superseded classifier.

The last one is the point of the exercise: a CLI that silently runs the
wrong classifier is worse than one that has no command at all.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

import scripts.run_pipeline as cli

CASELAW_COMMANDS = [
    "cases",
    "signals",
    "validate-clusters",
    "decide",
    "review-queue",
    "apply-reviews",
    "evaluate",
    "freeze",
]


def _parser_commands() -> dict:
    """Every subcommand the real parser exposes, name -> subparser."""

    captured: dict[str, argparse.ArgumentParser] = {}
    original = argparse.ArgumentParser.parse_args

    def _capture(self, *args, **kwargs):
        captured["parser"] = self
        raise SystemExit(0)  # stop before any command runs

    argparse.ArgumentParser.parse_args = _capture
    try:
        with pytest.raises(SystemExit):
            cli.main()
    finally:
        argparse.ArgumentParser.parse_args = original

    parser = captured["parser"]
    actions = [
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    ]
    assert actions, "the CLI should expose subcommands"
    return dict(actions[0].choices)


# ---------------------------------------------------------------------------
# 1. The commands exist
# ---------------------------------------------------------------------------


def test_every_caselaw_phase_has_a_command():
    commands = _parser_commands()
    missing = [c for c in CASELAW_COMMANDS if c not in commands]
    assert not missing, f"phases unreachable from the CLI: {missing}"


def test_retired_classify_is_still_listed_so_it_fails_loudly():
    """Removing it outright would give a bare argparse error instead."""

    assert "classify" in _parser_commands()


@pytest.mark.parametrize("command", CASELAW_COMMANDS)
def test_each_command_has_a_handler(command):
    commands = _parser_commands()
    assert commands[command].get_default("func") is not None


# ---------------------------------------------------------------------------
# 2. The retired classifier is unreachable
# ---------------------------------------------------------------------------


def test_classify_exits_without_running_the_old_classifier(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.cmd_classify(argparse.Namespace())

    assert exit_info.value.code != 0, "a retired command must not exit 0"
    out = capsys.readouterr().out
    assert "retired" in out.lower()
    assert "decide" in out, "the message should name the replacement"


def test_the_cli_does_not_import_the_superseded_flow():
    """The old flow writes no current state -- it must not be wired here."""

    source = open(cli.__file__).read()
    # Mentions in comments/messages are fine; an import is not.
    imports = [
        line for line in source.splitlines()
        if "import" in line and "classification_flow" in line
        and "domain_classification_flow" not in line
    ]
    assert imports == []


# ---------------------------------------------------------------------------
# 3. Each handler calls its phase with the arguments it was given
# ---------------------------------------------------------------------------


def test_signals_calls_phase_3(monkeypatch):
    seen = {}

    def _fake(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            run_id="sig1", signal_version="v", corpus_size=10, processed=10,
            skipped_already_done=0, n_clusters=2, noise_documents=1,
            noise_share=0.1, stats={},
        )

    monkeypatch.setattr(
        "orchestration.dags.domain_signal_flow.run_domain_signals", _fake
    )
    cli.cmd_signals(argparse.Namespace(
        run_id="sig1", limit=5, batch_size=7, no_llm=True
    ))

    assert seen["run_id"] == "sig1"
    assert seen["limit"] == 5
    assert seen["batch_size"] == 7
    assert seen["llm_enabled"] is False


def test_signals_without_no_llm_leaves_the_configured_default(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "orchestration.dags.domain_signal_flow.run_domain_signals",
        lambda **kw: seen.update(kw) or SimpleNamespace(
            run_id="s", signal_version="v", corpus_size=0, processed=0,
            skipped_already_done=0, n_clusters=0, noise_documents=0,
            noise_share=0.0, stats={},
        ),
    )
    cli.cmd_signals(argparse.Namespace(
        run_id="s", limit=None, batch_size=None, no_llm=False
    ))
    assert seen["llm_enabled"] is None


def test_validate_clusters_calls_phase_4(monkeypatch):
    seen = {}
    report = SimpleNamespace(
        n_clusters=3, noise_share=0.05, weighted_purity=0.9, purity_lift=0.3,
        adjusted_rand_index=0.5, normalized_mutual_info=0.5,
        verdict_reasons=["a reason"],
    )
    monkeypatch.setattr(
        "orchestration.dags.cluster_validation_flow.run_cluster_validation",
        lambda **kw: seen.update(kw) or SimpleNamespace(
            run_id="sig1", n_documents=10, verdict="useful", best=report,
            report_path="var/x.json", cluster_is_a_useful_signal=True,
        ),
    )
    cli.cmd_validate_clusters(argparse.Namespace(
        run_id="sig1", limit=None, output_dir=None
    ))
    assert seen["run_id"] == "sig1"


def test_validate_clusters_tells_you_to_disable_a_useless_signal(monkeypatch, capsys):
    monkeypatch.setattr(
        "orchestration.dags.cluster_validation_flow.run_cluster_validation",
        lambda **kw: SimpleNamespace(
            run_id="sig1", n_documents=10, verdict="not_useful", best=None,
            report_path=None, cluster_is_a_useful_signal=False,
        ),
    )
    cli.cmd_validate_clusters(argparse.Namespace(
        run_id="sig1", limit=None, output_dir=None
    ))
    assert "--no-cluster-signal" in capsys.readouterr().out


def _decision_result(**overrides):
    base = dict(
        run_id="dec1", signal_run_id="sig1", signals_available=10, decided=10,
        skipped_already_done=0, cluster_signal_enabled=True, clusters_profiled=2,
        auto_accepted=8, auto_accept_rate=0.8, needs_review=2, review_rate=0.2,
        dropped_off_domain=0, by_domain={}, by_review_reason={},
        write_outcomes={}, protected_by_review=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_decide_calls_phase_5(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "orchestration.dags.domain_classification_flow.run_domain_classification",
        lambda **kw: seen.update(kw) or _decision_result(),
    )
    cli.cmd_decide(argparse.Namespace(
        run_id="dec1", signal_run_id="sig1", limit=3, batch_size=4,
        no_cluster_signal=False, dry_run=False,
    ))

    assert seen["run_id"] == "dec1"
    assert seen["signal_run_id"] == "sig1"
    # None, not True: the phase consults Phase 4's verdict and fails closed.
    # The CLI must not be able to force the cluster signal on.
    assert seen["cluster_signal_enabled"] is None
    assert seen["write_current_state"] is True


def test_decide_flags_map_to_the_phase_parameters(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "orchestration.dags.domain_classification_flow.run_domain_classification",
        lambda **kw: seen.update(kw) or _decision_result(cluster_signal_enabled=False),
    )
    cli.cmd_decide(argparse.Namespace(
        run_id="dec1", signal_run_id="sig1", limit=None, batch_size=None,
        no_cluster_signal=True, dry_run=True,
    ))

    assert seen["cluster_signal_enabled"] is False
    assert seen["write_current_state"] is False


def test_review_queue_calls_phase_6(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "src.classification.review_store.export_review_queue",
        lambda run_id, **kw: seen.update({"run_id": run_id, **kw}) or {
            "queued": 2, "csv": "q.csv", "jsonl": "q.jsonl",
        },
    )
    cli.cmd_review_queue(argparse.Namespace(
        run_id="dec1", output_dir="/tmp/queue", limit=None
    ))
    assert seen["run_id"] == "dec1"
    assert str(seen["output_dir"]) == "/tmp/queue"


def test_apply_reviews_loads_then_applies(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "src.classification.review_store.load_review_decisions_csv",
        lambda path: calls.append(("load", str(path))) or [
            {"doc_id": "d1", "decision": "human_accepted", "reviewer": "r"}
        ],
    )
    monkeypatch.setattr(
        "src.classification.review_store.apply_review_decisions",
        lambda decisions, **kw: calls.append(("apply", len(decisions), kw))
        or {"recorded": 1, "outcomes": {"updated": 1}},
    )
    cli.cmd_apply_reviews(argparse.Namespace(csv="q.csv", run_id="dec1"))

    assert calls[0] == ("load", "q.csv")
    assert calls[1][0] == "apply" and calls[1][1] == 1
    assert calls[1][2]["decision_run_id"] == "dec1"


def test_apply_reviews_with_no_completed_rows_does_nothing(monkeypatch, capsys):
    monkeypatch.setattr(
        "src.classification.review_store.load_review_decisions_csv", lambda p: []
    )

    def _must_not_run(*a, **k):  # pragma: no cover
        raise AssertionError("nothing should be applied")

    monkeypatch.setattr(
        "src.classification.review_store.apply_review_decisions", _must_not_run
    )
    cli.cmd_apply_reviews(argparse.Namespace(csv="q.csv", run_id="dec1"))
    assert "nothing to apply" in capsys.readouterr().out


def test_apply_reviews_reports_a_bad_decision_value(monkeypatch, capsys):
    """A reviewer's typo must stop the run, not be silently skipped."""

    def _raise(path):
        raise ValueError("unknown decision 'accpeted'")

    monkeypatch.setattr(
        "src.classification.review_store.load_review_decisions_csv", _raise
    )
    with pytest.raises(SystemExit):
        cli.cmd_apply_reviews(argparse.Namespace(csv="q.csv", run_id=None))
    assert "unknown decision" in capsys.readouterr().out


def test_evaluate_calls_phase_7_and_prints_its_own_summary(monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(
        "orchestration.dags.evaluation_flow.run_evaluation",
        lambda **kw: seen.update(kw) or SimpleNamespace(
            report_paths={"text": "e.txt", "json": "e.json"}
        ),
    )
    monkeypatch.setattr(
        "orchestration.dags.evaluation_flow.render_evaluation_summary",
        lambda result: "THE PHASE 7 SUMMARY",
    )
    cli.cmd_evaluate(argparse.Namespace(
        run_id="eval1", decision_run_id="dec1", signal_run_id="sig1",
        labels_file=None, output_dir=None,
    ))

    assert seen["decision_run_id"] == "dec1"
    assert seen["signal_run_id"] == "sig1"
    # The CLI must not rebuild the report itself.
    assert "THE PHASE 7 SUMMARY" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 4. freeze evaluates first, and respects a refusal
# ---------------------------------------------------------------------------


def _readiness(verdict, can_freeze):
    return SimpleNamespace(
        verdict=verdict, answer="because", blockers=[], reservations=[],
        can_freeze=can_freeze,
    )


def test_freeze_runs_the_evaluation_before_snapshotting(monkeypatch):
    order = []
    monkeypatch.setattr(
        "orchestration.dags.evaluation_flow.run_evaluation",
        lambda **kw: order.append("evaluate") or SimpleNamespace(
            verdict="ready", can_freeze=True,
            readiness=_readiness("ready", True),
        ),
    )
    monkeypatch.setattr(
        "src.evaluation.dataset_freeze.freeze_corpus",
        lambda *a, **kw: order.append("freeze") or SimpleNamespace(
            freeze_id="c1", document_count=5, domain_counts={},
            human_reviewed_count=0, human_reviewed_share=0.0,
            taxonomy_version="t@v1", readiness_verdict="ready",
        ),
    )
    cli.cmd_freeze(argparse.Namespace(
        freeze_id="c1", decision_run_id="dec1", signal_run_id="sig1",
        eval_run_id=None, labels_file=None, notes=None, no_reservations=False,
    ))

    assert order == ["evaluate", "freeze"], "the evaluation must come first"


def test_freeze_stops_when_the_verdict_forbids_it(monkeypatch, capsys):
    """The gate lives in freeze_corpus; the CLI must surface its refusal.

    Deliberately not re-checked in the CLI: one source of truth for "may
    this corpus be frozen" is worth more than saving a function call.
    """

    from src.common.exceptions import ConfigurationError

    monkeypatch.setattr(
        "orchestration.dags.evaluation_flow.run_evaluation",
        lambda **kw: SimpleNamespace(
            verdict="not_evaluable", can_freeze=False,
            readiness=_readiness("not_evaluable", False),
        ),
    )

    def _refuse(*a, **kw):
        assert kw["readiness"].can_freeze is False, "the verdict must be passed through"
        raise ConfigurationError("readiness verdict is 'not_evaluable'")

    monkeypatch.setattr("src.evaluation.dataset_freeze.freeze_corpus", _refuse)

    with pytest.raises(SystemExit) as exit_info:
        cli.cmd_freeze(argparse.Namespace(
            freeze_id="c1", decision_run_id="dec1", signal_run_id=None,
            eval_run_id=None, labels_file=None, notes=None,
            no_reservations=False,
        ))
    assert exit_info.value.code != 0
    assert "not_evaluable" in capsys.readouterr().out


def test_freeze_reports_a_refusal_from_the_store(monkeypatch, capsys):
    from src.common.exceptions import ConfigurationError

    monkeypatch.setattr(
        "orchestration.dags.evaluation_flow.run_evaluation",
        lambda **kw: SimpleNamespace(
            verdict="ready", can_freeze=True, readiness=_readiness("ready", True)
        ),
    )

    def _refuse(*a, **k):
        raise ConfigurationError("no accepted documents found")

    monkeypatch.setattr("src.evaluation.dataset_freeze.freeze_corpus", _refuse)

    with pytest.raises(SystemExit):
        cli.cmd_freeze(argparse.Namespace(
            freeze_id="c1", decision_run_id="dec1", signal_run_id=None,
            eval_run_id=None, labels_file=None, notes=None,
            no_reservations=False,
        ))
    assert "no accepted documents" in capsys.readouterr().out


def test_freeze_default_eval_run_id_is_derived_from_the_freeze_id(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "orchestration.dags.evaluation_flow.run_evaluation",
        lambda **kw: seen.update(kw) or SimpleNamespace(
            verdict="ready", can_freeze=True, readiness=_readiness("ready", True)
        ),
    )
    monkeypatch.setattr(
        "src.evaluation.dataset_freeze.freeze_corpus",
        lambda *a, **kw: SimpleNamespace(
            freeze_id="corpus_v1", document_count=1, domain_counts={},
            human_reviewed_count=0, human_reviewed_share=0.0,
            taxonomy_version="t", readiness_verdict="ready",
        ),
    )
    cli.cmd_freeze(argparse.Namespace(
        freeze_id="corpus_v1", decision_run_id="dec1", signal_run_id=None,
        eval_run_id=None, labels_file=None, notes=None, no_reservations=False,
    ))
    assert seen["run_id"] == "eval_corpus_v1"
