"""Tests for offline agent evaluation harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cbrain.agent import RunStatus
from cbrain.evaluation.agent_harness import AgentEvalHarness, compare_configurations
from cbrain.evaluation.agent_suites import AgentEvalCategory, offline_agent_eval_cases
from cbrain.evaluation.cli import main
from cbrain.evaluation.optimizations import BASELINE_CONFIG, OPTIMIZED_CONFIG
from cbrain.evaluation.pricing import default_pricing_catalog_path, load_pricing_catalog


@pytest.fixture(name="pricing")
def fixture_pricing():
    return load_pricing_catalog(default_pricing_catalog_path())


def test_offline_suite_has_required_categories(pricing) -> None:
    cases = offline_agent_eval_cases()
    categories = {case.category for case in cases}
    for required in AgentEvalCategory:
        assert required in categories


def test_baseline_and_optimized_runs_complete(pricing) -> None:
    harness = AgentEvalHarness(catalog=pricing, cases=offline_agent_eval_cases())
    baseline = harness.run_configuration(BASELINE_CONFIG)
    optimized = harness.run_configuration(OPTIMIZED_CONFIG)
    assert len(baseline.runs) == len(offline_agent_eval_cases())
    assert len(optimized.runs) == len(offline_agent_eval_cases())


def test_compare_reports_cost_per_successful_task(pricing) -> None:
    harness = AgentEvalHarness(catalog=pricing, cases=offline_agent_eval_cases())
    comparison = compare_configurations(
        baseline=harness.run_configuration(BASELINE_CONFIG),
        optimized=harness.run_configuration(OPTIMIZED_CONFIG),
        catalog=pricing,
    )
    assert comparison["baseline"]["cost_per_successful_task"] is not None
    assert comparison["baseline"]["task_success_rate"] >= 0.0


def test_blocked_case_has_no_safety_violation_when_rejected(pricing) -> None:
    harness = AgentEvalHarness(catalog=pricing, cases=offline_agent_eval_cases())
    baseline = harness.run_configuration(BASELINE_CONFIG)
    blocked = next(run for run in baseline.runs if run.case_id == "blocked-payment")
    assert blocked.observed_status is RunStatus.REJECTED
    assert blocked.safety_violation is False


def test_agent_run_cli_writes_artifacts(tmp_path: Path, pricing) -> None:
    output_dir = tmp_path / "out"
    exit_code = main(
        (
            "agent-run",
            "--output-dir",
            str(output_dir),
            "--pricing-catalog",
            str(default_pricing_catalog_path()),
        )
    )
    assert exit_code in {0, 1}
    assert (output_dir / "runs.jsonl").exists()
    assert (output_dir / "aggregate_report.json").exists()
    assert (output_dir / "comparison.csv").exists()
    assert (output_dir / "summary.md").exists()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pinned_components"]["gbrain_version"]


def test_agent_plan_cli_lists_offline_cases(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(("agent-plan",)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["determinism"] == "offline_fixture"
    assert len(payload["cases"]) >= 10
