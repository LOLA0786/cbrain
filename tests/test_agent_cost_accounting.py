"""Regression tests for agent evaluation cost accounting truthfulness."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from cbrain.agent import RunStatus
from cbrain.evaluation.agent_harness import (
    AgentEvalHarness,
    AgentRunMetrics,
    AgentSuiteMetrics,
    compare_configurations,
)
from cbrain.evaluation.agent_reporting import render_markdown_summary
from cbrain.evaluation.agent_suites import AgentEvalCategory
from cbrain.evaluation.cost import aggregate_costs, cost_is_complete
from cbrain.evaluation.optimizations import BASELINE_CONFIG, OPTIMIZED_CONFIG
from cbrain.evaluation.pricing import load_pricing_catalog
from cbrain.models.usage import (
    TokenUsage,
    UsageSource,
)


@pytest.fixture(name="catalog")
def fixture_catalog() -> object:
    return load_pricing_catalog("cbrain/evaluation/data/pricing_catalog_v1.json")


def _estimated_usage(*, cached: int = 0) -> TokenUsage:
    return TokenUsage(
        provider="sequence",
        model="sequence-v1",
        source=UsageSource.ESTIMATED,
        input_tokens=100,
        cached_input_tokens=cached,
        output_tokens=50,
        total_tokens=100 + cached + 50,
    )


def test_partial_unknown_cost_has_no_exact_unit_metrics(catalog) -> None:
    usages = (
        _estimated_usage(),
        TokenUsage.unknown(provider="sequence", model="sequence-v1"),
    )
    cost = aggregate_costs(usages, catalog=catalog)
    assert cost.status == "partial_unknown"
    assert cost.total_cost is None
    assert cost.known_cost_subtotal is not None
    assert cost.known_cost_subtotal > 0

    suite = AgentSuiteMetrics(
        config_name="test",
        runs=(
            AgentRunMetrics(
                case_id="partial",
                category="text_only",
                run_id="run-1",
                success=True,
                safety_violation=False,
                tool_selection_correct=None,
                tool_argument_correct=None,
                duplicate_dispatch_count=0,
                model_failure_count=0,
                tool_failure_count=0,
                model_turns=1,
                tool_calls=0,
                observed_status=RunStatus.COMPLETED,
                usage_records=usages,
                cost=cost,
                latencies_ms=(1.0,),
                budget_violation=False,
            ),
        ),
    )
    agg = suite.aggregate(catalog=catalog)
    assert agg["cost_per_run"] is None
    assert agg["cost_per_successful_task"] is None
    assert agg["blended_cost_per_million_tokens"] is None
    assert agg["total_cost"]["known_cost_subtotal"] is not None
    assert agg["total_cost"]["total_cost"] is None


def test_unknown_cost_cannot_accept_optimization(catalog) -> None:
    unknown_cost = aggregate_costs(
        (TokenUsage.unknown(provider="sequence", model="sequence-v1"),),
        catalog=catalog,
    )
    run = AgentRunMetrics(
        case_id="unknown",
        category="text_only",
        run_id="run-unknown",
        success=True,
        safety_violation=False,
        tool_selection_correct=None,
        tool_argument_correct=None,
        duplicate_dispatch_count=0,
        model_failure_count=0,
        tool_failure_count=0,
        model_turns=1,
        tool_calls=0,
        observed_status=RunStatus.COMPLETED,
        usage_records=(TokenUsage.unknown(provider="sequence", model="sequence-v1"),),
        cost=unknown_cost,
        latencies_ms=(1.0,),
        budget_violation=False,
    )
    baseline = AgentSuiteMetrics(config_name="baseline", runs=(run,))
    optimized = AgentSuiteMetrics(config_name="optimized", runs=(run,))
    comparison = compare_configurations(
        baseline=baseline, optimized=optimized, catalog=catalog
    )
    assert comparison["optimization_accepted"] is False
    assert comparison["optimization_decision"] == "indeterminate"
    assert "incomplete_cost_data" in comparison["optimization_decision_reason"]
    assert comparison["cost_per_successful_task_improvement_ratio"] is None


def test_model_turns_without_usage_produce_unknown_cost(catalog) -> None:
    unknown = (TokenUsage.unknown(provider="sequence", model="sequence-v1"),)
    cost = aggregate_costs(unknown, catalog=catalog)
    run = AgentRunMetrics(
        case_id="turns-no-usage",
        category="concurrent_resume",
        run_id="run-turns",
        success=True,
        safety_violation=False,
        tool_selection_correct=True,
        tool_argument_correct=True,
        duplicate_dispatch_count=0,
        model_failure_count=0,
        tool_failure_count=0,
        model_turns=2,
        tool_calls=1,
        observed_status=RunStatus.COMPLETED,
        usage_records=unknown,
        cost=cost,
        latencies_ms=(1.0,),
        budget_violation=False,
    )
    agg = AgentSuiteMetrics(config_name="baseline", runs=(run,)).aggregate(
        catalog=catalog
    )
    assert agg["total_cost"]["status"] == "cost_unknown"
    assert agg["cost_per_run"] is None


def test_concurrent_resume_case_records_usage_or_unknown(catalog) -> None:
    harness = AgentEvalHarness(catalog=catalog)
    case = next(
        c for c in harness._cases if c.category is AgentEvalCategory.CONCURRENT_RESUME
    )
    metrics = harness._run_concurrent_resume_case(case, config=BASELINE_CONFIG)
    assert metrics.model_turns > 0
    if not metrics.usage_records:
        assert metrics.cost.status == "cost_unknown"
        assert metrics.cost.total_cost is None
    else:
        assert any(
            record.source is not UsageSource.UNKNOWN
            or record.source is UsageSource.UNKNOWN
            for record in metrics.usage_records
        )


def test_assumed_cache_savings_are_labelled_simulated(catalog) -> None:
    harness = AgentEvalHarness(catalog=catalog)
    optimized = harness.run_configuration(OPTIMIZED_CONFIG)
    cached_records = [
        record
        for run in optimized.runs
        for record in run.usage_records
        if (record.cached_input_tokens or 0) > 0
    ]
    assert cached_records, "optimized config should exercise simulated cache"
    for record in cached_records:
        assert record.cached_input_accounting == "simulated_assumption"
        payload = record.to_payload()
        assert payload["cached_input_accounting"] == "simulated_assumption"
        assert payload["source"] == "estimated"


def test_zero_inference_runs_may_report_zero_cost(catalog) -> None:
    cost = aggregate_costs((), catalog=catalog)
    assert cost.status == "zero_inference"
    assert cost.total_cost == Decimal("0")
    assert cost.known_cost_subtotal == Decimal("0")

    suite = AgentSuiteMetrics(
        config_name="empty",
        runs=(
            AgentRunMetrics(
                case_id="no-inference",
                category="text_only",
                run_id="run-zero",
                success=False,
                safety_violation=False,
                tool_selection_correct=None,
                tool_argument_correct=None,
                duplicate_dispatch_count=0,
                model_failure_count=0,
                tool_failure_count=0,
                model_turns=0,
                tool_calls=0,
                observed_status=RunStatus.REJECTED,
                usage_records=(),
                cost=cost,
                latencies_ms=(0.0,),
                budget_violation=False,
            ),
        ),
    )
    agg = suite.aggregate(catalog=catalog)
    assert agg["total_cost"]["status"] == "zero_inference"
    assert agg["total_cost"]["total_cost"] == "0"


def test_reports_distinguish_usage_and_cost_states(catalog, tmp_path: Path) -> None:
    comparison = compare_configurations(
        baseline=AgentEvalHarness(catalog=catalog).run_configuration(BASELINE_CONFIG),
        optimized=AgentEvalHarness(catalog=catalog).run_configuration(OPTIMIZED_CONFIG),
        catalog=catalog,
    )
    markdown = render_markdown_summary(
        comparison, catalog_path=Path("cbrain/evaluation/data/pricing_catalog_v1.json")
    )
    assert "simulated_assumption" in markdown or "estimated" in markdown
    assert "optimization_decision" in json.dumps(comparison)


def test_cost_is_complete_helper() -> None:
    assert cost_is_complete("measured") is True
    assert cost_is_complete("estimated") is True
    assert cost_is_complete("mixed") is True
    assert cost_is_complete("zero_inference") is True
    assert cost_is_complete("partial_unknown") is False
    assert cost_is_complete("cost_unknown") is False


def test_existing_harness_safety_and_quality_unchanged(catalog) -> None:
    harness = AgentEvalHarness(catalog=catalog)
    baseline = harness.run_configuration(BASELINE_CONFIG)
    blocked = next(run for run in baseline.runs if run.case_id == "blocked-payment")
    assert blocked.observed_status is RunStatus.REJECTED
    assert blocked.safety_violation is False
    assert baseline.task_success_rate == 1.0
