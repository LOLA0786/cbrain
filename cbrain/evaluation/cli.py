"""Read-only CLI for inspecting the domain evaluation catalog."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cbrain.company.kinds import CompanyAgentKind
from cbrain.models import FiveProviderSettings, build_five_provider_router

from .agent_harness import AgentEvalHarness, compare_configurations
from .agent_reporting import write_agent_eval_artifacts
from .agent_suites import default_agent_eval_cases, offline_agent_eval_cases
from .catalog import Scenario, default_catalog
from .company_gates import evaluate_release_gates
from .company_harness import CompanyEvalHarness
from .company_reporting import write_company_eval_artifacts
from .company_suites import cases_for_agent, plan_payload
from .model_matrix import (
    FiveModelMatrixRunner,
    ProposalOutcome,
    default_model_tasks,
    default_tool_bindings,
)
from .operator_loop import (
    load_offline_eval_identities,
    run_operator_loop,
    write_operator_artifacts,
)
from .operator_tasks import operator_plan_payload
from .optimizations import BASELINE_CONFIG, OPTIMIZED_CONFIG
from .pricing import default_pricing_catalog_path, load_pricing_catalog


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cbrain-eval",
        description="Inspect CBrain's deterministic domain evaluation catalog.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("catalog", help="Print the complete catalog as JSON")
    scenario = subcommands.add_parser("scenario", help="Print one scenario as JSON")
    scenario.add_argument("scenario_id")
    subcommands.add_parser(
        "model-plan",
        help="Print credential-free tools and five-model tasks as JSON",
    )
    generate = subcommands.add_parser(
        "model-generate",
        help="Call all five configured model providers and print proposals",
    )
    generate.add_argument(
        "--confirm-live-api",
        action="store_true",
        help="Confirm that live, potentially billable provider APIs may be called",
    )
    subcommands.add_parser(
        "agent-plan",
        help="Print offline agent evaluation suites as JSON",
    )
    agent_run = subcommands.add_parser(
        "agent-run",
        help="Run offline baseline and optimized agent evaluation harness",
    )
    agent_run.add_argument(
        "--output-dir",
        default="agent-eval-output",
        help="Directory for JSONL, JSON, CSV, Markdown, and manifest artifacts",
    )
    agent_run.add_argument(
        "--pricing-catalog",
        default=str(default_pricing_catalog_path()),
        help="Path to replaceable pricing catalog JSON",
    )
    company_plan = subcommands.add_parser(
        "company-plan",
        help="Print offline company-agent scenario counts and categories",
    )
    company_plan.add_argument(
        "--agent",
        default="all",
        choices=("all", "gtm", "operations", "legal", "accounts"),
    )
    operator_plan = subcommands.add_parser(
        "operator-plan",
        help="Print accounts/legal/coding operator-loop tasks",
    )
    operator_plan.add_argument(
        "--agent",
        default="all",
        choices=("all", "accounts", "legal", "coding"),
    )
    operator_run = subcommands.add_parser(
        "operator-run",
        help="Run offline operator loop with named approval resume",
    )
    operator_run.add_argument(
        "--agent",
        default="all",
        choices=("all", "accounts", "legal", "coding"),
    )
    operator_run.add_argument(
        "--output-dir",
        default="operator-loop-output",
        help="Directory for operator evidence JSONL and suite JSON",
    )
    company_run = subcommands.add_parser(
        "company-run",
        help="Run offline company-agent evaluation suites",
    )
    company_run.add_argument(
        "--agent",
        default="all",
        choices=("all", "gtm", "operations", "legal", "accounts"),
    )
    company_run.add_argument(
        "--output-dir",
        default="company-eval-output",
        help="Directory for JSONL, JSON, CSV, Markdown, and manifest artifacts",
    )
    company_run.add_argument(
        "--pricing-catalog",
        default=str(default_pricing_catalog_path()),
        help="Path to replaceable pricing catalog JSON",
    )
    parsed = parser.parse_args(arguments)
    catalog = default_catalog()
    exit_code = 0
    if parsed.command == "catalog":
        payload = catalog.to_payload()
    elif parsed.command == "scenario":
        scenario_value = catalog.scenario(parsed.scenario_id)
        payload = _scenario_payload(scenario_value)
    elif parsed.command == "model-plan":
        payload = _model_plan_payload()
    elif parsed.command == "agent-plan":
        payload = _agent_plan_payload()
    elif parsed.command == "agent-run":
        pricing_path = Path(parsed.pricing_catalog)
        pricing = load_pricing_catalog(pricing_path)
        harness = AgentEvalHarness(catalog=pricing, cases=offline_agent_eval_cases())
        baseline = harness.run_configuration(BASELINE_CONFIG)
        optimized = harness.run_configuration(OPTIMIZED_CONFIG)
        comparison = compare_configurations(
            baseline=baseline,
            optimized=optimized,
            catalog=pricing,
        )
        write_agent_eval_artifacts(
            output_dir=parsed.output_dir,
            comparison=comparison,
            catalog=pricing,
            catalog_path=pricing_path,
            baseline=baseline,
            optimized=optimized,
        )
        payload = comparison
        if not comparison["optimization_accepted"]:
            exit_code = 1
    elif parsed.command == "company-plan":
        payload = plan_payload(_parse_company_agent(parsed.agent))
    elif parsed.command == "company-run":
        pricing_path = Path(parsed.pricing_catalog)
        pricing = load_pricing_catalog(pricing_path)
        kind = _parse_company_agent(parsed.agent)
        cases = cases_for_agent(kind)
        metrics = CompanyEvalHarness(catalog=pricing, cases=cases).run()
        aggregate = metrics.aggregate(catalog=pricing)
        gates = evaluate_release_gates(metrics)
        write_company_eval_artifacts(
            output_dir=parsed.output_dir,
            metrics=metrics,
            aggregate=aggregate,
            gates=gates,
            catalog=pricing,
            catalog_path=pricing_path,
        )
        payload = {"aggregate": aggregate, "release_gates": gates.to_payload()}
        if not gates.passed:
            exit_code = 1
    elif parsed.command == "operator-plan":
        payload = operator_plan_payload(_parse_operator_agent(parsed.agent))
    elif parsed.command == "operator-run":
        allowed, principals, max_ttl = load_offline_eval_identities(
            Path(__file__).resolve().parents[2] / "tests" / "operator_fixtures.py"
        )
        result = run_operator_loop(
            _parse_operator_agent(parsed.agent),
            allowed_approvers=allowed,
            principal_by_agent=principals,
            max_ttl_seconds=max_ttl,
        )
        write_operator_artifacts(parsed.output_dir, result)
        payload = result.to_payload()
        if not result.passed:
            exit_code = 1
    else:
        if not parsed.confirm_live_api:
            parser.error("model-generate requires --confirm-live-api")
        router = build_five_provider_router(FiveProviderSettings.from_environment())
        report = FiveModelMatrixRunner(router=router, catalog=catalog).run()
        payload = report.to_payload()
        payload = {
            **payload,
            "determinism": "nondeterministic_live_provider",
        }
        if any(
            item.outcome is ProposalOutcome.CONTROL_FAILURE for item in report.attempts
        ):
            exit_code = 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code


def _scenario_payload(scenario: Scenario) -> dict[str, Any]:
    return {
        "schema": "cbrain-domain-scenario/v1",
        "scenario_id": scenario.scenario_id,
        "title": scenario.title,
        "kind": scenario.kind.value,
        "steps": [
            {
                "step_id": step.step_id,
                "capability": step.capability,
                "arguments": step.arguments,
                "repeat": step.repeat,
                "fault": step.fault.value,
            }
            for step in scenario.steps
        ],
    }


def _model_plan_payload() -> dict[str, Any]:
    return {
        "schema": "cbrain-model-plan/v1",
        "tools": [
            {
                "capability": binding.capability,
                "name": binding.tool.name,
                "description": binding.tool.description,
                "input_schema": binding.tool.input_schema,
            }
            for binding in default_tool_bindings()
        ],
        "tasks": [
            {
                "task_id": task.task_id,
                "scenario_id": task.scenario_id,
                "step_id": task.step_id,
                "prompt": task.prompt,
            }
            for task in default_model_tasks()
        ],
    }


def _agent_plan_payload() -> dict[str, Any]:
    return {
        "schema": "cbrain-agent-eval-plan/v1",
        "determinism": "offline_fixture",
        "cases": [
            {
                "case_id": case.case_id,
                "category": case.category.value,
                "title": case.title,
                "task": case.task,
                "expected_status": case.expected_status.value,
                "safety_sensitive": case.safety_sensitive,
                "requires_durable_store": case.requires_durable_store,
            }
            for case in default_agent_eval_cases()
        ],
    }


def _parse_operator_agent(value: str) -> CompanyAgentKind | None:
    if value == "all":
        return None
    return CompanyAgentKind(value)


def _parse_company_agent(value: str) -> CompanyAgentKind | None:
    if value == "all":
        return None
    return CompanyAgentKind(value)


if __name__ == "__main__":
    raise SystemExit(main())
