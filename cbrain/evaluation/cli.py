"""Read-only CLI for inspecting the domain evaluation catalog."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

from cbrain.models import FiveProviderSettings, build_five_provider_router

from .catalog import Scenario, default_catalog
from .model_matrix import (
    FiveModelMatrixRunner,
    ProposalOutcome,
    default_model_tasks,
    default_tool_bindings,
)


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
    else:
        if not parsed.confirm_live_api:
            parser.error("model-generate requires --confirm-live-api")
        router = build_five_provider_router(FiveProviderSettings.from_environment())
        report = FiveModelMatrixRunner(router=router, catalog=catalog).run()
        payload = report.to_payload()
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


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
