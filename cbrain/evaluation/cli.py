"""Read-only CLI for inspecting the domain evaluation catalog."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

from .catalog import Scenario, default_catalog


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cbrain-eval",
        description="Inspect CBrain's deterministic domain evaluation catalog.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("catalog", help="Print the complete catalog as JSON")
    scenario = subcommands.add_parser("scenario", help="Print one scenario as JSON")
    scenario.add_argument("scenario_id")
    parsed = parser.parse_args(arguments)
    catalog = default_catalog()
    if parsed.command == "catalog":
        payload = catalog.to_payload()
    else:
        scenario_value = catalog.scenario(parsed.scenario_id)
        payload = _scenario_payload(scenario_value)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
