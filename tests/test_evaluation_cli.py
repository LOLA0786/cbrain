from __future__ import annotations

import json

import pytest

from cbrain.evaluation.cli import main


def test_catalog_cli_prints_machine_readable_catalog(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(("catalog",)) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "cbrain-domain-catalog/v1"
    assert len(payload["capabilities"]) == 9
    assert len(payload["scenarios"]) == 8


def test_scenario_cli_prints_finance_chain(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(("scenario", "payments.app-fraud-chain")) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "cbrain-domain-scenario/v1"
    assert [step["capability"] for step in payload["steps"]] == [
        "payments.limit.modify",
        "payments.beneficiary.add",
        "payments.transfer.initiate",
    ]


def test_model_plan_cli_contains_five_model_tasks_without_credentials(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(("model-plan",)) == 0

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["schema"] == "cbrain-model-plan/v1"
    assert len(payload["tools"]) == 9
    assert len(payload["tasks"]) == 10
    lowered = output.casefold()
    assert "api_key" not in lowered
    assert "password" not in lowered
    assert "credential" not in lowered


def test_live_model_cli_requires_explicit_cost_confirmation() -> None:
    with pytest.raises(SystemExit):
        main(("model-generate",))
