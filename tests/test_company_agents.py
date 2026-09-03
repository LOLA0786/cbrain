"""Tests for configuration-driven company agents and offline evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cbrain.agent import ToolRegistry, ToolRegistryError
from cbrain.company.handlers import build_handlers
from cbrain.company.kinds import MATRIX_AGENT_KINDS, CompanyAgentKind
from cbrain.company.profiles import all_company_profiles, spec_for_kind
from cbrain.company.risk import ToolRiskLevel
from cbrain.company.simulators import SimulatorValidationError, load_fixture_bundle
from cbrain.company.spec import SCENARIO_SUITE_VERSION
from cbrain.evaluation.cli import main
from cbrain.evaluation.company_gates import evaluate_release_gates
from cbrain.evaluation.company_gateway import CompanyRiskGateway
from cbrain.evaluation.company_harness import CompanyEvalHarness
from cbrain.evaluation.company_scenarios import (
    SCENARIOS_PER_AGENT,
    CompanyScenarioCategory,
    validate_suite,
)
from cbrain.evaluation.company_suites import (
    all_company_eval_cases,
    cases_for_agent,
)
from cbrain.evaluation.pricing import default_pricing_catalog_path, load_pricing_catalog


@pytest.fixture(name="pricing")
def fixture_pricing():
    return load_pricing_catalog(default_pricing_catalog_path())


def test_six_profiles_share_foundation_contract() -> None:
    specs = all_company_profiles()
    assert len(specs) == 6
    ids = {spec.agent_id for spec in specs}
    assert len(ids) == 6


def test_tool_allowlists_are_isolated() -> None:
    gtm = spec_for_kind(CompanyAgentKind.GTM).profile.permitted_tools
    ops = spec_for_kind(CompanyAgentKind.OPERATIONS).profile.permitted_tools
    legal = spec_for_kind(CompanyAgentKind.LEGAL).profile.permitted_tools
    accounts = spec_for_kind(CompanyAgentKind.ACCOUNTS).profile.permitted_tools
    coding = spec_for_kind(CompanyAgentKind.CODING).profile.permitted_tools
    procurement = spec_for_kind(CompanyAgentKind.PROCUREMENT).profile.permitted_tools
    assert gtm.isdisjoint(ops)
    assert gtm.isdisjoint(legal)
    assert gtm.isdisjoint(accounts)
    assert gtm.isdisjoint(coding)
    assert gtm.isdisjoint(procurement)
    assert ops.isdisjoint(accounts)
    assert legal.isdisjoint(coding)
    assert accounts.isdisjoint(coding)
    assert accounts.isdisjoint(procurement)
    assert coding.isdisjoint(procurement)


def test_review_and_block_never_execute_handlers() -> None:
    spec = spec_for_kind(CompanyAgentKind.GTM)
    bundle = load_fixture_bundle("default")
    gateway = CompanyRiskGateway(spec)
    handlers = build_handlers(kind=CompanyAgentKind.GTM, bundle=bundle)
    from cbrain.contracts import ActionIntent

    gateway = CompanyRiskGateway(spec)
    for tool, risk in spec.tool_risks.items():
        if risk is ToolRiskLevel.ALLOW:
            continue
        action = ActionIntent.capture(
            agent_id=spec.agent_id,
            framework="test",
            tool_name=tool,
            capability=spec.tools.get(tool).capability,
            arguments=_sample_arguments(tool),
        )
        execution = gateway.decide_and_execute(action, handlers[tool])
        assert execution.tool_executed is False
        assert tool not in gateway.handler_calls


def test_scenario_suite_has_two_hundred_cases() -> None:
    cases = all_company_eval_cases()
    validate_suite(cases)
    assert len(cases) == 200
    for kind in MATRIX_AGENT_KINDS:
        agent_cases = cases_for_agent(kind)
        assert len(agent_cases) == SCENARIOS_PER_AGENT


def test_scenario_category_counts() -> None:
    expected = {
        CompanyScenarioCategory.NORMAL: 20,
        CompanyScenarioCategory.EDGE: 10,
        CompanyScenarioCategory.ADVERSARIAL: 10,
        CompanyScenarioCategory.AUTHORIZATION: 5,
        CompanyScenarioCategory.CRASH_CONCURRENCY_COST: 5,
    }
    for kind in MATRIX_AGENT_KINDS:
        agent_cases = cases_for_agent(kind)
        for category, count in expected.items():
            actual = sum(1 for case in agent_cases if case.category is category)
            assert actual == count


def test_malformed_fixture_rejected() -> None:
    with pytest.raises(SimulatorValidationError):
        load_fixture_bundle("unknown-fixture")


def test_offline_company_run_keeps_safety_gates_and_zero_route_divergence(
    pricing,
) -> None:
    harness = CompanyEvalHarness(
        catalog=pricing,
        cases=all_company_eval_cases(),
        model_routes=("offline",),
    )
    metrics = harness.run()
    gates = evaluate_release_gates(metrics)
    assert gates.metrics["unauthorized_executions"] == 0
    assert gates.metrics["approval_bypasses"] == 0
    assert gates.metrics["safety_violations"] == 0
    assert gates.metrics["duplicate_dispatches"] == 0
    assert metrics.decision_divergence_count == 0
    assert gates.metrics["decision_divergence_count"] == 0
    assert metrics.route_invariance.incomplete_comparison_count == 0
    assert gates.metrics["incomplete_route_comparison_count"] == 0
    assert gates.passed is True


def test_company_plan_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(("company-plan", "--agent", "all")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_cases"] == 200
    assert payload["scenario_suite_version"] == SCENARIO_SUITE_VERSION


def test_company_run_cli_writes_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    exit_code = main(
        (
            "company-run",
            "--agent",
            "gtm",
            "--output-dir",
            str(output_dir),
            "--pricing-catalog",
            str(default_pricing_catalog_path()),
        )
    )
    assert exit_code == 0
    for name in (
        "runs.jsonl",
        "aggregate_report.json",
        "comparison.csv",
        "summary.md",
        "manifest.json",
    ):
        assert (output_dir / name).exists()


def test_company_run_artifacts_are_deterministic(tmp_path: Path) -> None:
    dirs = []
    for index in range(2):
        output_dir = tmp_path / f"run-{index}"
        assert (
            main(
                (
                    "company-run",
                    "--agent",
                    "legal",
                    "--output-dir",
                    str(output_dir),
                )
            )
            == 0
        )
        dirs.append(output_dir)

    def digest(directory: Path) -> str:
        parts: list[bytes] = []
        runs = []
        for line in (directory / "runs.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            payload.pop("latencies_ms", None)
            runs.append(payload)
        parts.append(json.dumps(runs, sort_keys=True).encode("utf-8"))
        aggregate = json.loads(
            (directory / "aggregate_report.json").read_text(encoding="utf-8")
        )
        agg = aggregate.get("aggregate", {})
        agg.pop("average_latency_ms", None)
        agg.pop("p95_latency_ms", None)
        for run in agg.get("runs", []):
            run.pop("latencies_ms", None)
        parts.append(json.dumps(aggregate, sort_keys=True).encode("utf-8"))
        parts.append((directory / "comparison.csv").read_bytes())
        parts.append((directory / "summary.md").read_bytes())
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        manifest.pop("cbrain_commit", None)
        parts.append(json.dumps(manifest, sort_keys=True).encode("utf-8"))
        return hashlib.sha256(b"".join(parts)).hexdigest()

    assert digest(dirs[0]) == digest(dirs[1])


def test_accounts_decimal_validation() -> None:
    bundle = load_fixture_bundle("default")
    handlers = build_handlers(kind=CompanyAgentKind.ACCOUNTS, bundle=bundle)
    with pytest.raises(SimulatorValidationError):
        handlers["prepare_payment"](
            {
                "invoice_id": "inv-1",
                "amount_minor": "NaN",
                "currency": "USD",
            }
        )


def test_forge_tool_name_fails_closed() -> None:
    spec = spec_for_kind(CompanyAgentKind.GTM)
    registry = ToolRegistry(list(spec.tools._tools.values()))
    with pytest.raises(ToolRegistryError):
        registry.get("execute_payment")


def _sample_arguments(tool: str) -> dict[str, object]:
    samples: dict[str, dict[str, object]] = {
        "send_email": {"recipient": "a@b.com", "subject": "s", "body": "b"},
        "bulk_campaign": {"campaign_id": "c1", "recipients": ["a@b.com"]},
        "delete_lead": {"lead_id": "lead-1"},
        "propose_restart": {"service_id": "svc-api"},
        "propose_infra_change": {"change_id": "c1", "description": "d"},
        "delete_data": {"resource_id": "r1"},
        "sign_contract": {"contract_id": "ctr-v1"},
        "file_document": {"document_id": "d1", "court": "US"},
        "send_commitment": {"matter_id": "matter-a", "text": "t"},
        "provide_legal_advice": {"matter_id": "matter-a", "conclusion": "c"},
        "prepare_email": {"recipient": "a@b.com", "subject": "s", "body": "b"},
        "execute_payment": {
            "invoice_id": "inv-1",
            "amount_minor": "10000",
            "currency": "USD",
            "beneficiary_id": "vendor-1",
        },
        "change_bank_details": {"vendor_id": "vendor-1", "account_ref": "bank"},
        "change_vendor": {"vendor_id": "vendor-1", "new_name": "n"},
        "prepare_payment": {
            "invoice_id": "inv-1",
            "amount_minor": "10000",
            "currency": "USD",
        },
        "award_quote": {
            "rfq_id": "rfq-steel-1",
            "quote_id": "quote-oracle-1",
            "amount_minor": "10000",
            "currency": "USD",
        },
        "create_purchase_requisition": {
            "rfq_id": "rfq-steel-1",
            "quote_id": "quote-oracle-1",
            "material_id": "mat-steel-rod",
            "quantity": 10,
            "amount_minor": "10000",
            "currency": "USD",
        },
        "release_purchase_order": {
            "pr_id": "pr-1",
            "amount_minor": "10000",
            "currency": "USD",
        },
        "change_vendor_bank": {"vendor_id": "vendor-oracle-1", "account_ref": "bank"},
        "post_erp_payment": {
            "vendor_id": "vendor-oracle-1",
            "amount_minor": "10000",
            "currency": "USD",
        },
    }
    return samples.get(tool, {"query": "q"})
