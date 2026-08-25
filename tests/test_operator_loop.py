"""Operator-loop approval resume for accounts, legal, and coding agents."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cbrain import ExecutionStatus, GovernedRuntime
from cbrain.company.approval import (
    ApprovalBoundedGateway,
    ApprovalInbox,
    ApprovalInboxError,
)
from cbrain.company.authority import CompanyExecutionContext
from cbrain.company.governance import CompanyRiskGateway
from cbrain.company.handlers import build_handlers
from cbrain.company.kinds import OPERATOR_AGENT_KINDS, CompanyAgentKind
from cbrain.company.profiles import spec_for_kind
from cbrain.company.simulators import load_fixture_bundle
from cbrain.contracts import ActionIntent
from cbrain.evaluation.cli import main
from cbrain.evaluation.company_harness import canonical_action_intent_digest
from cbrain.evaluation.operator_loop import (
    run_operator_loop,
    run_operator_task,
    write_operator_artifacts,
)
from cbrain.evaluation.operator_tasks import (
    OPERATOR_TASKS_PER_AGENT,
    all_operator_tasks,
    operator_tasks_for,
)


def test_operator_suite_has_eight_tasks_per_vertical() -> None:
    tasks = all_operator_tasks()
    assert len(tasks) == 24
    for kind in OPERATOR_AGENT_KINDS:
        assert len(operator_tasks_for(kind)) == OPERATOR_TASKS_PER_AGENT


def test_operator_loop_passes_offline() -> None:
    result = run_operator_loop()
    assert result.passed is True
    assert result.to_payload()["live_provider_calls"] == 0
    assert result.to_payload()["real_model_quality"] == "not_evaluated"
    assert result.to_payload()["decision_authority"] == "company_test_gateway"


def test_review_without_approval_never_runs_handler() -> None:
    task = next(
        item
        for item in operator_tasks_for(CompanyAgentKind.ACCOUNTS)
        if item.task_id.endswith("006")
    )
    record = run_operator_task(task)
    assert record.first_status == ExecutionStatus.REVIEW_REQUIRED.value
    assert record.final_status == ExecutionStatus.REVIEW_REQUIRED.value
    assert record.handler_invocations == 0
    assert record.approved_by is None
    assert record.success is True


def test_named_approval_executes_same_intent_once() -> None:
    task = next(
        item
        for item in operator_tasks_for(CompanyAgentKind.ACCOUNTS)
        if item.require_approval
    )
    spec = spec_for_kind(task.agent_kind)
    bundle = load_fixture_bundle("default")
    inbox = ApprovalInbox()
    inner = CompanyRiskGateway(spec, bundle=bundle)
    gateway = ApprovalBoundedGateway(
        inner, inbox, digest_for=canonical_action_intent_digest
    )
    runtime = GovernedRuntime(gateway)
    handlers = build_handlers(kind=task.agent_kind, bundle=bundle)
    calls = {"count": 0}

    def counted(arguments: dict[str, object]) -> object:
        calls["count"] += 1
        return handlers[task.tool](arguments)

    action = ActionIntent.capture(
        agent_id=spec.agent_id,
        framework="test",
        tool_name=task.tool,
        capability=spec.tools.get(task.tool).capability,
        arguments=dict(task.arguments),
        request_id="operator-same-intent",
    )
    first = runtime.execute(action, counted)
    assert first.status is ExecutionStatus.REVIEW_REQUIRED
    assert calls["count"] == 0
    inbox.approve(action.request_id, actor_id="controller-1")
    second = runtime.execute(action, counted)
    assert second.status is ExecutionStatus.EXECUTED
    assert calls["count"] == 1
    third = runtime.execute(action, counted)
    assert third.status is ExecutionStatus.BLOCKED
    assert calls["count"] == 1
    assert third.reason == "approval_already_consumed"


def test_frozen_inbox_refuses_approval_and_dispatch() -> None:
    task = next(
        item
        for item in operator_tasks_for(CompanyAgentKind.LEGAL)
        if item.require_approval
    )
    record = run_operator_task(task, freeze_before_approve=True)
    assert record.frozen is True
    assert record.handler_invocations == 0
    assert record.success is False
    assert record.final_status == ExecutionStatus.CONTROL_FAILURE.value


def test_approve_without_parked_action_fails() -> None:
    inbox = ApprovalInbox()
    with pytest.raises(ApprovalInboxError, match="no parked action"):
        inbox.approve("missing", actor_id="controller-1")


def test_coding_block_tools_never_mutate_workspace() -> None:
    bundle = load_fixture_bundle("default")
    before = bundle.coding.snapshot()
    for task in operator_tasks_for(CompanyAgentKind.CODING):
        if task.expected_decision.value != "block":
            continue
        record = run_operator_task(task)
        assert record.handler_invocations == 0
        assert record.final_status == ExecutionStatus.BLOCKED.value
    after = load_fixture_bundle("default").coding.snapshot()
    assert after["files"] == before["files"]
    assert after["pull_requests"] == []


def test_evidence_omits_arguments_and_cli_writes_artifacts(tmp_path: Path) -> None:
    result = run_operator_loop(CompanyAgentKind.CODING)
    write_operator_artifacts(tmp_path, result)
    line = (tmp_path / "evidence.jsonl").read_text(encoding="utf-8").splitlines()[0]
    payload = json.loads(line)
    assert "arguments" not in payload
    assert "API_KEY" not in json.dumps(payload)
    assert payload["schema"] == "cbrain-operator-evidence/v1"
    assert main(("operator-plan", "--agent", "all")) == 0
    assert (
        main(
            (
                "operator-run",
                "--agent",
                "coding",
                "--output-dir",
                str(tmp_path / "cli"),
            )
        )
        == 0
    )


def test_legal_commitment_stays_review_without_approval() -> None:
    spec = spec_for_kind(CompanyAgentKind.LEGAL)
    bundle = load_fixture_bundle("default")
    context = CompanyExecutionContext(
        principal_id="legal-operator",
        agent_kind=CompanyAgentKind.LEGAL,
        permitted_matter_ids=frozenset({"matter-a"}),
        authorization_scope_id="scope-legal",
    )
    gateway = CompanyRiskGateway(spec, context=context, bundle=bundle)
    handlers = build_handlers(
        kind=CompanyAgentKind.LEGAL, bundle=bundle, context=context
    )
    action = ActionIntent.capture(
        agent_id=spec.agent_id,
        framework="test",
        tool_name="send_commitment",
        capability=spec.tools.get("send_commitment").capability,
        arguments={"matter_id": "matter-a", "text": "We agree."},
    )
    execution = gateway.decide_and_execute(action, handlers["send_commitment"])
    assert execution.status is ExecutionStatus.REVIEW_REQUIRED
    assert execution.tool_executed is False
