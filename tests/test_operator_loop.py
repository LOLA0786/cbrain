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
    ApprovalPrincipal,
    ApprovalRole,
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
    APPROVER_DIRECTORY,
    CODE_REVIEWER_PRINCIPAL,
    CONTROLLER_PRINCIPAL,
    COUNSEL_PRINCIPAL,
    run_operator_loop,
    run_operator_task,
    write_operator_artifacts,
)
from cbrain.evaluation.operator_tasks import (
    OPERATOR_TASKS_PER_AGENT,
    all_operator_tasks,
    operator_tasks_for,
)


def test_operator_suite_has_nine_tasks_per_vertical() -> None:
    tasks = all_operator_tasks()
    assert len(tasks) == 27
    assert sum(task.freeze_before_approval for task in tasks) == 3
    for kind in OPERATOR_AGENT_KINDS:
        assert len(operator_tasks_for(kind)) == OPERATOR_TASKS_PER_AGENT


def test_operator_loop_passes_offline_with_explicit_frozen_evidence() -> None:
    result = run_operator_loop()
    assert result.passed is True
    assert result.to_payload()["live_provider_calls"] == 0
    assert result.to_payload()["real_model_quality"] == "not_evaluated"
    assert result.to_payload()["decision_authority"] == "company_test_gateway"
    frozen = [record for record in result.records if record.frozen]
    assert len(frozen) == 3
    assert {record.agent_kind for record in frozen} == {
        "accounts",
        "legal",
        "coding",
    }
    assert all(
        record.final_status == ExecutionStatus.CONTROL_FAILURE for record in frozen
    )
    assert all(record.tool_executed is False for record in frozen)
    assert all(record.retryable is False for record in frozen)
    assert all(record.handler_invocations == 0 for record in frozen)


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


def test_role_bound_approval_executes_same_intent_once() -> None:
    task = next(
        item
        for item in operator_tasks_for(CompanyAgentKind.ACCOUNTS)
        if item.require_approval and not item.freeze_before_approval
    )
    spec = spec_for_kind(task.agent_kind)
    bundle = load_fixture_bundle("default")
    inbox = ApprovalInbox(allowed_approvers=APPROVER_DIRECTORY)
    inner = CompanyRiskGateway(spec, bundle=bundle)
    gateway = ApprovalBoundedGateway(
        inner,
        inbox,
        digest_for=canonical_action_intent_digest,
        required_role_for=lambda _: ApprovalRole.CONTROLLER,
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
    inbox.approve(action.request_id, principal=CONTROLLER_PRINCIPAL)
    second = runtime.execute(action, counted)
    assert second.status is ExecutionStatus.EXECUTED
    assert calls["count"] == 1
    third = runtime.execute(action, counted)
    assert third.status is ExecutionStatus.BLOCKED
    assert calls["count"] == 1
    assert third.reason == "approval_already_consumed"


def test_wrong_role_unknown_actor_and_duplicate_approval_fail_closed() -> None:
    task = next(
        item
        for item in operator_tasks_for(CompanyAgentKind.ACCOUNTS)
        if item.require_approval and not item.freeze_before_approval
    )
    spec = spec_for_kind(task.agent_kind)
    bundle = load_fixture_bundle("default")
    inbox = ApprovalInbox(allowed_approvers=APPROVER_DIRECTORY)
    inner = CompanyRiskGateway(spec, bundle=bundle)
    gateway = ApprovalBoundedGateway(
        inner,
        inbox,
        digest_for=canonical_action_intent_digest,
        required_role_for=lambda _: ApprovalRole.CONTROLLER,
    )
    runtime = GovernedRuntime(gateway)
    calls = {"count": 0}

    def counted(arguments: dict[str, object]) -> object:
        calls["count"] += 1
        return arguments

    action = ActionIntent.capture(
        agent_id=spec.agent_id,
        framework="test",
        tool_name=task.tool,
        capability=spec.tools.get(task.tool).capability,
        arguments=dict(task.arguments),
        request_id="operator-role-bound",
    )
    assert runtime.execute(action, counted).status is ExecutionStatus.REVIEW_REQUIRED
    with pytest.raises(ApprovalInboxError, match="role cannot approve"):
        inbox.approve(action.request_id, principal=COUNSEL_PRINCIPAL)
    with pytest.raises(ApprovalInboxError, match="not an allowed approver"):
        inbox.approve(
            action.request_id,
            principal=ApprovalPrincipal(
                actor_id="unknown-controller",
                role=ApprovalRole.CONTROLLER,
            ),
        )
    inbox.approve(action.request_id, principal=CONTROLLER_PRINCIPAL)
    with pytest.raises(ApprovalInboxError, match="already recorded"):
        inbox.approve(action.request_id, principal=CONTROLLER_PRINCIPAL)
    assert calls["count"] == 0


@pytest.mark.parametrize(
    ("kind", "expected_role", "expected_actor"),
    (
        (
            CompanyAgentKind.ACCOUNTS,
            ApprovalRole.CONTROLLER,
            CONTROLLER_PRINCIPAL.actor_id,
        ),
        (CompanyAgentKind.LEGAL, ApprovalRole.COUNSEL, COUNSEL_PRINCIPAL.actor_id),
        (
            CompanyAgentKind.CODING,
            ApprovalRole.CODE_REVIEWER,
            CODE_REVIEWER_PRINCIPAL.actor_id,
        ),
    ),
)
def test_each_vertical_uses_its_own_approver_role(
    kind: CompanyAgentKind,
    expected_role: ApprovalRole,
    expected_actor: str,
) -> None:
    record = next(
        item for item in run_operator_loop(kind).records if item.approved_by is not None
    )
    assert record.required_approval_role == expected_role.value
    assert record.approved_role == expected_role.value
    assert record.approved_by == expected_actor


def test_frozen_inbox_refuses_approval_and_dispatch_as_expected_safety_case() -> None:
    for kind in OPERATOR_AGENT_KINDS:
        task = next(
            item for item in operator_tasks_for(kind) if item.freeze_before_approval
        )
        record = run_operator_task(task)
        assert record.frozen is True
        assert record.handler_invocations == 0
        assert record.success is True
        assert record.final_status == ExecutionStatus.CONTROL_FAILURE.value
        assert record.tool_executed is False
        assert record.retryable is False


def test_approve_without_parked_action_fails() -> None:
    inbox = ApprovalInbox(allowed_approvers=APPROVER_DIRECTORY)
    with pytest.raises(ApprovalInboxError, match="no parked action"):
        inbox.approve("missing", principal=CONTROLLER_PRINCIPAL)


def test_review_without_deployment_owned_role_fails_before_dispatch() -> None:
    task = next(
        item
        for item in operator_tasks_for(CompanyAgentKind.ACCOUNTS)
        if item.require_approval and not item.freeze_before_approval
    )
    spec = spec_for_kind(task.agent_kind)
    bundle = load_fixture_bundle("default")
    inbox = ApprovalInbox(allowed_approvers=APPROVER_DIRECTORY)
    gateway = ApprovalBoundedGateway(
        CompanyRiskGateway(spec, bundle=bundle),
        inbox,
        digest_for=canonical_action_intent_digest,
        required_role_for=lambda _: None,
    )
    calls = {"count": 0}

    def counted(arguments: dict[str, object]) -> object:
        calls["count"] += 1
        return arguments

    action = ActionIntent.capture(
        agent_id=spec.agent_id,
        framework="test",
        tool_name=task.tool,
        capability=spec.tools.get(task.tool).capability,
        arguments=dict(task.arguments),
    )
    execution = GovernedRuntime(gateway).execute(action, counted)
    assert execution.status is ExecutionStatus.CONTROL_FAILURE
    assert execution.tool_executed is False
    assert execution.retryable is False
    assert calls["count"] == 0


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
    evidence = (tmp_path / "evidence.jsonl").read_text(encoding="utf-8")
    payloads = [json.loads(line) for line in evidence.splitlines()]
    serialized = json.dumps(payloads)
    assert "arguments" not in serialized
    assert "API_KEY" not in serialized
    assert all(
        payload["schema"] == "cbrain-operator-evidence/v1" for payload in payloads
    )
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
