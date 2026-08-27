"""Operator-loop approval resume for accounts, legal, and coding agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cbrain import ExecutionStatus, GovernedRuntime
from cbrain.company.approval import (
    CODE_REVIEWER_ACTOR,
    CONTROLLER_ACTOR,
    COUNSEL_ACTOR,
    ApprovalBoundedGateway,
    ApprovalInbox,
    ApprovalInboxError,
    ApproverRole,
    TrustedCallerContext,
    trusted_caller,
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
    FORBIDDEN_EVIDENCE_KEYS,
    run_operator_loop,
    run_operator_task,
    write_operator_artifacts,
)
from cbrain.evaluation.operator_tasks import (
    OPERATOR_TASKS_PER_AGENT,
    all_operator_tasks,
    operator_tasks_for,
)


def _review_task(kind: CompanyAgentKind, tool: str):
    return next(
        item
        for item in operator_tasks_for(kind)
        if item.tool == tool
        and item.require_approval
        and not item.freeze_before_approval
    )


def test_operator_suite_has_nine_tasks_per_vertical() -> None:
    tasks = all_operator_tasks()
    assert len(tasks) == 27
    for kind in OPERATOR_AGENT_KINDS:
        assert len(operator_tasks_for(kind)) == OPERATOR_TASKS_PER_AGENT
        freeze = [
            item for item in operator_tasks_for(kind) if item.freeze_before_approval
        ]
        assert len(freeze) == 1


def test_operator_loop_passes_offline() -> None:
    result = run_operator_loop()
    payload = result.to_payload()
    assert result.passed is True
    assert payload["live_provider_calls"] == 0
    assert payload["real_model_quality"] == "not_evaluated"
    assert payload["decision_authority"] == "company_test_gateway"
    assert payload["execution_mode"] == "offline_fixture"
    assert payload["indeterminate_retryable"] is False
    assert payload["total_tasks"] == 27


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
    assert record.trusted_approver_role is None
    assert record.success is True


def test_named_approval_executes_same_intent_once() -> None:
    task = _review_task(CompanyAgentKind.ACCOUNTS, "execute_payment")
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
    inbox.approve(action.request_id, caller=trusted_caller(CONTROLLER_ACTOR))
    second = runtime.execute(action, counted)
    assert second.status is ExecutionStatus.EXECUTED
    assert calls["count"] == 1
    third = runtime.execute(action, counted)
    assert third.status is ExecutionStatus.BLOCKED
    assert calls["count"] == 1
    assert third.reason == "approval_already_consumed"


def test_wrong_role_unknown_blank_and_mismatch_never_run_handler() -> None:
    task = _review_task(CompanyAgentKind.ACCOUNTS, "execute_payment")
    spec = spec_for_kind(task.agent_kind)
    bundle = load_fixture_bundle("default")
    inbox = ApprovalInbox()
    inner = CompanyRiskGateway(spec, bundle=bundle)
    gateway = ApprovalBoundedGateway(
        inner, inbox, digest_for=canonical_action_intent_digest
    )
    runtime = GovernedRuntime(gateway)
    calls: list[Any] = []

    action = ActionIntent.capture(
        agent_id=spec.agent_id,
        framework="test",
        tool_name=task.tool,
        capability=spec.tools.get(task.tool).capability,
        arguments=dict(task.arguments),
        request_id="operator-role-bound",
    )
    first = runtime.execute(action, lambda arguments: calls.append(arguments))
    assert first.status is ExecutionStatus.REVIEW_REQUIRED
    assert calls == []

    with pytest.raises(ApprovalInboxError, match="wrong role"):
        inbox.approve(action.request_id, caller=trusted_caller(COUNSEL_ACTOR))
    with pytest.raises(ApprovalInboxError, match="unknown actor"):
        inbox.approve(
            action.request_id,
            caller=TrustedCallerContext(
                actor_id="operator-unknown-1", role=ApproverRole.CONTROLLER
            ),
        )
    with pytest.raises(ApprovalInboxError, match="actor_id must be non-empty"):
        trusted_caller("   ")
    with pytest.raises(ApprovalInboxError, match="actor mismatch"):
        inbox.approve(
            action.request_id,
            caller=TrustedCallerContext(
                actor_id=CONTROLLER_ACTOR, role=ApproverRole.COUNSEL
            ),
        )
    with pytest.raises(ApprovalInboxError, match="duplicate approval"):
        inbox.approve(action.request_id, caller=trusted_caller(CONTROLLER_ACTOR))
        inbox.approve(action.request_id, caller=trusted_caller(CONTROLLER_ACTOR))

    # The first of the duplicate pair succeeded; consume once, then refuse.
    second = runtime.execute(action, lambda arguments: calls.append(arguments))
    assert second.status is ExecutionStatus.EXECUTED
    assert len(calls) == 1
    third = runtime.execute(action, lambda arguments: calls.append(arguments))
    assert third.status is ExecutionStatus.BLOCKED
    assert len(calls) == 1


def test_action_intent_actor_fields_cannot_authorize() -> None:
    task = _review_task(CompanyAgentKind.LEGAL, "send_commitment")
    spec = spec_for_kind(task.agent_kind)
    bundle = load_fixture_bundle("default")
    context = CompanyExecutionContext(
        principal_id="legal-operator",
        agent_kind=CompanyAgentKind.LEGAL,
        permitted_matter_ids=frozenset({"matter-a"}),
        authorization_scope_id="scope-legal-intent",
    )
    inbox = ApprovalInbox()
    inner = CompanyRiskGateway(spec, context=context, bundle=bundle)
    gateway = ApprovalBoundedGateway(
        inner, inbox, digest_for=canonical_action_intent_digest
    )
    runtime = GovernedRuntime(gateway)
    calls: list[Any] = []
    poisoned = dict(task.arguments)
    poisoned.update(
        {
            "actor_id": COUNSEL_ACTOR,
            "role": ApproverRole.COUNSEL.value,
            "approver": COUNSEL_ACTOR,
        }
    )
    action = ActionIntent.capture(
        agent_id=spec.agent_id,
        framework="test",
        tool_name=task.tool,
        capability=spec.tools.get(task.tool).capability,
        arguments=poisoned,
        request_id="operator-poisoned-intent",
    )
    first = runtime.execute(action, lambda arguments: calls.append(arguments))
    assert first.status is ExecutionStatus.REVIEW_REQUIRED
    assert calls == []
    with pytest.raises(ApprovalInboxError, match="wrong role"):
        inbox.approve(action.request_id, caller=trusted_caller(CONTROLLER_ACTOR))
    assert calls == []


def test_fixture_actors_match_required_roles() -> None:
    accounts = _review_task(CompanyAgentKind.ACCOUNTS, "execute_payment")
    legal = _review_task(CompanyAgentKind.LEGAL, "send_commitment")
    patch = _review_task(CompanyAgentKind.CODING, "apply_patch")
    pull = _review_task(CompanyAgentKind.CODING, "open_pull_request")
    assert accounts.required_role is ApproverRole.CONTROLLER
    assert accounts.fixture_actor_id == CONTROLLER_ACTOR
    assert legal.required_role is ApproverRole.COUNSEL
    assert legal.fixture_actor_id == COUNSEL_ACTOR
    assert patch.required_role is ApproverRole.CODE_REVIEWER
    assert patch.fixture_actor_id == CODE_REVIEWER_ACTOR
    assert pull.required_role is ApproverRole.CODE_REVIEWER
    assert pull.fixture_actor_id == CODE_REVIEWER_ACTOR
    for task in (accounts, legal, patch, pull):
        record = run_operator_task(task)
        assert record.success is True
        assert record.handler_invocations == 1
        assert record.approved_by == task.fixture_actor_id
        assert record.trusted_approver_role == task.required_role.value


def test_frozen_inbox_refuses_approval_and_dispatch() -> None:
    task = _review_task(CompanyAgentKind.LEGAL, "send_commitment")
    record = run_operator_task(task, freeze_before_approve=True)
    assert record.frozen is True
    assert record.handler_invocations == 0
    assert record.approved_by is None
    assert record.success is False
    assert record.final_status == ExecutionStatus.INDETERMINATE.value
    assert record.retryable is False


def test_suite_contains_explicit_freeze_before_approval_evidence(
    tmp_path: Path,
) -> None:
    result = run_operator_loop()
    write_operator_artifacts(tmp_path, result)
    frozen = [record for record in result.records if record.frozen]
    assert len(frozen) == 3
    kinds = {record.agent_kind for record in frozen}
    assert kinds == {"accounts", "legal", "coding"}
    for record in frozen:
        payload = record.to_payload()
        assert payload["frozen"] is True
        assert payload["handler_invocations"] == 0
        assert payload["approved_by"] is None
        assert payload["trusted_approver_role"] is None
        assert payload["final_status"] == ExecutionStatus.INDETERMINATE.value
        assert payload["retryable"] is False
        assert payload["success"] is True
    non_freeze = [record for record in result.records if not record.frozen]
    assert len(non_freeze) == 24
    assert all(record.frozen is False for record in non_freeze)


def test_approve_without_parked_action_fails() -> None:
    inbox = ApprovalInbox()
    with pytest.raises(ApprovalInboxError, match="no parked action"):
        inbox.approve("missing", caller=trusted_caller(CONTROLLER_ACTOR))


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


def test_evidence_omits_arguments_credentials_endpoints_and_payloads(
    tmp_path: Path,
) -> None:
    result = run_operator_loop()
    write_operator_artifacts(tmp_path, result)
    rendered = (tmp_path / "evidence.jsonl").read_text(encoding="utf-8")
    suite = (tmp_path / "operator_suite.json").read_text(encoding="utf-8")
    combined = rendered + suite
    for line in rendered.splitlines():
        payload = json.loads(line)
        for key in FORBIDDEN_EVIDENCE_KEYS:
            assert key not in payload
        assert "arguments" not in payload
        assert payload["schema"] == "cbrain-operator-evidence/v1"
        assert payload["decision_authority"] == "company_test_gateway"
        assert payload["live_provider_calls"] == 0
        assert payload["real_model_quality"] == "not_evaluated"
        assert payload["execution_mode"] == "offline_fixture"
    forbidden_fragments = (
        "API_KEY",
        "inv-1",
        "vendor-1",
        "10000",
        "beneficiary_id",
        "amount_minor",
        "src/app.py",
        "https://",
        "http://",
        "password",
        "credential",
        "We agree to the redline.",
        "def mul(a, b)",
    )
    for fragment in forbidden_fragments:
        assert fragment not in combined
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
