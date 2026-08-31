"""Operator-loop approval resume for accounts, legal, and coding agents."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

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
    run_operator_loop,
    run_operator_task,
    write_operator_artifacts,
)
from cbrain.evaluation.operator_tasks import (
    OPERATOR_TASKS_PER_AGENT,
    all_operator_tasks,
    operator_tasks_for,
)


def _load_operator_fixtures() -> ModuleType:
    path = Path(__file__).with_name("operator_fixtures.py")
    spec = importlib.util.spec_from_file_location("cbrain_operator_fixtures", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"missing operator fixtures at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_FX = _load_operator_fixtures()
APPROVER_DIRECTORY = _FX.APPROVER_DIRECTORY
CODE_REVIEWER_PRINCIPAL = _FX.CODE_REVIEWER_PRINCIPAL
CONTROLLER_PRINCIPAL = _FX.CONTROLLER_PRINCIPAL
COUNSEL_PRINCIPAL = _FX.COUNSEL_PRINCIPAL
FIXTURE_MAX_TTL_SECONDS = _FX.FIXTURE_MAX_TTL_SECONDS
PRINCIPAL_BY_AGENT = _FX.PRINCIPAL_BY_AGENT

_FIXTURE_IDENTITIES = (
    "operator-controller-1",
    "operator-counsel-1",
    "operator-code-reviewer-1",
    "operator-buyer-lead-1",
)


def _inbox() -> ApprovalInbox:
    return ApprovalInbox(
        allowed_approvers=APPROVER_DIRECTORY,
        max_ttl_seconds=FIXTURE_MAX_TTL_SECONDS,
    )


def _run_loop(kind: CompanyAgentKind | None = None):
    return run_operator_loop(
        kind,
        allowed_approvers=APPROVER_DIRECTORY,
        principal_by_agent=PRINCIPAL_BY_AGENT,
        max_ttl_seconds=FIXTURE_MAX_TTL_SECONDS,
    )


def _run_task(task, **kwargs):
    return run_operator_task(
        task,
        allowed_approvers=APPROVER_DIRECTORY,
        principal_by_agent=PRINCIPAL_BY_AGENT,
        max_ttl_seconds=FIXTURE_MAX_TTL_SECONDS,
        **kwargs,
    )


def test_operator_suite_has_nine_tasks_per_vertical() -> None:
    tasks = all_operator_tasks()
    assert len(tasks) == 36
    assert sum(task.freeze_before_approval for task in tasks) == 4
    for kind in OPERATOR_AGENT_KINDS:
        assert len(operator_tasks_for(kind)) == OPERATOR_TASKS_PER_AGENT


def test_operator_loop_passes_offline_with_explicit_frozen_evidence() -> None:
    result = _run_loop()
    assert result.passed is True
    assert result.to_payload()["live_provider_calls"] == 0
    assert result.to_payload()["real_model_quality"] == "not_evaluated"
    assert result.to_payload()["decision_authority"] == "company_test_gateway"
    frozen = [record for record in result.records if record.frozen]
    assert len(frozen) == 4
    assert {record.agent_kind for record in frozen} == {
        "accounts",
        "legal",
        "coding",
        "procurement",
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
    record = _run_task(task)
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
    inbox = _inbox()
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
    inbox = _inbox()
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
        (
            CompanyAgentKind.PROCUREMENT,
            ApprovalRole.BUYER_LEAD,
            _FX.BUYER_LEAD_PRINCIPAL.actor_id,
        ),
    ),
)
def test_each_vertical_uses_its_own_approver_role(
    kind: CompanyAgentKind,
    expected_role: ApprovalRole,
    expected_actor: str,
) -> None:
    record = next(
        item for item in _run_loop(kind).records if item.approved_by is not None
    )
    assert record.required_approval_role == expected_role.value
    assert record.approved_role == expected_role.value
    assert record.approved_by == expected_actor


def test_frozen_inbox_refuses_approval_and_dispatch_as_expected_safety_case() -> None:
    for kind in OPERATOR_AGENT_KINDS:
        task = next(
            item for item in operator_tasks_for(kind) if item.freeze_before_approval
        )
        record = _run_task(task)
        assert record.frozen is True
        assert record.handler_invocations == 0
        assert record.success is True
        assert record.final_status == ExecutionStatus.CONTROL_FAILURE.value
        assert record.tool_executed is False
        assert record.retryable is False


def test_approve_without_parked_action_fails() -> None:
    inbox = _inbox()
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
    inbox = _inbox()
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
        record = _run_task(task)
        assert record.handler_invocations == 0
        assert record.final_status == ExecutionStatus.BLOCKED.value
    after = load_fixture_bundle("default").coding.snapshot()
    assert after["files"] == before["files"]
    assert after["pull_requests"] == []


def test_evidence_omits_arguments_and_cli_writes_artifacts(tmp_path: Path) -> None:
    result = _run_loop(CompanyAgentKind.CODING)
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


def _payment_action() -> tuple[ActionIntent, str]:
    task = next(
        item
        for item in operator_tasks_for(CompanyAgentKind.ACCOUNTS)
        if item.require_approval and not item.freeze_before_approval
    )
    spec = spec_for_kind(task.agent_kind)
    action = ActionIntent.capture(
        agent_id=spec.agent_id,
        framework="test",
        tool_name=task.tool,
        capability=spec.tools.get(task.tool).capability,
        arguments=dict(task.arguments),
        request_id="operator-correctness",
    )
    return action, canonical_action_intent_digest(action)


def test_freeze_before_approval_or_send_is_control_failure() -> None:
    action, _digest = _payment_action()
    spec = spec_for_kind(CompanyAgentKind.ACCOUNTS)
    bundle = load_fixture_bundle("default")
    inbox = _inbox()
    inbox.freeze("frozen_before_send")
    calls = {"count": 0}
    execution = GovernedRuntime(
        ApprovalBoundedGateway(
            CompanyRiskGateway(spec, bundle=bundle),
            inbox,
            digest_for=canonical_action_intent_digest,
            required_role_for=lambda _: ApprovalRole.CONTROLLER,
        )
    ).execute(action, lambda arguments: calls.__setitem__("count", calls["count"] + 1))
    assert execution.status is ExecutionStatus.CONTROL_FAILURE
    assert execution.tool_executed is False
    assert execution.retryable is False
    assert calls["count"] == 0
    assert execution.status is not ExecutionStatus.INDETERMINATE


def test_handler_failure_after_send_is_indeterminate_and_not_retryable() -> None:
    action, digest = _payment_action()
    spec = spec_for_kind(CompanyAgentKind.ACCOUNTS)
    bundle = load_fixture_bundle("default")
    inbox = _inbox()
    runtime = GovernedRuntime(
        ApprovalBoundedGateway(
            CompanyRiskGateway(spec, bundle=bundle),
            inbox,
            digest_for=canonical_action_intent_digest,
            required_role_for=lambda _: ApprovalRole.CONTROLLER,
        )
    )
    first = runtime.execute(action, lambda arguments: arguments)
    assert first.status is ExecutionStatus.REVIEW_REQUIRED
    inbox.approve(action.request_id, principal=CONTROLLER_PRINCIPAL)

    def exploding(arguments: dict[str, object]) -> object:
        raise RuntimeError("target unreachable after send")

    result = runtime.execute(action, exploding)
    assert result.status is ExecutionStatus.INDETERMINATE
    assert result.tool_executed is None
    assert result.retryable is False
    assert digest == canonical_action_intent_digest(action)


def test_required_role_is_immutable_and_rechecked_on_consume() -> None:
    action, digest = _payment_action()
    inbox = _inbox()
    inbox.park(action, digest=digest, required_role=ApprovalRole.CONTROLLER)
    with pytest.raises(ApprovalInboxError, match="already parked"):
        inbox.park(action, digest=digest, required_role=ApprovalRole.COUNSEL)
    inbox.approve(action.request_id, principal=CONTROLLER_PRINCIPAL)
    assert (
        inbox.consume_if_approved(
            action, digest=digest, required_role=ApprovalRole.COUNSEL
        )
        is False
    )
    roles = {"current": ApprovalRole.CONTROLLER}
    spec = spec_for_kind(CompanyAgentKind.ACCOUNTS)
    bundle = load_fixture_bundle("default")
    inbox2 = _inbox()
    calls = {"count": 0}
    runtime = GovernedRuntime(
        ApprovalBoundedGateway(
            CompanyRiskGateway(spec, bundle=bundle),
            inbox2,
            digest_for=canonical_action_intent_digest,
            required_role_for=lambda _: roles["current"],
        )
    )
    parked = runtime.execute(action, lambda arguments: calls.__setitem__("count", 1))
    assert parked.status is ExecutionStatus.REVIEW_REQUIRED
    inbox2.approve(action.request_id, principal=CONTROLLER_PRINCIPAL)
    roles["current"] = ApprovalRole.COUNSEL
    second = runtime.execute(action, lambda arguments: calls.__setitem__("count", 1))
    assert second.status is ExecutionStatus.CONTROL_FAILURE
    assert second.tool_executed is False
    assert calls["count"] == 0


def test_ttl_rejects_bool_nan_infinity_non_numeric_non_positive_and_over_max() -> None:
    action, digest = _payment_action()
    inbox = _inbox()
    inbox.park(action, digest=digest, required_role=ApprovalRole.CONTROLLER)
    rejected = (
        True,
        False,
        float("nan"),
        float("inf"),
        float("-inf"),
        "3600",
        object(),
        0,
        -1,
        FIXTURE_MAX_TTL_SECONDS + 1,
    )
    for ttl in rejected:
        with pytest.raises(ApprovalInboxError, match="ttl_seconds"):
            inbox.approve(
                action.request_id,
                principal=CONTROLLER_PRINCIPAL,
                ttl_seconds=ttl,
            )
    record = inbox.approve(
        action.request_id,
        principal=CONTROLLER_PRINCIPAL,
        ttl_seconds=FIXTURE_MAX_TTL_SECONDS,
    )
    assert record.expires_at == record.approved_at + FIXTURE_MAX_TTL_SECONDS


def test_approval_inbox_requires_injected_directory_and_max_ttl() -> None:
    with pytest.raises(TypeError):
        ApprovalInbox()
    with pytest.raises(TypeError):
        ApprovalInbox(allowed_approvers=APPROVER_DIRECTORY)


def test_fixture_actor_ids_live_only_under_tests() -> None:
    root = Path(__file__).resolve().parents[1] / "cbrain"
    leaked: list[str] = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for identity in _FIXTURE_IDENTITIES:
            if identity in text:
                leaked.append(f"{path.relative_to(root.parent)}:{identity}")
    assert leaked == []


def test_evidence_preserves_certainty_fields_without_payloads(tmp_path: Path) -> None:
    result = _run_loop()
    write_operator_artifacts(tmp_path, result)
    payloads = [
        json.loads(line)
        for line in (tmp_path / "evidence.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    required = (
        "tool_executed",
        "required_approval_role",
        "approved_by",
        "approved_role",
        "retryable",
        "final_status",
        "intent_digest",
    )
    forbidden = (
        "arguments",
        "credentials",
        "credential",
        "endpoint",
        "endpoints",
        "payload",
        "raw_payload",
        "API_KEY",
        "https://",
        "beneficiary_id",
        "amount_minor",
    )
    rendered = json.dumps(payloads)
    for payload in payloads:
        for field in required:
            assert field in payload
        assert payload["decision_authority"] == "company_test_gateway"
        assert payload["live_provider_calls"] == 0
        assert payload["real_model_quality"] == "not_evaluated"
        assert payload["execution_mode"] == "offline_fixture"
    for token in forbidden:
        assert token not in rendered
    frozen = [payload for payload in payloads if payload["frozen"] is True]
    assert len(frozen) == 4
    assert all(item["final_status"] == "CONTROL_FAILURE" for item in frozen)
    assert all(item["tool_executed"] is False for item in frozen)
    assert all(item["retryable"] is False for item in frozen)
