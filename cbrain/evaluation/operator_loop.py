"""Offline operator loop: park REVIEW, role-bound approve, execute once, freeze closed."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cbrain import ActionIntent, ExecutionStatus, GovernedExecution, GovernedRuntime
from cbrain.company.approval import (
    ApprovalBoundedGateway,
    ApprovalInbox,
    ApprovalInboxError,
    ApprovalRecord,
    TrustedCallerContext,
    trusted_caller,
)
from cbrain.company.authority import CompanyExecutionContext
from cbrain.company.governance import CompanyRiskGateway
from cbrain.company.handlers import build_handlers
from cbrain.company.kinds import CompanyAgentKind
from cbrain.company.profiles import spec_for_kind
from cbrain.company.simulators import load_fixture_bundle
from cbrain.evaluation.company_harness import canonical_action_intent_digest
from cbrain.evaluation.company_scenarios import ExpectedDecision

from .operator_tasks import OperatorTask, operator_tasks_for

EVIDENCE_SCHEMA = "cbrain-operator-evidence/v1"
FORBIDDEN_EVIDENCE_KEYS = frozenset(
    {
        "arguments",
        "credential",
        "credentials",
        "endpoint",
        "endpoints",
        "payload",
        "raw_payload",
        "secret",
        "secrets",
        "tool_payload",
        "api_key",
        "password",
        "token",
    }
)


@dataclass(frozen=True, slots=True)
class OperatorRunRecord:
    task_id: str
    agent_kind: str
    tool: str
    request_id: str
    action_intent_digest: str
    first_status: str
    final_status: str
    handler_invocations: int
    approved_by: str | None
    trusted_approver_role: str | None
    frozen: bool
    retryable: bool
    success: bool
    reason: str

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "schema": EVIDENCE_SCHEMA,
            "task_id": self.task_id,
            "agent_kind": self.agent_kind,
            "tool": self.tool,
            "request_id": self.request_id,
            "action_intent_digest": self.action_intent_digest,
            "first_status": self.first_status,
            "final_status": self.final_status,
            "handler_invocations": self.handler_invocations,
            "approved_by": self.approved_by,
            "trusted_approver_role": self.trusted_approver_role,
            "frozen": self.frozen,
            "retryable": self.retryable,
            "success": self.success,
            "reason": self.reason,
            "decision_authority": "company_test_gateway",
            "execution_mode": "offline_fixture",
            "live_provider_calls": 0,
            "real_model_quality": "not_evaluated",
        }
        _assert_evidence_privacy(payload)
        return payload


@dataclass(frozen=True, slots=True)
class OperatorSuiteResult:
    records: tuple[OperatorRunRecord, ...]

    @property
    def passed(self) -> bool:
        return all(record.success for record in self.records)

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "schema": "cbrain-operator-suite/v1",
            "passed": self.passed,
            "total_tasks": len(self.records),
            "successes": sum(1 for record in self.records if record.success),
            "live_provider_calls": 0,
            "real_model_quality": "not_evaluated",
            "decision_authority": "company_test_gateway",
            "execution_mode": "offline_fixture",
            "indeterminate_retryable": False,
            "records": [record.to_payload() for record in self.records],
        }
        _assert_evidence_privacy(payload)
        return payload


def run_operator_loop(kind: CompanyAgentKind | None = None) -> OperatorSuiteResult:
    records = tuple(run_operator_task(task) for task in operator_tasks_for(kind))
    return OperatorSuiteResult(records=records)


def run_operator_task(
    task: OperatorTask,
    *,
    caller: TrustedCallerContext | None = None,
    freeze_before_approve: bool | None = None,
) -> OperatorRunRecord:
    spec = spec_for_kind(task.agent_kind)
    bundle = load_fixture_bundle("default")
    context = _context_for(task)
    inbox = ApprovalInbox()
    inner = CompanyRiskGateway(spec, context=context, bundle=bundle)
    gateway = ApprovalBoundedGateway(
        inner, inbox, digest_for=canonical_action_intent_digest
    )
    runtime = GovernedRuntime(gateway)
    handlers = build_handlers(kind=task.agent_kind, bundle=bundle, context=context)
    handler_calls = {"count": 0}

    def counted(arguments: Mapping[str, Any]) -> Any:
        handler_calls["count"] += 1
        return handlers[task.tool](arguments)

    action = ActionIntent.capture(
        agent_id=spec.agent_id,
        framework="cbrain-operator-loop",
        tool_name=task.tool,
        capability=spec.tools.get(task.tool).capability,
        arguments=dict(task.arguments),
        request_id=f"operator-{task.task_id}",
    )
    digest = canonical_action_intent_digest(action)
    first = runtime.execute(action, counted)
    approval: ApprovalRecord | None = None
    final = first
    freeze = (
        task.freeze_before_approval
        if freeze_before_approve is None
        else freeze_before_approve
    )
    if freeze:
        inbox.freeze("indeterminate_not_retried")
    if task.require_approval and first.status is ExecutionStatus.REVIEW_REQUIRED:
        try:
            approval = inbox.approve(
                action.request_id, caller=_caller_for(task, caller)
            )
            final = runtime.execute(action, counted)
        except ApprovalInboxError as exc:
            if freeze or task.freeze_before_approval:
                final = GovernedExecution(
                    status=ExecutionStatus.INDETERMINATE,
                    request_id=action.request_id,
                    tool_executed=None,
                    reason=str(exc),
                    retryable=False,
                )
            else:
                final = GovernedExecution(
                    status=ExecutionStatus.CONTROL_FAILURE,
                    request_id=action.request_id,
                    tool_executed=False,
                    reason=str(exc),
                    retryable=False,
                )
    if final.status is ExecutionStatus.INDETERMINATE:
        inbox.freeze("indeterminate_not_retried")
    success = _task_succeeded(
        task,
        first=first,
        final=final,
        handler_calls=handler_calls["count"],
        frozen=inbox.frozen,
        approved=approval is not None,
        unexpected_freeze=freeze and not task.freeze_before_approval,
    )
    return OperatorRunRecord(
        task_id=task.task_id,
        agent_kind=task.agent_kind.value,
        tool=task.tool,
        request_id=action.request_id,
        action_intent_digest=digest,
        first_status=first.status.value,
        final_status=final.status.value,
        handler_invocations=handler_calls["count"],
        approved_by=None if approval is None else approval.actor_id,
        trusted_approver_role=None if approval is None else approval.role.value,
        frozen=inbox.frozen,
        retryable=final.retryable,
        success=success,
        reason=final.reason,
    )


def write_operator_artifacts(
    output_dir: str | Path, result: OperatorSuiteResult
) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "operator_suite.json").write_text(
        json.dumps(result.to_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        json.dumps(record.to_payload(), sort_keys=True) for record in result.records
    ]
    (directory / "evidence.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (directory / "summary.md").write_text(_summary_markdown(result), encoding="utf-8")


def _caller_for(
    task: OperatorTask, override: TrustedCallerContext | None
) -> TrustedCallerContext:
    if override is not None:
        return override
    if task.fixture_actor_id is None:
        raise ApprovalInboxError("task has no fixture actor")
    return trusted_caller(task.fixture_actor_id)


def _task_succeeded(
    task: OperatorTask,
    *,
    first: Any,
    final: Any,
    handler_calls: int,
    frozen: bool,
    approved: bool,
    unexpected_freeze: bool,
) -> bool:
    if task.freeze_before_approval:
        return (
            frozen
            and not approved
            and handler_calls == 0
            and final.status is ExecutionStatus.INDETERMINATE
            and final.retryable is False
            and first.status is ExecutionStatus.REVIEW_REQUIRED
        )
    expected_final = {
        ExpectedDecision.ALLOW: ExecutionStatus.EXECUTED,
        ExpectedDecision.REVIEW: (
            ExecutionStatus.EXECUTED
            if task.require_approval
            else ExecutionStatus.REVIEW_REQUIRED
        ),
        ExpectedDecision.BLOCK: ExecutionStatus.BLOCKED,
    }[task.expected_decision]
    if frozen or unexpected_freeze:
        return False
    if final.status is ExecutionStatus.INDETERMINATE:
        return False
    if handler_calls != (1 if task.expect_handler else 0):
        return False
    if task.require_approval and first.status is not ExecutionStatus.REVIEW_REQUIRED:
        return False
    return final.status is expected_final


def _context_for(task: OperatorTask) -> CompanyExecutionContext:
    if task.authorized_matters is not None:
        permitted = task.authorized_matters
    elif task.agent_kind is CompanyAgentKind.LEGAL:
        permitted = frozenset({"matter-a"})
    else:
        permitted = frozenset()
    return CompanyExecutionContext(
        principal_id=f"{task.agent_kind.value}-operator",
        agent_kind=task.agent_kind,
        permitted_matter_ids=permitted,
        authorization_scope_id=f"scope-{task.task_id}",
    )


def _summary_markdown(result: OperatorSuiteResult) -> str:
    freeze_count = sum(1 for record in result.records if record.frozen)
    lines = [
        "# Operator loop",
        "",
        f"- Passed: `{str(result.passed).lower()}`",
        f"- Tasks: {len(result.records)}",
        f"- Explicit freeze-before-approval records: {freeze_count}",
        "- Decision authority: `company_test_gateway`",
        "- Live provider calls: `0`",
        "- Real-model quality: `not_evaluated`",
        "- Execution mode: `offline_fixture`",
        "",
        "REVIEW tools execute only after a trusted-role approval of the parked",
        "ActionIntent. Identity and role come from caller context, never from",
        "model arguments. INDETERMINATE runs are frozen and never retried.",
        "The existing non-freeze tasks do not demonstrate freezing.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _assert_evidence_privacy(payload: Mapping[str, Any]) -> None:
    for key in _walk_keys(payload):
        normalized = key.lower().replace("-", "_")
        if normalized in FORBIDDEN_EVIDENCE_KEYS:
            raise ValueError(f"operator evidence must not contain {key!r}")


def _walk_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str):
                keys.append(key)
            keys.extend(_walk_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.extend(_walk_keys(item))
    return keys


__all__ = [
    "EVIDENCE_SCHEMA",
    "FORBIDDEN_EVIDENCE_KEYS",
    "OperatorRunRecord",
    "OperatorSuiteResult",
    "run_operator_loop",
    "run_operator_task",
    "write_operator_artifacts",
]
