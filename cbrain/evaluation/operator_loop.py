"""Offline operator loop for role-bound REVIEW resume.

Park a REVIEW ActionIntent, approve from injected caller context, execute
the handler once, and freeze closed. A freeze before approval or send is
CONTROL_FAILURE with tool_executed=False. Only a possible post-send
execution may be INDETERMINATE. Approver identities are injected; this
module does not ship fixture actor directories.
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from cbrain import ActionIntent, ExecutionStatus, GovernedExecution, GovernedRuntime
from cbrain.company.approval import (
    ApprovalBoundedGateway,
    ApprovalInbox,
    ApprovalInboxError,
    ApprovalPrincipal,
    ApprovalRecord,
    ApprovalRole,
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

_APPROVAL_ROLE_BY_TOOL = MappingProxyType(
    {
        "execute_payment": ApprovalRole.CONTROLLER,
        "send_commitment": ApprovalRole.COUNSEL,
        "apply_patch": ApprovalRole.CODE_REVIEWER,
        "propose_patch": ApprovalRole.CODE_REVIEWER,
        "open_pull_request": ApprovalRole.CODE_REVIEWER,
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
    tool_executed: bool | None
    retryable: bool
    handler_invocations: int
    required_approval_role: str | None
    approved_by: str | None
    approved_role: str | None
    frozen: bool
    success: bool
    reason: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": EVIDENCE_SCHEMA,
            "task_id": self.task_id,
            "agent_kind": self.agent_kind,
            "tool": self.tool,
            "request_id": self.request_id,
            "intent_digest": self.action_intent_digest,
            "action_intent_digest": self.action_intent_digest,
            "first_status": self.first_status,
            "final_status": self.final_status,
            "tool_executed": self.tool_executed,
            "retryable": self.retryable,
            "handler_invocations": self.handler_invocations,
            "required_approval_role": self.required_approval_role,
            "approved_by": self.approved_by,
            "approved_role": self.approved_role,
            "frozen": self.frozen,
            "success": self.success,
            "reason": self.reason,
            "decision_authority": "company_test_gateway",
            "execution_mode": "offline_fixture",
            "live_provider_calls": 0,
            "real_model_quality": "not_evaluated",
        }


@dataclass(frozen=True, slots=True)
class OperatorSuiteResult:
    records: tuple[OperatorRunRecord, ...]

    @property
    def passed(self) -> bool:
        return all(record.success for record in self.records)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "cbrain-operator-suite/v1",
            "passed": self.passed,
            "total_tasks": len(self.records),
            "successes": sum(1 for record in self.records if record.success),
            "live_provider_calls": 0,
            "real_model_quality": "not_evaluated",
            "decision_authority": "company_test_gateway",
            "execution_mode": "offline_fixture",
            "records": [record.to_payload() for record in self.records],
        }


def load_offline_eval_identities(
    path: Path,
) -> tuple[
    Mapping[ApprovalRole, Collection[str]],
    Mapping[CompanyAgentKind, ApprovalPrincipal],
    float,
]:
    """Load a deployment-owned or test-injected approver directory from disk."""
    spec = importlib.util.spec_from_file_location(
        "cbrain_offline_operator_fixtures", path
    )
    if spec is None or spec.loader is None:
        raise ApprovalInboxError("approver directory file is missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (
        module.APPROVER_DIRECTORY,
        module.PRINCIPAL_BY_AGENT,
        float(module.FIXTURE_MAX_TTL_SECONDS),
    )


def run_operator_loop(
    kind: CompanyAgentKind | None = None,
    *,
    allowed_approvers: Mapping[ApprovalRole, Collection[str]],
    principal_by_agent: Mapping[CompanyAgentKind, ApprovalPrincipal],
    max_ttl_seconds: float,
) -> OperatorSuiteResult:
    records = tuple(
        run_operator_task(
            task,
            allowed_approvers=allowed_approvers,
            principal_by_agent=principal_by_agent,
            max_ttl_seconds=max_ttl_seconds,
        )
        for task in operator_tasks_for(kind)
    )
    return OperatorSuiteResult(records=records)


def run_operator_task(
    task: OperatorTask,
    *,
    allowed_approvers: Mapping[ApprovalRole, Collection[str]],
    principal_by_agent: Mapping[CompanyAgentKind, ApprovalPrincipal],
    max_ttl_seconds: float,
    principal: ApprovalPrincipal | None = None,
    freeze_before_approve: bool = False,
) -> OperatorRunRecord:
    spec = spec_for_kind(task.agent_kind)
    bundle = load_fixture_bundle("default")
    context = _context_for(task)
    inbox = ApprovalInbox(
        allowed_approvers=allowed_approvers,
        max_ttl_seconds=max_ttl_seconds,
    )
    inner = CompanyRiskGateway(spec, context=context, bundle=bundle)
    gateway = ApprovalBoundedGateway(
        inner,
        inbox,
        digest_for=canonical_action_intent_digest,
        required_role_for=_required_approval_role,
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
    required_role = _required_approval_role(action)
    first = runtime.execute(action, counted)
    approval: ApprovalRecord | None = None
    final = first
    expected_freeze = task.freeze_before_approval or freeze_before_approve
    if expected_freeze:
        inbox.freeze("operator_control_state_frozen")
    if task.require_approval and first.status is ExecutionStatus.REVIEW_REQUIRED:
        selected_principal = principal or principal_by_agent[task.agent_kind]
        try:
            approval = inbox.approve(
                action.request_id,
                principal=selected_principal,
            )
            final = runtime.execute(action, counted)
        except ApprovalInboxError as exc:
            if inbox.frozen:
                final = runtime.execute(action, counted)
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
        expected_freeze=expected_freeze,
    )
    return OperatorRunRecord(
        task_id=task.task_id,
        agent_kind=task.agent_kind.value,
        tool=task.tool,
        request_id=action.request_id,
        action_intent_digest=digest,
        first_status=first.status.value,
        final_status=final.status.value,
        tool_executed=final.tool_executed,
        retryable=final.retryable,
        handler_invocations=handler_calls["count"],
        required_approval_role=None if required_role is None else required_role.value,
        approved_by=None if approval is None else approval.actor_id,
        approved_role=None if approval is None else approval.actor_role.value,
        frozen=inbox.frozen,
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


def _task_succeeded(
    task: OperatorTask,
    *,
    first: GovernedExecution,
    final: GovernedExecution,
    handler_calls: int,
    frozen: bool,
    expected_freeze: bool,
) -> bool:
    if expected_freeze:
        return (
            first.status is ExecutionStatus.REVIEW_REQUIRED
            and final.status is ExecutionStatus.CONTROL_FAILURE
            and final.tool_executed is False
            and final.retryable is False
            and handler_calls == 0
            and frozen
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
    if frozen or final.status is ExecutionStatus.INDETERMINATE:
        return False
    if handler_calls != (1 if task.expect_handler else 0):
        return False
    if task.require_approval and first.status is not ExecutionStatus.REVIEW_REQUIRED:
        return False
    return final.status is expected_final


def _required_approval_role(action: ActionIntent) -> ApprovalRole | None:
    return _APPROVAL_ROLE_BY_TOOL.get(action.tool_name)


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
    lines = [
        "# Operator loop",
        "",
        f"- Passed: `{str(result.passed).lower()}`",
        f"- Tasks: {len(result.records)}",
        "- Decision authority: `company_test_gateway`",
        "- Live provider calls: `0`",
        "- Real-model quality: `not_evaluated`",
        "",
        "REVIEW tools execute only after a role-bound approval of the parked",
        "ActionIntent. A pre-send frozen inbox fails with CONTROL_FAILURE and",
        "does not dispatch. A possible post-send execution would be recorded as",
        "INDETERMINATE, frozen, and never retried.",
        "",
    ]
    return "\n".join(lines) + "\n"


__all__ = [
    "EVIDENCE_SCHEMA",
    "OperatorRunRecord",
    "OperatorSuiteResult",
    "load_offline_eval_identities",
    "run_operator_loop",
    "run_operator_task",
    "write_operator_artifacts",
]
