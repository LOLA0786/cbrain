"""Offline operator loop: park REVIEW, named approve, execute once, freeze closed."""

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
CONTROLLER_ACTOR = "operator-controller-1"


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
            "action_intent_digest": self.action_intent_digest,
            "first_status": self.first_status,
            "final_status": self.final_status,
            "handler_invocations": self.handler_invocations,
            "approved_by": self.approved_by,
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
            "records": [record.to_payload() for record in self.records],
        }


def run_operator_loop(
    kind: CompanyAgentKind | None = None,
    *,
    actor_id: str = CONTROLLER_ACTOR,
) -> OperatorSuiteResult:
    records = tuple(
        run_operator_task(task, actor_id=actor_id) for task in operator_tasks_for(kind)
    )
    return OperatorSuiteResult(records=records)


def run_operator_task(
    task: OperatorTask,
    *,
    actor_id: str = CONTROLLER_ACTOR,
    freeze_before_approve: bool = False,
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
    handlers = build_handlers(
        kind=task.agent_kind, bundle=bundle, context=context
    )
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
    if freeze_before_approve:
        inbox.freeze("indeterminate_not_retried")
    if task.require_approval and first.status is ExecutionStatus.REVIEW_REQUIRED:
        try:
            approval = inbox.approve(action.request_id, actor_id=actor_id)
            final = runtime.execute(action, counted)
        except ApprovalInboxError as exc:
            final = GovernedExecution(
                status=ExecutionStatus.CONTROL_FAILURE,
                request_id=action.request_id,
                tool_executed=False,
                reason=str(exc),
            )
    if final.status is ExecutionStatus.INDETERMINATE:
        inbox.freeze("indeterminate_not_retried")
    success = _task_succeeded(
        task,
        first=first,
        final=final,
        handler_calls=handler_calls["count"],
        frozen=inbox.frozen,
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
    first: Any,
    final: Any,
    handler_calls: int,
    frozen: bool,
) -> bool:
    expected_final = {
        ExpectedDecision.ALLOW: ExecutionStatus.EXECUTED,
        ExpectedDecision.REVIEW: (
            ExecutionStatus.EXECUTED
            if task.require_approval
            else ExecutionStatus.REVIEW_REQUIRED
        ),
        ExpectedDecision.BLOCK: ExecutionStatus.BLOCKED,
    }[task.expected_decision]
    if frozen:
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
    lines = [
        "# Operator loop",
        "",
        f"- Passed: `{str(result.passed).lower()}`",
        f"- Tasks: {len(result.records)}",
        "- Decision authority: `company_test_gateway`",
        "- Live provider calls: `0`",
        "- Real-model quality: `not_evaluated`",
        "",
        "REVIEW tools execute only after a named human approval of the parked",
        "ActionIntent. INDETERMINATE runs are frozen and never retried.",
        "",
    ]
    return "\n".join(lines) + "\n"


__all__ = [
    "CONTROLLER_ACTOR",
    "EVIDENCE_SCHEMA",
    "OperatorRunRecord",
    "OperatorSuiteResult",
    "run_operator_loop",
    "run_operator_task",
    "write_operator_artifacts",
]
