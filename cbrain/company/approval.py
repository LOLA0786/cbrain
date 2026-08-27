"""Named-human approval inbox for REVIEW_REQUIRED company actions.

This does not decide policy. It records who approved a parked ActionIntent
and allows the existing gateway to run the handler once after approval.
Approver identity and role come only from trusted caller context — never from
ActionIntent or model arguments. Frozen inboxes never approve and never dispatch.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from cbrain import ActionIntent, ExecutionStatus, GovernedExecution

from .governance import CompanyRiskGateway


class ApprovalInboxError(ValueError):
    """The approval inbox rejected park, approve, or consume."""


class ApproverRole(StrEnum):
    CONTROLLER = "controller"
    COUNSEL = "counsel"
    CODE_REVIEWER = "code_reviewer"


TOOL_REQUIRED_APPROVER_ROLE: Mapping[str, ApproverRole] = MappingProxyType(
    {
        "execute_payment": ApproverRole.CONTROLLER,
        "send_commitment": ApproverRole.COUNSEL,
        "apply_patch": ApproverRole.CODE_REVIEWER,
        "open_pull_request": ApproverRole.CODE_REVIEWER,
    }
)

FIXTURE_ACTORS: Mapping[str, ApproverRole] = MappingProxyType(
    {
        "operator-controller-1": ApproverRole.CONTROLLER,
        "operator-counsel-1": ApproverRole.COUNSEL,
        "operator-code-reviewer-1": ApproverRole.CODE_REVIEWER,
    }
)

CONTROLLER_ACTOR = "operator-controller-1"
COUNSEL_ACTOR = "operator-counsel-1"
CODE_REVIEWER_ACTOR = "operator-code-reviewer-1"


@dataclass(frozen=True, slots=True)
class TrustedCallerContext:
    """Control-plane caller identity. Never constructed from ActionIntent."""

    actor_id: str
    role: ApproverRole

    def __post_init__(self) -> None:
        if not self.actor_id.strip():
            raise ApprovalInboxError("actor_id must be non-empty")
        if not isinstance(self.role, ApproverRole):
            raise ApprovalInboxError("trusted role must be an ApproverRole")


@dataclass(frozen=True, slots=True)
class ApprovalRecord:
    request_id: str
    actor_id: str
    role: ApproverRole
    intent_digest: str
    approved_at: float
    expires_at: float
    consumed: bool = False


@dataclass(frozen=True, slots=True)
class _ParkedAction:
    digest: str
    required_role: ApproverRole | None


def required_approver_role_for(tool_name: str) -> ApproverRole | None:
    return TOOL_REQUIRED_APPROVER_ROLE.get(tool_name)


def trusted_caller(
    actor_id: str,
    *,
    directory: Mapping[str, ApproverRole] = FIXTURE_ACTORS,
    role: ApproverRole | None = None,
) -> TrustedCallerContext:
    """Build caller context from a trusted directory, not from model arguments."""
    if not actor_id.strip():
        raise ApprovalInboxError("actor_id must be non-empty")
    try:
        directory_role = directory[actor_id]
    except KeyError as exc:
        raise ApprovalInboxError("unknown actor") from exc
    resolved = directory_role if role is None else role
    if resolved is not directory_role:
        raise ApprovalInboxError("actor mismatch")
    return TrustedCallerContext(actor_id=actor_id, role=resolved)


def fixture_actor_for_role(role: ApproverRole) -> str:
    for actor_id, assigned in FIXTURE_ACTORS.items():
        if assigned is role:
            return actor_id
    raise ApprovalInboxError(f"no fixture actor for role {role.value}")


class ApprovalInbox:
    """Park REVIEW actions, accept one role-bound approval, consume once."""

    def __init__(
        self,
        *,
        actors: Mapping[str, ApproverRole] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._clock = clock or time.time
        self._lock = threading.Lock()
        self._actors: Mapping[str, ApproverRole] = MappingProxyType(
            dict(FIXTURE_ACTORS if actors is None else actors)
        )
        self._pending: dict[str, _ParkedAction] = {}
        self._approvals: dict[str, ApprovalRecord] = {}
        self._frozen_reason: str | None = None

    @property
    def frozen(self) -> bool:
        return self._frozen_reason is not None

    @property
    def frozen_reason(self) -> str | None:
        return self._frozen_reason

    def freeze(self, reason: str) -> None:
        if not reason.strip():
            raise ApprovalInboxError("freeze reason must be non-empty")
        with self._lock:
            self._frozen_reason = reason

    def park(
        self,
        action: ActionIntent,
        *,
        digest: str,
        required_role: ApproverRole | None = None,
    ) -> None:
        if not digest.strip():
            raise ApprovalInboxError("intent digest must be non-empty")
        role = required_role
        if role is None:
            role = required_approver_role_for(action.tool_name)
        with self._lock:
            if self._frozen_reason is not None:
                raise ApprovalInboxError("inbox is frozen")
            existing = self._pending.get(action.request_id)
            if existing is not None and existing.digest != digest:
                raise ApprovalInboxError("request_id already parked for another intent")
            self._pending[action.request_id] = _ParkedAction(
                digest=digest, required_role=role
            )

    def approve(
        self,
        request_id: str,
        *,
        caller: TrustedCallerContext,
        ttl_seconds: float = 3600.0,
    ) -> ApprovalRecord:
        if not request_id.strip():
            raise ApprovalInboxError("request_id must be non-empty")
        if ttl_seconds <= 0:
            raise ApprovalInboxError("ttl_seconds must be positive")
        with self._lock:
            if self._frozen_reason is not None:
                raise ApprovalInboxError("inbox is frozen")
            parked = self._pending.get(request_id)
            if parked is None:
                raise ApprovalInboxError("no parked action for request_id")
            if parked.required_role is None:
                raise ApprovalInboxError("no required approver role for parked action")
            directory_role = self._actors.get(caller.actor_id)
            if directory_role is None:
                raise ApprovalInboxError("unknown actor")
            if caller.role is not directory_role:
                raise ApprovalInboxError("actor mismatch")
            if caller.role is not parked.required_role:
                raise ApprovalInboxError("wrong role")
            current = self._approvals.get(request_id)
            if current is not None and current.consumed:
                raise ApprovalInboxError("approval already consumed")
            if current is not None:
                raise ApprovalInboxError("duplicate approval")
            now = self._clock()
            record = ApprovalRecord(
                request_id=request_id,
                actor_id=caller.actor_id,
                role=caller.role,
                intent_digest=parked.digest,
                approved_at=now,
                expires_at=now + ttl_seconds,
            )
            self._approvals[request_id] = record
            return record

    def consume_if_approved(self, action: ActionIntent, *, digest: str) -> bool:
        with self._lock:
            if self._frozen_reason is not None:
                return False
            record = self._approvals.get(action.request_id)
            if record is None or record.consumed:
                return False
            if record.intent_digest != digest:
                return False
            if self._clock() >= record.expires_at:
                return False
            self._approvals[action.request_id] = ApprovalRecord(
                request_id=record.request_id,
                actor_id=record.actor_id,
                role=record.role,
                intent_digest=record.intent_digest,
                approved_at=record.approved_at,
                expires_at=record.expires_at,
                consumed=True,
            )
            return True

    def record_for(self, request_id: str) -> ApprovalRecord | None:
        with self._lock:
            return self._approvals.get(request_id)


class ApprovalBoundedGateway:
    """CompanyRiskGateway plus one-shot approval for REVIEW tools."""

    independent_execution = False

    def __init__(
        self,
        inner: CompanyRiskGateway,
        inbox: ApprovalInbox,
        *,
        digest_for: Callable[[ActionIntent], str],
    ) -> None:
        self._inner = inner
        self._inbox = inbox
        self._digest_for = digest_for
        self.handler_calls: list[str] = []

    @property
    def actions(self) -> list[ActionIntent]:
        return self._inner.actions

    @property
    def last_status(self) -> ExecutionStatus | None:
        return self._inner.last_status

    def decide_and_execute(
        self,
        action: ActionIntent,
        handler: Callable[[Mapping[str, Any]], Any],
    ) -> GovernedExecution:
        if self._inbox.frozen:
            execution = GovernedExecution(
                status=ExecutionStatus.CONTROL_FAILURE,
                request_id=action.request_id,
                tool_executed=False,
                reason=f"inbox_frozen:{self._inbox.frozen_reason}",
                retryable=False,
            )
            self._inner.last_status = execution.status
            return execution
        digest = self._digest_for(action)
        first = self._inner.decide_and_execute(action, handler)
        if first.status is not ExecutionStatus.REVIEW_REQUIRED:
            return first
        if self._inbox.consume_if_approved(action, digest=digest):
            output = handler(action.arguments)
            self.handler_calls.append(action.tool_name)
            self._inner.handler_calls.append(action.tool_name)
            execution = GovernedExecution(
                status=ExecutionStatus.EXECUTED,
                request_id=action.request_id,
                tool_executed=True,
                reason="approved_review",
                output=output,
            )
            self._inner.last_status = execution.status
            return execution
        record = self._inbox.record_for(action.request_id)
        if record is not None and record.consumed:
            execution = GovernedExecution(
                status=ExecutionStatus.BLOCKED,
                request_id=action.request_id,
                tool_executed=False,
                reason="approval_already_consumed",
            )
            self._inner.last_status = execution.status
            return execution
        self._inbox.park(
            action,
            digest=digest,
            required_role=required_approver_role_for(action.tool_name),
        )
        return first


__all__ = [
    "CODE_REVIEWER_ACTOR",
    "CONTROLLER_ACTOR",
    "COUNSEL_ACTOR",
    "FIXTURE_ACTORS",
    "TOOL_REQUIRED_APPROVER_ROLE",
    "ApprovalBoundedGateway",
    "ApprovalInbox",
    "ApprovalInboxError",
    "ApprovalRecord",
    "ApproverRole",
    "TrustedCallerContext",
    "fixture_actor_for_role",
    "required_approver_role_for",
    "trusted_caller",
]
