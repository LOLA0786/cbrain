"""Role-bound human approval inbox for REVIEW_REQUIRED company actions.

This module does not decide policy or authenticate people. A trusted identity
adapter supplies an :class:`ApprovalPrincipal`; the inbox binds that principal
to a deployment-owned role directory, the parked ActionIntent digest, and a
single consumption. Frozen inboxes never approve and never dispatch.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from cbrain import ActionIntent, ExecutionStatus, GovernedExecution

from .governance import CompanyRiskGateway


class ApprovalInboxError(ValueError):
    """The approval inbox rejected park, approve, or consume."""


class ApprovalRole(StrEnum):
    """Deployment-owned roles allowed to approve consequential actions."""

    CONTROLLER = "controller"
    COUNSEL = "counsel"
    CODE_REVIEWER = "code_reviewer"
    BUYER_LEAD = "buyer_lead"


@dataclass(frozen=True, slots=True)
class ApprovalPrincipal:
    """Identity and role asserted by a trusted authentication adapter."""

    actor_id: str
    role: ApprovalRole

    def __post_init__(self) -> None:
        if not self.actor_id.strip():
            raise ApprovalInboxError("actor_id must be non-empty")
        if not isinstance(self.role, ApprovalRole):
            raise ApprovalInboxError("role must be an ApprovalRole")


@dataclass(frozen=True, slots=True)
class PendingApproval:
    request_id: str
    intent_digest: str
    required_role: ApprovalRole


@dataclass(frozen=True, slots=True)
class ApprovalRecord:
    request_id: str
    actor_id: str
    actor_role: ApprovalRole
    intent_digest: str
    approved_at: float
    expires_at: float
    consumed: bool = False


class ApprovalInbox:
    """Park REVIEW actions, accept one role-bound approval, consume once."""

    def __init__(
        self,
        *,
        allowed_approvers: Mapping[ApprovalRole, Collection[str]],
        max_ttl_seconds: float,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._max_ttl_seconds = _require_finite_seconds(
            max_ttl_seconds, name="max_ttl_seconds"
        )
        if self._max_ttl_seconds <= 0:
            raise ApprovalInboxError("max_ttl_seconds must be positive")
        normalized: dict[ApprovalRole, frozenset[str]] = {}
        for role, actor_ids in allowed_approvers.items():
            if not isinstance(role, ApprovalRole):
                raise ApprovalInboxError("approver directory contains an invalid role")
            actors = frozenset(actor_id for actor_id in actor_ids if actor_id.strip())
            if len(actors) != len(actor_ids):
                raise ApprovalInboxError("approver actor_id must be non-empty")
            normalized[role] = actors
        self._allowed_approvers = MappingProxyType(normalized)
        self._clock = clock or time.time
        self._lock = threading.Lock()
        self._pending: dict[str, PendingApproval] = {}
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
        required_role: ApprovalRole,
    ) -> None:
        if not digest.strip():
            raise ApprovalInboxError("intent digest must be non-empty")
        if not isinstance(required_role, ApprovalRole):
            raise ApprovalInboxError("required_role must be an ApprovalRole")
        with self._lock:
            if self._frozen_reason is not None:
                raise ApprovalInboxError("inbox is frozen")
            pending = PendingApproval(
                request_id=action.request_id,
                intent_digest=digest,
                required_role=required_role,
            )
            existing = self._pending.get(action.request_id)
            if existing is not None and existing != pending:
                raise ApprovalInboxError(
                    "request_id already parked for another approval"
                )
            self._pending[action.request_id] = pending

    def approve(
        self,
        request_id: str,
        *,
        principal: ApprovalPrincipal,
        ttl_seconds: float = 3600.0,
    ) -> ApprovalRecord:
        if not request_id.strip():
            raise ApprovalInboxError("request_id must be non-empty")
        if not isinstance(principal, ApprovalPrincipal):
            raise ApprovalInboxError("principal must come from the identity adapter")
        ttl = _require_finite_seconds(ttl_seconds, name="ttl_seconds")
        if ttl <= 0:
            raise ApprovalInboxError("ttl_seconds must be positive")
        if ttl > self._max_ttl_seconds:
            raise ApprovalInboxError("ttl_seconds exceeds deployment maximum")
        with self._lock:
            if self._frozen_reason is not None:
                raise ApprovalInboxError("inbox is frozen")
            pending = self._pending.get(request_id)
            if pending is None:
                raise ApprovalInboxError("no parked action for request_id")
            if principal.role is not pending.required_role:
                raise ApprovalInboxError("principal role cannot approve this action")
            allowed = self._allowed_approvers.get(principal.role, frozenset())
            if principal.actor_id not in allowed:
                raise ApprovalInboxError("principal is not an allowed approver")
            if request_id in self._approvals:
                raise ApprovalInboxError("approval already recorded")
            now = self._clock()
            record = ApprovalRecord(
                request_id=request_id,
                actor_id=principal.actor_id,
                actor_role=principal.role,
                intent_digest=pending.intent_digest,
                approved_at=now,
                expires_at=now + ttl,
            )
            self._approvals[request_id] = record
            return record

    def consume_if_approved(
        self,
        action: ActionIntent,
        *,
        digest: str,
        required_role: ApprovalRole,
    ) -> bool:
        if not isinstance(required_role, ApprovalRole):
            raise ApprovalInboxError("required_role must be an ApprovalRole")
        with self._lock:
            if self._frozen_reason is not None:
                return False
            pending = self._pending.get(action.request_id)
            record = self._approvals.get(action.request_id)
            if pending is None or record is None or record.consumed:
                return False
            if pending.required_role is not required_role:
                return False
            if pending.intent_digest != digest or record.intent_digest != digest:
                return False
            if pending.required_role is not record.actor_role:
                return False
            if self._clock() >= record.expires_at:
                return False
            self._approvals[action.request_id] = ApprovalRecord(
                request_id=record.request_id,
                actor_id=record.actor_id,
                actor_role=record.actor_role,
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
    """CompanyRiskGateway plus one-shot, role-bound approval for REVIEW tools."""

    independent_execution = False

    def __init__(
        self,
        inner: CompanyRiskGateway,
        inbox: ApprovalInbox,
        *,
        digest_for: Callable[[ActionIntent], str],
        required_role_for: Callable[[ActionIntent], ApprovalRole | None],
    ) -> None:
        self._inner = inner
        self._inbox = inbox
        self._digest_for = digest_for
        self._required_role_for = required_role_for
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
        required_role = self._required_role_for(action)
        if required_role is None:
            execution = GovernedExecution(
                status=ExecutionStatus.CONTROL_FAILURE,
                request_id=action.request_id,
                tool_executed=False,
                reason="approval_role_not_configured",
                retryable=False,
            )
            self._inner.last_status = execution.status
            return execution
        if self._inbox.consume_if_approved(
            action, digest=digest, required_role=required_role
        ):
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
        if record is not None:
            execution = GovernedExecution(
                status=ExecutionStatus.CONTROL_FAILURE,
                request_id=action.request_id,
                tool_executed=False,
                reason="approval_role_mismatch",
                retryable=False,
            )
            self._inner.last_status = execution.status
            return execution
        self._inbox.park(action, digest=digest, required_role=required_role)
        return first


def _require_finite_seconds(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ApprovalInboxError(f"{name} must be a finite number")
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        raise ApprovalInboxError(f"{name} must be a finite number")
    return number


__all__ = [
    "ApprovalBoundedGateway",
    "ApprovalInbox",
    "ApprovalInboxError",
    "ApprovalPrincipal",
    "ApprovalRecord",
    "ApprovalRole",
    "PendingApproval",
]
