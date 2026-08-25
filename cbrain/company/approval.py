"""Named-human approval inbox for REVIEW_REQUIRED company actions.

This does not decide policy. It records who approved a parked ActionIntent
and allows the existing gateway to run the handler once after approval.
Frozen inboxes never approve and never dispatch.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from cbrain import ActionIntent, ExecutionStatus, GovernedExecution

from .governance import CompanyRiskGateway


class ApprovalInboxError(ValueError):
    """The approval inbox rejected park, approve, or consume."""


@dataclass(frozen=True, slots=True)
class ApprovalRecord:
    request_id: str
    actor_id: str
    intent_digest: str
    approved_at: float
    expires_at: float
    consumed: bool = False


class ApprovalInbox:
    """Park REVIEW actions, accept one named approval, consume once."""

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.time
        self._lock = threading.Lock()
        self._pending: dict[str, str] = {}
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

    def park(self, action: ActionIntent, *, digest: str) -> None:
        if not digest.strip():
            raise ApprovalInboxError("intent digest must be non-empty")
        with self._lock:
            if self._frozen_reason is not None:
                raise ApprovalInboxError("inbox is frozen")
            existing = self._pending.get(action.request_id)
            if existing is not None and existing != digest:
                raise ApprovalInboxError("request_id already parked for another intent")
            self._pending[action.request_id] = digest

    def approve(
        self,
        request_id: str,
        *,
        actor_id: str,
        ttl_seconds: float = 3600.0,
    ) -> ApprovalRecord:
        if not request_id.strip() or not actor_id.strip():
            raise ApprovalInboxError("request_id and actor_id must be non-empty")
        if ttl_seconds <= 0:
            raise ApprovalInboxError("ttl_seconds must be positive")
        with self._lock:
            if self._frozen_reason is not None:
                raise ApprovalInboxError("inbox is frozen")
            digest = self._pending.get(request_id)
            if digest is None:
                raise ApprovalInboxError("no parked action for request_id")
            current = self._approvals.get(request_id)
            if current is not None and current.consumed:
                raise ApprovalInboxError("approval already consumed")
            now = self._clock()
            record = ApprovalRecord(
                request_id=request_id,
                actor_id=actor_id,
                intent_digest=digest,
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
        self._inbox.park(action, digest=digest)
        return first


__all__ = [
    "ApprovalBoundedGateway",
    "ApprovalInbox",
    "ApprovalInboxError",
    "ApprovalRecord",
]
