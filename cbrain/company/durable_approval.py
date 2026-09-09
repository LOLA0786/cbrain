"""Durable REVIEW parking with conflict-safe worker leases.

Lease expiry returns the action to PARKED. It is never proof that an
external write did not happen, and it never auto-dispatches.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from cbrain import ActionIntent
from cbrain.company.approval import (
    ApprovalInboxError,
    ApprovalPrincipal,
    ApprovalRole,
)


class ReviewState(StrEnum):
    PARKED = "PARKED"
    LEASED = "LEASED"
    APPROVED = "APPROVED"
    CONSUMED = "CONSUMED"
    APPROVAL_EXPIRED = "APPROVAL_EXPIRED"
    FROZEN = "FROZEN"


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    organisation_id: str
    request_id: str
    intent_digest: str
    required_role: ApprovalRole
    state: ReviewState
    lease_owner: str | None
    lease_expires_at: float | None
    actor_id: str | None
    approved_at: float | None
    approval_expires_at: float | None
    consumed_at: float | None
    frozen_reason: str | None
    version: int


_SCHEMA = """
CREATE TABLE IF NOT EXISTS durable_reviews (
    organisation_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    intent_digest TEXT NOT NULL,
    required_role TEXT NOT NULL,
    state TEXT NOT NULL,
    lease_owner TEXT,
    lease_expires_at REAL,
    actor_id TEXT,
    approved_at REAL,
    approval_expires_at REAL,
    consumed_at REAL,
    frozen_reason TEXT,
    version INTEGER NOT NULL,
    event_log TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (organisation_id, request_id),
    CHECK (length(organisation_id) > 0),
    CHECK (length(request_id) > 0),
    CHECK (length(intent_digest) > 0),
    CHECK (version >= 1)
);
"""


class DurableReviewStore:
    """SQLite-backed REVIEW state shared across workers."""

    def __init__(
        self,
        path: str | Path,
        *,
        organisation_id: str,
        allowed_approvers: Mapping[ApprovalRole, Collection[str]],
        max_ttl_seconds: float,
        max_lease_seconds: float,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not organisation_id.strip():
            raise ApprovalInboxError("organisation_id must be non-empty")
        if max_ttl_seconds <= 0 or max_lease_seconds <= 0:
            raise ApprovalInboxError("TTL and lease windows must be positive")
        self._organisation_id = organisation_id
        self._max_ttl_seconds = float(max_ttl_seconds)
        self._max_lease_seconds = float(max_lease_seconds)
        self._clock = clock or time.time
        normalized: dict[ApprovalRole, frozenset[str]] = {}
        for role, actor_ids in allowed_approvers.items():
            if not isinstance(role, ApprovalRole):
                raise ApprovalInboxError("approver directory contains an invalid role")
            actors = frozenset(a for a in actor_ids if a.strip())
            if len(actors) != len(tuple(actor_ids)):
                raise ApprovalInboxError("approver actor_id must be non-empty")
            normalized[role] = actors
        self._allowed_approvers = MappingProxyType(normalized)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(
            str(path),
            isolation_level=None,
            check_same_thread=False,
            timeout=5.0,
        )
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.executescript(_SCHEMA)

    def close(self) -> None:
        self._connection.close()

    def park(
        self,
        action: ActionIntent,
        *,
        digest: str,
        required_role: ApprovalRole,
    ) -> ReviewRecord:
        if not digest.strip():
            raise ApprovalInboxError("intent digest must be non-empty")
        if not isinstance(required_role, ApprovalRole):
            raise ApprovalInboxError("required_role must be an ApprovalRole")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                existing = self._load(action.request_id)
                if existing is not None:
                    if existing.state is ReviewState.FROZEN:
                        raise ApprovalInboxError("inbox is frozen")
                    if (
                        existing.intent_digest != digest
                        or existing.required_role is not required_role
                    ):
                        raise ApprovalInboxError(
                            "request_id already parked for another approval"
                        )
                    if existing.state in {
                        ReviewState.PARKED,
                        ReviewState.LEASED,
                        ReviewState.APPROVED,
                    }:
                        self._connection.execute("COMMIT")
                        return existing
                    raise ApprovalInboxError(
                        f"cannot park request in state {existing.state.value}"
                    )
                record = ReviewRecord(
                    organisation_id=self._organisation_id,
                    request_id=action.request_id,
                    intent_digest=digest,
                    required_role=required_role,
                    state=ReviewState.PARKED,
                    lease_owner=None,
                    lease_expires_at=None,
                    actor_id=None,
                    approved_at=None,
                    approval_expires_at=None,
                    consumed_at=None,
                    frozen_reason=None,
                    version=1,
                )
                self._insert(record, event="parked")
                self._connection.execute("COMMIT")
                return record
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def claim_lease(
        self,
        request_id: str,
        *,
        worker_id: str,
        lease_seconds: float | None = None,
    ) -> ReviewRecord:
        if not worker_id.strip():
            raise ApprovalInboxError("worker_id must be non-empty")
        lease_window = (
            self._max_lease_seconds if lease_seconds is None else float(lease_seconds)
        )
        if lease_window <= 0 or lease_window > self._max_lease_seconds:
            raise ApprovalInboxError("lease_seconds outside deployment maximum")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                record = self._expire_lease_if_needed(request_id)
                if record is None:
                    raise ApprovalInboxError("no parked action for request_id")
                if record.state is ReviewState.FROZEN:
                    raise ApprovalInboxError("inbox is frozen")
                now = self._clock()
                if record.state is ReviewState.LEASED:
                    if (
                        record.lease_owner == worker_id
                        and record.lease_expires_at is not None
                        and now < record.lease_expires_at
                    ):
                        renewed = self._replace(
                            record,
                            state=ReviewState.LEASED,
                            set_lease=True,
                            lease_owner=worker_id,
                            lease_expires_at=now + lease_window,
                            event="lease_renewed",
                        )
                        self._connection.execute("COMMIT")
                        return renewed
                    raise ApprovalInboxError("review lease held by another worker")
                if record.state is not ReviewState.PARKED:
                    raise ApprovalInboxError(
                        f"cannot lease request in state {record.state.value}"
                    )
                leased = self._replace(
                    record,
                    state=ReviewState.LEASED,
                    set_lease=True,
                    lease_owner=worker_id,
                    lease_expires_at=now + lease_window,
                    event="leased",
                )
                self._connection.execute("COMMIT")
                return leased
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def release_lease(self, request_id: str, *, worker_id: str) -> ReviewRecord:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                record = self._expire_lease_if_needed(request_id)
                if record is None:
                    raise ApprovalInboxError("no parked action for request_id")
                if record.state is not ReviewState.LEASED:
                    self._connection.execute("COMMIT")
                    return record
                if record.lease_owner != worker_id:
                    raise ApprovalInboxError("only the lease owner may release")
                parked = self._replace(
                    record,
                    state=ReviewState.PARKED,
                    clear_lease=True,
                    event="lease_released",
                )
                self._connection.execute("COMMIT")
                return parked
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def approve(
        self,
        request_id: str,
        *,
        principal: ApprovalPrincipal,
        ttl_seconds: float = 3600.0,
    ) -> ReviewRecord:
        if not isinstance(principal, ApprovalPrincipal):
            raise ApprovalInboxError("principal must come from the identity adapter")
        ttl = float(ttl_seconds)
        if ttl <= 0 or ttl > self._max_ttl_seconds:
            raise ApprovalInboxError("ttl_seconds outside deployment maximum")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                record = self._expire_lease_if_needed(request_id)
                if record is None:
                    raise ApprovalInboxError("no parked action for request_id")
                if record.state is ReviewState.FROZEN:
                    raise ApprovalInboxError("inbox is frozen")
                if record.state not in {ReviewState.PARKED, ReviewState.LEASED}:
                    raise ApprovalInboxError(
                        f"cannot approve request in state {record.state.value}"
                    )
                if principal.role is not record.required_role:
                    raise ApprovalInboxError(
                        "principal role cannot approve this action"
                    )
                allowed = self._allowed_approvers.get(principal.role, frozenset())
                if principal.actor_id not in allowed:
                    raise ApprovalInboxError("principal is not an allowed approver")
                now = self._clock()
                approved = self._replace(
                    record,
                    state=ReviewState.APPROVED,
                    clear_lease=True,
                    set_approval=True,
                    actor_id=principal.actor_id,
                    approved_at=now,
                    approval_expires_at=now + ttl,
                    event="approved",
                )
                self._connection.execute("COMMIT")
                return approved
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def consume_if_approved(
        self,
        action: ActionIntent,
        *,
        digest: str,
        required_role: ApprovalRole,
    ) -> bool:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                record = self._expire_approval_if_needed(action.request_id)
                if record is None:
                    self._connection.execute("COMMIT")
                    return False
                if record.state is ReviewState.FROZEN:
                    self._connection.execute("COMMIT")
                    return False
                if record.state is not ReviewState.APPROVED:
                    self._connection.execute("COMMIT")
                    return False
                if (
                    record.required_role is not required_role
                    or record.intent_digest != digest
                ):
                    self._connection.execute("COMMIT")
                    return False
                if (
                    record.approval_expires_at is not None
                    and self._clock() >= record.approval_expires_at
                ):
                    self._replace(
                        record,
                        state=ReviewState.APPROVAL_EXPIRED,
                        event="approval_expired",
                    )
                    self._connection.execute("COMMIT")
                    return False
                self._replace(
                    record,
                    state=ReviewState.CONSUMED,
                    set_consumed=True,
                    consumed_at=self._clock(),
                    event="consumed",
                )
                self._connection.execute("COMMIT")
                return True
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def freeze(self, reason: str) -> None:
        if not reason.strip():
            raise ApprovalInboxError("freeze reason must be non-empty")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                rows = self._connection.execute(
                    """
                    SELECT request_id FROM durable_reviews
                    WHERE organisation_id = ?
                    """,
                    (self._organisation_id,),
                ).fetchall()
                for (request_id,) in rows:
                    record = self._load(request_id)
                    if record is None or record.state is ReviewState.FROZEN:
                        continue
                    self._replace(
                        record,
                        state=ReviewState.FROZEN,
                        clear_lease=True,
                        set_frozen=True,
                        frozen_reason=reason,
                        event="frozen",
                    )
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def get(self, request_id: str) -> ReviewRecord | None:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                record = self._expire_lease_if_needed(request_id)
                record = self._expire_approval_if_needed(request_id) or record
                self._connection.execute("COMMIT")
                return record
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def events(self, request_id: str) -> list[dict[str, Any]]:
        row = self._connection.execute(
            """
            SELECT event_log FROM durable_reviews
            WHERE organisation_id = ? AND request_id = ?
            """,
            (self._organisation_id, request_id),
        ).fetchone()
        if row is None:
            return []
        return list(json.loads(row[0]))

    def _expire_lease_if_needed(self, request_id: str) -> ReviewRecord | None:
        record = self._load(request_id)
        if record is None:
            return None
        if (
            record.state is ReviewState.LEASED
            and record.lease_expires_at is not None
            and self._clock() >= record.lease_expires_at
        ):
            # Lease expiry is not proof the action never happened.
            return self._replace(
                record,
                state=ReviewState.PARKED,
                clear_lease=True,
                event="lease_expired",
            )
        return record

    def _expire_approval_if_needed(self, request_id: str) -> ReviewRecord | None:
        record = self._load(request_id)
        if record is None:
            return None
        if (
            record.state is ReviewState.APPROVED
            and record.approval_expires_at is not None
            and self._clock() >= record.approval_expires_at
        ):
            return self._replace(
                record,
                state=ReviewState.APPROVAL_EXPIRED,
                event="approval_expired",
            )
        return record

    def _load(self, request_id: str) -> ReviewRecord | None:
        row = self._connection.execute(
            """
            SELECT organisation_id, request_id, intent_digest, required_role, state,
                   lease_owner, lease_expires_at, actor_id, approved_at,
                   approval_expires_at, consumed_at, frozen_reason, version
            FROM durable_reviews
            WHERE organisation_id = ? AND request_id = ?
            """,
            (self._organisation_id, request_id),
        ).fetchone()
        if row is None:
            return None
        return ReviewRecord(
            organisation_id=row[0],
            request_id=row[1],
            intent_digest=row[2],
            required_role=ApprovalRole(row[3]),
            state=ReviewState(row[4]),
            lease_owner=row[5],
            lease_expires_at=row[6],
            actor_id=row[7],
            approved_at=row[8],
            approval_expires_at=row[9],
            consumed_at=row[10],
            frozen_reason=row[11],
            version=row[12],
        )

    def _insert(self, record: ReviewRecord, *, event: str) -> None:
        events = [{"at": self._clock(), "event": event, "state": record.state.value}]
        self._connection.execute(
            """
            INSERT INTO durable_reviews (
                organisation_id, request_id, intent_digest, required_role, state,
                lease_owner, lease_expires_at, actor_id, approved_at,
                approval_expires_at, consumed_at, frozen_reason, version, event_log
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.organisation_id,
                record.request_id,
                record.intent_digest,
                record.required_role.value,
                record.state.value,
                record.lease_owner,
                record.lease_expires_at,
                record.actor_id,
                record.approved_at,
                record.approval_expires_at,
                record.consumed_at,
                record.frozen_reason,
                record.version,
                json.dumps(events),
            ),
        )

    def _replace(
        self,
        record: ReviewRecord,
        *,
        event: str,
        state: ReviewState | None = None,
        clear_lease: bool = False,
        lease_owner: str | None = None,
        lease_expires_at: float | None = None,
        set_lease: bool = False,
        actor_id: str | None = None,
        approved_at: float | None = None,
        approval_expires_at: float | None = None,
        set_approval: bool = False,
        consumed_at: float | None = None,
        set_consumed: bool = False,
        frozen_reason: str | None = None,
        set_frozen: bool = False,
    ) -> ReviewRecord:
        updated = ReviewRecord(
            organisation_id=record.organisation_id,
            request_id=record.request_id,
            intent_digest=record.intent_digest,
            required_role=record.required_role,
            state=record.state if state is None else state,
            lease_owner=(
                None
                if clear_lease
                else (lease_owner if set_lease else record.lease_owner)
            ),
            lease_expires_at=(
                None
                if clear_lease
                else (lease_expires_at if set_lease else record.lease_expires_at)
            ),
            actor_id=actor_id if set_approval else record.actor_id,
            approved_at=approved_at if set_approval else record.approved_at,
            approval_expires_at=(
                approval_expires_at if set_approval else record.approval_expires_at
            ),
            consumed_at=consumed_at if set_consumed else record.consumed_at,
            frozen_reason=frozen_reason if set_frozen else record.frozen_reason,
            version=record.version + 1,
        )
        events = self.events(record.request_id)
        events.append(
            {
                "at": self._clock(),
                "event": event,
                "state": updated.state.value,
                "version": updated.version,
            }
        )
        cursor = self._connection.execute(
            """
            UPDATE durable_reviews SET
                intent_digest = ?, required_role = ?, state = ?,
                lease_owner = ?, lease_expires_at = ?, actor_id = ?,
                approved_at = ?, approval_expires_at = ?, consumed_at = ?,
                frozen_reason = ?, version = ?, event_log = ?
            WHERE organisation_id = ? AND request_id = ? AND version = ?
            """,
            (
                updated.intent_digest,
                updated.required_role.value,
                updated.state.value,
                updated.lease_owner,
                updated.lease_expires_at,
                updated.actor_id,
                updated.approved_at,
                updated.approval_expires_at,
                updated.consumed_at,
                updated.frozen_reason,
                updated.version,
                json.dumps(events),
                record.organisation_id,
                record.request_id,
                record.version,
            ),
        )
        if cursor.rowcount != 1:
            raise ApprovalInboxError("concurrent review update conflict")
        return updated


__all__ = [
    "DurableReviewStore",
    "ReviewRecord",
    "ReviewState",
]
