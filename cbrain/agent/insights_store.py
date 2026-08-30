"""Append-only learning stores for insights signals and approvals."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path

from .insights import (
    ApprovalRecord,
    InsightsError,
    LearningSignal,
    LearningStoreError,
    validate_approval_columns,
    validate_signal_columns,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS learning_signals (
    signal_id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    kind TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    observed_at REAL NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    record_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS learning_approvals (
    approval_id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    candidate_hash TEXT NOT NULL,
    profile_fingerprint TEXT NOT NULL,
    reviewer_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    approved_at REAL NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    record_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_learning_signals_agent
    ON learning_signals(agent_id, observed_at);
CREATE INDEX IF NOT EXISTS idx_learning_approvals_agent
    ON learning_approvals(candidate_hash, profile_fingerprint);
"""

_DEFAULT_BUSY_TIMEOUT_SECONDS = 5.0


def _unique_ids(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    unique: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str) or not item.strip():
            raise LearningStoreError(f"{field_name} must be non-empty strings")
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return tuple(unique)


def _signal_in_bounds(
    signal: LearningSignal,
    *,
    agent_id: str | None,
    since: float | None,
    until: float | None,
) -> bool:
    if agent_id is not None and signal.agent_id != agent_id:
        return False
    if since is not None and signal.observed_at < since:
        return False
    return until is None or signal.observed_at <= until


def _load_signal_payload(raw: object) -> LearningSignal:
    if not isinstance(raw, str):
        raise LearningStoreError("stored learning signal is corrupted")
    try:
        return LearningSignal.from_json(raw)
    except InsightsError as exc:
        raise LearningStoreError("stored learning signal is corrupted") from exc


def _load_approval_payload(raw: object) -> ApprovalRecord:
    if not isinstance(raw, str):
        raise LearningStoreError("stored approval record is corrupted")
    try:
        return ApprovalRecord.from_json(raw)
    except InsightsError as exc:
        raise LearningStoreError("stored approval record is corrupted") from exc


class InMemoryLearningStore:
    """Thread-safe in-memory append-only learning store."""

    def __init__(self) -> None:
        self._signals: dict[str, LearningSignal] = {}
        self._signals_by_key: dict[str, LearningSignal] = {}
        self._approvals: dict[str, ApprovalRecord] = {}
        self._approvals_by_key: dict[str, ApprovalRecord] = {}
        self._lock = threading.Lock()

    def append_signal(self, signal: LearningSignal) -> LearningSignal:
        with self._lock:
            existing = self._signals_by_key.get(signal.idempotency_key)
            if existing is not None:
                if existing.content_hash != signal.content_hash:
                    raise LearningStoreError("idempotent retry payload mismatch")
                return existing
            if signal.signal_id in self._signals:
                raise LearningStoreError("learning signal already exists")
            self._signals[signal.signal_id] = signal
            self._signals_by_key[signal.idempotency_key] = signal
            return signal

    def list_signals(
        self,
        *,
        agent_id: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> tuple[LearningSignal, ...]:
        with self._lock:
            selected = [
                signal
                for signal in self._signals.values()
                if _signal_in_bounds(
                    signal, agent_id=agent_id, since=since, until=until
                )
            ]
        return tuple(
            sorted(selected, key=lambda item: (item.observed_at, item.signal_id))
        )

    def load_signals(self, signal_ids: Sequence[str]) -> tuple[LearningSignal, ...]:
        unique = _unique_ids(signal_ids, "signal ids")
        with self._lock:
            try:
                return tuple(self._signals[item] for item in unique)
            except KeyError as exc:
                raise LearningStoreError("missing learning evidence") from exc

    def append_approval(self, approval: ApprovalRecord) -> ApprovalRecord:
        with self._lock:
            existing = self._approvals_by_key.get(approval.idempotency_key)
            if existing is not None:
                if existing.content_hash != approval.content_hash:
                    raise LearningStoreError("idempotent retry payload mismatch")
                return existing
            if approval.approval_id in self._approvals:
                raise LearningStoreError("approval already exists")
            self._approvals[approval.approval_id] = approval
            self._approvals_by_key[approval.idempotency_key] = approval
            return approval

    def list_approvals(
        self,
        *,
        agent_id: str | None = None,
    ) -> tuple[ApprovalRecord, ...]:
        with self._lock:
            selected = [
                approval
                for approval in self._approvals.values()
                if agent_id is None or approval.agent_id == agent_id
            ]
        return tuple(sorted(selected, key=lambda item: item.approval_id))

    def load_approvals(self, approval_ids: Sequence[str]) -> tuple[ApprovalRecord, ...]:
        unique = _unique_ids(approval_ids, "approval ids")
        with self._lock:
            try:
                return tuple(self._approvals[item] for item in unique)
            except KeyError as exc:
                raise LearningStoreError("missing approval evidence") from exc


class SQLiteLearningStore:
    """Crash-safe append-only learning store backed by SQLite WAL."""

    def __init__(
        self,
        path: str | Path,
        *,
        busy_timeout_seconds: float = _DEFAULT_BUSY_TIMEOUT_SECONDS,
    ) -> None:
        self._path = str(path)
        self._connection = sqlite3.connect(
            self._path,
            isolation_level=None,
            timeout=busy_timeout_seconds,
        )
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript(_SCHEMA)

    def close(self) -> None:
        self._connection.close()

    def append_signal(self, signal: LearningSignal) -> LearningSignal:
        row = self._connection.execute(
            """
            SELECT schema_version, kind, agent_id, content_hash, observed_at,
                   idempotency_key, record_json
            FROM learning_signals
            WHERE idempotency_key = ?
            """,
            (signal.idempotency_key,),
        ).fetchone()
        if row is not None:
            existing = self._signal_from_row(row)
            if existing.content_hash != signal.content_hash:
                raise LearningStoreError("idempotent retry payload mismatch")
            return existing
        try:
            self._connection.execute(
                """
                INSERT INTO learning_signals (
                    signal_id, schema_version, kind, agent_id, content_hash,
                    observed_at, idempotency_key, record_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal.signal_id,
                    signal.schema_version,
                    signal.kind.value,
                    signal.agent_id,
                    signal.content_hash,
                    signal.observed_at,
                    signal.idempotency_key,
                    signal.to_json(),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise LearningStoreError("learning signal already exists") from exc
        return signal

    def list_signals(
        self,
        *,
        agent_id: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> tuple[LearningSignal, ...]:
        clauses = []
        params: list[object] = []
        if agent_id is not None:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if since is not None:
            clauses.append("observed_at >= ?")
            params.append(since)
        if until is not None:
            clauses.append("observed_at <= ?")
            params.append(until)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._connection.execute(
            f"""
            SELECT schema_version, kind, agent_id, content_hash, observed_at,
                   idempotency_key, record_json
            FROM learning_signals
            {where}
            ORDER BY observed_at ASC, signal_id ASC
            """,
            params,
        ).fetchall()
        return tuple(self._signal_from_row(row) for row in rows)

    def load_signals(self, signal_ids: Sequence[str]) -> tuple[LearningSignal, ...]:
        unique = _unique_ids(signal_ids, "signal ids")
        if not unique:
            return ()
        placeholders = ", ".join("?" for _ in unique)
        rows = self._connection.execute(
            f"""
            SELECT signal_id, schema_version, kind, agent_id, content_hash,
                   observed_at, idempotency_key, record_json
            FROM learning_signals
            WHERE signal_id IN ({placeholders})
            """,
            unique,
        ).fetchall()
        by_id = {row[0]: self._signal_from_row(row[1:]) for row in rows}
        missing = [item for item in unique if item not in by_id]
        if missing:
            raise LearningStoreError("missing learning evidence")
        return tuple(by_id[item] for item in unique)

    def append_approval(self, approval: ApprovalRecord) -> ApprovalRecord:
        row = self._connection.execute(
            """
            SELECT schema_version, candidate_hash, profile_fingerprint, reviewer_id,
                   content_hash, approved_at, idempotency_key, record_json
            FROM learning_approvals
            WHERE idempotency_key = ?
            """,
            (approval.idempotency_key,),
        ).fetchone()
        if row is not None:
            existing = self._approval_from_row(row)
            if existing.content_hash != approval.content_hash:
                raise LearningStoreError("idempotent retry payload mismatch")
            return existing
        try:
            self._connection.execute(
                """
                INSERT INTO learning_approvals (
                    approval_id, schema_version, candidate_hash, profile_fingerprint,
                    reviewer_id, content_hash, approved_at, idempotency_key, record_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    approval.approval_id,
                    approval.schema_version,
                    approval.candidate_hash,
                    approval.profile_fingerprint,
                    approval.reviewer_id,
                    approval.content_hash,
                    approval.approved_at,
                    approval.idempotency_key,
                    approval.to_json(),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise LearningStoreError("approval already exists") from exc
        return approval

    def list_approvals(
        self,
        *,
        agent_id: str | None = None,
    ) -> tuple[ApprovalRecord, ...]:
        rows = self._connection.execute(
            """
            SELECT schema_version, candidate_hash, profile_fingerprint, reviewer_id,
                   content_hash, approved_at, idempotency_key, record_json
            FROM learning_approvals
            ORDER BY approval_id ASC
            """
        ).fetchall()
        approvals = tuple(self._approval_from_row(row) for row in rows)
        if agent_id is None:
            return approvals
        return tuple(item for item in approvals if item.agent_id == agent_id)

    def load_approvals(self, approval_ids: Sequence[str]) -> tuple[ApprovalRecord, ...]:
        unique = _unique_ids(approval_ids, "approval ids")
        if not unique:
            return ()
        placeholders = ", ".join("?" for _ in unique)
        rows = self._connection.execute(
            f"""
            SELECT approval_id, schema_version, candidate_hash, profile_fingerprint,
                   reviewer_id, content_hash, approved_at, idempotency_key, record_json
            FROM learning_approvals
            WHERE approval_id IN ({placeholders})
            """,
            unique,
        ).fetchall()
        by_id = {row[0]: self._approval_from_row(row[1:]) for row in rows}
        missing = [item for item in unique if item not in by_id]
        if missing:
            raise LearningStoreError("missing approval evidence")
        return tuple(by_id[item] for item in unique)

    def _signal_from_row(self, row: tuple[object, ...]) -> LearningSignal:
        (
            schema_version,
            kind,
            agent_id,
            content_hash,
            observed_at,
            idempotency_key,
            raw,
        ) = row
        signal = _load_signal_payload(raw)
        if not isinstance(kind, str) or not isinstance(agent_id, str):
            raise LearningStoreError("stored learning signal is corrupted")
        if not isinstance(content_hash, str) or not isinstance(idempotency_key, str):
            raise LearningStoreError("stored learning signal is corrupted")
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise LearningStoreError("stored learning signal is corrupted")
        if not isinstance(observed_at, (int, float)) or isinstance(observed_at, bool):
            raise LearningStoreError("stored learning signal is corrupted")
        validate_signal_columns(
            signal,
            schema_version=schema_version,
            kind=kind,
            agent_id=agent_id,
            content_hash=content_hash,
            observed_at=float(observed_at),
            idempotency_key=idempotency_key,
        )
        return signal

    def _approval_from_row(self, row: tuple[object, ...]) -> ApprovalRecord:
        (
            schema_version,
            candidate_hash,
            profile_fingerprint,
            reviewer_id,
            content_hash,
            approved_at,
            idempotency_key,
            raw,
        ) = row
        approval = _load_approval_payload(raw)
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise LearningStoreError("stored approval record is corrupted")
        if not isinstance(candidate_hash, str) or not isinstance(
            profile_fingerprint, str
        ):
            raise LearningStoreError("stored approval record is corrupted")
        if not isinstance(reviewer_id, str) or not isinstance(content_hash, str):
            raise LearningStoreError("stored approval record is corrupted")
        if not isinstance(idempotency_key, str):
            raise LearningStoreError("stored approval record is corrupted")
        if not isinstance(approved_at, (int, float)) or isinstance(approved_at, bool):
            raise LearningStoreError("stored approval record is corrupted")
        validate_approval_columns(
            approval,
            schema_version=schema_version,
            candidate_hash=candidate_hash,
            profile_fingerprint=profile_fingerprint,
            reviewer_id=reviewer_id,
            content_hash=content_hash,
            approved_at=float(approved_at),
            idempotency_key=idempotency_key,
        )
        return approval


class ReadOnlyLearningStore:
    """Read-only facade used by report generation."""

    def __init__(self, inner: InMemoryLearningStore | SQLiteLearningStore) -> None:
        self._inner = inner

    def append_signal(self, signal: LearningSignal) -> LearningSignal:
        raise LearningStoreError("insights report is read-only")

    def list_signals(
        self,
        *,
        agent_id: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> tuple[LearningSignal, ...]:
        return self._inner.list_signals(agent_id=agent_id, since=since, until=until)

    def load_signals(self, signal_ids: Sequence[str]) -> tuple[LearningSignal, ...]:
        return self._inner.load_signals(signal_ids)

    def append_approval(self, approval: ApprovalRecord) -> ApprovalRecord:
        raise LearningStoreError("insights report is read-only")

    def list_approvals(
        self,
        *,
        agent_id: str | None = None,
    ) -> tuple[ApprovalRecord, ...]:
        return self._inner.list_approvals(agent_id=agent_id)

    def load_approvals(self, approval_ids: Sequence[str]) -> tuple[ApprovalRecord, ...]:
        return self._inner.load_approvals(approval_ids)


__all__ = [
    "InMemoryLearningStore",
    "ReadOnlyLearningStore",
    "SQLiteLearningStore",
]
