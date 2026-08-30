"""Append-only learning stores for insights signals and approvals."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .insights import (
    STORE_SCHEMA_VERSION,
    ApprovalRecord,
    InsightsError,
    LearningSignal,
    LearningStoreError,
    finalize_signal,
    parse_record_json,
    validate_appended_approval,
    validate_appended_signal,
    validate_approval_columns,
    validate_signal_columns,
    verify_v1_signal_columns,
    verify_v1_signal_payload,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS learning_store_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS learning_signals (
    signal_id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    kind TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    observed_at REAL NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    idempotency_digest TEXT NOT NULL,
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
    idempotency_digest TEXT NOT NULL,
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
        parse_record_json(raw)
        return LearningSignal.from_json(raw)
    except InsightsError as exc:
        raise LearningStoreError("stored learning signal is corrupted") from exc


def _load_approval_payload(raw: object) -> ApprovalRecord:
    if not isinstance(raw, str):
        raise LearningStoreError("stored approval record is corrupted")
    try:
        parse_record_json(raw)
        return ApprovalRecord.from_json(raw)
    except InsightsError as exc:
        raise LearningStoreError("stored approval record is corrupted") from exc


def _table_names(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    return {str(name) for (name,) in rows}


def _read_store_schema_version(connection: sqlite3.Connection) -> int | None:
    tables = _table_names(connection)
    if "learning_store_meta" not in tables:
        if "learning_signals" in tables or "learning_approvals" in tables:
            return 1
        return None
    row = connection.execute(
        "SELECT value FROM learning_store_meta WHERE key = 'schema_version'"
    ).fetchone()
    if row is None:
        return 1
    try:
        return int(row[0])
    except (TypeError, ValueError) as exc:
        raise LearningStoreError("unsupported learning store schema") from exc


def _require_current_schema(connection: sqlite3.Connection) -> None:
    version = _read_store_schema_version(connection)
    if version is None:
        raise LearningStoreError("learning store schema is missing")
    if version != STORE_SCHEMA_VERSION:
        raise LearningStoreError(
            f"unsupported learning store schema v{version}; explicit migration required"
        )


class InMemoryLearningStore:
    """Thread-safe in-memory append-only learning store."""

    def __init__(self) -> None:
        self._signals: dict[str, LearningSignal] = {}
        self._signals_by_key: dict[str, LearningSignal] = {}
        self._approvals: dict[str, ApprovalRecord] = {}
        self._approvals_by_key: dict[str, ApprovalRecord] = {}
        self._lock = threading.Lock()

    def append_signal(self, signal: LearningSignal) -> LearningSignal:
        signal = validate_appended_signal(signal)
        with self._lock:
            existing = self._signals_by_key.get(signal.idempotency_key)
            if existing is not None:
                if existing.idempotency_digest != signal.idempotency_digest:
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
        approval = validate_appended_approval(approval)
        with self._lock:
            existing = self._approvals_by_key.get(approval.idempotency_key)
            if existing is not None:
                if existing.idempotency_digest != approval.idempotency_digest:
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
        read_only: bool = False,
    ) -> None:
        self._path = str(Path(path))
        self._read_only = read_only
        if read_only:
            resolved = Path(self._path)
            if not resolved.is_file():
                raise LearningStoreError("learning store does not exist")
            uri = resolved.resolve().as_uri() + "?mode=ro"
            self._connection = sqlite3.connect(
                uri,
                uri=True,
                isolation_level=None,
                timeout=busy_timeout_seconds,
            )
            self._connection.execute("PRAGMA query_only = ON")
            _require_current_schema(self._connection)
            return
        self._connection = sqlite3.connect(
            self._path,
            isolation_level=None,
            timeout=busy_timeout_seconds,
        )
        existing = _read_store_schema_version(self._connection)
        if existing is None:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.executescript(_SCHEMA)
            self._connection.execute(
                """
                INSERT OR REPLACE INTO learning_store_meta (key, value)
                VALUES ('schema_version', ?)
                """,
                (str(STORE_SCHEMA_VERSION),),
            )
            return
        if existing != STORE_SCHEMA_VERSION:
            self._connection.close()
            raise LearningStoreError(
                f"unsupported learning store schema v{existing}; "
                "explicit migration required"
            )
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript(_SCHEMA)

    @classmethod
    def open_readonly(
        cls,
        path: str | Path,
        *,
        busy_timeout_seconds: float = _DEFAULT_BUSY_TIMEOUT_SECONDS,
    ) -> SQLiteLearningStore:
        return cls(
            path,
            busy_timeout_seconds=busy_timeout_seconds,
            read_only=True,
        )

    def close(self) -> None:
        self._connection.close()

    def append_signal(self, signal: LearningSignal) -> LearningSignal:
        if self._read_only:
            raise LearningStoreError("insights report is read-only")
        signal = validate_appended_signal(signal)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            try:
                self._connection.execute(
                    """
                    INSERT INTO learning_signals (
                        signal_id, schema_version, kind, agent_id, content_hash,
                        observed_at, idempotency_key, idempotency_digest, record_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        signal.signal_id,
                        signal.schema_version,
                        signal.kind.value,
                        signal.agent_id,
                        signal.content_hash,
                        signal.observed_at,
                        signal.idempotency_key,
                        signal.idempotency_digest,
                        signal.to_json(),
                    ),
                )
            except sqlite3.IntegrityError:
                row = self._connection.execute(
                    """
                    SELECT schema_version, kind, agent_id, content_hash, observed_at,
                           idempotency_key, idempotency_digest, record_json
                    FROM learning_signals
                    WHERE idempotency_key = ?
                    """,
                    (signal.idempotency_key,),
                ).fetchone()
                if row is None:
                    raise LearningStoreError("learning signal already exists") from None
                existing = self._signal_from_row(row)
                if existing.idempotency_digest != signal.idempotency_digest:
                    raise LearningStoreError(
                        "idempotent retry payload mismatch"
                    ) from None
                self._connection.execute("COMMIT")
                return existing
            self._connection.execute("COMMIT")
            return signal
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

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
                   idempotency_key, idempotency_digest, record_json
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
                   observed_at, idempotency_key, idempotency_digest, record_json
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
        if self._read_only:
            raise LearningStoreError("insights report is read-only")
        approval = validate_appended_approval(approval)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            try:
                self._connection.execute(
                    """
                    INSERT INTO learning_approvals (
                        approval_id, schema_version, candidate_hash,
                        profile_fingerprint, reviewer_id, content_hash,
                        approved_at, idempotency_key, idempotency_digest, record_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        approval.idempotency_digest,
                        approval.to_json(),
                    ),
                )
            except sqlite3.IntegrityError:
                row = self._connection.execute(
                    """
                    SELECT schema_version, candidate_hash, profile_fingerprint,
                           reviewer_id, content_hash, approved_at, idempotency_key,
                           idempotency_digest, record_json
                    FROM learning_approvals
                    WHERE idempotency_key = ?
                    """,
                    (approval.idempotency_key,),
                ).fetchone()
                if row is None:
                    raise LearningStoreError("approval already exists") from None
                existing = self._approval_from_row(row)
                if existing.idempotency_digest != approval.idempotency_digest:
                    raise LearningStoreError(
                        "idempotent retry payload mismatch"
                    ) from None
                self._connection.execute("COMMIT")
                return existing
            self._connection.execute("COMMIT")
            return approval
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

    def list_approvals(
        self,
        *,
        agent_id: str | None = None,
    ) -> tuple[ApprovalRecord, ...]:
        rows = self._connection.execute(
            """
            SELECT schema_version, candidate_hash, profile_fingerprint, reviewer_id,
                   content_hash, approved_at, idempotency_key, idempotency_digest,
                   record_json
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
                   reviewer_id, content_hash, approved_at, idempotency_key,
                   idempotency_digest, record_json
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
            idempotency_digest,
            raw,
        ) = row
        signal = _load_signal_payload(raw)
        if not isinstance(kind, str) or not isinstance(agent_id, str):
            raise LearningStoreError("stored learning signal is corrupted")
        if not isinstance(content_hash, str) or not isinstance(idempotency_key, str):
            raise LearningStoreError("stored learning signal is corrupted")
        if not isinstance(idempotency_digest, str):
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
            idempotency_digest=idempotency_digest,
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
            idempotency_digest,
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
        if not isinstance(idempotency_key, str) or not isinstance(
            idempotency_digest, str
        ):
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
            idempotency_digest=idempotency_digest,
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


_V1_SIGNAL_COLUMNS = (
    "signal_id",
    "schema_version",
    "kind",
    "agent_id",
    "content_hash",
    "observed_at",
    "idempotency_key",
    "record_json",
)
_V1_ALLOWED_TABLES = frozenset(
    {"learning_signals", "learning_approvals", "sqlite_sequence"}
)
_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


def _paths_resolve_same(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    if left.exists() and right.exists():
        try:
            return os.path.samefile(left, right)
        except OSError:
            return False
    return False


def _sidecar_paths(path: Path) -> tuple[Path, ...]:
    return tuple(Path(str(path) + suffix) for suffix in _SQLITE_SIDECAR_SUFFIXES)


def _cleanup_sqlite_artifacts(path: Path) -> None:
    for candidate in (path, *_sidecar_paths(path)):
        try:
            if candidate.exists() or candidate.is_symlink():
                candidate.unlink()
        except OSError:
            continue


def _column_names(connection: sqlite3.Connection, table: str) -> tuple[str, ...]:
    rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
    return tuple(str(row[1]) for row in rows)


def migrate_learning_store_v1_to_v2(
    source: str | Path, destination: str | Path
) -> None:
    """Explicit verified rewrite of a v1 store. Never invoked on ordinary open."""
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file():
        raise LearningStoreError("learning store does not exist")
    if _paths_resolve_same(source_path, destination_path):
        raise LearningStoreError("migration source and destination must differ")
    if destination_path.exists() or any(
        sidecar.exists() for sidecar in _sidecar_paths(destination_path)
    ):
        raise LearningStoreError("migration destination already exists")

    uri = source_path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    verified: list[dict[str, Any]] = []
    try:
        connection.execute("PRAGMA query_only = ON")
        version = _read_store_schema_version(connection)
        if version != 1:
            raise LearningStoreError("explicit migration requires a v1 store")
        tables = _table_names(connection)
        unexpected = tables - _V1_ALLOWED_TABLES
        if unexpected:
            raise LearningStoreError("v1 store schema is invalid")
        if "learning_signals" not in tables:
            raise LearningStoreError("explicit migration requires a v1 store")
        if _column_names(connection, "learning_signals") != _V1_SIGNAL_COLUMNS:
            raise LearningStoreError("v1 store schema is invalid")
        approval_rows: list[tuple[object, ...]] = []
        if "learning_approvals" in tables:
            approval_rows = connection.execute(
                "SELECT record_json FROM learning_approvals"
            ).fetchall()
        if approval_rows:
            raise LearningStoreError(
                "v1 approvals cannot be silently reinterpreted; "
                "re-approve after signal migration"
            )
        signal_rows = connection.execute(
            """
            SELECT signal_id, schema_version, kind, agent_id, content_hash,
                   observed_at, idempotency_key, record_json
            FROM learning_signals
            """
        ).fetchall()
        for (
            signal_id,
            schema_version,
            kind,
            agent_id,
            content_hash,
            observed_at,
            idempotency_key,
            raw,
        ) in signal_rows:
            if not isinstance(raw, str):
                raise LearningStoreError("v1 signal is not verifiable")
            try:
                payload = parse_record_json(raw)
                fields = verify_v1_signal_payload(payload, raw)
                verify_v1_signal_columns(
                    payload,
                    signal_id=signal_id,
                    schema_version=schema_version,
                    kind=kind,
                    agent_id=agent_id,
                    content_hash=content_hash,
                    observed_at=observed_at,
                    idempotency_key=idempotency_key,
                )
            except InsightsError as exc:
                raise LearningStoreError("v1 signal is not verifiable") from exc
            verified.append(fields)
    finally:
        connection.close()

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.migrate-",
        suffix=".db",
        dir=destination_path.parent,
    )
    os.close(handle)
    tmp_path = Path(tmp_name)
    published = False
    try:
        upgraded = SQLiteLearningStore(tmp_path)
        try:
            for fields in verified:
                upgraded.append_signal(
                    finalize_signal(
                        kind=fields["kind"],
                        agent_id=fields["agent_id"],
                        idempotency_key=fields["idempotency_key"],
                        observed_at=fields["observed_at"],
                        reviewer_id=fields["reviewer_id"],
                        feedback_text=fields["feedback_text"],
                        run_id=fields["run_id"],
                        run_status=fields["run_status"],
                        tool_name=fields["tool_name"],
                        execution_status=fields["execution_status"],
                    )
                )
        finally:
            upgraded.close()
        check = SQLiteLearningStore(tmp_path)
        try:
            loaded = check.list_signals()
            if len(loaded) != len(verified):
                raise LearningStoreError("migrated store failed verification")
            by_key = {item.idempotency_key: item for item in loaded}
            for fields in verified:
                signal = by_key.get(str(fields["idempotency_key"]))
                if signal is None:
                    raise LearningStoreError("migrated store failed verification")
                validate_appended_signal(signal)
                if (
                    signal.schema_version != STORE_SCHEMA_VERSION
                    or signal.kind != fields["kind"]
                    or signal.agent_id != fields["agent_id"]
                    or signal.feedback_text != fields["feedback_text"]
                    or signal.reviewer_id != fields["reviewer_id"]
                    or signal.run_id != fields["run_id"]
                    or signal.run_status != fields["run_status"]
                    or signal.tool_name != fields["tool_name"]
                    or signal.execution_status != fields["execution_status"]
                    or signal.observed_at != fields["observed_at"]
                ):
                    raise LearningStoreError("migrated store failed verification")
        finally:
            check.close()
        os.replace(tmp_path, destination_path)
        published = True
    except Exception:
        _cleanup_sqlite_artifacts(tmp_path)
        if not published:
            _cleanup_sqlite_artifacts(destination_path)
        raise
    _cleanup_sqlite_artifacts(tmp_path)


__all__ = [
    "InMemoryLearningStore",
    "ReadOnlyLearningStore",
    "SQLiteLearningStore",
    "migrate_learning_store_v1_to_v2",
]
