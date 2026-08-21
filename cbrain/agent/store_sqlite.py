"""SQLite durable run store using the Python standard library."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .durable import (
    DurableRunState,
    RunStoreError,
    StoredRunRecord,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS durable_runs (
    run_id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    version INTEGER NOT NULL,
    durable_state TEXT NOT NULL,
    record_json TEXT NOT NULL
);
"""


class SQLiteRunStore:
    """Crash-safe durable run store backed by SQLite."""

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._connection = sqlite3.connect(self._path, isolation_level=None)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript(_SCHEMA)

    def close(self) -> None:
        self._connection.close()

    def create(self, record: StoredRunRecord) -> None:
        try:
            self._connection.execute(
                """
                INSERT INTO durable_runs (
                    run_id, schema_version, version, durable_state, record_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.schema_version,
                    record.version,
                    record.durable_state.value,
                    record.to_json(),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise RunStoreError(f"run {record.run_id!r} already exists") from exc

    def load(self, run_id: str) -> StoredRunRecord:
        row = self._connection.execute(
            "SELECT record_json FROM durable_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise RunStoreError(f"unknown run {run_id!r}")
        raw = row[0]
        if not isinstance(raw, str):
            raise RunStoreError("stored run record is corrupted")
        return StoredRunRecord.from_json(raw)

    def save(
        self,
        record: StoredRunRecord,
        *,
        expected_version: int,
    ) -> StoredRunRecord:
        updated = StoredRunRecord(
            schema_version=record.schema_version,
            version=expected_version + 1,
            run_id=record.run_id,
            durable_state=record.durable_state,
            profile_fingerprint=record.profile_fingerprint,
            task=record.task,
            run_metadata=record.run_metadata,
            started_at=record.started_at,
            deadline=record.deadline,
            next_step=record.next_step,
            tool_calls=record.tool_calls,
            model_turns=record.model_turns,
            messages=record.messages,
            events=record.events,
            pending_request_id=record.pending_request_id,
            pending_idempotency_key=record.pending_idempotency_key,
            pending_action=record.pending_action,
            pending_tool_call_id=record.pending_tool_call_id,
            pending_tool_name=record.pending_tool_name,
            pending_execution_status=record.pending_execution_status,
            pending_execution_output=record.pending_execution_output,
            pending_execution_reason=record.pending_execution_reason,
            terminal_status=record.terminal_status,
            final_text=record.final_text,
            terminal_metadata=record.terminal_metadata,
        )
        cursor = self._connection.execute(
            """
            UPDATE durable_runs
            SET version = ?, durable_state = ?, record_json = ?
            WHERE run_id = ? AND version = ?
            """,
            (
                updated.version,
                updated.durable_state.value,
                updated.to_json(),
                updated.run_id,
                expected_version,
            ),
        )
        if cursor.rowcount != 1:
            raise RunStoreError("optimistic version conflict")
        return updated

    def claim_tool_dispatch(
        self,
        run_id: str,
        *,
        expected_version: int,
    ) -> StoredRunRecord | None:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                """
                SELECT version, durable_state, record_json
                FROM durable_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if row is None:
                raise RunStoreError(f"unknown run {run_id!r}")
            version, durable_state, raw = row
            prepared_state = DurableRunState.TOOL_PREPARED.value
            if version != expected_version or durable_state != prepared_state:
                self._connection.execute("ROLLBACK")
                return None
            record = StoredRunRecord.from_json(raw)
            updated = StoredRunRecord(
                schema_version=record.schema_version,
                version=expected_version + 1,
                run_id=record.run_id,
                durable_state=DurableRunState.TOOL_IN_FLIGHT,
                profile_fingerprint=record.profile_fingerprint,
                task=record.task,
                run_metadata=record.run_metadata,
                started_at=record.started_at,
                deadline=record.deadline,
                next_step=record.next_step,
                tool_calls=record.tool_calls,
                model_turns=record.model_turns,
                messages=record.messages,
                events=record.events,
                pending_request_id=record.pending_request_id,
                pending_idempotency_key=record.pending_idempotency_key,
                pending_action=record.pending_action,
                pending_tool_call_id=record.pending_tool_call_id,
                pending_tool_name=record.pending_tool_name,
                pending_execution_status=record.pending_execution_status,
                pending_execution_output=record.pending_execution_output,
                pending_execution_reason=record.pending_execution_reason,
                terminal_status=record.terminal_status,
                final_text=record.final_text,
                terminal_metadata=record.terminal_metadata,
            )
            cursor = self._connection.execute(
                """
                UPDATE durable_runs
                SET version = ?, durable_state = ?, record_json = ?
                WHERE run_id = ? AND version = ?
                """,
                (
                    updated.version,
                    updated.durable_state.value,
                    updated.to_json(),
                    run_id,
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                self._connection.execute("ROLLBACK")
                return None
            self._connection.execute("COMMIT")
            return updated
        except Exception:
            self._connection.execute("ROLLBACK")
            raise


__all__ = ["SQLiteRunStore"]
