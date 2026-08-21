"""In-memory durable run store for deterministic tests."""

from __future__ import annotations

import threading

from .durable import (
    DurableRunState,
    RunStoreError,
    StoredRunRecord,
)


class InMemoryRunStore:
    """Thread-safe in-memory ``RunStore`` implementation."""

    def __init__(self) -> None:
        self._records: dict[str, StoredRunRecord] = {}
        self._lock = threading.Lock()

    def create(self, record: StoredRunRecord) -> None:
        with self._lock:
            if record.run_id in self._records:
                raise RunStoreError(f"run {record.run_id!r} already exists")
            self._records[record.run_id] = record

    def load(self, run_id: str) -> StoredRunRecord:
        with self._lock:
            try:
                return self._records[run_id]
            except KeyError as exc:
                raise RunStoreError(f"unknown run {run_id!r}") from exc

    def save(
        self,
        record: StoredRunRecord,
        *,
        expected_version: int,
    ) -> StoredRunRecord:
        with self._lock:
            current = self._records.get(record.run_id)
            if current is None:
                raise RunStoreError(f"unknown run {record.run_id!r}")
            if current.version != expected_version:
                raise RunStoreError("optimistic version conflict")
            updated = StoredRunRecord(
                schema_version=record.schema_version,
                version=expected_version + 1,
                run_id=record.run_id,
                durable_state=record.durable_state,
                profile_fingerprint=record.profile_fingerprint,
                task=record.task,
                run_metadata=record.run_metadata,
                started_at_utc=record.started_at_utc,
                expires_at_utc=record.expires_at_utc,
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
            self._records[record.run_id] = updated
            return updated

    def claim_tool_dispatch(
        self,
        run_id: str,
        *,
        expected_version: int,
    ) -> StoredRunRecord | None:
        with self._lock:
            current = self._records.get(run_id)
            if current is None:
                raise RunStoreError(f"unknown run {run_id!r}")
            if (
                current.version != expected_version
                or current.durable_state is not DurableRunState.TOOL_PREPARED
            ):
                return None
            updated = StoredRunRecord(
                schema_version=current.schema_version,
                version=expected_version + 1,
                run_id=current.run_id,
                durable_state=DurableRunState.TOOL_IN_FLIGHT,
                profile_fingerprint=current.profile_fingerprint,
                task=current.task,
                run_metadata=current.run_metadata,
                started_at_utc=current.started_at_utc,
                expires_at_utc=current.expires_at_utc,
                next_step=current.next_step,
                tool_calls=current.tool_calls,
                model_turns=current.model_turns,
                messages=current.messages,
                events=current.events,
                pending_request_id=current.pending_request_id,
                pending_idempotency_key=current.pending_idempotency_key,
                pending_action=current.pending_action,
                pending_tool_call_id=current.pending_tool_call_id,
                pending_tool_name=current.pending_tool_name,
                pending_execution_status=current.pending_execution_status,
                pending_execution_output=current.pending_execution_output,
                pending_execution_reason=current.pending_execution_reason,
                terminal_status=current.terminal_status,
                final_text=current.final_text,
                terminal_metadata=current.terminal_metadata,
            )
            self._records[run_id] = updated
            return updated


__all__ = ["InMemoryRunStore"]
