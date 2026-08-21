"""Durable run records, serialization, and store protocol."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from cbrain.models import Message, MessageRole

from .contracts import RunEvent, RunEventKind, RunInput, RunResult, RunStatus
from .profile import AgentProfile

STORE_SCHEMA_VERSION = 1


class RunStoreError(ValueError):
    """Stored run record is missing, inconsistent, or cannot be updated."""


class DurableRunState(StrEnum):
    RUNNING = "RUNNING"
    TOOL_PREPARED = "TOOL_PREPARED"
    TOOL_IN_FLIGHT = "TOOL_IN_FLIGHT"
    TOOL_COMPLETED = "TOOL_COMPLETED"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


_TERMINAL_DURABLE = frozenset(
    {
        DurableRunState.COMPLETED,
        DurableRunState.FAILED,
        DurableRunState.CANCELLED,
        DurableRunState.RECOVERY_REQUIRED,
    }
)


@dataclass(frozen=True, slots=True)
class StoredRunRecord:
    schema_version: int
    version: int
    run_id: str
    durable_state: DurableRunState
    profile_fingerprint: str
    task: str
    run_metadata: Mapping[str, Any]
    started_at: float
    deadline: float
    next_step: int
    tool_calls: int
    model_turns: int
    messages: tuple[Mapping[str, Any], ...]
    events: tuple[Mapping[str, Any], ...]
    pending_request_id: str | None = None
    pending_idempotency_key: str | None = None
    pending_action: Mapping[str, Any] | None = None
    pending_tool_call_id: str | None = None
    pending_tool_name: str | None = None
    pending_execution_status: str | None = None
    pending_execution_output: Any | None = None
    pending_execution_reason: str | None = None
    terminal_status: RunStatus | None = None
    final_text: str | None = None
    terminal_metadata: Mapping[str, Any] | None = None

    def to_json(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "version": self.version,
            "run_id": self.run_id,
            "durable_state": self.durable_state.value,
            "profile_fingerprint": self.profile_fingerprint,
            "task": self.task,
            "run_metadata": dict(self.run_metadata),
            "started_at": self.started_at,
            "deadline": self.deadline,
            "next_step": self.next_step,
            "tool_calls": self.tool_calls,
            "model_turns": self.model_turns,
            "messages": [dict(item) for item in self.messages],
            "events": [dict(item) for item in self.events],
            "pending_request_id": self.pending_request_id,
            "pending_idempotency_key": self.pending_idempotency_key,
            "pending_action": (
                dict(self.pending_action) if self.pending_action is not None else None
            ),
            "pending_tool_call_id": self.pending_tool_call_id,
            "pending_tool_name": self.pending_tool_name,
            "pending_execution_status": self.pending_execution_status,
            "pending_execution_output": self.pending_execution_output,
            "pending_execution_reason": self.pending_execution_reason,
            "terminal_status": (
                self.terminal_status.value if self.terminal_status is not None else None
            ),
            "final_text": self.final_text,
            "terminal_metadata": (
                dict(self.terminal_metadata)
                if self.terminal_metadata is not None
                else None
            ),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> StoredRunRecord:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RunStoreError("stored run record is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise RunStoreError("stored run record must be a JSON object")
        return from_mapping(payload)


class RunStore(Protocol):
    def create(self, record: StoredRunRecord) -> None: ...

    def load(self, run_id: str) -> StoredRunRecord: ...

    def save(
        self,
        record: StoredRunRecord,
        *,
        expected_version: int,
    ) -> StoredRunRecord: ...

    def claim_tool_dispatch(
        self,
        run_id: str,
        *,
        expected_version: int,
    ) -> StoredRunRecord | None: ...


def profile_fingerprint(profile: AgentProfile) -> str:
    limits = profile.limits
    payload = {
        "agent_id": profile.agent_id,
        "instructions": profile.instructions,
        "model_route": profile.model_route,
        "permitted_tools": sorted(profile.permitted_tools),
        "max_model_turns": profile.max_model_turns,
        "max_tool_calls": profile.max_tool_calls,
        "timeout_seconds": profile.timeout_seconds,
        "limits": {
            "max_context_messages": limits.max_context_messages,
            "max_model_text_chars": limits.max_model_text_chars,
            "max_observation_chars": limits.max_observation_chars,
            "max_output_tokens": limits.max_output_tokens,
            "max_task_chars": limits.max_task_chars,
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def serialize_message(message: Message) -> dict[str, Any]:
    return {
        "role": message.role.value,
        "content": message.content,
        "tool_call_id": message.tool_call_id,
        "tool_name": message.tool_name,
    }


def deserialize_message(payload: Mapping[str, Any]) -> Message:
    role = payload.get("role")
    content = payload.get("content")
    if not isinstance(role, str) or not isinstance(content, str):
        raise RunStoreError("stored message is invalid")
    return Message(
        role=MessageRole(role),
        content=content,
        tool_call_id=payload.get("tool_call_id"),
        tool_name=payload.get("tool_name"),
    )


def serialize_event(event: RunEvent) -> dict[str, Any]:
    return {
        "kind": event.kind.value,
        "step": event.step,
        "timestamp": event.timestamp,
        "detail": dict(event.detail),
    }


def deserialize_event(payload: Mapping[str, Any]) -> RunEvent:
    kind = payload.get("kind")
    step = payload.get("step")
    timestamp = payload.get("timestamp")
    detail = payload.get("detail")
    if (
        not isinstance(kind, str)
        or not isinstance(step, int)
        or not isinstance(timestamp, (int, float))
        or not isinstance(detail, Mapping)
    ):
        raise RunStoreError("stored event is invalid")
    return RunEvent(
        kind=RunEventKind(kind),
        step=step,
        timestamp=float(timestamp),
        detail=dict(detail),
    )


def record_to_run_result(record: StoredRunRecord) -> RunResult:
    if record.terminal_status is None:
        raise RunStoreError("stored run is not terminal")
    events = tuple(deserialize_event(item) for item in record.events)
    metadata = dict(record.terminal_metadata or record.run_metadata)
    return RunResult(
        run_id=record.run_id,
        status=record.terminal_status,
        final_text=record.final_text,
        tool_calls=record.tool_calls,
        model_turns=record.model_turns,
        events=events,
        metadata=metadata,
    )


def from_mapping(payload: Mapping[str, Any]) -> StoredRunRecord:
    schema_version = payload.get("schema_version")
    if schema_version != STORE_SCHEMA_VERSION:
        raise RunStoreError(
            f"unsupported stored run schema version: {schema_version!r}"
        )
    version = payload.get("version")
    run_id = payload.get("run_id")
    durable_state = payload.get("durable_state")
    profile_fingerprint = payload.get("profile_fingerprint")
    task = payload.get("task")
    run_metadata = payload.get("run_metadata")
    started_at = payload.get("started_at")
    deadline = payload.get("deadline")
    next_step = payload.get("next_step")
    tool_calls = payload.get("tool_calls")
    model_turns = payload.get("model_turns")
    messages = payload.get("messages")
    events = payload.get("events")
    if (
        not isinstance(version, int)
        or version < 0
        or not isinstance(run_id, str)
        or not run_id.strip()
        or not isinstance(durable_state, str)
        or not isinstance(profile_fingerprint, str)
        or not isinstance(task, str)
        or not isinstance(run_metadata, dict)
        or not isinstance(started_at, (int, float))
        or not isinstance(deadline, (int, float))
        or not isinstance(next_step, int)
        or not isinstance(tool_calls, int)
        or not isinstance(model_turns, int)
        or not isinstance(messages, list)
        or not isinstance(events, list)
    ):
        raise RunStoreError("stored run record is incomplete or inconsistent")
    try:
        state = DurableRunState(durable_state)
    except ValueError as exc:
        raise RunStoreError(f"unknown durable state {durable_state!r}") from exc
    terminal_status_raw = payload.get("terminal_status")
    terminal_status = (
        RunStatus(terminal_status_raw) if isinstance(terminal_status_raw, str) else None
    )
    terminal_metadata = payload.get("terminal_metadata")
    if terminal_metadata is not None and not isinstance(terminal_metadata, dict):
        raise RunStoreError("terminal_metadata must be a mapping")
    pending_action = payload.get("pending_action")
    if pending_action is not None and not isinstance(pending_action, dict):
        raise RunStoreError("pending_action must be a mapping")
    return StoredRunRecord(
        schema_version=schema_version,
        version=version,
        run_id=run_id,
        durable_state=state,
        profile_fingerprint=profile_fingerprint,
        task=task,
        run_metadata=run_metadata,
        started_at=float(started_at),
        deadline=float(deadline),
        next_step=next_step,
        tool_calls=tool_calls,
        model_turns=model_turns,
        messages=tuple(item for item in messages if isinstance(item, dict)),
        events=tuple(item for item in events if isinstance(item, dict)),
        pending_request_id=_optional_str(payload.get("pending_request_id")),
        pending_idempotency_key=_optional_str(payload.get("pending_idempotency_key")),
        pending_action=pending_action,
        pending_tool_call_id=_optional_str(payload.get("pending_tool_call_id")),
        pending_tool_name=_optional_str(payload.get("pending_tool_name")),
        pending_execution_status=_optional_str(payload.get("pending_execution_status")),
        pending_execution_output=payload.get("pending_execution_output"),
        pending_execution_reason=_optional_str(payload.get("pending_execution_reason")),
        terminal_status=terminal_status,
        final_text=_optional_str(payload.get("final_text")),
        terminal_metadata=terminal_metadata,
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RunStoreError("expected non-empty string field")
    return value


def new_running_record(
    *,
    run_id: str,
    profile: AgentProfile,
    run_input: RunInput,
    started_at: float,
    deadline: float,
    messages: tuple[Message, ...],
    events: tuple[RunEvent, ...],
) -> StoredRunRecord:
    return StoredRunRecord(
        schema_version=STORE_SCHEMA_VERSION,
        version=0,
        run_id=run_id,
        durable_state=DurableRunState.RUNNING,
        profile_fingerprint=profile_fingerprint(profile),
        task=run_input.task,
        run_metadata=dict(run_input.metadata or {}),
        started_at=started_at,
        deadline=deadline,
        next_step=1,
        tool_calls=0,
        model_turns=0,
        messages=tuple(serialize_message(item) for item in messages),
        events=tuple(serialize_event(item) for item in events),
    )


__all__ = [
    "DurableRunState",
    "RunStore",
    "RunStoreError",
    "STORE_SCHEMA_VERSION",
    "StoredRunRecord",
    "deserialize_event",
    "deserialize_message",
    "from_mapping",
    "new_running_record",
    "profile_fingerprint",
    "record_to_run_result",
    "serialize_event",
    "serialize_message",
]
