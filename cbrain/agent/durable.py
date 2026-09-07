"""Durable run records, serialization, and store protocol."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from cbrain.models import (
    Message,
    MessageRole,
    ModelError,
    ProviderContinuation,
    ToolCall,
)

from .contracts import RunEvent, RunEventKind, RunInput, RunResult, RunStatus
from .profile import AgentProfile

STORE_SCHEMA_VERSION = 2


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

_PENDING_TOOL_STATES = frozenset(
    {
        DurableRunState.TOOL_PREPARED,
        DurableRunState.TOOL_IN_FLIGHT,
        DurableRunState.TOOL_COMPLETED,
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
    started_at_utc: float
    expires_at_utc: float
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
            "started_at_utc": self.started_at_utc,
            "expires_at_utc": self.expires_at_utc,
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


def compute_deadline_monotonic(
    *,
    expires_at_utc: float,
    wall_clock: Callable[[], float],
    monotonic_clock: Callable[[], float],
) -> float:
    remaining = expires_at_utc - wall_clock()
    return monotonic_clock() + max(0.0, remaining)


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
        "knowledge_required_for_tools": profile.knowledge_required_for_tools,
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
    payload: dict[str, Any] = {
        "role": message.role.value,
        "content": message.content,
        "tool_call_id": message.tool_call_id,
        "tool_name": message.tool_name,
    }
    if message.tool_call is not None:
        call = message.tool_call
        payload["tool_call"] = {
            "call_id": call.call_id,
            "name": call.name,
            "arguments": call.arguments,
        }
        if call.continuation is not None:
            turn = call.continuation
            payload["tool_call"]["continuation"] = {
                "provider": turn.provider,
                "model": turn.model,
                "wire_format": turn.wire_format,
                "message": turn.message,
            }
    return payload


def deserialize_message(payload: Mapping[str, Any]) -> Message:
    role = payload.get("role")
    content = payload.get("content")
    if not isinstance(role, str) or not isinstance(content, str):
        raise RunStoreError("stored message is invalid")
    try:
        message_role = MessageRole(role)
    except ValueError as exc:
        raise RunStoreError("stored message role is invalid") from exc
    try:
        tool_call = None
        if payload.get("tool_call") is not None:
            raw = payload["tool_call"]
            if not isinstance(raw, dict) or not isinstance(raw.get("arguments"), dict):
                raise RunStoreError("stored assistant call is invalid")
            continuation = None
            if raw.get("continuation") is not None:
                turn = raw["continuation"]
                if not isinstance(turn, dict) or not isinstance(
                    turn.get("message"), dict
                ):
                    raise RunStoreError("stored continuation is invalid")
                continuation = ProviderContinuation.capture(
                    provider=_required_text(turn.get("provider"), "provider"),
                    model=_required_text(turn.get("model"), "model"),
                    wire_format=_required_text(turn.get("wire_format"), "wire_format"),
                    message=turn["message"],
                )
            tool_call = ToolCall.capture(
                call_id=_required_text(raw.get("call_id"), "call_id"),
                name=_required_text(raw.get("name"), "name"),
                arguments=raw["arguments"],
                continuation=continuation,
            )
        return Message(
            role=message_role,
            content=content,
            tool_call_id=payload.get("tool_call_id"),
            tool_name=payload.get("tool_name"),
            tool_call=tool_call,
        )
    except ModelError as exc:
        raise RunStoreError("stored assistant history is invalid") from exc


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
    if not isinstance(kind, str):
        raise RunStoreError("stored event kind is invalid")
    step_value = _non_negative_int(step, "event step", minimum=0)
    timestamp_value = _finite_number(timestamp, "event timestamp")
    if not isinstance(detail, Mapping):
        raise RunStoreError("stored event detail is invalid")
    try:
        event_kind = RunEventKind(kind)
    except ValueError as exc:
        raise RunStoreError("stored event kind is invalid") from exc
    return RunEvent(
        kind=event_kind,
        step=step_value,
        timestamp=timestamp_value,
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


def in_flight_run_result(
    record: StoredRunRecord,
    *,
    reason: str = "tool execution is in flight or its outcome is unknown",
) -> RunResult:
    events = tuple(deserialize_event(item) for item in record.events)
    metadata = {**dict(record.run_metadata), "reason": reason}
    return RunResult(
        run_id=record.run_id,
        status=RunStatus.EXECUTION_IN_FLIGHT,
        final_text=None,
        tool_calls=record.tool_calls,
        model_turns=record.model_turns,
        events=events,
        metadata=metadata,
    )


def observe_loaded_run(record: StoredRunRecord) -> RunResult:
    if record.durable_state in _TERMINAL_DURABLE:
        return record_to_run_result(record)
    if record.durable_state is DurableRunState.TOOL_IN_FLIGHT:
        return in_flight_run_result(record)
    raise RunStoreError(
        f"cannot observe non-terminal run in durable state {record.durable_state.value}"
    )


def from_mapping(payload: Mapping[str, Any]) -> StoredRunRecord:
    schema_version = payload.get("schema_version")
    if schema_version != STORE_SCHEMA_VERSION:
        raise RunStoreError(
            f"unsupported stored run schema version: {schema_version!r}"
        )
    version = _non_negative_int(payload.get("version"), "version", minimum=0)
    run_id = _required_text(payload.get("run_id"), "run_id")
    durable_state_raw = payload.get("durable_state")
    if not isinstance(durable_state_raw, str):
        raise RunStoreError("durable_state must be a string")
    try:
        state = DurableRunState(durable_state_raw)
    except ValueError as exc:
        raise RunStoreError(f"unknown durable state {durable_state_raw!r}") from exc
    profile_fp = _required_text(
        payload.get("profile_fingerprint"), "profile_fingerprint"
    )
    task = _required_text(payload.get("task"), "task")
    run_metadata = payload.get("run_metadata")
    if not isinstance(run_metadata, dict):
        raise RunStoreError("run_metadata must be a mapping")
    started_at_utc = _finite_number(payload.get("started_at_utc"), "started_at_utc")
    expires_at_utc = _finite_number(payload.get("expires_at_utc"), "expires_at_utc")
    next_step = _non_negative_int(payload.get("next_step"), "next_step", minimum=1)
    tool_calls = _non_negative_int(payload.get("tool_calls"), "tool_calls", minimum=0)
    model_turns = _non_negative_int(
        payload.get("model_turns"), "model_turns", minimum=0
    )
    messages_raw = payload.get("messages")
    events_raw = payload.get("events")
    if not isinstance(messages_raw, list) or not isinstance(events_raw, list):
        raise RunStoreError("messages and events must be lists")
    messages = _validate_message_list(messages_raw)
    events = _validate_event_list(events_raw)
    terminal_status_raw = payload.get("terminal_status")
    terminal_status = None
    if terminal_status_raw is not None:
        if not isinstance(terminal_status_raw, str):
            raise RunStoreError("terminal_status must be a string")
        try:
            terminal_status = RunStatus(terminal_status_raw)
        except ValueError as exc:
            raise RunStoreError("terminal_status is invalid") from exc
    terminal_metadata = payload.get("terminal_metadata")
    if terminal_metadata is not None and not isinstance(terminal_metadata, dict):
        raise RunStoreError("terminal_metadata must be a mapping")
    pending_action = payload.get("pending_action")
    if pending_action is not None and not isinstance(pending_action, dict):
        raise RunStoreError("pending_action must be a mapping")
    final_text_raw = payload.get("final_text")
    final_text = None
    if final_text_raw is not None:
        final_text = _optional_str(final_text_raw)
    record = StoredRunRecord(
        schema_version=schema_version,
        version=version,
        run_id=run_id,
        durable_state=state,
        profile_fingerprint=profile_fp,
        task=task,
        run_metadata=run_metadata,
        started_at_utc=started_at_utc,
        expires_at_utc=expires_at_utc,
        next_step=next_step,
        tool_calls=tool_calls,
        model_turns=model_turns,
        messages=messages,
        events=events,
        pending_request_id=_optional_str(payload.get("pending_request_id")),
        pending_idempotency_key=_optional_str(payload.get("pending_idempotency_key")),
        pending_action=pending_action,
        pending_tool_call_id=_optional_str(payload.get("pending_tool_call_id")),
        pending_tool_name=_optional_str(payload.get("pending_tool_name")),
        pending_execution_status=_optional_str(payload.get("pending_execution_status")),
        pending_execution_output=payload.get("pending_execution_output"),
        pending_execution_reason=_optional_str(payload.get("pending_execution_reason")),
        terminal_status=terminal_status,
        final_text=final_text,
        terminal_metadata=terminal_metadata,
    )
    _validate_state_fields(record)
    return record


def validate_record_columns(
    record: StoredRunRecord,
    *,
    schema_version: int,
    version: int,
    durable_state: str,
) -> None:
    if record.schema_version != schema_version:
        raise RunStoreError("schema_version column mismatch")
    if record.version != version:
        raise RunStoreError("version column mismatch")
    if record.durable_state.value != durable_state:
        raise RunStoreError("durable_state column mismatch")


def new_running_record(
    *,
    run_id: str,
    profile: AgentProfile,
    run_input: RunInput,
    started_at_utc: float,
    expires_at_utc: float,
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
        started_at_utc=started_at_utc,
        expires_at_utc=expires_at_utc,
        next_step=1,
        tool_calls=0,
        model_turns=0,
        messages=tuple(serialize_message(item) for item in messages),
        events=tuple(serialize_event(item) for item in events),
    )


def _validate_message_list(messages: list[Any]) -> tuple[Mapping[str, Any], ...]:
    validated: list[Mapping[str, Any]] = []
    for index, item in enumerate(messages):
        if not isinstance(item, dict):
            raise RunStoreError(f"message at index {index} is invalid")
        validated.append(dict(item))
    return tuple(validated)


def _validate_event_list(events: list[Any]) -> tuple[Mapping[str, Any], ...]:
    validated: list[Mapping[str, Any]] = []
    for index, item in enumerate(events):
        if not isinstance(item, dict):
            raise RunStoreError(f"event at index {index} is invalid")
        validated.append(dict(item))
    return tuple(validated)


def _validate_state_fields(record: StoredRunRecord) -> None:
    if record.durable_state in _TERMINAL_DURABLE:
        if record.terminal_status is None:
            raise RunStoreError("terminal run missing terminal_status")
        return
    if record.terminal_status is not None:
        raise RunStoreError("non-terminal run must not include terminal_status")
    if record.durable_state in _PENDING_TOOL_STATES:
        for field_name in (
            "pending_request_id",
            "pending_idempotency_key",
            "pending_action",
            "pending_tool_call_id",
            "pending_tool_name",
        ):
            if getattr(record, field_name) is None:
                raise RunStoreError(
                    f"{record.durable_state.value} requires {field_name}"
                )
    if (
        record.durable_state is DurableRunState.TOOL_COMPLETED
        and record.pending_execution_status is None
    ):
        raise RunStoreError("TOOL_COMPLETED requires pending_execution_status")


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunStoreError(f"{field_name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise RunStoreError(f"{field_name} must be finite")
    return number


def _non_negative_int(value: object, field_name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RunStoreError(f"{field_name} must be an integer")
    if value < minimum:
        raise RunStoreError(f"{field_name} must be >= {minimum}")
    return value


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunStoreError(f"{field_name} must be a non-empty string")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RunStoreError("expected non-empty string field")
    return value


__all__ = [
    "DurableRunState",
    "RunStore",
    "RunStoreError",
    "STORE_SCHEMA_VERSION",
    "StoredRunRecord",
    "compute_deadline_monotonic",
    "deserialize_event",
    "deserialize_message",
    "from_mapping",
    "in_flight_run_result",
    "new_running_record",
    "observe_loaded_run",
    "profile_fingerprint",
    "record_to_run_result",
    "serialize_event",
    "serialize_message",
    "validate_record_columns",
]
