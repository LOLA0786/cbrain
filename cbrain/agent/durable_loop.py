"""Durable checkpoint helpers for the foundation agent loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from cbrain.contracts import ActionIntent, ContractError, GovernedExecution
from cbrain.models import Message, MessageRole, ModelError, ToolCall, validate_history

from .contracts import RunEvent, RunInput, RunResult, RunStatus
from .durable import (
    DurableRunState,
    RunStore,
    RunStoreError,
    StoredRunRecord,
    compute_deadline_monotonic,
    deserialize_event,
    deserialize_message,
    in_flight_run_result,
    new_running_record,
    profile_fingerprint,
    record_to_run_result,
    serialize_event,
    serialize_message,
)
from .profile import AgentProfile


@dataclass
class DurableRunContext:
    run_input: RunInput
    run_id: str
    started_at_utc: float
    deadline_monotonic: float
    events: list[RunEvent]
    messages: list[Message]
    tool_calls: int
    model_turns: int
    next_step: int
    record: StoredRunRecord
    merge_completed_tool: bool = False


def initialize_durable_run(
    *,
    profile: AgentProfile,
    run_input: RunInput,
    run_store: RunStore,
    started_at_utc: float,
    expires_at_utc: float,
    wall_clock: Callable[[], float],
    monotonic_clock: Callable[[], float],
    run_id: str,
    instructions: str,
    started_event: RunEvent,
) -> DurableRunContext | RunResult:
    if run_input.run_id:
        try:
            record = run_store.load(run_input.run_id)
        except RunStoreError:
            record = None
    else:
        record = None

    if record is not None:
        return resume_durable_run(
            profile=profile,
            run_input=run_input,
            record=record,
            wall_clock=wall_clock,
            monotonic_clock=monotonic_clock,
        )

    messages = [
        Message(role=MessageRole.SYSTEM, content=instructions),
        Message(role=MessageRole.USER, content=run_input.task),
    ]
    events = [started_event]
    record = new_running_record(
        run_id=run_id,
        profile=profile,
        run_input=run_input,
        started_at_utc=started_at_utc,
        expires_at_utc=expires_at_utc,
        messages=tuple(messages),
        events=tuple(events),
    )
    run_store.create(record)
    deadline_monotonic = compute_deadline_monotonic(
        expires_at_utc=expires_at_utc,
        wall_clock=wall_clock,
        monotonic_clock=monotonic_clock,
    )
    return DurableRunContext(
        run_input=run_input,
        run_id=run_id,
        started_at_utc=started_at_utc,
        deadline_monotonic=deadline_monotonic,
        events=events,
        messages=messages,
        tool_calls=0,
        model_turns=0,
        next_step=1,
        record=record,
    )


def resume_durable_run(
    *,
    profile: AgentProfile,
    run_input: RunInput,
    record: StoredRunRecord,
    wall_clock: Callable[[], float],
    monotonic_clock: Callable[[], float],
) -> DurableRunContext | RunResult:
    if profile_fingerprint(profile) != record.profile_fingerprint:
        raise RunStoreError("agent profile fingerprint mismatch")
    if run_input.task.strip() != record.task.strip():
        raise RunStoreError("run task mismatch")

    if record.durable_state in {
        DurableRunState.COMPLETED,
        DurableRunState.FAILED,
        DurableRunState.CANCELLED,
        DurableRunState.RECOVERY_REQUIRED,
    }:
        return record_to_run_result(record)

    if record.durable_state is DurableRunState.TOOL_IN_FLIGHT:
        return in_flight_run_result(record)

    merge_completed_tool = record.durable_state is DurableRunState.TOOL_COMPLETED
    deadline_monotonic = compute_deadline_monotonic(
        expires_at_utc=record.expires_at_utc,
        wall_clock=wall_clock,
        monotonic_clock=monotonic_clock,
    )
    return DurableRunContext(
        run_input=run_input,
        run_id=record.run_id,
        started_at_utc=record.started_at_utc,
        deadline_monotonic=deadline_monotonic,
        events=[deserialize_event(item) for item in record.events],
        messages=[deserialize_message(item) for item in record.messages],
        tool_calls=record.tool_calls,
        model_turns=record.model_turns,
        next_step=record.next_step,
        record=record,
        merge_completed_tool=merge_completed_tool,
    )


def resolve_claim_conflict(run_store: RunStore, run_id: str) -> RunResult:
    current = run_store.load(run_id)
    if current.durable_state in {
        DurableRunState.COMPLETED,
        DurableRunState.FAILED,
        DurableRunState.CANCELLED,
        DurableRunState.RECOVERY_REQUIRED,
    }:
        return record_to_run_result(current)
    if current.durable_state is DurableRunState.TOOL_IN_FLIGHT:
        return in_flight_run_result(
            current,
            reason="another worker claimed tool dispatch",
        )
    return in_flight_run_result(
        current,
        reason="tool dispatch claim conflict",
    )


def persist_record(
    run_store: RunStore,
    record: StoredRunRecord,
    *,
    durable_state: DurableRunState | None = None,
    events: list[RunEvent] | None = None,
    messages: list[Message] | None = None,
    tool_calls: int | None = None,
    model_turns: int | None = None,
    next_step: int | None = None,
    pending_request_id: str | None = None,
    pending_idempotency_key: str | None = None,
    pending_action: dict[str, Any] | None = None,
    pending_tool_call_id: str | None = None,
    pending_tool_name: str | None = None,
    pending_execution_status: str | None = None,
    pending_execution_output: Any | None = None,
    pending_execution_reason: str | None = None,
    clear_pending: bool = False,
    terminal_status: RunStatus | None = None,
    final_text: str | None = None,
    terminal_metadata: dict[str, Any] | None = None,
) -> StoredRunRecord:
    updated = StoredRunRecord(
        schema_version=record.schema_version,
        version=record.version,
        run_id=record.run_id,
        durable_state=durable_state or record.durable_state,
        profile_fingerprint=record.profile_fingerprint,
        task=record.task,
        run_metadata=record.run_metadata,
        started_at_utc=record.started_at_utc,
        expires_at_utc=record.expires_at_utc,
        next_step=next_step if next_step is not None else record.next_step,
        tool_calls=tool_calls if tool_calls is not None else record.tool_calls,
        model_turns=model_turns if model_turns is not None else record.model_turns,
        messages=(
            tuple(serialize_message(item) for item in messages)
            if messages is not None
            else record.messages
        ),
        events=(
            tuple(serialize_event(item) for item in events)
            if events is not None
            else record.events
        ),
        pending_request_id=(
            None
            if clear_pending
            else (
                pending_request_id
                if pending_request_id is not None
                else record.pending_request_id
            )
        ),
        pending_idempotency_key=(
            None
            if clear_pending
            else (
                pending_idempotency_key
                if pending_idempotency_key is not None
                else record.pending_idempotency_key
            )
        ),
        pending_action=(
            None
            if clear_pending
            else (
                pending_action if pending_action is not None else record.pending_action
            )
        ),
        pending_tool_call_id=(
            None
            if clear_pending
            else (
                pending_tool_call_id
                if pending_tool_call_id is not None
                else record.pending_tool_call_id
            )
        ),
        pending_tool_name=(
            None
            if clear_pending
            else (
                pending_tool_name
                if pending_tool_name is not None
                else record.pending_tool_name
            )
        ),
        pending_execution_status=(
            None
            if clear_pending
            else (
                pending_execution_status
                if pending_execution_status is not None
                else record.pending_execution_status
            )
        ),
        pending_execution_output=(
            None
            if clear_pending
            else (
                pending_execution_output
                if pending_execution_output is not None
                else record.pending_execution_output
            )
        ),
        pending_execution_reason=(
            None
            if clear_pending
            else (
                pending_execution_reason
                if pending_execution_reason is not None
                else record.pending_execution_reason
            )
        ),
        terminal_status=(
            terminal_status if terminal_status is not None else record.terminal_status
        ),
        final_text=final_text if final_text is not None else record.final_text,
        terminal_metadata=(
            terminal_metadata
            if terminal_metadata is not None
            else record.terminal_metadata
        ),
    )
    return run_store.save(updated, expected_version=record.version)


def finalize_durable_record(
    record: StoredRunRecord,
    *,
    durable_state: DurableRunState,
    terminal_status: RunStatus,
    final_text: str | None = None,
    terminal_metadata: dict[str, Any] | None = None,
    events: list[RunEvent] | None = None,
    messages: list[Message] | None = None,
    tool_calls: int | None = None,
    model_turns: int | None = None,
) -> StoredRunRecord:
    return StoredRunRecord(
        schema_version=record.schema_version,
        version=record.version,
        run_id=record.run_id,
        durable_state=durable_state,
        profile_fingerprint=record.profile_fingerprint,
        task=record.task,
        run_metadata=record.run_metadata,
        started_at_utc=record.started_at_utc,
        expires_at_utc=record.expires_at_utc,
        next_step=record.next_step,
        tool_calls=tool_calls if tool_calls is not None else record.tool_calls,
        model_turns=model_turns if model_turns is not None else record.model_turns,
        messages=(
            tuple(serialize_message(item) for item in messages)
            if messages is not None
            else record.messages
        ),
        events=(
            tuple(serialize_event(item) for item in events)
            if events is not None
            else record.events
        ),
        terminal_status=terminal_status,
        final_text=final_text,
        terminal_metadata=terminal_metadata or dict(record.run_metadata),
    )


def restore_action(record: StoredRunRecord) -> ActionIntent:
    if record.pending_action is None:
        raise RunStoreError("missing pending action")
    try:
        return ActionIntent.capture(**dict(record.pending_action))
    except ContractError as exc:
        raise RunStoreError("pending action is invalid") from exc


def restore_tool_call(record: StoredRunRecord) -> ToolCall:
    if (
        record.pending_tool_call_id is None
        or record.pending_tool_name is None
        or record.pending_action is None
    ):
        raise RunStoreError("missing pending tool call")
    arguments = record.pending_action.get("arguments")
    if not isinstance(arguments, dict):
        raise RunStoreError("pending tool arguments are invalid")
    messages = [deserialize_message(item) for item in record.messages]
    try:
        validate_history(messages, allow_pending=True)
        pending = ToolCall.capture(
            call_id=record.pending_tool_call_id,
            name=record.pending_tool_name,
            arguments=arguments,
        )
    except ModelError as exc:
        raise RunStoreError("pending assistant history is invalid") from exc
    call = messages[-1].tool_call if messages else None
    if call is None:
        raise RunStoreError(
            "pending tool has no saved assistant turn; migration required"
        )
    if (call.call_id, call.name, call._arguments_json) != (
        pending.call_id,
        pending.name,
        pending._arguments_json,
    ):
        raise RunStoreError("pending action and assistant call do not match")
    return call


def restore_execution(record: StoredRunRecord) -> GovernedExecution:
    from cbrain.contracts import ExecutionStatus

    if record.pending_execution_status is None or record.pending_request_id is None:
        raise RunStoreError("missing pending execution result")
    try:
        status = ExecutionStatus(record.pending_execution_status)
    except ValueError as exc:
        raise RunStoreError("pending execution status is invalid") from exc
    tool_executed: bool | None
    if status is ExecutionStatus.EXECUTED:
        tool_executed = True
    elif status is ExecutionStatus.INDETERMINATE:
        tool_executed = None
    else:
        tool_executed = False
    return GovernedExecution(
        status=status,
        request_id=record.pending_request_id,
        tool_executed=tool_executed,
        reason=record.pending_execution_reason or "",
        output=record.pending_execution_output,
    )


__all__ = [
    "DurableRunContext",
    "finalize_durable_record",
    "initialize_durable_run",
    "persist_record",
    "resolve_claim_conflict",
    "restore_action",
    "restore_execution",
    "restore_tool_call",
    "resume_durable_run",
]
