"""Run lifecycle types for the foundation agent."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any


class RunStatus(StrEnum):
    COMPLETED = "completed"
    REJECTED = "rejected"
    INVALID_MODEL_RESPONSE = "invalid_model_response"
    MODEL_FAILURE = "model_failure"
    TOOL_FAILURE = "tool_failure"
    LIMIT_REACHED = "limit_reached"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    RECOVERY_REQUIRED = "recovery_required"


class RunEventKind(StrEnum):
    RUN_STARTED = "run_started"
    MODEL_TURN = "model_turn"
    MODEL_RESPONSE = "model_response"
    TOOL_REQUESTED = "tool_requested"
    TOOL_COMPLETED = "tool_completed"
    TOOL_REJECTED = "tool_rejected"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"


@dataclass(frozen=True, slots=True)
class RunInput:
    task: str
    run_id: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("task must be non-empty")
        if self.run_id is not None and (
            not isinstance(self.run_id, str) or not self.run_id.strip()
        ):
            raise ValueError("run_id must be non-empty when provided")
        if self.metadata is not None:
            if not isinstance(self.metadata, Mapping):
                raise ValueError("metadata must be a mapping")
            object.__setattr__(
                self,
                "metadata",
                MappingProxyType(dict(self.metadata)),
            )


@dataclass(frozen=True, slots=True)
class RunEvent:
    kind: RunEventKind
    step: int
    timestamp: float
    detail: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RunResult:
    run_id: str
    status: RunStatus
    final_text: str | None
    tool_calls: int
    model_turns: int
    events: tuple[RunEvent, ...]
    metadata: Mapping[str, Any]

    @property
    def request_ids(self) -> tuple[str, ...]:
        collected: list[str] = []
        for event in self.events:
            request_id = event.detail.get("request_id")
            if isinstance(request_id, str) and request_id:
                collected.append(request_id)
        return tuple(collected)


__all__ = [
    "RunEvent",
    "RunEventKind",
    "RunInput",
    "RunResult",
    "RunStatus",
]
