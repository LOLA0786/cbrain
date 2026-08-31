"""Reusable agent profile for configuration-driven specialization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .limits import RunLimits, finite_positive_seconds


@dataclass(frozen=True, slots=True)
class AgentProfile:
    """Domain-neutral agent configuration."""

    agent_id: str
    instructions: str
    model_route: str
    permitted_tools: frozenset[str]
    max_model_turns: int
    max_tool_calls: int
    timeout_seconds: float
    limits: RunLimits = RunLimits()
    metadata: Mapping[str, Any] | None = None
    knowledge_required_for_tools: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, str) or not self.agent_id.strip():
            raise ValueError("agent_id must be non-empty")
        if not isinstance(self.instructions, str) or not self.instructions.strip():
            raise ValueError("instructions must be non-empty")
        if not isinstance(self.model_route, str) or not self.model_route.strip():
            raise ValueError("model_route must be non-empty")
        if not isinstance(self.permitted_tools, frozenset):
            raise ValueError("permitted_tools must be a frozenset")
        if any(
            not isinstance(name, str) or not name.strip()
            for name in self.permitted_tools
        ):
            raise ValueError("permitted tool names must be non-empty strings")
        if (
            isinstance(self.max_model_turns, bool)
            or not isinstance(self.max_model_turns, int)
            or self.max_model_turns < 1
        ):
            raise ValueError("max_model_turns must be positive")
        if (
            isinstance(self.max_tool_calls, bool)
            or not isinstance(self.max_tool_calls, int)
            or self.max_tool_calls < 0
        ):
            raise ValueError("max_tool_calls must be non-negative")
        object.__setattr__(
            self,
            "timeout_seconds",
            finite_positive_seconds(self.timeout_seconds, "timeout_seconds"),
        )
        if not isinstance(self.limits, RunLimits):
            raise ValueError("limits must be a RunLimits")
        if not isinstance(self.knowledge_required_for_tools, bool):
            raise ValueError("knowledge_required_for_tools must be a bool")
        if self.metadata is not None:
            if not isinstance(self.metadata, Mapping):
                raise ValueError("metadata must be a mapping")
            object.__setattr__(
                self,
                "metadata",
                MappingProxyType(dict(self.metadata)),
            )


__all__ = ["AgentProfile"]
