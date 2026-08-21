"""Reusable foundation agent for configuration-driven company agents."""

from .contracts import RunEvent, RunEventKind, RunInput, RunResult, RunStatus
from .foundation import FoundationAgent, FoundationAgentError
from .limits import RunLimits, finite_positive_seconds
from .profile import AgentProfile
from .tools import GovernedTool, ToolRegistry, ToolRegistryError

__all__ = [
    "AgentProfile",
    "FoundationAgent",
    "FoundationAgentError",
    "GovernedTool",
    "RunEvent",
    "RunEventKind",
    "RunInput",
    "RunLimits",
    "RunResult",
    "RunStatus",
    "ToolRegistry",
    "ToolRegistryError",
    "finite_positive_seconds",
]
