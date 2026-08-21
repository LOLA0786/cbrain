"""Reusable foundation agent for configuration-driven company agents."""

from .contracts import RunEvent, RunEventKind, RunInput, RunResult, RunStatus
from .durable import (
    DurableRunState,
    RunStore,
    RunStoreError,
    StoredRunRecord,
    profile_fingerprint,
)
from .foundation import FoundationAgent, FoundationAgentError
from .limits import RunLimits, finite_positive_seconds
from .profile import AgentProfile
from .store_memory import InMemoryRunStore
from .store_sqlite import SQLiteRunStore
from .tools import GovernedTool, ToolRegistry, ToolRegistryError

__all__ = [
    "AgentProfile",
    "DurableRunState",
    "FoundationAgent",
    "FoundationAgentError",
    "GovernedTool",
    "InMemoryRunStore",
    "RunEvent",
    "RunEventKind",
    "RunInput",
    "RunLimits",
    "RunResult",
    "RunStatus",
    "RunStore",
    "RunStoreError",
    "SQLiteRunStore",
    "StoredRunRecord",
    "ToolRegistry",
    "ToolRegistryError",
    "finite_positive_seconds",
    "profile_fingerprint",
]
