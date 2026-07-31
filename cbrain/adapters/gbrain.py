from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .hermes import HermesCapabilityMap

GBRAIN_VERSION = "0.42.67.0"
GBRAIN_UPSTREAM_COMMIT = "c6dc0adf26a2d20df1147d2ec87c8922ca86d410"

GBRAIN_READ_TOOL_CAPABILITIES = MappingProxyType(
    {
        "get_page": "memory.page.read",
        "list_pages": "memory.page.list",
        "search": "memory.search.keyword",
        "query": "memory.search.hybrid",
        "get_tags": "memory.tags.read",
        "get_links": "memory.graph.read",
        "get_backlinks": "memory.graph.read",
        "traverse_graph": "memory.graph.traverse",
        "get_timeline": "memory.timeline.read",
        "get_stats": "memory.health.read",
        "get_health": "memory.health.read",
        "get_brain_identity": "memory.identity.read",
        "list_skills": "skills.list",
        "get_skill": "skills.read",
        "resolve_slugs": "memory.page.resolve",
    }
)


class GBrainConfigurationError(ValueError):
    """Raised when trusted GBrain process configuration is invalid."""


@dataclass(frozen=True, slots=True)
class GBrainStdioConfig:
    """Hermes MCP configuration for the pinned real GBrain executable."""

    command: str
    home: str
    connect_timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if not self.command or not Path(self.command).is_absolute():
            raise GBrainConfigurationError("GBrain command must be an absolute path")
        if not self.home or not Path(self.home).is_absolute():
            raise GBrainConfigurationError("GBRAIN_HOME must be an absolute path")
        if self.connect_timeout_seconds <= 0:
            raise GBrainConfigurationError(
                "GBrain connection timeout must be greater than zero"
            )

    @property
    def selected_tools(self) -> tuple[str, ...]:
        return tuple(GBRAIN_READ_TOOL_CAPABILITIES)

    def capability_map(self) -> HermesCapabilityMap:
        return HermesCapabilityMap(GBRAIN_READ_TOOL_CAPABILITIES)

    def hermes_server_config(self) -> dict[str, Any]:
        """Return Hermes' real mcp_servers.gbrain configuration shape."""

        return {
            "command": self.command,
            "args": ["serve"],
            "env": {
                "GBRAIN_HOME": self.home,
                "DATABASE_URL": "",
                "GBRAIN_DATABASE_URL": "",
            },
            "connect_timeout": self.connect_timeout_seconds,
            "enabled": True,
            "tools": {
                "include": list(self.selected_tools),
            },
        }


__all__ = [
    "GBRAIN_READ_TOOL_CAPABILITIES",
    "GBRAIN_UPSTREAM_COMMIT",
    "GBRAIN_VERSION",
    "GBrainConfigurationError",
    "GBrainStdioConfig",
]
