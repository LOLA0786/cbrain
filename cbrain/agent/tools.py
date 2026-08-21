"""Tool registration and allowlisting for foundation agents."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from cbrain.models import ToolDefinition


class ToolRegistryError(ValueError):
    """Tool registration or resolution failed."""


@dataclass(frozen=True, slots=True)
class GovernedTool:
    name: str
    capability: str
    description: str
    input_schema: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ToolRegistryError("tool name must be non-empty")
        if not isinstance(self.capability, str) or not self.capability.strip():
            raise ToolRegistryError("tool capability must be non-empty")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ToolRegistryError("tool description must be non-empty")
        if not isinstance(self.input_schema, Mapping):
            raise ToolRegistryError("tool input_schema must be a mapping")
        object.__setattr__(
            self,
            "input_schema",
            MappingProxyType(dict(self.input_schema)),
        )

    def to_definition(self) -> ToolDefinition:
        return ToolDefinition.capture(
            name=self.name,
            description=self.description,
            input_schema=dict(self.input_schema),
        )


class ToolRegistry:
    """Immutable registry of tools available to a foundation agent."""

    def __init__(self, tools: Iterable[GovernedTool]) -> None:
        values: dict[str, GovernedTool] = {}
        for tool in tools:
            if tool.name in values:
                raise ToolRegistryError(f"duplicate tool name {tool.name!r}")
            values[tool.name] = tool
        self._tools = MappingProxyType(values)

    def get(self, name: str) -> GovernedTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolRegistryError(f"unknown tool {name!r}") from exc

    def contains(self, name: str) -> bool:
        return name in self._tools

    def definitions_for(self, permitted: frozenset[str]) -> tuple[ToolDefinition, ...]:
        missing = sorted(permitted - set(self._tools))
        if missing:
            raise ToolRegistryError(f"profile permits unregistered tools: {missing}")
        return tuple(self._tools[name].to_definition() for name in sorted(permitted))


__all__ = ["GovernedTool", "ToolRegistry", "ToolRegistryError"]
