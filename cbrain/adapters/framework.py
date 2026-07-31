from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol

from ..contracts import ActionIntent
from .privatevault import PrivateVaultDecision, PrivateVaultVerdict


class FrameworkAdapterError(ValueError):
    """Raised when trusted framework adapter configuration is invalid."""


class CapabilityResolver(Protocol):
    def resolve(self, tool_name: str) -> str:
        """Resolve a framework tool to its governed capability."""


class FrameworkDecisionClient(Protocol):
    def decide(self, action: ActionIntent) -> PrivateVaultDecision:
        """Return the authoritative PrivateVault decision."""


class StaticCapabilityMap:
    """Immutable, default-deny tool-to-capability mapping."""

    def __init__(self, capabilities: Mapping[str, str]) -> None:
        values: dict[str, str] = {}
        for tool_name, capability in capabilities.items():
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise FrameworkAdapterError("Tool names must be non-empty strings")
            if not isinstance(capability, str) or not capability.strip():
                raise FrameworkAdapterError(
                    f"Capability for {tool_name!r} must be non-empty"
                )
            values[tool_name] = capability

        if not values:
            raise FrameworkAdapterError(
                "At least one tool capability must be configured"
            )
        self._values = MappingProxyType(values)

    def resolve(self, tool_name: str) -> str:
        try:
            return self._values[tool_name]
        except KeyError as exc:
            raise FrameworkAdapterError(
                f"Tool {tool_name!r} has no configured capability"
            ) from exc


class FrameworkDisposition(StrEnum):
    BLOCKED = "BLOCKED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    EXECUTION_GATE_REQUIRED = "EXECUTION_GATE_REQUIRED"
    CONTROL_FAILURE = "CONTROL_FAILURE"


@dataclass(frozen=True, slots=True)
class FrameworkGuardResult:
    disposition: FrameworkDisposition
    message: str
    request_id: str | None
    action: ActionIntent | None

    @property
    def blocks_execution(self) -> bool:
        return True


class FrameworkDecisionGuard:
    """Normalize framework tool calls and fail closed on every current path.

    PrivateVault ALLOW remains blocked until the signed execution gateway is
    connected. Concrete framework adapters therefore never call their native
    tool handler in the current production-runtime alpha.
    """

    def __init__(
        self,
        *,
        client: FrameworkDecisionClient,
        agent_id: str,
        capabilities: CapabilityResolver,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise FrameworkAdapterError("agent_id must be a non-empty trusted value")

        self._client = client
        self._agent_id = agent_id
        self._capabilities = capabilities
        self._clock = clock

    def _request_id(
        self,
        *,
        framework: str,
        invocation_id: str,
    ) -> str:
        if not framework.strip() or not invocation_id.strip():
            raise FrameworkAdapterError("framework and invocation_id are required")

        material = "\x1f".join((framework, self._agent_id, invocation_id)).encode(
            "utf-8"
        )
        return f"{framework}-" + hashlib.sha256(material).hexdigest()

    def evaluate(
        self,
        *,
        framework: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        invocation_id: str,
        context: Mapping[str, Any] | None = None,
    ) -> FrameworkGuardResult:
        try:
            request_id = self._request_id(
                framework=framework,
                invocation_id=invocation_id,
            )
            action = ActionIntent.capture(
                agent_id=self._agent_id,
                framework=framework,
                tool_name=tool_name,
                capability=self._capabilities.resolve(tool_name),
                arguments=arguments,
                context=context,
                request_id=request_id,
                idempotency_key=request_id,
                timestamp=self._clock(),
            )
            decision = self._client.decide(action)

            if decision.verdict is PrivateVaultVerdict.BLOCK:
                disposition = FrameworkDisposition.BLOCKED
            elif decision.verdict is PrivateVaultVerdict.REQUIRE_APPROVAL:
                disposition = FrameworkDisposition.REVIEW_REQUIRED
            elif decision.verdict is PrivateVaultVerdict.ALLOW:
                disposition = FrameworkDisposition.EXECUTION_GATE_REQUIRED
            else:
                disposition = FrameworkDisposition.CONTROL_FAILURE

            return FrameworkGuardResult(
                disposition=disposition,
                message=(f"CBRAIN_{disposition.value} request_id={request_id}"),
                request_id=request_id,
                action=action,
            )
        except Exception:
            return FrameworkGuardResult(
                disposition=FrameworkDisposition.CONTROL_FAILURE,
                message="CBRAIN_CONTROL_FAILURE",
                request_id=None,
                action=None,
            )


__all__ = [
    "CapabilityResolver",
    "FrameworkAdapterError",
    "FrameworkDecisionClient",
    "FrameworkDecisionGuard",
    "FrameworkDisposition",
    "FrameworkGuardResult",
    "StaticCapabilityMap",
]
