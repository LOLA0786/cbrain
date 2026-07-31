from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, Literal, Protocol, TypedDict

from ..contracts import ActionIntent
from .privatevault import PrivateVaultDecision, PrivateVaultVerdict


class HermesAdapterError(ValueError):
    """Raised when the trusted Hermes adapter configuration is invalid."""


class HermesBlockDirective(TypedDict):
    action: Literal["block"]
    message: str


class DecisionClient(Protocol):
    def decide(self, action: ActionIntent) -> PrivateVaultDecision:
        """Return the authoritative PrivateVault decision."""


class HermesPluginContext(Protocol):
    def register_hook(self, hook_name: str, callback: Callable[..., Any]) -> None:
        """Register a Hermes lifecycle hook."""


class HermesCapabilityMap:
    """Immutable, default-deny mapping from Hermes tools to capabilities."""

    def __init__(self, capabilities: Mapping[str, str]) -> None:
        values: dict[str, str] = {}

        for tool_name, capability in capabilities.items():
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise HermesAdapterError("Hermes tool names must be non-empty strings")
            if not isinstance(capability, str) or not capability.strip():
                raise HermesAdapterError(
                    f"Capability for Hermes tool {tool_name!r} must be non-empty"
                )
            values[tool_name] = capability

        if not values:
            raise HermesAdapterError(
                "At least one Hermes capability must be configured"
            )

        self._values = MappingProxyType(values)

    def resolve(self, tool_name: str) -> str:
        try:
            return self._values[tool_name]
        except KeyError as exc:
            raise HermesAdapterError(
                f"Hermes tool {tool_name!r} has no configured capability"
            ) from exc


class HermesPreToolDecisionHook:
    """Translate Hermes calls into PrivateVault decisions.

    This hook deliberately blocks PrivateVault ALLOW decisions until the separate
    execution-authorization, exact-byte dispatch, witness, and closure adapter is
    connected. That prevents a decision-only integration from being mistaken for
    production execution enforcement.
    """

    def __init__(
        self,
        *,
        client: DecisionClient,
        agent_id: str,
        capabilities: HermesCapabilityMap,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise HermesAdapterError("agent_id must be a non-empty trusted value")

        self._client = client
        self._agent_id = agent_id
        self._capabilities = capabilities
        self._clock = clock

    @staticmethod
    def _block(message: str) -> HermesBlockDirective:
        return {"action": "block", "message": message}

    def _request_id(self, *, session_id: str, tool_call_id: str) -> str:
        if not session_id or not tool_call_id:
            raise HermesAdapterError(
                "Hermes session_id and tool_call_id are required for governance"
            )

        material = "\x1f".join((self._agent_id, session_id, tool_call_id)).encode(
            "utf-8"
        )
        return "hermes-" + hashlib.sha256(material).hexdigest()

    def pre_tool_call(
        self,
        *,
        tool_name: str,
        args: dict[str, Any] | None = None,
        task_id: str = "",
        session_id: str = "",
        tool_call_id: str = "",
        turn_id: str = "",
        api_request_id: str = "",
        middleware_trace: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> HermesBlockDirective:
        """Return a valid block directive on every path, including failures."""

        try:
            capability = self._capabilities.resolve(tool_name)
            request_id = self._request_id(
                session_id=session_id,
                tool_call_id=tool_call_id,
            )
            action = ActionIntent.capture(
                agent_id=self._agent_id,
                framework="hermes",
                tool_name=tool_name,
                capability=capability,
                arguments=args or {},
                context={
                    "task_id": task_id,
                    "session_id": session_id,
                    "tool_call_id": tool_call_id,
                    "turn_id": turn_id,
                    "api_request_id": api_request_id,
                    "middleware_trace": middleware_trace or [],
                },
                request_id=request_id,
                idempotency_key=request_id,
                timestamp=self._clock(),
            )
            decision = self._client.decide(action)

            if decision.verdict is PrivateVaultVerdict.BLOCK:
                return self._block(f"CBRAIN_BLOCKED request_id={request_id}")

            if decision.verdict is PrivateVaultVerdict.REQUIRE_APPROVAL:
                return self._block(f"CBRAIN_REVIEW_REQUIRED request_id={request_id}")

            if decision.verdict is PrivateVaultVerdict.ALLOW:
                return self._block(
                    f"CBRAIN_EXECUTION_GATE_REQUIRED request_id={request_id}"
                )

            return self._block(f"CBRAIN_INVALID_VERDICT request_id={request_id}")
        except Exception:
            # Hermes swallows hook exceptions and otherwise proceeds. Never let a
            # control-plane failure escape this callback.
            return self._block("CBRAIN_CONTROL_FAILURE")


def register_hermes_hook(
    context: HermesPluginContext,
    hook: HermesPreToolDecisionHook,
) -> None:
    """Register CBrain through Hermes' real pre_tool_call plugin API."""

    context.register_hook("pre_tool_call", hook.pre_tool_call)


__all__ = [
    "DecisionClient",
    "HermesAdapterError",
    "HermesBlockDirective",
    "HermesCapabilityMap",
    "HermesPluginContext",
    "HermesPreToolDecisionHook",
    "register_hermes_hook",
]
