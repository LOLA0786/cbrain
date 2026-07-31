from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from cbrain.adapters.hermes import (
    HermesAdapterError,
    HermesCapabilityMap,
    HermesPreToolDecisionHook,
    register_hermes_hook,
)
from cbrain.adapters.privatevault import (
    PrivateVaultDecision,
    PrivateVaultVerdict,
)
from cbrain.contracts import ActionIntent


class RecordingClient:
    def __init__(self, verdict: PrivateVaultVerdict) -> None:
        self.verdict = verdict
        self.actions: list[ActionIntent] = []

    def decide(self, action: ActionIntent) -> PrivateVaultDecision:
        self.actions.append(action)
        return PrivateVaultDecision(
            verdict=self.verdict,
            triggered_by="test-policy",
            reason="test reason",
            _record_json=b"{}",
        )


class RaisingClient:
    def decide(self, action: ActionIntent) -> PrivateVaultDecision:
        raise RuntimeError("PrivateVault unavailable")


class RecordingPluginContext:
    def __init__(self) -> None:
        self.hooks: list[tuple[str, Callable[..., Any]]] = []

    def register_hook(
        self,
        hook_name: str,
        callback: Callable[..., Any],
    ) -> None:
        self.hooks.append((hook_name, callback))


def make_hook(
    verdict: PrivateVaultVerdict,
) -> tuple[HermesPreToolDecisionHook, RecordingClient]:
    client = RecordingClient(verdict)
    hook = HermesPreToolDecisionHook(
        client=client,
        agent_id="agent-1",
        capabilities=HermesCapabilityMap({"terminal": "system.command.execute"}),
        clock=lambda: 1234.5,
    )
    return hook, client


@pytest.mark.parametrize(
    ("verdict", "marker"),
    [
        (PrivateVaultVerdict.BLOCK, "CBRAIN_BLOCKED"),
        (
            PrivateVaultVerdict.REQUIRE_APPROVAL,
            "CBRAIN_REVIEW_REQUIRED",
        ),
        (
            PrivateVaultVerdict.ALLOW,
            "CBRAIN_EXECUTION_GATE_REQUIRED",
        ),
    ],
)
def test_every_privatevault_verdict_blocks_before_execution(
    verdict: PrivateVaultVerdict,
    marker: str,
) -> None:
    hook, client = make_hook(verdict)

    directive = hook.pre_tool_call(
        tool_name="terminal",
        args={"command": "pwd"},
        session_id="session-1",
        tool_call_id="tool-1",
    )

    assert directive["action"] == "block"
    assert directive["message"].startswith(marker)
    assert len(client.actions) == 1
    assert client.actions[0].agent_id == "agent-1"
    assert client.actions[0].framework == "hermes"
    assert client.actions[0].tool_name == "terminal"
    assert client.actions[0].capability == "system.command.execute"


def test_privatevault_failure_becomes_static_fail_closed_directive() -> None:
    hook = HermesPreToolDecisionHook(
        client=RaisingClient(),
        agent_id="agent-1",
        capabilities=HermesCapabilityMap({"terminal": "system.command.execute"}),
    )

    directive = hook.pre_tool_call(
        tool_name="terminal",
        args={"command": "pwd"},
        session_id="session-1",
        tool_call_id="tool-1",
    )

    assert directive == {
        "action": "block",
        "message": "CBRAIN_CONTROL_FAILURE",
    }


def test_unknown_tool_fails_closed_without_calling_privatevault() -> None:
    hook, client = make_hook(PrivateVaultVerdict.ALLOW)

    directive = hook.pre_tool_call(
        tool_name="unregistered_tool",
        args={},
        session_id="session-1",
        tool_call_id="tool-1",
    )

    assert directive["message"] == "CBRAIN_CONTROL_FAILURE"
    assert client.actions == []


@pytest.mark.parametrize(
    ("session_id", "tool_call_id"),
    [
        ("", "tool-1"),
        ("session-1", ""),
    ],
)
def test_missing_hermes_identity_fails_closed(
    session_id: str,
    tool_call_id: str,
) -> None:
    hook, client = make_hook(PrivateVaultVerdict.ALLOW)

    directive = hook.pre_tool_call(
        tool_name="terminal",
        args={},
        session_id=session_id,
        tool_call_id=tool_call_id,
    )

    assert directive["message"] == "CBRAIN_CONTROL_FAILURE"
    assert client.actions == []


def test_request_id_is_deterministic_for_same_hermes_tool_call() -> None:
    hook, client = make_hook(PrivateVaultVerdict.BLOCK)

    first = hook.pre_tool_call(
        tool_name="terminal",
        args={"command": "pwd"},
        session_id="session-1",
        tool_call_id="tool-1",
    )
    second = hook.pre_tool_call(
        tool_name="terminal",
        args={"command": "pwd"},
        session_id="session-1",
        tool_call_id="tool-1",
    )

    assert first == second
    assert client.actions[0].request_id == client.actions[1].request_id
    assert client.actions[0].idempotency_key == client.actions[1].idempotency_key


def test_capability_map_is_copied_and_default_deny() -> None:
    source = {"terminal": "system.command.execute"}
    capabilities = HermesCapabilityMap(source)
    source.clear()

    assert capabilities.resolve("terminal") == "system.command.execute"
    with pytest.raises(HermesAdapterError):
        capabilities.resolve("memory")


def test_registers_real_hermes_pre_tool_call_hook_name() -> None:
    hook, _client = make_hook(PrivateVaultVerdict.BLOCK)
    context = RecordingPluginContext()

    register_hermes_hook(context, hook)

    assert len(context.hooks) == 1
    assert context.hooks[0][0] == "pre_tool_call"
    assert context.hooks[0][1] == hook.pre_tool_call
