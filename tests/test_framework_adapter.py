from __future__ import annotations

import pytest

from cbrain.adapters.framework import (
    FrameworkAdapterError,
    FrameworkDecisionGuard,
    FrameworkDisposition,
    StaticCapabilityMap,
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
            triggered_by="test",
            reason="test",
            _record_json=b"{}",
        )


class RaisingClient:
    def decide(self, action: ActionIntent) -> PrivateVaultDecision:
        raise RuntimeError("control plane unavailable")


def guard(
    verdict: PrivateVaultVerdict,
) -> tuple[FrameworkDecisionGuard, RecordingClient]:
    client = RecordingClient(verdict)
    return (
        FrameworkDecisionGuard(
            client=client,
            agent_id="agent-1",
            capabilities=StaticCapabilityMap({"search": "knowledge.search"}),
            clock=lambda: 1234.5,
        ),
        client,
    )


@pytest.mark.parametrize(
    ("verdict", "disposition"),
    [
        (
            PrivateVaultVerdict.BLOCK,
            FrameworkDisposition.BLOCKED,
        ),
        (
            PrivateVaultVerdict.REQUIRE_APPROVAL,
            FrameworkDisposition.REVIEW_REQUIRED,
        ),
        (
            PrivateVaultVerdict.ALLOW,
            FrameworkDisposition.EXECUTION_GATE_REQUIRED,
        ),
    ],
)
def test_every_verdict_remains_fail_closed(
    verdict: PrivateVaultVerdict,
    disposition: FrameworkDisposition,
) -> None:
    runtime, client = guard(verdict)

    result = runtime.evaluate(
        framework="langchain",
        tool_name="search",
        arguments={"query": "private vault"},
        invocation_id="call-1",
    )

    assert result.disposition is disposition
    assert result.blocks_execution is True
    assert result.request_id is not None
    assert result.action is client.actions[0]


def test_control_failure_is_static_and_does_not_leak_exception() -> None:
    runtime = FrameworkDecisionGuard(
        client=RaisingClient(),
        agent_id="agent-1",
        capabilities=StaticCapabilityMap({"search": "knowledge.search"}),
    )

    result = runtime.evaluate(
        framework="langchain",
        tool_name="search",
        arguments={},
        invocation_id="call-1",
    )

    assert result.disposition is FrameworkDisposition.CONTROL_FAILURE
    assert result.message == "CBRAIN_CONTROL_FAILURE"
    assert result.action is None


def test_unknown_tool_fails_closed_without_calling_client() -> None:
    runtime, client = guard(PrivateVaultVerdict.ALLOW)

    result = runtime.evaluate(
        framework="crewai",
        tool_name="delete_everything",
        arguments={},
        invocation_id="call-1",
    )

    assert result.disposition is FrameworkDisposition.CONTROL_FAILURE
    assert client.actions == []


def test_request_identity_is_deterministic() -> None:
    runtime, _client = guard(PrivateVaultVerdict.BLOCK)

    first = runtime.evaluate(
        framework="autogen",
        tool_name="search",
        arguments={"query": "one"},
        invocation_id="call-1",
    )
    second = runtime.evaluate(
        framework="autogen",
        tool_name="search",
        arguments={"query": "two"},
        invocation_id="call-1",
    )

    assert first.request_id == second.request_id


def test_different_frameworks_get_different_request_ids() -> None:
    runtime, _client = guard(PrivateVaultVerdict.BLOCK)

    langchain = runtime.evaluate(
        framework="langchain",
        tool_name="search",
        arguments={},
        invocation_id="call-1",
    )
    autogen = runtime.evaluate(
        framework="autogen",
        tool_name="search",
        arguments={},
        invocation_id="call-1",
    )

    assert langchain.request_id != autogen.request_id


def test_static_capability_map_is_copied_and_default_deny() -> None:
    source = {"search": "knowledge.search"}
    capabilities = StaticCapabilityMap(source)
    source.clear()

    assert capabilities.resolve("search") == "knowledge.search"
    with pytest.raises(FrameworkAdapterError):
        capabilities.resolve("delete")


@pytest.mark.parametrize(
    "invocation_id",
    ["", "   "],
)
def test_missing_invocation_identity_fails_closed(
    invocation_id: str,
) -> None:
    runtime, client = guard(PrivateVaultVerdict.ALLOW)

    result = runtime.evaluate(
        framework="langchain",
        tool_name="search",
        arguments={},
        invocation_id=invocation_id,
    )

    assert result.disposition is FrameworkDisposition.CONTROL_FAILURE
    assert client.actions == []
