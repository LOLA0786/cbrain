"""Wire-contract and restart regressions; all transports/gateways here are doubles."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from cbrain import ActionIntent, ExecutionStatus, GovernedExecution, GovernedRuntime
from cbrain.agent import (
    AgentProfile,
    FoundationAgent,
    FoundationAgentError,
    GovernedTool,
    RunInput,
    RunStatus,
    ToolRegistry,
)
from cbrain.agent.durable import DurableRunState, StoredRunRecord
from cbrain.agent.store_sqlite import SQLiteRunStore
from cbrain.models import (
    AnthropicAdapter,
    CompletionRequest,
    GoogleAdapter,
    Message,
    MessageRole,
    ModelContractError,
    ModelRouter,
    OpenAICompatibleAdapter,
    ProviderContinuation,
    ToolCall,
)

PROVIDERS = ["openai", "xai", "runpod", "anthropic", "google"]
MODEL = "configured-test-model"
TASK = RunInput(task="Read account-7", run_id="history-test-run")


class ScriptedTransport:
    def __init__(self, *responses: dict[str, Any]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def post_json(self, path: str, payload: Mapping[str, Any], headers: Any) -> Any:
        self.requests.append(copy.deepcopy(dict(payload)))
        assert self.responses, "unexpected provider request"
        return self.responses.pop(0)


class RecordingGateway:
    independent_execution = False

    def __init__(self) -> None:
        self.actions: list[ActionIntent] = []

    def decide_and_execute(
        self, action: ActionIntent, handler: Any
    ) -> GovernedExecution:
        self.actions.append(action)
        return GovernedExecution(
            status=ExecutionStatus.EXECUTED,
            request_id=action.request_id,
            tool_executed=True,
            reason="test_only",
            output=handler(action.arguments),
        )


class SimulatedCrash(RuntimeError):
    pass


class CrashStore(SQLiteRunStore):
    def __init__(self, path: Path, phase: str) -> None:
        super().__init__(path)
        self.phase = phase

    def save(
        self, record: StoredRunRecord, *, expected_version: int
    ) -> StoredRunRecord:
        saved = super().save(record, expected_version=expected_version)
        states = {
            "prepared": DurableRunState.TOOL_PREPARED,
            "completed": DurableRunState.TOOL_COMPLETED,
            "running": DurableRunState.RUNNING,
        }
        if saved.durable_state is states[self.phase] and (
            self.phase != "running" or saved.tool_calls == 1
        ):
            raise SimulatedCrash(self.phase)
        return saved


def responses(
    provider: str, *, rich: bool = False
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if provider == "anthropic":
        blocks: list[dict[str, Any]] = []
        if rich:
            blocks = [
                {"type": "thinking", "thinking": "", "signature": "opaque-signature"},
                {"type": "redacted_thinking", "data": "opaque-redacted-block"},
                {"type": "text", "text": "Checking the record."},
            ]
        blocks.append(
            {
                "type": "tool_use",
                "id": "call-1",
                "name": "lookup",
                "input": {"id": "account-7"},
            }
        )
        assistant = {"role": "assistant", "content": blocks}
        return (
            {"content": blocks, "stop_reason": "tool_use"},
            {
                "content": [{"type": "text", "text": "active"}],
                "stop_reason": "end_turn",
            },
            assistant,
        )
    if provider == "google":
        parts: list[dict[str, Any]] = []
        if rich:
            parts = [
                {"text": "Checking the record.", "thoughtSignature": "text-signature"}
            ]
        parts.append(
            {
                "functionCall": {
                    "id": "call-1",
                    "name": "lookup",
                    "args": {"id": "account-7"},
                },
                "thoughtSignature": "call-signature",
            }
        )
        assistant = {"role": "model", "parts": parts}
        return (
            {"candidates": [{"content": assistant, "finishReason": "STOP"}]},
            {
                "candidates": [
                    {"content": {"parts": [{"text": "active"}]}, "finishReason": "STOP"}
                ]
            },
            assistant,
        )
    assistant = {
        "role": "assistant",
        "content": "Checking the record." if rich else None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{ "id": "account-7" }'},
            }
        ],
    }
    if rich:
        assistant["reasoning_content"] = "provider-continuation"
    return (
        {"choices": [{"message": assistant, "finish_reason": "tool_calls"}]},
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "active"},
                    "finish_reason": "stop",
                }
            ]
        },
        assistant,
    )


def adapter(provider: str, transport: ScriptedTransport, *, model: str = MODEL) -> Any:
    kwargs = {"model": model, "transport": transport, "headers_provider": lambda: {}}
    if provider == "anthropic":
        return AnthropicAdapter(**kwargs)
    if provider == "google":
        return GoogleAdapter(**kwargs)
    return OpenAICompatibleAdapter(provider=provider, **kwargs)


def build_agent(
    model: Any,
    gateway: RecordingGateway,
    store: SQLiteRunStore | None = None,
    knowledge: Any = None,
) -> FoundationAgent:
    return FoundationAgent(
        profile=AgentProfile(
            agent_id="history-agent",
            instructions="Read the requested record.",
            model_route="test",
            permitted_tools=frozenset({"lookup"}),
            max_model_turns=4,
            max_tool_calls=2,
            timeout_seconds=30,
        ),
        runtime=GovernedRuntime(gateway),
        model_router=ModelRouter({"test": model}),
        tools=ToolRegistry(
            [
                GovernedTool(
                    name="lookup",
                    capability="records.read",
                    description="Read a record",
                    input_schema={
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                    },
                )
            ]
        ),
        handlers={"lookup": lambda args: {"id": args["id"], "state": "active"}},
        run_store=store,
        clock=lambda: 10.0,
        wall_clock=lambda: 1_800_000_000.0,
        knowledge=knowledge,
    )


def assert_followup(
    provider: str, payload: dict[str, Any], assistant: dict[str, Any]
) -> None:
    messages = payload["contents" if provider == "google" else "messages"]
    index = next(i for i, message in enumerate(messages) if message == assistant)
    result = messages[index + 1]
    if provider == "google":
        result = result["parts"][0]["functionResponse"]
        assert result["id"] == "call-1"
        assert result["name"] == "lookup"
        observation = result["response"]["content"]
    elif provider == "anthropic":
        result = result["content"][0]
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "call-1"
        observation = result["content"]
    else:
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call-1"
        observation = result["content"]
    assert json.loads(observation)["output"] == {"id": "account-7", "state": "active"}
    assert sum(message == assistant for message in messages) == 1


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize(
    "phase", ["none", "durable", "prepared", "completed", "running"]
)
def test_provider_followup_survives_sqlite_restart(
    tmp_path: Path, provider: str, phase: str
) -> None:
    first, final, assistant = responses(provider)
    gateway = RecordingGateway()
    path = tmp_path / "runs.sqlite"
    if phase in {"none", "durable"}:
        transport = ScriptedTransport(first, final)
        store = SQLiteRunStore(path) if phase == "durable" else None
        try:
            result = build_agent(adapter(provider, transport), gateway, store).run(TASK)
        finally:
            if store:
                store.close()
        payload = transport.requests[1]
    else:
        crashing = CrashStore(path, phase)
        try:
            with pytest.raises(SimulatedCrash):
                build_agent(
                    adapter(provider, ScriptedTransport(first)), gateway, crashing
                ).run(TASK)
        finally:
            crashing.close()
        assert len(gateway.actions) == (0 if phase == "prepared" else 1)
        reopened = SQLiteRunStore(path)
        transport = ScriptedTransport(final)
        try:
            result = build_agent(adapter(provider, transport), gateway, reopened).run(
                TASK
            )
        finally:
            reopened.close()
        payload = transport.requests[0]
    assert result.status is RunStatus.COMPLETED
    assert result.final_text == "active"
    assert len(gateway.actions) == 1
    assert_followup(provider, payload, assistant)


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
def test_rich_assistant_turn_is_preserved_across_restart(
    tmp_path: Path, provider: str
) -> None:
    first, final, assistant = responses(provider, rich=True)
    path = tmp_path / "runs.sqlite"
    gateway = RecordingGateway()
    crashing = CrashStore(path, "completed")
    try:
        with pytest.raises(SimulatedCrash):
            build_agent(
                adapter(provider, ScriptedTransport(first)), gateway, crashing
            ).run(TASK)
    finally:
        crashing.close()
    transport = ScriptedTransport(final)
    reopened = SQLiteRunStore(path)
    try:
        result = build_agent(adapter(provider, transport), gateway, reopened).run(TASK)
    finally:
        reopened.close()
    assert result.status is RunStatus.COMPLETED
    assert len(gateway.actions) == 1
    assert_followup(provider, transport.requests[0], assistant)


def followup(call: ToolCall) -> CompletionRequest:
    return CompletionRequest(
        messages=(
            Message(MessageRole.USER, "Read a record"),
            call.as_message(),
            Message(
                MessageRole.TOOL,
                "active",
                tool_call_id=call.call_id,
                tool_name=call.name,
            ),
        )
    )


@pytest.mark.parametrize("provider", PROVIDERS)
def test_continuation_cannot_be_sent_to_different_model(provider: str) -> None:
    first, _, _ = responses(provider)
    call = adapter(provider, ScriptedTransport(first)).complete(
        CompletionRequest(messages=(Message(MessageRole.USER, "Read"),))
    )
    transport = ScriptedTransport()
    with pytest.raises(ModelContractError, match="route mismatch"):
        adapter(provider, transport, model="other-model").complete(followup(call))
    assert not transport.requests


def test_continuation_cannot_cross_provider_routes() -> None:
    first, _, _ = responses("openai")
    call = adapter("openai", ScriptedTransport(first)).complete(
        CompletionRequest(messages=(Message(MessageRole.USER, "Read"),))
    )
    transport = ScriptedTransport()
    with pytest.raises(ModelContractError, match="route mismatch"):
        adapter("xai", transport).complete(followup(call))
    assert not transport.requests


@pytest.mark.parametrize(
    "kind", ["orphan", "wrong_id", "wrong_name", "interposed", "missing", "duplicate"]
)
def test_invalid_call_result_history_is_rejected(kind: str) -> None:
    call = ToolCall.capture(call_id="call-1", name="lookup", arguments={})
    user = Message(MessageRole.USER, "Read")
    result = Message(
        MessageRole.TOOL, "active", tool_call_id="call-1", tool_name="lookup"
    )
    variants = {
        "orphan": (user, result),
        "wrong_id": (user, call.as_message(), replace(result, tool_call_id="other")),
        "wrong_name": (user, call.as_message(), replace(result, tool_name="other")),
        "interposed": (user, call.as_message(), user, result),
        "missing": (user, call.as_message()),
        "duplicate": (user, call.as_message(), result, call.as_message(), result),
    }
    with pytest.raises(ModelContractError):
        CompletionRequest(messages=variants[kind])


@pytest.mark.parametrize("change", ["missing", "arguments", "id"])
def test_pending_history_mismatch_never_dispatches(tmp_path: Path, change: str) -> None:
    first, final, _ = responses("openai")
    path = tmp_path / "runs.sqlite"
    gateway = RecordingGateway()
    crashing = CrashStore(path, "prepared")
    try:
        with pytest.raises(SimulatedCrash):
            build_agent(
                adapter("openai", ScriptedTransport(first)), gateway, crashing
            ).run(TASK)
    finally:
        crashing.close()
    store = SQLiteRunStore(path)
    try:
        record = store.load(TASK.run_id)
        if change == "missing":
            record = replace(record, messages=record.messages[:-1])
        elif change == "id":
            record = replace(record, pending_tool_call_id="other")
        else:
            record = replace(
                record,
                pending_action={**record.pending_action, "arguments": {"id": "other"}},
            )
        store.save(record, expected_version=record.version)
        transport = ScriptedTransport(final)
        with pytest.raises(FoundationAgentError, match="assistant"):
            build_agent(adapter("openai", transport), gateway, store).run(TASK)
        assert not gateway.actions
        assert not transport.requests
    finally:
        store.close()


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
def test_partial_tool_response_never_reaches_gateway(provider: str) -> None:
    first, _, _ = responses(provider)
    if provider == "google":
        first["candidates"][0]["finishReason"] = "MAX_TOKENS"
    elif provider == "anthropic":
        first["stop_reason"] = "max_tokens"
    else:
        first["choices"][0]["finish_reason"] = "length"
    gateway = RecordingGateway()
    result = build_agent(adapter(provider, ScriptedTransport(first)), gateway).run(TASK)
    assert result.status is RunStatus.MODEL_FAILURE
    assert not gateway.actions


@pytest.mark.parametrize("provider", ["anthropic", "google"])
def test_multiple_tool_calls_remain_non_executable(provider: str) -> None:
    first, _, _ = responses(provider)
    blocks = (
        first["content"]
        if provider == "anthropic"
        else first["candidates"][0]["content"]["parts"]
    )
    blocks.append(copy.deepcopy(blocks[-1]))
    gateway = RecordingGateway()
    result = build_agent(adapter(provider, ScriptedTransport(first)), gateway).run(TASK)
    assert result.status is RunStatus.MODEL_FAILURE
    assert not gateway.actions


def test_google_local_id_is_not_invented_on_provider_wire() -> None:
    first, final, assistant = responses("google")
    del assistant["parts"][0]["functionCall"]["id"]
    transport = ScriptedTransport(first, final)
    gateway = RecordingGateway()
    result = build_agent(adapter("google", transport), gateway).run(TASK)
    assert result.status is RunStatus.COMPLETED
    response = transport.requests[1]["contents"][-1]["parts"][0]["functionResponse"]
    assert "id" not in response
    assert len(gateway.actions) == 1


@pytest.mark.parametrize("durable", [False, True])
def test_duplicate_provider_call_id_never_dispatches_twice(
    tmp_path: Path, durable: bool
) -> None:
    first, final, _ = responses("openai")
    gateway = RecordingGateway()
    transport = ScriptedTransport(first, copy.deepcopy(first), final)
    store = SQLiteRunStore(tmp_path / "duplicate.sqlite") if durable else None
    try:
        result = build_agent(adapter("openai", transport), gateway, store).run(TASK)
    finally:
        if store:
            store.close()
    assert result.status is RunStatus.INVALID_MODEL_RESPONSE
    assert len(gateway.actions) == 1
    assert len(transport.requests) == 2


@pytest.mark.parametrize("phase", ["prepared", "completed"])
def test_refreshed_knowledge_does_not_split_call_and_result(
    tmp_path: Path, phase: str
) -> None:
    from types import SimpleNamespace

    class Knowledge:
        def retrieve(self, query: Any) -> Any:
            return SimpleNamespace(
                quoted_evidence="UNTRUSTED_SOURCE_CONTEXT: test evidence"
            )

    run = replace(
        TASK,
        metadata={
            "tenant_id": "tenant-1",
            "collection_id": "collection-1",
            "principal_id": "reader",
        },
    )
    first, final, assistant = responses("anthropic", rich=True)
    path = tmp_path / "knowledge.sqlite"
    gateway = RecordingGateway()
    crashing = CrashStore(path, phase)
    try:
        with pytest.raises(SimulatedCrash):
            build_agent(
                adapter("anthropic", ScriptedTransport(first)),
                gateway,
                crashing,
                Knowledge(),
            ).run(run)
    finally:
        crashing.close()
    reopened = SQLiteRunStore(path)
    transport = ScriptedTransport(final)
    try:
        result = build_agent(
            adapter("anthropic", transport), gateway, reopened, Knowledge()
        ).run(run)
    finally:
        reopened.close()
    assert result.status is RunStatus.COMPLETED
    assert len(gateway.actions) == 1
    assert_followup("anthropic", transport.requests[0], assistant)
    assert (
        "UNTRUSTED_SOURCE_CONTEXT" in transport.requests[0]["messages"][-1]["content"]
    )


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
def test_continuation_is_immutable_and_survives_serialization(provider: str) -> None:
    from cbrain.agent.durable import deserialize_message, serialize_message

    first, _, assistant = responses(provider, rich=True)
    call = adapter(provider, ScriptedTransport(first)).complete(
        CompletionRequest(messages=(Message(MessageRole.USER, "Read"),))
    )
    assert call.continuation is not None
    expected = copy.deepcopy(assistant)
    assistant.clear()
    call.continuation.message.clear()
    restored = deserialize_message(serialize_message(call.as_message()))
    assert restored.tool_call == call
    assert restored.tool_call.continuation.message == expected
    assert "opaque-signature" not in repr(restored)
    assert "call-signature" not in repr(restored)


@pytest.mark.parametrize(
    "change", ["role", "headers", "name", "arguments", "numeric_type"]
)
def test_continuation_cannot_change_the_neutral_call(change: str) -> None:
    _, _, assistant = responses("openai")
    arguments: dict[str, Any] = {"id": "account-7"}
    if change == "role":
        assistant["role"] = "system"
    elif change == "headers":
        assistant["headers"] = {"X-Test": "not-transport-configuration"}
    elif change == "name":
        assistant["tool_calls"][0]["function"]["name"] = "other"
    elif change == "arguments":
        assistant["tool_calls"][0]["function"]["arguments"] = '{"id":"other"}'
    else:
        arguments = {"id": True}
        assistant["tool_calls"][0]["function"]["arguments"] = '{"id":1}'
    with pytest.raises(ModelContractError):
        ToolCall.capture(
            call_id="call-1",
            name="lookup",
            arguments=arguments,
            continuation=ProviderContinuation.capture(
                provider="openai",
                model=MODEL,
                wire_format="chat_completions",
                message=assistant,
            ),
        )


def test_oversized_continuation_is_rejected_before_execution() -> None:
    first, _, assistant = responses("openai")
    assistant["content"] = "a" * 262144
    gateway = RecordingGateway()
    result = build_agent(adapter("openai", ScriptedTransport(first)), gateway).run(TASK)
    assert result.status is RunStatus.MODEL_FAILURE
    assert not gateway.actions


def test_history_is_included_in_estimated_input_usage() -> None:
    from cbrain.models import InstrumentedModelAdapter, TextOutput

    class FinalModel:
        provider = "test"
        model = MODEL

        def complete(self, request: CompletionRequest) -> TextOutput:
            return TextOutput("done")

    measured = InstrumentedModelAdapter(adapter=FinalModel(), route_id="test")
    call = ToolCall.capture(call_id="c1", name="lookup", arguments={"id": "x" * 800})
    measured.complete(followup(call))
    assert measured.observations[0].usage.input_tokens > 200
