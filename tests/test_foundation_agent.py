"""Foundation agent loop: bounded reasoning, governed tool execution."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

import pytest

from cbrain import ActionIntent, ExecutionStatus, GovernedExecution, GovernedRuntime
from cbrain.agent import (
    AgentProfile,
    FoundationAgent,
    FoundationAgentError,
    GovernedTool,
    RunEventKind,
    RunInput,
    RunLimits,
    RunStatus,
    ToolRegistry,
)
from cbrain.models import (
    CompletionRequest,
    MessageRole,
    ModelError,
    ModelRouter,
    TextOutput,
    ToolCall,
)


class SequenceModel:
    """Deterministic model that returns a scripted output sequence."""

    def __init__(self, outputs: list[Any]) -> None:
        self._outputs = list(outputs)
        self.requests: list[CompletionRequest] = []

    @property
    def provider(self) -> str:
        return "sequence"

    @property
    def model(self) -> str:
        return "sequence-v1"

    def complete(self, request: CompletionRequest) -> TextOutput | ToolCall:
        self.requests.append(request)
        if not self._outputs:
            raise ModelError("no scripted outputs remain")
        return self._outputs.pop(0)


class AllowGateway:
    """Minimal gateway that executes the handler once and returns EXECUTED."""

    def __init__(self, *, independent_execution: bool = False) -> None:
        self.independent_execution = independent_execution
        self.actions: list[ActionIntent] = []

    def decide_and_execute(
        self,
        action: ActionIntent,
        handler: Callable[[Mapping[str, Any]], Any],
    ) -> GovernedExecution:
        self.actions.append(action)
        output = handler(action.arguments)
        return GovernedExecution(
            status=ExecutionStatus.EXECUTED,
            request_id=action.request_id,
            tool_executed=True,
            reason="test_allow",
            output=output,
        )


class BlockGateway:
    def __init__(self) -> None:
        self.independent_execution = False

    def decide_and_execute(
        self,
        action: ActionIntent,
        handler: Callable[[Mapping[str, Any]], Any],
    ) -> GovernedExecution:
        return GovernedExecution(
            status=ExecutionStatus.BLOCKED,
            request_id=action.request_id,
            tool_executed=False,
            reason="test_block",
        )


class IndeterminateOnceGateway:
    def __init__(self) -> None:
        self.independent_execution = False
        self.calls = 0

    def decide_and_execute(
        self,
        action: ActionIntent,
        handler: Callable[[Mapping[str, Any]], Any],
    ) -> GovernedExecution:
        self.calls += 1
        handler(action.arguments)
        if self.calls == 1:
            return GovernedExecution(
                status=ExecutionStatus.INDETERMINATE,
                request_id=action.request_id,
                tool_executed=None,
                reason="maybe_executed",
                retryable=False,
            )
        return GovernedExecution(
            status=ExecutionStatus.EXECUTED,
            request_id=action.request_id,
            tool_executed=True,
            reason="retry_should_not_happen",
            output="retried",
        )


def lookup_tool() -> GovernedTool:
    return GovernedTool(
        name="lookup",
        capability="data.lookup",
        description="Look up a record by id",
        input_schema={"type": "object", "properties": {"id": {"type": "string"}}},
    )


def write_tool() -> GovernedTool:
    return GovernedTool(
        name="write_note",
        capability="notes.write",
        description="Write a note",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
        },
    )


def build_agent(
    *,
    profile: AgentProfile,
    model: SequenceModel,
    gateway: object,
    registry: ToolRegistry,
    handlers: Mapping[str, Callable[[Mapping[str, Any]], Any]],
    clock: Callable[[], float] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> FoundationAgent:
    return FoundationAgent(
        profile=profile,
        runtime=GovernedRuntime(gateway),
        model_router=ModelRouter({profile.model_route: model}),
        tools=registry,
        handlers=handlers,
        clock=clock or (lambda: 1_700_000_000.0),
        run_id_factory=lambda: "run-test-001",
        request_id_factory=lambda run_id, step: f"{run_id}-step-{step}",
        cancelled=cancelled or (lambda: False),
    )


def research_profile() -> AgentProfile:
    return AgentProfile(
        agent_id="research-agent",
        instructions="Answer with concise facts.",
        model_route="local",
        permitted_tools=frozenset({"lookup"}),
        max_model_turns=4,
        max_tool_calls=2,
        timeout_seconds=30.0,
    )


def writer_profile() -> AgentProfile:
    return AgentProfile(
        agent_id="writer-agent",
        instructions="Draft short notes.",
        model_route="local",
        permitted_tools=frozenset({"write_note"}),
        max_model_turns=3,
        max_tool_calls=1,
        timeout_seconds=10.0,
    )


def test_task_completes_without_tool() -> None:
    model = SequenceModel([TextOutput(text="done without tools")])
    agent = build_agent(
        profile=research_profile(),
        model=model,
        gateway=AllowGateway(),
        registry=ToolRegistry([lookup_tool()]),
        handlers={"lookup": lambda _args: {"ok": True}},
    )

    result = agent.run(RunInput(task="Summarize the quarterly update"))

    assert result.status is RunStatus.COMPLETED
    assert result.final_text == "done without tools"
    assert result.tool_calls == 0
    assert result.run_id == "run-test-001"
    assert any(event.kind is RunEventKind.RUN_STARTED for event in result.events)


def test_valid_tool_call_completes_through_governed_runtime() -> None:
    model = SequenceModel(
        [
            ToolCall.capture(
                call_id="call-1",
                name="lookup",
                arguments={"id": "acct-7"},
            ),
            TextOutput(text="account acct-7 is active"),
        ]
    )
    gateway = AllowGateway()
    direct_calls = 0

    def handler(arguments: Mapping[str, Any]) -> dict[str, Any]:
        nonlocal direct_calls
        direct_calls += 1
        return {"id": arguments["id"], "status": "active"}

    agent = build_agent(
        profile=research_profile(),
        model=model,
        gateway=gateway,
        registry=ToolRegistry([lookup_tool()]),
        handlers={"lookup": handler},
    )

    result = agent.run(RunInput(task="Check account acct-7"))

    assert result.status is RunStatus.COMPLETED
    assert result.tool_calls == 1
    assert len(gateway.actions) == 1
    assert gateway.actions[0].capability == "data.lookup"
    assert gateway.actions[0].request_id == "run-test-001-step-1"
    assert direct_calls == 1
    tool_event = next(
        event for event in result.events if event.kind is RunEventKind.TOOL_COMPLETED
    )
    assert tool_event.detail["request_id"] == "run-test-001-step-1"


def test_tool_handler_not_invoked_outside_governed_runtime() -> None:
    model = SequenceModel(
        [
            ToolCall.capture(call_id="c1", name="lookup", arguments={"id": "1"}),
            TextOutput(text="ok"),
        ]
    )
    invocations: list[str] = []

    class TrackingGateway:
        independent_execution = False

        def decide_and_execute(self, action, handler):
            invocations.append("gateway")
            return GovernedExecution(
                status=ExecutionStatus.EXECUTED,
                request_id=action.request_id,
                tool_executed=True,
                reason="tracked",
                output=handler(action.arguments),
            )

    def handler(_arguments: Mapping[str, Any]) -> dict[str, str]:
        invocations.append("handler")
        return {"seen": "inside_gateway"}

    build_agent(
        profile=research_profile(),
        model=model,
        gateway=TrackingGateway(),
        registry=ToolRegistry([lookup_tool()]),
        handlers={"lookup": handler},
    ).run(RunInput(task="lookup"))

    assert invocations == ["gateway", "handler"]


def test_unknown_tool_rejected_before_execution() -> None:
    model = SequenceModel(
        [
            ToolCall.capture(
                call_id="c1",
                name="delete_everything",
                arguments={"confirm": True},
            ),
        ]
    )
    gateway = AllowGateway()

    result = build_agent(
        profile=research_profile(),
        model=model,
        gateway=gateway,
        registry=ToolRegistry([lookup_tool()]),
        handlers={"lookup": lambda _a: {}},
    ).run(RunInput(task="danger"))

    assert result.status is RunStatus.INVALID_MODEL_RESPONSE
    assert gateway.actions == []
    assert result.tool_calls == 0


def test_tool_not_in_profile_rejected_before_execution() -> None:
    model = SequenceModel(
        [
            ToolCall.capture(
                call_id="c1",
                name="write_note",
                arguments={"text": "hello"},
            ),
        ]
    )
    gateway = AllowGateway()
    registry = ToolRegistry([lookup_tool(), write_tool()])

    result = build_agent(
        profile=research_profile(),
        model=model,
        gateway=gateway,
        registry=registry,
        handlers={"lookup": lambda _a: {}},
    ).run(RunInput(task="write"))

    assert result.status is RunStatus.INVALID_MODEL_RESPONSE
    assert gateway.actions == []


def test_malformed_model_output_rejected() -> None:
    class BrokenModel:
        provider = "broken"
        model = "broken"

        def complete(self, _request: CompletionRequest) -> TextOutput:
            raise ModelError("provider exploded")

    agent = FoundationAgent(
        profile=research_profile(),
        runtime=GovernedRuntime(AllowGateway()),
        model_router=ModelRouter({"local": BrokenModel()}),
        tools=ToolRegistry([lookup_tool()]),
        handlers={"lookup": lambda _a: {}},
        clock=lambda: 1_700_000_000.0,
        run_id_factory=lambda: "run-broken",
        request_id_factory=lambda run_id, step: f"{run_id}-{step}",
    )

    result = agent.run(RunInput(task="fail"))

    assert result.status is RunStatus.MODEL_FAILURE


def test_maximum_step_protection() -> None:
    model = SequenceModel(
        [
            ToolCall.capture(call_id="1", name="lookup", arguments={"id": "1"}),
            ToolCall.capture(call_id="2", name="lookup", arguments={"id": "2"}),
            ToolCall.capture(call_id="3", name="lookup", arguments={"id": "3"}),
        ]
    )
    profile = AgentProfile(
        agent_id="research-agent",
        instructions="Keep looking things up.",
        model_route="local",
        permitted_tools=frozenset({"lookup"}),
        max_model_turns=2,
        max_tool_calls=5,
        timeout_seconds=30.0,
    )

    result = build_agent(
        profile=profile,
        model=model,
        gateway=AllowGateway(),
        registry=ToolRegistry([lookup_tool()]),
        handlers={"lookup": lambda args: args},
    ).run(RunInput(task="loop"))

    assert result.status is RunStatus.LIMIT_REACHED
    assert result.tool_calls == 2


def test_tool_failure_preserves_structured_trace() -> None:
    model = SequenceModel(
        [
            ToolCall.capture(call_id="c1", name="lookup", arguments={"id": "x"}),
        ]
    )

    result = build_agent(
        profile=research_profile(),
        model=model,
        gateway=BlockGateway(),
        registry=ToolRegistry([lookup_tool()]),
        handlers={"lookup": lambda _a: {}},
    ).run(RunInput(task="blocked lookup"))

    assert result.status is RunStatus.REJECTED
    assert result.tool_calls == 1
    failure = next(
        event for event in result.events if event.kind is RunEventKind.TOOL_REJECTED
    )
    assert failure.detail["execution_status"] == ExecutionStatus.BLOCKED.value
    assert failure.detail["request_id"] == "run-test-001-step-1"


def test_timeout_terminates_run() -> None:
    times = iter([0.0, 6.0])
    profile = AgentProfile(
        agent_id="research-agent",
        instructions="Work quickly.",
        model_route="local",
        permitted_tools=frozenset({"lookup"}),
        max_model_turns=4,
        max_tool_calls=2,
        timeout_seconds=5.0,
    )
    model = SequenceModel(
        [
            ToolCall.capture(call_id="c1", name="lookup", arguments={"id": "1"}),
            TextOutput(text="late"),
        ]
    )

    result = build_agent(
        profile=profile,
        model=model,
        gateway=AllowGateway(),
        registry=ToolRegistry([lookup_tool()]),
        handlers={"lookup": lambda args: args},
        clock=lambda: next(times),
    ).run(RunInput(task="slow"))

    assert result.status is RunStatus.TIMED_OUT


def test_cancellation_terminates_run() -> None:
    calls = {"count": 0}

    def cancelled() -> bool:
        calls["count"] += 1
        return calls["count"] > 1

    model = SequenceModel(
        [
            ToolCall.capture(call_id="c1", name="lookup", arguments={"id": "1"}),
            TextOutput(text="never"),
        ]
    )

    result = build_agent(
        profile=research_profile(),
        model=model,
        gateway=AllowGateway(),
        registry=ToolRegistry([lookup_tool()]),
        handlers={"lookup": lambda args: args},
        cancelled=cancelled,
    ).run(RunInput(task="cancel me"))

    assert result.status is RunStatus.CANCELLED


def test_request_and_run_identifiers_propagate() -> None:
    model = SequenceModel(
        [
            ToolCall.capture(call_id="c1", name="lookup", arguments={"id": "9"}),
            TextOutput(text="found"),
        ]
    )
    gateway = AllowGateway()

    result = build_agent(
        profile=research_profile(),
        model=model,
        gateway=gateway,
        registry=ToolRegistry([lookup_tool()]),
        handlers={"lookup": lambda args: args},
    ).run(RunInput(task="trace", run_id="custom-run-42"))

    assert result.run_id == "custom-run-42"
    assert gateway.actions[0].request_id == "custom-run-42-step-1"
    assert gateway.actions[0].agent_id == "research-agent"


def test_indeterminate_tool_execution_is_not_retried() -> None:
    gateway = IndeterminateOnceGateway()
    model = SequenceModel(
        [
            ToolCall.capture(call_id="c1", name="lookup", arguments={"id": "1"}),
            ToolCall.capture(call_id="c2", name="lookup", arguments={"id": "2"}),
        ]
    )

    result = build_agent(
        profile=research_profile(),
        model=model,
        gateway=gateway,
        registry=ToolRegistry([lookup_tool()]),
        handlers={"lookup": lambda args: args},
    ).run(RunInput(task="risky"))

    assert gateway.calls == 1
    assert result.status is RunStatus.TOOL_FAILURE
    assert result.tool_calls == 1


def test_two_profiles_share_foundation_implementation() -> None:
    lookup_model = SequenceModel(
        [
            ToolCall.capture(call_id="l1", name="lookup", arguments={"id": "1"}),
            TextOutput(text="research done"),
        ]
    )
    write_model = SequenceModel([TextOutput(text="draft ready")])

    research = build_agent(
        profile=research_profile(),
        model=lookup_model,
        gateway=AllowGateway(),
        registry=ToolRegistry([lookup_tool(), write_tool()]),
        handlers={"lookup": lambda args: args},
    )
    writer = build_agent(
        profile=writer_profile(),
        model=write_model,
        gateway=AllowGateway(),
        registry=ToolRegistry([lookup_tool(), write_tool()]),
        handlers={"write_note": lambda args: args},
    )

    research_result = research.run(RunInput(task="research"))
    writer_result = writer.run(RunInput(task="write"))

    assert research_result.status is RunStatus.COMPLETED
    assert research_result.tool_calls == 1
    assert writer_result.status is RunStatus.COMPLETED
    assert writer_result.tool_calls == 0
    assert research_result.final_text == "research done"
    assert writer_result.final_text == "draft ready"


def test_foundation_agent_has_no_privatevault_imports() -> None:
    import ast
    from pathlib import Path

    root = Path("cbrain/agent")
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "privatevault" not in alias.name
                    assert "agent_dna" not in alias.name
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "privatevault" not in node.module
                assert "agent_dna" not in node.module


def test_run_input_validation() -> None:
    agent = build_agent(
        profile=research_profile(),
        model=SequenceModel([TextOutput(text="ok")]),
        gateway=AllowGateway(),
        registry=ToolRegistry([lookup_tool()]),
        handlers={"lookup": lambda _a: {}},
    )

    with pytest.raises(ValueError, match="task"):
        agent.run(RunInput(task="   "))


def test_model_receives_tool_definitions_for_permitted_tools_only() -> None:
    model = SequenceModel([TextOutput(text="ok")])
    build_agent(
        profile=research_profile(),
        model=model,
        gateway=AllowGateway(),
        registry=ToolRegistry([lookup_tool(), write_tool()]),
        handlers={"lookup": lambda _a: {}},
    ).run(RunInput(task="hello"))

    assert len(model.requests) == 1
    tool_names = {tool.name for tool in model.requests[0].tools}
    assert tool_names == {"lookup"}


def test_tool_observation_is_structured_for_follow_up_turn() -> None:
    model = SequenceModel(
        [
            ToolCall.capture(call_id="c1", name="lookup", arguments={"id": "7"}),
            TextOutput(text="done"),
        ]
    )
    build_agent(
        profile=research_profile(),
        model=model,
        gateway=AllowGateway(),
        registry=ToolRegistry([lookup_tool()]),
        handlers={"lookup": lambda _a: {"id": "7", "status": "active"}},
    ).run(RunInput(task="trace observation"))

    assert len(model.requests) == 2
    observation = model.requests[1].messages[-1]
    assert observation.role is MessageRole.TOOL
    payload = json.loads(observation.content)
    assert payload["status"] == "EXECUTED"
    assert payload["output"]["status"] == "active"
    assert payload["request_id"] == "run-test-001-step-1"


def test_extra_handlers_rejected_at_construction() -> None:
    with pytest.raises(FoundationAgentError, match="non-permitted"):
        build_agent(
            profile=research_profile(),
            model=SequenceModel([TextOutput(text="ok")]),
            gateway=AllowGateway(),
            registry=ToolRegistry([lookup_tool(), write_tool()]),
            handlers={
                "lookup": lambda _a: {},
                "write_note": lambda _a: {},
            },
        )


def test_timeout_seconds_rejects_nan() -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        AgentProfile(
            agent_id="agent",
            instructions="x",
            model_route="local",
            permitted_tools=frozenset({"lookup"}),
            max_model_turns=1,
            max_tool_calls=0,
            timeout_seconds=float("nan"),
        )


def test_task_length_bound_enforced() -> None:
    profile = AgentProfile(
        agent_id="agent",
        instructions="x",
        model_route="local",
        permitted_tools=frozenset({"lookup"}),
        max_model_turns=2,
        max_tool_calls=1,
        timeout_seconds=30.0,
        limits=RunLimits(max_task_chars=8),
    )
    result = build_agent(
        profile=profile,
        model=SequenceModel([TextOutput(text="ok")]),
        gateway=AllowGateway(),
        registry=ToolRegistry([lookup_tool()]),
        handlers={"lookup": lambda _a: {}},
    ).run(RunInput(task="123456789"))

    assert result.status is RunStatus.INVALID_MODEL_RESPONSE


def test_model_text_length_bound_enforced() -> None:
    profile = AgentProfile(
        agent_id="agent",
        instructions="x",
        model_route="local",
        permitted_tools=frozenset({"lookup"}),
        max_model_turns=2,
        max_tool_calls=0,
        timeout_seconds=30.0,
        limits=RunLimits(max_model_text_chars=4),
    )
    result = build_agent(
        profile=profile,
        model=SequenceModel([TextOutput(text="12345")]),
        gateway=AllowGateway(),
        registry=ToolRegistry([lookup_tool()]),
        handlers={"lookup": lambda _a: {}},
    ).run(RunInput(task="hello"))

    assert result.status is RunStatus.INVALID_MODEL_RESPONSE


def test_observation_length_bound_enforced() -> None:
    profile = AgentProfile(
        agent_id="agent",
        instructions="x",
        model_route="local",
        permitted_tools=frozenset({"lookup"}),
        max_model_turns=3,
        max_tool_calls=2,
        timeout_seconds=30.0,
        limits=RunLimits(max_observation_chars=32),
    )
    model = SequenceModel(
        [
            ToolCall.capture(call_id="c1", name="lookup", arguments={"id": "1"}),
            TextOutput(text="done"),
        ]
    )
    result = build_agent(
        profile=profile,
        model=model,
        gateway=AllowGateway(),
        registry=ToolRegistry([lookup_tool()]),
        handlers={"lookup": lambda _a: {"payload": "x" * 100}},
    ).run(RunInput(task="big observation"))

    assert result.status is RunStatus.LIMIT_REACHED


def test_cancellation_before_first_model_turn() -> None:
    model = SequenceModel([TextOutput(text="never")])
    result = build_agent(
        profile=research_profile(),
        model=model,
        gateway=AllowGateway(),
        registry=ToolRegistry([lookup_tool()]),
        handlers={"lookup": lambda _a: {}},
        cancelled=lambda: True,
    ).run(RunInput(task="stop now"))

    assert result.status is RunStatus.CANCELLED
    assert model.requests == []


def test_blocked_gateway_never_invokes_handler() -> None:
    handler_calls = 0

    def handler(_arguments: Mapping[str, Any]) -> None:
        nonlocal handler_calls
        handler_calls += 1

    build_agent(
        profile=research_profile(),
        model=SequenceModel(
            [ToolCall.capture(call_id="c1", name="lookup", arguments={"id": "1"})]
        ),
        gateway=BlockGateway(),
        registry=ToolRegistry([lookup_tool()]),
        handlers={"lookup": handler},
    ).run(RunInput(task="blocked"))

    assert handler_calls == 0


def test_handler_exception_becomes_tool_failure_without_retry() -> None:
    class ExplodingGateway:
        independent_execution = False
        calls = 0

        def decide_and_execute(self, action, handler):
            self.calls += 1
            handler(action.arguments)
            raise RuntimeError("closure failed")

    gateway = ExplodingGateway()
    model = SequenceModel(
        [
            ToolCall.capture(call_id="c1", name="lookup", arguments={"id": "1"}),
            ToolCall.capture(call_id="c2", name="lookup", arguments={"id": "2"}),
        ]
    )
    result = build_agent(
        profile=research_profile(),
        model=model,
        gateway=gateway,
        registry=ToolRegistry([lookup_tool()]),
        handlers={"lookup": lambda _a: (_ for _ in ()).throw(ValueError("boom"))},
    ).run(RunInput(task="explode"))

    assert gateway.calls == 1
    assert result.status is RunStatus.TOOL_FAILURE
    assert result.tool_calls == 1


def test_sequential_runs_do_not_leak_messages() -> None:
    model = SequenceModel(
        [
            ToolCall.capture(call_id="c1", name="lookup", arguments={"id": "1"}),
            TextOutput(text="first"),
            TextOutput(text="second"),
        ]
    )
    agent = build_agent(
        profile=research_profile(),
        model=model,
        gateway=AllowGateway(),
        registry=ToolRegistry([lookup_tool()]),
        handlers={"lookup": lambda args: args},
    )

    first = agent.run(RunInput(task="first task", run_id="run-a"))
    second = agent.run(RunInput(task="second task", run_id="run-b"))

    assert first.status is RunStatus.COMPLETED
    assert second.status is RunStatus.COMPLETED
    assert len(model.requests) == 3
    assert model.requests[0].messages[1].content == "first task"
    assert model.requests[2].messages[1].content == "second task"
    assert all(
        message.role is not MessageRole.TOOL for message in model.requests[2].messages
    )


def test_failed_run_then_successful_run() -> None:
    model = SequenceModel(
        [
            ToolCall.capture(
                call_id="c1",
                name="delete_everything",
                arguments={"confirm": True},
            ),
            TextOutput(text="recovered"),
        ]
    )
    agent = build_agent(
        profile=research_profile(),
        model=model,
        gateway=AllowGateway(),
        registry=ToolRegistry([lookup_tool()]),
        handlers={"lookup": lambda _a: {}},
    )
    failed = agent.run(RunInput(task="fail first", run_id="run-fail"))
    recovered = agent.run(RunInput(task="ok now", run_id="run-ok"))

    assert failed.status is RunStatus.INVALID_MODEL_RESPONSE
    assert recovered.status is RunStatus.COMPLETED
    assert recovered.run_id == "run-ok"


def test_run_id_whitespace_rejected() -> None:
    with pytest.raises(ValueError, match="run_id"):
        RunInput(task="hello", run_id="   ")


def test_metadata_snapshot_is_immutable() -> None:
    metadata = {"team": "platform"}
    run_input = RunInput(task="hello", metadata=metadata)
    metadata["team"] = "mutated"
    assert run_input.metadata is not None
    assert run_input.metadata["team"] == "platform"


def test_tool_schema_mutation_does_not_change_registry() -> None:
    schema: dict[str, object] = {
        "type": "object",
        "properties": {"id": {"type": "string"}},
    }
    tool = GovernedTool(
        name="lookup",
        capability="data.lookup",
        description="lookup",
        input_schema=schema,
    )
    schema["properties"] = {}
    assert tool.input_schema["properties"] == {"id": {"type": "string"}}
