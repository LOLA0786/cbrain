"""Durable, crash-safe foundation agent runs."""

from __future__ import annotations

import ast
import sqlite3
import threading
from collections.abc import Callable, Mapping
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
from cbrain.agent.durable import (
    DurableRunState,
    RunStoreError,
    StoredRunRecord,
    from_mapping,
    profile_fingerprint,
)
from cbrain.agent.store_memory import InMemoryRunStore
from cbrain.agent.store_sqlite import SQLiteRunStore
from cbrain.models import (
    CompletionRequest,
    ModelError,
    ModelRouter,
    TextOutput,
    ToolCall,
)


class CrashSimulation(RuntimeError):
    """Simulated process crash for durable-run tests."""


class SequenceModel:
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
    independent_execution = False

    def __init__(self) -> None:
        self.actions: list[ActionIntent] = []

    def decide_and_execute(
        self,
        action: ActionIntent,
        handler: Callable[[Mapping[str, Any]], Any],
    ) -> GovernedExecution:
        self.actions.append(action)
        return GovernedExecution(
            status=ExecutionStatus.EXECUTED,
            request_id=action.request_id,
            tool_executed=True,
            reason="test_allow",
            output=handler(action.arguments),
        )


class CrashAfterPreparedStore(InMemoryRunStore):
    def save(
        self,
        record: StoredRunRecord,
        *,
        expected_version: int,
    ) -> StoredRunRecord:
        updated = super().save(record, expected_version=expected_version)
        if updated.durable_state is DurableRunState.TOOL_PREPARED:
            raise CrashSimulation("crash after tool prepared")
        return updated


class CrashAfterCompletedStore(InMemoryRunStore):
    def save(
        self,
        record: StoredRunRecord,
        *,
        expected_version: int,
    ) -> StoredRunRecord:
        updated = super().save(record, expected_version=expected_version)
        if updated.durable_state is DurableRunState.TOOL_COMPLETED:
            raise CrashSimulation("crash after tool completed")
        return updated


def payment_tool() -> GovernedTool:
    return GovernedTool(
        name="submit_payment",
        capability="payments.submit",
        description="Submit a payment instruction",
        input_schema={
            "type": "object",
            "properties": {
                "account": {"type": "string"},
                "amount_cents": {"type": "integer"},
            },
        },
    )


def research_profile(*, permitted: frozenset[str] | None = None) -> AgentProfile:
    return AgentProfile(
        agent_id="payments-agent",
        instructions="Execute payment tools carefully.",
        model_route="local",
        permitted_tools=permitted or frozenset({"submit_payment"}),
        max_model_turns=4,
        max_tool_calls=2,
        timeout_seconds=30.0,
    )


def build_durable_agent(
    *,
    store: InMemoryRunStore | SQLiteRunStore,
    model: SequenceModel,
    gateway: AllowGateway | None = None,
    profile: AgentProfile | None = None,
    handler_calls: dict[str, int] | None = None,
    handlers: Mapping[str, Callable[[Mapping[str, Any]], Any]] | None = None,
) -> FoundationAgent:
    calls = handler_calls if handler_calls is not None else {"count": 0}
    resolved_profile = profile or research_profile()

    def default_handler(arguments: Mapping[str, Any]) -> dict[str, Any]:
        calls["count"] += 1
        return {
            "account": arguments["account"],
            "amount_cents": arguments["amount_cents"],
            "status": "submitted",
        }

    resolved_handlers = handlers or {"submit_payment": default_handler}

    return FoundationAgent(
        profile=resolved_profile,
        runtime=GovernedRuntime(gateway or AllowGateway()),
        model_router=ModelRouter({"local": model}),
        tools=ToolRegistry([payment_tool()]),
        handlers=resolved_handlers,
        clock=lambda: 1_700_000_000.0,
        run_id_factory=lambda: "run-durable-001",
        request_id_factory=lambda run_id, step: f"{run_id}-step-{step}",
        run_store=store,
    )


def test_normal_durable_run_completes() -> None:
    store = InMemoryRunStore()
    gateway = AllowGateway()
    model = SequenceModel(
        [
            ToolCall.capture(
                call_id="pay-1",
                name="submit_payment",
                arguments={"account": "acct-1", "amount_cents": 5000},
            ),
            TextOutput(text="payment submitted"),
        ]
    )
    agent = build_durable_agent(store=store, model=model, gateway=gateway)
    result = agent.run(RunInput(task="Pay vendor invoice", run_id="run-durable-001"))

    assert result.status is RunStatus.COMPLETED
    assert result.tool_calls == 1
    assert len(gateway.actions) == 1
    persisted = store.load("run-durable-001")
    assert persisted.durable_state is DurableRunState.COMPLETED


def test_resume_after_crash_before_tool_dispatch() -> None:
    store = CrashAfterPreparedStore()
    gateway = AllowGateway()
    model = SequenceModel(
        [
            ToolCall.capture(
                call_id="pay-1",
                name="submit_payment",
                arguments={"account": "acct-1", "amount_cents": 5000},
            ),
            TextOutput(text="payment submitted"),
        ]
    )
    agent = build_durable_agent(store=store, model=model, gateway=gateway)
    with pytest.raises(CrashSimulation):
        agent.run(RunInput(task="Pay vendor invoice", run_id="run-durable-001"))

    prepared = store.load("run-durable-001")
    assert prepared.durable_state is DurableRunState.TOOL_PREPARED
    assert prepared.pending_request_id == "run-durable-001-step-1"
    assert prepared.pending_idempotency_key == "run-durable-001-step-1"

    recovered = build_durable_agent(store=store, model=model, gateway=gateway).run(
        RunInput(task="Pay vendor invoice", run_id="run-durable-001")
    )
    assert recovered.status is RunStatus.COMPLETED
    assert len(gateway.actions) == 1
    assert gateway.actions[0].request_id == "run-durable-001-step-1"
    assert gateway.actions[0].idempotency_key == "run-durable-001-step-1"


def test_crash_while_tool_marked_in_flight() -> None:
    prepared_store = CrashAfterPreparedStore()
    model = SequenceModel(
        [
            ToolCall.capture(
                call_id="pay-1",
                name="submit_payment",
                arguments={"account": "acct-1", "amount_cents": 5000},
            ),
            TextOutput(text="never"),
        ]
    )
    first = build_durable_agent(store=prepared_store, model=model)
    with pytest.raises(CrashSimulation):
        first.run(RunInput(task="Pay vendor invoice", run_id="run-durable-001"))
    prepared = prepared_store.load("run-durable-001")
    assert prepared.durable_state is DurableRunState.TOOL_PREPARED
    inflight = StoredRunRecord(
        schema_version=prepared.schema_version,
        version=prepared.version + 1,
        run_id=prepared.run_id,
        durable_state=DurableRunState.TOOL_IN_FLIGHT,
        profile_fingerprint=prepared.profile_fingerprint,
        task=prepared.task,
        run_metadata=prepared.run_metadata,
        started_at=prepared.started_at,
        deadline=prepared.deadline,
        next_step=prepared.next_step,
        tool_calls=prepared.tool_calls,
        model_turns=prepared.model_turns,
        messages=prepared.messages,
        events=prepared.events,
        pending_request_id=prepared.pending_request_id,
        pending_idempotency_key=prepared.pending_idempotency_key,
        pending_action=prepared.pending_action,
        pending_tool_call_id=prepared.pending_tool_call_id,
        pending_tool_name=prepared.pending_tool_name,
    )
    store = InMemoryRunStore()
    store._records["run-durable-001"] = inflight
    assert store.load("run-durable-001").durable_state is DurableRunState.TOOL_IN_FLIGHT


def test_inflight_resume_requires_recovery_without_handler() -> None:
    store = InMemoryRunStore()
    handler_calls = {"count": 0}
    prepared_store = CrashAfterPreparedStore()
    model = SequenceModel(
        [
            ToolCall.capture(
                call_id="pay-1",
                name="submit_payment",
                arguments={"account": "acct-1", "amount_cents": 5000},
            ),
            TextOutput(text="done"),
        ]
    )
    first = build_durable_agent(
        store=prepared_store,
        model=model,
        handler_calls=handler_calls,
    )
    with pytest.raises(CrashSimulation):
        first.run(RunInput(task="Pay vendor invoice", run_id="run-durable-001"))
    prepared = prepared_store.load("run-durable-001")
    inflight = StoredRunRecord(
        schema_version=prepared.schema_version,
        version=prepared.version + 1,
        run_id=prepared.run_id,
        durable_state=DurableRunState.TOOL_IN_FLIGHT,
        profile_fingerprint=prepared.profile_fingerprint,
        task=prepared.task,
        run_metadata=prepared.run_metadata,
        started_at=prepared.started_at,
        deadline=prepared.deadline,
        next_step=prepared.next_step,
        tool_calls=prepared.tool_calls,
        model_turns=prepared.model_turns,
        messages=prepared.messages,
        events=prepared.events,
        pending_request_id=prepared.pending_request_id,
        pending_idempotency_key=prepared.pending_idempotency_key,
        pending_action=prepared.pending_action,
        pending_tool_call_id=prepared.pending_tool_call_id,
        pending_tool_name=prepared.pending_tool_name,
    )
    store._records["run-durable-001"] = inflight
    gateway = AllowGateway()
    recovered = build_durable_agent(
        store=store,
        model=model,
        gateway=gateway,
        handler_calls=handler_calls,
    )
    result = recovered.run(
        RunInput(task="Pay vendor invoice", run_id="run-durable-001")
    )
    assert result.status is RunStatus.RECOVERY_REQUIRED
    assert handler_calls["count"] == 0
    assert gateway.actions == []


def test_crash_after_tool_completion_resumes_without_duplicate() -> None:
    store = CrashAfterCompletedStore()
    gateway = AllowGateway()
    handler_calls = {"count": 0}
    model = SequenceModel(
        [
            ToolCall.capture(
                call_id="pay-1",
                name="submit_payment",
                arguments={"account": "acct-1", "amount_cents": 5000},
            ),
            TextOutput(text="payment submitted"),
        ]
    )
    agent = build_durable_agent(
        store=store,
        model=model,
        gateway=gateway,
        handler_calls=handler_calls,
    )
    with pytest.raises(CrashSimulation):
        agent.run(RunInput(task="Pay vendor invoice", run_id="run-durable-001"))
    assert handler_calls["count"] == 1
    assert len(gateway.actions) == 1

    recovered = build_durable_agent(
        store=store,
        model=model,
        gateway=gateway,
        handler_calls=handler_calls,
    )
    result = recovered.run(
        RunInput(task="Pay vendor invoice", run_id="run-durable-001")
    )
    assert result.status is RunStatus.COMPLETED
    assert handler_calls["count"] == 1
    assert len(gateway.actions) == 1


def test_stable_request_id_and_idempotency_key_across_resume() -> None:
    store = CrashAfterPreparedStore()
    gateway = AllowGateway()
    model = SequenceModel(
        [
            ToolCall.capture(
                call_id="pay-1",
                name="submit_payment",
                arguments={"account": "acct-1", "amount_cents": 5000},
            ),
            TextOutput(text="done"),
        ]
    )
    first = build_durable_agent(store=store, model=model, gateway=gateway)
    with pytest.raises(CrashSimulation):
        first.run(RunInput(task="Pay vendor invoice", run_id="run-durable-001"))
    build_durable_agent(store=store, model=model, gateway=gateway).run(
        RunInput(task="Pay vendor invoice", run_id="run-durable-001")
    )
    assert gateway.actions[0].request_id == "run-durable-001-step-1"
    assert gateway.actions[0].idempotency_key == "run-durable-001-step-1"


def test_corrupted_sqlite_record_fails_closed(tmp_path: Path) -> None:
    db_path = tmp_path / "runs.db"
    store = SQLiteRunStore(db_path)
    store.create(
        StoredRunRecord(
            schema_version=1,
            version=0,
            run_id="broken",
            durable_state=DurableRunState.RUNNING,
            profile_fingerprint="abc",
            task="task",
            run_metadata={},
            started_at=0.0,
            deadline=10.0,
            next_step=1,
            tool_calls=0,
            model_turns=0,
            messages=(),
            events=(),
        )
    )
    store.close()
    connection = sqlite3.connect(db_path)
    connection.execute(
        "UPDATE durable_runs SET record_json = ? WHERE run_id = ?",
        ("not-json", "broken"),
    )
    connection.commit()
    connection.close()
    broken_store = SQLiteRunStore(db_path)
    with pytest.raises(RunStoreError):
        broken_store.load("broken")


def test_unknown_schema_version_fails_closed() -> None:
    with pytest.raises(RunStoreError, match="unsupported"):
        from_mapping({"schema_version": 999, "run_id": "x"})


def test_agent_profile_mismatch_blocks_resume() -> None:
    store = CrashAfterPreparedStore()
    model = SequenceModel(
        [
            ToolCall.capture(
                call_id="pay-1",
                name="submit_payment",
                arguments={"account": "acct-1", "amount_cents": 5000},
            ),
            TextOutput(text="done"),
        ]
    )
    first = build_durable_agent(store=store, model=model)
    with pytest.raises(CrashSimulation):
        first.run(RunInput(task="Pay vendor invoice", run_id="run-durable-001"))
    changed = AgentProfile(
        agent_id="payments-agent",
        instructions="Different instructions.",
        model_route="local",
        permitted_tools=frozenset({"submit_payment"}),
        max_model_turns=4,
        max_tool_calls=2,
        timeout_seconds=30.0,
    )
    with pytest.raises(FoundationAgentError, match="fingerprint"):
        build_durable_agent(store=store, model=model, profile=changed).run(
            RunInput(task="Pay vendor invoice", run_id="run-durable-001")
        )


def test_tool_permission_mismatch_blocks_resume() -> None:
    store = CrashAfterPreparedStore()
    model = SequenceModel(
        [
            ToolCall.capture(
                call_id="pay-1",
                name="submit_payment",
                arguments={"account": "acct-1", "amount_cents": 5000},
            ),
            TextOutput(text="done"),
        ]
    )
    first = build_durable_agent(store=store, model=model)
    with pytest.raises(CrashSimulation):
        first.run(RunInput(task="Pay vendor invoice", run_id="run-durable-001"))
    restricted = research_profile(permitted=frozenset({"submit_payment", "audit_log"}))
    with pytest.raises(FoundationAgentError, match="fingerprint"):
        build_durable_agent(
            store=store,
            model=model,
            profile=restricted,
            handlers={"submit_payment": lambda _a: {}, "audit_log": lambda _a: {}},
        ).run(RunInput(task="Pay vendor invoice", run_id="run-durable-001"))


def test_two_resume_attempts_cannot_both_claim_dispatch() -> None:
    store = InMemoryRunStore()
    profile = research_profile()
    prepared = StoredRunRecord(
        schema_version=1,
        version=0,
        run_id="run-durable-001",
        durable_state=DurableRunState.TOOL_PREPARED,
        profile_fingerprint=profile_fingerprint(profile),
        task="Pay vendor invoice",
        run_metadata={},
        started_at=0.0,
        deadline=100.0,
        next_step=1,
        tool_calls=0,
        model_turns=1,
        messages=(),
        events=(),
        pending_request_id="run-durable-001-step-1",
        pending_idempotency_key="run-durable-001-step-1",
        pending_action={
            "agent_id": "payments-agent",
            "framework": "cbrain-foundation",
            "tool_name": "submit_payment",
            "capability": "payments.submit",
            "arguments": {"account": "acct-1", "amount_cents": 5000},
            "request_id": "run-durable-001-step-1",
            "idempotency_key": "run-durable-001-step-1",
            "timestamp": 1.0,
            "context": {"run_id": "run-durable-001", "tool_call_id": "pay-1"},
        },
        pending_tool_call_id="pay-1",
        pending_tool_name="submit_payment",
    )
    store.create(prepared)
    claims: list[bool] = []

    def claim_once() -> None:
        claimed = store.claim_tool_dispatch(
            "run-durable-001",
            expected_version=0,
        )
        claims.append(claimed is not None)

    threads = [threading.Thread(target=claim_once) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(claims) == 1


def test_completed_run_is_idempotently_readable() -> None:
    store = InMemoryRunStore()
    model = SequenceModel([TextOutput(text="done")])
    agent = build_durable_agent(store=store, model=model)
    first = agent.run(RunInput(task="Pay vendor invoice", run_id="run-durable-001"))
    second = build_durable_agent(store=store, model=model).run(
        RunInput(task="Pay vendor invoice", run_id="run-durable-001")
    )
    assert first.status is RunStatus.COMPLETED
    assert second.status is RunStatus.COMPLETED
    assert second.final_text == first.final_text


def test_non_durable_foundation_agent_remains_compatible() -> None:
    model = SequenceModel([TextOutput(text="ephemeral")])
    agent = FoundationAgent(
        profile=research_profile(),
        runtime=GovernedRuntime(AllowGateway()),
        model_router=ModelRouter({"local": model}),
        tools=ToolRegistry([payment_tool()]),
        handlers={"submit_payment": lambda _a: {"ok": True}},
        clock=lambda: 1_700_000_000.0,
    )
    result = agent.run(RunInput(task="hello"))
    assert result.status is RunStatus.COMPLETED
    assert result.final_text == "ephemeral"


def test_durable_core_has_no_privatevault_imports() -> None:
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
