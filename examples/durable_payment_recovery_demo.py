"""Deterministic demo: payment tool crash leaves run in recovery required."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cbrain import GovernedRuntime
from cbrain.agent import (
    AgentProfile,
    FoundationAgent,
    GovernedTool,
    RunInput,
    RunStatus,
    ToolRegistry,
)
from cbrain.agent.durable import DurableRunState, StoredRunRecord
from cbrain.agent.store_memory import InMemoryRunStore
from cbrain.models import CompletionRequest, ModelRouter, TextOutput, ToolCall


class SequenceModel:
    def __init__(self, outputs: list[Any]) -> None:
        self._outputs = list(outputs)

    @property
    def provider(self) -> str:
        return "demo"

    @property
    def model(self) -> str:
        return "demo-v1"

    def complete(self, request: CompletionRequest) -> TextOutput | ToolCall:
        if not self._outputs:
            raise RuntimeError("no scripted outputs remain")
        return self._outputs.pop(0)


class AllowGateway:
    independent_execution = False

    def decide_and_execute(self, action: Any, handler: Any) -> Any:
        from cbrain import ExecutionStatus, GovernedExecution

        return GovernedExecution(
            status=ExecutionStatus.EXECUTED,
            request_id=action.request_id,
            tool_executed=True,
            reason="demo_allow",
            output=handler(action.arguments),
        )


class CrashAfterInFlightClaimStore(InMemoryRunStore):
    """Persist TOOL_IN_FLIGHT, then simulate a process crash before execution."""

    def claim_tool_dispatch(
        self,
        run_id: str,
        *,
        expected_version: int,
    ) -> StoredRunRecord | None:
        claimed = super().claim_tool_dispatch(run_id, expected_version=expected_version)
        if claimed is not None:
            raise RuntimeError("simulated crash after in-flight marker")
        return claimed


def main() -> None:
    store = CrashAfterInFlightClaimStore()
    profile = AgentProfile(
        agent_id="demo-payments-agent",
        instructions="Submit payments only through governed tools.",
        model_route="local",
        permitted_tools=frozenset({"submit_payment"}),
        max_model_turns=3,
        max_tool_calls=1,
        timeout_seconds=60.0,
    )
    registry = ToolRegistry(
        [
            GovernedTool(
                name="submit_payment",
                capability="payments.submit",
                description="Submit a simulated payment instruction",
                input_schema={
                    "type": "object",
                    "properties": {
                        "account": {"type": "string"},
                        "amount_cents": {"type": "integer"},
                    },
                },
            )
        ]
    )
    model = SequenceModel(
        [
            ToolCall.capture(
                call_id="pay-1",
                name="submit_payment",
                arguments={"account": "vendor-42", "amount_cents": 125_000},
            ),
            TextOutput(text="payment complete"),
        ]
    )
    handler_calls = {"count": 0}

    def submit_payment(arguments: Mapping[str, Any]) -> dict[str, Any]:
        handler_calls["count"] += 1
        return {
            "account": arguments["account"],
            "amount_cents": arguments["amount_cents"],
            "status": "submitted",
        }

    agent = FoundationAgent(
        profile=profile,
        runtime=GovernedRuntime(AllowGateway()),
        model_router=ModelRouter({"local": model}),
        tools=registry,
        handlers={"submit_payment": submit_payment},
        clock=lambda: 1_700_000_000.0,
        run_id_factory=lambda: "demo-payment-run",
        request_id_factory=lambda run_id, step: f"{run_id}-step-{step}",
        run_store=store,
    )

    print("Starting durable payment run...")
    try:
        agent.run(
            RunInput(
                task="Pay vendor-42 USD 1,250.00",
                run_id="demo-payment-run",
            )
        )
    except RuntimeError as exc:
        print(f"Simulated crash: {exc}")

    inflight = store.load("demo-payment-run")
    print(f"Persisted durable state after crash: {inflight.durable_state.value}")
    print(f"Handler invocations before restart: {handler_calls['count']}")

    print("\nRestarting CBrain to resume the run...")
    restarted = FoundationAgent(
        profile=profile,
        runtime=GovernedRuntime(AllowGateway()),
        model_router=ModelRouter({"local": model}),
        tools=registry,
        handlers={"submit_payment": submit_payment},
        clock=lambda: 1_700_000_100.0,
        run_id_factory=lambda: "demo-payment-run",
        request_id_factory=lambda run_id, step: f"{run_id}-step-{step}",
        run_store=store,
    )
    result = restarted.run(
        RunInput(task="Pay vendor-42 USD 1,250.00", run_id="demo-payment-run")
    )
    print(f"Resume status: {result.status.value}")
    print(f"Handler invocations after resume: {handler_calls['count']}")
    print(
        "CBrain refused to repeat the consequential payment action; "
        "reconciliation is required before continuing."
    )

    assert handler_calls["count"] == 0
    assert result.status is RunStatus.EXECUTION_IN_FLIGHT
    persisted = store.load("demo-payment-run")
    assert persisted.durable_state is DurableRunState.TOOL_IN_FLIGHT


if __name__ == "__main__":
    main()
