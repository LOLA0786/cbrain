from __future__ import annotations

from typing import Any

import pytest

from cbrain import (
    ActionIntent,
    ContractError,
    ExecutionStatus,
    GovernedExecution,
    GovernedRuntime,
)


def action(arguments: dict[str, Any] | None = None) -> ActionIntent:
    return ActionIntent.capture(
        request_id="req-1",
        agent_id="agent-1",
        framework="langchain",
        tool_name="transfer_funds",
        capability="payment.execute",
        timestamp=1_700_000_000.0,
        arguments=arguments or {
            "amount": "100.00",
            "currency": "USD",
        },
    )


class StubGateway:
    def __init__(
        self,
        status: ExecutionStatus,
        *,
        call_handler: bool,
        raise_after_call: bool = False,
        request_id: str = "req-1",
    ) -> None:
        self.status = status
        self.call_handler = call_handler
        self.raise_after_call = raise_after_call
        self.request_id = request_id

    def decide_and_execute(self, intent, handler):
        output = None

        if self.call_handler:
            output = handler(intent.arguments)

        if self.raise_after_call:
            raise OSError("simulated closure failure")

        return GovernedExecution(
            status=self.status,
            request_id=self.request_id,
            tool_executed=self.call_handler,
            reason="stub",
            output=output,
        )


class DoubleDispatchGateway:
    def decide_and_execute(self, intent, handler):
        handler(intent.arguments)
        handler(intent.arguments)
        raise AssertionError("second dispatch should have failed")


@pytest.mark.parametrize(
    "status",
    [
        ExecutionStatus.BLOCKED,
        ExecutionStatus.REVIEW_REQUIRED,
    ],
)
def test_non_allow_decisions_never_call_tool(status):
    calls = 0

    def tool(_arguments):
        nonlocal calls
        calls += 1

    result = GovernedRuntime(
        StubGateway(status, call_handler=False)
    ).execute(action(), tool)

    assert result.status is status
    assert result.tool_executed is False
    assert calls == 0


def test_executed_calls_tool_exactly_once():
    calls = 0

    def tool(arguments):
        nonlocal calls
        calls += 1
        return arguments["amount"]

    result = GovernedRuntime(
        StubGateway(
            ExecutionStatus.EXECUTED,
            call_handler=True,
        )
    ).execute(action(), tool)

    assert result.status is ExecutionStatus.EXECUTED
    assert result.output == "100.00"
    assert calls == 1


def test_failure_after_tool_call_is_indeterminate():
    result = GovernedRuntime(
        StubGateway(
            ExecutionStatus.EXECUTED,
            call_handler=True,
            raise_after_call=True,
        )
    ).execute(action(), lambda _arguments: "sent")

    assert result.status is ExecutionStatus.INDETERMINATE
    assert result.tool_executed is True
    assert result.retryable is False


def test_gateway_failure_before_tool_call_is_fail_closed():
    class BrokenGateway:
        def decide_and_execute(self, intent, handler):
            raise ConnectionError("offline")

    result = GovernedRuntime(BrokenGateway()).execute(
        action(),
        lambda _arguments: pytest.fail("tool must not execute"),
    )

    assert result.status is ExecutionStatus.CONTROL_FAILURE
    assert result.tool_executed is False
    assert result.retryable is False


def test_gateway_cannot_dispatch_twice():
    calls = 0

    def tool(_arguments):
        nonlocal calls
        calls += 1

    result = GovernedRuntime(
        DoubleDispatchGateway()
    ).execute(action(), tool)

    assert result.status is ExecutionStatus.INDETERMINATE
    assert result.tool_executed is True
    assert result.retryable is False
    assert calls == 1


def test_gateway_cannot_claim_unproven_execution():
    result = GovernedRuntime(
        StubGateway(
            ExecutionStatus.EXECUTED,
            call_handler=False,
        )
    ).execute(action(), lambda _arguments: "not called")

    assert result.status is ExecutionStatus.CONTROL_FAILURE
    assert result.tool_executed is False


def test_gateway_result_must_match_request():
    result = GovernedRuntime(
        StubGateway(
            ExecutionStatus.EXECUTED,
            call_handler=True,
            request_id="different-request",
        )
    ).execute(action(), lambda _arguments: "sent")

    assert result.status is ExecutionStatus.INDETERMINATE
    assert result.tool_executed is True
    assert result.retryable is False
    assert result.reason == "request_id_mismatch"


def test_action_is_immutable_json_snapshot():
    arguments = {"nested": {"amount": 100}}
    intent = action(arguments)
    arguments["nested"]["amount"] = 999

    assert intent.arguments == {"nested": {"amount": 100}}


def test_action_rejects_non_finite_json():
    with pytest.raises(ContractError, match="finite JSON"):
        action({"amount": float("nan")})


def test_action_rejects_non_string_object_keys():
    with pytest.raises(ContractError, match="keys must be strings"):
        action({1: "not allowed"})


def test_privatevault_payload_matches_real_decide_contract():
    payload = action().privatevault_decide_payload()

    assert set(payload) == {
        "agent_id",
        "capability",
        "timestamp",
        "arguments",
        "context",
        "evidence",
        "request_id",
    }
    assert payload["context"]["cbrain"]["framework"] == "langchain"
    assert payload["context"]["cbrain"]["tool_name"] == "transfer_funds"
