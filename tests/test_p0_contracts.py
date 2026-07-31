from __future__ import annotations

import pytest

from cbrain.contracts import (
    ContractError,
    ExecutionStatus,
    GovernedExecution,
)


@pytest.mark.parametrize(
    "tool_executed",
    [False, True],
)
def test_indeterminate_requires_unknown_execution_state(
    tool_executed: bool,
) -> None:
    with pytest.raises(
        ContractError,
        match="INDETERMINATE requires",
    ):
        GovernedExecution(
            status=ExecutionStatus.INDETERMINATE,
            request_id="request-001",
            tool_executed=tool_executed,
            reason="outcome unknown",
            retryable=False,
        )


def test_indeterminate_is_never_retryable() -> None:
    with pytest.raises(
        ContractError,
        match="CONTROL_FAILURE",
    ):
        GovernedExecution(
            status=ExecutionStatus.INDETERMINATE,
            request_id="request-001",
            tool_executed=None,
            reason="outcome unknown",
            retryable=True,
        )


def test_proven_control_failure_may_be_retryable() -> None:
    result = GovernedExecution(
        status=ExecutionStatus.CONTROL_FAILURE,
        request_id="request-001",
        tool_executed=False,
        reason="authority unavailable before dispatch",
        retryable=True,
    )

    assert result.retryable is True
