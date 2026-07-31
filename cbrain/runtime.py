from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .contracts import ActionIntent, ExecutionStatus, GovernedExecution
from .ports import PrivateVaultGateway, ToolHandler


class GovernedRuntime:
    """Framework-neutral governed execution entry point."""

    def __init__(self, gateway: PrivateVaultGateway) -> None:
        self._gateway = gateway

    def execute(
        self,
        action: ActionIntent,
        handler: ToolHandler,
    ) -> GovernedExecution:
        handler_entries = 0

        def invoke_once(arguments: Mapping[str, Any]) -> Any:
            nonlocal handler_entries
            handler_entries += 1

            if handler_entries > 1:
                raise RuntimeError("tool handler invoked more than once")

            return handler(arguments)

        try:
            result = self._gateway.decide_and_execute(
                action,
                invoke_once,
            )
        except Exception as exc:
            execution_possible = handler_entries > 0

            return GovernedExecution(
                status=(
                    ExecutionStatus.INDETERMINATE
                    if execution_possible
                    else ExecutionStatus.CONTROL_FAILURE
                ),
                request_id=action.request_id,
                tool_executed=execution_possible,
                reason=(
                    "privatevault_gateway_error:"
                    f"{type(exc).__name__}"
                ),
                retryable=False,
            )

        if not isinstance(result, GovernedExecution):
            return self._contract_failure(
                action,
                handler_entries,
                "invalid_gateway_result",
            )

        if result.request_id != action.request_id:
            return self._contract_failure(
                action,
                handler_entries,
                "request_id_mismatch",
            )

        if (
            result.status is ExecutionStatus.EXECUTED
            and handler_entries != 1
        ):
            return self._contract_failure(
                action,
                handler_entries,
                "unproven_execution",
            )

        if (
            result.status is not ExecutionStatus.EXECUTED
            and handler_entries != 0
        ):
            return self._contract_failure(
                action,
                handler_entries,
                "tool_called_without_executed_status",
            )

        return result

    @staticmethod
    def _contract_failure(
        action: ActionIntent,
        handler_entries: int,
        reason: str,
    ) -> GovernedExecution:
        execution_possible = handler_entries > 0

        return GovernedExecution(
            status=(
                ExecutionStatus.INDETERMINATE
                if execution_possible
                else ExecutionStatus.CONTROL_FAILURE
            ),
            request_id=action.request_id,
            tool_executed=execution_possible,
            reason=reason,
            retryable=False,
        )
