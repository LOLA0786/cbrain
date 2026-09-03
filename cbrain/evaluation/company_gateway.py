"""Company eval/fixture policy gateway.

This is the `company_test_gateway` decision authority. It is not PrivateVault
and must not be imported by production adapters, execution, or the agent loop.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from cbrain import ActionIntent, ExecutionStatus, GovernedExecution
from cbrain.company.authority import CompanyExecutionContext
from cbrain.company.risk import ToolRiskLevel
from cbrain.company.simulators import CompanySimulatorBundle
from cbrain.company.spec import CompanyAgentSpec
from cbrain.company.validation import validate_company_action


class CompanyRiskGateway:
    """Maps tool risk and argument validation to ALLOW, REVIEW, or BLOCK."""

    independent_execution = False
    decision_authority = "company_test_gateway"

    def __init__(
        self,
        spec: CompanyAgentSpec,
        *,
        context: CompanyExecutionContext | None = None,
        bundle: CompanySimulatorBundle | None = None,
    ) -> None:
        self._spec = spec
        self._context = context
        self._bundle = bundle
        self.actions: list[ActionIntent] = []
        self.handler_calls: list[str] = []
        self.last_status: ExecutionStatus | None = None

    def decide_and_execute(
        self,
        action: ActionIntent,
        handler: Callable[[Mapping[str, Any]], Any],
    ) -> GovernedExecution:
        self.actions.append(action)
        validation = validate_company_action(
            kind=self._spec.kind,
            action=action,
            context=self._context,
            bundle=self._bundle,
        )
        if not validation.ok:
            execution = GovernedExecution(
                status=ExecutionStatus.BLOCKED,
                request_id=action.request_id,
                tool_executed=False,
                reason=validation.reason or "blocked_by_validation",
            )
            self.last_status = execution.status
            return execution
        tool_name = action.tool_name
        risk = self._spec.decision_for_tool(tool_name)
        if risk is ToolRiskLevel.BLOCK:
            execution = GovernedExecution(
                status=ExecutionStatus.BLOCKED,
                request_id=action.request_id,
                tool_executed=False,
                reason="blocked_by_policy",
            )
            self.last_status = execution.status
            return execution
        if risk is ToolRiskLevel.REVIEW:
            execution = GovernedExecution(
                status=ExecutionStatus.REVIEW_REQUIRED,
                request_id=action.request_id,
                tool_executed=False,
                reason="review_required",
            )
            self.last_status = execution.status
            return execution
        self.handler_calls.append(tool_name)
        execution = GovernedExecution(
            status=ExecutionStatus.EXECUTED,
            request_id=action.request_id,
            tool_executed=True,
            reason="allow",
            output=handler(action.arguments),
        )
        self.last_status = execution.status
        return execution


__all__ = ["CompanyRiskGateway"]
