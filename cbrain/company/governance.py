"""Risk-based governed gateway for company agents."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from cbrain import ActionIntent, ExecutionStatus, GovernedExecution

from .risk import ToolRiskLevel
from .spec import CompanyAgentSpec


class CompanyRiskGateway:
    """Maps tool risk classification to ALLOW, REVIEW, or BLOCK decisions."""

    independent_execution = False

    def __init__(self, spec: CompanyAgentSpec) -> None:
        self._spec = spec
        self.actions: list[ActionIntent] = []
        self.handler_calls: list[str] = []

    def decide_and_execute(
        self,
        action: ActionIntent,
        handler: Callable[[Mapping[str, Any]], Any],
    ) -> GovernedExecution:
        self.actions.append(action)
        tool_name = action.tool_name
        risk = self._spec.decision_for_tool(tool_name)
        if risk is ToolRiskLevel.BLOCK:
            return GovernedExecution(
                status=ExecutionStatus.BLOCKED,
                request_id=action.request_id,
                tool_executed=False,
                reason="blocked_by_policy",
            )
        if risk is ToolRiskLevel.REVIEW:
            return GovernedExecution(
                status=ExecutionStatus.REVIEW_REQUIRED,
                request_id=action.request_id,
                tool_executed=False,
                reason="review_required",
            )
        self.handler_calls.append(tool_name)
        return GovernedExecution(
            status=ExecutionStatus.EXECUTED,
            request_id=action.request_id,
            tool_executed=True,
            reason="allow",
            output=handler(action.arguments),
        )


__all__ = ["CompanyRiskGateway"]
