"""Typed offline scenarios for company-agent evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from cbrain.agent import RunStatus
from cbrain.company.kinds import MATRIX_AGENT_KINDS, CompanyAgentKind
from cbrain.company.risk import ToolRiskLevel
from cbrain.company.simulators import load_fixture_bundle
from cbrain.models import TextOutput, ToolCall


class CompanyScenarioCategory(StrEnum):
    NORMAL = "normal"
    EDGE = "edge"
    ADVERSARIAL = "adversarial"
    AUTHORIZATION = "authorization"
    CRASH_CONCURRENCY_COST = "crash_concurrency_cost"


class ExpectedDecision(StrEnum):
    ALLOW = "allow"
    REVIEW = "review"
    BLOCK = "block"


_CATEGORY_COUNTS: dict[CompanyScenarioCategory, int] = {
    CompanyScenarioCategory.NORMAL: 20,
    CompanyScenarioCategory.EDGE: 10,
    CompanyScenarioCategory.ADVERSARIAL: 10,
    CompanyScenarioCategory.AUTHORIZATION: 5,
    CompanyScenarioCategory.CRASH_CONCURRENCY_COST: 5,
}

SCENARIOS_PER_AGENT = sum(_CATEGORY_COUNTS.values())


@dataclass(frozen=True, slots=True)
class CompanyEvalCase:
    case_id: str
    agent_kind: CompanyAgentKind
    category: CompanyScenarioCategory
    title: str
    task: str
    fixture_id: str
    model_outputs: tuple[Any, ...]
    expected_tool: str | None
    expected_arguments: Mapping[str, Any] | None
    expected_decision: ExpectedDecision
    expected_status: RunStatus
    expect_state_mutation: bool
    expected_citations: tuple[str, ...]
    expected_duplicate_dispatch: int
    expect_safety_violation: bool
    requires_durable_store: bool
    concurrent_resume: bool = False
    model_route: str = "offline"
    expect_citation_grounding: bool = False
    authorized_matters: frozenset[str] | None = None
    expect_action_intent: bool = True

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id must be non-empty")
        if not self.title.strip() or not self.task.strip():
            raise ValueError("title and task must be non-empty")
        if not self.fixture_id.strip():
            raise ValueError("fixture_id must be non-empty")
        if not self.model_outputs:
            raise ValueError("model_outputs must be non-empty")
        if self.expected_arguments is not None and not isinstance(
            self.expected_arguments, Mapping
        ):
            raise ValueError("expected_arguments must be a mapping")
        if self.expected_arguments is not None:
            object.__setattr__(
                self,
                "expected_arguments",
                MappingProxyType(dict(self.expected_arguments)),
            )
        load_fixture_bundle(self.fixture_id)
        parts = self.case_id.split("-", 2)
        if len(parts) < 3:
            raise ValueError(f"case_id must be agent-category-index: {self.case_id!r}")


def validate_suite(cases: tuple[CompanyEvalCase, ...]) -> None:
    seen: set[str] = set()
    by_agent: dict[CompanyAgentKind, dict[CompanyScenarioCategory, int]] = {
        kind: dict.fromkeys(CompanyScenarioCategory, 0)
        for kind in MATRIX_AGENT_KINDS
    }
    for case in cases:
        if case.case_id in seen:
            raise ValueError(f"duplicate case_id {case.case_id!r}")
        seen.add(case.case_id)
        by_agent[case.agent_kind][case.category] += 1
    for kind in MATRIX_AGENT_KINDS:
        for category, expected in _CATEGORY_COUNTS.items():
            actual = by_agent[kind][category]
            if actual != expected:
                raise ValueError(
                    f"{kind.value} {category.value}: expected {expected}, got {actual}"
                )


def decision_from_risk(risk: ToolRiskLevel) -> ExpectedDecision:
    if risk is ToolRiskLevel.ALLOW:
        return ExpectedDecision.ALLOW
    if risk is ToolRiskLevel.REVIEW:
        return ExpectedDecision.REVIEW
    return ExpectedDecision.BLOCK


def _tool_output(
    *,
    call_id: str,
    name: str,
    arguments: Mapping[str, Any],
    finish: bool = True,
) -> tuple[Any, ...]:
    outputs: list[Any] = [
        ToolCall.capture(call_id=call_id, name=name, arguments=dict(arguments))
    ]
    if finish:
        outputs.append(TextOutput(text="done"))
    return tuple(outputs)


def expected_status_for_decision(decision: ExpectedDecision) -> RunStatus:
    if decision is ExpectedDecision.ALLOW:
        return RunStatus.COMPLETED
    return RunStatus.REJECTED


__all__ = [
    "CompanyEvalCase",
    "CompanyScenarioCategory",
    "ExpectedDecision",
    "SCENARIOS_PER_AGENT",
    "decision_from_risk",
    "expected_status_for_decision",
    "validate_suite",
]
