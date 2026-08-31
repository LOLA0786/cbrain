"""Typed company-agent specification composing foundation-agent primitives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from cbrain.agent import AgentProfile, ToolRegistry

from .kinds import CompanyAgentKind
from .risk import ToolRiskLevel, risk_for_tool
from .tools import company_tool_registry, tools_for_kind

COMPANY_AGENT_VERSION = "0.6.0"
SCENARIO_SUITE_VERSION = "company-offline-v0.4.1"


@dataclass(frozen=True, slots=True)
class EvaluationThresholds:
    tool_selection_accuracy: float = 0.95
    tool_argument_accuracy: float = 0.95
    source_grounding_accuracy: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "tool_selection_accuracy",
            "tool_argument_accuracy",
            "source_grounding_accuracy",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or value < 0 or value > 1:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class CompanyAgentSpec:
    kind: CompanyAgentKind
    version: str
    profile: AgentProfile
    tools: ToolRegistry
    tool_risks: Mapping[str, ToolRiskLevel]
    required_simulator: str
    evaluation_thresholds: EvaluationThresholds
    review_tools: frozenset[str]
    block_tools: frozenset[str]

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("version must be non-empty")
        if not self.required_simulator.strip():
            raise ValueError("required_simulator must be non-empty")
        permitted = self.profile.permitted_tools
        unknown = set(self.tool_risks) - permitted
        if unknown:
            raise ValueError(
                f"tool_risks contains unpermitted tools: {sorted(unknown)}"
            )
        missing = permitted - set(self.tool_risks)
        if missing:
            raise ValueError(f"missing tool risk for: {sorted(missing)}")
        if not isinstance(self.evaluation_thresholds, EvaluationThresholds):
            raise ValueError("evaluation_thresholds must be EvaluationThresholds")
        object.__setattr__(
            self,
            "tool_risks",
            MappingProxyType(dict(self.tool_risks)),
        )

    @property
    def agent_id(self) -> str:
        return self.profile.agent_id

    def decision_for_tool(self, tool_name: str) -> ToolRiskLevel:
        try:
            return self.tool_risks[tool_name]
        except KeyError as exc:
            raise ValueError(f"unknown tool {tool_name!r}") from exc


def build_company_spec(
    *,
    kind: CompanyAgentKind,
    profile: AgentProfile,
    required_simulator: str,
    evaluation_thresholds: EvaluationThresholds | None = None,
) -> CompanyAgentSpec:
    permitted = profile.permitted_tools
    registry = company_tool_registry()
    registry.validate_permitted(permitted)
    risks = {name: risk_for_tool(kind, name) for name in permitted}
    review_tools = frozenset(
        name for name, risk in risks.items() if risk is ToolRiskLevel.REVIEW
    )
    block_tools = frozenset(
        name for name, risk in risks.items() if risk is ToolRiskLevel.BLOCK
    )
    return CompanyAgentSpec(
        kind=kind,
        version=COMPANY_AGENT_VERSION,
        profile=profile,
        tools=ToolRegistry(tools_for_kind(kind)),
        tool_risks=risks,
        required_simulator=required_simulator,
        evaluation_thresholds=evaluation_thresholds or EvaluationThresholds(),
        review_tools=review_tools,
        block_tools=block_tools,
    )


__all__ = [
    "COMPANY_AGENT_VERSION",
    "SCENARIO_SUITE_VERSION",
    "CompanyAgentKind",
    "CompanyAgentSpec",
    "EvaluationThresholds",
    "build_company_spec",
]
