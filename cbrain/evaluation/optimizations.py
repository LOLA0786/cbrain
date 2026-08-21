"""Configurable cost optimizations for agent evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from cbrain.agent import RunLimits

from .agent_suites import AgentEvalCase, AgentEvalCategory


@dataclass(frozen=True, slots=True)
class AgentEvalOptimizationConfig:
    name: str
    simple_task_route: str | None = None
    max_output_tokens: int | None = None
    cached_system_prompt: bool = False
    dedupe_context: bool = False
    max_observation_chars: int | None = None
    early_termination: bool = False
    budget_usd: Decimal | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("optimization config name must be non-empty")
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if self.max_observation_chars is not None and self.max_observation_chars < 1:
            raise ValueError("max_observation_chars must be positive")
        if self.budget_usd is not None and self.budget_usd <= 0:
            raise ValueError("budget_usd must be positive")


BASELINE_CONFIG = AgentEvalOptimizationConfig(name="baseline")

OPTIMIZED_CONFIG = AgentEvalOptimizationConfig(
    name="optimized",
    simple_task_route="cheap",
    max_output_tokens=256,
    cached_system_prompt=True,
    dedupe_context=True,
    max_observation_chars=512,
    early_termination=True,
)


def select_model_route(
    case: AgentEvalCase,
    *,
    default_route: str,
    config: AgentEvalOptimizationConfig,
) -> str:
    if (
        config.simple_task_route is not None
        and case.simple_task
        and case.category is AgentEvalCategory.TEXT_ONLY
    ):
        return config.simple_task_route
    return default_route


def apply_run_limits(
    limits: RunLimits,
    *,
    config: AgentEvalOptimizationConfig,
) -> RunLimits:
    observation = (
        config.max_observation_chars
        if config.max_observation_chars is not None
        else limits.max_observation_chars
    )
    return RunLimits(
        max_context_messages=limits.max_context_messages,
        max_model_text_chars=limits.max_model_text_chars,
        max_observation_chars=observation,
        max_output_tokens=(
            config.max_output_tokens
            if config.max_output_tokens is not None
            else limits.max_output_tokens
        ),
        max_task_chars=limits.max_task_chars,
    )


__all__ = [
    "AgentEvalOptimizationConfig",
    "BASELINE_CONFIG",
    "OPTIMIZED_CONFIG",
    "apply_run_limits",
    "select_model_route",
]
