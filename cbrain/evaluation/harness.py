"""Evaluation harness for model traces and governed scenario execution."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from cbrain.adapters.privatevault import (
    PrivateVaultDecision,
    PrivateVaultVerdict,
)
from cbrain.contracts import ActionIntent, GovernedExecution

from .catalog import ScenarioCatalog

REPORT_SCHEMA = "cbrain-evaluation-report/v1"


class EvaluationError(RuntimeError):
    """The evaluation input or control path was invalid."""


class DecisionDivergenceError(EvaluationError):
    """Identical decision inputs produced different dispositions."""


class DecisionProbe(Protocol):
    def decide(self, action: ActionIntent) -> PrivateVaultDecision:
        """Return the authoritative PrivateVault decision for one probe."""


class ScenarioExecutor(Protocol):
    def execute(self, action: ActionIntent) -> GovernedExecution:
        """Execute one scenario step through the governed runtime."""


@dataclass(frozen=True, slots=True)
class StepReference:
    scenario_id: str
    step_id: str


@dataclass(frozen=True, slots=True)
class ModelTrace:
    """Tool proposals captured from one model run."""

    model_id: str
    proposals: tuple[StepReference, ...]

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if not self.proposals:
            raise ValueError("model trace must contain at least one proposal")


@dataclass(frozen=True, slots=True)
class DecisionObservation:
    model_id: str
    scenario_id: str
    step_id: str
    intent_digest: str
    disposition: str
    gate_activated: bool
    triggered_by: str
    reason: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "scenario_id": self.scenario_id,
            "step_id": self.step_id,
            "intent_digest": self.intent_digest,
            "disposition": self.disposition,
            "gate_activated": self.gate_activated,
            "triggered_by": self.triggered_by,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class DecisionDivergence:
    intent_digest: str
    dispositions: tuple[str, ...]
    models: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "intent_digest": self.intent_digest,
            "dispositions": list(self.dispositions),
            "models": list(self.models),
        }


@dataclass(frozen=True, slots=True)
class DecisionMatrixReport:
    observations: tuple[DecisionObservation, ...]
    divergences: tuple[DecisionDivergence, ...]

    @property
    def decision_divergence_count(self) -> int:
        return len(self.divergences)

    @property
    def activation_frequency(self) -> dict[str, dict[str, int]]:
        summaries: dict[str, dict[str, int]] = {}
        for observation in self.observations:
            summary = summaries.setdefault(
                observation.model_id,
                {"proposals": 0, "gate_activations": 0},
            )
            summary["proposals"] += 1
            if observation.gate_activated:
                summary["gate_activations"] += 1
        return summaries

    def assert_zero_divergence(self) -> None:
        if self.divergences:
            raise DecisionDivergenceError(
                f"{len(self.divergences)} identical intent(s) diverged"
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": REPORT_SCHEMA,
            "decision_divergence_count": self.decision_divergence_count,
            "activation_frequency": self.activation_frequency,
            "observations": [item.to_payload() for item in self.observations],
            "divergences": [item.to_payload() for item in self.divergences],
        }


class DecisionMatrixHarness:
    """Compare PrivateVault decisions across captured model proposals.

    The factory should provide an isolated decision client for each model run.
    The model identifier is never placed inside the ActionIntent, so identical
    proposals remain byte-for-byte identical decision inputs.
    """

    def __init__(
        self,
        *,
        catalog: ScenarioCatalog,
        decision_client_factory: Callable[[str], DecisionProbe],
        agent_id: str,
        framework: str = "cbrain-evaluation",
    ) -> None:
        if not agent_id.strip() or not framework.strip():
            raise ValueError("agent_id and framework must be non-empty")
        self._catalog = catalog
        self._decision_client_factory = decision_client_factory
        self._agent_id = agent_id
        self._framework = framework

    def evaluate(self, traces: Iterable[ModelTrace]) -> DecisionMatrixReport:
        trace_list = tuple(traces)
        if not trace_list:
            raise EvaluationError("at least one model trace is required")
        model_ids = [trace.model_id for trace in trace_list]
        if len(model_ids) != len(set(model_ids)):
            raise EvaluationError("model trace IDs must be unique")
        observations: list[DecisionObservation] = []
        for trace in trace_list:
            client = self._decision_client_factory(trace.model_id)
            prior_by_scenario: dict[str, list[str]] = defaultdict(list)
            for reference in trace.proposals:
                scenario = self._catalog.scenario(reference.scenario_id)
                step = scenario.step(reference.step_id)
                action = scenario.capture_action(
                    step,
                    agent_id=self._agent_id,
                    framework=self._framework,
                    prior_capabilities=tuple(prior_by_scenario[scenario.scenario_id]),
                )
                observation = self._observe(
                    client=client,
                    trace=trace,
                    reference=reference,
                    action=action,
                )
                observations.append(observation)
                prior_by_scenario[scenario.scenario_id].append(step.capability)
        return DecisionMatrixReport(
            observations=tuple(observations),
            divergences=_find_divergences(observations),
        )

    @staticmethod
    def _observe(
        *,
        client: DecisionProbe,
        trace: ModelTrace,
        reference: StepReference,
        action: ActionIntent,
    ) -> DecisionObservation:
        digest = _intent_digest(action)
        try:
            decision = client.decide(action)
            if not isinstance(decision, PrivateVaultDecision):
                raise EvaluationError("decision client returned an invalid result")
            disposition = decision.verdict.value
            triggered_by = decision.triggered_by
            reason = decision.reason
        except Exception as exc:
            disposition = "control_failure"
            triggered_by = "decision-client"
            reason = f"decision_unavailable:{type(exc).__name__}"
        return DecisionObservation(
            model_id=trace.model_id,
            scenario_id=reference.scenario_id,
            step_id=reference.step_id,
            intent_digest=digest,
            disposition=disposition,
            gate_activated=(disposition != PrivateVaultVerdict.ALLOW.value),
            triggered_by=triggered_by,
            reason=reason,
        )


@dataclass(frozen=True, slots=True)
class ExecutionObservation:
    scenario_id: str
    step_id: str
    attempt: int
    capability: str
    status: str
    tool_executed: bool | None
    retryable: bool
    reason: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "step_id": self.step_id,
            "attempt": self.attempt,
            "capability": self.capability,
            "status": self.status,
            "tool_executed": self.tool_executed,
            "retryable": self.retryable,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ScenarioExecutionReport:
    observations: tuple[ExecutionObservation, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "cbrain-scenario-execution-report/v1",
            "observations": [item.to_payload() for item in self.observations],
        }


class ScenarioExecutionHarness:
    """Drive scenarios through a real governed executor in declared order."""

    def __init__(
        self,
        *,
        catalog: ScenarioCatalog,
        executor: ScenarioExecutor,
        agent_id: str,
        framework: str = "cbrain-evaluation",
    ) -> None:
        if not agent_id.strip() or not framework.strip():
            raise ValueError("agent_id and framework must be non-empty")
        self._catalog = catalog
        self._executor = executor
        self._agent_id = agent_id
        self._framework = framework

    def run(self, scenario_ids: Iterable[str]) -> ScenarioExecutionReport:
        selected = tuple(scenario_ids)
        if not selected:
            raise EvaluationError("at least one scenario is required")
        observations: list[ExecutionObservation] = []
        for scenario_id in selected:
            scenario = self._catalog.scenario(scenario_id)
            prior: list[str] = []
            for step in scenario.steps:
                action = scenario.capture_action(
                    step,
                    agent_id=self._agent_id,
                    framework=self._framework,
                    prior_capabilities=tuple(prior),
                )
                for attempt in range(1, step.repeat + 1):
                    result = self._executor.execute(action)
                    if not isinstance(result, GovernedExecution):
                        raise EvaluationError(
                            "scenario executor returned an invalid result"
                        )
                    observations.append(
                        ExecutionObservation(
                            scenario_id=scenario.scenario_id,
                            step_id=step.step_id,
                            attempt=attempt,
                            capability=step.capability,
                            status=result.status.value,
                            tool_executed=result.tool_executed,
                            retryable=result.retryable,
                            reason=result.reason,
                        )
                    )
                prior.append(step.capability)
        return ScenarioExecutionReport(tuple(observations))


def _intent_digest(action: ActionIntent) -> str:
    encoded = json.dumps(
        action.privatevault_decide_payload(),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _find_divergences(
    observations: Iterable[DecisionObservation],
) -> tuple[DecisionDivergence, ...]:
    grouped: dict[str, list[DecisionObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.intent_digest].append(observation)
    divergences: list[DecisionDivergence] = []
    for digest, items in sorted(grouped.items()):
        dispositions = sorted({item.disposition for item in items})
        if len(dispositions) > 1:
            divergences.append(
                DecisionDivergence(
                    intent_digest=digest,
                    dispositions=tuple(dispositions),
                    models=tuple(sorted({item.model_id for item in items})),
                )
            )
    return tuple(divergences)


__all__ = [
    "DecisionDivergence",
    "DecisionDivergenceError",
    "DecisionMatrixHarness",
    "DecisionMatrixReport",
    "DecisionObservation",
    "DecisionProbe",
    "EvaluationError",
    "ExecutionObservation",
    "ModelTrace",
    "REPORT_SCHEMA",
    "ScenarioExecutionHarness",
    "ScenarioExecutionReport",
    "ScenarioExecutor",
    "StepReference",
]
