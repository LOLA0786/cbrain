from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from cbrain import ActionIntent, ExecutionStatus, GovernedExecution
from cbrain.adapters.privatevault import PrivateVaultDecision, PrivateVaultVerdict
from cbrain.evaluation import (
    Consequence,
    DecisionDivergenceError,
    DecisionMatrixHarness,
    ModelTrace,
    ScenarioExecutionHarness,
    StepReference,
    default_catalog,
)


class Probe:
    def __init__(
        self,
        decide: Callable[[ActionIntent], PrivateVaultVerdict],
    ) -> None:
        self._decide = decide
        self.actions: list[ActionIntent] = []

    def decide(self, action: ActionIntent) -> PrivateVaultDecision:
        self.actions.append(action)
        verdict = self._decide(action)
        return PrivateVaultDecision(
            verdict=verdict,
            triggered_by="evaluation-policy",
            reason=f"test-{verdict.value}",
            request_id=action.request_id,
            _record_json=json.dumps(
                {"request_id": action.request_id},
                separators=(",", ":"),
            ).encode(),
        )


def trace(model_id: str, *references: tuple[str, str]) -> ModelTrace:
    return ModelTrace(
        model_id=model_id,
        proposals=tuple(StepReference(*reference) for reference in references),
    )


def test_default_catalog_separates_consequence_from_policy() -> None:
    catalog = default_catalog()

    transfer = catalog.capabilities["payments.transfer.initiate"]
    assert transfer.consequences == frozenset(
        {
            Consequence.MUTATING,
            Consequence.EXTERNAL,
            Consequence.IRREVERSIBLE,
        }
    )
    chain = catalog.scenario("payments.app-fraud-chain")
    assert [step.capability for step in chain.steps] == [
        "payments.limit.modify",
        "payments.beneficiary.add",
        "payments.transfer.initiate",
    ]
    payload = catalog.to_payload()
    assert "risk" not in json.dumps(payload).lower()
    assert "verdict" not in json.dumps(payload).lower()


def test_same_intent_has_zero_divergence_across_models() -> None:
    probes: dict[str, Probe] = {}

    def factory(model_id: str) -> Probe:
        probe = Probe(
            lambda action: (
                PrivateVaultVerdict.ALLOW
                if action.capability.endswith("read")
                else PrivateVaultVerdict.REQUIRE_APPROVAL
            )
        )
        probes[model_id] = probe
        return probe

    harness = DecisionMatrixHarness(
        catalog=default_catalog(),
        decision_client_factory=factory,
        agent_id="evaluation-agent",
    )
    report = harness.evaluate(
        (
            trace(
                "anthropic",
                ("crm.contact-lookup", "read-contact"),
                ("crm.contact-export", "export-contacts"),
            ),
            trace(
                "openai",
                ("crm.contact-lookup", "read-contact"),
                ("crm.contact-export", "export-contacts"),
            ),
            trace(
                "runpod-hermes",
                ("crm.contact-lookup", "read-contact"),
            ),
        )
    )

    report.assert_zero_divergence()
    assert report.decision_divergence_count == 0
    assert report.activation_frequency == {
        "anthropic": {"proposals": 2, "gate_activations": 1},
        "openai": {"proposals": 2, "gate_activations": 1},
        "runpod-hermes": {"proposals": 1, "gate_activations": 0},
    }
    anthropic_payload = probes["anthropic"].actions[0].privatevault_decide_payload()
    openai_payload = probes["openai"].actions[0].privatevault_decide_payload()
    assert anthropic_payload == openai_payload
    assert "anthropic" not in json.dumps(anthropic_payload)


def test_harness_detects_real_decision_divergence() -> None:
    verdicts = {
        "model-a": PrivateVaultVerdict.ALLOW,
        "model-b": PrivateVaultVerdict.BLOCK,
    }
    harness = DecisionMatrixHarness(
        catalog=default_catalog(),
        decision_client_factory=lambda model_id: Probe(
            lambda action: verdicts[model_id]
        ),
        agent_id="evaluation-agent",
    )

    report = harness.evaluate(
        (
            trace("model-a", ("crm.contact-lookup", "read-contact")),
            trace("model-b", ("crm.contact-lookup", "read-contact")),
        )
    )

    assert report.decision_divergence_count == 1
    assert report.divergences[0].dispositions == ("allow", "block")
    with pytest.raises(DecisionDivergenceError, match="1 identical intent"):
        report.assert_zero_divergence()


def test_decision_failure_is_fail_closed_in_report() -> None:
    class BrokenProbe:
        def decide(self, action: ActionIntent) -> PrivateVaultDecision:
            raise TimeoutError("control plane unavailable")

    harness = DecisionMatrixHarness(
        catalog=default_catalog(),
        decision_client_factory=lambda model_id: BrokenProbe(),
        agent_id="evaluation-agent",
    )

    report = harness.evaluate(
        (trace("model-a", ("crm.contact-lookup", "read-contact")),)
    )

    observation = report.observations[0]
    assert observation.disposition == "control_failure"
    assert observation.gate_activated is True
    assert observation.reason == "decision_unavailable:TimeoutError"


def test_execution_harness_replays_exact_same_intent() -> None:
    class Executor:
        def __init__(self) -> None:
            self.actions: list[ActionIntent] = []

        def execute(self, action: ActionIntent) -> GovernedExecution:
            self.actions.append(action)
            return GovernedExecution(
                status=ExecutionStatus.BLOCKED,
                request_id=action.request_id,
                tool_executed=False,
                reason="privatevault_block:replay",
            )

    executor = Executor()
    harness = ScenarioExecutionHarness(
        catalog=default_catalog(),
        executor=executor,
        agent_id="evaluation-agent",
    )

    report = harness.run(("payments.transfer-replay",))

    assert len(report.observations) == 2
    assert [item.attempt for item in report.observations] == [1, 2]
    assert executor.actions[0] == executor.actions[1]
    assert all(item.tool_executed is False for item in report.observations)
