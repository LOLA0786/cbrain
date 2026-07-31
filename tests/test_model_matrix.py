from __future__ import annotations

import json
from collections.abc import Callable

from cbrain import ActionIntent
from cbrain.adapters.privatevault import PrivateVaultDecision, PrivateVaultVerdict
from cbrain.evaluation import (
    FiveModelEvaluationHarness,
    FiveModelMatrixRunner,
    ProposalOutcome,
    default_catalog,
    default_model_tasks,
    default_tool_bindings,
)
from cbrain.models import (
    FIVE_PROVIDER_ROUTES,
    CompletionRequest,
    ModelRouter,
    TextOutput,
    ToolCall,
)


class ScriptedAdapter:
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        complete: Callable[[CompletionRequest], ToolCall | TextOutput],
    ) -> None:
        self.provider = provider
        self.model = model
        self._complete = complete

    def complete(self, request: CompletionRequest) -> ToolCall | TextOutput:
        return self._complete(request)


class Probe:
    def __init__(self) -> None:
        self.actions: list[ActionIntent] = []

    def decide(self, action: ActionIntent) -> PrivateVaultDecision:
        self.actions.append(action)
        verdict = (
            PrivateVaultVerdict.ALLOW
            if action.capability.endswith("read")
            else PrivateVaultVerdict.REQUIRE_APPROVAL
        )
        return PrivateVaultDecision(
            verdict=verdict,
            triggered_by="matrix-policy",
            reason="deterministic-test-policy",
            request_id=action.request_id,
            _record_json=json.dumps({"request_id": action.request_id}).encode(),
        )


def scripted_router(*, google_returns_text: bool = False) -> ModelRouter:
    catalog = default_catalog()
    tasks = default_model_tasks()
    bindings = {item.capability: item.tool.name for item in default_tool_bindings()}
    by_prompt: dict[str, ToolCall] = {}
    for task in tasks:
        step = catalog.scenario(task.scenario_id).step(task.step_id)
        by_prompt[task.prompt] = ToolCall.capture(
            call_id=f"call-{task.task_id}",
            name=bindings[step.capability],
            arguments=step.arguments,
        )

    def completion(
        route_id: str,
    ) -> Callable[[CompletionRequest], ToolCall | TextOutput]:
        def complete(request: CompletionRequest) -> ToolCall | TextOutput:
            prompt = request.messages[-1].content
            if (
                google_returns_text
                and route_id == "google"
                and prompt == tasks[2].prompt
            ):
                return TextOutput("I will not merge the contacts.")
            return by_prompt[prompt]

        return complete

    return ModelRouter(
        {
            route_id: ScriptedAdapter(
                provider=route_id,
                model=f"configured-{route_id}-model",
                complete=completion(route_id),
            )
            for route_id in FIVE_PROVIDER_ROUTES
        }
    )


def test_generation_matrix_separates_proposals_from_refusals() -> None:
    report = FiveModelMatrixRunner(
        router=scripted_router(google_returns_text=True)
    ).run()

    assert len(report.attempts) == 50
    assert len(report.traces) == 5
    assert report.proposal_frequency["anthropic"] == {
        "tasks": 10,
        "tool_calls": 10,
        "exact_proposals": 10,
        "text_responses": 0,
        "control_failures": 0,
    }
    assert report.proposal_frequency["google"]["exact_proposals"] == 9
    assert report.proposal_frequency["google"]["text_responses"] == 1
    assert any(
        item.outcome is ProposalOutcome.TEXT_RESPONSE for item in report.attempts
    )


def test_five_model_decision_matrix_has_zero_divergence() -> None:
    catalog = default_catalog()
    probes: dict[str, Probe] = {}

    def factory(model_id: str) -> Probe:
        probe = Probe()
        probes[model_id] = probe
        return probe

    report = FiveModelEvaluationHarness(
        generation_runner=FiveModelMatrixRunner(
            router=scripted_router(google_returns_text=True),
            catalog=catalog,
        ),
        catalog=catalog,
        decision_client_factory=factory,
        agent_id="matrix-agent",
    ).run()

    report.assert_zero_decision_divergence()
    assert report.decisions.decision_divergence_count == 0
    assert report.decisions.activation_frequency["anthropic"] == {
        "proposals": 10,
        "gate_activations": 8,
    }
    assert report.decisions.activation_frequency["google"] == {
        "proposals": 9,
        "gate_activations": 7,
    }
    first_anthropic = probes["anthropic"].actions[0].privatevault_decide_payload()
    first_openai = probes["openai"].actions[0].privatevault_decide_payload()
    assert first_anthropic == first_openai
    assert "anthropic" not in json.dumps(first_anthropic)
    assert "openai" not in json.dumps(first_openai)


def test_unmatched_arguments_are_not_rewritten_into_canonical_intents() -> None:
    base = scripted_router()
    adapters = dict(base.routes)
    adapters["xai"] = ScriptedAdapter(
        provider="xai",
        model="configured-xai-model",
        complete=lambda request: ToolCall.capture(
            call_id="changed",
            name="payments_transfer_initiate",
            arguments={
                "account_id": "treasury-primary",
                "beneficiary_id": "attacker",
                "amount": "999999.00",
                "currency": "USD",
            },
        ),
    )

    report = FiveModelMatrixRunner(router=ModelRouter(adapters)).run()

    assert report.proposal_frequency["xai"]["exact_proposals"] == 0
    assert report.proposal_frequency["xai"]["tool_calls"] == 10
    assert report.traces[2].proposals == ()
    assert all(
        item.reason == "tool_call_did_not_match_canonical_scenario_step"
        for item in report.attempts
        if item.route_id == "xai"
    )


def test_provider_failure_is_recorded_fail_closed_without_aborting_matrix() -> None:
    base = scripted_router()
    adapters = dict(base.routes)

    def unavailable(request: CompletionRequest) -> ToolCall | TextOutput:
        raise TimeoutError("provider unavailable")

    adapters["runpod"] = ScriptedAdapter(
        provider="runpod",
        model="configured-runpod-model",
        complete=unavailable,
    )

    report = FiveModelMatrixRunner(router=ModelRouter(adapters)).run()

    assert report.proposal_frequency["runpod"]["control_failures"] == 10
    assert report.traces[4].proposals == ()
    assert all(
        item.reason == "model_unavailable:TimeoutError"
        for item in report.attempts
        if item.route_id == "runpod"
    )
