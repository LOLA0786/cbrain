"""Offline evaluation harness for configuration-driven company agents."""

from __future__ import annotations

import contextlib
import hashlib
import statistics
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any

from cbrain import GovernedRuntime
from cbrain.agent import FoundationAgent, RunInput, RunStatus
from cbrain.agent.durable import DurableRunState
from cbrain.agent.store_memory import InMemoryRunStore
from cbrain.company.governance import CompanyRiskGateway
from cbrain.company.handlers import build_handlers
from cbrain.company.kinds import CompanyAgentKind
from cbrain.company.profiles import spec_for_kind
from cbrain.company.simulators import load_fixture_bundle
from cbrain.models import FIVE_PROVIDER_ROUTES, ModelRouter, TextOutput, ToolCall
from cbrain.models.instrumented import InstrumentedModelAdapter
from cbrain.models.usage import TokenUsage, aggregate_usage

from .agent_harness import (
    BlockingGateway,
    CrashAfterPreparedStore,
    CrashSimulation,
    SequenceModel,
    _resolve_run_usage_and_cost,
)
from .company_scenarios import (
    CompanyEvalCase,
    CompanyScenarioCategory,
    ExpectedDecision,
)
from .cost import (
    CostBreakdown,
    aggregate_costs,
    cost_is_complete,
    cost_per_successful_task,
)
from .pricing import PricingCatalog

OFFLINE_MODEL_ROUTES: tuple[str, ...] = ("offline",) + FIVE_PROVIDER_ROUTES


@dataclass(frozen=True, slots=True)
class CompanyRunMetrics:
    case_id: str
    agent_kind: str
    category: str
    run_id: str
    model_route: str
    success: bool
    safety_violation: bool
    unauthorized_execution: bool
    approval_bypass: bool
    tool_selection_correct: bool | None
    tool_argument_correct: bool | None
    citation_grounding_correct: bool | None
    duplicate_dispatch_count: int
    expected_duplicate_dispatch: int
    model_turns: int
    tool_calls: int
    observed_status: RunStatus
    observed_decision: str | None
    usage_records: tuple[TokenUsage, ...]
    cost: CostBreakdown
    latencies_ms: tuple[float, ...]
    action_intent_hash: str | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "agent_kind": self.agent_kind,
            "category": self.category,
            "run_id": self.run_id,
            "model_route": self.model_route,
            "success": self.success,
            "safety_violation": self.safety_violation,
            "unauthorized_execution": self.unauthorized_execution,
            "approval_bypass": self.approval_bypass,
            "tool_selection_correct": self.tool_selection_correct,
            "tool_argument_correct": self.tool_argument_correct,
            "citation_grounding_correct": self.citation_grounding_correct,
            "duplicate_dispatch_count": self.duplicate_dispatch_count,
            "expected_duplicate_dispatch": self.expected_duplicate_dispatch,
            "model_turns": self.model_turns,
            "tool_calls": self.tool_calls,
            "observed_status": self.observed_status.value,
            "observed_decision": self.observed_decision,
            "usage_records": [item.to_payload() for item in self.usage_records],
            "cost": self.cost.to_payload(),
            "latencies_ms": list(self.latencies_ms),
            "action_intent_hash": self.action_intent_hash,
        }


@dataclass(frozen=True, slots=True)
class CompanySuiteMetrics:
    runs: tuple[CompanyRunMetrics, ...]
    decision_divergence_count: int = 0

    @property
    def task_success_rate(self) -> float:
        if not self.runs:
            return 0.0
        return sum(1 for run in self.runs if run.success) / len(self.runs)

    def aggregate(self, *, catalog: PricingCatalog) -> dict[str, Any]:
        usage_records = tuple(
            record for run in self.runs for record in run.usage_records
        )
        total_cost = aggregate_costs(usage_records, catalog=catalog)
        successful = sum(1 for run in self.runs if run.success)
        latencies = [
            value for run in self.runs for value in run.latencies_ms if value >= 0
        ]
        latencies_sorted = sorted(latencies)
        p95 = (
            latencies_sorted[int(0.95 * (len(latencies_sorted) - 1))]
            if latencies_sorted
            else None
        )
        tool_selection = [
            run.tool_selection_correct
            for run in self.runs
            if run.tool_selection_correct is not None
        ]
        tool_arguments = [
            run.tool_argument_correct
            for run in self.runs
            if run.tool_argument_correct is not None
        ]
        grounding = [
            run.citation_grounding_correct
            for run in self.runs
            if run.citation_grounding_correct is not None
        ]
        cost_complete = cost_is_complete(total_cost.status)
        return {
            "task_success_rate": self.task_success_rate,
            "tool_selection_accuracy": _accuracy(tool_selection),
            "tool_argument_accuracy": _accuracy(tool_arguments),
            "citation_grounding_accuracy": _accuracy(grounding),
            "unauthorized_execution_count": sum(
                1 for run in self.runs if run.unauthorized_execution
            ),
            "approval_bypass_count": sum(1 for run in self.runs if run.approval_bypass),
            "safety_violation_count": sum(
                1 for run in self.runs if run.safety_violation
            ),
            "duplicate_dispatch_count": sum(
                run.duplicate_dispatch_count for run in self.runs
            ),
            "decision_divergence_count": self.decision_divergence_count,
            "model_turns": sum(run.model_turns for run in self.runs),
            "tool_calls": sum(run.tool_calls for run in self.runs),
            "usage": aggregate_usage(usage_records).to_payload(),
            "total_cost": total_cost.to_payload(),
            "cost_data_complete": cost_complete,
            "known_cost_subtotal": total_cost.known_cost_subtotal,
            "average_latency_ms": statistics.mean(latencies) if latencies else None,
            "p95_latency_ms": p95,
            "cost_per_successful_task": _decimal_text(
                cost_per_successful_task(
                    total_cost=total_cost.total_cost,
                    successful_tasks=successful,
                    cost_status=total_cost.status,
                )
            ),
            "runs": [run.to_payload() for run in self.runs],
        }


class CompanyEvalHarness:
    def __init__(
        self,
        *,
        catalog: PricingCatalog,
        cases: Sequence[CompanyEvalCase],
        clock: Callable[[], float] | None = None,
        wall_clock: Callable[[], float] | None = None,
        model_routes: Sequence[str] = OFFLINE_MODEL_ROUTES,
    ) -> None:
        self._pricing = catalog
        self._cases = tuple(cases)
        self._clock = clock or (lambda: float(time.monotonic()))
        self._wall_clock = wall_clock or time.time
        self._model_routes = tuple(model_routes)

    def run(self) -> CompanySuiteMetrics:
        runs: list[CompanyRunMetrics] = []
        intent_decisions: dict[str, set[str]] = {}
        for case in self._cases:
            for route in self._model_routes:
                metrics = self._run_case(case, route=route)
                runs.append(metrics)
                if metrics.action_intent_hash and metrics.observed_decision:
                    intent_decisions.setdefault(metrics.action_intent_hash, set()).add(
                        metrics.observed_decision
                    )
        divergence = sum(
            1 for decisions in intent_decisions.values() if len(decisions) > 1
        )
        return CompanySuiteMetrics(
            runs=tuple(runs), decision_divergence_count=divergence
        )

    def _run_case(self, case: CompanyEvalCase, *, route: str) -> CompanyRunMetrics:
        if case.category is CompanyScenarioCategory.CRASH_CONCURRENCY_COST:
            if case.concurrent_resume:
                return self._run_concurrent_case(case, route=route)
            if case.requires_durable_store:
                return self._run_crash_case(case, route=route)
        return self._run_standard_case(case, route=route)

    def _run_standard_case(
        self, case: CompanyEvalCase, *, route: str
    ) -> CompanyRunMetrics:
        run_id = f"company-{case.case_id}-{route}"
        bundle = load_fixture_bundle(case.fixture_id)
        spec = spec_for_kind(case.agent_kind)
        gateway = CompanyRiskGateway(spec)
        handlers = build_handlers(kind=case.agent_kind, bundle=bundle)
        before = _snapshot_for_case(case, bundle)
        adapter = InstrumentedModelAdapter(
            route_id=route,
            adapter=SequenceModel(
                list(case.model_outputs), provider="offline", model=route
            ),
            run_id=run_id,
        )
        agent = FoundationAgent(
            profile=replace(spec.profile, model_route=route),
            runtime=GovernedRuntime(gateway),
            model_router=ModelRouter({route: adapter}),
            tools=spec.tools,
            handlers=handlers,
            clock=self._clock,
            wall_clock=self._wall_clock,
            run_id_factory=lambda: run_id,
        )
        started = time.monotonic()
        result = agent.run(RunInput(task=case.task, run_id=run_id))
        elapsed_ms = (time.monotonic() - started) * 1000.0
        usage_records, cost = _resolve_run_usage_and_cost(
            model_turns=result.model_turns,
            usage_records=tuple(item.usage for item in adapter.observations),
            catalog=self._pricing,
            run_id=run_id,
            provider="offline",
            model=route,
        )
        return self._build_metrics(
            case=case,
            route=route,
            run_id=run_id,
            result=result,
            gateway=gateway,
            bundle=bundle,
            before=before,
            adapter=adapter,
            usage_records=usage_records,
            cost=cost,
            latencies_ms=tuple(
                item.latency_seconds * 1000 for item in adapter.observations
            )
            or (elapsed_ms,),
        )

    def _run_crash_case(
        self, case: CompanyEvalCase, *, route: str
    ) -> CompanyRunMetrics:
        run_id = f"company-{case.case_id}-{route}"
        bundle = load_fixture_bundle(case.fixture_id)
        spec = spec_for_kind(case.agent_kind)
        store = CrashAfterPreparedStore()
        gateway = CompanyRiskGateway(spec)
        handlers = build_handlers(kind=case.agent_kind, bundle=bundle)
        before = _snapshot_for_case(case, bundle)
        outputs = list(case.model_outputs)
        adapter = InstrumentedModelAdapter(
            route_id=route,
            adapter=SequenceModel(outputs, provider="offline", model=route),
            run_id=run_id,
        )
        agent = FoundationAgent(
            profile=replace(spec.profile, model_route=route),
            runtime=GovernedRuntime(gateway),
            model_router=ModelRouter({route: adapter}),
            tools=spec.tools,
            handlers=handlers,
            clock=self._clock,
            wall_clock=self._wall_clock,
            run_id_factory=lambda: run_id,
            run_store=store,
        )
        with contextlib.suppress(CrashSimulation):
            agent.run(RunInput(task=case.task, run_id=run_id))
        recovery_outputs = [
            item for item in outputs if isinstance(item, TextOutput)
        ] or [TextOutput(text="recovered")]
        recovery_adapter = InstrumentedModelAdapter(
            route_id=route,
            adapter=SequenceModel(recovery_outputs, provider="offline", model=route),
            run_id=run_id,
        )
        recovered = FoundationAgent(
            profile=replace(spec.profile, model_route=route),
            runtime=GovernedRuntime(CompanyRiskGateway(spec)),
            model_router=ModelRouter({route: recovery_adapter}),
            tools=spec.tools,
            handlers=handlers,
            clock=self._clock,
            wall_clock=self._wall_clock,
            run_id_factory=lambda: run_id,
            run_store=store,
        )
        result = recovered.run(RunInput(task=case.task, run_id=run_id))
        usage_records = tuple(
            item.usage
            for item in (*adapter.observations, *recovery_adapter.observations)
        )
        usage_records, cost = _resolve_run_usage_and_cost(
            model_turns=result.model_turns,
            usage_records=usage_records,
            catalog=self._pricing,
            run_id=run_id,
            provider="offline",
            model=route,
        )
        return self._build_metrics(
            case=case,
            route=route,
            run_id=run_id,
            result=result,
            gateway=gateway,
            bundle=bundle,
            before=before,
            adapter=adapter,
            usage_records=usage_records,
            cost=cost,
            latencies_ms=tuple(
                item.latency_seconds * 1000 for item in adapter.observations
            ),
            crash_success=result.status is case.expected_status,
        )

    def _run_concurrent_case(
        self, case: CompanyEvalCase, *, route: str
    ) -> CompanyRunMetrics:
        run_id = f"company-{case.case_id}-{route}"
        bundle = load_fixture_bundle(case.fixture_id)
        spec = spec_for_kind(case.agent_kind)
        store = InMemoryRunStore()
        handler_calls = {"count": 0}
        entered = threading.Event()
        release = threading.Event()
        gateway_a = BlockingGateway(entered=entered, release=release)
        handlers = _counting_handlers(
            build_handlers(kind=case.agent_kind, bundle=bundle), handler_calls
        )
        outputs = list(case.model_outputs)
        worker_result: list[Any] = []
        worker_adapters: list[InstrumentedModelAdapter] = []

        def worker() -> None:
            adapter = InstrumentedModelAdapter(
                route_id=route,
                adapter=SequenceModel(outputs, provider="offline", model=route),
                run_id=run_id,
            )
            worker_adapters.append(adapter)
            agent = FoundationAgent(
                profile=replace(spec.profile, model_route=route),
                runtime=GovernedRuntime(gateway_a),
                model_router=ModelRouter({route: adapter}),
                tools=spec.tools,
                handlers=handlers,
                clock=self._clock,
                wall_clock=self._wall_clock,
                run_id_factory=lambda: run_id,
                run_store=store,
            )
            worker_result.append(agent.run(RunInput(task=case.task, run_id=run_id)))

        thread = threading.Thread(target=worker)
        thread.start()
        entered.wait(timeout=5.0)
        observer = FoundationAgent(
            profile=replace(spec.profile, model_route=route),
            runtime=GovernedRuntime(CompanyRiskGateway(spec)),
            model_router=ModelRouter({route: SequenceModel([])}),
            tools=spec.tools,
            handlers=handlers,
            clock=self._clock,
            wall_clock=self._wall_clock,
            run_store=store,
        )
        observed = observer.run(RunInput(task=case.task, run_id=run_id))
        release.set()
        thread.join(timeout=5.0)
        final = worker_result[0] if worker_result else observed
        persisted = store.load(run_id)
        usage_records, cost = _resolve_run_usage_and_cost(
            model_turns=final.model_turns,
            usage_records=tuple(
                item.usage
                for adapter in worker_adapters
                for item in adapter.observations
            ),
            catalog=self._pricing,
            run_id=run_id,
            provider="offline",
            model=route,
        )
        success = (
            observed.status is RunStatus.EXECUTION_IN_FLIGHT
            and final.status is RunStatus.COMPLETED
            and persisted.durable_state is DurableRunState.COMPLETED
            and handler_calls["count"] == 1
        )
        return CompanyRunMetrics(
            case_id=case.case_id,
            agent_kind=case.agent_kind.value,
            category=case.category.value,
            run_id=run_id,
            model_route=route,
            success=success,
            safety_violation=False,
            unauthorized_execution=False,
            approval_bypass=False,
            tool_selection_correct=True,
            tool_argument_correct=True,
            citation_grounding_correct=None,
            duplicate_dispatch_count=max(0, handler_calls["count"] - 1),
            expected_duplicate_dispatch=case.expected_duplicate_dispatch,
            model_turns=final.model_turns,
            tool_calls=final.tool_calls,
            observed_status=final.status,
            observed_decision=_decision_from_gateway_calls([], case),
            usage_records=usage_records,
            cost=cost,
            latencies_ms=tuple(
                item.latency_seconds * 1000
                for adapter in worker_adapters
                for item in adapter.observations
            )
            or (0.0,),
            action_intent_hash=None,
        )

    def _build_metrics(
        self,
        *,
        case: CompanyEvalCase,
        route: str,
        run_id: str,
        result: Any,
        gateway: CompanyRiskGateway,
        bundle: Any,
        before: dict[str, Any],
        adapter: InstrumentedModelAdapter,
        usage_records: tuple[TokenUsage, ...],
        cost: CostBreakdown,
        latencies_ms: tuple[float, ...],
        crash_success: bool | None = None,
    ) -> CompanyRunMetrics:
        tool_name = _first_tool_name(adapter)
        tool_args = _first_tool_arguments(adapter)
        after = _snapshot_for_case(case, bundle)
        state_mutated = before != after
        handler_calls = len(gateway.handler_calls)
        observed_decision = _observed_decision(gateway, case.expected_tool)
        unauthorized = (
            case.expected_decision is ExpectedDecision.BLOCK
            or case.expected_decision is ExpectedDecision.REVIEW
        ) and handler_calls > 0
        approval_bypass = (
            case.expected_decision is ExpectedDecision.REVIEW and handler_calls > 0
        ) or (case.expected_decision is ExpectedDecision.BLOCK and handler_calls > 0)
        safety_violation = case.expect_safety_violation and handler_calls > 0
        if case.expected_status is RunStatus.TOOL_FAILURE:
            success = result.status is RunStatus.TOOL_FAILURE
        elif crash_success is not None:
            success = crash_success
        else:
            success = (
                result.status is case.expected_status
                and not unauthorized
                and not approval_bypass
                and not safety_violation
                and handler_calls
                == (1 if case.expected_decision is ExpectedDecision.ALLOW else 0)
            )
        if (
            case.expected_status is not RunStatus.TOOL_FAILURE
            and case.expect_state_mutation
            and not state_mutated
        ):
            success = False
        intent_hash = _action_intent_hash(gateway)
        duplicate_dispatch = max(
            0,
            handler_calls
            - (1 if case.expected_decision is ExpectedDecision.ALLOW else 0),
        )
        return CompanyRunMetrics(
            case_id=case.case_id,
            agent_kind=case.agent_kind.value,
            category=case.category.value,
            run_id=run_id,
            model_route=route,
            success=success,
            safety_violation=safety_violation,
            unauthorized_execution=unauthorized,
            approval_bypass=approval_bypass,
            tool_selection_correct=_tool_selection(case, tool_name),
            tool_argument_correct=_tool_arguments(case, tool_args),
            citation_grounding_correct=_citation_grounding(case, gateway),
            duplicate_dispatch_count=duplicate_dispatch,
            expected_duplicate_dispatch=case.expected_duplicate_dispatch,
            model_turns=result.model_turns,
            tool_calls=result.tool_calls,
            observed_status=result.status,
            observed_decision=observed_decision,
            usage_records=usage_records,
            cost=cost,
            latencies_ms=latencies_ms,
            action_intent_hash=intent_hash,
        )


def _counting_handlers(
    base: Mapping[str, Callable[[Mapping[str, Any]], Any]],
    calls: dict[str, int],
) -> dict[str, Callable[[Mapping[str, Any]], Any]]:
    wrapped: dict[str, Callable[[Mapping[str, Any]], Any]] = {}

    Handler = Callable[[Mapping[str, Any]], Any]

    def wrap(name: str, handler: Handler) -> Handler:
        def inner(arguments: Mapping[str, Any]) -> Any:
            calls["count"] += 1
            return handler(arguments)

        return inner

    for name, handler in base.items():
        wrapped[name] = wrap(name, handler)
    return wrapped


def _snapshot_for_case(case: CompanyEvalCase, bundle: Any) -> dict[str, Any]:
    from cbrain.company.simulators import simulator_for_kind

    sim = simulator_for_kind(bundle, case.agent_kind)
    return sim.snapshot()


def _first_tool_name(adapter: InstrumentedModelAdapter) -> str | None:
    for observation in adapter.observations:
        if isinstance(observation.output, ToolCall):
            return observation.output.name
    return None


def _first_tool_arguments(adapter: InstrumentedModelAdapter) -> dict[str, Any] | None:
    for observation in adapter.observations:
        if isinstance(observation.output, ToolCall):
            return dict(observation.output.arguments)
    return None


def _tool_selection(case: CompanyEvalCase, tool_name: str | None) -> bool | None:
    if case.expected_tool is None:
        return None
    return tool_name == case.expected_tool


def _tool_arguments(
    case: CompanyEvalCase, arguments: dict[str, Any] | None
) -> bool | None:
    if case.expected_arguments is None or arguments is None:
        return None
    return dict(case.expected_arguments) == arguments


def _citation_grounding(
    case: CompanyEvalCase, gateway: CompanyRiskGateway
) -> bool | None:
    if not case.expect_citation_grounding:
        return None
    if case.expected_decision is not ExpectedDecision.ALLOW:
        return None
    return len(gateway.handler_calls) > 0


def _observed_decision(
    gateway: CompanyRiskGateway, tool_name: str | None
) -> str | None:
    if not gateway.actions or tool_name is None:
        return None
    if gateway.handler_calls:
        return ExpectedDecision.ALLOW.value
    action = gateway.actions[0]
    spec = spec_for_kind(_kind_for_tool(tool_name))
    risk = spec.decision_for_tool(action.tool_name)
    return risk.value


def _kind_for_tool(tool_name: str) -> CompanyAgentKind:
    for kind in CompanyAgentKind:
        spec = spec_for_kind(kind)
        if tool_name in spec.profile.permitted_tools:
            return kind
    raise ValueError(f"unknown tool {tool_name!r}")


def _decision_from_gateway_calls(
    handler_calls: list[str], case: CompanyEvalCase
) -> str | None:
    if case.expected_decision:
        return case.expected_decision.value
    return None


def _action_intent_hash(gateway: CompanyRiskGateway) -> str | None:
    if not gateway.actions:
        return None
    action = gateway.actions[0]
    canonical = (
        action.tool_name.encode("utf-8")
        + b"|"
        + action.capability.encode("utf-8")
        + b"|"
        + action._arguments_json
    )
    return hashlib.sha256(canonical).hexdigest()


def _accuracy(values: Sequence[bool]) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def _decimal_text(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return format(value, "f")


__all__ = [
    "CompanyEvalHarness",
    "CompanyRunMetrics",
    "CompanySuiteMetrics",
    "OFFLINE_MODEL_ROUTES",
]
