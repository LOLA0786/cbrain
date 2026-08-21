"""Agent evaluation harness with quality, safety, latency, and cost metrics."""

from __future__ import annotations

import contextlib
import statistics
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, cast

from cbrain import ActionIntent, ExecutionStatus, GovernedExecution, GovernedRuntime
from cbrain.agent import (
    AgentProfile,
    FoundationAgent,
    GovernedTool,
    RunInput,
    RunLimits,
    RunStatus,
    ToolRegistry,
)
from cbrain.agent.durable import DurableRunState, StoredRunRecord
from cbrain.agent.store_memory import InMemoryRunStore
from cbrain.models import (
    CompletionRequest,
    ModelError,
    ModelRouter,
    TextOutput,
    ToolCall,
)
from cbrain.models.instrumented import InstrumentedModelAdapter
from cbrain.models.usage import TokenUsage, aggregate_usage

from .agent_suites import (
    AgentEvalCase,
    AgentEvalCategory,
    GatewayKind,
    default_agent_eval_cases,
)
from .cost import (
    CostBreakdown,
    aggregate_costs,
    blended_cost_per_million_tokens,
    cost_is_complete,
    cost_per_successful_task,
    reconcile_usage_with_model_turns,
    usage_inconsistent_cost,
)
from .optimizations import (
    AgentEvalOptimizationConfig,
    apply_run_limits,
    select_model_route,
)
from .pricing import PricingCatalog


class AgentEvalError(RuntimeError):
    """Agent evaluation could not be executed safely."""


@dataclass(frozen=True, slots=True)
class AgentRunMetrics:
    case_id: str
    category: str
    run_id: str
    success: bool
    safety_violation: bool
    tool_selection_correct: bool | None
    tool_argument_correct: bool | None
    duplicate_dispatch_count: int
    model_failure_count: int
    tool_failure_count: int
    model_turns: int
    tool_calls: int
    observed_status: RunStatus
    usage_records: tuple[TokenUsage, ...]
    cost: CostBreakdown
    latencies_ms: tuple[float, ...]
    budget_violation: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "run_id": self.run_id,
            "success": self.success,
            "safety_violation": self.safety_violation,
            "tool_selection_correct": self.tool_selection_correct,
            "tool_argument_correct": self.tool_argument_correct,
            "duplicate_dispatch_count": self.duplicate_dispatch_count,
            "model_failure_count": self.model_failure_count,
            "tool_failure_count": self.tool_failure_count,
            "model_turns": self.model_turns,
            "tool_calls": self.tool_calls,
            "observed_status": self.observed_status.value,
            "usage_records": [item.to_payload() for item in self.usage_records],
            "cost": self.cost.to_payload(),
            "latencies_ms": list(self.latencies_ms),
            "budget_violation": self.budget_violation,
        }


@dataclass(frozen=True, slots=True)
class AgentSuiteMetrics:
    config_name: str
    runs: tuple[AgentRunMetrics, ...]

    @property
    def task_success_rate(self) -> float:
        if not self.runs:
            return 0.0
        return sum(1 for run in self.runs if run.success) / len(self.runs)

    @property
    def safety_violation_rate(self) -> float:
        if not self.runs:
            return 0.0
        return sum(1 for run in self.runs if run.safety_violation) / len(self.runs)

    def aggregate(self, *, catalog: PricingCatalog) -> dict[str, Any]:
        usage_records = tuple(
            record for run in self.runs for record in run.usage_records
        )
        usage_totals = aggregate_usage(usage_records)
        total_cost = aggregate_costs(usage_records, catalog=catalog)
        successful = sum(1 for run in self.runs if run.success)
        failed_cost = aggregate_costs(
            tuple(
                record
                for run in self.runs
                if not run.success
                for record in run.usage_records
            ),
            catalog=catalog,
        )
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
        cost_complete = cost_is_complete(total_cost.status)
        return {
            "config_name": self.config_name,
            "task_success_rate": self.task_success_rate,
            "safety_violation_rate": self.safety_violation_rate,
            "tool_selection_accuracy": _accuracy(tool_selection),
            "tool_argument_accuracy": _accuracy(tool_arguments),
            "duplicate_dispatch_count": sum(
                run.duplicate_dispatch_count for run in self.runs
            ),
            "model_failure_count": sum(run.model_failure_count for run in self.runs),
            "tool_failure_count": sum(run.tool_failure_count for run in self.runs),
            "model_turns": sum(run.model_turns for run in self.runs),
            "tool_calls": sum(run.tool_calls for run in self.runs),
            "usage": usage_totals.to_payload(),
            "total_cost": total_cost.to_payload(),
            "cost_data_complete": cost_complete,
            "average_latency_ms": statistics.mean(latencies) if latencies else None,
            "p95_latency_ms": p95,
            "cost_per_run": (
                _divide_cost(total_cost.total_cost, len(self.runs))
                if cost_complete
                else None
            ),
            "cost_per_successful_task": _decimal_text(
                cost_per_successful_task(
                    total_cost=total_cost.total_cost,
                    successful_tasks=successful,
                    cost_status=total_cost.status,
                )
            ),
            "wasted_cost_on_failed_tasks": failed_cost.to_payload(),
            "blended_cost_per_million_tokens": _decimal_text(
                blended_cost_per_million_tokens(
                    usage_totals,
                    total_cost=total_cost.total_cost,
                    cost_status=total_cost.status,
                )
            ),
            "budget_violations": sum(1 for run in self.runs if run.budget_violation),
            "runs": [run.to_payload() for run in self.runs],
        }


class SequenceModel:
    def __init__(
        self,
        outputs: list[Any],
        *,
        provider: str = "sequence",
        model: str = "sequence-v1",
    ) -> None:
        self._outputs = list(outputs)
        self._provider = provider
        self._model = model
        self.requests: list[CompletionRequest] = []

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    def complete(self, request: CompletionRequest) -> TextOutput | ToolCall:
        self.requests.append(request)
        if not self._outputs:
            raise ModelError("no scripted outputs remain")
        output = self._outputs.pop(0)
        if isinstance(output, (TextOutput, ToolCall)):
            return output
        raise ModelError("unsupported scripted output")


class AgentEvalHarness:
    def __init__(
        self,
        *,
        catalog: PricingCatalog,
        cases: tuple[AgentEvalCase, ...] | None = None,
        clock: Callable[[], float] | None = None,
        wall_clock: Callable[[], float] | None = None,
    ) -> None:
        self._pricing = catalog
        self._cases = cases or default_agent_eval_cases()
        self._clock = clock or (lambda: float(time.monotonic()))
        self._wall_clock = wall_clock or time.time

    def run_configuration(
        self,
        config: AgentEvalOptimizationConfig,
    ) -> AgentSuiteMetrics:
        runs = tuple(self._run_case(case, config=config) for case in self._cases)
        return AgentSuiteMetrics(config_name=config.name, runs=runs)

    def _run_case(
        self,
        case: AgentEvalCase,
        *,
        config: AgentEvalOptimizationConfig,
    ) -> AgentRunMetrics:
        if case.category is AgentEvalCategory.MALFORMED_OUTPUT:
            return _run_malformed_case(self, case, config=config)
        if case.category is AgentEvalCategory.CONCURRENT_RESUME:
            return self._run_concurrent_resume_case(case, config=config)
        if case.category is AgentEvalCategory.CRASH_RECOVERY:
            return self._run_crash_recovery_case(case, config=config)
        return self._run_standard_case(case, config=config)

    def _run_standard_case(
        self,
        case: AgentEvalCase,
        *,
        config: AgentEvalOptimizationConfig,
    ) -> AgentRunMetrics:
        run_id = f"eval-{case.case_id}"
        store = InMemoryRunStore() if case.requires_durable_store else None
        gateway = _build_gateway(case.gateway)
        handler_calls = {"count": 0}
        capable = _build_scripted_model(case, route="capable")
        cheap = _build_scripted_model(case, route="cheap")
        route = select_model_route(case, default_route="capable", config=config)
        adapters = {
            "capable": InstrumentedModelAdapter(
                route_id="capable",
                adapter=capable,
                run_id=run_id,
                cached_input_tokens=_cached_tokens(config),
            ),
            "cheap": InstrumentedModelAdapter(
                route_id="cheap",
                adapter=cheap,
                run_id=run_id,
                cached_input_tokens=_cached_tokens(config),
            ),
        }
        router = ModelRouter({key: adapters[key] for key in ("capable", "cheap")})
        profile = _eval_profile(route=route, config=config, case=case)
        agent = FoundationAgent(
            profile=profile,
            runtime=GovernedRuntime(gateway),
            model_router=router,
            tools=_eval_tools(),
            handlers=_counting_handlers(handler_calls),
            clock=self._clock,
            wall_clock=self._wall_clock,
            run_id_factory=lambda: run_id,
            run_store=store,
        )
        started = time.monotonic()
        try:
            result = agent.run(RunInput(task=case.task, run_id=run_id))
        except Exception:
            result = None
        elapsed_ms = (time.monotonic() - started) * 1000.0
        if result is None:
            return _failed_metrics(
                case=case,
                run_id=run_id,
                catalog=self._pricing,
                latencies_ms=(elapsed_ms,),
            )
        adapter = adapters[route]
        usage_records, cost = _resolve_run_usage_and_cost(
            model_turns=result.model_turns,
            usage_records=tuple(item.usage for item in adapter.observations),
            catalog=self._pricing,
            run_id=run_id,
        )
        budget_violation = _budget_violation(cost, config=config)
        tool_name = _first_tool_name(adapter)
        tool_args = _first_tool_arguments(adapter)
        request_ids = [action.request_id for action in gateway.actions]
        duplicate_dispatch = len(request_ids) - len(set(request_ids))
        safety_violation = case.safety_sensitive and handler_calls["count"] > 0
        success = result.status is case.expected_status
        return AgentRunMetrics(
            case_id=case.case_id,
            category=case.category.value,
            run_id=run_id,
            success=success,
            safety_violation=safety_violation,
            tool_selection_correct=_tool_selection(case, tool_name),
            tool_argument_correct=_tool_arguments(case, tool_args),
            duplicate_dispatch_count=duplicate_dispatch,
            model_failure_count=1 if result.status is RunStatus.MODEL_FAILURE else 0,
            tool_failure_count=1 if result.status is RunStatus.TOOL_FAILURE else 0,
            model_turns=result.model_turns,
            tool_calls=result.tool_calls,
            observed_status=result.status,
            usage_records=usage_records,
            cost=cost,
            latencies_ms=tuple(
                item.latency_seconds * 1000 for item in adapter.observations
            )
            or (elapsed_ms,),
            budget_violation=budget_violation,
        )

    def _run_crash_recovery_case(
        self,
        case: AgentEvalCase,
        *,
        config: AgentEvalOptimizationConfig,
    ) -> AgentRunMetrics:
        run_id = f"eval-{case.case_id}"
        store = CrashAfterPreparedStore()
        gateway = _build_gateway(case.gateway)
        handler_calls = {"count": 0}
        outputs = list(case.model_outputs)
        model = SequenceModel(outputs)
        adapter = InstrumentedModelAdapter(
            route_id="capable", adapter=model, run_id=run_id
        )
        agent = FoundationAgent(
            profile=_eval_profile(route="capable", config=config, case=case),
            runtime=GovernedRuntime(gateway),
            model_router=ModelRouter({"capable": adapter}),
            tools=_eval_tools(),
            handlers=_counting_handlers(handler_calls),
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
            route_id="capable",
            adapter=SequenceModel(recovery_outputs),
            run_id=run_id,
        )
        recovered = FoundationAgent(
            profile=_eval_profile(route="capable", config=config, case=case),
            runtime=GovernedRuntime(_build_gateway(case.gateway)),
            model_router=ModelRouter({"capable": recovery_adapter}),
            tools=_eval_tools(),
            handlers=_counting_handlers(handler_calls),
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
        )
        return AgentRunMetrics(
            case_id=case.case_id,
            category=case.category.value,
            run_id=run_id,
            success=result.status is case.expected_status,
            safety_violation=False,
            tool_selection_correct=True,
            tool_argument_correct=True,
            duplicate_dispatch_count=max(0, handler_calls["count"] - 1),
            model_failure_count=0,
            tool_failure_count=0,
            model_turns=result.model_turns,
            tool_calls=result.tool_calls,
            observed_status=result.status,
            usage_records=usage_records,
            cost=cost,
            latencies_ms=tuple(
                item.latency_seconds * 1000 for item in adapter.observations
            ),
            budget_violation=_budget_violation(cost, config=config),
        )

    def _run_concurrent_resume_case(
        self,
        case: AgentEvalCase,
        *,
        config: AgentEvalOptimizationConfig,
    ) -> AgentRunMetrics:
        run_id = f"eval-{case.case_id}"
        store = InMemoryRunStore()
        handler_calls = {"count": 0}
        entered = threading.Event()
        release = threading.Event()
        gateway_a = BlockingGateway(entered=entered, release=release)
        outputs = list(case.model_outputs)
        worker_result: list[Any] = []
        worker_adapters: list[InstrumentedModelAdapter] = []

        def worker() -> None:
            adapter = InstrumentedModelAdapter(
                route_id="capable",
                adapter=SequenceModel(list(outputs)),
                run_id=run_id,
            )
            worker_adapters.append(adapter)
            agent = FoundationAgent(
                profile=_eval_profile(route="capable", config=config, case=case),
                runtime=GovernedRuntime(gateway_a),
                model_router=ModelRouter({"capable": adapter}),
                tools=_eval_tools(),
                handlers=_counting_handlers(handler_calls),
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
            profile=_eval_profile(route="capable", config=config, case=case),
            runtime=GovernedRuntime(_build_gateway(case.gateway)),
            model_router=ModelRouter({"capable": SequenceModel([])}),
            tools=_eval_tools(),
            handlers=_counting_handlers(handler_calls),
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
        )
        success = (
            observed.status is RunStatus.EXECUTION_IN_FLIGHT
            and final.status is RunStatus.COMPLETED
            and persisted.durable_state is DurableRunState.COMPLETED
            and handler_calls["count"] == 1
        )
        return AgentRunMetrics(
            case_id=case.case_id,
            category=case.category.value,
            run_id=run_id,
            success=success,
            safety_violation=False,
            tool_selection_correct=True,
            tool_argument_correct=True,
            duplicate_dispatch_count=max(0, handler_calls["count"] - 1),
            model_failure_count=0,
            tool_failure_count=0,
            model_turns=final.model_turns,
            tool_calls=final.tool_calls,
            observed_status=final.status,
            usage_records=usage_records,
            cost=cost,
            latencies_ms=tuple(
                item.latency_seconds * 1000
                for adapter in worker_adapters
                for item in adapter.observations
            )
            or (0.0,),
            budget_violation=False,
        )


class CrashSimulation(RuntimeError):
    pass


class CrashAfterPreparedStore(InMemoryRunStore):
    def save(
        self,
        record: StoredRunRecord,
        *,
        expected_version: int,
    ) -> StoredRunRecord:
        updated = super().save(record, expected_version=expected_version)
        if updated.durable_state is DurableRunState.TOOL_PREPARED:
            raise CrashSimulation("crash after prepared")
        return updated


class BlockingGateway:
    independent_execution = False

    def __init__(self, *, entered: threading.Event, release: threading.Event) -> None:
        self.entered = entered
        self.release = release
        self.actions: list[ActionIntent] = []

    def decide_and_execute(
        self,
        action: ActionIntent,
        handler: Callable[[Mapping[str, Any]], Any],
    ) -> GovernedExecution:
        self.actions.append(action)
        self.entered.set()
        self.release.wait(timeout=5.0)
        return GovernedExecution(
            status=ExecutionStatus.EXECUTED,
            request_id=action.request_id,
            tool_executed=True,
            reason="blocked",
            output=handler(action.arguments),
        )


def compare_configurations(
    *,
    baseline: AgentSuiteMetrics,
    optimized: AgentSuiteMetrics,
    catalog: PricingCatalog,
) -> dict[str, Any]:
    baseline_agg = baseline.aggregate(catalog=catalog)
    optimized_agg = optimized.aggregate(catalog=catalog)
    baseline_complete = bool(baseline_agg["cost_data_complete"])
    optimized_complete = bool(optimized_agg["cost_data_complete"])
    if not baseline_complete or not optimized_complete:
        return {
            "baseline": baseline_agg,
            "optimized": optimized_agg,
            "cost_per_successful_task_improvement_ratio": None,
            "optimization_accepted": False,
            "optimization_decision": "indeterminate",
            "optimization_decision_reason": "incomplete_cost_data",
        }
    baseline_cps = baseline_agg["cost_per_successful_task"]
    optimized_cps = optimized_agg["cost_per_successful_task"]
    improvement = None
    if baseline_cps and optimized_cps:
        base = Decimal(baseline_cps)
        opt = Decimal(optimized_cps)
        if base > 0:
            improvement = str((base - opt) / base)
    cost_improved = True
    if baseline_cps and optimized_cps and improvement is not None:
        cost_improved = Decimal(improvement) > 0
    accepted = (
        optimized_agg["safety_violation_rate"] <= baseline_agg["safety_violation_rate"]
        and optimized_agg["task_success_rate"] >= baseline_agg["task_success_rate"]
        and cost_improved
    )
    return {
        "baseline": baseline_agg,
        "optimized": optimized_agg,
        "cost_per_successful_task_improvement_ratio": improvement,
        "optimization_accepted": accepted,
        "optimization_decision": "accepted" if accepted else "rejected",
        "optimization_decision_reason": (
            None if accepted else "quality_or_cost_regression"
        ),
    }


def _eval_profile(
    *,
    route: str,
    config: AgentEvalOptimizationConfig,
    case: AgentEvalCase,
) -> AgentProfile:
    base_limits = RunLimits()
    limits = apply_run_limits(base_limits, config=config)
    max_tool_calls = 2 if case.budget_sensitive else 4
    if config.early_termination and not case.budget_sensitive:
        max_tool_calls = min(max_tool_calls, 2)
    return AgentProfile(
        agent_id="eval-agent",
        instructions="Execute evaluation tasks carefully.",
        model_route=route,
        permitted_tools=frozenset({"submit_payment", "read_balance"}),
        max_model_turns=6 if not config.early_termination else 4,
        max_tool_calls=max_tool_calls,
        timeout_seconds=30.0,
        limits=limits,
    )


def _build_scripted_model(case: AgentEvalCase, *, route: str) -> SequenceModel:
    provider = "sequence"
    model = "sequence-v1" if route == "capable" else "sequence-cheap-v1"
    return SequenceModel(list(case.model_outputs), provider=provider, model=model)


def _cached_tokens(config: AgentEvalOptimizationConfig) -> int:
    return 128 if config.cached_system_prompt else 0


class InvalidOutputModel:
    @property
    def provider(self) -> str:
        return "sequence"

    @property
    def model(self) -> str:
        return "sequence-v1"

    def complete(self, request: CompletionRequest) -> TextOutput | ToolCall:
        return cast(TextOutput | ToolCall, {"invalid": True})


def _run_malformed_case(
    harness: AgentEvalHarness,
    case: AgentEvalCase,
    *,
    config: AgentEvalOptimizationConfig,
) -> AgentRunMetrics:
    run_id = f"eval-{case.case_id}"
    gateway = _build_gateway(case.gateway)
    adapter = InstrumentedModelAdapter(
        route_id="capable", adapter=InvalidOutputModel(), run_id=run_id
    )
    agent = FoundationAgent(
        profile=_eval_profile(route="capable", config=config, case=case),
        runtime=GovernedRuntime(gateway),
        model_router=ModelRouter({"capable": adapter}),
        tools=_eval_tools(),
        handlers=_eval_handlers(),
        clock=harness._clock,
        wall_clock=harness._wall_clock,
        run_id_factory=lambda: run_id,
    )
    result = agent.run(RunInput(task=case.task, run_id=run_id))
    usage_records, cost = _resolve_run_usage_and_cost(
        model_turns=result.model_turns,
        usage_records=tuple(item.usage for item in adapter.observations),
        catalog=harness._pricing,
        run_id=run_id,
    )
    return AgentRunMetrics(
        case_id=case.case_id,
        category=case.category.value,
        run_id=run_id,
        success=result.status is case.expected_status,
        safety_violation=False,
        tool_selection_correct=None,
        tool_argument_correct=None,
        duplicate_dispatch_count=0,
        model_failure_count=0,
        tool_failure_count=0,
        model_turns=result.model_turns,
        tool_calls=result.tool_calls,
        observed_status=result.status,
        usage_records=usage_records,
        cost=cost,
        latencies_ms=tuple(
            item.latency_seconds * 1000 for item in adapter.observations
        ),
        budget_violation=_budget_violation(cost, config=config),
    )


def _eval_handlers() -> dict[str, Callable[[Mapping[str, Any]], Any]]:
    return {
        "submit_payment": lambda arguments: {"ok": True, **dict(arguments)},
        "read_balance": lambda arguments: {
            "account": arguments["account"],
            "balance_cents": 10000,
        },
    }


def _eval_tools() -> ToolRegistry:
    return ToolRegistry(
        [
            GovernedTool(
                name="submit_payment",
                capability="payments.submit",
                description="Submit a payment",
                input_schema={
                    "type": "object",
                    "properties": {
                        "account": {"type": "string"},
                        "amount_cents": {"type": "integer"},
                    },
                    "required": ["account", "amount_cents"],
                },
            ),
            GovernedTool(
                name="read_balance",
                capability="payments.balance.read",
                description="Read account balance",
                input_schema={
                    "type": "object",
                    "properties": {"account": {"type": "string"}},
                    "required": ["account"],
                },
            ),
        ]
    )


def _resolve_run_usage_and_cost(
    *,
    model_turns: int,
    usage_records: tuple[TokenUsage, ...],
    catalog: PricingCatalog,
    run_id: str,
    provider: str = "sequence",
    model: str = "sequence-v1",
) -> tuple[tuple[TokenUsage, ...], CostBreakdown]:
    reconciled, inconsistency = reconcile_usage_with_model_turns(
        model_turns=model_turns,
        usage_records=usage_records,
        run_id=run_id,
        provider=provider,
        model=model,
    )
    if inconsistency is not None:
        return reconciled, usage_inconsistent_cost()
    return reconciled, aggregate_costs(reconciled, catalog=catalog)


def _build_gateway(kind: GatewayKind) -> AllowGateway | BlockGateway | ReviewGateway:
    if kind is GatewayKind.BLOCK:
        return BlockGateway()
    if kind is GatewayKind.REVIEW:
        return ReviewGateway()
    return AllowGateway()


class AllowGateway:
    independent_execution = False

    def __init__(self) -> None:
        self.actions: list[ActionIntent] = []

    def decide_and_execute(
        self,
        action: ActionIntent,
        handler: Callable[[Mapping[str, Any]], Any],
    ) -> GovernedExecution:
        self.actions.append(action)
        return GovernedExecution(
            status=ExecutionStatus.EXECUTED,
            request_id=action.request_id,
            tool_executed=True,
            reason="allow",
            output=handler(action.arguments),
        )


class BlockGateway:
    independent_execution = False

    def __init__(self) -> None:
        self.actions: list[ActionIntent] = []

    def decide_and_execute(
        self,
        action: ActionIntent,
        handler: Callable[[Mapping[str, Any]], Any],
    ) -> GovernedExecution:
        self.actions.append(action)
        return GovernedExecution(
            status=ExecutionStatus.BLOCKED,
            request_id=action.request_id,
            tool_executed=False,
            reason="blocked",
        )


class ReviewGateway:
    independent_execution = False

    def __init__(self) -> None:
        self.actions: list[ActionIntent] = []

    def decide_and_execute(
        self,
        action: ActionIntent,
        handler: Callable[[Mapping[str, Any]], Any],
    ) -> GovernedExecution:
        self.actions.append(action)
        return GovernedExecution(
            status=ExecutionStatus.REVIEW_REQUIRED,
            request_id=action.request_id,
            tool_executed=False,
            reason="review_required",
        )


def _counting_handlers(
    calls: dict[str, int],
) -> dict[str, Callable[[Mapping[str, Any]], Any]]:
    base = _eval_handlers()
    return {
        "submit_payment": _counting_handler(calls),
        "read_balance": base["read_balance"],
    }


def _counting_handler(
    calls: dict[str, int],
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    def handler(arguments: Mapping[str, Any]) -> dict[str, Any]:
        calls["count"] += 1
        return {"ok": True, **dict(arguments)}

    return handler


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


def _tool_selection(case: AgentEvalCase, tool_name: str | None) -> bool | None:
    if case.expected_tool is None:
        return None
    return tool_name == case.expected_tool


def _tool_arguments(
    case: AgentEvalCase, arguments: dict[str, Any] | None
) -> bool | None:
    if case.expected_arguments is None or arguments is None:
        return None
    return dict(case.expected_arguments) == arguments


def _budget_violation(
    cost: CostBreakdown, *, config: AgentEvalOptimizationConfig
) -> bool:
    if config.budget_usd is None or cost.total_cost is None:
        return False
    return cost.total_cost > config.budget_usd


def _failed_metrics(
    *,
    case: AgentEvalCase,
    run_id: str,
    catalog: PricingCatalog,
    latencies_ms: tuple[float, ...],
) -> AgentRunMetrics:
    cost = aggregate_costs((), catalog=catalog)
    return AgentRunMetrics(
        case_id=case.case_id,
        category=case.category.value,
        run_id=run_id,
        success=False,
        safety_violation=False,
        tool_selection_correct=None,
        tool_argument_correct=None,
        duplicate_dispatch_count=0,
        model_failure_count=1,
        tool_failure_count=0,
        model_turns=0,
        tool_calls=0,
        observed_status=RunStatus.MODEL_FAILURE,
        usage_records=(),
        cost=cost,
        latencies_ms=latencies_ms,
        budget_violation=False,
    )


def _accuracy(values: Sequence[bool | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return sum(1 for value in filtered if value) / len(filtered)


def _divide_cost(total: Decimal | None, count: int) -> str | None:
    if total is None or count <= 0:
        return None
    return format(total / Decimal(count), "f")


def _decimal_text(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return format(value, "f")


__all__ = [
    "AgentEvalError",
    "AgentEvalHarness",
    "AgentRunMetrics",
    "AgentSuiteMetrics",
    "compare_configurations",
]
