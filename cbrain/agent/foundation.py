"""Bounded reasoning and tool loop for reusable foundation agents."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

from cbrain.contracts import ActionIntent, ContractError, ExecutionStatus
from cbrain.knowledge.errors import KnowledgeUnavailable
from cbrain.models import (
    CompletionRequest,
    Message,
    MessageRole,
    ModelError,
    ModelRouter,
    TextOutput,
    ToolCall,
)
from cbrain.runtime import GovernedRuntime

from .contracts import RunEvent, RunEventKind, RunInput, RunResult, RunStatus
from .durable import DurableRunState, RunStore, RunStoreError, record_to_run_result
from .durable_loop import (
    DurableRunContext,
    finalize_durable_record,
    initialize_durable_run,
    persist_record,
    resolve_claim_conflict,
    restore_action,
    restore_execution,
    restore_tool_call,
)
from .profile import AgentProfile
from .tools import ToolRegistry, ToolRegistryError


class FoundationAgentError(ValueError):
    """Foundation agent input or configuration is invalid."""


class FoundationAgent:
    """Domain-neutral agent loop over model routing and governed execution.

    ``run()`` is synchronous and cooperative. Timeout and cancellation checks
    occur only between model turns. An in-flight model call or governed tool
    handler is not interrupted.

    The injected ``clock`` must be monotonic (for example ``time.monotonic``).
    Durable runs persist an absolute UTC expiration via ``wall_clock``.

    One ``FoundationAgent`` instance may be reused across sequential runs, but
    concurrent ``run()`` calls on the same instance are not supported.
    """

    def __init__(
        self,
        *,
        profile: AgentProfile,
        runtime: GovernedRuntime,
        model_router: ModelRouter,
        tools: ToolRegistry,
        handlers: Mapping[str, Callable[[Mapping[str, Any]], Any]],
        clock: Callable[[], float] | None = None,
        run_id_factory: Callable[[], str] | None = None,
        request_id_factory: Callable[[str, int], str] | None = None,
        cancelled: Callable[[], bool] | None = None,
        run_store: RunStore | None = None,
        wall_clock: Callable[[], float] | None = None,
        knowledge: Any | None = None,
    ) -> None:
        self._profile = profile
        self._runtime = runtime
        self._model_router = model_router
        self._tools = tools
        self._handlers = dict(handlers)
        self._clock = clock or time.monotonic
        self._wall_clock = wall_clock or time.time
        self._run_id_factory = run_id_factory or (lambda: str(uuid.uuid4()))
        self._request_id_factory = request_id_factory or (
            lambda run_id, step: f"{run_id}-step-{step}"
        )
        self._cancelled = cancelled or (lambda: False)
        self._run_store = run_store
        self._knowledge = knowledge
        self._validate_handlers()

    def run(self, run_input: RunInput) -> RunResult:
        if self._run_store is None:
            return self._run_ephemeral(run_input)
        return self._run_durable(run_input)

    def _run_ephemeral(self, run_input: RunInput) -> RunResult:
        if not isinstance(run_input.task, str) or not run_input.task.strip():
            raise FoundationAgentError("task must be non-empty")

        limits = self._profile.limits
        if len(run_input.task) > limits.max_task_chars:
            return self._failed(
                run_id=run_input.run_id or self._run_id_factory(),
                status=RunStatus.INVALID_MODEL_RESPONSE,
                events=[],
                metadata=self._metadata(
                    run_input,
                    reason="task exceeds max_task_chars",
                ),
                tool_calls=0,
                model_turns=0,
            )

        run_id = run_input.run_id or self._run_id_factory()
        started_at = self._clock()
        deadline = started_at + float(self._profile.timeout_seconds)
        events: list[RunEvent] = []
        messages: list[Message] = [
            Message(role=MessageRole.SYSTEM, content=self._profile.instructions),
            Message(role=MessageRole.USER, content=run_input.task),
        ]
        knowledge_message, knowledge_tools_blocked = self._knowledge_context(run_input)
        if knowledge_message is not None:
            messages.append(knowledge_message)
        tool_calls = 0
        model_turns = 0

        events.append(
            RunEvent(
                kind=RunEventKind.RUN_STARTED,
                step=0,
                timestamp=started_at,
                detail=_snapshot_detail({"run_id": run_id}),
            )
        )

        try:
            tool_definitions = self._tools.definitions_for(
                self._profile.permitted_tools
            )
        except ToolRegistryError as exc:
            return self._failed(
                run_id=run_id,
                status=RunStatus.MODEL_FAILURE,
                events=events,
                metadata=self._metadata(run_input, reason=str(exc)),
                tool_calls=tool_calls,
                model_turns=model_turns,
            )

        for step in range(1, self._profile.max_model_turns + 1):
            if self._cancelled():
                return self._terminal(
                    run_id=run_id,
                    status=RunStatus.CANCELLED,
                    events=events,
                    metadata=self._metadata(run_input),
                    tool_calls=tool_calls,
                    model_turns=model_turns,
                )

            now = self._clock()
            if now > deadline:
                return self._terminal(
                    run_id=run_id,
                    status=RunStatus.TIMED_OUT,
                    events=events,
                    metadata=self._metadata(run_input),
                    tool_calls=tool_calls,
                    model_turns=model_turns,
                )

            if len(messages) > limits.max_context_messages:
                return self._terminal(
                    run_id=run_id,
                    status=RunStatus.LIMIT_REACHED,
                    events=events,
                    metadata=self._metadata(
                        run_input,
                        reason="context message limit reached",
                    ),
                    tool_calls=tool_calls,
                    model_turns=model_turns,
                )

            events.append(
                RunEvent(
                    kind=RunEventKind.MODEL_TURN,
                    step=step,
                    timestamp=now,
                    detail=_snapshot_detail({"run_id": run_id}),
                )
            )

            try:
                output = self._model_router.complete(
                    self._profile.model_route,
                    CompletionRequest(
                        messages=tuple(messages),
                        tools=tool_definitions,
                        max_output_tokens=limits.max_output_tokens,
                    ),
                )
            except ModelError as exc:
                return self._failed(
                    run_id=run_id,
                    status=RunStatus.MODEL_FAILURE,
                    events=events,
                    metadata=self._metadata(run_input, reason=str(exc)),
                    tool_calls=tool_calls,
                    model_turns=model_turns,
                )

            model_turns += 1

            if isinstance(output, TextOutput):
                if len(output.text) > limits.max_model_text_chars:
                    return self._failed(
                        run_id=run_id,
                        status=RunStatus.INVALID_MODEL_RESPONSE,
                        events=events,
                        metadata=self._metadata(
                            run_input,
                            reason="model text exceeds max_model_text_chars",
                        ),
                        tool_calls=tool_calls,
                        model_turns=model_turns,
                    )
                events.append(
                    RunEvent(
                        kind=RunEventKind.MODEL_RESPONSE,
                        step=step,
                        timestamp=self._clock(),
                        detail=_snapshot_detail({"kind": "text"}),
                    )
                )
                events.append(
                    RunEvent(
                        kind=RunEventKind.RUN_COMPLETED,
                        step=step,
                        timestamp=self._clock(),
                        detail=_snapshot_detail({"run_id": run_id}),
                    )
                )
                return RunResult(
                    run_id=run_id,
                    status=RunStatus.COMPLETED,
                    final_text=output.text,
                    tool_calls=tool_calls,
                    model_turns=model_turns,
                    events=tuple(events),
                    metadata=self._metadata(run_input),
                )

            if not isinstance(output, ToolCall):
                return self._failed(
                    run_id=run_id,
                    status=RunStatus.INVALID_MODEL_RESPONSE,
                    events=events,
                    metadata=self._metadata(
                        run_input,
                        reason="model returned an unsupported output type",
                    ),
                    tool_calls=tool_calls,
                    model_turns=model_turns,
                )

            if output.name not in self._profile.permitted_tools:
                return self._failed(
                    run_id=run_id,
                    status=RunStatus.INVALID_MODEL_RESPONSE,
                    events=events,
                    metadata=self._metadata(
                        run_input,
                        reason=f"tool {output.name!r} is not permitted for this agent",
                    ),
                    tool_calls=tool_calls,
                    model_turns=model_turns,
                )

            if not self._tools.contains(output.name):
                return self._failed(
                    run_id=run_id,
                    status=RunStatus.INVALID_MODEL_RESPONSE,
                    events=events,
                    metadata=self._metadata(
                        run_input,
                        reason=f"tool {output.name!r} is not registered",
                    ),
                    tool_calls=tool_calls,
                    model_turns=model_turns,
                )

            if tool_calls >= self._profile.max_tool_calls:
                return self._terminal(
                    run_id=run_id,
                    status=RunStatus.LIMIT_REACHED,
                    events=events,
                    metadata=self._metadata(run_input),
                    tool_calls=tool_calls,
                    model_turns=model_turns,
                )

            tool = self._tools.get(output.name)
            request_id = self._request_id_factory(run_id, step)
            if knowledge_tools_blocked:
                return self._failed(
                    run_id=run_id,
                    status=RunStatus.TOOL_FAILURE,
                    events=events,
                    metadata=self._metadata(
                        run_input,
                        reason="KNOWLEDGE_UNAVAILABLE",
                    ),
                    tool_calls=tool_calls,
                    model_turns=model_turns,
                )
            try:
                action = ActionIntent.capture(
                    agent_id=self._profile.agent_id,
                    framework="cbrain-foundation",
                    tool_name=output.name,
                    capability=tool.capability,
                    arguments=output.arguments,
                    request_id=request_id,
                    idempotency_key=request_id,
                    timestamp=now,
                    context={"run_id": run_id, "tool_call_id": output.call_id},
                )
            except ContractError as exc:
                return self._failed(
                    run_id=run_id,
                    status=RunStatus.INVALID_MODEL_RESPONSE,
                    events=events,
                    metadata=self._metadata(run_input, reason=str(exc)),
                    tool_calls=tool_calls,
                    model_turns=model_turns,
                )

            events.append(
                RunEvent(
                    kind=RunEventKind.TOOL_REQUESTED,
                    step=step,
                    timestamp=now,
                    detail=_snapshot_detail(
                        {
                            "tool_name": output.name,
                            "capability": tool.capability,
                            "request_id": request_id,
                        }
                    ),
                )
            )

            handler = self._handlers[output.name]
            execution = self._runtime.execute(action, handler)
            tool_calls += 1

            if execution.status is ExecutionStatus.EXECUTED:
                observation = {
                    "status": execution.status.value,
                    "request_id": request_id,
                    "output": execution.output,
                }
                observation_text = json.dumps(observation, sort_keys=True, default=str)
                if len(observation_text) > limits.max_observation_chars:
                    return self._failed(
                        run_id=run_id,
                        status=RunStatus.LIMIT_REACHED,
                        events=events,
                        metadata=self._metadata(
                            run_input,
                            reason="tool observation exceeds max_observation_chars",
                        ),
                        tool_calls=tool_calls,
                        model_turns=model_turns,
                    )
                events.append(
                    RunEvent(
                        kind=RunEventKind.TOOL_COMPLETED,
                        step=step,
                        timestamp=self._clock(),
                        detail=_snapshot_detail(
                            {
                                "request_id": request_id,
                                "execution_status": execution.status.value,
                            }
                        ),
                    )
                )
                messages.append(
                    Message(
                        role=MessageRole.TOOL,
                        content=observation_text,
                        tool_call_id=output.call_id,
                        tool_name=output.name,
                    )
                )
                continue

            events.append(
                RunEvent(
                    kind=RunEventKind.TOOL_REJECTED,
                    step=step,
                    timestamp=self._clock(),
                    detail=_snapshot_detail(
                        {
                            "request_id": request_id,
                            "execution_status": execution.status.value,
                            "reason": execution.reason,
                        }
                    ),
                )
            )

            if execution.status is ExecutionStatus.INDETERMINATE:
                return self._failed(
                    run_id=run_id,
                    status=RunStatus.TOOL_FAILURE,
                    events=events,
                    metadata=self._metadata(run_input, reason=execution.reason),
                    tool_calls=tool_calls,
                    model_turns=model_turns,
                )

            if execution.status in {
                ExecutionStatus.BLOCKED,
                ExecutionStatus.REVIEW_REQUIRED,
            }:
                return self._failed(
                    run_id=run_id,
                    status=RunStatus.REJECTED,
                    events=events,
                    metadata=self._metadata(run_input, reason=execution.reason),
                    tool_calls=tool_calls,
                    model_turns=model_turns,
                )

            if execution.status is ExecutionStatus.CONTROL_FAILURE:
                return self._failed(
                    run_id=run_id,
                    status=RunStatus.TOOL_FAILURE,
                    events=events,
                    metadata=self._metadata(run_input, reason=execution.reason),
                    tool_calls=tool_calls,
                    model_turns=model_turns,
                )

            return self._failed(
                run_id=run_id,
                status=RunStatus.TOOL_FAILURE,
                events=events,
                metadata=self._metadata(
                    run_input,
                    reason=f"unexpected execution status: {execution.status.value}",
                ),
                tool_calls=tool_calls,
                model_turns=model_turns,
            )

        return self._terminal(
            run_id=run_id,
            status=RunStatus.LIMIT_REACHED,
            events=events,
            metadata=self._metadata(run_input),
            tool_calls=tool_calls,
            model_turns=model_turns,
        )

    def _run_durable(self, run_input: RunInput) -> RunResult:
        if not isinstance(run_input.task, str) or not run_input.task.strip():
            raise FoundationAgentError("task must be non-empty")
        if self._run_store is None:
            raise FoundationAgentError("durable run requested without a run store")

        limits = self._profile.limits
        run_id = run_input.run_id or self._run_id_factory()
        started_at_utc = self._wall_clock()
        expires_at_utc = started_at_utc + float(self._profile.timeout_seconds)
        started_at_monotonic = self._clock()

        if len(run_input.task) > limits.max_task_chars:
            return self._failed(
                run_id=run_id,
                status=RunStatus.INVALID_MODEL_RESPONSE,
                events=[],
                metadata=self._metadata(
                    run_input,
                    reason="task exceeds max_task_chars",
                ),
                tool_calls=0,
                model_turns=0,
            )

        started_event = RunEvent(
            kind=RunEventKind.RUN_STARTED,
            step=0,
            timestamp=started_at_monotonic,
            detail=_snapshot_detail({"run_id": run_id}),
        )
        try:
            initialized = initialize_durable_run(
                profile=self._profile,
                run_input=RunInput(
                    task=run_input.task,
                    run_id=run_id,
                    metadata=run_input.metadata,
                ),
                run_store=self._run_store,
                started_at_utc=started_at_utc,
                expires_at_utc=expires_at_utc,
                wall_clock=self._wall_clock,
                monotonic_clock=self._clock,
                run_id=run_id,
                instructions=self._profile.instructions,
                started_event=started_event,
            )
        except RunStoreError as exc:
            raise FoundationAgentError(str(exc)) from exc

        if isinstance(initialized, RunResult):
            return initialized

        ctx = initialized
        knowledge_message, knowledge_tools_blocked = self._knowledge_context(run_input)
        if knowledge_message is not None:
            ctx.messages.append(knowledge_message)
        try:
            tool_definitions = self._tools.definitions_for(
                self._profile.permitted_tools
            )
        except ToolRegistryError as exc:
            return self._persist_terminal(
                ctx,
                status=RunStatus.MODEL_FAILURE,
                metadata=self._metadata(run_input, reason=str(exc)),
            )

        resume_phase: str | None = None
        if ctx.record.durable_state is DurableRunState.TOOL_PREPARED:
            resume_phase = "prepared"
        elif ctx.merge_completed_tool:
            resume_phase = "completed"

        for step in range(ctx.next_step, self._profile.max_model_turns + 1):
            if self._cancelled():
                return self._persist_terminal(ctx, status=RunStatus.CANCELLED)

            now = self._clock()
            if now > ctx.deadline_monotonic:
                return self._persist_terminal(ctx, status=RunStatus.TIMED_OUT)

            if len(ctx.messages) > limits.max_context_messages:
                return self._persist_terminal(
                    ctx,
                    status=RunStatus.LIMIT_REACHED,
                    metadata=self._metadata(
                        run_input,
                        reason="context message limit reached",
                    ),
                )

            output: TextOutput | ToolCall | None = None
            tool_resume_completed = False
            tool_resume_prepared = False
            if resume_phase == "completed":
                output = restore_tool_call(ctx.record)
                tool_resume_completed = True
                resume_phase = None
            elif resume_phase == "prepared":
                output = restore_tool_call(ctx.record)
                tool_resume_prepared = True
                resume_phase = None
            else:
                ctx.events.append(
                    RunEvent(
                        kind=RunEventKind.MODEL_TURN,
                        step=step,
                        timestamp=now,
                        detail=_snapshot_detail({"run_id": run_id}),
                    )
                )
                try:
                    output = self._model_router.complete(
                        self._profile.model_route,
                        CompletionRequest(
                            messages=tuple(ctx.messages),
                            tools=tool_definitions,
                            max_output_tokens=limits.max_output_tokens,
                        ),
                    )
                except ModelError as exc:
                    return self._persist_terminal(
                        ctx,
                        status=RunStatus.MODEL_FAILURE,
                        metadata=self._metadata(run_input, reason=str(exc)),
                    )
                ctx.model_turns += 1
                ctx.record = persist_record(
                    self._run_store,
                    ctx.record,
                    durable_state=DurableRunState.RUNNING,
                    events=ctx.events,
                    messages=ctx.messages,
                    model_turns=ctx.model_turns,
                    next_step=step,
                )

            if (
                not tool_resume_completed
                and not tool_resume_prepared
                and isinstance(output, TextOutput)
            ):
                if len(output.text) > limits.max_model_text_chars:
                    return self._persist_terminal(
                        ctx,
                        status=RunStatus.INVALID_MODEL_RESPONSE,
                        metadata=self._metadata(
                            run_input,
                            reason="model text exceeds max_model_text_chars",
                        ),
                    )
                ctx.events.append(
                    RunEvent(
                        kind=RunEventKind.MODEL_RESPONSE,
                        step=step,
                        timestamp=self._clock(),
                        detail=_snapshot_detail({"kind": "text"}),
                    )
                )
                return self._persist_terminal(
                    ctx,
                    status=RunStatus.COMPLETED,
                    final_text=output.text,
                )

            if output is None or not isinstance(output, ToolCall):
                if not tool_resume_completed and not tool_resume_prepared:
                    return self._persist_terminal(
                        ctx,
                        status=RunStatus.INVALID_MODEL_RESPONSE,
                        metadata=self._metadata(
                            run_input,
                            reason="model returned an unsupported output type",
                        ),
                    )
                raise FoundationAgentError("durable resume missing tool call")

            if not tool_resume_completed and not tool_resume_prepared:
                if output.name not in self._profile.permitted_tools:
                    return self._persist_terminal(
                        ctx,
                        status=RunStatus.INVALID_MODEL_RESPONSE,
                        metadata=self._metadata(
                            run_input,
                            reason=(
                                f"tool {output.name!r} is not permitted for this agent"
                            ),
                        ),
                    )
                if not self._tools.contains(output.name):
                    return self._persist_terminal(
                        ctx,
                        status=RunStatus.INVALID_MODEL_RESPONSE,
                        metadata=self._metadata(
                            run_input,
                            reason=f"tool {output.name!r} is not registered",
                        ),
                    )
                if ctx.tool_calls >= self._profile.max_tool_calls:
                    return self._persist_terminal(ctx, status=RunStatus.LIMIT_REACHED)

            tool_result = self._execute_durable_tool(
                ctx=ctx,
                run_input=run_input,
                step=step,
                output=output,
                now=now,
                resume_completed=tool_resume_completed,
                resume_prepared=tool_resume_prepared,
                knowledge_tools_blocked=knowledge_tools_blocked,
            )
            if isinstance(tool_result, RunResult):
                return tool_result
            ctx = tool_result
            ctx.record = persist_record(
                self._run_store,
                ctx.record,
                durable_state=DurableRunState.RUNNING,
                events=ctx.events,
                messages=ctx.messages,
                tool_calls=ctx.tool_calls,
                model_turns=ctx.model_turns,
                next_step=step + 1,
                clear_pending=True,
            )

        return self._persist_terminal(ctx, status=RunStatus.LIMIT_REACHED)

    def _execute_durable_tool(
        self,
        *,
        ctx: DurableRunContext,
        run_input: RunInput,
        step: int,
        output: ToolCall,
        now: float,
        resume_completed: bool,
        resume_prepared: bool,
        knowledge_tools_blocked: bool = False,
    ) -> DurableRunContext | RunResult:
        assert self._run_store is not None
        limits = self._profile.limits
        tool = self._tools.get(output.name)
        request_id = ctx.record.pending_request_id or self._request_id_factory(
            ctx.run_id, step
        )

        if resume_completed:
            execution = restore_execution(ctx.record)
        else:
            if not resume_prepared:
                if knowledge_tools_blocked:
                    return self._persist_terminal(
                        ctx,
                        status=RunStatus.TOOL_FAILURE,
                        metadata=self._metadata(
                            run_input,
                            reason="KNOWLEDGE_UNAVAILABLE",
                        ),
                    )
                try:
                    action = ActionIntent.capture(
                        agent_id=self._profile.agent_id,
                        framework="cbrain-foundation",
                        tool_name=output.name,
                        capability=tool.capability,
                        arguments=output.arguments,
                        request_id=request_id,
                        idempotency_key=request_id,
                        timestamp=now,
                        context={"run_id": ctx.run_id, "tool_call_id": output.call_id},
                    )
                except ContractError as exc:
                    return self._persist_terminal(
                        ctx,
                        status=RunStatus.INVALID_MODEL_RESPONSE,
                        metadata=self._metadata(run_input, reason=str(exc)),
                    )

                ctx.events.append(
                    RunEvent(
                        kind=RunEventKind.TOOL_REQUESTED,
                        step=step,
                        timestamp=now,
                        detail=_snapshot_detail(
                            {
                                "tool_name": output.name,
                                "capability": tool.capability,
                                "request_id": request_id,
                            }
                        ),
                    )
                )
                ctx.record = persist_record(
                    self._run_store,
                    ctx.record,
                    durable_state=DurableRunState.TOOL_PREPARED,
                    events=ctx.events,
                    messages=ctx.messages,
                    model_turns=ctx.model_turns,
                    next_step=step,
                    pending_request_id=request_id,
                    pending_idempotency_key=request_id,
                    pending_action={
                        "agent_id": self._profile.agent_id,
                        "framework": "cbrain-foundation",
                        "tool_name": output.name,
                        "capability": tool.capability,
                        "arguments": output.arguments,
                        "request_id": request_id,
                        "idempotency_key": request_id,
                        "timestamp": now,
                        "context": {
                            "run_id": ctx.run_id,
                            "tool_call_id": output.call_id,
                        },
                    },
                    pending_tool_call_id=output.call_id,
                    pending_tool_name=output.name,
                )
            else:
                action = restore_action(ctx.record)

            claimed = self._run_store.claim_tool_dispatch(
                ctx.run_id,
                expected_version=ctx.record.version,
            )
            if claimed is None:
                return resolve_claim_conflict(self._run_store, ctx.run_id)

            ctx.record = claimed
            handler = self._handlers[output.name]
            execution = self._runtime.execute(action, handler)
            ctx.tool_calls += 1
            ctx.record = persist_record(
                self._run_store,
                ctx.record,
                durable_state=DurableRunState.TOOL_COMPLETED,
                events=ctx.events,
                messages=ctx.messages,
                tool_calls=ctx.tool_calls,
                model_turns=ctx.model_turns,
                next_step=step,
                pending_execution_status=execution.status.value,
                pending_execution_output=execution.output,
                pending_execution_reason=execution.reason,
            )

        if execution.status is ExecutionStatus.EXECUTED:
            observation = {
                "status": execution.status.value,
                "request_id": request_id,
                "output": execution.output,
            }
            observation_text = json.dumps(observation, sort_keys=True, default=str)
            if len(observation_text) > limits.max_observation_chars:
                return self._persist_terminal(
                    ctx,
                    status=RunStatus.LIMIT_REACHED,
                    metadata=self._metadata(
                        run_input,
                        reason="tool observation exceeds max_observation_chars",
                    ),
                )
            ctx.events.append(
                RunEvent(
                    kind=RunEventKind.TOOL_COMPLETED,
                    step=step,
                    timestamp=self._clock(),
                    detail=_snapshot_detail(
                        {
                            "request_id": request_id,
                            "execution_status": execution.status.value,
                        }
                    ),
                )
            )
            ctx.messages.append(
                Message(
                    role=MessageRole.TOOL,
                    content=observation_text,
                    tool_call_id=output.call_id,
                    tool_name=output.name,
                )
            )
            return ctx

        ctx.events.append(
            RunEvent(
                kind=RunEventKind.TOOL_REJECTED,
                step=step,
                timestamp=self._clock(),
                detail=_snapshot_detail(
                    {
                        "request_id": request_id,
                        "execution_status": execution.status.value,
                        "reason": execution.reason,
                    }
                ),
            )
        )
        if execution.status is ExecutionStatus.INDETERMINATE:
            return self._persist_terminal(
                ctx,
                status=RunStatus.TOOL_FAILURE,
                metadata=self._metadata(run_input, reason=execution.reason),
            )
        if execution.status in {
            ExecutionStatus.BLOCKED,
            ExecutionStatus.REVIEW_REQUIRED,
        }:
            return self._persist_terminal(
                ctx,
                status=RunStatus.REJECTED,
                metadata=self._metadata(run_input, reason=execution.reason),
            )
        if execution.status is ExecutionStatus.CONTROL_FAILURE:
            return self._persist_terminal(
                ctx,
                status=RunStatus.TOOL_FAILURE,
                metadata=self._metadata(run_input, reason=execution.reason),
            )
        return self._persist_terminal(
            ctx,
            status=RunStatus.TOOL_FAILURE,
            metadata=self._metadata(
                run_input,
                reason=f"unexpected execution status: {execution.status.value}",
            ),
        )

    def _persist_terminal(
        self,
        ctx: DurableRunContext,
        *,
        status: RunStatus,
        final_text: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RunResult:
        assert self._run_store is not None
        durable_state = {
            RunStatus.COMPLETED: DurableRunState.COMPLETED,
            RunStatus.CANCELLED: DurableRunState.CANCELLED,
            RunStatus.RECOVERY_REQUIRED: DurableRunState.RECOVERY_REQUIRED,
        }.get(status, DurableRunState.FAILED)
        terminal_metadata = dict(metadata or self._metadata(ctx.run_input))
        if status is not RunStatus.COMPLETED:
            ctx.events.append(
                RunEvent(
                    kind=RunEventKind.RUN_FAILED,
                    step=len(ctx.events),
                    timestamp=0.0,
                    detail=_snapshot_detail({"status": status.value}),
                )
            )
        else:
            ctx.events.append(
                RunEvent(
                    kind=RunEventKind.RUN_COMPLETED,
                    step=len(ctx.events),
                    timestamp=self._clock(),
                    detail=_snapshot_detail({"run_id": ctx.run_id}),
                )
            )
        terminal = finalize_durable_record(
            ctx.record,
            durable_state=durable_state,
            terminal_status=status,
            final_text=final_text,
            terminal_metadata=terminal_metadata,
            events=ctx.events,
            messages=ctx.messages,
            tool_calls=ctx.tool_calls,
            model_turns=ctx.model_turns,
        )
        saved = self._run_store.save(terminal, expected_version=ctx.record.version)
        return record_to_run_result(saved)

    def _knowledge_context(self, run_input: RunInput) -> tuple[Message | None, bool]:
        if self._knowledge is None:
            return None, False
        metadata = dict(run_input.metadata or {})
        tenant_id = metadata.get("tenant_id")
        collection_id = metadata.get("collection_id")
        principal_id = metadata.get("principal_id")
        required = self._profile.knowledge_required_for_tools
        if not all(
            isinstance(value, str) and value.strip()
            for value in (tenant_id, collection_id, principal_id)
        ):
            return None, required
        from cbrain.knowledge.contracts import RetrievalQuery

        try:
            context = self._knowledge.retrieve(
                RetrievalQuery(
                    tenant_id=str(tenant_id),
                    collection_id=str(collection_id),
                    principal_id=str(principal_id),
                    text=run_input.task,
                    top_k=4,
                    created_at=self._wall_clock(),
                )
            )
        except KnowledgeUnavailable:
            return None, required
        except Exception:
            return None, required
        return (
            Message(role=MessageRole.USER, content=context.quoted_evidence),
            False,
        )

    def _validate_handlers(self) -> None:
        missing = sorted(self._profile.permitted_tools - set(self._handlers))
        if missing:
            raise FoundationAgentError(
                f"missing handlers for permitted tools: {missing}"
            )
        extra = sorted(set(self._handlers) - self._profile.permitted_tools)
        if extra:
            raise FoundationAgentError(
                f"handlers provided for non-permitted tools: {extra}"
            )

    @staticmethod
    def _metadata(
        run_input: RunInput,
        *,
        reason: str | None = None,
    ) -> Mapping[str, Any]:
        payload: dict[str, Any] = {}
        if run_input.metadata:
            payload.update(dict(run_input.metadata))
        if reason is not None:
            payload["reason"] = reason
        return MappingProxyType(payload)

    @staticmethod
    def _terminal(
        *,
        run_id: str,
        status: RunStatus,
        events: list[RunEvent],
        metadata: Mapping[str, Any],
        tool_calls: int,
        model_turns: int,
    ) -> RunResult:
        events.append(
            RunEvent(
                kind=RunEventKind.RUN_FAILED,
                step=len(events),
                timestamp=0.0,
                detail=_snapshot_detail({"status": status.value}),
            )
        )
        return RunResult(
            run_id=run_id,
            status=status,
            final_text=None,
            tool_calls=tool_calls,
            model_turns=model_turns,
            events=tuple(events),
            metadata=metadata,
        )

    @staticmethod
    def _failed(
        *,
        run_id: str,
        status: RunStatus,
        events: list[RunEvent],
        metadata: Mapping[str, Any],
        tool_calls: int,
        model_turns: int,
    ) -> RunResult:
        events.append(
            RunEvent(
                kind=RunEventKind.RUN_FAILED,
                step=len(events),
                timestamp=0.0,
                detail=_snapshot_detail({"status": status.value}),
            )
        )
        return RunResult(
            run_id=run_id,
            status=status,
            final_text=None,
            tool_calls=tool_calls,
            model_turns=model_turns,
            events=tuple(events),
            metadata=metadata,
        )


def _snapshot_detail(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        json.loads(json.dumps(dict(value), sort_keys=True, default=str))
    )


__all__ = ["FoundationAgent", "FoundationAgentError"]
