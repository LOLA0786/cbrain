"""Bounded reasoning and tool loop for reusable foundation agents."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

from cbrain.contracts import ActionIntent, ContractError, ExecutionStatus
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
    Wall-clock time can move backwards and break deadline enforcement.

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
    ) -> None:
        self._profile = profile
        self._runtime = runtime
        self._model_router = model_router
        self._tools = tools
        self._handlers = dict(handlers)
        self._clock = clock or time.monotonic
        self._run_id_factory = run_id_factory or (lambda: str(uuid.uuid4()))
        self._request_id_factory = request_id_factory or (
            lambda run_id, step: f"{run_id}-step-{step}"
        )
        self._cancelled = cancelled or (lambda: False)
        self._validate_handlers()

    def run(self, run_input: RunInput) -> RunResult:
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
