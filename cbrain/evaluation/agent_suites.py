"""Deterministic offline agent evaluation suites."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from cbrain.agent import RunStatus
from cbrain.models import TextOutput, ToolCall


class AgentEvalCategory(StrEnum):
    TEXT_ONLY = "text_only"
    TOOL_SELECTION = "tool_selection"
    STRUCTURED_ARGUMENTS = "structured_arguments"
    MULTI_STEP_TOOL = "multi_step_tool"
    BLOCKED_ACTION = "blocked_action"
    REVIEW_REQUIRED = "review_required"
    CRASH_RECOVERY = "crash_recovery"
    CONCURRENT_RESUME = "concurrent_resume"
    MALFORMED_OUTPUT = "malformed_output"
    CONTEXT_GROWTH = "context_growth"
    BUDGET_EXHAUSTION = "budget_exhaustion"


class GatewayKind(StrEnum):
    ALLOW = "allow"
    BLOCK = "block"
    REVIEW = "review"


@dataclass(frozen=True, slots=True)
class AgentEvalCase:
    case_id: str
    category: AgentEvalCategory
    title: str
    task: str
    model_outputs: tuple[Any, ...]
    expected_status: RunStatus
    expected_tool: str | None = None
    expected_arguments: Mapping[str, Any] | None = None
    gateway: GatewayKind = GatewayKind.ALLOW
    safety_sensitive: bool = False
    requires_durable_store: bool = False
    simple_task: bool = False
    budget_sensitive: bool = False

    def __post_init__(self) -> None:
        if not self.case_id.strip() or not self.title.strip() or not self.task.strip():
            raise ValueError("case_id, title, and task must be non-empty")
        if not self.model_outputs:
            raise ValueError("model_outputs must be non-empty")


def default_agent_eval_cases() -> tuple[AgentEvalCase, ...]:
    return (
        AgentEvalCase(
            case_id="text-only-greeting",
            category=AgentEvalCategory.TEXT_ONLY,
            title="Answer with text only",
            task="Reply with a short greeting.",
            model_outputs=(TextOutput(text="Hello from CBrain."),),
            expected_status=RunStatus.COMPLETED,
            simple_task=True,
        ),
        AgentEvalCase(
            case_id="tool-selection-payment",
            category=AgentEvalCategory.TOOL_SELECTION,
            title="Select the payment tool",
            task="Submit a payment to vendor acct-1 for 5000 cents.",
            model_outputs=(
                ToolCall.capture(
                    call_id="pay-1",
                    name="submit_payment",
                    arguments={"account": "acct-1", "amount_cents": 5000},
                ),
                TextOutput(text="payment submitted"),
            ),
            expected_status=RunStatus.COMPLETED,
            expected_tool="submit_payment",
        ),
        AgentEvalCase(
            case_id="structured-arguments-payment",
            category=AgentEvalCategory.STRUCTURED_ARGUMENTS,
            title="Provide structured payment arguments",
            task="Pay vendor acct-2 exactly 2500 cents.",
            model_outputs=(
                ToolCall.capture(
                    call_id="pay-2",
                    name="submit_payment",
                    arguments={"account": "acct-2", "amount_cents": 2500},
                ),
                TextOutput(text="done"),
            ),
            expected_status=RunStatus.COMPLETED,
            expected_tool="submit_payment",
            expected_arguments={"account": "acct-2", "amount_cents": 2500},
        ),
        AgentEvalCase(
            case_id="multi-step-payment",
            category=AgentEvalCategory.MULTI_STEP_TOOL,
            title="Read balance then submit payment",
            task="Check balance for acct-1 then pay 1000 cents.",
            model_outputs=(
                ToolCall.capture(
                    call_id="bal-1",
                    name="read_balance",
                    arguments={"account": "acct-1"},
                ),
                ToolCall.capture(
                    call_id="pay-3",
                    name="submit_payment",
                    arguments={"account": "acct-1", "amount_cents": 1000},
                ),
                TextOutput(text="completed multi-step payment"),
            ),
            expected_status=RunStatus.COMPLETED,
            expected_tool="read_balance",
        ),
        AgentEvalCase(
            case_id="blocked-payment",
            category=AgentEvalCategory.BLOCKED_ACTION,
            title="Blocked payment must not execute",
            task="Submit a blocked payment.",
            model_outputs=(
                ToolCall.capture(
                    call_id="pay-block",
                    name="submit_payment",
                    arguments={"account": "acct-1", "amount_cents": 100},
                ),
            ),
            expected_status=RunStatus.REJECTED,
            expected_tool="submit_payment",
            gateway=GatewayKind.BLOCK,
            safety_sensitive=True,
        ),
        AgentEvalCase(
            case_id="review-required-payment",
            category=AgentEvalCategory.REVIEW_REQUIRED,
            title="Review-required payment stops safely",
            task="Submit a payment requiring review.",
            model_outputs=(
                ToolCall.capture(
                    call_id="pay-review",
                    name="submit_payment",
                    arguments={"account": "acct-1", "amount_cents": 9000},
                ),
            ),
            expected_status=RunStatus.REJECTED,
            expected_tool="submit_payment",
            gateway=GatewayKind.REVIEW,
            safety_sensitive=True,
        ),
        AgentEvalCase(
            case_id="crash-recovery-prepared",
            category=AgentEvalCategory.CRASH_RECOVERY,
            title="Resume after crash before dispatch",
            task="Submit payment after simulated crash.",
            model_outputs=(
                ToolCall.capture(
                    call_id="pay-crash",
                    name="submit_payment",
                    arguments={"account": "acct-1", "amount_cents": 5000},
                ),
                TextOutput(text="recovered"),
            ),
            expected_status=RunStatus.COMPLETED,
            expected_tool="submit_payment",
            requires_durable_store=True,
        ),
        AgentEvalCase(
            case_id="concurrent-resume-inflight",
            category=AgentEvalCategory.CONCURRENT_RESUME,
            title="Second worker observes in-flight execution",
            task="Submit payment while another worker is active.",
            model_outputs=(
                ToolCall.capture(
                    call_id="pay-concurrent",
                    name="submit_payment",
                    arguments={"account": "acct-1", "amount_cents": 5000},
                ),
                TextOutput(text="done"),
            ),
            expected_status=RunStatus.COMPLETED,
            expected_tool="submit_payment",
            requires_durable_store=True,
        ),
        AgentEvalCase(
            case_id="malformed-model-output",
            category=AgentEvalCategory.MALFORMED_OUTPUT,
            title="Reject unsupported model output",
            task="Return something invalid.",
            model_outputs=(object(),),
            expected_status=RunStatus.INVALID_MODEL_RESPONSE,
        ),
        AgentEvalCase(
            case_id="context-growth",
            category=AgentEvalCategory.CONTEXT_GROWTH,
            title="Handle growing context across turns",
            task="Summarize two tool observations.",
            model_outputs=(
                ToolCall.capture(
                    call_id="bal-a",
                    name="read_balance",
                    arguments={"account": "acct-1"},
                ),
                ToolCall.capture(
                    call_id="bal-b",
                    name="read_balance",
                    arguments={"account": "acct-2"},
                ),
                TextOutput(text="balances summarized"),
            ),
            expected_status=RunStatus.COMPLETED,
        ),
        AgentEvalCase(
            case_id="budget-exhaustion",
            category=AgentEvalCategory.BUDGET_EXHAUSTION,
            title="Stop when token budget is exhausted",
            task="Keep calling tools until budget is exhausted.",
            model_outputs=tuple(
                ToolCall.capture(
                    call_id=f"pay-loop-{index}",
                    name="submit_payment",
                    arguments={"account": "acct-1", "amount_cents": 100},
                )
                for index in range(8)
            )
            + (TextOutput(text="should not reach"),),
            expected_status=RunStatus.LIMIT_REACHED,
            budget_sensitive=True,
        ),
    )


def offline_agent_eval_cases() -> tuple[AgentEvalCase, ...]:
    return default_agent_eval_cases()


CaseFilter = Callable[[AgentEvalCase], bool]


def filter_cases(
    cases: tuple[AgentEvalCase, ...],
    *,
    categories: set[AgentEvalCategory] | None = None,
    include_durable: bool = True,
) -> tuple[AgentEvalCase, ...]:
    selected: list[AgentEvalCase] = []
    for case in cases:
        if categories is not None and case.category not in categories:
            continue
        if case.requires_durable_store and not include_durable:
            continue
        selected.append(case)
    return tuple(selected)


__all__ = [
    "AgentEvalCase",
    "AgentEvalCategory",
    "CaseFilter",
    "GatewayKind",
    "default_agent_eval_cases",
    "filter_cases",
    "offline_agent_eval_cases",
]
