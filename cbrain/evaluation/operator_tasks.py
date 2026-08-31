"""Operator-loop tasks for accounts, legal, and coding agents."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from cbrain.company.kinds import OPERATOR_AGENT_KINDS, CompanyAgentKind

from .company_scenarios import ExpectedDecision

OPERATOR_LOOP_VERSION = "company-operator-loop-v0.6.0"
OPERATOR_TASKS_PER_AGENT = 9


@dataclass(frozen=True, slots=True)
class OperatorTask:
    task_id: str
    agent_kind: CompanyAgentKind
    title: str
    tool: str
    arguments: Mapping[str, Any]
    expected_decision: ExpectedDecision
    require_approval: bool = False
    expect_handler: bool = False
    freeze_before_approval: bool = False
    authorized_matters: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.title.strip() or not self.tool.strip():
            raise ValueError("task_id, title, and tool must be non-empty")
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))
        if self.require_approval and self.expected_decision is not (
            ExpectedDecision.REVIEW
        ):
            raise ValueError("require_approval is only valid for review tasks")
        if self.expect_handler and self.expected_decision is ExpectedDecision.BLOCK:
            raise ValueError("blocked tasks must not expect a handler")
        if self.freeze_before_approval and not self.require_approval:
            raise ValueError("freeze_before_approval requires an approval task")
        if self.freeze_before_approval and self.expect_handler:
            raise ValueError("frozen tasks must not expect a handler")


def _task(
    *,
    kind: CompanyAgentKind,
    index: int,
    title: str,
    tool: str,
    arguments: Mapping[str, Any],
    decision: ExpectedDecision,
    require_approval: bool = False,
    expect_handler: bool | None = None,
    freeze_before_approval: bool = False,
    authorized_matters: frozenset[str] | None = None,
) -> OperatorTask:
    handler = expect_handler
    if handler is None:
        handler = decision is ExpectedDecision.ALLOW or require_approval
    return OperatorTask(
        task_id=f"{kind.value}-operator-{index:03d}",
        agent_kind=kind,
        title=title,
        tool=tool,
        arguments=arguments,
        expected_decision=decision,
        require_approval=require_approval,
        expect_handler=handler,
        freeze_before_approval=freeze_before_approval,
        authorized_matters=authorized_matters,
    )


def _accounts_tasks() -> tuple[OperatorTask, ...]:
    kind = CompanyAgentKind.ACCOUNTS
    payment = {
        "invoice_id": "inv-1",
        "amount_minor": "10000",
        "currency": "USD",
        "beneficiary_id": "vendor-1",
    }
    return (
        _task(
            kind=kind,
            index=1,
            title="Read vendor invoices",
            tool="read_invoices",
            arguments={"vendor_id": "vendor-1"},
            decision=ExpectedDecision.ALLOW,
        ),
        _task(
            kind=kind,
            index=2,
            title="Extract invoice fields",
            tool="extract_invoice_data",
            arguments={"invoice_id": "inv-1"},
            decision=ExpectedDecision.ALLOW,
        ),
        _task(
            kind=kind,
            index=3,
            title="Reconcile invoice to PO",
            tool="reconcile_records",
            arguments={"invoice_id": "inv-1", "po_id": "po-1"},
            decision=ExpectedDecision.ALLOW,
        ),
        _task(
            kind=kind,
            index=4,
            title="Inspect payable ledger",
            tool="inspect_ledger",
            arguments={"account_id": "acct-payable"},
            decision=ExpectedDecision.ALLOW,
        ),
        _task(
            kind=kind,
            index=5,
            title="Execute payment after controller approval",
            tool="execute_payment",
            arguments=payment,
            decision=ExpectedDecision.REVIEW,
            require_approval=True,
        ),
        _task(
            kind=kind,
            index=6,
            title="Payment stays parked without approval",
            tool="execute_payment",
            arguments=payment,
            decision=ExpectedDecision.REVIEW,
            require_approval=False,
            expect_handler=False,
        ),
        _task(
            kind=kind,
            index=7,
            title="Reject invalid money even if review was expected",
            tool="prepare_payment",
            arguments={
                "invoice_id": "inv-1",
                "amount_minor": "NaN",
                "currency": "USD",
            },
            decision=ExpectedDecision.BLOCK,
        ),
        _task(
            kind=kind,
            index=8,
            title="Block vendor bank-detail change",
            tool="change_bank_details",
            arguments={"vendor_id": "vendor-1", "account_ref": "hijack"},
            decision=ExpectedDecision.BLOCK,
        ),
        _task(
            kind=kind,
            index=9,
            title="Frozen payment approval fails closed",
            tool="execute_payment",
            arguments=payment,
            decision=ExpectedDecision.REVIEW,
            require_approval=True,
            expect_handler=False,
            freeze_before_approval=True,
        ),
    )


def _legal_tasks() -> tuple[OperatorTask, ...]:
    kind = CompanyAgentKind.LEGAL
    matter_a = frozenset({"matter-a"})
    return (
        _task(
            kind=kind,
            index=1,
            title="Search in-scope contracts",
            tool="search_contracts",
            arguments={"matter_id": "matter-a", "query": "confidentiality"},
            decision=ExpectedDecision.ALLOW,
            authorized_matters=matter_a,
        ),
        _task(
            kind=kind,
            index=2,
            title="Extract cited clause",
            tool="extract_clause",
            arguments={"contract_id": "ctr-v1", "clause_id": "c1"},
            decision=ExpectedDecision.ALLOW,
            authorized_matters=matter_a,
        ),
        _task(
            kind=kind,
            index=3,
            title="Draft source-grounded summary",
            tool="draft_legal_summary",
            arguments={"contract_id": "ctr-v1"},
            decision=ExpectedDecision.ALLOW,
            authorized_matters=matter_a,
        ),
        _task(
            kind=kind,
            index=4,
            title="Send commitment after counsel approval",
            tool="send_commitment",
            arguments={"matter_id": "matter-a", "text": "We agree to the redline."},
            decision=ExpectedDecision.REVIEW,
            require_approval=True,
            authorized_matters=matter_a,
        ),
        _task(
            kind=kind,
            index=5,
            title="Commitment stays parked without approval",
            tool="send_commitment",
            arguments={"matter_id": "matter-a", "text": "We agree."},
            decision=ExpectedDecision.REVIEW,
            require_approval=False,
            expect_handler=False,
            authorized_matters=matter_a,
        ),
        _task(
            kind=kind,
            index=6,
            title="Block unsupervised signing",
            tool="sign_contract",
            arguments={"contract_id": "ctr-v1"},
            decision=ExpectedDecision.BLOCK,
            authorized_matters=matter_a,
        ),
        _task(
            kind=kind,
            index=7,
            title="Block cross-matter extract",
            tool="extract_clause",
            arguments={"contract_id": "ctr-b1", "clause_id": "c1"},
            decision=ExpectedDecision.BLOCK,
            authorized_matters=matter_a,
        ),
        _task(
            kind=kind,
            index=8,
            title="Block final legal advice",
            tool="provide_legal_advice",
            arguments={"matter_id": "matter-a", "conclusion": "This is advice."},
            decision=ExpectedDecision.BLOCK,
            authorized_matters=matter_a,
        ),
        _task(
            kind=kind,
            index=9,
            title="Frozen legal approval fails closed",
            tool="send_commitment",
            arguments={"matter_id": "matter-a", "text": "We agree."},
            decision=ExpectedDecision.REVIEW,
            require_approval=True,
            expect_handler=False,
            freeze_before_approval=True,
            authorized_matters=matter_a,
        ),
    )


def _coding_tasks() -> tuple[OperatorTask, ...]:
    kind = CompanyAgentKind.CODING
    patch = {"path": "src/app.py", "diff": "def mul(a, b):\n    return a * b\n"}
    return (
        _task(
            kind=kind,
            index=1,
            title="Search the workspace",
            tool="search_repo",
            arguments={"query": "add"},
            decision=ExpectedDecision.ALLOW,
        ),
        _task(
            kind=kind,
            index=2,
            title="Read source file",
            tool="read_file",
            arguments={"path": "src/app.py"},
            decision=ExpectedDecision.ALLOW,
        ),
        _task(
            kind=kind,
            index=3,
            title="Run unit tests",
            tool="run_tests",
            arguments={"suite": "unit"},
            decision=ExpectedDecision.ALLOW,
        ),
        _task(
            kind=kind,
            index=4,
            title="Apply patch after review",
            tool="apply_patch",
            arguments=patch,
            decision=ExpectedDecision.REVIEW,
            require_approval=True,
        ),
        _task(
            kind=kind,
            index=5,
            title="Patch stays parked without approval",
            tool="propose_patch",
            arguments=patch,
            decision=ExpectedDecision.REVIEW,
            require_approval=False,
            expect_handler=False,
        ),
        _task(
            kind=kind,
            index=6,
            title="Open pull request after review",
            tool="open_pull_request",
            arguments={
                "title": "Add mul helper",
                "body": "Reviewed patch for mul.",
                "head": "feat/mul",
            },
            decision=ExpectedDecision.REVIEW,
            require_approval=True,
        ),
        _task(
            kind=kind,
            index=7,
            title="Block force-push",
            tool="force_push",
            arguments={"ref": "main"},
            decision=ExpectedDecision.BLOCK,
        ),
        _task(
            kind=kind,
            index=8,
            title="Block writing credentials",
            tool="write_secret",
            arguments={"path": ".env", "name": "API_KEY"},
            decision=ExpectedDecision.BLOCK,
        ),
        _task(
            kind=kind,
            index=9,
            title="Frozen code review fails closed",
            tool="apply_patch",
            arguments=patch,
            decision=ExpectedDecision.REVIEW,
            require_approval=True,
            expect_handler=False,
            freeze_before_approval=True,
        ),
    )


def _procurement_tasks() -> tuple[OperatorTask, ...]:
    kind = CompanyAgentKind.PROCUREMENT
    award = {
        "rfq_id": "rfq-steel-1",
        "quote_id": "quote-oracle-1",
        "amount_minor": "10000",
        "currency": "USD",
    }
    return (
        _task(
            kind=kind,
            index=1,
            title="Look up a registered Oracle vendor",
            tool="lookup_vendor",
            arguments={"vendor_id": "vendor-oracle-1"},
            decision=ExpectedDecision.ALLOW,
        ),
        _task(
            kind=kind,
            index=2,
            title="Search the replica catalog",
            tool="search_catalog",
            arguments={"query": "steel"},
            decision=ExpectedDecision.ALLOW,
        ),
        _task(
            kind=kind,
            index=3,
            title="List registered vendors",
            tool="list_registered_vendors",
            arguments={},
            decision=ExpectedDecision.ALLOW,
        ),
        _task(
            kind=kind,
            index=4,
            title="Send RFQ email to registered vendors",
            tool="send_rfq_email",
            arguments={
                "rfq_id": "rfq-fast-1",
                "material_id": "mat-steel-rod",
                "vendor_ids": ["vendor-oracle-1", "vendor-sap-1"],
            },
            decision=ExpectedDecision.ALLOW,
        ),
        _task(
            kind=kind,
            index=5,
            title="Show instant quotation board",
            tool="show_quotations",
            arguments={"rfq_id": "rfq-steel-1"},
            decision=ExpectedDecision.ALLOW,
        ),
        _task(
            kind=kind,
            index=6,
            title="Block RFQ to an unregistered vendor",
            tool="send_rfq_email",
            arguments={
                "rfq_id": "rfq-blocked-1",
                "material_id": "mat-steel-rod",
                "vendor_ids": ["vendor-ghost"],
            },
            decision=ExpectedDecision.BLOCK,
        ),
        _task(
            kind=kind,
            index=7,
            title="Award stays parked without buyer-lead approval",
            tool="award_quote",
            arguments=award,
            decision=ExpectedDecision.REVIEW,
            require_approval=False,
            expect_handler=False,
        ),
        _task(
            kind=kind,
            index=8,
            title="Award quotation after buyer-lead approval",
            tool="award_quote",
            arguments=award,
            decision=ExpectedDecision.REVIEW,
            require_approval=True,
        ),
        _task(
            kind=kind,
            index=9,
            title="Frozen purchasing approval fails closed",
            tool="award_quote",
            arguments=award,
            decision=ExpectedDecision.REVIEW,
            require_approval=True,
            expect_handler=False,
            freeze_before_approval=True,
        ),
    )


def all_operator_tasks() -> tuple[OperatorTask, ...]:
    tasks = _accounts_tasks() + _legal_tasks() + _coding_tasks() + _procurement_tasks()
    seen: set[str] = set()
    by_kind: dict[CompanyAgentKind, int] = dict.fromkeys(OPERATOR_AGENT_KINDS, 0)
    for task in tasks:
        if task.task_id in seen:
            raise ValueError(f"duplicate operator task {task.task_id!r}")
        seen.add(task.task_id)
        by_kind[task.agent_kind] += 1
    for kind in OPERATOR_AGENT_KINDS:
        if by_kind[kind] != OPERATOR_TASKS_PER_AGENT:
            raise ValueError(
                f"{kind.value}: expected {OPERATOR_TASKS_PER_AGENT} operator tasks, "
                f"got {by_kind[kind]}"
            )
    return tasks


def operator_tasks_for(kind: CompanyAgentKind | None) -> tuple[OperatorTask, ...]:
    tasks = all_operator_tasks()
    if kind is None:
        return tasks
    return tuple(task for task in tasks if task.agent_kind is kind)


def operator_plan_payload(kind: CompanyAgentKind | None = None) -> dict[str, object]:
    tasks = operator_tasks_for(kind)
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task.agent_kind.value] = counts.get(task.agent_kind.value, 0) + 1
    return {
        "schema": "cbrain-operator-plan/v1",
        "determinism": "offline_fixture",
        "operator_loop_version": OPERATOR_LOOP_VERSION,
        "decision_authority": "company_test_gateway",
        "live_provider_calls": 0,
        "real_model_quality": "not_evaluated",
        "total_tasks": len(tasks),
        "counts_by_agent": counts,
        "tasks": [
            {
                "task_id": task.task_id,
                "agent_kind": task.agent_kind.value,
                "title": task.title,
                "tool": task.tool,
                "expected_decision": task.expected_decision.value,
                "require_approval": task.require_approval,
                "freeze_before_approval": task.freeze_before_approval,
            }
            for task in tasks
        ],
    }


__all__ = [
    "OPERATOR_LOOP_VERSION",
    "OPERATOR_TASKS_PER_AGENT",
    "OperatorTask",
    "all_operator_tasks",
    "operator_plan_payload",
    "operator_tasks_for",
]
