"""Adversarial regressions for procurement extract and proof hardening."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

import pytest
from operator_fixtures import APPROVER_DIRECTORY, BUYER_LEAD_PRINCIPAL

from cbrain import ExecutionStatus, GovernedRuntime
from cbrain.adapters.privatevault_execution import (
    AgentDNAVerifiers,
    ExecutionAuthorizationBinding,
    PrivateVaultAgentDNAUnavailable,
    PrivateVaultAgentDNAVerifier,
    PrivateVaultEvidenceRejected,
    PrivateVaultExecutionError,
)
from cbrain.company.approval import (
    ApprovalBoundedGateway,
    ApprovalInbox,
    ApprovalRole,
)
from cbrain.company.governance import CompanyRiskGateway
from cbrain.company.handlers import build_handlers
from cbrain.company.kinds import CompanyAgentKind
from cbrain.company.profiles import spec_for_kind
from cbrain.company.risk import ToolRiskLevel, risk_for_tool
from cbrain.company.simulators import load_fixture_bundle
from cbrain.company.spec import CompanyAgentSpec, build_company_spec
from cbrain.contracts import ActionIntent, GovernedExecution
from cbrain.dispatch import PreparedDispatch
from cbrain.evaluation.company_harness import canonical_action_intent_digest
from cbrain.procurement import (
    BUYER_PROFILE,
    VENDOR_ONBOARDING_PROFILE,
    ErpSystem,
    ProcurementError,
    build_privatevault_procurement_proof,
    build_procurement_proof,
)
from cbrain.procurement.erp import parse_json_boolean, records_from_mapping

NOW = "2026-07-31T12:00:30Z"
ZERO = "sha256:" + ("0" * 64)
ONE = "sha256:" + ("1" * 64)
Call = tuple[tuple[object, ...], dict[str, object]]
Verifier = Callable[..., object]


@dataclass(frozen=True, slots=True)
class Report:
    ok: bool = True
    evidence_state: str = "VERIFIED"
    decision_conformance: str = "CONFORMANT"
    reason_code: str = "VALID"
    accountable_principal: str | None = "component@example"


def _action(
    spec: CompanyAgentSpec, tool: str, arguments: dict[str, object]
) -> ActionIntent:
    return ActionIntent.capture(
        agent_id=spec.agent_id,
        framework="test",
        tool_name=tool,
        capability=spec.tools.get(tool).capability,
        arguments=arguments,
    )


def _vendor_row(registered: object) -> dict[str, object]:
    row: dict[str, object] = {
        "vendor_id": "vendor-coerced-1",
        "name": "Coerced Steel",
        "email": "coerced@example.com",
    }
    if registered is not _MISSING:
        row["registered"] = registered
    return row


_MISSING = object()


@pytest.mark.parametrize("value", ["false", "0", 1, None, _MISSING])
def test_extract_registered_rejects_non_json_booleans(value: object) -> None:
    if value is _MISSING:
        row = _vendor_row(_MISSING)
        assert "registered" not in row
    else:
        row = _vendor_row(value)
    with pytest.raises(ProcurementError, match="JSON boolean"):
        records_from_mapping({"vendors": [row]}, system=ErpSystem.ORACLE)


@pytest.mark.parametrize("value", ["false", "0", 1, None])
def test_parse_json_boolean_rejects_coercible_impostors(value: object) -> None:
    with pytest.raises(ProcurementError, match="JSON boolean"):
        parse_json_boolean(value, field_name="registered")


def test_parse_json_boolean_accepts_only_actual_bools() -> None:
    assert parse_json_boolean(False, field_name="registered") is False
    assert parse_json_boolean(True, field_name="registered") is True


def test_json_false_does_not_register_a_vendor() -> None:
    replica = records_from_mapping(
        {
            "vendors": [
                {
                    "vendor_id": "vendor-json-false",
                    "name": "JSON False Co",
                    "email": "false@example.com",
                    "registered": False,
                }
            ]
        },
        system=ErpSystem.ORACLE,
    )
    assert replica.vendors()[0].registered is False


def _award_executed() -> tuple[
    CompanyAgentSpec,
    ActionIntent,
    GovernedExecution,
    dict[str, Any],
]:
    spec = spec_for_kind(CompanyAgentKind.PROCUREMENT)
    bundle = load_fixture_bundle("default")
    inner = CompanyRiskGateway(spec, bundle=bundle)
    inbox = ApprovalInbox(
        allowed_approvers=APPROVER_DIRECTORY,
        max_ttl_seconds=3600.0,
    )
    gateway = ApprovalBoundedGateway(
        inner,
        inbox,
        digest_for=canonical_action_intent_digest,
        required_role_for=lambda _: ApprovalRole.BUYER_LEAD,
    )
    handlers = build_handlers(kind=CompanyAgentKind.PROCUREMENT, bundle=bundle)
    action = _action(
        spec,
        "award_quote",
        {
            "rfq_id": "rfq-steel-1",
            "quote_id": "quote-oracle-1",
            "amount_minor": "10000",
            "currency": "USD",
        },
    )
    first = GovernedRuntime(gateway).execute(action, handlers["award_quote"])
    assert first.status is ExecutionStatus.REVIEW_REQUIRED
    inbox.approve(action.request_id, principal=BUYER_LEAD_PRINCIPAL)
    final = GovernedRuntime(gateway).execute(action, handlers["award_quote"])
    assert final.status is ExecutionStatus.EXECUTED
    board = handlers["show_quotations"]({"rfq_id": "rfq-steel-1"})
    return spec, action, final, board


def test_offline_proof_rejects_non_executed_statuses() -> None:
    spec, action, final, board = _award_executed()
    for status, tool_executed in (
        (ExecutionStatus.BLOCKED, False),
        (ExecutionStatus.REVIEW_REQUIRED, False),
        (ExecutionStatus.CONTROL_FAILURE, False),
        (ExecutionStatus.INDETERMINATE, None),
    ):
        failed = GovernedExecution(
            status=status,
            request_id=action.request_id,
            tool_executed=tool_executed,
            reason="not-executed",
            output=final.output,
        )
        with pytest.raises(ProcurementError, match="EXECUTED"):
            build_procurement_proof(action=action, execution=failed, quote_board=board)


def test_offline_proof_rejects_field_and_board_mismatches() -> None:
    spec, action, final, board = _award_executed()
    swapped = deepcopy(dict(board))
    swapped["quotations"] = deepcopy(list(board["quotations"]))
    swapped["quotations"][0] = dict(swapped["quotations"][0])
    swapped["quotations"][0]["amount_minor"] = "1"
    with pytest.raises(ProcurementError, match="quote-board digest"):
        build_procurement_proof(action=action, execution=final, quote_board=swapped)

    other = ActionIntent.capture(
        agent_id=spec.agent_id,
        framework="test",
        tool_name="award_quote",
        capability=spec.tools.get("award_quote").capability,
        arguments={
            "rfq_id": "rfq-steel-1",
            "quote_id": "quote-sap-1",
            "amount_minor": "11000",
            "currency": "USD",
        },
        request_id=action.request_id,
    )
    with pytest.raises(ProcurementError, match="does not match"):
        build_procurement_proof(action=other, execution=final, quote_board=board)


def test_offline_proof_recomputes_digest_and_ignores_caller_authority() -> None:
    _, action, final, board = _award_executed()
    proof = build_procurement_proof(action=action, execution=final, quote_board=board)
    assert proof.action_intent_digest == canonical_action_intent_digest(action)
    assert proof.decision_authority == "company_test_gateway"
    assert proof.decision_receipt_digest is None
    with pytest.raises(TypeError):
        build_procurement_proof(
            action=action,
            execution=final,
            quote_board=board,
            action_intent_digest="deadbeef",
        )


def test_rfq_idempotency_replays_identical_payload_and_blocks_conflicts() -> None:
    spec = spec_for_kind(CompanyAgentKind.PROCUREMENT)
    bundle = load_fixture_bundle("default")
    gateway = CompanyRiskGateway(spec, bundle=bundle)
    handlers = build_handlers(kind=CompanyAgentKind.PROCUREMENT, bundle=bundle)
    arguments = {
        "rfq_id": "rfq-idem-1",
        "material_id": "mat-steel-rod",
        "vendor_ids": ["vendor-oracle-1", "vendor-sap-1"],
    }
    first = gateway.decide_and_execute(
        _action(spec, "send_rfq_email", arguments),
        handlers["send_rfq_email"],
    )
    assert first.status is ExecutionStatus.EXECUTED
    outbound_after_first = len(bundle.procurement.outbound_rfqs)
    second = gateway.decide_and_execute(
        _action(spec, "send_rfq_email", arguments),
        handlers["send_rfq_email"],
    )
    assert second.status is ExecutionStatus.EXECUTED
    assert second.output == first.output
    assert len(bundle.procurement.outbound_rfqs) == outbound_after_first
    conflict = gateway.decide_and_execute(
        _action(
            spec,
            "send_rfq_email",
            {
                "rfq_id": "rfq-idem-1",
                "material_id": "mat-steel-rod",
                "vendor_ids": ["vendor-oracle-1", "vendor-sql-1"],
            },
        ),
        handlers["send_rfq_email"],
    )
    assert conflict.status is ExecutionStatus.BLOCKED
    assert conflict.tool_executed is False
    assert conflict.reason == "PROCUREMENT_RFQ_IDEMPOTENCY_CONFLICT"


def test_prohibited_bank_and_payment_tools_remain_blocked_when_readded() -> None:
    assert "change_vendor_bank" not in BUYER_PROFILE.permitted_tools
    assert "post_erp_payment" not in VENDOR_ONBOARDING_PROFILE.permitted_tools
    assert (
        risk_for_tool(CompanyAgentKind.PROCUREMENT, "change_vendor_bank")
        is ToolRiskLevel.BLOCK
    )
    assert (
        risk_for_tool(CompanyAgentKind.PROCUREMENT, "post_erp_payment")
        is ToolRiskLevel.BLOCK
    )
    profile = replace(
        BUYER_PROFILE,
        agent_id="company-procurement-adversarial-v0.6",
        permitted_tools=BUYER_PROFILE.permitted_tools
        | frozenset({"change_vendor_bank", "post_erp_payment"}),
    )
    spec = build_company_spec(
        kind=CompanyAgentKind.PROCUREMENT,
        profile=profile,
        required_simulator="procurement_erp_simulator",
    )
    bundle = load_fixture_bundle("default")
    gateway = CompanyRiskGateway(spec, bundle=bundle)
    handlers = build_handlers(kind=CompanyAgentKind.PROCUREMENT, bundle=bundle)
    bank = gateway.decide_and_execute(
        _action(
            spec,
            "change_vendor_bank",
            {"vendor_id": "vendor-oracle-1", "account_ref": "bank"},
        ),
        handlers["change_vendor_bank"],
    )
    payment = gateway.decide_and_execute(
        _action(
            spec,
            "post_erp_payment",
            {
                "vendor_id": "vendor-oracle-1",
                "amount_minor": "10000",
                "currency": "USD",
            },
        ),
        handlers["post_erp_payment"],
    )
    assert bank.status is ExecutionStatus.BLOCKED
    assert payment.status is ExecutionStatus.BLOCKED
    assert bank.tool_executed is False
    assert payment.tool_executed is False
    simulated = handlers["change_vendor_bank"](
        {"vendor_id": "vendor-oracle-1", "account_ref": "bank"}
    )
    assert simulated["delivery_mode"] == "simulated"
    assert simulated["live_erp_posted"] is False
    assert simulated["privatevault_authorized"] is False
    assert "changed" not in simulated


def _recording_verifier(
    *,
    stage: str,
    report: object,
    calls: dict[str, list[Call]],
) -> Verifier:
    def invoke(*args: object, **kwargs: object) -> object:
        calls[stage].append((args, kwargs))
        return report

    return invoke


def _recording_adapter() -> PrivateVaultAgentDNAVerifier:
    calls: dict[str, list[Call]] = {
        "authorization": [],
        "dispatch": [],
        "closure": [],
    }
    verifiers = AgentDNAVerifiers(
        verify_execution_authorization=_recording_verifier(
            stage="authorization",
            report=Report(),
            calls=calls,
        ),
        verify_dispatch_witness=_recording_verifier(
            stage="dispatch",
            report=Report(),
            calls=calls,
        ),
        verify_closure_chain=_recording_verifier(
            stage="closure",
            report=Report(),
            calls=calls,
        ),
    )
    return PrivateVaultAgentDNAVerifier(verifiers)


def _verified_closure_for(
    action: ActionIntent,
    *,
    decision_id: str,
    verifier: PrivateVaultAgentDNAVerifier,
) -> Any:
    prepared = PreparedDispatch.capture(
        request_id=action.request_id,
        dispatch={
            "transport": "https",
            "destination": "procurement.example",
            "operation": "POST /v1/award",
            "wire_content_type": "application/json",
            "wire_content_encoding": "identity",
            "tool_id": "procurement.v1",
            "tool_schema_digest": ZERO,
            "tool_artifact_digest": ONE,
            "credential_audience": "procurement.example",
            "idempotency_key_digest": ZERO,
            "retry_policy_digest": ONE,
        },
        wire_bytes=b'{"award":true}',
        peer_identity_bytes=b"tls-spki:procurement.example:v1",
    )
    binding = ExecutionAuthorizationBinding.capture(
        request_id=action.request_id,
        action=action.privatevault_decide_payload(),
        prepared_dispatch=prepared,
        decision_receipt_digest=ZERO,
        authority_receipt_digest=ONE,
        approval_artifact_digest=ZERO,
        state_snapshot_digest=ZERO,
        policy_bundle_digest=ONE,
        obligations_digest=ONE,
        at_time=NOW,
    )
    authorization = verifier.verify_authorization(
        authorization={
            "decision_id": decision_id,
            "expires_at": "2099-01-01T00:00:00Z",
            "signed": "authorization",
        },
        trust_bundle={"signed": "trust-bundle"},
        binding=binding,
        already_consumed=False,
    )
    dispatch = verifier.verify_dispatch(
        verified_authorization=authorization,
        witness={"signed": "witness"},
    )
    return verifier.verify_closure(
        verified_dispatch=dispatch,
        closure={"signed": "closure"},
    )


def test_privatevault_proof_requires_verified_closure_and_rechecks_evidence() -> None:
    _, action, final, board = _award_executed()
    verifier = _recording_adapter()
    pv_execution = GovernedExecution(
        status=ExecutionStatus.EXECUTED,
        request_id=action.request_id,
        tool_executed=True,
        reason="allow",
        decision_id="dec-pv-1",
        output=final.output,
    )
    closure = _verified_closure_for(action, decision_id="dec-pv-1", verifier=verifier)
    proof = build_privatevault_procurement_proof(
        action=action,
        execution=pv_execution,
        quote_board=board,
        verified_closure=closure,
        verifier=verifier,
    )
    payload = proof.to_payload()
    assert payload["decision_authority"] == "privatevault"
    assert payload["decision_id"] == "dec-pv-1"
    assert payload["decision_receipt_digest"] == ZERO
    assert payload["authority_receipt_digest"] == ONE
    assert payload["action_intent_digest"] == canonical_action_intent_digest(action)
    with pytest.raises(TypeError):
        build_privatevault_procurement_proof(
            action=action,
            execution=pv_execution,
            quote_board=board,
            verified_closure=closure,
            verifier=verifier,
            decision_receipt_digest=ZERO,
        )
    with pytest.raises(ProcurementError, match="VerifiedClosure"):
        build_privatevault_procurement_proof(
            action=action,
            execution=pv_execution,
            quote_board=board,
            verified_closure=object(),  # type: ignore[arg-type]
            verifier=verifier,
        )


def test_privatevault_proof_rejects_decision_and_action_mismatch() -> None:
    _, action, final, board = _award_executed()
    verifier = _recording_adapter()
    closure = _verified_closure_for(action, decision_id="dec-pv-1", verifier=verifier)
    mismatched = GovernedExecution(
        status=ExecutionStatus.EXECUTED,
        request_id=action.request_id,
        tool_executed=True,
        reason="allow",
        decision_id="dec-other",
        output=final.output,
    )
    with pytest.raises(ProcurementError, match="decision_id"):
        build_privatevault_procurement_proof(
            action=action,
            execution=mismatched,
            quote_board=board,
            verified_closure=closure,
            verifier=verifier,
        )
    other = ActionIntent.capture(
        agent_id=action.agent_id,
        framework=action.framework,
        tool_name=action.tool_name,
        capability=action.capability,
        arguments=dict(action.arguments),
    )
    with pytest.raises(ProcurementError, match="request_id|action digest"):
        build_privatevault_procurement_proof(
            action=other,
            execution=GovernedExecution(
                status=ExecutionStatus.EXECUTED,
                request_id=other.request_id,
                tool_executed=True,
                reason="allow",
                decision_id="dec-pv-1",
                output=final.output,
            ),
            quote_board=board,
            verified_closure=closure,
            verifier=verifier,
        )


def test_forged_privatevault_closure_fails_real_agent_dna() -> None:
    try:
        AgentDNAVerifiers.load()
    except PrivateVaultAgentDNAUnavailable:
        pytest.skip("pinned PrivateVault Agent DNA is unavailable")
    _, action, final, board = _award_executed()
    recording = _recording_adapter()
    closure = _verified_closure_for(action, decision_id="dec-pv-1", verifier=recording)
    pv_execution = GovernedExecution(
        status=ExecutionStatus.EXECUTED,
        request_id=action.request_id,
        tool_executed=True,
        reason="allow",
        decision_id="dec-pv-1",
        output=final.output,
    )
    with pytest.raises(
        (
            PrivateVaultEvidenceRejected,
            PrivateVaultExecutionError,
            ProcurementError,
        )
    ):
        build_privatevault_procurement_proof(
            action=action,
            execution=pv_execution,
            quote_board=board,
            verified_closure=closure,
        )
