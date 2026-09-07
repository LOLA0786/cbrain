"""Phase 2: trusted security-event evidence under PV_LOOP_EVENTS_REQUIRED."""

from __future__ import annotations

from typing import Any

import pytest
from pinned_privatevault_server import pinned_privatevault_server
from test_execution_gateway_conformance import MemoryConsumptionStore
from test_privatevault_http_contract import (
    ARGUMENTS,
    BUYER,
    ORG,
    PO_CREATE,
    Harness,
    evidence,
    intent,
    now_rfc3339,
)

from cbrain import ExecutionStatus, GovernedRuntime
from cbrain.execution.authorize_client import (
    AuthorizationRefused,
    PrivateVaultAuthorizationClient,
)
from cbrain.execution.security_events import SealedDecisionInvokeEvidence


@pytest.fixture()
def loop_pv(tmp_path, monkeypatch):
    with pinned_privatevault_server(
        tmp_path,
        monkeypatch,
        organisation_id=ORG,
        grants={
            BUYER: (PO_CREATE,),
            "other-agent": (PO_CREATE,),
        },
        blocked_capabilities=(),
        loop_events_required=True,
    ) as server:
        yield server


def test_loop_events_required_refuses_mint_without_security_events(loop_pv):
    """Reproduce the Phase 2 gap: bound ALLOW, but authorize omits events."""
    harness = Harness(loop_pv)
    # Issuer without a trusted provider — matches pre-fix CBrain.
    harness.issuer = PrivateVaultAuthorizationClient(
        harness.transport,
        organisation_id=ORG,
        evidence_digests=evidence(),
        require_security_events=False,
        security_event_provider=None,
    )
    action = intent()
    decision, planned = harness.bound_decision(action)
    assert decision.verdict.value == "allow"

    with pytest.raises(
        AuthorizationRefused,
        match="AUTHORIZE_LOOP_EVENTS_REQUIRED",
    ):
        harness.issuer.issue(action=action, decision=decision, planned=planned)

    body = harness.authorize_body(
        decision.record, planned, request_id=action.request_id
    )
    response = harness.raw_authorize(body)
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["reason_code"] == "AUTHORIZE_LOOP_EVENTS_REQUIRED"


def test_trusted_sealed_decision_events_allow_mint_under_loop_gate(loop_pv):
    harness = Harness(loop_pv)
    harness.issuer = PrivateVaultAuthorizationClient(
        harness.transport,
        organisation_id=ORG,
        evidence_digests=evidence(),
        require_security_events=True,
        security_event_provider=SealedDecisionInvokeEvidence(clock=now_rfc3339),
    )
    action = intent()
    decision, planned = harness.bound_decision(action)
    issued = harness.issuer.issue(action=action, decision=decision, planned=planned)
    assert issued.authorization["max_uses"] == 1
    assert issued.authorization["action"] == planned.action

    # Direct HTTP with the same trusted event shape also mints.
    body = harness.authorize_body(
        decision.record, planned, request_id=f"{action.request_id}-http"
    )
    events = SealedDecisionInvokeEvidence(clock=now_rfc3339).events_for_authorize(
        action=action, decision=decision, planned=planned
    )
    # Fresh event_id so mint claim key stays the decision; events batch is new.
    body["security_events"] = [
        {**dict(events[0]), "event_id": f"invoke-{action.request_id}-http"}
    ]
    # Same decision already minted — expect replay of identical permit or
    # binding conflict if request_id differs. Use the same request_id.
    body["request_id"] = action.request_id
    body["security_events"] = [dict(events[0])]
    response = harness.raw_authorize(body)
    assert response.status_code == 200, response.text
    assert "authorization" in response.json()


def test_missing_trusted_provider_refuses_before_http_when_required(loop_pv):
    harness = Harness(loop_pv)
    harness.issuer = PrivateVaultAuthorizationClient(
        harness.transport,
        organisation_id=ORG,
        evidence_digests=evidence(),
        require_security_events=True,
        security_event_provider=None,
    )
    action = intent()
    decision, planned = harness.bound_decision(action)
    with pytest.raises(AuthorizationRefused, match="no trusted evidence provider"):
        harness.issuer.issue(action=action, decision=decision, planned=planned)
    assert harness.transport.paths() == ["/v1/decide"]


def test_circular_security_events_refuse_mint(loop_pv):
    harness = Harness(loop_pv)
    action = intent()
    decision, planned = harness.bound_decision(action)
    body = harness.authorize_body(
        decision.record, planned, request_id=action.request_id
    )
    digest = decision.record["action_digest"]
    body["security_events"] = [
        {
            "spec": "pv-agent-security-event/1.0",
            "event_id": "a",
            "trace_id": "trace-loop",
            "occurred_at": "2026-09-07T15:00:00Z",
            "source_agent_id": "agent-a",
            "target_agent_id": "agent-b",
            "relation": "DELEGATES",
            "action_digest": digest,
            "authorization_id": "auth-a",
            "authorization_state": "VERIFIED",
            "parent_event_id": None,
        },
        {
            "spec": "pv-agent-security-event/1.0",
            "event_id": "b",
            "trace_id": "trace-loop",
            "occurred_at": "2026-09-07T15:00:01Z",
            "source_agent_id": "agent-b",
            "target_agent_id": "agent-a",
            "relation": "APPROVES",
            "action_digest": digest,
            "authorization_id": "auth-b",
            "authorization_state": "VERIFIED",
            "parent_event_id": "a",
        },
    ]
    response = harness.raw_authorize(body)
    assert response.status_code == 403, response.text
    detail = response.json()["detail"]
    assert detail["triggered_by"] == "loop_discovery"
    assert detail["decision"] in ("BLOCK", "REVIEW")
    assert "authorization" not in response.json()


def test_governed_runtime_executes_with_trusted_loop_evidence(loop_pv):
    harness = Harness(loop_pv)
    harness.issuer = PrivateVaultAuthorizationClient(
        harness.transport,
        organisation_id=ORG,
        evidence_digests=evidence(),
        require_security_events=True,
        security_event_provider=SealedDecisionInvokeEvidence(clock=now_rfc3339),
    )
    store = MemoryConsumptionStore()
    calls: list[Any] = []
    result = GovernedRuntime(harness.gateway(store, calls)).execute(
        intent(request_id="req-loop-exec-001"),
        lambda arguments: calls.append(arguments),
    )
    assert result.status is ExecutionStatus.EXECUTED, result.reason
    assert calls == [ARGUMENTS]


@pytest.fixture()
def secure_pv(tmp_path, monkeypatch):
    with pinned_privatevault_server(
        tmp_path,
        monkeypatch,
        organisation_id=ORG,
        grants={BUYER: (PO_CREATE,)},
        loop_events_required=True,
        secure_profile=True,
    ) as server:
        yield server


def test_secure_profile_mints_with_signed_receipt_and_trusted_events(secure_pv):
    """Fully provisioned PV_SECURE_PROFILE=1: signed decide + evidenced mint."""
    from agent_dna.signer import verify_trusted_envelope

    harness = Harness(secure_pv)
    harness.issuer = PrivateVaultAuthorizationClient(
        harness.transport,
        organisation_id=ORG,
        evidence_digests=evidence(),
        require_security_events=True,
        security_event_provider=SealedDecisionInvokeEvidence(clock=now_rfc3339),
    )
    action = intent(request_id="req-secure-001")
    decision, planned = harness.bound_decision(action)
    assert decision.verdict.value == "allow"
    record = decision.record
    envelope_response = harness.server.client.get(
        f"/v1/envelope/{record['record_hash']}",
        headers={"X-API-Key": harness.server.api_keys[BUYER]},
    )
    assert envelope_response.status_code == 200, envelope_response.text
    envelope = envelope_response.json()["envelope"]
    assert verify_trusted_envelope(
        envelope,
        record["record_hash"],
        trusted_keys=frozenset({envelope["public_key"]}),
    )

    issued = harness.issuer.issue(action=action, decision=decision, planned=planned)
    assert issued.authorization["max_uses"] == 1

    store = MemoryConsumptionStore()
    calls: list[Any] = []
    result = GovernedRuntime(harness.gateway(store, calls)).execute(
        intent(request_id="req-secure-exec-001"),
        lambda arguments: calls.append(arguments),
    )
    assert result.status is ExecutionStatus.EXECUTED, result.reason
    assert calls == [ARGUMENTS]
