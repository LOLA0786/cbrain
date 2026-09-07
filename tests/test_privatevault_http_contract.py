"""The decision-to-permit HTTP contract against the pinned PrivateVault server.

Everything on the PrivateVault side is the real pinned `api.server`: API-key
authentication, the grants and policy files, DRP 0.2 sealing on `/v1/decide`,
and the sealed-ALLOW binding on `/v1/authorize`. Everything on the CBrain side
is the production client code: `PrivateVaultDecisionClient`,
`PrivateVaultAuthorizationClient`, `HttpDispatchPlanner` and
`PrivateVaultExecutionGateway` with the real Agent DNA verifiers.

What is proven: one bound ALLOW mints and reaches EXECUTED; an audit-only
decision, a changed action, a changed dispatch, another agent's record, a
BLOCK and an unapproved REVIEW cannot mint. What is not proven here: an
independent dispatcher (the transport is in-process), and the deployment
origin of the non-decision evidence digests, which the pinned server seals but
does not resolve.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

import pytest

from cbrain import ActionIntent, ExecutionStatus, GovernedRuntime
from cbrain.adapters.privatevault import (
    MINTABLE_DECISION_PROTOCOL,
    PrivateVaultDecisionClient,
    PrivateVaultProtocolError,
    PrivateVaultVerdict,
)
from cbrain.adapters.privatevault_claim import PrivateVaultAuthorizationClaimCoordinator
from cbrain.adapters.privatevault_consumption import PrivateVaultAuthorizationUseFactory
from cbrain.adapters.privatevault_execution import (
    AgentDNAVerifiers,
    PrivateVaultAgentDNAVerifier,
)
from cbrain.execution import (
    InProcessDispatchTransport,
    PlannedDispatch,
    PrivateVaultExecutionGateway,
    WitnessIdentity,
)
from cbrain.execution.authorize_client import (
    AuthorizationRefused,
    EvidenceDigests,
    PrivateVaultAuthorizationClient,
)
from cbrain.execution.planner import HttpDispatchPlanner, ToolRoute
from pinned_privatevault_server import (
    PinnedServer,
    TestClientTransport,
    pinned_privatevault_server,
)
from test_execution_gateway_conformance import (
    Closer,
    MemoryConsumptionStore,
    SealingDispatchTransport,
)

execution_v01 = pytest.importorskip("agent_dna.execution_v01")
dispatch_v01 = pytest.importorskip("agent_dna.dispatch_v01")
closure_v01 = pytest.importorskip("agent_dna.closure_v01")

ORG = "steel.example"
BUYER = "buyer-agent"
OTHER = "other-agent"

PO_CREATE = "procurement.po.create"
PO_AMEND = "procurement.po.amend"  # deliberately not granted -> REVIEW
BANK_CHANGE = "procurement.vendor.bank_change"  # blocked by policy

ZERO = "sha256:" + ("0" * 64)
ONE = "sha256:" + ("1" * 64)
PEER_IDENTITY = "tls-spki-sha256:" + ("a" * 64)

ARGUMENTS = {"vendor_id": "V-001", "material": "HR-COIL-3MM", "quantity_tonnes": 40}


def route(capability: str) -> ToolRoute:
    return ToolRoute(
        tool_id=f"{capability}.v1",
        capability=capability,
        destination="erp-sandbox.example",
        operation="POST /v1/purchase-orders",
        credential_audience="erp-sandbox.example",
        peer_identity=PEER_IDENTITY,
        allowed_parameters=tuple(sorted(ARGUMENTS)),
        required_parameters=tuple(sorted(ARGUMENTS)),
    )


ROUTES = {name: route(name) for name in (PO_CREATE, PO_AMEND, BANK_CHANGE)}


def planner(agent_id: str = BUYER) -> HttpDispatchPlanner:
    return HttpDispatchPlanner(
        ROUTES,
        subject_principal=f"{agent_id}@{ORG}",
        subject_key_id=agent_id,
    )


def intent(
    capability: str = PO_CREATE,
    *,
    request_id: str = "req-po-001",
    agent_id: str = BUYER,
) -> ActionIntent:
    return ActionIntent.capture(
        request_id=request_id,
        idempotency_key=request_id,
        agent_id=agent_id,
        framework="test",
        tool_name=capability,
        capability=capability,
        timestamp=datetime.now(UTC).timestamp(),
        arguments=dict(ARGUMENTS),
    )


def evidence() -> EvidenceDigests:
    return EvidenceDigests(
        authority_receipt_digest=ONE,
        state_snapshot_digest=ZERO,
        policy_bundle_digest=ONE,
        obligations_digest=ZERO,
    )


def now_rfc3339() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


@pytest.fixture()
def pv(tmp_path, monkeypatch):
    with pinned_privatevault_server(
        tmp_path,
        monkeypatch,
        organisation_id=ORG,
        grants={BUYER: (PO_CREATE,), OTHER: (PO_CREATE,)},
        blocked_capabilities=(BANK_CHANGE,),
    ) as server:
        yield server


class Harness:
    """Production clients wired to the pinned server for one agent."""

    def __init__(self, server: PinnedServer, agent_id: str = BUYER) -> None:
        self.server = server
        self.agent_id = agent_id
        self.transport: TestClientTransport = server.transport(agent_id)
        self.decisions = PrivateVaultDecisionClient(self.transport)
        self.issuer = PrivateVaultAuthorizationClient(
            self.transport,
            organisation_id=ORG,
            evidence_digests=evidence(),
        )
        self.planner = planner(agent_id)

    def bound_decision(self, action: ActionIntent) -> tuple[Any, PlannedDispatch]:
        from cbrain.adapters.privatevault import DecisionBinding

        planned = self.planner.plan(action)
        binding = DecisionBinding.capture(
            action,
            execution_action=planned.action,
            dispatch=planned.prepared.dispatch,
        )
        return self.decisions.decide(action, binding=binding), planned

    def gateway(
        self,
        store: MemoryConsumptionStore,
        calls: list[Any],
    ) -> PrivateVaultExecutionGateway:
        keyring = self.server.keyring
        verifier = PrivateVaultAgentDNAVerifier(
            AgentDNAVerifiers(
                verify_execution_authorization=(
                    execution_v01.verify_execution_authorization
                ),
                verify_dispatch_witness=dispatch_v01.verify_dispatch_witness,
                verify_closure_chain=closure_v01.verify_closure_chain,
            )
        )
        transport = SealingDispatchTransport(
            InProcessDispatchTransport(
                handler_runner=lambda arguments: None,
                witness_signer=dispatch_v01.create_dispatch_witness,
                signing_key=keyring.witness,
                identity=WitnessIdentity(
                    witness_component_id="dispatcher-01",
                    signer_key_id="witness-signer-01",
                    independent=False,
                ),
            ),
            Closer(keyring),
        )
        return PrivateVaultExecutionGateway(
            decision_client=self.decisions,
            planner=self.planner,
            issuer=self.issuer,
            claim_coordinator=PrivateVaultAuthorizationClaimCoordinator(
                verifier=verifier,
                use_factory=PrivateVaultAuthorizationUseFactory(
                    execution_v01.execution_authorization_digest
                ),
                consumption_store=store,
            ),
            verifier=verifier,
            transport=transport,
            clock=now_rfc3339,
            witness_id_factory=lambda action: f"witness-{action.request_id}",
        )

    def raw_authorize(self, body: dict[str, Any]) -> Any:
        return self.server.client.post(
            "/v1/authorize",
            json=body,
            headers={"X-API-Key": self.server.api_keys[self.agent_id]},
        )

    def authorize_body(
        self,
        record: dict[str, Any],
        planned: PlannedDispatch,
        *,
        request_id: str,
        with_reference: bool = True,
    ) -> dict[str, Any]:
        prepared = planned.prepared
        body: dict[str, Any] = {
            "request_id": request_id,
            "agent_id": self.agent_id,
            "organisation_id": ORG,
            "action": dict(planned.action),
            "dispatch": dict(prepared.dispatch),
            "expected_wire_bytes_digest": sha256(prepared.wire_bytes),
            "expected_wire_bytes_length": len(prepared.wire_bytes),
            "expected_peer_identity_digest": sha256(prepared.peer_identity_bytes),
            "decision_receipt_digest": "sha256:" + record["record_hash"],
            "authority_receipt_digest": ONE,
            "state_snapshot_digest": ZERO,
            "policy_bundle_digest": ONE,
            "obligations_digest": ZERO,
        }
        if with_reference:
            body["decision_id"] = record["decision_id"]
        return body


def assert_no_permit(response: Any, reason_code: str) -> None:
    assert response.status_code in (403, 404, 422), response.text
    body = response.json()
    assert "authorization" not in body
    assert "execution_authorization_id" not in json.dumps(body)
    assert body["detail"]["reason_code"] == reason_code, body


# --------------------------------------------------------------------------
# The bound ALLOW path


def test_bound_decide_seals_a_mintable_record(pv):
    harness = Harness(pv)

    decision, planned = harness.bound_decision(intent())

    assert decision.verdict is PrivateVaultVerdict.ALLOW
    record = decision.record
    assert record["protocol_version"] == MINTABLE_DECISION_PROTOCOL
    assert record["agent_id"] == BUYER
    assert record["capability"] == PO_CREATE
    assert record["action_digest"].startswith("sha256:")
    assert record["dispatch_context_digest"].startswith("sha256:")
    ((path, sent),) = harness.transport.calls
    assert path == "/v1/decide"
    assert sent["execution_action"] == planned.action
    assert sent["arguments"] == ARGUMENTS


def test_bound_allow_mints_a_permit_answering_this_request(pv):
    harness = Harness(pv)
    action = intent()
    decision, planned = harness.bound_decision(action)

    issued = harness.issuer.issue(action=action, decision=decision, planned=planned)

    authorization = issued.authorization
    assert authorization["execution_authorization_id"].startswith("eauth-")
    assert authorization["request_id"] == action.request_id
    assert authorization["organisation_id"] == ORG
    assert authorization["action"] == planned.action
    assert authorization["dispatch"] == planned.prepared.dispatch
    assert authorization["decision_receipt_digest"] == (
        "sha256:" + decision.record["record_hash"]
    )
    assert authorization["max_uses"] == 1
    assert issued.trust_bundle["organisation_id"] == ORG
    assert harness.transport.paths() == ["/v1/decide", "/v1/authorize"]
    _, sent = harness.transport.calls[1]
    assert sent["decision_id"] == decision.record["decision_id"]
    assert sent["record_hash"] == decision.record["record_hash"]


def test_governed_runtime_reaches_executed_through_the_pinned_server(pv):
    harness = Harness(pv)
    store = MemoryConsumptionStore()
    calls: list[Any] = []

    result = GovernedRuntime(harness.gateway(store, calls)).execute(
        intent(),
        lambda arguments: calls.append(arguments) or {"po_id": "PO-1001"},
    )

    assert result.status is ExecutionStatus.EXECUTED, result.reason
    assert result.tool_executed is True
    assert result.output == {"po_id": "PO-1001"}
    assert calls == [ARGUMENTS]
    assert result.decision_id
    assert harness.transport.paths() == ["/v1/decide", "/v1/authorize"]


# --------------------------------------------------------------------------
# What cannot mint


def test_audit_only_decision_cannot_mint(pv):
    """Without the binding the server seals drp/0.1; the client refuses to issue
    and the server refuses a reference-less mint."""
    harness = Harness(pv)
    action = intent()
    planned = harness.planner.plan(action)

    audit_only = harness.decisions.decide(action)
    assert audit_only.verdict is PrivateVaultVerdict.ALLOW
    assert audit_only.record["protocol_version"] != MINTABLE_DECISION_PROTOCOL

    with pytest.raises(AuthorizationRefused, match="audit-only"):
        harness.issuer.issue(action=action, decision=audit_only, planned=planned)
    assert harness.transport.paths() == ["/v1/decide"]

    # The server agrees, on both refusal paths.
    assert_no_permit(
        harness.raw_authorize(
            harness.authorize_body(
                audit_only.record,
                planned,
                request_id=action.request_id,
                with_reference=False,
            )
        ),
        "AUTHORIZE_DECISION_REQUIRED",
    )
    assert_no_permit(
        harness.raw_authorize(
            harness.authorize_body(
                audit_only.record, planned, request_id=action.request_id
            )
        ),
        "AUTHORIZE_PROTOCOL_NOT_AUTHORIZING",
    )


def test_bound_decide_refuses_a_server_that_did_not_seal_the_binding(pv):
    """If the pinned server ever answered a bound request with an audit-only
    record, the client must not carry that decision forward."""
    harness = Harness(pv)
    action = intent()
    planned = harness.planner.plan(action)
    from cbrain.adapters.privatevault import DecisionBinding

    binding = DecisionBinding.capture(
        action,
        execution_action=planned.action,
        dispatch=planned.prepared.dispatch,
    )

    class StripsBinding:
        def __init__(self, inner: TestClientTransport) -> None:
            self._inner = inner

        def post_json(self, path, payload):
            stripped = {
                key: value
                for key, value in payload.items()
                if key not in {"execution_action", "dispatch_context"}
            }
            return self._inner.post_json(path, stripped)

    with pytest.raises(PrivateVaultProtocolError, match="audit-only"):
        PrivateVaultDecisionClient(StripsBinding(harness.transport)).decide(
            action, binding=binding
        )


def test_changed_action_after_decision_cannot_mint(pv):
    harness = Harness(pv)
    action = intent()
    decision, planned = harness.bound_decision(action)

    widened = dict(planned.action)
    widened["parameters"] = {**ARGUMENTS, "quantity_tonnes": 400}
    tampered = PlannedDispatch(action=widened, prepared=planned.prepared)

    with pytest.raises(
        AuthorizationRefused,
        match="status 403:AUTHORIZE_ARGUMENTS_DIGEST_MISMATCH",
    ):
        harness.issuer.issue(action=action, decision=decision, planned=tampered)

    rerouted = dict(planned.action)
    rerouted["resource"] = "erp-production.example"
    with pytest.raises(
        AuthorizationRefused,
        match="status 403:AUTHORIZE_ACTION_DIGEST_MISMATCH",
    ):
        harness.issuer.issue(
            action=action,
            decision=decision,
            planned=PlannedDispatch(action=rerouted, prepared=planned.prepared),
        )


def test_changed_dispatch_after_decision_cannot_mint(pv):
    from cbrain.dispatch import PreparedDispatch

    harness = Harness(pv)
    action = intent()
    decision, planned = harness.bound_decision(action)

    redirected = dict(planned.prepared.dispatch)
    redirected["destination"] = "erp-production.example"
    tampered = PlannedDispatch(
        action=planned.action,
        prepared=PreparedDispatch.capture(
            request_id=action.request_id,
            dispatch=redirected,
            wire_bytes=planned.prepared.wire_bytes,
            peer_identity_bytes=planned.prepared.peer_identity_bytes,
        ),
    )

    with pytest.raises(
        AuthorizationRefused,
        match="status 403:AUTHORIZE_DISPATCH_CONTEXT_DIGEST_MISMATCH",
    ):
        harness.issuer.issue(action=action, decision=decision, planned=tampered)


def test_another_agents_record_cannot_mint(pv):
    buyer = Harness(pv, BUYER)
    other = Harness(pv, OTHER)
    buyer_decision, _ = buyer.bound_decision(intent())

    other_action = intent(agent_id=OTHER, request_id="req-other-001")
    other_planned = other.planner.plan(other_action)

    # The client sees the record was sealed for someone else.
    with pytest.raises(AuthorizationRefused, match="different agent_id"):
        other.issuer.issue(
            action=other_action, decision=buyer_decision, planned=other_planned
        )
    assert other.transport.paths() == []

    # So does the server, if the client check is bypassed.
    assert_no_permit(
        other.raw_authorize(
            other.authorize_body(
                buyer_decision.record,
                other_planned,
                request_id=other_action.request_id,
            )
        ),
        "AUTHORIZE_AGENT_MISMATCH",
    )


def test_block_cannot_mint_and_never_calls_authorize(pv):
    harness = Harness(pv)
    action = intent(BANK_CHANGE)
    decision, planned = harness.bound_decision(action)
    assert decision.verdict is PrivateVaultVerdict.BLOCK
    assert decision.record["protocol_version"] == MINTABLE_DECISION_PROTOCOL

    with pytest.raises(AuthorizationRefused, match="only an ALLOW"):
        harness.issuer.issue(action=action, decision=decision, planned=planned)
    assert harness.transport.paths() == ["/v1/decide"]

    assert_no_permit(
        harness.raw_authorize(
            harness.authorize_body(
                decision.record, planned, request_id=action.request_id
            )
        ),
        "AUTHORIZE_DECISION_NOT_ALLOW",
    )

    store = MemoryConsumptionStore()
    calls: list[Any] = []
    gateway_transport = Harness(pv)
    result = GovernedRuntime(gateway_transport.gateway(store, calls)).execute(
        intent(BANK_CHANGE, request_id="req-bank-002"),
        lambda arguments: calls.append(arguments),
    )
    assert result.status is ExecutionStatus.BLOCKED
    assert result.tool_executed is False
    assert calls == []
    assert gateway_transport.transport.paths() == ["/v1/decide"]


def test_unapproved_review_cannot_mint_and_never_calls_authorize(pv):
    harness = Harness(pv)
    action = intent(PO_AMEND)
    decision, planned = harness.bound_decision(action)
    assert decision.verdict is PrivateVaultVerdict.REQUIRE_APPROVAL
    assert decision.triggered_by == "authorization"

    with pytest.raises(AuthorizationRefused, match="only an ALLOW"):
        harness.issuer.issue(action=action, decision=decision, planned=planned)
    assert harness.transport.paths() == ["/v1/decide"]

    assert_no_permit(
        harness.raw_authorize(
            harness.authorize_body(
                decision.record, planned, request_id=action.request_id
            )
        ),
        "AUTHORIZE_DECISION_NOT_ALLOW",
    )

    store = MemoryConsumptionStore()
    calls: list[Any] = []
    gateway_harness = Harness(pv)
    result = GovernedRuntime(gateway_harness.gateway(store, calls)).execute(
        intent(PO_AMEND, request_id="req-amend-002"),
        lambda arguments: calls.append(arguments),
    )
    assert result.status is ExecutionStatus.REVIEW_REQUIRED
    assert result.tool_executed is False
    assert calls == []
    assert gateway_harness.transport.paths() == ["/v1/decide"]


def test_a_second_mint_for_the_same_decision_with_other_bindings_conflicts(pv):
    """The permit is bound at mint time; the same sealed decision cannot be
    re-minted for different wire bytes."""
    from cbrain.dispatch import PreparedDispatch

    harness = Harness(pv)
    action = intent()
    decision, planned = harness.bound_decision(action)
    harness.issuer.issue(action=action, decision=decision, planned=planned)

    other_bytes = PlannedDispatch(
        action=planned.action,
        prepared=PreparedDispatch.capture(
            request_id=action.request_id,
            dispatch=planned.prepared.dispatch,
            wire_bytes=planned.prepared.wire_bytes + b" ",
            peer_identity_bytes=planned.prepared.peer_identity_bytes,
        ),
    )

    with pytest.raises(
        AuthorizationRefused,
        match="status 403:AUTHORIZE_PERMIT_BINDING_CONFLICT",
    ):
        harness.issuer.issue(action=action, decision=decision, planned=other_bytes)
