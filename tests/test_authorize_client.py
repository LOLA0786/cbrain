"""`/v1/authorize` client: the request names the sealed ALLOW, and the permit
must answer exactly that request.

These are contract tests against a recording transport. The same client is
exercised against the real pinned PrivateVault HTTP server in
`tests/test_privatevault_http_contract.py`.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from cbrain import ActionIntent
from cbrain.adapters.privatevault import (
    HttpJsonResponse,
    PrivateVaultDecision,
    PrivateVaultVerdict,
)
from cbrain.dispatch import PreparedDispatch
from cbrain.execution import PlannedDispatch
from cbrain.execution.authorize_client import (
    AuthorizationRefused,
    EvidenceDigests,
    PrivateVaultAuthorizationClient,
)

ZERO = "sha256:" + ("0" * 64)
ONE = "sha256:" + ("1" * 64)
TWO = "sha256:" + ("2" * 64)
RECORD_HASH = "3" * 64
ORG = "store.example"
WIRE = b'{"amount":100}'
PEER = b"tls-spki-sha256:" + (b"a" * 64)


def digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def intent() -> ActionIntent:
    return ActionIntent.capture(
        request_id="req-1",
        idempotency_key="req-1",
        agent_id="agent-1",
        framework="test",
        tool_name="payments.execute",
        capability="payments.execute",
        timestamp=1_700_000_000.0,
        arguments={"amount": 100},
    )


def execution_action() -> dict[str, Any]:
    return {
        "subject_principal": f"agent-1@{ORG}",
        "subject_key_id": "agent-1",
        "action": "payments.execute",
        "resource": "ledger:primary",
        "parameters": {"amount": 100},
    }


def dispatch() -> dict[str, Any]:
    return {
        "transport": "https",
        "destination": "ledger.example",
        "operation": "POST /v1/payments",
        "wire_content_type": "application/json",
        "wire_content_encoding": "identity",
        "tool_id": "payments.execute.v1",
        "tool_schema_digest": ZERO,
        "tool_artifact_digest": ONE,
        "credential_audience": "ledger.example",
        "idempotency_key_digest": ZERO,
        "retry_policy_digest": ONE,
    }


def planned() -> PlannedDispatch:
    return PlannedDispatch(
        action=execution_action(),
        prepared=PreparedDispatch.capture(
            request_id="req-1",
            dispatch=dispatch(),
            wire_bytes=WIRE,
            peer_identity_bytes=PEER,
        ),
    )


def record(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "protocol_version": "drp/0.2",
        "decision_id": "decision-1",
        "agent_id": "agent-1",
        "capability": "payments.execute",
        "decision": "allow",
        "request_id": "req-1",
        "action_digest": ZERO,
        "dispatch_context_digest": ONE,
        "record_hash": RECORD_HASH,
    }
    base.update(overrides)
    return base


def decision(
    verdict: PrivateVaultVerdict = PrivateVaultVerdict.ALLOW,
    **overrides: Any,
) -> PrivateVaultDecision:
    return PrivateVaultDecision(
        verdict=verdict,
        triggered_by="policy",
        reason="test",
        request_id="req-1",
        _record_json=json.dumps(record(**overrides)).encode(),
    )


def evidence() -> EvidenceDigests:
    return EvidenceDigests(
        authority_receipt_digest=ONE,
        state_snapshot_digest=ZERO,
        policy_bundle_digest=ONE,
        obligations_digest=ZERO,
    )


def permit_for(sent: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """An authorization echoing the request the way the pinned server does."""
    authorization = {
        "execution_authorization_id": "eauth-1",
        "organisation_id": sent["organisation_id"],
        "request_id": sent["request_id"],
        "decision_receipt_digest": sent["decision_receipt_digest"],
        "authority_receipt_digest": sent["authority_receipt_digest"],
        "approval_artifact_digest": sent["approval_artifact_digest"],
        "action": sent["action"],
        "expected_wire_bytes_digest": sent["expected_wire_bytes_digest"],
        "expected_wire_bytes_length": sent["expected_wire_bytes_length"],
        "expected_peer_identity_digest": sent["expected_peer_identity_digest"],
        "dispatch": sent["dispatch"],
        "state_snapshot_digest": sent["state_snapshot_digest"],
        "policy_bundle_digest": sent["policy_bundle_digest"],
        "obligations_digest": sent["obligations_digest"],
        "max_uses": 1,
    }
    authorization.update(overrides)
    return authorization


class EchoTransport:
    """Answers `/v1/authorize` with a permit bound to whatever was sent."""

    def __init__(self, **permit_overrides: Any) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self._overrides = permit_overrides

    def post_json(self, path: str, payload: Any) -> HttpJsonResponse:
        sent = json.loads(json.dumps(payload))
        self.requests.append((path, sent))
        return HttpJsonResponse(
            status_code=200,
            body={
                "authorization": permit_for(sent, **self._overrides),
                "trust_bundle": {"organisation_id": ORG, "keys": []},
                "at_time": "2026-09-07T09:00:00Z",
            },
        )


class RefusingTransport:
    def __init__(self, status_code: int, body: dict[str, Any]) -> None:
        self._response = HttpJsonResponse(status_code=status_code, body=body)
        self.calls = 0

    def post_json(self, path: str, payload: Any) -> HttpJsonResponse:
        self.calls += 1
        return self._response


def client(transport: Any) -> PrivateVaultAuthorizationClient:
    return PrivateVaultAuthorizationClient(
        transport,
        organisation_id=ORG,
        evidence_digests=evidence(),
    )


def test_request_names_the_sealed_allow_and_derives_its_receipt_digest():
    transport = EchoTransport()

    issued = client(transport).issue(
        action=intent(), decision=decision(), planned=planned()
    )

    ((path, sent),) = transport.requests
    assert path == "/v1/authorize"
    assert sent["decision_id"] == "decision-1"
    assert sent["record_hash"] == RECORD_HASH
    assert sent["decision_receipt_digest"] == "sha256:" + RECORD_HASH
    assert sent["agent_id"] == "agent-1"
    assert sent["organisation_id"] == ORG
    assert sent["action"] == execution_action()
    assert sent["dispatch"] == dispatch()
    assert sent["expected_wire_bytes_digest"] == digest(WIRE)
    assert sent["expected_wire_bytes_length"] == len(WIRE)
    assert sent["expected_peer_identity_digest"] == digest(PEER)
    assert sent["approval_artifact_digest"] is None
    assert issued.binding_digests["decision_receipt_digest"] == (
        "sha256:" + RECORD_HASH
    )
    assert issued.binding_digests["at_time"] == "2026-09-07T09:00:00Z"


def test_prefixed_record_hash_is_normalised_not_double_prefixed():
    transport = EchoTransport()

    client(transport).issue(
        action=intent(),
        decision=decision(record_hash="sha256:" + RECORD_HASH),
        planned=planned(),
    )

    ((_, sent),) = transport.requests
    assert sent["record_hash"] == RECORD_HASH
    assert sent["decision_receipt_digest"] == "sha256:" + RECORD_HASH


@pytest.mark.parametrize(
    ("verdict", "overrides", "message"),
    [
        (PrivateVaultVerdict.BLOCK, {"decision": "block"}, "only an ALLOW"),
        (
            PrivateVaultVerdict.REQUIRE_APPROVAL,
            {"decision": "require_approval"},
            "only an ALLOW",
        ),
        (
            PrivateVaultVerdict.ALLOW,
            {"protocol_version": "drp/0.1"},
            "audit-only",
        ),
        (PrivateVaultVerdict.ALLOW, {"record_hash": ""}, "record_hash"),
        (PrivateVaultVerdict.ALLOW, {"decision_id": " "}, "decision_id"),
        (PrivateVaultVerdict.ALLOW, {"agent_id": "agent-2"}, "different agent_id"),
        (PrivateVaultVerdict.ALLOW, {"decision": "block"}, "does not carry an ALLOW"),
        (PrivateVaultVerdict.ALLOW, {"request_id": "req-9"}, "different request"),
    ],
)
def test_unmintable_decisions_are_refused_before_any_request(
    verdict,
    overrides,
    message,
):
    transport = EchoTransport()

    with pytest.raises(AuthorizationRefused, match=message):
        client(transport).issue(
            action=intent(),
            decision=decision(verdict, **overrides),
            planned=planned(),
        )

    assert transport.requests == []


def test_server_refusal_carries_the_reason_code():
    transport = RefusingTransport(
        403,
        {
            "detail": {
                "reason_code": "AUTHORIZE_ACTION_DIGEST_MISMATCH",
                "detail": "authorize mint refused: sealed ALLOW binding failed",
            }
        },
    )

    with pytest.raises(
        AuthorizationRefused,
        match="status 403:AUTHORIZE_ACTION_DIGEST_MISMATCH",
    ):
        client(transport).issue(action=intent(), decision=decision(), planned=planned())


def test_server_refusal_without_reason_code_is_still_refused():
    transport = RefusingTransport(503, {"detail": "signer is not configured"})

    with pytest.raises(AuthorizationRefused, match="status 503:unspecified"):
        client(transport).issue(action=intent(), decision=decision(), planned=planned())


@pytest.mark.parametrize(
    "override",
    [
        {"decision_receipt_digest": TWO},
        {"authority_receipt_digest": TWO},
        {"state_snapshot_digest": TWO},
        {"policy_bundle_digest": TWO},
        {"obligations_digest": TWO},
        {"approval_artifact_digest": TWO},
        {"request_id": "req-2"},
        {"organisation_id": "other.example"},
        {"expected_wire_bytes_digest": TWO},
        {"expected_wire_bytes_length": 1},
        {"expected_peer_identity_digest": TWO},
        {"max_uses": 2},
        {"dispatch": {**dispatch(), "destination": "evil.example"}},
        {"action": {**execution_action(), "resource": "ledger:other"}},
    ],
)
def test_permit_that_does_not_echo_the_request_is_refused(override):
    transport = EchoTransport(**override)

    with pytest.raises(AuthorizationRefused, match="does not match|different|not "):
        client(transport).issue(action=intent(), decision=decision(), planned=planned())


def test_evidence_digests_must_be_well_formed():
    with pytest.raises(ValueError, match="authority_receipt_digest"):
        EvidenceDigests(
            authority_receipt_digest="not-a-digest",
            state_snapshot_digest=ZERO,
            policy_bundle_digest=ONE,
            obligations_digest=ZERO,
        )

    with pytest.raises(ValueError, match="approval_artifact_digest"):
        EvidenceDigests(
            authority_receipt_digest=ONE,
            state_snapshot_digest=ZERO,
            policy_bundle_digest=ONE,
            obligations_digest=ZERO,
            approval_artifact_digest="sha256:short",
        )


def test_organisation_id_is_required():
    with pytest.raises(ValueError, match="organisation_id"):
        PrivateVaultAuthorizationClient(
            EchoTransport(), organisation_id=" ", evidence_digests=evidence()
        )
