from __future__ import annotations

from typing import Any

import pytest

from cbrain import ActionIntent
from cbrain.adapters import (
    DecisionBinding,
    HttpJsonResponse,
    PrivateVaultBindingError,
    PrivateVaultDecisionClient,
    PrivateVaultHttpTransport,
    PrivateVaultProtocolError,
    PrivateVaultVerdict,
)

ZERO = "sha256:" + ("0" * 64)
ONE = "sha256:" + ("1" * 64)
HASH = "2" * 64


def execution_action() -> dict[str, Any]:
    return {
        "subject_principal": "agent-1@example",
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
        "serialization": "pv-json-parameters/0.1",
    }


def binding() -> DecisionBinding:
    return DecisionBinding.capture(
        action(),
        execution_action=execution_action(),
        dispatch=dispatch(),
    )


def mintable_record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "protocol_version": "drp/0.2",
        "decision_id": "decision-1",
        "agent_id": "agent-1",
        "capability": "payments.execute",
        "decision": "allow",
        "request_id": "req-1",
        "action_digest": ZERO,
        "dispatch_context_digest": ONE,
        "record_hash": HASH,
    }
    record.update(overrides)
    return record


def action() -> ActionIntent:
    return ActionIntent.capture(
        request_id="req-1",
        idempotency_key="idem-1",
        agent_id="agent-1",
        framework="hermes",
        tool_name="payments.execute",
        capability="payments.execute",
        timestamp=1_700_000_000.0,
        arguments={"amount": 100},
        context={"tenant_id": "tenant-1"},
        evidence={"source": "test"},
    )


class FakeTransport:
    def __init__(
        self,
        response: HttpJsonResponse,
    ) -> None:
        self.response = response
        self.path = None
        self.payload = None

    def post_json(self, path, payload):
        self.path = path
        self.payload = payload
        return self.response


def response(
    decision: str,
    status_code: int,
) -> HttpJsonResponse:
    return HttpJsonResponse(
        status_code=status_code,
        body={
            "decision": decision,
            "triggered_by": "policy",
            "reason": "test decision",
            "record": {
                "decision_id": "decision-1",
                "agent_id": "agent-1",
                "request_id": "req-1",
            },
        },
    )


@pytest.mark.parametrize(
    ("raw_decision", "status", "expected"),
    [
        ("allow", 200, PrivateVaultVerdict.ALLOW),
        (
            "require_approval",
            202,
            PrivateVaultVerdict.REQUIRE_APPROVAL,
        ),
        ("block", 403, PrivateVaultVerdict.BLOCK),
    ],
)
def test_maps_real_privatevault_decisions(
    raw_decision,
    status,
    expected,
):
    transport = FakeTransport(response(raw_decision, status))

    result = PrivateVaultDecisionClient(transport).decide(action())

    assert result.verdict is expected
    assert result.request_id == "req-1"
    assert result.reason == "test decision"
    assert result.record["decision_id"] == "decision-1"


def test_sends_real_decide_request_shape():
    transport = FakeTransport(response("allow", 200))
    intent = action()

    PrivateVaultDecisionClient(transport).decide(intent)

    assert transport.path == "/v1/decide"
    assert set(transport.payload) == {
        "agent_id",
        "capability",
        "timestamp",
        "arguments",
        "context",
        "evidence",
        "request_id",
    }
    assert transport.payload["agent_id"] == intent.agent_id
    assert transport.payload["request_id"] == intent.request_id
    assert transport.payload["context"]["cbrain"] == {
        "framework": "hermes",
        "tool_name": "payments.execute",
        "idempotency_key": "idem-1",
    }


@pytest.mark.parametrize(
    ("decision", "wrong_status"),
    [
        ("allow", 403),
        ("require_approval", 200),
        ("block", 200),
    ],
)
def test_rejects_status_decision_contradiction(
    decision,
    wrong_status,
):
    client = PrivateVaultDecisionClient(FakeTransport(response(decision, wrong_status)))

    with pytest.raises(
        PrivateVaultProtocolError,
        match="contradicts",
    ):
        client.decide(action())


def test_rejects_unknown_decision():
    client = PrivateVaultDecisionClient(FakeTransport(response("maybe", 200)))

    with pytest.raises(
        PrivateVaultProtocolError,
        match="unknown decision",
    ):
        client.decide(action())


def test_rejects_missing_record():
    malformed = {
        "decision": "allow",
        "triggered_by": "policy",
        "reason": "allowed",
    }

    client = PrivateVaultDecisionClient(
        FakeTransport(
            HttpJsonResponse(
                status_code=200,
                body=malformed,
            )
        )
    )

    with pytest.raises(
        PrivateVaultProtocolError,
        match="'record'",
    ):
        client.decide(action())


def test_record_is_immutable_snapshot():
    record: dict[str, Any] = {
        "decision_id": "decision-1",
        "request_id": "req-1",
        "nested": {"value": 1},
    }

    client = PrivateVaultDecisionClient(
        FakeTransport(
            HttpJsonResponse(
                status_code=200,
                body={
                    "decision": "allow",
                    "triggered_by": "policy",
                    "reason": "allowed",
                    "record": record,
                },
            )
        )
    )

    result = client.decide(action())
    record["nested"]["value"] = 999

    assert result.record["nested"]["value"] == 1


def test_rejects_decision_for_different_request():
    client = PrivateVaultDecisionClient(
        FakeTransport(
            HttpJsonResponse(
                status_code=200,
                body={
                    "decision": "allow",
                    "triggered_by": "policy",
                    "reason": "allowed",
                    "record": {
                        "decision_id": "decision-2",
                        "request_id": "req-2",
                    },
                },
            )
        )
    )

    with pytest.raises(
        PrivateVaultProtocolError,
        match="different request_id",
    ):
        client.decide(action())


def test_rejects_decision_without_request_binding():
    client = PrivateVaultDecisionClient(
        FakeTransport(
            HttpJsonResponse(
                status_code=200,
                body={
                    "decision": "allow",
                    "triggered_by": "policy",
                    "reason": "allowed",
                    "record": {
                        "decision_id": "decision-1",
                    },
                },
            )
        )
    )

    with pytest.raises(
        PrivateVaultProtocolError,
        match="not bound",
    ):
        client.decide(action())


def test_bound_decide_request_carries_execution_action_and_dispatch_context():
    transport = FakeTransport(
        HttpJsonResponse(
            status_code=200,
            body={
                "decision": "allow",
                "triggered_by": "policy",
                "reason": "allowed",
                "record": mintable_record(),
            },
        )
    )

    result = PrivateVaultDecisionClient(transport).decide(action(), binding=binding())

    assert result.verdict is PrivateVaultVerdict.ALLOW
    assert transport.path == "/v1/decide"
    payload = transport.payload
    assert payload["execution_action"] == execution_action()
    assert payload["dispatch_context"] == {
        "adapter": "https",
        "transport": "https",
        "operation": "POST /v1/payments",
        "destination": "ledger.example",
        "wire_content_type": "application/json",
        "serialization": "pv-json-parameters/0.1",
    }
    # The audit fields are unchanged; the pinned server derives the digests
    # itself and refuses caller-authored ones.
    assert "action_digest" not in payload["execution_action"]
    assert "dispatch_context_digest" not in payload["dispatch_context"]
    assert payload["arguments"] == payload["execution_action"]["parameters"]


def test_unbound_decide_request_stays_audit_only():
    transport = FakeTransport(response("allow", 200))

    PrivateVaultDecisionClient(transport).decide(action())

    assert "execution_action" not in transport.payload
    assert "dispatch_context" not in transport.payload


def test_bound_decide_refuses_an_audit_only_record():
    """A drp/0.1 answer to a bound request could never mint; refuse it now."""
    transport = FakeTransport(
        HttpJsonResponse(
            status_code=200,
            body={
                "decision": "allow",
                "triggered_by": "policy",
                "reason": "allowed",
                "record": {
                    "protocol_version": "drp/0.1",
                    "decision_id": "decision-1",
                    "agent_id": "agent-1",
                    "capability": "payments.execute",
                    "request_id": "req-1",
                    "record_hash": HASH,
                },
            },
        )
    )

    with pytest.raises(PrivateVaultProtocolError, match="audit-only"):
        PrivateVaultDecisionClient(transport).decide(action(), binding=binding())


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"action_digest": "not-a-digest"}, "action_digest"),
        ({"dispatch_context_digest": None}, "dispatch_context_digest"),
        ({"record_hash": "abc"}, "record_hash"),
        ({"decision_id": ""}, "decision_id"),
        ({"agent_id": "agent-2"}, "different agent_id"),
        ({"capability": "payments.refund"}, "different capability"),
    ],
)
def test_bound_decide_refuses_records_the_mint_binding_cannot_read(
    override,
    message,
):
    transport = FakeTransport(
        HttpJsonResponse(
            status_code=200,
            body={
                "decision": "allow",
                "triggered_by": "policy",
                "reason": "allowed",
                "record": mintable_record(**override),
            },
        )
    )

    with pytest.raises(PrivateVaultProtocolError, match=message):
        PrivateVaultDecisionClient(transport).decide(action(), binding=binding())


def test_binding_refuses_parameters_that_differ_from_the_governed_arguments():
    widened = execution_action()
    widened["parameters"] = {"amount": 100, "beneficiary": "x"}

    with pytest.raises(PrivateVaultBindingError, match="parameters"):
        DecisionBinding.capture(action(), execution_action=widened, dispatch=dispatch())


def test_binding_refuses_action_or_subject_that_differ_from_the_intent():
    other_capability = {**execution_action(), "action": "payments.refund"}
    with pytest.raises(PrivateVaultBindingError, match="capability"):
        DecisionBinding.capture(
            action(), execution_action=other_capability, dispatch=dispatch()
        )

    other_subject = {**execution_action(), "subject_key_id": "agent-2"}
    with pytest.raises(PrivateVaultBindingError, match="agent_id"):
        DecisionBinding.capture(
            action(), execution_action=other_subject, dispatch=dispatch()
        )


def test_binding_refuses_extra_or_missing_execution_action_fields():
    extra = {**execution_action(), "action_digest": ZERO}
    with pytest.raises(
        PrivateVaultBindingError, match="unexpected=\\['action_digest'\\]"
    ):
        DecisionBinding.capture(action(), execution_action=extra, dispatch=dispatch())

    missing = execution_action()
    del missing["resource"]
    with pytest.raises(PrivateVaultBindingError, match="missing=\\['resource'\\]"):
        DecisionBinding.capture(action(), execution_action=missing, dispatch=dispatch())


def test_binding_refuses_dispatch_missing_a_sealed_context_field():
    broken = dispatch()
    broken["operation"] = ""

    with pytest.raises(PrivateVaultBindingError, match="dispatch_context.operation"):
        DecisionBinding.capture(
            action(), execution_action=execution_action(), dispatch=broken
        )


def test_binding_is_an_immutable_snapshot():
    source = execution_action()
    captured = DecisionBinding.capture(
        action(), execution_action=source, dispatch=dispatch()
    )
    source["parameters"]["amount"] = 999
    captured.execution_action["parameters"]["amount"] = 555

    assert captured.execution_action["parameters"]["amount"] == 100


def test_remote_privatevault_requires_https():
    with pytest.raises(
        ValueError,
        match="requires HTTPS",
    ):
        PrivateVaultHttpTransport(
            base_url="http://privatevault.example",
            headers_provider=lambda: {},
        )


def test_local_http_requires_explicit_opt_in():
    with pytest.raises(
        ValueError,
        match="requires HTTPS",
    ):
        PrivateVaultHttpTransport(
            base_url="http://127.0.0.1:8765",
            headers_provider=lambda: {},
        )

    transport = PrivateVaultHttpTransport(
        base_url="http://127.0.0.1:8765",
        headers_provider=lambda: {},
        allow_insecure_localhost=True,
    )

    assert transport is not None


def test_base_url_rejects_embedded_credentials():
    with pytest.raises(
        ValueError,
        match="must not contain credentials",
    ):
        PrivateVaultHttpTransport(
            base_url="https://user:secret@privatevault.example",
            headers_provider=lambda: {},
        )
