from __future__ import annotations

from typing import Any

import pytest

from cbrain import ActionIntent
from cbrain.adapters import (
    HttpJsonResponse,
    PrivateVaultDecisionClient,
    PrivateVaultHttpTransport,
    PrivateVaultProtocolError,
    PrivateVaultVerdict,
)


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
