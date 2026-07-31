from __future__ import annotations

import pytest

from cbrain.contracts import ContractError
from cbrain.dispatch import PreparedDispatch

ZERO = "sha256:" + ("0" * 64)
ONE = "sha256:" + ("1" * 64)


def metadata() -> dict[str, str]:
    return {
        "transport": "https",
        "destination": "payments.example",
        "operation": "POST /v1/payments",
        "wire_content_type": "application/json",
        "wire_content_encoding": "identity",
        "tool_id": "payments.execute.v1",
        "tool_schema_digest": ZERO,
        "tool_artifact_digest": ONE,
        "credential_audience": "payments.example",
        "idempotency_key_digest": ZERO,
        "retry_policy_digest": ONE,
    }


def capture(
    dispatch=None,
    wire_bytes=b'{"amount":100}',
    peer_identity_bytes=b"payments.example",
):
    return PreparedDispatch.capture(
        request_id="req-1",
        dispatch=dispatch or metadata(),
        wire_bytes=wire_bytes,
        peer_identity_bytes=peer_identity_bytes,
    )


def test_captures_real_privatevault_dispatch_shape():
    prepared = capture()

    assert prepared.request_id == "req-1"
    assert prepared.dispatch == metadata()
    assert prepared.wire_bytes == b'{"amount":100}'
    assert prepared.peer_identity_bytes == b"payments.example"


def test_dispatch_is_immutable_snapshot():
    original = metadata()
    prepared = capture(original)
    original["destination"] = "attacker.example"

    assert prepared.dispatch["destination"] == "payments.example"


def test_rejects_missing_dispatch_field():
    invalid = metadata()
    invalid.pop("credential_audience")

    with pytest.raises(
        ContractError,
        match="missing=.*credential_audience",
    ):
        capture(invalid)


def test_rejects_unexpected_dispatch_field():
    invalid = metadata()
    invalid["untrusted"] = "value"

    with pytest.raises(
        ContractError,
        match="unexpected=.*untrusted",
    ):
        capture(invalid)


def test_rejects_invalid_digest():
    invalid = metadata()
    invalid["tool_artifact_digest"] = "not-a-digest"

    with pytest.raises(
        ContractError,
        match="tool_artifact_digest",
    ):
        capture(invalid)


@pytest.mark.parametrize(
    ("wire_bytes", "peer_identity_bytes"),
    [
        (b"", b"payments.example"),
        (b'{"amount":100}', b""),
    ],
)
def test_rejects_empty_boundary_evidence(
    wire_bytes,
    peer_identity_bytes,
):
    with pytest.raises(
        ContractError,
        match="non-empty immutable bytes",
    ):
        capture(
            wire_bytes=wire_bytes,
            peer_identity_bytes=peer_identity_bytes,
        )
