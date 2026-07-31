from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from cbrain.adapters.privatevault_consumption import (
    PrivateVaultAuthorizationUseFactory,
)
from cbrain.adapters.privatevault_execution import (
    ExecutionAuthorizationBinding,
    PrivateVaultExecutionError,
    VerifiedAuthorization,
    VerifiedEvidence,
)
from cbrain.dispatch import PreparedDispatch

ZERO = "sha256:" + ("0" * 64)
ONE = "sha256:" + ("1" * 64)


def _prepared() -> PreparedDispatch:
    return PreparedDispatch.capture(
        request_id="request-001",
        dispatch={
            "transport": "https",
            "destination": "payments.example",
            "operation": "POST /v1/payments",
            "wire_content_type": "application/json",
            "wire_content_encoding": "identity",
            "tool_id": "payments.v1",
            "tool_schema_digest": ZERO,
            "tool_artifact_digest": ONE,
            "credential_audience": "payments.example",
            "idempotency_key_digest": ZERO,
            "retry_policy_digest": ONE,
        },
        wire_bytes=b'{"amount":400000}',
        peer_identity_bytes=(b"tls-spki:payments.example:v1"),
    )


def _verified(
    overrides: Mapping[str, Any] | None = None,
) -> VerifiedAuthorization:
    authorization: dict[str, Any] = {
        "organisation_id": "bank.example",
        "execution_authorization_id": ("exec-auth-001"),
        "request_id": "request-001",
        "signature": {
            "key_id": "runtime-key-01",
        },
    }
    authorization.update(overrides or {})

    binding = ExecutionAuthorizationBinding.capture(
        request_id="request-001",
        action={
            "subject_principal": "agent@example",
            "action": "payments.execute",
        },
        prepared_dispatch=_prepared(),
        decision_receipt_digest=ZERO,
        authority_receipt_digest=ONE,
        approval_artifact_digest=ZERO,
        state_snapshot_digest=ZERO,
        policy_bundle_digest=ONE,
        obligations_digest=ONE,
        at_time="2026-07-31T12:00:30Z",
    )

    return VerifiedAuthorization(
        binding=binding,
        evidence=VerifiedEvidence(
            stage="authorization",
            accountable_principal=("runtime@example"),
            reason_code="VALID",
        ),
        _authorization_json=json.dumps(
            authorization,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8"),
        _trust_bundle_json=b'{"trusted":true}',
    )


def test_claim_uses_real_verified_document_shape() -> None:
    observed: list[dict[str, Any]] = []

    def digest(
        document: Mapping[str, Any],
    ) -> object:
        observed.append(dict(document))
        return ZERO

    claim = PrivateVaultAuthorizationUseFactory(digest).create(
        verified_authorization=_verified(),
        claimed_at="2026-07-31T12:00:31Z",
    )

    assert claim.organisation_id == "bank.example"
    assert claim.execution_authorization_id == "exec-auth-001"
    assert claim.authorization_digest == ZERO
    assert claim.request_id == "request-001"
    assert observed[0]["signature"] == {"key_id": "runtime-key-01"}


@pytest.mark.parametrize(
    "overrides",
    [
        {"organisation_id": ""},
        {"execution_authorization_id": ""},
        {"request_id": ""},
        {"request_id": "different-request"},
    ],
)
def test_invalid_signed_identity_fails_closed(
    overrides: Mapping[str, Any],
) -> None:
    factory = PrivateVaultAuthorizationUseFactory(lambda document: ZERO)

    with pytest.raises(PrivateVaultExecutionError):
        factory.create(
            verified_authorization=_verified(overrides),
            claimed_at=("2026-07-31T12:00:31Z"),
        )


def test_digest_failure_does_not_expose_details() -> None:
    def digest(
        document: Mapping[str, Any],
    ) -> object:
        del document
        raise RuntimeError("sensitive signing implementation detail")

    factory = PrivateVaultAuthorizationUseFactory(digest)

    with pytest.raises(
        PrivateVaultExecutionError,
        match="digest failed:RuntimeError",
    ) as raised:
        factory.create(
            verified_authorization=_verified(),
            claimed_at=("2026-07-31T12:00:31Z"),
        )

    assert "sensitive signing" not in str(raised.value)


def test_non_text_digest_fails_closed() -> None:
    factory = PrivateVaultAuthorizationUseFactory(lambda document: None)

    with pytest.raises(
        PrivateVaultExecutionError,
        match="digest is invalid",
    ):
        factory.create(
            verified_authorization=_verified(),
            claimed_at=("2026-07-31T12:00:31Z"),
        )


def test_malformed_digest_fails_closed() -> None:
    factory = PrivateVaultAuthorizationUseFactory(lambda document: "not-a-digest")

    with pytest.raises(
        PrivateVaultExecutionError,
        match="claim is invalid",
    ):
        factory.create(
            verified_authorization=_verified(),
            claimed_at=("2026-07-31T12:00:31Z"),
        )
