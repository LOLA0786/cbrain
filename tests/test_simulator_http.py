from __future__ import annotations

import json
from decimal import Decimal

import pytest

from cbrain.simulators import (
    CRM_CONTACT_READ,
    BearerTokenDigestVerifier,
    Contact,
    CRMSimulator,
    LedgerAccount,
    LedgerSimulator,
    SimulatorApplication,
    encode_simulator_request,
)

TOKEN = "sidecar-only-target-token"
OPERATION = "POST /v1/crm/contacts/read"


def application() -> SimulatorApplication:
    return SimulatorApplication(
        simulator=CRMSimulator((Contact("contact-1", "Alice", "alice@example.test"),)),
        operations={OPERATION: CRM_CONTACT_READ},
        credential_verifier=BearerTokenDigestVerifier.from_token(TOKEN),
    )


def headers(token: str = TOKEN) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Content-Encoding": "identity",
    }


def test_http_application_executes_canonical_exact_body() -> None:
    response = application().handle(
        operation=OPERATION,
        headers=headers(),
        wire_bytes=encode_simulator_request(
            request_id="request-1",
            idempotency_key="request-1",
            arguments={"contact_id": "contact-1"},
        ),
    )

    assert response.status_code == 200
    body = json.loads(response.body)
    assert body["schema"] == "cbrain-simulator-effect/v1"
    assert body["capability"] == CRM_CONTACT_READ
    assert body["result"]["contact"]["contact_id"] == "contact-1"


def test_http_application_rejects_missing_sidecar_credential() -> None:
    response = application().handle(
        operation=OPERATION,
        headers={"Content-Type": "application/json"},
        wire_bytes=b"{}",
    )

    assert response.status_code == 401
    assert TOKEN.encode() not in response.body
    assert json.loads(response.body)["reason"] == "INVALID_TARGET_CREDENTIAL"


def test_http_application_rejects_unknown_operation_before_mutation() -> None:
    response = application().handle(
        operation="POST /v1/crm/unknown",
        headers=headers(),
        wire_bytes=encode_simulator_request(
            request_id="request-1",
            idempotency_key="request-1",
            arguments={"contact_id": "contact-1"},
        ),
    )

    assert response.status_code == 404
    assert json.loads(response.body)["reason"] == "OPERATION_NOT_FOUND"


def test_http_application_rejects_extra_request_fields() -> None:
    body = json.loads(
        encode_simulator_request(
            request_id="request-1",
            idempotency_key="request-1",
            arguments={"contact_id": "contact-1"},
        )
    )
    body["credential"] = "must-not-enter-body"

    response = application().handle(
        operation=OPERATION,
        headers=headers(),
        wire_bytes=json.dumps(body).encode(),
    )

    assert response.status_code == 400
    assert json.loads(response.body)["reason"] == "MALFORMED_SIMULATOR_REQUEST"


def test_http_application_rejects_credential_header_injection() -> None:
    verifier = BearerTokenDigestVerifier.from_token(TOKEN)

    assert verifier.verify({"Authorization": f"Bearer {TOKEN}\r\nX-Evil: 1"}) is False


def test_application_refuses_cross_domain_route_configuration() -> None:
    with pytest.raises(ValueError, match="simulator domain"):
        SimulatorApplication(
            simulator=LedgerSimulator(
                (
                    LedgerAccount(
                        "treasury-primary",
                        "USD",
                        Decimal("100.00"),
                        Decimal("50.00"),
                    ),
                )
            ),
            operations={OPERATION: CRM_CONTACT_READ},
            credential_verifier=BearerTokenDigestVerifier.from_token(TOKEN),
        )
