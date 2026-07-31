from __future__ import annotations

import json

import pytest

from cbrain import ActionIntent
from cbrain.evaluation import default_catalog
from cbrain.simulators import (
    SIMULATOR_REQUEST_SCHEMA,
    BearerTokenDigestVerifier,
    Contact,
    CRMSimulator,
    SimulatorApplication,
    SimulatorDispatchPlanner,
    SimulatorPlanningError,
    SimulatorTargetBinding,
)

ZERO = "sha256:" + ("0" * 64)
ONE = "sha256:" + ("1" * 64)
PEER = b"tls-cert-sha256:" + (b"a" * 64)


def planner() -> SimulatorDispatchPlanner:
    catalog = default_catalog()
    return SimulatorDispatchPlanner(
        catalog=catalog,
        targets={
            "crm-simulator": SimulatorTargetBinding(PEER, ZERO, ONE),
            "ledger-simulator": SimulatorTargetBinding(PEER, ONE, ZERO),
        },
        tool_schema_digests={capability: ZERO for capability in catalog.capabilities},
    )


def intent(
    *,
    capability: str = "payments.transfer.initiate",
    tool_name: str | None = None,
) -> ActionIntent:
    return ActionIntent.capture(
        request_id="request-1",
        idempotency_key="idempotency-1",
        agent_id="finance-agent",
        framework="test",
        tool_name=tool_name or capability,
        capability=capability,
        timestamp=1_700_000_000.0,
        arguments={
            "account_id": "treasury-primary",
            "beneficiary_id": "beneficiary-17",
            "amount": "50000.00",
            "currency": "USD",
        },
    )


def test_planner_builds_exact_sidecar_ready_request() -> None:
    planned = planner().plan(intent())
    dispatch = planned.prepared.dispatch
    body = json.loads(planned.prepared.wire_bytes)

    assert dispatch["transport"] == "https"
    assert dispatch["destination"] == "ledger-simulator"
    assert dispatch["operation"] == "POST /v1/payments/transfers/initiate"
    assert dispatch["credential_audience"] == "ledger-simulator"
    assert dispatch["tool_artifact_digest"] == ONE
    assert planned.prepared.peer_identity_bytes == PEER
    assert body == {
        "schema": SIMULATOR_REQUEST_SCHEMA,
        "request_id": "request-1",
        "idempotency_key": "idempotency-1",
        "arguments": intent().arguments,
    }
    assert planned.action == {
        "subject_principal": "finance-agent",
        "action": "payments.transfer.initiate",
        "resource": "payments:ledger-simulator",
        "parameters": intent().arguments,
    }


def test_planned_bytes_execute_against_real_simulator_application() -> None:
    action = ActionIntent.capture(
        request_id="request-contact-read",
        idempotency_key="request-contact-read",
        agent_id="crm-agent",
        framework="test",
        tool_name="crm.contact.read",
        capability="crm.contact.read",
        timestamp=1_700_000_000.0,
        arguments={"contact_id": "contact-1"},
    )
    planned = planner().plan(action)
    application = SimulatorApplication(
        simulator=CRMSimulator((Contact("contact-1", "Alice", "alice@example.test"),)),
        operations={
            "POST /v1/crm/contacts/read": "crm.contact.read",
        },
        credential_verifier=BearerTokenDigestVerifier.from_token("target-token"),
    )

    response = application.handle(
        operation=planned.prepared.dispatch["operation"],
        headers={
            "Authorization": "Bearer target-token",
            "Content-Type": planned.prepared.dispatch["wire_content_type"],
            "Content-Encoding": planned.prepared.dispatch["wire_content_encoding"],
        },
        wire_bytes=planned.prepared.wire_bytes,
    )

    assert response.status_code == 200
    assert json.loads(response.body)["result"]["contact"]["contact_id"] == "contact-1"


def test_arguments_cannot_override_route_or_credential() -> None:
    action = ActionIntent.capture(
        request_id="request-route-injection",
        idempotency_key="request-route-injection",
        agent_id="finance-agent",
        framework="test",
        tool_name="payments.balance.read",
        capability="payments.balance.read",
        timestamp=1_700_000_000.0,
        arguments={
            "account_id": "treasury-primary",
            "destination": "attacker.example",
            "credential_audience": "attacker.example",
        },
    )

    planned = planner().plan(action)

    assert planned.prepared.dispatch["destination"] == "ledger-simulator"
    assert planned.prepared.dispatch["credential_audience"] == "ledger-simulator"
    body = json.loads(planned.prepared.wire_bytes)
    assert body["arguments"]["destination"] == "attacker.example"


def test_planner_refuses_unknown_or_mismatched_tool() -> None:
    with pytest.raises(SimulatorPlanningError, match="not in the domain catalog"):
        planner().plan(intent(capability="payments.unknown"))

    with pytest.raises(SimulatorPlanningError, match="tool_name"):
        planner().plan(intent(tool_name="payments.limit.modify"))


def test_planner_requires_complete_deployment_evidence() -> None:
    catalog = default_catalog()

    with pytest.raises(ValueError, match="missing simulator targets"):
        SimulatorDispatchPlanner(
            catalog=catalog,
            targets={
                "ledger-simulator": SimulatorTargetBinding(PEER, ZERO, ONE),
            },
            tool_schema_digests={
                capability: ZERO for capability in catalog.capabilities
            },
        )
