from __future__ import annotations

import json
from typing import Any

import pytest

from cbrain import ActionIntent, ExecutionStatus, GovernedExecution, GovernedRuntime
from cbrain.adapters.privatevault import PrivateVaultDecision, PrivateVaultVerdict
from cbrain.adapters.privatevault_execution import ExecutionAuthorizationBinding
from cbrain.dispatch import PreparedDispatch
from cbrain.execution import (
    CredentialHeader,
    DispatchTransportError,
    HandlerNotInvoked,
    IssuedAuthorization,
    PlannedDispatch,
    PrivateVaultExecutionGateway,
    SidecarDispatchService,
    SidecarDispatchTransport,
    TargetResponse,
    WitnessIdentity,
)

ZERO = "sha256:" + ("0" * 64)
ONE = "sha256:" + ("1" * 64)
WIRE_BYTES = b'{"amount":50000,"beneficiary_id":"vendor-17"}'
PEER_BYTES = b"tls-cert-sha256:" + (b"a" * 64)
SECRET = "pv-sidecar-test-secret"


def dispatch_document() -> dict[str, Any]:
    return {
        "transport": "https",
        "destination": "ledger.example",
        "operation": "POST /v1/transfers",
        "wire_content_type": "application/json",
        "wire_content_encoding": "identity",
        "tool_id": "payments.transfer.initiate.v1",
        "tool_schema_digest": ZERO,
        "tool_artifact_digest": ONE,
        "credential_audience": "ledger.example",
        "idempotency_key_digest": ZERO,
        "retry_policy_digest": ONE,
    }


def action_document() -> dict[str, Any]:
    return {
        "subject_principal": "finance-agent@example",
        "action": "payments.transfer.initiate",
        "resource": "ledger:primary",
        "parameters": {
            "amount": 50000,
            "beneficiary_id": "vendor-17",
        },
    }


def prepared() -> PreparedDispatch:
    return PreparedDispatch.capture(
        request_id="request-sidecar-1",
        dispatch=dispatch_document(),
        wire_bytes=WIRE_BYTES,
        peer_identity_bytes=PEER_BYTES,
    )


def binding() -> ExecutionAuthorizationBinding:
    return ExecutionAuthorizationBinding.capture(
        request_id="request-sidecar-1",
        action=action_document(),
        prepared_dispatch=prepared(),
        decision_receipt_digest=ZERO,
        authority_receipt_digest=ONE,
        approval_artifact_digest=None,
        state_snapshot_digest=ZERO,
        policy_bundle_digest=ONE,
        obligations_digest=ZERO,
        at_time="2026-07-31T12:00:30Z",
    )


def intent() -> ActionIntent:
    return ActionIntent.capture(
        request_id="request-sidecar-1",
        idempotency_key="request-sidecar-1",
        agent_id="finance-agent",
        framework="test",
        tool_name="payments.transfer.initiate",
        capability="payments.transfer.initiate",
        timestamp=1_700_000_000.0,
        arguments={"amount": 50000, "beneficiary_id": "vendor-17"},
    )


class Claimant:
    def __init__(self) -> None:
        self.calls = 0
        self.fail = False

    def verify_and_claim(
        self, *, authorization, trust_bundle, binding, claimed_at=None
    ):
        self.calls += 1
        assert claimed_at == "2026-07-31T12:00:41Z"
        if self.fail:
            raise RuntimeError("already consumed")
        return object()


class Credentials:
    def __init__(self) -> None:
        self.audiences: list[str] = []

    def resolve(self, audience: str) -> CredentialHeader:
        self.audiences.append(audience)
        return CredentialHeader("Authorization", f"Bearer {SECRET}")


class Channel:
    def __init__(self, peer_identity: bytes = PEER_BYTES) -> None:
        self.peer_identity_bytes = peer_identity
        self.send_calls = 0
        self.closed = False
        self.sent_bytes: bytes | None = None
        self.credential: CredentialHeader | None = None
        self.fail_after_send = False
        self.output: Any = {"transfer_id": "tx-1"}

    def send_exact(
        self,
        *,
        wire_bytes,
        wire_content_type,
        wire_content_encoding,
        credential,
    ):
        self.send_calls += 1
        self.sent_bytes = wire_bytes
        self.credential = credential
        if self.fail_after_send:
            raise TimeoutError("target result unavailable")
        return TargetResponse(
            dispatch_outcome="ACKNOWLEDGED",
            response_status="200",
            response_bytes=b'{"transfer_id":"tx-1"}',
            effect_state="CONFIRMED",
            output=self.output,
        )

    def close(self) -> None:
        self.closed = True


class Connector:
    def __init__(self, channel: Channel) -> None:
        self.channel = channel
        self.dispatches: list[dict[str, Any]] = []

    def open(self, dispatch):
        self.dispatches.append(dict(dispatch))
        return self.channel


class Closer:
    def __init__(self) -> None:
        self.calls = 0

    def seal(self, **kwargs):
        self.calls += 1
        return {
            "closure_id": "closure-1",
            "request_id": kwargs["authorization"]["request_id"],
            "witness_id": kwargs["witness"]["dispatch_witness_id"],
        }


class RPC:
    def __init__(self, service: SidecarDispatchService) -> None:
        self.service = service
        self.payloads: list[MappingLike] = []

    def call(self, payload):
        self.payloads.append(dict(payload))
        return self.service.handle(payload)


MappingLike = dict[str, Any]


def make_transport(
    *,
    claimant: Claimant | None = None,
    channel: Channel | None = None,
) -> tuple[SidecarDispatchTransport, Claimant, Channel, Credentials, Closer, RPC]:
    actual_claimant = claimant or Claimant()
    actual_channel = channel or Channel()
    credentials = Credentials()
    closer = Closer()

    def sign_witness(metadata, **kwargs):
        assert metadata["observed_at"] == "2026-07-31T12:00:41Z"
        assert kwargs["wire_bytes"] is WIRE_BYTES or kwargs["wire_bytes"] == WIRE_BYTES
        assert kwargs["peer_identity_bytes"] == PEER_BYTES
        return {
            "dispatch_witness_id": metadata["dispatch_witness_id"],
            "request_id": kwargs["authorization"]["request_id"],
            "wire_bytes_seen": len(kwargs["wire_bytes"]),
        }

    service = SidecarDispatchService(
        claimant=actual_claimant,
        connector=Connector(actual_channel),
        credential_provider=credentials,
        witness_signer=sign_witness,
        witness_signing_key=object(),
        identity=WitnessIdentity(
            witness_component_id="dispatcher-sidecar-1",
            signer_key_id="witness-key-1",
            independent=True,
        ),
        closure_writer=closer,
        clock=lambda: "2026-07-31T12:00:41Z",
    )
    rpc = RPC(service)
    transport = SidecarDispatchTransport(
        rpc=rpc,
        identity=WitnessIdentity(
            witness_component_id="dispatcher-sidecar-1",
            signer_key_id="witness-key-1",
            independent=True,
        ),
    )
    return transport, actual_claimant, actual_channel, credentials, closer, rpc


def dispatch_through(transport: SidecarDispatchTransport):
    bound = binding()
    return transport.dispatch(
        authorization={
            "request_id": bound.request_id,
            "execution_authorization_id": "eauth-1",
        },
        trust_bundle={"bundle": "test"},
        action=bound.action,
        prepared=bound.prepared_dispatch,
        witness_id="witness-1",
        observed_at="2026-07-31T12:00:40Z",
        attempt=1,
        binding=bound,
    )


def test_sidecar_sends_exact_authorized_bytes_and_returns_closed_evidence():
    transport, claimant, channel, credentials, closer, rpc = make_transport()

    result = dispatch_through(transport)

    assert channel.sent_bytes == WIRE_BYTES
    assert channel.send_calls == 1
    assert channel.closed is True
    assert claimant.calls == 1
    assert credentials.audiences == ["ledger.example"]
    assert channel.credential is not None
    assert channel.credential.value == f"Bearer {SECRET}"
    assert closer.calls == 1
    assert result.closure == {
        "closure_id": "closure-1",
        "request_id": "request-sidecar-1",
        "witness_id": "witness-1",
    }
    assert result.output == {"transfer_id": "tx-1"}
    assert SECRET not in json.dumps(rpc.payloads)


def test_peer_identity_mismatch_refuses_before_claim_or_send():
    channel = Channel(b"tls-cert-sha256:" + (b"b" * 64))
    transport, claimant, _, _, closer, _ = make_transport(channel=channel)

    with pytest.raises(HandlerNotInvoked, match="pre_dispatch_refused"):
        dispatch_through(transport)

    assert claimant.calls == 0
    assert channel.send_calls == 0
    assert closer.calls == 0


def test_consumed_authorization_refuses_before_send():
    claimant = Claimant()
    claimant.fail = True
    transport, _, channel, _, closer, _ = make_transport(claimant=claimant)

    with pytest.raises(HandlerNotInvoked, match="pre_dispatch_refused"):
        dispatch_through(transport)

    assert channel.send_calls == 0
    assert closer.calls == 0


def test_failure_after_send_is_indeterminate_at_transport_boundary():
    channel = Channel()
    channel.fail_after_send = True
    transport, claimant, _, _, closer, _ = make_transport(channel=channel)

    with pytest.raises(DispatchTransportError, match="post_dispatch_unproven"):
        dispatch_through(transport)

    assert claimant.calls == 1
    assert channel.send_calls == 1
    assert closer.calls == 0


def test_non_json_result_after_send_is_indeterminate_not_retryable():
    channel = Channel()
    channel.output = object()
    transport, claimant, _, _, closer, _ = make_transport(channel=channel)

    with pytest.raises(DispatchTransportError, match="response_unavailable"):
        dispatch_through(transport)

    assert claimant.calls == 1
    assert channel.send_calls == 1
    assert closer.calls == 1


class IndependentGateway:
    independent_execution = True

    def decide_and_execute(self, action, handler):
        return GovernedExecution(
            status=ExecutionStatus.EXECUTED,
            request_id=action.request_id,
            tool_executed=True,
            reason="independent_execution_closed",
            output={"transfer_id": "tx-1"},
        )


def test_runtime_accepts_closed_independent_execution_without_local_handler():
    local_calls = 0

    def local_handler(arguments):
        nonlocal local_calls
        local_calls += 1

    result = GovernedRuntime(IndependentGateway()).execute(intent(), local_handler)

    assert result.status is ExecutionStatus.EXECUTED
    assert result.tool_executed is True
    assert local_calls == 0


def test_unmarked_gateway_cannot_claim_execution_without_local_handler():
    class UnmarkedGateway:
        def decide_and_execute(self, action, handler):
            return GovernedExecution(
                status=ExecutionStatus.EXECUTED,
                request_id=action.request_id,
                tool_executed=True,
                reason="unproven",
            )

    result = GovernedRuntime(UnmarkedGateway()).execute(
        intent(), lambda arguments: pytest.fail("must not run")
    )

    assert result.status is ExecutionStatus.CONTROL_FAILURE
    assert result.tool_executed is False
    assert result.reason == "unproven_execution"


class DecisionClient:
    def decide(self, action):
        return PrivateVaultDecision(
            verdict=PrivateVaultVerdict.ALLOW,
            triggered_by="policy",
            reason="allowed",
            request_id=action.request_id,
            _record_json=b'{"decision_id":"decision-1"}',
        )


class Planner:
    def plan(self, action):
        return PlannedDispatch(action=action_document(), prepared=prepared())


class Issuer:
    def issue(self, *, action, decision, planned):
        return IssuedAuthorization(
            authorization={
                "request_id": action.request_id,
                "execution_authorization_id": "eauth-1",
            },
            trust_bundle={"bundle": "test"},
            binding_digests={
                "decision_receipt_digest": ZERO,
                "authority_receipt_digest": ONE,
                "approval_artifact_digest": None,
                "state_snapshot_digest": ZERO,
                "policy_bundle_digest": ONE,
                "obligations_digest": ZERO,
                "at_time": "2026-07-31T12:00:30Z",
            },
        )


class Verifier:
    def verify_authorization(self, **kwargs):
        return object()

    def verify_dispatch(self, **kwargs):
        return object()

    def verify_closure(self, **kwargs):
        return object()


class NeverLocalClaim:
    def verify_and_claim(self, **kwargs):
        raise AssertionError("independent authorization must be claimed by sidecar")


class NeverLocalClose:
    def seal(self, **kwargs):
        raise AssertionError("independent closure must be signed by sidecar")


def test_concrete_gateway_delegates_claim_send_witness_and_close_to_sidecar():
    transport, claimant, channel, _, closer, _ = make_transport()
    gateway = PrivateVaultExecutionGateway(
        decision_client=DecisionClient(),
        planner=Planner(),
        issuer=Issuer(),
        claim_coordinator=NeverLocalClaim(),
        verifier=Verifier(),
        transport=transport,
        closure_writer=NeverLocalClose(),
        clock=lambda: "2026-07-31T12:00:40Z",
        witness_id_factory=lambda action: "witness-1",
    )
    local_calls = 0

    def local_handler(arguments):
        nonlocal local_calls
        local_calls += 1

    result = GovernedRuntime(gateway).execute(intent(), local_handler)

    assert result.status is ExecutionStatus.EXECUTED, result.reason
    assert result.output == {"transfer_id": "tx-1"}
    assert local_calls == 0
    assert claimant.calls == 1
    assert channel.send_calls == 1
    assert closer.calls == 1
