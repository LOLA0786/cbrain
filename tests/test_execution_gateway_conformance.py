"""End-to-end governed execution against the real Agent DNA verifiers.

No doubles for the evidence layer. Real Ed25519 keys, real signing, real
verification, real single-use consumption. This is the test that proves the
gateway can reach EXECUTED and that it refuses to when any stage is broken.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

nacl_signing = pytest.importorskip("nacl.signing")
SigningKey = nacl_signing.SigningKey

authority_v01 = pytest.importorskip("agent_dna.authority_v01")
execution_v01 = pytest.importorskip("agent_dna.execution_v01")
dispatch_v01 = pytest.importorskip("agent_dna.dispatch_v01")
closure_v01 = pytest.importorskip("agent_dna.closure_v01")

from cbrain import ActionIntent, ExecutionStatus, GovernedRuntime  # noqa: E402
from cbrain.adapters.privatevault import (  # noqa: E402
    PrivateVaultDecision,
    PrivateVaultVerdict,
)
from cbrain.adapters.privatevault_claim import (  # noqa: E402
    PrivateVaultAuthorizationClaimCoordinator,
)
from cbrain.adapters.privatevault_consumption import (  # noqa: E402
    PrivateVaultAuthorizationUseFactory,
)
from cbrain.adapters.privatevault_execution import (  # noqa: E402
    AgentDNAVerifiers,
    PrivateVaultAgentDNAVerifier,
)
from cbrain.dispatch import PreparedDispatch  # noqa: E402
from cbrain.execution import (  # noqa: E402
    DispatchTransportError,
    InProcessDispatchTransport,
    IssuedAuthorization,
    PlannedDispatch,
    PrivateVaultExecutionGateway,
    WitnessIdentity,
)

CANONICALIZATION = authority_v01.CANONICALIZATION
TRUST_SPEC = authority_v01.TRUST_SPEC
sha256_digest = authority_v01.sha256_digest
encode_public_key = authority_v01.encode_public_key
sha256_bytes_digest = execution_v01.sha256_bytes_digest

ZERO = "sha256:" + ("0" * 64)
ONE = "sha256:" + ("1" * 64)
ORG = "store.example"

WIRE_BYTES = b'{"account":"4471","amount":400000,"currency":"INR"}'
PEER_BYTES = b"tls-spki:payments.store.example:v3"


class Keyring:
    def __init__(self) -> None:
        self.execution = SigningKey.generate()
        self.witness = SigningKey.generate()
        self.closure = SigningKey.generate()

    def trust_bundle(self) -> dict[str, Any]:
        return {
            "spec": TRUST_SPEC,
            "canonicalization": CANONICALIZATION,
            "organisation_id": ORG,
            "bundle_version": 1,
            "pinned_at": "2026-07-31T11:00:00Z",
            "keys": [
                {
                    "key_id": "execution-signer-01",
                    "principal": f"execution-runtime@{ORG}",
                    "algorithm": "ed25519",
                    "public_key": encode_public_key(self.execution),
                    "usages": ["execution_authorization_signer"],
                },
                {
                    "key_id": "witness-signer-01",
                    "principal": f"dispatcher@{ORG}",
                    "algorithm": "ed25519",
                    "public_key": encode_public_key(self.witness),
                    "usages": ["dispatch_witness_signer"],
                },
                {
                    "key_id": "closure-signer-01",
                    "principal": f"closer@{ORG}",
                    "algorithm": "ed25519",
                    "public_key": encode_public_key(self.closure),
                    "usages": ["closure_signer"],
                },
            ],
        }


def action_document() -> dict[str, Any]:
    return {
        "subject_principal": f"refund-agent@{ORG}",
        "subject_key_id": "refund-agent-01",
        "action": "payments.refund",
        "resource": "account:4471",
        "parameters": {
            "amount": {"minor_units": 400000, "currency": "INR"},
            "case_id": "case-7821",
        },
    }


def dispatch_document() -> dict[str, Any]:
    return {
        "transport": "https",
        "destination": "payments.store.example",
        "operation": "POST /v1/refunds",
        "wire_content_type": "application/json",
        "wire_content_encoding": "identity",
        "tool_id": "payments.refund.v3",
        "tool_schema_digest": ZERO,
        "tool_artifact_digest": ONE,
        "credential_audience": "payments.store.example",
        "idempotency_key_digest": ZERO,
        "retry_policy_digest": ONE,
    }


def intent() -> ActionIntent:
    return ActionIntent.capture(
        request_id="request-001",
        idempotency_key="request-001",
        agent_id="refund-agent-01",
        framework="test",
        tool_name="payments.refund",
        capability="payments.refund",
        timestamp=1_700_000_000.0,
        arguments={"amount": 400000},
    )


class StubDecisionClient:
    def __init__(self, verdict: PrivateVaultVerdict) -> None:
        self.verdict = verdict

    def decide(self, action: ActionIntent) -> PrivateVaultDecision:
        return PrivateVaultDecision(
            verdict=self.verdict,
            triggered_by="policy",
            reason="test",
            request_id=action.request_id,
            _record_json=json.dumps({"decision_id": "decision-1"}).encode(),
        )


class Planner:
    def plan(self, action: ActionIntent) -> PlannedDispatch:
        return PlannedDispatch(
            action=action_document(),
            prepared=PreparedDispatch.capture(
                request_id=action.request_id,
                dispatch=dispatch_document(),
                wire_bytes=WIRE_BYTES,
                peer_identity_bytes=PEER_BYTES,
            ),
        )


class Issuer:
    """Stands in for the /v1/authorize endpoint, using the real signer."""

    def __init__(self, keys: Keyring, authorization_id: str = "eauth-001") -> None:
        self._keys = keys
        self._authorization_id = authorization_id

    def issue(self, *, action, decision, planned) -> IssuedAuthorization:
        act = planned.action
        unsigned = {
            "spec": execution_v01.EXECUTION_AUTHORIZATION_SPEC,
            "canonicalization": CANONICALIZATION,
            "execution_authorization_id": self._authorization_id,
            "organisation_id": ORG,
            "request_id": action.request_id,
            "issued_at": "2026-07-31T12:00:00Z",
            "not_before": "2026-07-31T12:00:00Z",
            "expires_at": "2026-07-31T12:01:00Z",
            "nonce": "execution-nonce-0001",
            "decision_receipt_digest": ZERO,
            "authority_receipt_digest": ONE,
            "approval_artifact_digest": ZERO,
            "action": act,
            "action_digest": sha256_digest(act),
            "expected_wire_bytes_digest": sha256_bytes_digest(WIRE_BYTES),
            "expected_wire_bytes_length": len(WIRE_BYTES),
            "expected_peer_identity_digest": sha256_bytes_digest(PEER_BYTES),
            "dispatch": planned.prepared.dispatch,
            "state_snapshot_digest": ZERO,
            "policy_bundle_digest": ONE,
            "trust_bundle_digest": sha256_digest(self._keys.trust_bundle()),
            "obligations_digest": ZERO,
            "max_uses": 1,
            "signer_key_id": "execution-signer-01",
        }

        return IssuedAuthorization(
            authorization=execution_v01.sign_execution_authorization(
                unsigned, self._keys.execution
            ),
            trust_bundle=self._keys.trust_bundle(),
            binding_digests={
                "decision_receipt_digest": ZERO,
                "authority_receipt_digest": ONE,
                "approval_artifact_digest": ZERO,
                "state_snapshot_digest": ZERO,
                "policy_bundle_digest": ONE,
                "obligations_digest": ZERO,
                "at_time": "2026-07-31T12:00:30Z",
            },
        )


class Closer:
    def __init__(self, keys: Keyring) -> None:
        self._keys = keys

    def seal(
        self,
        *,
        authorization,
        witness,
        trust_bundle,
        dispatch_outcome,
        response_status,
        response_bytes,
        effect_state,
        idempotency_key_digest,
    ):
        unsigned = {
            "spec": closure_v01.CLOSURE_RECORD_SPEC,
            "canonicalization": CANONICALIZATION,
            "closure_id": "closure-001",
            "organisation_id": authorization["organisation_id"],
            "request_id": authorization["request_id"],
            "execution_authorization_id": (authorization["execution_authorization_id"]),
            "execution_authorization_digest": (
                execution_v01.execution_authorization_digest(authorization)
            ),
            "dispatch_witness_id": witness["dispatch_witness_id"],
            "dispatch_witness_digest": (dispatch_v01.dispatch_witness_digest(witness)),
            "closed_at": "2026-07-31T12:00:45Z",
            "closure_component_id": "closer-01",
            "dispatch_outcome": dispatch_outcome,
            "response_status": response_status,
            "response_bytes_digest": sha256_bytes_digest(response_bytes),
            "response_bytes_length": len(response_bytes),
            "effect_state": effect_state,
            "effect_evidence_digest": ZERO,
            "idempotency_key_digest": idempotency_key_digest,
            "authorization_use_count": 1,
            "trust_bundle_digest": sha256_digest(trust_bundle),
            "signer_key_id": "closure-signer-01",
        }
        return closure_v01.sign_closure_record(unsigned, self._keys.closure)


class MemoryConsumptionStore:
    def __init__(self) -> None:
        self._claimed: set[tuple[str, str]] = set()
        self._digests: set[str] = set()

    def is_consumed(self, authorization) -> bool:
        key = (
            authorization.organisation_id,
            authorization.execution_authorization_id,
        )
        return key in self._claimed or (
            authorization.authorization_digest in self._digests
        )

    def claim_once(self, authorization) -> bool:
        key = (
            authorization.organisation_id,
            authorization.execution_authorization_id,
        )
        if key in self._claimed or authorization.authorization_digest in self._digests:
            return False
        self._claimed.add(key)
        self._digests.add(authorization.authorization_digest)
        return True


def build_gateway(
    keys: Keyring,
    store: MemoryConsumptionStore,
    verdict: PrivateVaultVerdict = PrivateVaultVerdict.ALLOW,
    independent_witness: bool = False,
) -> PrivateVaultExecutionGateway:
    verifiers = AgentDNAVerifiers(
        verify_execution_authorization=(execution_v01.verify_execution_authorization),
        verify_dispatch_witness=dispatch_v01.verify_dispatch_witness,
        verify_closure_chain=closure_v01.verify_closure_chain,
    )
    verifier = PrivateVaultAgentDNAVerifier(verifiers)

    return PrivateVaultExecutionGateway(
        decision_client=StubDecisionClient(verdict),
        planner=Planner(),
        issuer=Issuer(keys),
        claim_coordinator=PrivateVaultAuthorizationClaimCoordinator(
            verifier=verifier,
            use_factory=PrivateVaultAuthorizationUseFactory(
                execution_v01.execution_authorization_digest
            ),
            consumption_store=store,
        ),
        verifier=verifier,
        transport=InProcessDispatchTransport(
            handler_runner=lambda arguments: None,
            witness_signer=dispatch_v01.create_dispatch_witness,
            signing_key=keys.witness,
            identity=WitnessIdentity(
                witness_component_id="dispatcher-01",
                signer_key_id="witness-signer-01",
                independent=independent_witness,
            ),
        ),
        closure_writer=Closer(keys),
        clock=lambda: "2026-07-31T12:00:40Z",
        witness_id_factory=lambda action: "witness-001",
    )


def test_allow_reaches_executed_through_the_real_evidence_chain():
    keys = Keyring()
    store = MemoryConsumptionStore()
    calls: list[Any] = []

    result = GovernedRuntime(build_gateway(keys, store)).execute(
        intent(),
        lambda arguments: calls.append(arguments) or {"refund_id": "rf-1"},
    )

    assert result.status is ExecutionStatus.EXECUTED, result.reason
    assert result.tool_executed is True
    assert result.output == {"refund_id": "rf-1"}
    assert len(calls) == 1


def test_block_never_reaches_the_handler():
    keys = Keyring()
    store = MemoryConsumptionStore()
    calls: list[Any] = []

    result = GovernedRuntime(
        build_gateway(keys, store, PrivateVaultVerdict.BLOCK)
    ).execute(intent(), lambda arguments: calls.append(arguments))

    assert result.status is ExecutionStatus.BLOCKED
    assert result.tool_executed is False
    assert calls == []


def test_require_approval_never_reaches_the_handler():
    keys = Keyring()
    store = MemoryConsumptionStore()
    calls: list[Any] = []

    result = GovernedRuntime(
        build_gateway(keys, store, PrivateVaultVerdict.REQUIRE_APPROVAL)
    ).execute(intent(), lambda arguments: calls.append(arguments))

    assert result.status is ExecutionStatus.REVIEW_REQUIRED
    assert result.tool_executed is False
    assert calls == []


def test_replayed_authorization_is_refused_before_execution():
    keys = Keyring()
    store = MemoryConsumptionStore()
    calls: list[Any] = []

    first = GovernedRuntime(build_gateway(keys, store)).execute(
        intent(), lambda arguments: calls.append(arguments)
    )
    assert first.status is ExecutionStatus.EXECUTED

    second = GovernedRuntime(build_gateway(keys, store)).execute(
        intent(), lambda arguments: calls.append(arguments)
    )

    assert second.status is ExecutionStatus.CONTROL_FAILURE
    assert second.tool_executed is False
    assert len(calls) == 1, "a replayed permit must never re-enter the handler"


def test_handler_failure_is_indeterminate_and_not_retryable():
    keys = Keyring()
    store = MemoryConsumptionStore()

    def exploding(arguments):
        raise RuntimeError("target unreachable after send")

    result = GovernedRuntime(build_gateway(keys, store)).execute(intent(), exploding)

    assert result.status is ExecutionStatus.INDETERMINATE
    assert result.tool_executed is None
    assert result.retryable is False


def test_tampered_wire_bytes_are_refused():
    keys = Keyring()
    store = MemoryConsumptionStore()
    calls: list[Any] = []

    gateway = build_gateway(keys, store)

    class TamperedPlanner:
        def plan(self, action):
            return PlannedDispatch(
                action=action_document(),
                prepared=PreparedDispatch.capture(
                    request_id=action.request_id,
                    dispatch=dispatch_document(),
                    wire_bytes=b'{"account":"4471","amount":900000}',
                    peer_identity_bytes=PEER_BYTES,
                ),
            )

    gateway._planner = TamperedPlanner()

    result = GovernedRuntime(gateway).execute(
        intent(), lambda arguments: calls.append(arguments)
    )

    assert result.status is ExecutionStatus.CONTROL_FAILURE
    assert calls == [], "bytes not covered by the permit must never dispatch"


def test_in_process_transport_refuses_to_claim_independence():
    keys = Keyring()
    with pytest.raises(DispatchTransportError):
        InProcessDispatchTransport(
            handler_runner=lambda arguments: None,
            witness_signer=dispatch_v01.create_dispatch_witness,
            signing_key=keys.witness,
            identity=WitnessIdentity(
                witness_component_id="dispatcher-01",
                signer_key_id="witness-signer-01",
                independent=True,
            ),
        )
