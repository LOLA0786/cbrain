from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from cbrain.adapters.privatevault_claim import (
    PrivateVaultAuthorizationClaimCoordinator,
)
from cbrain.adapters.privatevault_consumption import (
    PrivateVaultAuthorizationUseFactory,
)
from cbrain.adapters.privatevault_execution import (
    AgentDNAVerifiers,
    ExecutionAuthorizationBinding,
    PrivateVaultAgentDNAVerifier,
    PrivateVaultEvidenceRejected,
    PrivateVaultExecutionError,
)
from cbrain.consumption import (
    AuthorizationConsumptionError,
    AuthorizationUse,
)
from cbrain.dispatch import PreparedDispatch

ZERO = "sha256:" + ("0" * 64)
ONE = "sha256:" + ("1" * 64)


@dataclass(frozen=True, slots=True)
class Report:
    ok: bool = True
    evidence_state: str = "VERIFIED"
    decision_conformance: str = "CONFORMANT"
    reason_code: str = "VALID"
    accountable_principal: str | None = "runtime@example"


class Store:
    def __init__(
        self,
        events: list[str],
        *,
        consumed: bool = False,
        claim_result: bool = True,
        lookup_error: Exception | None = None,
        claim_error: Exception | None = None,
    ) -> None:
        self.events = events
        self.consumed = consumed
        self.claim_result = claim_result
        self.lookup_error = lookup_error
        self.claim_error = claim_error
        self.lookup_values: list[AuthorizationUse] = []
        self.claim_values: list[AuthorizationUse] = []

    def is_consumed(
        self,
        authorization: AuthorizationUse,
    ) -> bool:
        self.events.append("lookup")
        self.lookup_values.append(authorization)

        if self.lookup_error is not None:
            raise self.lookup_error

        return self.consumed

    def claim_once(
        self,
        authorization: AuthorizationUse,
    ) -> bool:
        self.events.append("claim")
        self.claim_values.append(authorization)

        if self.claim_error is not None:
            raise self.claim_error

        return self.claim_result


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


def _binding() -> ExecutionAuthorizationBinding:
    return ExecutionAuthorizationBinding.capture(
        request_id="request-001",
        action={
            "subject_principal": "agent@example",
            "subject_key_id": "agent-key-01",
            "action": "payments.execute",
            "resource": "account:4471",
            "parameters": {
                "amount": 400000,
            },
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


def _authorization() -> dict[str, Any]:
    return {
        "organisation_id": "bank.example",
        "execution_authorization_id": ("exec-auth-001"),
        "request_id": "request-001",
        "signature": {
            "key_id": "runtime-key-01",
        },
    }


def _verifier(
    events: list[str],
    *,
    accept_consumed: bool = False,
) -> PrivateVaultAgentDNAVerifier:
    def authorization_verifier(
        *args: object,
        **kwargs: object,
    ) -> object:
        del args
        events.append("verify")

        if kwargs["already_consumed"] is True and not accept_consumed:
            return Report(
                ok=False,
                evidence_state="VERIFIED",
                decision_conformance=("NON_CONFORMANT"),
                reason_code=("EXECUTION_AUTHORIZATION_CONSUMED"),
            )

        return Report()

    def unused_verifier(
        *args: object,
        **kwargs: object,
    ) -> object:
        del args, kwargs
        return Report()

    return PrivateVaultAgentDNAVerifier(
        AgentDNAVerifiers(
            verify_execution_authorization=(authorization_verifier),
            verify_dispatch_witness=unused_verifier,
            verify_closure_chain=unused_verifier,
        )
    )


def _coordinator(
    events: list[str],
    store: Store,
    *,
    accept_consumed: bool = False,
    digests: list[str] | None = None,
) -> PrivateVaultAuthorizationClaimCoordinator:
    digest_values = iter(digests or [ZERO])

    def digest(
        document: Mapping[str, Any],
    ) -> object:
        del document
        events.append("digest")

        try:
            return next(digest_values)
        except StopIteration:
            return ZERO

    return PrivateVaultAuthorizationClaimCoordinator(
        verifier=_verifier(
            events,
            accept_consumed=accept_consumed,
        ),
        use_factory=(PrivateVaultAuthorizationUseFactory(digest)),
        consumption_store=store,
    )


def test_verify_then_atomic_claim_order() -> None:
    events: list[str] = []
    store = Store(events)

    verified = _coordinator(
        events,
        store,
    ).verify_and_claim(
        authorization=_authorization(),
        trust_bundle={"trusted": True},
        binding=_binding(),
    )

    assert events == [
        "digest",
        "lookup",
        "verify",
        "digest",
        "claim",
    ]
    assert verified.authorization["execution_authorization_id"] == "exec-auth-001"
    assert store.lookup_values[0] == (store.claim_values[0])


def test_claim_records_actual_consumption_time() -> None:
    events: list[str] = []
    store = Store(events)

    _coordinator(events, store).verify_and_claim(
        authorization=_authorization(),
        trust_bundle={"trusted": True},
        binding=_binding(),
        claimed_at="2026-07-31T12:00:41Z",
    )

    assert store.lookup_values[0].claimed_at == "2026-07-31T12:00:41Z"
    assert store.claim_values[0].claimed_at == "2026-07-31T12:00:41Z"


def test_previously_consumed_is_rejected_by_agent_dna() -> None:
    events: list[str] = []
    store = Store(
        events,
        consumed=True,
    )

    with pytest.raises(
        PrivateVaultEvidenceRejected,
        match="EXECUTION_AUTHORIZATION_CONSUMED",
    ):
        _coordinator(
            events,
            store,
        ).verify_and_claim(
            authorization=_authorization(),
            trust_bundle={"trusted": True},
            binding=_binding(),
        )

    assert events == [
        "digest",
        "lookup",
        "verify",
    ]
    assert store.claim_values == []


def test_defensive_check_rejects_consumed_even_if_verifier_is_wrong() -> None:
    events: list[str] = []
    store = Store(
        events,
        consumed=True,
    )

    with pytest.raises(
        PrivateVaultEvidenceRejected,
        match="EXECUTION_AUTHORIZATION_CONSUMED",
    ):
        _coordinator(
            events,
            store,
            accept_consumed=True,
        ).verify_and_claim(
            authorization=_authorization(),
            trust_bundle={"trusted": True},
            binding=_binding(),
        )

    assert "claim" not in events


def test_competing_claim_loses_atomic_race() -> None:
    events: list[str] = []
    store = Store(
        events,
        claim_result=False,
    )

    with pytest.raises(
        PrivateVaultEvidenceRejected,
        match="EXECUTION_AUTHORIZATION_CONSUMED",
    ):
        _coordinator(
            events,
            store,
        ).verify_and_claim(
            authorization=_authorization(),
            trust_bundle={"trusted": True},
            binding=_binding(),
        )

    assert events[-1] == "claim"


def test_lookup_failure_stops_before_verification() -> None:
    events: list[str] = []
    store = Store(
        events,
        lookup_error=AuthorizationConsumptionError("lookup unavailable"),
    )

    with pytest.raises(
        AuthorizationConsumptionError,
        match="lookup unavailable",
    ):
        _coordinator(
            events,
            store,
        ).verify_and_claim(
            authorization=_authorization(),
            trust_bundle={"trusted": True},
            binding=_binding(),
        )

    assert events == [
        "digest",
        "lookup",
    ]


def test_claim_failure_occurs_before_any_dispatch() -> None:
    events: list[str] = []
    store = Store(
        events,
        claim_error=AuthorizationConsumptionError("claim unavailable"),
    )

    with pytest.raises(
        AuthorizationConsumptionError,
        match="claim unavailable",
    ):
        _coordinator(
            events,
            store,
        ).verify_and_claim(
            authorization=_authorization(),
            trust_bundle={"trusted": True},
            binding=_binding(),
        )

    assert events == [
        "digest",
        "lookup",
        "verify",
        "digest",
        "claim",
    ]


def test_digest_change_during_verification_fails_closed() -> None:
    events: list[str] = []
    store = Store(events)

    with pytest.raises(
        PrivateVaultExecutionError,
        match="changed during verification",
    ):
        _coordinator(
            events,
            store,
            digests=[ZERO, ONE],
        ).verify_and_claim(
            authorization=_authorization(),
            trust_bundle={"trusted": True},
            binding=_binding(),
        )

    assert "claim" not in events
