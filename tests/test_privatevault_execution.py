from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from cbrain.adapters.privatevault_execution import (
    AgentDNAVerifiers,
    ExecutionAuthorizationBinding,
    PrivateVaultAgentDNAVerifier,
    PrivateVaultEvidenceRejected,
    PrivateVaultExecutionError,
)
from cbrain.contracts import ContractError
from cbrain.dispatch import PreparedDispatch

ZERO = "sha256:" + ("0" * 64)
ONE = "sha256:" + ("1" * 64)

Call = tuple[tuple[object, ...], dict[str, object]]
Verifier = Callable[..., object]


@dataclass(frozen=True, slots=True)
class Report:
    ok: bool = True
    evidence_state: str = "VERIFIED"
    decision_conformance: str = "CONFORMANT"
    reason_code: str = "VALID"
    accountable_principal: str | None = "component@example"


def _prepared(
    request_id: str = "request-001",
) -> PreparedDispatch:
    return PreparedDispatch.capture(
        request_id=request_id,
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
        peer_identity_bytes=b"tls-spki:payments.example:v1",
    )


def _action() -> dict[str, Any]:
    return {
        "subject_principal": "agent@example",
        "subject_key_id": "agent-key-01",
        "action": "payments.execute",
        "resource": "account:4471",
        "parameters": {
            "amount": 400000,
            "currency": "INR",
        },
    }


def _binding(
    *,
    request_id: str = "request-001",
    prepared: PreparedDispatch | None = None,
) -> ExecutionAuthorizationBinding:
    return ExecutionAuthorizationBinding.capture(
        request_id=request_id,
        action=_action(),
        prepared_dispatch=prepared or _prepared(request_id),
        decision_receipt_digest=ZERO,
        authority_receipt_digest=ONE,
        approval_artifact_digest=ZERO,
        state_snapshot_digest=ZERO,
        policy_bundle_digest=ONE,
        obligations_digest=ONE,
        at_time="2026-07-31T12:00:30Z",
    )


def _recording_verifier(
    *,
    stage: str,
    report: object,
    calls: dict[str, list[Call]],
) -> Verifier:
    def invoke(
        *args: object,
        **kwargs: object,
    ) -> object:
        calls[stage].append((args, kwargs))
        return report

    return invoke


def _suite(
    *,
    authorization_report: object | None = None,
    dispatch_report: object | None = None,
    closure_report: object | None = None,
) -> tuple[
    PrivateVaultAgentDNAVerifier,
    dict[str, list[Call]],
]:
    calls: dict[str, list[Call]] = {
        "authorization": [],
        "dispatch": [],
        "closure": [],
    }

    verifiers = AgentDNAVerifiers(
        verify_execution_authorization=_recording_verifier(
            stage="authorization",
            report=authorization_report or Report(),
            calls=calls,
        ),
        verify_dispatch_witness=_recording_verifier(
            stage="dispatch",
            report=dispatch_report or Report(),
            calls=calls,
        ),
        verify_closure_chain=_recording_verifier(
            stage="closure",
            report=closure_report or Report(),
            calls=calls,
        ),
    )

    return PrivateVaultAgentDNAVerifier(verifiers), calls


def test_complete_chain_uses_exact_frozen_values() -> None:
    verifier, calls = _suite()
    binding = _binding()

    authorization = verifier.verify_authorization(
        authorization={"signed": "authorization"},
        trust_bundle={"signed": "trust-bundle"},
        binding=binding,
        already_consumed=False,
    )
    dispatch = verifier.verify_dispatch(
        verified_authorization=authorization,
        witness={"signed": "witness"},
    )
    closure = verifier.verify_closure(
        verified_dispatch=dispatch,
        closure={"signed": "closure"},
    )

    authorization_args, authorization_kwargs = calls["authorization"][0]

    assert authorization_args == (
        {"signed": "authorization"},
        {"signed": "trust-bundle"},
    )
    assert authorization_kwargs["expected_request_id"] == "request-001"
    assert authorization_kwargs["expected_action"] == _action()
    assert (
        authorization_kwargs["expected_dispatch"] == binding.prepared_dispatch.dispatch
    )
    assert (
        authorization_kwargs["expected_wire_bytes"]
        == binding.prepared_dispatch.wire_bytes
    )
    assert (
        authorization_kwargs["expected_peer_identity_bytes"]
        == binding.prepared_dispatch.peer_identity_bytes
    )
    assert authorization_kwargs["already_consumed"] is False

    dispatch_args, dispatch_kwargs = calls["dispatch"][0]

    assert dispatch_args == (
        {"signed": "witness"},
        {"signed": "authorization"},
        {"signed": "trust-bundle"},
    )
    assert dispatch_kwargs["wire_bytes"] == (binding.prepared_dispatch.wire_bytes)
    assert dispatch_kwargs["peer_identity_bytes"] == (
        binding.prepared_dispatch.peer_identity_bytes
    )

    closure_args, closure_kwargs = calls["closure"][0]

    assert closure_args == (
        {"signed": "authorization"},
        {"signed": "witness"},
        {"signed": "closure"},
        {"signed": "trust-bundle"},
    )
    assert closure_kwargs["require_witness_independence"] is True
    assert closure.closure == {"signed": "closure"}


def test_binding_snapshots_action_before_authorization() -> None:
    action = _action()

    binding = ExecutionAuthorizationBinding.capture(
        request_id="request-001",
        action=action,
        prepared_dispatch=_prepared(),
        decision_receipt_digest=ZERO,
        authority_receipt_digest=ONE,
        approval_artifact_digest=ZERO,
        state_snapshot_digest=ZERO,
        policy_bundle_digest=ONE,
        obligations_digest=ONE,
        at_time="2026-07-31T12:00:30Z",
    )

    action["parameters"]["amount"] = 999999999

    assert binding.action["parameters"]["amount"] == 400000


@pytest.mark.parametrize(
    "report",
    [
        Report(
            ok=False,
            evidence_state="INVALID",
            decision_conformance="NOT_ASSESSABLE",
            reason_code="SIGNATURE_INVALID",
        ),
        Report(
            ok=False,
            evidence_state="UNVERIFIABLE",
            decision_conformance="NOT_ASSESSABLE",
            reason_code="TRUST_BUNDLE_UNAVAILABLE",
        ),
        Report(
            ok=False,
            evidence_state="VERIFIED",
            decision_conformance="NON_CONFORMANT",
            reason_code="ACTION_MISMATCH",
        ),
    ],
)
def test_authorization_requires_verified_and_conformant(
    report: Report,
) -> None:
    verifier, _ = _suite(
        authorization_report=report,
    )

    with pytest.raises(PrivateVaultEvidenceRejected) as raised:
        verifier.verify_authorization(
            authorization={"signed": "authorization"},
            trust_bundle={"signed": "trust-bundle"},
            binding=_binding(),
            already_consumed=False,
        )

    assert raised.value.reason_code == report.reason_code


def test_missing_trust_bundle_fails_closed() -> None:
    verifier, _ = _suite()

    with pytest.raises(
        PrivateVaultEvidenceRejected,
        match="TRUST_BUNDLE_UNAVAILABLE",
    ):
        verifier.verify_authorization(
            authorization={"signed": "authorization"},
            trust_bundle=None,
            binding=_binding(),
            already_consumed=False,
        )


def test_missing_dispatch_witness_fails_closed() -> None:
    verifier, _ = _suite()

    authorization = verifier.verify_authorization(
        authorization={"signed": "authorization"},
        trust_bundle={"signed": "trust-bundle"},
        binding=_binding(),
        already_consumed=False,
    )

    with pytest.raises(
        PrivateVaultEvidenceRejected,
        match="DISPATCH_WITNESS_ABSENT",
    ):
        verifier.verify_dispatch(
            verified_authorization=authorization,
            witness=None,
        )


def test_missing_closure_record_fails_closed() -> None:
    verifier, _ = _suite()

    authorization = verifier.verify_authorization(
        authorization={"signed": "authorization"},
        trust_bundle={"signed": "trust-bundle"},
        binding=_binding(),
        already_consumed=False,
    )
    dispatch = verifier.verify_dispatch(
        verified_authorization=authorization,
        witness={"signed": "witness"},
    )

    with pytest.raises(
        PrivateVaultEvidenceRejected,
        match="CLOSURE_RECORD_ABSENT",
    ):
        verifier.verify_closure(
            verified_dispatch=dispatch,
            closure=None,
        )


def test_verifier_exception_becomes_safe_control_error() -> None:
    def explode(
        *args: object,
        **kwargs: object,
    ) -> object:
        del args, kwargs
        raise RuntimeError("sensitive internal failure")

    verifier = PrivateVaultAgentDNAVerifier(
        AgentDNAVerifiers(
            verify_execution_authorization=explode,
            verify_dispatch_witness=explode,
            verify_closure_chain=explode,
        )
    )

    with pytest.raises(
        PrivateVaultExecutionError,
        match="authorization_verification_failed:RuntimeError",
    ) as raised:
        verifier.verify_authorization(
            authorization={"signed": "authorization"},
            trust_bundle={"signed": "trust-bundle"},
            binding=_binding(),
            already_consumed=False,
        )

    assert "sensitive internal failure" not in str(raised.value)


def test_consumed_authorization_state_is_forwarded() -> None:
    verifier, calls = _suite(
        authorization_report=Report(
            ok=False,
            evidence_state="VERIFIED",
            decision_conformance="NON_CONFORMANT",
            reason_code="EXECUTION_AUTHORIZATION_CONSUMED",
        )
    )

    with pytest.raises(
        PrivateVaultEvidenceRejected,
        match="EXECUTION_AUTHORIZATION_CONSUMED",
    ):
        verifier.verify_authorization(
            authorization={"signed": "authorization"},
            trust_bundle={"signed": "trust-bundle"},
            binding=_binding(),
            already_consumed=True,
        )

    _, kwargs = calls["authorization"][0]
    assert kwargs["already_consumed"] is True


def test_request_id_mismatch_is_rejected_at_capture() -> None:
    with pytest.raises(
        ContractError,
        match="request_id mismatch",
    ):
        _binding(
            request_id="request-001",
            prepared=_prepared("request-002"),
        )


def test_malformed_digest_is_rejected_at_capture() -> None:
    with pytest.raises(
        ContractError,
        match="decision_receipt_digest",
    ):
        ExecutionAuthorizationBinding.capture(
            request_id="request-001",
            action=_action(),
            prepared_dispatch=_prepared(),
            decision_receipt_digest="not-a-digest",
            authority_receipt_digest=ONE,
            approval_artifact_digest=ZERO,
            state_snapshot_digest=ZERO,
            policy_bundle_digest=ONE,
            obligations_digest=ONE,
            at_time="2026-07-31T12:00:30Z",
        )
