from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any, cast

from cbrain.contracts import (
    ContractError,
    _restore_object,
    _snapshot_object,
)
from cbrain.dispatch import PreparedDispatch

Verifier = Callable[..., object]


class PrivateVaultExecutionError(RuntimeError):
    """Base failure for Agent DNA execution verification."""


class PrivateVaultAgentDNAUnavailable(PrivateVaultExecutionError):
    """The pinned Agent DNA implementation cannot be loaded."""


class PrivateVaultEvidenceRejected(PrivateVaultExecutionError):
    """Agent DNA rejected an execution-evidence stage."""

    def __init__(
        self,
        *,
        stage: str,
        evidence_state: str,
        decision_conformance: str,
        reason_code: str,
    ) -> None:
        self.stage = stage
        self.evidence_state = evidence_state
        self.decision_conformance = decision_conformance
        self.reason_code = reason_code

        super().__init__(
            f"{stage}_rejected:{evidence_state}:{decision_conformance}:{reason_code}"
        )


@dataclass(frozen=True, slots=True)
class AgentDNAVerifiers:
    """Real verifier functions loaded from PrivateVault Agent DNA."""

    verify_execution_authorization: Verifier
    verify_dispatch_witness: Verifier
    verify_closure_chain: Verifier

    @classmethod
    def load(cls) -> AgentDNAVerifiers:
        try:
            execution = import_module("agent_dna.execution_v01")
            dispatch = import_module("agent_dna.dispatch_v01")
            closure = import_module("agent_dna.closure_v01")
        except (ImportError, ModuleNotFoundError) as exc:
            raise PrivateVaultAgentDNAUnavailable(
                "pinned PrivateVault Agent DNA is unavailable"
            ) from exc

        return cls(
            verify_execution_authorization=_required_callable(
                execution,
                "verify_execution_authorization",
            ),
            verify_dispatch_witness=_required_callable(
                dispatch,
                "verify_dispatch_witness",
            ),
            verify_closure_chain=_required_callable(
                closure,
                "verify_closure_chain",
            ),
        )


@dataclass(frozen=True, slots=True)
class ExecutionAuthorizationBinding:
    """Immutable expectations for one authorized dispatch."""

    request_id: str
    prepared_dispatch: PreparedDispatch
    _action_json: bytes
    decision_receipt_digest: str
    authority_receipt_digest: str
    approval_artifact_digest: str | None
    state_snapshot_digest: str
    policy_bundle_digest: str
    obligations_digest: str
    at_time: str

    def __post_init__(self) -> None:
        if self.request_id != self.prepared_dispatch.request_id:
            raise ContractError("authorization and dispatch request_id mismatch")

        if not isinstance(self._action_json, bytes):
            raise ContractError("_action_json must be bytes")

        _restore_object(self._action_json)

        for name in (
            "decision_receipt_digest",
            "authority_receipt_digest",
            "state_snapshot_digest",
            "policy_bundle_digest",
            "obligations_digest",
        ):
            _require_digest(getattr(self, name), name)

        if self.approval_artifact_digest is not None:
            _require_digest(
                self.approval_artifact_digest,
                "approval_artifact_digest",
            )

        if not isinstance(self.at_time, str) or not self.at_time.strip():
            raise ContractError("at_time must be non-empty RFC3339 text")

    @classmethod
    def capture(
        cls,
        *,
        request_id: str,
        action: Mapping[str, Any],
        prepared_dispatch: PreparedDispatch,
        decision_receipt_digest: str,
        authority_receipt_digest: str,
        approval_artifact_digest: str | None,
        state_snapshot_digest: str,
        policy_bundle_digest: str,
        obligations_digest: str,
        at_time: str,
    ) -> ExecutionAuthorizationBinding:
        return cls(
            request_id=request_id,
            prepared_dispatch=prepared_dispatch,
            _action_json=_snapshot_object(
                action,
                "execution_action",
            ),
            decision_receipt_digest=decision_receipt_digest,
            authority_receipt_digest=authority_receipt_digest,
            approval_artifact_digest=approval_artifact_digest,
            state_snapshot_digest=state_snapshot_digest,
            policy_bundle_digest=policy_bundle_digest,
            obligations_digest=obligations_digest,
            at_time=at_time,
        )

    @property
    def action(self) -> dict[str, Any]:
        return _restore_object(self._action_json)


@dataclass(frozen=True, slots=True)
class VerifiedEvidence:
    stage: str
    accountable_principal: str | None
    reason_code: str


@dataclass(frozen=True, slots=True)
class VerifiedAuthorization:
    binding: ExecutionAuthorizationBinding
    evidence: VerifiedEvidence
    _authorization_json: bytes
    _trust_bundle_json: bytes

    @property
    def authorization(self) -> dict[str, Any]:
        return _restore_object(self._authorization_json)

    @property
    def trust_bundle(self) -> dict[str, Any]:
        return _restore_object(self._trust_bundle_json)


@dataclass(frozen=True, slots=True)
class VerifiedDispatch:
    authorization: VerifiedAuthorization
    evidence: VerifiedEvidence
    _witness_json: bytes

    @property
    def witness(self) -> dict[str, Any]:
        return _restore_object(self._witness_json)


@dataclass(frozen=True, slots=True)
class VerifiedClosure:
    dispatch: VerifiedDispatch
    evidence: VerifiedEvidence
    _closure_json: bytes

    @property
    def closure(self) -> dict[str, Any]:
        return _restore_object(self._closure_json)


class PrivateVaultAgentDNAVerifier:
    """Verify authorization, dispatch, and closure in strict order."""

    def __init__(
        self,
        verifiers: AgentDNAVerifiers | None = None,
    ) -> None:
        self._verifiers = verifiers or AgentDNAVerifiers.load()

    def verify_authorization(
        self,
        *,
        authorization: Mapping[str, Any],
        trust_bundle: Mapping[str, Any] | None,
        binding: ExecutionAuthorizationBinding,
        already_consumed: bool,
    ) -> VerifiedAuthorization:
        authorization_json = _snapshot_object(
            authorization,
            "execution_authorization",
        )
        authorization_document = _restore_object(authorization_json)

        trust_bundle_json = (
            None
            if trust_bundle is None
            else _snapshot_object(
                trust_bundle,
                "trust_bundle",
            )
        )
        trust_bundle_document = (
            None if trust_bundle_json is None else _restore_object(trust_bundle_json)
        )

        prepared = binding.prepared_dispatch

        report = self._invoke(
            "authorization",
            self._verifiers.verify_execution_authorization,
            authorization_document,
            trust_bundle_document,
            expected_request_id=binding.request_id,
            expected_action=binding.action,
            expected_dispatch=prepared.dispatch,
            expected_decision_receipt_digest=(binding.decision_receipt_digest),
            expected_authority_receipt_digest=(binding.authority_receipt_digest),
            expected_approval_artifact_digest=(binding.approval_artifact_digest),
            expected_state_snapshot_digest=(binding.state_snapshot_digest),
            expected_policy_bundle_digest=(binding.policy_bundle_digest),
            expected_obligations_digest=(binding.obligations_digest),
            expected_wire_bytes=prepared.wire_bytes,
            expected_peer_identity_bytes=(prepared.peer_identity_bytes),
            at_time=binding.at_time,
            already_consumed=already_consumed,
        )

        evidence = _require_accepted(
            "authorization",
            report,
        )

        if trust_bundle_json is None:
            raise PrivateVaultEvidenceRejected(
                stage="authorization",
                evidence_state="UNVERIFIABLE",
                decision_conformance="NOT_ASSESSABLE",
                reason_code="TRUST_BUNDLE_UNAVAILABLE",
            )

        return VerifiedAuthorization(
            binding=binding,
            evidence=evidence,
            _authorization_json=authorization_json,
            _trust_bundle_json=trust_bundle_json,
        )

    def verify_dispatch(
        self,
        *,
        verified_authorization: VerifiedAuthorization,
        witness: Mapping[str, Any] | None,
    ) -> VerifiedDispatch:
        witness_json = (
            None
            if witness is None
            else _snapshot_object(
                witness,
                "dispatch_witness",
            )
        )
        witness_document = (
            None if witness_json is None else _restore_object(witness_json)
        )

        binding = verified_authorization.binding
        prepared = binding.prepared_dispatch

        report = self._invoke(
            "dispatch",
            self._verifiers.verify_dispatch_witness,
            witness_document,
            verified_authorization.authorization,
            verified_authorization.trust_bundle,
            observed_action=binding.action,
            observed_dispatch=prepared.dispatch,
            wire_bytes=prepared.wire_bytes,
            peer_identity_bytes=(prepared.peer_identity_bytes),
        )

        evidence = _require_accepted(
            "dispatch",
            report,
        )

        if witness_json is None:
            raise PrivateVaultEvidenceRejected(
                stage="dispatch",
                evidence_state="ABSENT",
                decision_conformance="NOT_ASSESSABLE",
                reason_code="DISPATCH_WITNESS_ABSENT",
            )

        return VerifiedDispatch(
            authorization=verified_authorization,
            evidence=evidence,
            _witness_json=witness_json,
        )

    def verify_closure(
        self,
        *,
        verified_dispatch: VerifiedDispatch,
        closure: Mapping[str, Any] | None,
    ) -> VerifiedClosure:
        closure_json = (
            None
            if closure is None
            else _snapshot_object(
                closure,
                "closure_record",
            )
        )
        closure_document = (
            None if closure_json is None else _restore_object(closure_json)
        )

        verified_authorization = verified_dispatch.authorization

        report = self._invoke(
            "closure",
            self._verifiers.verify_closure_chain,
            verified_authorization.authorization,
            verified_dispatch.witness,
            closure_document,
            verified_authorization.trust_bundle,
            require_witness_independence=True,
        )

        evidence = _require_accepted(
            "closure",
            report,
        )

        if closure_json is None:
            raise PrivateVaultEvidenceRejected(
                stage="closure",
                evidence_state="ABSENT",
                decision_conformance="NOT_ASSESSABLE",
                reason_code="CLOSURE_RECORD_ABSENT",
            )

        return VerifiedClosure(
            dispatch=verified_dispatch,
            evidence=evidence,
            _closure_json=closure_json,
        )

    @staticmethod
    def _invoke(
        stage: str,
        verifier: Verifier,
        *args: object,
        **kwargs: object,
    ) -> object:
        try:
            return verifier(*args, **kwargs)
        except Exception as exc:
            raise PrivateVaultExecutionError(
                f"{stage}_verification_failed:{type(exc).__name__}"
            ) from exc


def _required_callable(
    module: object,
    name: str,
) -> Verifier:
    value = getattr(module, name, None)

    if not callable(value):
        raise PrivateVaultAgentDNAUnavailable(
            f"Agent DNA does not export required verifier {name!r}"
        )

    return cast(Verifier, value)


def _enum_label(value: object) -> str:
    return str(getattr(value, "value", value)).upper()


def _report_text(
    report: object,
    name: str,
    fallback: str,
) -> str:
    value = getattr(report, name, None)

    if isinstance(value, str) and value:
        return value

    return fallback


def _with_failures(
    reason_code: str,
    report: object,
) -> str:
    """Carry Agent DNA's per-check failures into the rejection reason."""

    failures = getattr(report, "failures", ())

    if not isinstance(failures, (tuple, list)) or not failures:
        return reason_code

    detail = "; ".join(str(item) for item in failures[:3])
    return f"{reason_code}:{detail}"


def _require_accepted(
    stage: str,
    report: object,
) -> VerifiedEvidence:
    evidence_state = _enum_label(getattr(report, "evidence_state", "UNKNOWN"))
    conformance = _enum_label(
        getattr(
            report,
            "decision_conformance",
            "UNKNOWN",
        )
    )
    accepted = (
        getattr(report, "ok", None) is True
        and evidence_state == "VERIFIED"
        and conformance == "CONFORMANT"
    )

    # Agent DNA reports no reason_code on the accepted path. Substituting the
    # malformed-report sentinel there would stamp every successful verification
    # with a failure label in the evidence chain.
    reason_code = _report_text(
        report,
        "reason_code",
        "ACCEPTED" if accepted else "MALFORMED_VERIFICATION_REPORT",
    )

    if not accepted:
        raise PrivateVaultEvidenceRejected(
            stage=stage,
            evidence_state=evidence_state,
            decision_conformance=conformance,
            reason_code=_with_failures(reason_code, report),
        )

    principal = getattr(
        report,
        "accountable_principal",
        None,
    )

    return VerifiedEvidence(
        stage=stage,
        accountable_principal=(
            principal if isinstance(principal, str) and principal else None
        ),
        reason_code=reason_code,
    )


def _require_digest(
    value: object,
    path: str,
) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{path} must be a sha256 digest")

    prefix = "sha256:"
    hexadecimal = value.removeprefix(prefix)

    if (
        not value.startswith(prefix)
        or len(hexadecimal) != 64
        or any(character not in "0123456789abcdef" for character in hexadecimal)
    ):
        raise ContractError(f"{path} must be a sha256 digest")

    return value
