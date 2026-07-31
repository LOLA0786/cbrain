from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cbrain.adapters.privatevault_consumption import (
    PrivateVaultAuthorizationUseFactory,
)
from cbrain.adapters.privatevault_execution import (
    ExecutionAuthorizationBinding,
    PrivateVaultAgentDNAVerifier,
    PrivateVaultEvidenceRejected,
    PrivateVaultExecutionError,
    VerifiedAuthorization,
)
from cbrain.consumption import (
    AuthorizationConsumptionStore,
)
from cbrain.contracts import (
    _restore_object,
    _snapshot_object,
)


class PrivateVaultAuthorizationClaimCoordinator:
    """Verify and atomically consume one authorization before dispatch."""

    def __init__(
        self,
        *,
        verifier: PrivateVaultAgentDNAVerifier,
        use_factory: PrivateVaultAuthorizationUseFactory,
        consumption_store: AuthorizationConsumptionStore,
    ) -> None:
        self._verifier = verifier
        self._use_factory = use_factory
        self._consumption_store = consumption_store

    def verify_and_claim(
        self,
        *,
        authorization: Mapping[str, Any],
        trust_bundle: Mapping[str, Any] | None,
        binding: ExecutionAuthorizationBinding,
    ) -> VerifiedAuthorization:
        authorization_json = _snapshot_object(
            authorization,
            "execution_authorization",
        )
        authorization_document = _restore_object(authorization_json)

        candidate = self._use_factory.create_candidate(
            authorization=authorization_document,
            binding=binding,
            claimed_at=binding.at_time,
        )

        already_consumed = self._consumption_store.is_consumed(candidate)

        verified = self._verifier.verify_authorization(
            authorization=authorization_document,
            trust_bundle=trust_bundle,
            binding=binding,
            already_consumed=already_consumed,
        )

        if already_consumed:
            raise _consumed_error()

        verified_use = self._use_factory.create(
            verified_authorization=verified,
            claimed_at=binding.at_time,
        )

        if verified_use != candidate:
            raise PrivateVaultExecutionError(
                "authorization changed during verification"
            )

        if not self._consumption_store.claim_once(verified_use):
            raise _consumed_error()

        return verified


def _consumed_error() -> PrivateVaultEvidenceRejected:
    return PrivateVaultEvidenceRejected(
        stage="authorization",
        evidence_state="VERIFIED",
        decision_conformance="NON_CONFORMANT",
        reason_code=("EXECUTION_AUTHORIZATION_CONSUMED"),
    )
