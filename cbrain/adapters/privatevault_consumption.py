from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Any, cast

from cbrain.adapters.privatevault_execution import (
    ExecutionAuthorizationBinding,
    PrivateVaultAgentDNAUnavailable,
    PrivateVaultExecutionError,
    VerifiedAuthorization,
)
from cbrain.consumption import AuthorizationUse

DigestFunction = Callable[[Mapping[str, Any]], object]


class PrivateVaultAuthorizationUseFactory:
    """Build atomic-use claims with Agent DNA's real digest."""

    def __init__(
        self,
        digest_function: DigestFunction | None = None,
    ) -> None:
        self._digest_function = digest_function or _load_digest_function()

    def create(
        self,
        *,
        verified_authorization: VerifiedAuthorization,
        claimed_at: str,
    ) -> AuthorizationUse:
        return self.create_candidate(
            authorization=(verified_authorization.authorization),
            binding=verified_authorization.binding,
            claimed_at=claimed_at,
        )

    def create_candidate(
        self,
        *,
        authorization: Mapping[str, Any],
        binding: ExecutionAuthorizationBinding,
        claimed_at: str,
    ) -> AuthorizationUse:
        organisation_id = _required_text(
            authorization,
            "organisation_id",
        )
        authorization_id = _required_text(
            authorization,
            "execution_authorization_id",
        )
        signed_request_id = _required_text(
            authorization,
            "request_id",
        )

        if signed_request_id != binding.request_id:
            raise PrivateVaultExecutionError("authorization request_id mismatch")

        try:
            digest = self._digest_function(authorization)
        except Exception as exc:
            raise PrivateVaultExecutionError(
                f"execution authorization digest failed:{type(exc).__name__}"
            ) from exc

        if not isinstance(digest, str):
            raise PrivateVaultExecutionError(
                "execution authorization digest is invalid"
            )

        try:
            return AuthorizationUse(
                organisation_id=organisation_id,
                execution_authorization_id=(authorization_id),
                authorization_digest=digest,
                request_id=binding.request_id,
                claimed_at=claimed_at,
            )
        except ValueError as exc:
            raise PrivateVaultExecutionError(
                "execution authorization claim is invalid"
            ) from exc


def _load_digest_function() -> DigestFunction:
    try:
        module = import_module("agent_dna.execution_v01")
    except (ImportError, ModuleNotFoundError) as exc:
        raise PrivateVaultAgentDNAUnavailable(
            "pinned PrivateVault Agent DNA is unavailable"
        ) from exc

    value = getattr(
        module,
        "execution_authorization_digest",
        None,
    )

    if not callable(value):
        raise PrivateVaultAgentDNAUnavailable(
            "Agent DNA does not export 'execution_authorization_digest'"
        )

    return cast(DigestFunction, value)


def _required_text(
    value: Mapping[str, Any],
    key: str,
) -> str:
    field = value.get(key)

    if not isinstance(field, str) or not field.strip():
        raise PrivateVaultExecutionError(f"authorization field {key!r} is invalid")

    return field
