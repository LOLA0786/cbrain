"""HTTP client for PrivateVault `/v1/authorize`.

Mints a signed, single-use execution authorization. Every field of the
response is checked against the request that produced it before the permit is
allowed anywhere near a dispatch: a permit for a different request, a
different action, or different bytes is refused here rather than deeper in the
chain where the failure is harder to attribute.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from hmac import compare_digest
from typing import Any, cast

from cbrain.adapters.privatevault import (
    JsonTransport,
    PrivateVaultDecision,
    PrivateVaultProtocolError,
)
from cbrain.contracts import ActionIntent
from cbrain.execution.gateway import IssuedAuthorization, PlannedDispatch


class AuthorizationRefused(RuntimeError):
    """PrivateVault did not issue a usable execution authorization."""


@dataclass(frozen=True, slots=True)
class EvidenceDigests:
    """Digests the caller must supply to bind the permit to its context."""

    decision_receipt_digest: str
    authority_receipt_digest: str
    state_snapshot_digest: str
    policy_bundle_digest: str
    obligations_digest: str
    approval_artifact_digest: str | None = None


class PrivateVaultAuthorizationClient:
    """Calls `/v1/authorize` and verifies the permit answers this request."""

    def __init__(
        self,
        transport: JsonTransport,
        *,
        organisation_id: str,
        agent_id: str,
        evidence_digests: EvidenceDigests,
        path: str = "/v1/authorize",
    ) -> None:
        self._transport = transport
        self._organisation_id = organisation_id
        self._agent_id = agent_id
        self._digests = evidence_digests
        self._path = path

    def issue(
        self,
        *,
        action: ActionIntent,
        decision: PrivateVaultDecision,
        planned: PlannedDispatch,
    ) -> IssuedAuthorization:
        prepared = planned.prepared

        body: dict[str, Any] = {
            "request_id": action.request_id,
            "agent_id": self._agent_id,
            "organisation_id": self._organisation_id,
            "action": dict(planned.action),
            "dispatch": dict(prepared.dispatch),
            "expected_wire_bytes_digest": _digest(prepared.wire_bytes),
            "expected_wire_bytes_length": len(prepared.wire_bytes),
            "expected_peer_identity_digest": _digest(prepared.peer_identity_bytes),
            "decision_receipt_digest": self._digests.decision_receipt_digest,
            "authority_receipt_digest": self._digests.authority_receipt_digest,
            "approval_artifact_digest": (
                self._digests.approval_artifact_digest
            ),
            "state_snapshot_digest": self._digests.state_snapshot_digest,
            "policy_bundle_digest": self._digests.policy_bundle_digest,
            "obligations_digest": self._digests.obligations_digest,
        }

        try:
            response = self._transport.post_json(self._path, body)
        except Exception as exc:
            raise AuthorizationRefused(
                f"authorization transport failed:{type(exc).__name__}"
            ) from exc

        if response.status_code != 200:
            raise AuthorizationRefused(
                f"authorization rejected with status {response.status_code}"
            )

        authorization = _required_mapping(response.body, "authorization")
        trust_bundle = _required_mapping(response.body, "trust_bundle")
        at_time = _required_text(response.body, "at_time")

        self._verify_binding(authorization, action, planned, body)

        return IssuedAuthorization(
            authorization=_snapshot(authorization),
            trust_bundle=_snapshot(trust_bundle),
            binding_digests={
                "decision_receipt_digest": (
                    self._digests.decision_receipt_digest
                ),
                "authority_receipt_digest": (
                    self._digests.authority_receipt_digest
                ),
                "approval_artifact_digest": (
                    self._digests.approval_artifact_digest
                ),
                "state_snapshot_digest": self._digests.state_snapshot_digest,
                "policy_bundle_digest": self._digests.policy_bundle_digest,
                "obligations_digest": self._digests.obligations_digest,
                "at_time": at_time,
            },
        )

    def _verify_binding(
        self,
        authorization: Mapping[str, Any],
        action: ActionIntent,
        planned: PlannedDispatch,
        sent: Mapping[str, Any],
    ) -> None:
        """Refuse a permit that does not answer exactly this request.

        Without this the client would accept a valid signature over somebody
        else's action. The signature proves authenticity, not relevance.
        """
        echoed = authorization.get("request_id")
        if not isinstance(echoed, str) or not compare_digest(
            echoed, action.request_id
        ):
            raise AuthorizationRefused(
                "authorization answers a different request"
            )

        for field in (
            "expected_wire_bytes_digest",
            "expected_peer_identity_digest",
            "organisation_id",
        ):
            issued = authorization.get(field)
            expected = sent[field] if field != "organisation_id" else (
                self._organisation_id
            )
            if not isinstance(issued, str) or not compare_digest(
                issued, str(expected)
            ):
                raise AuthorizationRefused(
                    f"authorization {field} does not match the request"
                )

        if authorization.get("expected_wire_bytes_length") != len(
            planned.prepared.wire_bytes
        ):
            raise AuthorizationRefused(
                "authorization wire byte length does not match"
            )

        if authorization.get("max_uses") != 1:
            raise AuthorizationRefused(
                "authorization is not single-use"
            )

        if authorization.get("dispatch") != dict(planned.prepared.dispatch):
            raise AuthorizationRefused(
                "authorization dispatch does not match the prepared dispatch"
            )

        if authorization.get("action") != dict(planned.action):
            raise AuthorizationRefused(
                "authorization action does not match the requested action"
            )


def _digest(payload: bytes) -> str:
    """sha256 digest in the canonical prefixed form Agent DNA expects."""
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _required_mapping(body: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = body.get(key)
    if not isinstance(value, Mapping):
        raise PrivateVaultProtocolError(f"authorization response missing {key}")
    return value


def _required_text(body: Mapping[str, Any], key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PrivateVaultProtocolError(f"authorization response missing {key}")
    return value


def _snapshot(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Deep copy via JSON so later mutation cannot alter verified evidence."""
    return cast(Mapping[str, Any], json.loads(json.dumps(value, sort_keys=True)))


__all__ = [
    "AuthorizationRefused",
    "EvidenceDigests",
    "PrivateVaultAuthorizationClient",
]
