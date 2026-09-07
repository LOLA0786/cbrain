"""HTTP client for PrivateVault `/v1/authorize`.

Mints a signed, single-use execution authorization bound to a sealed ALLOW
decision. The pinned server refuses to mint unless the request names the
sealed record (`decision_id` and/or `record_hash`), carries a
`decision_receipt_digest` equal to that record's hash, and presents an action
and dispatch whose digests match the ones sealed at decide time. This client
builds exactly that request from the decision it was handed and the plan that
produced the decision, and then checks every field of the response against the
request before the permit is allowed anywhere near a dispatch: a permit for a
different request, a different action, different bytes, or different evidence
is refused here rather than deeper in the chain where the failure is harder to
attribute.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from hmac import compare_digest
from typing import Any, cast

from cbrain.adapters.privatevault import (
    JsonTransport,
    PrivateVaultDecision,
    PrivateVaultProtocolError,
    PrivateVaultVerdict,
    require_mintable_record,
)
from cbrain.contracts import ActionIntent
from cbrain.execution.gateway import IssuedAuthorization, PlannedDispatch

_SHA256_PREFIXED = re.compile(r"sha256:[0-9a-f]{64}")

_ECHOED_DIGEST_FIELDS = (
    "decision_receipt_digest",
    "authority_receipt_digest",
    "approval_artifact_digest",
    "state_snapshot_digest",
    "policy_bundle_digest",
    "obligations_digest",
)


class AuthorizationRefused(RuntimeError):
    """PrivateVault did not issue a usable execution authorization."""


@dataclass(frozen=True, slots=True)
class EvidenceDigests:
    """Deployment-supplied evidence references sealed into the permit.

    The pinned PrivateVault server checks these are well-formed sha256
    digests, seals them into the permit and its mint-binding digest, and
    echoes them back; it does not resolve them against a stored artifact.
    They therefore must come from deployment configuration that names real
    evidence, never be generated here. `decision_receipt_digest` is not a
    member: it is derived from the sealed decision record and the server
    refuses any other value.
    """

    authority_receipt_digest: str
    state_snapshot_digest: str
    policy_bundle_digest: str
    obligations_digest: str
    approval_artifact_digest: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "authority_receipt_digest",
            "state_snapshot_digest",
            "policy_bundle_digest",
            "obligations_digest",
        ):
            _require_digest(getattr(self, name), name)

        if self.approval_artifact_digest is not None:
            _require_digest(self.approval_artifact_digest, "approval_artifact_digest")


class PrivateVaultAuthorizationClient:
    """Calls `/v1/authorize` and verifies the permit answers this request."""

    def __init__(
        self,
        transport: JsonTransport,
        *,
        organisation_id: str,
        evidence_digests: EvidenceDigests,
        path: str = "/v1/authorize",
    ) -> None:
        if not isinstance(organisation_id, str) or not organisation_id.strip():
            raise ValueError("organisation_id must be non-empty text")

        self._transport = transport
        self._organisation_id = organisation_id
        self._digests = evidence_digests
        self._path = path

    def issue(
        self,
        *,
        action: ActionIntent,
        decision: PrivateVaultDecision,
        planned: PlannedDispatch,
    ) -> IssuedAuthorization:
        decision_id, record_hash = _sealed_allow_reference(decision, action)
        decision_receipt_digest = "sha256:" + record_hash
        prepared = planned.prepared

        body: dict[str, Any] = {
            "request_id": action.request_id,
            "agent_id": action.agent_id,
            "organisation_id": self._organisation_id,
            "decision_id": decision_id,
            "record_hash": record_hash,
            "action": dict(planned.action),
            "dispatch": dict(prepared.dispatch),
            "expected_wire_bytes_digest": _digest(prepared.wire_bytes),
            "expected_wire_bytes_length": len(prepared.wire_bytes),
            "expected_peer_identity_digest": _digest(prepared.peer_identity_bytes),
            "decision_receipt_digest": decision_receipt_digest,
            "authority_receipt_digest": self._digests.authority_receipt_digest,
            "approval_artifact_digest": self._digests.approval_artifact_digest,
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
                f"authorization rejected with status {response.status_code}:"
                f"{_refusal_reason(response.body)}"
            )

        authorization = _required_mapping(response.body, "authorization")
        trust_bundle = _required_mapping(response.body, "trust_bundle")
        at_time = _required_text(response.body, "at_time")

        self._verify_binding(authorization, action, planned, body)

        return IssuedAuthorization(
            authorization=_snapshot(authorization),
            trust_bundle=_snapshot(trust_bundle),
            binding_digests={
                "decision_receipt_digest": decision_receipt_digest,
                "authority_receipt_digest": self._digests.authority_receipt_digest,
                "approval_artifact_digest": self._digests.approval_artifact_digest,
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
        if not isinstance(echoed, str) or not compare_digest(echoed, action.request_id):
            raise AuthorizationRefused("authorization answers a different request")

        for field in (
            "expected_wire_bytes_digest",
            "expected_peer_identity_digest",
            "organisation_id",
        ):
            issued = authorization.get(field)
            expected = (
                sent[field] if field != "organisation_id" else (self._organisation_id)
            )
            if not isinstance(issued, str) or not compare_digest(issued, str(expected)):
                raise AuthorizationRefused(
                    f"authorization {field} does not match the request"
                )

        for field in _ECHOED_DIGEST_FIELDS:
            issued = authorization.get(field)
            expected = sent[field]
            if expected is None:
                if issued is not None:
                    raise AuthorizationRefused(
                        f"authorization {field} was not requested"
                    )
                continue
            if not isinstance(issued, str) or not compare_digest(issued, expected):
                raise AuthorizationRefused(
                    f"authorization {field} does not match the request"
                )

        if authorization.get("expected_wire_bytes_length") != len(
            planned.prepared.wire_bytes
        ):
            raise AuthorizationRefused("authorization wire byte length does not match")

        if authorization.get("max_uses") != 1:
            raise AuthorizationRefused("authorization is not single-use")

        if authorization.get("dispatch") != dict(planned.prepared.dispatch):
            raise AuthorizationRefused(
                "authorization dispatch does not match the prepared dispatch"
            )

        if authorization.get("action") != dict(planned.action):
            raise AuthorizationRefused(
                "authorization action does not match the requested action"
            )


def _sealed_allow_reference(
    decision: PrivateVaultDecision,
    action: ActionIntent,
) -> tuple[str, str]:
    """Resolve the sealed record this permit must be bound to.

    Only an ALLOW sealed under the mintable protocol can be referenced. The
    pinned server enforces the same conditions and additionally re-derives
    the action and dispatch digests; refusing here just makes the failure
    attributable before a doomed request is sent.
    """
    if decision.verdict is not PrivateVaultVerdict.ALLOW:
        raise AuthorizationRefused(
            f"only an ALLOW decision can mint; verdict was {decision.verdict.value}"
        )
    if decision.request_id != action.request_id:
        raise AuthorizationRefused("decision answers a different request")

    try:
        record = decision.record
        decision_id, record_hash = require_mintable_record(
            record,
            agent_id=action.agent_id,
            capability=action.capability,
        )
    except PrivateVaultProtocolError as exc:
        raise AuthorizationRefused(f"decision is not mintable: {exc}") from exc

    if record.get("decision") != PrivateVaultVerdict.ALLOW.value:
        raise AuthorizationRefused("sealed record does not carry an ALLOW decision")

    sealed_request_id = record.get("request_id")
    if sealed_request_id is not None and sealed_request_id != action.request_id:
        raise AuthorizationRefused("sealed record answers a different request")

    return decision_id, record_hash


def _refusal_reason(body: Mapping[str, Any]) -> str:
    """Surface the server's reason_code so refusals are attributable."""
    detail = body.get("detail")
    if isinstance(detail, Mapping):
        reason = detail.get("reason_code")
        if isinstance(reason, str) and reason:
            return reason
    return "unspecified"


def _digest(payload: bytes) -> str:
    """sha256 digest in the canonical prefixed form Agent DNA expects."""
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _require_digest(value: object, name: str) -> None:
    if not isinstance(value, str) or _SHA256_PREFIXED.fullmatch(value) is None:
        raise ValueError(f"{name} must be a sha256:<64 hex> digest")


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
