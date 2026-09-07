from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from hmac import compare_digest
from typing import Any, Protocol, cast

from cbrain.contracts import ActionIntent, _restore_object, _snapshot_object

# Pinned PrivateVault Agent DNA (upstreams.lock.json) only mints execution
# permits from a decision record sealed under this protocol version. A record
# sealed without both binding digests is audit-only and can never authorize.
MINTABLE_DECISION_PROTOCOL = "drp/0.2"

# The exact field sets the pinned server validates on `/v1/decide`
# (`agent_dna.action_v01.EXECUTION_ACTION_FIELDS` and
# `agent_dna.dispatch_context_v01.DISPATCH_CONTEXT_FIELDS`). Nothing may be
# added, omitted or renamed: an unexpected field is refused, and a missing one
# would let two different actions share a digest.
EXECUTION_ACTION_FIELDS = frozenset(
    {
        "subject_principal",
        "subject_key_id",
        "action",
        "resource",
        "parameters",
    }
)
DISPATCH_CONTEXT_FIELDS = frozenset(
    {
        "adapter",
        "transport",
        "operation",
        "destination",
        "wire_content_type",
    }
)

_SHA256_PREFIXED = re.compile(r"sha256:[0-9a-f]{64}")
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


class PrivateVaultAdapterError(RuntimeError):
    """Base failure for the PrivateVault adapter."""


class PrivateVaultTransportError(PrivateVaultAdapterError):
    """PrivateVault could not be reached securely."""


class PrivateVaultProtocolError(PrivateVaultAdapterError):
    """PrivateVault returned malformed or contradictory data."""


class PrivateVaultBindingError(PrivateVaultAdapterError):
    """The planned action cannot be bound into a mintable decision request."""


class PrivateVaultVerdict(StrEnum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    BLOCK = "block"


_EXPECTED_STATUS = {
    PrivateVaultVerdict.ALLOW: 200,
    PrivateVaultVerdict.REQUIRE_APPROVAL: 202,
    PrivateVaultVerdict.BLOCK: 403,
}


@dataclass(frozen=True, slots=True)
class HttpJsonResponse:
    status_code: int
    body: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            isinstance(self.status_code, bool)
            or not isinstance(self.status_code, int)
            or not 100 <= self.status_code <= 599
        ):
            raise PrivateVaultProtocolError("invalid HTTP status code")

        if not isinstance(self.body, Mapping):
            raise PrivateVaultProtocolError(
                "PrivateVault response must be a JSON object"
            )


class JsonTransport(Protocol):
    def post_json(
        self,
        path: str,
        payload: Mapping[str, Any],
    ) -> HttpJsonResponse:
        """POST one JSON object and return one JSON object."""


@dataclass(frozen=True, slots=True)
class PrivateVaultDecision:
    verdict: PrivateVaultVerdict
    triggered_by: str
    reason: str
    request_id: str
    _record_json: bytes

    @property
    def record(self) -> dict[str, Any]:
        restored: Any = json.loads(self._record_json)

        if not isinstance(restored, dict):
            raise PrivateVaultProtocolError(
                "stored PrivateVault record is not an object"
            )

        return cast(dict[str, Any], restored)


@dataclass(frozen=True, slots=True)
class DecisionBinding:
    """The exact planned action and dispatch a mintable decision must seal.

    The pinned server derives `action_digest` and `dispatch_context_digest`
    from these two objects and seals them into the decision record. The
    later `/v1/authorize` call is refused unless the action and dispatch it
    carries project onto the same digests. Capturing the binding from the
    plan, before the decision is requested, is what lets one immutable plan
    run through decision, issuance and dispatch.
    """

    _execution_action_json: bytes
    _dispatch_context_json: bytes

    @classmethod
    def capture(
        cls,
        action: ActionIntent,
        *,
        execution_action: Mapping[str, Any],
        dispatch: Mapping[str, Any],
    ) -> DecisionBinding:
        """Validate the plan against the intent the way the pinned server does.

        Mirrors `api.server._validate_decide_binding` and
        `agent_dna.dispatch_context_v01.dispatch_context_from_ea_dispatch`
        at the pinned commit so a doomed request is refused here, with an
        attributable reason, instead of as a 422 from PrivateVault. The
        server remains the authority; this is a fail-fast copy, not a
        replacement.
        """
        validated_action = _validate_execution_action(execution_action)

        if validated_action["action"] != action.capability:
            raise PrivateVaultBindingError(
                "execution_action.action must match the governed capability"
            )
        if validated_action["subject_key_id"] != action.agent_id:
            raise PrivateVaultBindingError(
                "execution_action.subject_key_id must match the agent_id"
            )
        if dict(validated_action["parameters"]) != action.arguments:
            raise PrivateVaultBindingError(
                "execution_action.parameters must match the governed arguments"
            )

        context = _dispatch_context_from_dispatch(dispatch)

        return cls(
            _execution_action_json=_snapshot_object(
                validated_action,
                "execution_action",
            ),
            _dispatch_context_json=_snapshot_object(
                context,
                "dispatch_context",
            ),
        )

    @property
    def execution_action(self) -> dict[str, Any]:
        return _restore_object(self._execution_action_json)

    @property
    def dispatch_context(self) -> dict[str, Any]:
        return _restore_object(self._dispatch_context_json)

    def decide_payload(self, action: ActionIntent) -> dict[str, Any]:
        """The mintable `/v1/decide` request: audit fields plus both bindings."""
        payload = action.privatevault_decide_payload()
        payload["execution_action"] = self.execution_action
        payload["dispatch_context"] = self.dispatch_context
        return payload


def _validate_execution_action(
    execution_action: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(execution_action, Mapping):
        raise PrivateVaultBindingError("execution_action must be a mapping")

    present = frozenset(execution_action)
    if present != EXECUTION_ACTION_FIELDS:
        missing = sorted(EXECUTION_ACTION_FIELDS - present)
        unexpected = sorted(present - EXECUTION_ACTION_FIELDS)
        raise PrivateVaultBindingError(
            "execution_action fields do not match the pinned contract; "
            f"missing={missing}, unexpected={unexpected}"
        )

    for name in ("subject_principal", "subject_key_id", "action", "resource"):
        value = execution_action[name]
        if not isinstance(value, str) or not value:
            raise PrivateVaultBindingError(
                f"execution_action.{name} must be a non-empty string"
            )

    parameters = execution_action["parameters"]
    if not isinstance(parameters, Mapping):
        raise PrivateVaultBindingError("execution_action.parameters must be a mapping")

    return dict(execution_action)


def _dispatch_context_from_dispatch(
    dispatch: Mapping[str, Any],
) -> dict[str, str]:
    """Project the EA dispatch onto the five sealed context fields.

    `adapter` is not a field of the EA dispatch object; the pinned server
    defaults it to `transport` when it projects the `/v1/authorize` dispatch
    back onto the sealed context. Sending the same default at decide time is
    what makes the two digests agree.
    """
    if not isinstance(dispatch, Mapping):
        raise PrivateVaultBindingError("dispatch must be a mapping")

    transport = dispatch.get("transport")
    adapter = dispatch.get("adapter")
    if not isinstance(adapter, str) or not adapter:
        adapter = transport

    candidate = {
        "adapter": adapter,
        "transport": transport,
        "operation": dispatch.get("operation"),
        "destination": dispatch.get("destination"),
        "wire_content_type": dispatch.get("wire_content_type"),
    }

    context: dict[str, str] = {}
    for name in sorted(DISPATCH_CONTEXT_FIELDS):
        value = candidate[name]
        if not isinstance(value, str) or not value:
            raise PrivateVaultBindingError(
                f"dispatch_context.{name} must be a non-empty string"
            )
        context[name] = value

    return context


def require_mintable_record(
    record: Mapping[str, Any],
    *,
    agent_id: str,
    capability: str,
) -> tuple[str, str]:
    """Check a sealed record has everything `/v1/authorize` will demand.

    Returns `(decision_id, record_hash)` with the hash as bare lowercase hex,
    which is how the pinned `DecisionRecord.to_dict()` emits it. Raises
    `PrivateVaultProtocolError` for an audit-only (drp/0.1) record or one
    that is missing the fields the mint binding reads.
    """
    protocol = record.get("protocol_version")
    if protocol != MINTABLE_DECISION_PROTOCOL:
        raise PrivateVaultProtocolError(
            "PrivateVault sealed an audit-only decision record; "
            f"protocol_version={protocol!r} cannot mint an execution permit"
        )

    for name in ("action_digest", "dispatch_context_digest"):
        value = record.get(name)
        if not isinstance(value, str) or _SHA256_PREFIXED.fullmatch(value) is None:
            raise PrivateVaultProtocolError(
                f"PrivateVault record field {name!r} is not a sealed sha256 digest"
            )

    decision_id = record.get("decision_id")
    if not isinstance(decision_id, str) or not decision_id.strip():
        raise PrivateVaultProtocolError(
            "PrivateVault record field 'decision_id' must be non-empty text"
        )

    record_hash = record.get("record_hash")
    if not isinstance(record_hash, str):
        raise PrivateVaultProtocolError(
            "PrivateVault record field 'record_hash' must be text"
        )
    bare_hash = record_hash.removeprefix("sha256:").lower()
    if _SHA256_HEX.fullmatch(bare_hash) is None:
        raise PrivateVaultProtocolError(
            "PrivateVault record field 'record_hash' is not a sha256 digest"
        )

    if record.get("agent_id") != agent_id:
        raise PrivateVaultProtocolError(
            "PrivateVault record is sealed for a different agent_id"
        )
    if record.get("capability") != capability:
        raise PrivateVaultProtocolError(
            "PrivateVault record is sealed for a different capability"
        )

    return decision_id, bare_hash


def _required_text(
    body: Mapping[str, Any],
    key: str,
) -> str:
    value = body.get(key)

    if not isinstance(value, str) or not value.strip():
        raise PrivateVaultProtocolError(
            f"PrivateVault response field {key!r} must be non-empty text"
        )

    return value


def _bound_request_id(
    body: Mapping[str, Any],
    expected_request_id: str,
) -> str:
    top_level = body.get("request_id")
    record = body.get("record")
    nested = record.get("request_id") if isinstance(record, Mapping) else None

    for location, value in (
        ("request_id", top_level),
        ("record.request_id", nested),
    ):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise PrivateVaultProtocolError(
                f"PrivateVault response field {location!r} must be non-empty text"
            )

    candidates = [
        value
        for value in (top_level, nested)
        if isinstance(value, str) and value.strip()
    ]

    if not candidates:
        raise PrivateVaultProtocolError(
            "PrivateVault decision is not bound to a request_id"
        )

    if len(candidates) == 2 and not compare_digest(
        candidates[0].encode("utf-8"),
        candidates[1].encode("utf-8"),
    ):
        raise PrivateVaultProtocolError(
            "PrivateVault response contains contradictory request_id values"
        )

    echoed_request_id = candidates[0]

    if not compare_digest(
        echoed_request_id.encode("utf-8"),
        expected_request_id.encode("utf-8"),
    ):
        raise PrivateVaultProtocolError(
            "PrivateVault decision answers a different request_id"
        )

    return echoed_request_id


def _snapshot_record(value: object) -> bytes:
    if not isinstance(value, Mapping):
        raise PrivateVaultProtocolError(
            "PrivateVault response field 'record' must be an object"
        )

    try:
        return json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PrivateVaultProtocolError(
            "PrivateVault record must contain finite JSON values"
        ) from exc


class PrivateVaultDecisionClient:
    """Strict client for the real PrivateVault `/v1/decide` contract.

    Without a `binding` the request is audit-only: the pinned server seals a
    drp/0.1 record that can never mint an execution permit. Callers that
    intend to execute must pass the `DecisionBinding` captured from the
    plan, and then receive a record checked to be mintable-shaped.
    """

    def __init__(self, transport: JsonTransport) -> None:
        self._transport = transport

    def decide(
        self,
        action: ActionIntent,
        *,
        binding: DecisionBinding | None = None,
    ) -> PrivateVaultDecision:
        payload = (
            action.privatevault_decide_payload()
            if binding is None
            else binding.decide_payload(action)
        )
        response = self._transport.post_json("/v1/decide", payload)

        raw_verdict = _required_text(
            response.body,
            "decision",
        )

        try:
            verdict = PrivateVaultVerdict(raw_verdict)
        except ValueError as exc:
            raise PrivateVaultProtocolError(
                "PrivateVault returned an unknown decision"
            ) from exc

        expected_status = _EXPECTED_STATUS[verdict]

        if response.status_code != expected_status:
            raise PrivateVaultProtocolError(
                "PrivateVault HTTP status contradicts its decision"
            )

        record_json = _snapshot_record(response.body.get("record"))
        request_id = _bound_request_id(
            response.body,
            action.request_id,
        )

        if binding is not None:
            # A bound request must come back sealed under the mintable
            # protocol. An audit-only answer here means the server did not
            # seal what was asked for, and no permit could follow.
            require_mintable_record(
                cast(Mapping[str, Any], json.loads(record_json)),
                agent_id=action.agent_id,
                capability=action.capability,
            )

        return PrivateVaultDecision(
            verdict=verdict,
            triggered_by=_required_text(
                response.body,
                "triggered_by",
            ),
            reason=_required_text(
                response.body,
                "reason",
            ),
            request_id=request_id,
            _record_json=record_json,
        )
