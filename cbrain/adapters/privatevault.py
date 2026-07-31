from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from hmac import compare_digest
from typing import Any, Protocol, cast

from cbrain.contracts import ActionIntent


class PrivateVaultAdapterError(RuntimeError):
    """Base failure for the PrivateVault adapter."""


class PrivateVaultTransportError(PrivateVaultAdapterError):
    """PrivateVault could not be reached securely."""


class PrivateVaultProtocolError(PrivateVaultAdapterError):
    """PrivateVault returned malformed or contradictory data."""


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
    """Strict client for the real PrivateVault `/v1/decide` contract."""

    def __init__(self, transport: JsonTransport) -> None:
        self._transport = transport

    def decide(
        self,
        action: ActionIntent,
    ) -> PrivateVaultDecision:
        response = self._transport.post_json(
            "/v1/decide",
            action.privatevault_decide_payload(),
        )

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
