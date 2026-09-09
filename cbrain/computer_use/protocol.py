"""JSON wire protocol between agent-side client and computer worker."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

WORKER_REQUEST_SCHEMA: Final = "cbrain-computer-worker/request-v1"
WORKER_RESPONSE_SCHEMA: Final = "cbrain-computer-worker/response-v1"

_METHODS: Final[frozenset[str]] = frozenset(
    {"observe", "navigate", "click", "type_text", "submit", "health"}
)


class ComputerProtocolError(ValueError):
    """Worker request or response violated the wire contract."""


def build_request(method: str, params: Mapping[str, Any]) -> dict[str, Any]:
    if method not in _METHODS:
        raise ComputerProtocolError(f"unsupported computer method {method!r}")
    return {
        "schema": WORKER_REQUEST_SCHEMA,
        "method": method,
        "params": dict(params),
    }


def parse_request(payload: object) -> tuple[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ComputerProtocolError("computer request must be an object")
    if payload.get("schema") != WORKER_REQUEST_SCHEMA:
        raise ComputerProtocolError("computer request schema mismatch")
    method = payload.get("method")
    if not isinstance(method, str) or method not in _METHODS:
        raise ComputerProtocolError("computer request method is invalid")
    params = payload.get("params")
    if not isinstance(params, dict):
        raise ComputerProtocolError("computer request params must be an object")
    return method, dict(params)


def build_response(
    *,
    ok: bool,
    result: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": WORKER_RESPONSE_SCHEMA,
        "ok": ok,
    }
    if result is not None:
        payload["result"] = dict(result)
    if error is not None:
        payload["error"] = error
    return payload


def parse_response(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ComputerProtocolError("computer response must be an object")
    if payload.get("schema") != WORKER_RESPONSE_SCHEMA:
        raise ComputerProtocolError("computer response schema mismatch")
    if not isinstance(payload.get("ok"), bool):
        raise ComputerProtocolError("computer response ok must be boolean")
    return dict(payload)


__all__ = [
    "ComputerProtocolError",
    "WORKER_REQUEST_SCHEMA",
    "WORKER_RESPONSE_SCHEMA",
    "build_request",
    "build_response",
    "parse_request",
    "parse_response",
]
