"""Agent-side client for the out-of-process computer worker.

``RemoteComputerBackend`` is the production ``ComputerBackend``: DEV_ONLY is
False. It never dials page URLs itself — only the worker endpoint.
"""

from __future__ import annotations

import json
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlsplit

from .protocol import (
    ComputerProtocolError,
    build_request,
    parse_response,
)
from .surface import (
    BackendActResult,
    BackendObservation,
    ComputerBackendError,
)


class RemoteComputerBackend:
    """JSON client to ``cbrain-computer-worker``. Not DEV_ONLY."""

    DEV_ONLY = False

    def __init__(
        self,
        *,
        endpoint: str,
        token: str,
        timeout_seconds: float = 30.0,
        max_response_bytes: int = 256 * 1024,
        allow_http_internal: bool = False,
    ) -> None:
        if not token.strip():
            raise ComputerBackendError("computer worker token must be non-empty")
        if timeout_seconds <= 0 or max_response_bytes <= 0:
            raise ComputerBackendError("computer client limits must be positive")

        self._endpoint = _normalize_endpoint(endpoint, allow_http_internal)
        self._token = token
        self._timeout_seconds = timeout_seconds
        self._max_response_bytes = max_response_bytes

    def observe(self, *, url_alias: str) -> BackendObservation:
        result = self._call("observe", {"url_alias": url_alias})
        return BackendObservation.from_wire(result)

    def navigate(self, *, url_alias: str, url: str) -> BackendActResult:
        result = self._call("navigate", {"url_alias": url_alias, "url": url})
        return _act_from_wire(result)

    def click(self, *, url_alias: str, selector: str) -> BackendActResult:
        result = self._call("click", {"url_alias": url_alias, "selector": selector})
        return _act_from_wire(result)

    def type_text(
        self, *, url_alias: str, selector: str, text: str
    ) -> BackendActResult:
        result = self._call(
            "type_text",
            {"url_alias": url_alias, "selector": selector, "text": text},
        )
        return _act_from_wire(result)

    def submit(self, *, url_alias: str, selector: str) -> BackendActResult:
        result = self._call("submit", {"url_alias": url_alias, "selector": selector})
        return _act_from_wire(result)

    def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        payload = build_request(method, params)
        body = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        request = urllib_request.Request(
            self._endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._token}",
            },
        )
        try:
            with urllib_request.urlopen(
                request, timeout=self._timeout_seconds
            ) as response:
                raw = response.read(self._max_response_bytes + 1)
        except urllib_error.HTTPError as exc:
            raw = exc.read(self._max_response_bytes + 1)
            return _result_from_raw(raw, self._max_response_bytes, http_error=exc)
        except Exception as exc:
            raise ComputerBackendError(
                f"computer worker request failed:{type(exc).__name__}"
            ) from exc

        return _result_from_raw(raw, self._max_response_bytes, http_error=None)


def _normalize_endpoint(endpoint: str, allow_http_internal: bool) -> str:
    parsed = urlsplit(endpoint)
    if parsed.query or parsed.fragment:
        raise ComputerBackendError(
            "computer worker endpoint must not contain query or fragment"
        )
    if not parsed.scheme or not parsed.netloc or not parsed.hostname:
        raise ComputerBackendError("computer worker endpoint must be an absolute URL")

    host = parsed.hostname
    if parsed.scheme == "https":
        pass
    elif allow_http_internal and parsed.scheme == "http":
        # Loopback or single-label compose/mesh DNS (agent still has no public net).
        if host not in {"127.0.0.1", "localhost"} and "." in host:
            raise ComputerBackendError(
                "http computer worker hosts must be loopback or single-label"
            )
    else:
        raise ComputerBackendError(
            "computer worker endpoint must be https "
            "(or http internal with allow_http_internal=True)"
        )

    base = endpoint.rstrip("/")
    if base.endswith("/v1/computer"):
        return base
    return base + "/v1/computer"


def _result_from_raw(
    raw: bytes,
    max_response_bytes: int,
    *,
    http_error: urllib_error.HTTPError | None,
) -> dict[str, Any]:
    if len(raw) > max_response_bytes:
        raise ComputerBackendError("computer worker response too large")
    try:
        decoded = parse_response(json.loads(raw))
    except (json.JSONDecodeError, ComputerProtocolError) as exc:
        if http_error is not None:
            raise ComputerBackendError(
                f"computer worker HTTP {http_error.code}"
            ) from exc
        raise ComputerBackendError("computer worker response invalid") from exc
    if not decoded.get("ok"):
        raise ComputerBackendError(str(decoded.get("error") or "computer worker error"))
    result = decoded.get("result")
    if not isinstance(result, dict):
        raise ComputerBackendError("computer worker result missing")
    return result


def _act_from_wire(payload: dict[str, Any]) -> BackendActResult:
    observation = None
    raw_obs = payload.get("observation")
    if isinstance(raw_obs, dict):
        observation = BackendObservation.from_wire(raw_obs)
    return BackendActResult(
        ok=bool(payload.get("ok", False)),
        detail=str(payload.get("detail", "")),
        observation=observation,
    )


__all__ = ["RemoteComputerBackend"]
