"""Bounded HTTPS JSON transport for model providers."""

from __future__ import annotations

import http.client
import json
import math
import os
import ssl
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast
from urllib.parse import SplitResult, urlsplit

from .contracts import (
    ModelCredentialError,
    ModelResponseError,
    ModelTransportError,
)


class JsonModelTransport(Protocol):
    def post_json(
        self,
        path: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> Mapping[str, Any]:
        """POST one JSON object and return one JSON object."""


class HeadersProvider(Protocol):
    def __call__(self) -> Mapping[str, str]:
        """Resolve provider headers outside model context."""


@dataclass(frozen=True, slots=True)
class EnvironmentCredentialHeaders:
    """Resolve one API credential at call time without storing its value."""

    environment_variable: str
    header_name: str
    prefix: str = ""

    def __post_init__(self) -> None:
        for path, value in (
            ("environment_variable", self.environment_variable),
            ("header_name", self.header_name),
        ):
            if not value.strip() or "\r" in value or "\n" in value:
                raise ValueError(f"{path} must be non-empty single-line text")
        if "\r" in self.prefix or "\n" in self.prefix:
            raise ValueError("prefix must be single-line text")

    def __call__(self) -> Mapping[str, str]:
        value = os.environ.get(self.environment_variable, "")
        if not value.strip():
            raise ModelCredentialError(
                f"required model credential {self.environment_variable} is unavailable"
            )
        if "\r" in value or "\n" in value:
            raise ModelCredentialError("model credential contains newlines")
        return {self.header_name: f"{self.prefix}{value}"}


class HttpsJsonModelTransport:
    """Strict synchronous transport with no redirect or response-body leakage."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float = 30.0,
        max_response_bytes: int = 4_194_304,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        parsed = urlsplit(base_url)
        self._validate_base_url(parsed)
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        if (
            isinstance(max_response_bytes, bool)
            or not isinstance(max_response_bytes, int)
            or max_response_bytes < 1
        ):
            raise ValueError("max_response_bytes must be positive")
        host = parsed.hostname
        if host is None:
            raise ValueError("base_url must contain a hostname")
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("base_url contains an invalid port") from exc
        self._host = host
        self._port = port
        self._base_path = parsed.path.rstrip("/")
        self._timeout_seconds = float(timeout_seconds)
        self._max_response_bytes = max_response_bytes
        self._ssl_context = ssl_context or ssl.create_default_context()

    @staticmethod
    def _validate_base_url(parsed: SplitResult) -> None:
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("model provider base_url must use HTTPS")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain query or fragment")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("base_url must not contain credentials")

    def post_json(
        self,
        path: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> Mapping[str, Any]:
        if (
            not isinstance(path, str)
            or not path.startswith("/")
            or path.startswith("//")
            or "?" in path
            or "#" in path
        ):
            raise ValueError("path must be an absolute origin path without a query")
        try:
            body = json.dumps(
                dict(payload),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ModelTransportError(
                "model request must contain finite JSON values"
            ) from exc
        request_headers = self._request_headers(headers)
        connection = http.client.HTTPSConnection(
            self._host,
            self._port,
            timeout=self._timeout_seconds,
            context=self._ssl_context,
        )
        try:
            connection.request(
                "POST",
                f"{self._base_path}{path}",
                body=body,
                headers=request_headers,
            )
            response = connection.getresponse()
            response_body = response.read(self._max_response_bytes + 1)
            status = response.status
            content_type = response.getheader("Content-Type", "")
        except (OSError, http.client.HTTPException) as exc:
            raise ModelTransportError("model provider transport failed") from exc
        finally:
            connection.close()
        if len(response_body) > self._max_response_bytes:
            raise ModelResponseError("model response exceeded size limit")
        if not 200 <= status <= 299:
            raise ModelTransportError(f"model provider returned HTTP {status}")
        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type != "application/json":
            raise ModelResponseError("model provider response was not JSON")
        try:
            decoded: object = json.loads(response_body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelResponseError("model provider returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise ModelResponseError("model provider response must be an object")
        return cast(dict[str, Any], decoded)

    @staticmethod
    def _request_headers(supplied: Mapping[str, str]) -> dict[str, str]:
        if not isinstance(supplied, Mapping):
            raise ModelCredentialError("headers provider must return a mapping")
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        prohibited = {"content-length", "content-type", "host"}
        for key, value in supplied.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ModelCredentialError("model headers must contain text")
            if key.casefold() in prohibited:
                raise ModelCredentialError(
                    "model headers attempted to override transport headers"
                )
            if "\r" in key or "\n" in key or "\r" in value or "\n" in value:
                raise ModelCredentialError("model headers contain newlines")
            headers[key] = value
        return headers


__all__ = [
    "EnvironmentCredentialHeaders",
    "HeadersProvider",
    "HttpsJsonModelTransport",
    "JsonModelTransport",
]
