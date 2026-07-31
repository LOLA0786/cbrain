from __future__ import annotations

import http.client
import json
import math
import ssl
from collections.abc import Callable, Mapping
from typing import Any, cast
from urllib.parse import SplitResult, urlsplit

from .privatevault import (
    HttpJsonResponse,
    PrivateVaultProtocolError,
    PrivateVaultTransportError,
)

HeadersProvider = Callable[[], Mapping[str, str]]


class PrivateVaultHttpTransport:
    """Bounded synchronous transport for PrivateVault JSON endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        headers_provider: HeadersProvider,
        timeout_seconds: float = 5.0,
        max_response_bytes: int = 1_048_576,
        allow_insecure_localhost: bool = False,
    ) -> None:
        parsed = urlsplit(base_url)
        self._validate_base_url(
            parsed,
            allow_insecure_localhost=allow_insecure_localhost,
        )

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

        self._scheme = parsed.scheme
        self._host = host
        self._port = port
        self._base_path = parsed.path.rstrip("/")
        self._headers_provider = headers_provider
        self._timeout_seconds = float(timeout_seconds)
        self._max_response_bytes = max_response_bytes
        self._ssl_context = ssl.create_default_context()

    @staticmethod
    def _validate_base_url(
        parsed: SplitResult,
        *,
        allow_insecure_localhost: bool,
    ) -> None:
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("base_url must be absolute")

        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain query or fragment")

        if parsed.username is not None or parsed.password is not None:
            raise ValueError("base_url must not contain credentials")

        if parsed.scheme == "https":
            return

        localhost = parsed.hostname in {
            "localhost",
            "127.0.0.1",
            "::1",
        }

        if not (parsed.scheme == "http" and allow_insecure_localhost and localhost):
            raise ValueError("PrivateVault requires HTTPS except explicit localhost")

    def post_json(
        self,
        path: str,
        payload: Mapping[str, Any],
    ) -> HttpJsonResponse:
        if (
            not isinstance(path, str)
            or not path.startswith("/")
            or path.startswith("//")
        ):
            raise ValueError("path must be an absolute origin path")

        try:
            request_body = json.dumps(
                dict(payload),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise PrivateVaultProtocolError(
                "request payload must contain finite JSON values"
            ) from exc

        connection = self._new_connection()

        try:
            connection.request(
                "POST",
                f"{self._base_path}{path}",
                body=request_body,
                headers=self._request_headers(),
            )
            response = connection.getresponse()
            response_body = response.read(self._max_response_bytes + 1)
            status_code = response.status
            content_type = response.getheader(
                "Content-Type",
                "",
            )
        except (
            OSError,
            http.client.HTTPException,
        ) as exc:
            raise PrivateVaultTransportError("PrivateVault transport failed") from exc
        finally:
            connection.close()

        if len(response_body) > self._max_response_bytes:
            raise PrivateVaultProtocolError("PrivateVault response exceeded size limit")

        media_type = content_type.split(";", 1)[0].strip().lower()

        if media_type != "application/json":
            raise PrivateVaultProtocolError(
                "PrivateVault response was not application/json"
            )

        try:
            decoded: Any = json.loads(response_body)
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise PrivateVaultProtocolError(
                "PrivateVault returned invalid JSON"
            ) from exc

        if not isinstance(decoded, dict):
            raise PrivateVaultProtocolError(
                "PrivateVault response must be a JSON object"
            )

        return HttpJsonResponse(
            status_code=status_code,
            body=cast(dict[str, Any], decoded),
        )

    def _request_headers(self) -> dict[str, str]:
        supplied = self._headers_provider()

        if not isinstance(supplied, Mapping):
            raise PrivateVaultTransportError("headers provider must return a mapping")

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        prohibited = {
            "content-length",
            "content-type",
            "host",
        }

        for key, value in supplied.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise PrivateVaultTransportError("authentication headers must be text")

            if key.lower() in prohibited:
                raise PrivateVaultTransportError(
                    "headers provider attempted to override transport headers"
                )

            if "\r" in key or "\n" in key or "\r" in value or "\n" in value:
                raise PrivateVaultTransportError(
                    "authentication headers contain newlines"
                )

            headers[key] = value

        return headers

    def _new_connection(
        self,
    ) -> http.client.HTTPConnection:
        if self._scheme == "https":
            return http.client.HTTPSConnection(
                self._host,
                self._port,
                timeout=self._timeout_seconds,
                context=self._ssl_context,
            )

        return http.client.HTTPConnection(
            self._host,
            self._port,
            timeout=self._timeout_seconds,
        )
