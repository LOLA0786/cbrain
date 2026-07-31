from __future__ import annotations

import http.client
from collections.abc import Mapping

import pytest

from cbrain.models import (
    HttpsJsonModelTransport,
    ModelCredentialError,
    ModelTransportError,
)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://api.example.com",
        "https://user:password@api.example.com",
        "https://api.example.com?api_key=secret",
        "https://api.example.com#fragment",
    ],
)
def test_model_transport_rejects_unsafe_base_urls(base_url: str) -> None:
    with pytest.raises(ValueError):
        HttpsJsonModelTransport(base_url=base_url)


def test_model_transport_headers_cannot_override_framing() -> None:
    with pytest.raises(ModelCredentialError, match="override"):
        HttpsJsonModelTransport._request_headers(  # noqa: SLF001
            {"Content-Type": "text/plain"}
        )


class FakeResponse:
    status = 401

    def read(self, amount: int) -> bytes:
        return b'{"error":"provider-secret-body"}'

    def getheader(self, name: str, default: str) -> str:
        return "application/json"


class FakeConnection:
    request_headers: Mapping[str, str] | None = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def request(
        self,
        method: str,
        path: str,
        body: bytes,
        headers: Mapping[str, str],
    ) -> None:
        self.request_headers = headers

    def getresponse(self) -> FakeResponse:
        return FakeResponse()

    def close(self) -> None:
        pass


def test_provider_error_does_not_leak_response_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(http.client, "HTTPSConnection", FakeConnection)
    transport = HttpsJsonModelTransport(base_url="https://api.example.com")

    with pytest.raises(ModelTransportError) as captured:
        transport.post_json(
            "/v1/complete",
            {"messages": []},
            {"Authorization": "Bearer request-secret"},
        )

    message = str(captured.value)
    assert "provider-secret-body" not in message
    assert "request-secret" not in message
    assert message == "model provider returned HTTP 401"


def test_model_request_path_cannot_contain_query_credentials() -> None:
    transport = HttpsJsonModelTransport(base_url="https://api.example.com")
    with pytest.raises(ValueError, match="without a query"):
        transport.post_json(
            "/v1/complete?key=secret",
            {},
            {},
        )
