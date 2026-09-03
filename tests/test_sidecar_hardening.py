"""Sidecar must close failed TLS sockets, refuse bad length, and pin client certs."""

from __future__ import annotations

import http.client
import ipaddress
import ssl
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from cbrain.execution.sidecar import (
    HTTPSRoute,
    PinnedHTTPSConnector,
    SidecarError,
    bind_sidecar_server,
)


def _strict_client_context() -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    return context


def _certificate(
    tmp_path: Path, *, name: str, server: bool = False
) -> tuple[Path, Path]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(UTC) - timedelta(minutes=1))
        .not_valid_after(datetime.now(UTC) + timedelta(days=1))
    )
    if server:
        builder = builder.add_extension(
            x509.SubjectAlternativeName(
                [x509.IPAddress(ipaddress.IPv4Address("127.0.0.1"))]
            ),
            critical=False,
        )
    certificate = builder.sign(key, hashes.SHA256())
    cert_path = tmp_path / f"{name}.pem"
    key_path = tmp_path / f"{name}.key"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return cert_path, key_path


def _server_context(cert_path: Path, key_path: Path, client_ca: Path) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_cert_chain(str(cert_path), str(key_path))
    context.load_verify_locations(cafile=str(client_ca))
    return context


def _mtls_client_context(
    *, server_ca: Path, client_cert: Path, client_key: Path
) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_verify_locations(cafile=str(server_ca))
    context.load_cert_chain(str(client_cert), str(client_key))
    return context


def test_pinned_channel_closes_socket_when_peer_certificate_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[bool] = []

    class FakeConnection:
        sock = None

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def connect(self) -> None:
            return None

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(http.client, "HTTPSConnection", FakeConnection)
    connector = PinnedHTTPSConnector(
        routes={
            "payments.internal.example": HTTPSRoute(
                destination="payments.internal.example",
                host="127.0.0.1",
                port=9443,
                operations=frozenset({"GET /"}),
                credential_audience="payments.internal.example",
            )
        },
        ssl_context=_strict_client_context(),
    )
    with pytest.raises(SidecarError, match="peer socket"):
        connector.open(
            {
                "destination": "payments.internal.example",
                "transport": "https",
                "operation": "GET /",
                "credential_audience": "payments.internal.example",
            }
        )
    assert closed == [True]


def test_missing_content_length_is_411_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    server_cert, server_key = _certificate(tmp_path, name="sidecar-server", server=True)
    client_cert, client_key = _certificate(tmp_path, name="sidecar-client")
    server = bind_sidecar_server(
        service=_UnusedService(),
        host="127.0.0.1",
        port=0,
        ssl_context=_server_context(server_cert, server_key, client_cert),
        allowed_client_principals=frozenset({"sidecar-client"}),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPSConnection(
            "127.0.0.1",
            int(server.server_address[1]),
            context=_mtls_client_context(
                server_ca=server_cert, client_cert=client_cert, client_key=client_key
            ),
        )
        try:
            connection.putrequest("POST", "/v1/dispatch")
            connection.endheaders()
            response = connection.getresponse()
            body = response.read()
        finally:
            connection.close()
    finally:
        server.shutdown()
        server.server_close()
    assert response.status == 411
    assert "Traceback" not in capsys.readouterr().err
    assert body


def test_unknown_client_certificate_is_403_with_empty_body(tmp_path: Path) -> None:
    server_cert, server_key = _certificate(tmp_path, name="sidecar-server", server=True)
    attacker_cert, attacker_key = _certificate(tmp_path, name="attacker")
    server = bind_sidecar_server(
        service=_UnusedService(),
        host="127.0.0.1",
        port=0,
        ssl_context=_server_context(server_cert, server_key, attacker_cert),
        allowed_client_principals=frozenset({"sidecar-client"}),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPSConnection(
            "127.0.0.1",
            int(server.server_address[1]),
            context=_mtls_client_context(
                server_ca=server_cert,
                client_cert=attacker_cert,
                client_key=attacker_key,
            ),
        )
        try:
            connection.request("POST", "/v1/dispatch", body=b"{}")
            response = connection.getresponse()
            body = response.read()
        finally:
            connection.close()
    finally:
        server.shutdown()
        server.server_close()
    assert response.status == 403
    assert body == b""


class _UnusedService:
    def handle(self, payload: Any) -> dict[str, Any]:
        raise AssertionError(f"dispatch must not run: {payload!r}")
