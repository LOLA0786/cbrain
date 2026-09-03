"""Planner, connector, and observed peer identity must share one pin format."""

from __future__ import annotations

import hashlib
import ipaddress
import ssl
import threading
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from cryptography.x509.oid import NameOID

from cbrain import ActionIntent
from cbrain.deploy.config import ConfigurationError, load
from cbrain.execution.planner import HttpDispatchPlanner, ToolRoute
from cbrain.execution.sidecar import HTTPSRoute, PinnedHTTPSConnector
from cbrain.execution.spki_cli import main as spki_main
from cbrain.execution.tls import (
    peer_identity_from_der_certificate,
    require_peer_identity,
)


def _certificate(tmp_path: Path, *, name: str) -> tuple[Path, Path, bytes]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(UTC) - timedelta(minutes=1))
        .not_valid_after(datetime.now(UTC) + timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.IPAddress(ipaddress.IPv4Address("127.0.0.1"))]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
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
    return cert_path, key_path, certificate.public_bytes(serialization.Encoding.DER)


def _serve_tls(cert_path: Path, key_path: Path) -> tuple[HTTPServer, int]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(str(cert_path), str(key_path))
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, int(server.server_address[1])


def _client_context(cert_path: Path) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_verify_locations(cafile=str(cert_path))
    return context


def test_spki_pin_matches_cryptography(tmp_path: Path) -> None:
    _cert_path, _key_path, der = _certificate(tmp_path, name="pin-a")
    certificate = x509.load_der_x509_certificate(der)
    expected = hashlib_spki(certificate)
    assert peer_identity_from_der_certificate(der) == expected
    assert require_peer_identity(expected) == expected


def hashlib_spki(certificate: x509.Certificate) -> str:
    spki = certificate.public_key().public_bytes(
        Encoding.DER, PublicFormat.SubjectPublicKeyInfo
    )
    return f"tls-spki-sha256:{hashlib.sha256(spki).hexdigest()}"


def test_planner_connector_and_observed_peer_match(tmp_path: Path) -> None:
    cert_path, key_path, der = _certificate(tmp_path, name="pin-match")
    pin = peer_identity_from_der_certificate(der)
    server, port = _serve_tls(cert_path, key_path)
    try:
        planned = HttpDispatchPlanner(
            {
                "payments.refund": ToolRoute(
                    tool_id="payments.refund.v3",
                    capability="payments.refund",
                    destination="payments.internal.example",
                    operation="GET /",
                    credential_audience="payments.internal.example",
                    peer_identity=pin,
                    allowed_parameters=("amount",),
                    required_parameters=("amount",),
                )
            },
            subject_principal="refund-agent@example",
            subject_key_id="refund-agent",
        ).plan(
            ActionIntent.capture(
                agent_id="refund-agent",
                framework="test",
                tool_name="payments.refund",
                capability="payments.refund",
                arguments={"amount": 1},
            )
        )
        connector = PinnedHTTPSConnector(
            routes={
                "payments.internal.example": HTTPSRoute(
                    destination="payments.internal.example",
                    host="127.0.0.1",
                    port=port,
                    operations=frozenset({"GET /"}),
                    credential_audience="payments.internal.example",
                )
            },
            ssl_context=_client_context(cert_path),
        )
        channel = connector.open(planned.prepared.dispatch)
        try:
            assert channel.peer_identity_bytes == planned.prepared.peer_identity_bytes
            assert channel.peer_identity_bytes == pin.encode("ascii")
        finally:
            channel.close()
    finally:
        server.shutdown()
        server.server_close()


def test_second_certificate_does_not_match_planned_pin(tmp_path: Path) -> None:
    _first_cert, _first_key, first_der = _certificate(tmp_path, name="pin-first")
    second_cert, second_key, second_der = _certificate(tmp_path, name="pin-second")
    planned_pin = peer_identity_from_der_certificate(first_der)
    observed_pin = peer_identity_from_der_certificate(second_der)
    assert planned_pin != observed_pin
    server, port = _serve_tls(second_cert, second_key)
    try:
        planned = HttpDispatchPlanner(
            {
                "payments.refund": ToolRoute(
                    tool_id="payments.refund.v3",
                    capability="payments.refund",
                    destination="payments.internal.example",
                    operation="GET /",
                    credential_audience="payments.internal.example",
                    peer_identity=planned_pin,
                    allowed_parameters=("amount",),
                    required_parameters=("amount",),
                )
            },
            subject_principal="refund-agent@example",
            subject_key_id="refund-agent",
        ).plan(
            ActionIntent.capture(
                agent_id="refund-agent",
                framework="test",
                tool_name="payments.refund",
                capability="payments.refund",
                arguments={"amount": 1},
            )
        )
        connector = PinnedHTTPSConnector(
            routes={
                "payments.internal.example": HTTPSRoute(
                    destination="payments.internal.example",
                    host="127.0.0.1",
                    port=port,
                    operations=frozenset({"GET /"}),
                    credential_audience="payments.internal.example",
                )
            },
            ssl_context=_client_context(second_cert),
        )
        channel = connector.open(planned.prepared.dispatch)
        try:
            assert channel.peer_identity_bytes != planned.prepared.peer_identity_bytes
            assert channel.peer_identity_bytes == observed_pin.encode("ascii")
        finally:
            channel.close()
    finally:
        server.shutdown()
        server.server_close()


def test_config_load_rejects_legacy_peer_identity(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        """
        {
          "organisation_id": "acme",
          "agent_id": "agent",
          "subject_principal": "agent@acme",
          "subject_key_id": "agent",
          "privatevault_base_url": "https://privatevault.example",
          "database_url": "postgresql://cbrain@127.0.0.1/cbrain",
          "witness_component_id": "dispatcher",
          "witness_signer_key_id": "witness",
          "routes": {
            "payments.refund": {
              "tool_id": "payments.refund.v3",
              "capability": "payments.refund",
              "destination": "payments.example",
              "operation": "POST /v1/refunds",
              "credential_audience": "payments.example",
              "peer_identity": "tls-spki:payments.example:v3",
              "allowed_parameters": ["amount"],
              "required_parameters": ["amount"]
            }
          }
        }
        """
    )
    with pytest.raises(ConfigurationError, match="peer_identity"):
        load(path)


def test_config_example_pins_are_canonical() -> None:
    config = load(Path("deploy/config.example.json"))
    for route in config.routes.values():
        require_peer_identity(route.peer_identity)


def test_cbrain_spki_prints_canonical_pin(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cert_path, key_path, der = _certificate(tmp_path, name="pin-cli")
    server, port = _serve_tls(cert_path, key_path)
    try:
        assert spki_main(["127.0.0.1", str(port)]) == 0
    finally:
        server.shutdown()
        server.server_close()
    assert capsys.readouterr().out.strip() == peer_identity_from_der_certificate(der)
