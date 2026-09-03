from __future__ import annotations

import hashlib
import json
import ssl
import threading
from datetime import UTC, datetime, timedelta
from http.client import HTTPSConnection
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from cbrain.simulators import (
    PROCUREMENT_MAIL_SEND,
    BearerTokenDigestVerifier,
    MailSimulator,
    SimulatorApplication,
    bind_simulator_server,
    encode_simulator_request,
)

TOKEN = "sidecar-mail-token"
OPERATION = "POST /v1/procurement/mail/send"
BODY = "PO-2026-0001\nFe500D 25t @ INR 48500\nKalika Steel Traders Pvt Ltd\n"


def _mail_app() -> tuple[MailSimulator, SimulatorApplication]:
    target = MailSimulator(credential_digest=hashlib.sha256(TOKEN.encode()).hexdigest())
    application = SimulatorApplication(
        simulator=target,
        operations={OPERATION: PROCUREMENT_MAIL_SEND},
        credential_verifier=BearerTokenDigestVerifier.from_token(TOKEN),
    )
    return target, application


def test_mail_send_keeps_exact_bytes_and_omits_body_from_receipt() -> None:
    target, _application = _mail_app()
    receipt = target.execute(
        PROCUREMENT_MAIL_SEND,
        request_id="mail-1",
        idempotency_key="mail-1",
        arguments={
            "to": "vendor@kalika.example",
            "subject": "Purchase order PO-2026-0001",
            "body": BODY,
        },
    )
    expected_digest = f"sha256:{hashlib.sha256(BODY.encode('utf-8')).hexdigest()}"
    assert receipt.result["body_sha256"] == expected_digest
    assert receipt.result["byte_length"] == len(BODY.encode("utf-8"))
    assert "body" not in receipt.result
    assert target.stored_body("mail-0001") == BODY.encode("utf-8")
    payload = json.dumps(receipt.to_payload())
    assert BODY not in payload
    assert TOKEN not in payload


def test_mail_http_adapter_requires_credential_digest() -> None:
    _target, application = _mail_app()
    wire = encode_simulator_request(
        request_id="mail-http",
        idempotency_key="mail-http",
        arguments={
            "to": "vendor@kalika.example",
            "subject": "PO",
            "body": BODY,
        },
    )
    denied = application.handle(
        operation=OPERATION,
        headers={"Content-Type": "application/json"},
        wire_bytes=wire,
    )
    assert denied.status_code == 401
    accepted = application.handle(
        operation=OPERATION,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
            "Content-Encoding": "identity",
        },
        wire_bytes=wire,
    )
    assert accepted.status_code == 200
    assert BODY.encode() not in accepted.body


def _certificate(tmp_path: Path) -> tuple[Path, Path]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "mail-target")])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(UTC) - timedelta(minutes=1))
        .not_valid_after(datetime.now(UTC) + timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    cert_path = tmp_path / "mail.pem"
    key_path = tmp_path / "mail.key"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return cert_path, key_path


def test_mail_tls_server_does_not_echo_request_content(tmp_path: Path) -> None:
    _target, application = _mail_app()
    cert_path, key_path = _certificate(tmp_path)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(str(cert_path), str(key_path))
    server = bind_simulator_server(
        application=application,
        host="127.0.0.1",
        port=0,
        ssl_context=context,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = int(server.server_address[1])
        client = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        client.check_hostname = False
        client.verify_mode = ssl.CERT_REQUIRED
        client.load_verify_locations(str(cert_path))
        connection = HTTPSConnection("127.0.0.1", port=port, context=client)
        body = encode_simulator_request(
            request_id="mail-tls",
            idempotency_key="mail-tls",
            arguments={
                "to": "vendor@kalika.example",
                "subject": "PO",
                "body": BODY,
            },
        )
        connection.request(
            "POST",
            "/v1/procurement/mail/send",
            body=body,
            headers={
                "Authorization": f"Bearer {TOKEN}",
                "Content-Type": "application/json",
                "Content-Encoding": "identity",
                "Content-Length": str(len(body)),
            },
        )
        response = connection.getresponse()
        payload = response.read()
        assert response.status == 200
        assert BODY.encode() not in payload
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
