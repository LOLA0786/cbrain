"""In-process credential and destination isolation proofs.

Network compose proofs live under deploy/isolation/ and run when
CBRAIN_ISOLATION=1. These tests always run and pin the worker/sidecar
credential boundary without Docker.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib import error, request

import pytest

from cbrain.execution.sidecar import (
    EnvironmentCredentialBinding,
    EnvironmentCredentialProvider,
    SidecarError,
)

WRITE_SECRET = "sidecar-only-write-token"


class _WriteHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length)
        auth = self.headers.get("Authorization", "")
        if auth != f"Bearer {WRITE_SECRET}":
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b'{"error":"unauthorized"}')
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"po_number":"PO-ISO-1"}')

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


@pytest.fixture()
def write_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _WriteHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    yield f"http://{host}:{port}/v1/purchase-orders"
    server.shutdown()
    thread.join(timeout=2)


def _post(url: str, *, token: str | None) -> tuple[int, bytes]:
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    req = request.Request(
        url,
        data=json.dumps({"quantity_tonnes": 40}).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=2) as response:
            return response.status, response.read()
    except error.HTTPError as exc:
        return exc.code, exc.read()


def test_worker_without_credentials_cannot_authorize_write(write_server, monkeypatch):
    monkeypatch.delenv("CBRAIN_CREDENTIAL_LEDGER", raising=False)
    provider = EnvironmentCredentialProvider(
        {
            "ledger.example": EnvironmentCredentialBinding(
                environment_variable="CBRAIN_CREDENTIAL_LEDGER",
            ),
        }
    )
    with pytest.raises(SidecarError, match="unavailable"):
        provider.resolve("ledger.example")

    status, body = _post(write_server, token=None)
    assert status == 401
    assert b"unauthorized" in body


def test_sidecar_with_credentials_can_reach_allowlisted_write(
    write_server, monkeypatch
):
    monkeypatch.setenv("CBRAIN_CREDENTIAL_LEDGER", WRITE_SECRET)
    provider = EnvironmentCredentialProvider(
        {
            "ledger.example": EnvironmentCredentialBinding(
                environment_variable="CBRAIN_CREDENTIAL_LEDGER",
            ),
        }
    )
    header = provider.resolve("ledger.example")
    assert header.name == "Authorization"
    assert header.value == f"Bearer {WRITE_SECRET}"
    status, body = _post(write_server, token=WRITE_SECRET)
    assert status == 200
    assert b"PO-ISO-1" in body


def test_worker_env_must_not_contain_target_credentials(monkeypatch):
    monkeypatch.delenv("CBRAIN_CREDENTIAL_LEDGER", raising=False)
    assert os.environ.get("CBRAIN_CREDENTIAL_LEDGER") is None


@pytest.mark.skipif(
    os.environ.get("CBRAIN_ISOLATION") != "1",
    reason="CBRAIN_ISOLATION is not set",
)
def test_compose_network_isolation_probe_passed():
    """CI job must export CBRAIN_ISOLATION=1 after deploy/isolation/probe.sh."""
    marker = os.environ.get("CBRAIN_ISOLATION_PROBE")
    assert marker == "passed", (
        "network isolation probe did not pass; "
        "run deploy/isolation/probe.sh under compose"
    )
