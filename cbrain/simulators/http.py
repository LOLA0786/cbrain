"""Strict HTTP adapter for executable CRM and ledger simulators."""

from __future__ import annotations

import hashlib
import json
import re
import ssl
from collections.abc import Mapping
from dataclasses import dataclass
from hmac import compare_digest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import MappingProxyType
from typing import Any, Protocol, cast

from .contracts import (
    SIMULATOR_REQUEST_SCHEMA,
    EffectReceipt,
    SimulatorBusinessError,
    SimulatorConflict,
    SimulatorContractError,
    SimulatorError,
    SimulatorNotFound,
    canonical_object,
    exact_fields,
    required_text,
)
from .crm import (
    CRM_CONTACT_MERGE,
    CRM_CONTACT_READ,
    CRM_EMAIL_SEND,
    CRM_EXPORT_CONTACTS,
)
from .ledger import (
    PAYMENTS_BALANCE_READ,
    PAYMENTS_BENEFICIARY_ADD,
    PAYMENTS_LIMIT_MODIFY,
    PAYMENTS_LIMIT_READ,
    PAYMENTS_TRANSFER_INITIATE,
)

SIMULATOR_OPERATIONS: Mapping[str, str] = MappingProxyType(
    {
        "POST /v1/crm/contacts/read": CRM_CONTACT_READ,
        "POST /v1/crm/contacts/merge": CRM_CONTACT_MERGE,
        "POST /v1/crm/emails/send": CRM_EMAIL_SEND,
        "POST /v1/crm/contacts/export": CRM_EXPORT_CONTACTS,
        "POST /v1/payments/balances/read": PAYMENTS_BALANCE_READ,
        "POST /v1/payments/limits/read": PAYMENTS_LIMIT_READ,
        "POST /v1/payments/limits/modify": PAYMENTS_LIMIT_MODIFY,
        "POST /v1/payments/beneficiaries/add": PAYMENTS_BENEFICIARY_ADD,
        "POST /v1/payments/transfers/initiate": PAYMENTS_TRANSFER_INITIATE,
    }
)
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


class ExecutableSimulator(Protocol):
    domain: str

    def execute(
        self,
        capability: str,
        *,
        request_id: str,
        idempotency_key: str,
        arguments: Mapping[str, Any],
    ) -> EffectReceipt:
        """Apply one business operation and return its effect receipt."""


class CredentialVerifier(Protocol):
    def verify(self, headers: Mapping[str, str]) -> bool:
        """Return whether the sidecar-injected target credential is valid."""


class BearerTokenDigestVerifier:
    """Verify a bearer token while storing only its SHA-256 digest."""

    def __init__(self, token_sha256: str) -> None:
        if _SHA256_HEX.fullmatch(token_sha256) is None:
            raise ValueError("token_sha256 must be 64 lowercase hex characters")
        self._token_sha256 = token_sha256

    @classmethod
    def from_token(cls, token: str) -> BearerTokenDigestVerifier:
        if not token:
            raise ValueError("token must be non-empty")
        return cls(hashlib.sha256(token.encode("utf-8")).hexdigest())

    def verify(self, headers: Mapping[str, str]) -> bool:
        authorization = ""
        for name, value in headers.items():
            if name.lower() == "authorization":
                authorization = value
                break
        prefix = "Bearer "
        if not authorization.startswith(prefix):
            return False
        token = authorization.removeprefix(prefix)
        if not token:
            return False
        observed = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return compare_digest(observed, self._token_sha256)


@dataclass(frozen=True, slots=True)
class SimulatorHTTPResponse:
    status_code: int
    body: bytes


class SimulatorApplication:
    """Route signed operations to one simulator without deciding policy."""

    def __init__(
        self,
        *,
        simulator: ExecutableSimulator,
        operations: Mapping[str, str],
        credential_verifier: CredentialVerifier,
    ) -> None:
        if not operations:
            raise ValueError("operations must be non-empty")
        unknown = set(operations) - set(SIMULATOR_OPERATIONS)
        if unknown:
            raise ValueError(f"operations contain unknown routes: {sorted(unknown)}")
        for operation, capability in operations.items():
            if SIMULATOR_OPERATIONS[operation] != capability:
                raise ValueError("operation capability does not match canonical route")
            if capability.partition(".")[0] != simulator.domain:
                raise ValueError("operation capability does not match simulator domain")
        self._simulator = simulator
        self._operations = dict(operations)
        self._credential_verifier = credential_verifier

    def handle(
        self,
        *,
        operation: str,
        headers: Mapping[str, str],
        wire_bytes: bytes,
    ) -> SimulatorHTTPResponse:
        if not self._credential_verifier.verify(headers):
            return _error_response(401, "INVALID_TARGET_CREDENTIAL")
        capability = self._operations.get(operation)
        if capability is None:
            return _error_response(404, "OPERATION_NOT_FOUND")
        if _header(headers, "content-type") != "application/json":
            return _error_response(415, "CONTENT_TYPE_NOT_SUPPORTED")
        content_encoding = _header(headers, "content-encoding") or "identity"
        if content_encoding != "identity":
            return _error_response(415, "CONTENT_ENCODING_NOT_SUPPORTED")
        try:
            payload = _decode_request(wire_bytes)
            arguments = payload["arguments"]
            if not isinstance(arguments, Mapping):
                raise SimulatorContractError("arguments must be a JSON object")
            receipt = self._simulator.execute(
                capability,
                request_id=required_text(payload["request_id"], "request_id"),
                idempotency_key=required_text(
                    payload["idempotency_key"], "idempotency_key"
                ),
                arguments=cast(Mapping[str, Any], arguments),
            )
        except SimulatorNotFound:
            return _error_response(404, "BUSINESS_OBJECT_NOT_FOUND")
        except SimulatorConflict:
            return _error_response(409, "BUSINESS_STATE_CONFLICT")
        except SimulatorBusinessError:
            return _error_response(422, "BUSINESS_CONSTRAINT_REJECTED")
        except (SimulatorContractError, UnicodeDecodeError, json.JSONDecodeError):
            return _error_response(400, "MALFORMED_SIMULATOR_REQUEST")
        except SimulatorError:
            return _error_response(500, "SIMULATOR_FAILURE")
        return SimulatorHTTPResponse(
            status_code=200,
            body=canonical_object(receipt.to_payload(), "effect receipt"),
        )


def encode_simulator_request(
    *,
    request_id: str,
    idempotency_key: str,
    arguments: Mapping[str, Any],
) -> bytes:
    """Encode the only body shape accepted by simulator targets."""

    return canonical_object(
        {
            "schema": SIMULATOR_REQUEST_SCHEMA,
            "request_id": required_text(request_id, "request_id"),
            "idempotency_key": required_text(idempotency_key, "idempotency_key"),
            "arguments": dict(arguments),
        },
        "simulator request",
    )


def serve_simulator(
    *,
    application: SimulatorApplication,
    host: str,
    port: int,
    ssl_context: ssl.SSLContext,
    max_request_bytes: int = 1024 * 1024,
) -> None:
    """Serve one target simulator over TLS without logging request content."""

    if not host.strip() or port < 1 or port > 65535:
        raise ValueError("simulator host or port is invalid")
    if max_request_bytes <= 0:
        raise ValueError("max_request_bytes must be positive")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            try:
                length = _content_length(self.headers.get("Content-Length"))
                if length > max_request_bytes:
                    self.send_error(413)
                    return
                wire_bytes = self.rfile.read(length)
                if len(wire_bytes) != length:
                    self.send_error(400)
                    return
                response = application.handle(
                    operation=f"POST {self.path}",
                    headers={key: value for key, value in self.headers.items()},
                    wire_bytes=wire_bytes,
                )
            except Exception:
                self.send_error(400)
                return
            self.send_response(response.status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response.body)))
            self.end_headers()
            self.wfile.write(response.body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    server.socket = ssl_context.wrap_socket(server.socket, server_side=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _decode_request(wire_bytes: bytes) -> dict[str, Any]:
    if not isinstance(wire_bytes, bytes) or not wire_bytes:
        raise SimulatorContractError("wire_bytes must be non-empty bytes")
    payload = json.loads(wire_bytes)
    if not isinstance(payload, dict):
        raise SimulatorContractError("simulator request must be a JSON object")
    exact_fields(
        payload,
        frozenset({"schema", "request_id", "idempotency_key", "arguments"}),
        "simulator request",
    )
    if payload["schema"] != SIMULATOR_REQUEST_SCHEMA:
        raise SimulatorContractError("simulator request schema is invalid")
    return payload


def _error_response(status_code: int, reason: str) -> SimulatorHTTPResponse:
    return SimulatorHTTPResponse(
        status_code=status_code,
        body=canonical_object(
            {"schema": "cbrain-simulator-error/v1", "reason": reason},
            "simulator error",
        ),
    )


def _header(headers: Mapping[str, str], name: str) -> str:
    for header_name, value in headers.items():
        if header_name.lower() == name:
            return value
    return ""


def _content_length(value: str | None) -> int:
    if value is None or not value.isdigit():
        raise SimulatorContractError("Content-Length is required")
    return int(value)


__all__ = [
    "BearerTokenDigestVerifier",
    "CredentialVerifier",
    "ExecutableSimulator",
    "SIMULATOR_OPERATIONS",
    "SimulatorApplication",
    "SimulatorHTTPResponse",
    "encode_simulator_request",
    "serve_simulator",
]
