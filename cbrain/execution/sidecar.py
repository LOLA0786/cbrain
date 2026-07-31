"""Independent sole-egress dispatcher sidecar.

The agent sends an authorization, its exact prepared bytes, and immutable
binding data to this boundary. The sidecar resolves credentials outside model
context, opens an allow-listed TLS route, verifies the observed peer identity,
atomically consumes the authorization, transmits the exact bytes, and signs the
witness and closure from what it actually observed.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import http.client
import json
import os
import re
import ssl
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol, cast
from urllib import request as urllib_request
from urllib.parse import urlsplit

from cbrain.adapters.privatevault_execution import ExecutionAuthorizationBinding
from cbrain.dispatch import PreparedDispatch
from cbrain.execution.gateway import ClosureWriter
from cbrain.execution.transport import (
    DispatchResult,
    DispatchTransportError,
    HandlerNotInvoked,
    WitnessIdentity,
)

SIDECAR_REQUEST_SCHEMA = "cbrain-dispatch-sidecar/request-v1"
SIDECAR_RESPONSE_SCHEMA = "cbrain-dispatch-sidecar/response-v1"
_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")

WitnessSigner = Callable[..., Mapping[str, Any]]


class SidecarError(RuntimeError):
    """Base failure for the independent dispatch boundary."""


class SidecarProtocolError(SidecarError):
    """A sidecar request or response violated the wire contract."""


class SidecarRPCError(SidecarError):
    """The agent could not obtain a conclusive sidecar response."""


class _SidecarDispatchFailure(SidecarError):
    def __init__(self, reason: str, *, execution_possible: bool) -> None:
        self.reason = reason
        self.execution_possible = execution_possible
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class CredentialHeader:
    """A credential resolved only inside the sidecar process."""

    name: str
    value: str = field(repr=False)

    def __post_init__(self) -> None:
        if _HEADER_NAME.fullmatch(self.name) is None:
            raise SidecarError("credential header name is invalid")
        if not self.value or "\r" in self.value or "\n" in self.value:
            raise SidecarError("credential value is invalid")


@dataclass(frozen=True, slots=True)
class EnvironmentCredentialBinding:
    """Map a signed audience to one sidecar-only environment variable."""

    environment_variable: str
    header_name: str = "Authorization"
    value_prefix: str = "Bearer "

    def __post_init__(self) -> None:
        if not self.environment_variable.strip():
            raise ValueError("environment_variable must be non-empty")
        if _HEADER_NAME.fullmatch(self.header_name) is None:
            raise ValueError("header_name is invalid")
        if "\r" in self.value_prefix or "\n" in self.value_prefix:
            raise ValueError("value_prefix is invalid")


class CredentialProvider(Protocol):
    def resolve(self, audience: str) -> CredentialHeader:
        """Resolve a credential without returning it to the agent process."""


class EnvironmentCredentialProvider:
    """Default-deny credential provider for production sidecars."""

    def __init__(
        self,
        bindings: Mapping[str, EnvironmentCredentialBinding],
    ) -> None:
        self._bindings = dict(bindings)

    def resolve(self, audience: str) -> CredentialHeader:
        binding = self._bindings.get(audience)
        if binding is None:
            raise SidecarError("credential audience is not configured")

        value = os.environ.get(binding.environment_variable, "")
        if not value:
            raise SidecarError("credential environment variable is unavailable")

        return CredentialHeader(
            name=binding.header_name,
            value=f"{binding.value_prefix}{value}",
        )


@dataclass(frozen=True, slots=True)
class TargetResponse:
    dispatch_outcome: str
    response_status: str
    response_bytes: bytes
    effect_state: str
    output: Any


class TargetChannel(Protocol):
    peer_identity_bytes: bytes

    def send_exact(
        self,
        *,
        wire_bytes: bytes,
        wire_content_type: str,
        wire_content_encoding: str,
        credential: CredentialHeader,
    ) -> TargetResponse:
        """Send the exact body bytes after the TLS peer is observed."""

    def close(self) -> None:
        """Release the target connection."""


class TargetConnector(Protocol):
    def open(self, dispatch: Mapping[str, Any]) -> TargetChannel:
        """Open an allow-listed route without transmitting action bytes."""


class AuthorizationClaimant(Protocol):
    def verify_and_claim(
        self,
        *,
        authorization: Mapping[str, Any],
        trust_bundle: Mapping[str, Any] | None,
        binding: ExecutionAuthorizationBinding,
        claimed_at: str | None = None,
    ) -> object:
        """Verify and atomically consume one authorization."""


class SidecarRPC(Protocol):
    def call(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Send one request to the independent sidecar process."""


class SidecarDispatchService:
    """Security-critical sidecar core, independent of its HTTP adapter."""

    def __init__(
        self,
        *,
        claimant: AuthorizationClaimant,
        connector: TargetConnector,
        credential_provider: CredentialProvider,
        witness_signer: WitnessSigner,
        witness_signing_key: Any,
        identity: WitnessIdentity,
        closure_writer: ClosureWriter,
        clock: Callable[[], str],
    ) -> None:
        if not identity.independent:
            raise SidecarError("sidecar witness identity must be independent")
        self._claimant = claimant
        self._connector = connector
        self._credential_provider = credential_provider
        self._witness_signer = witness_signer
        self._witness_signing_key = witness_signing_key
        self._identity = identity
        self._closure_writer = closure_writer
        self._clock = clock

    def handle(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        request_id = _possible_request_id(payload)
        try:
            envelope = _decode_request(payload)
        except Exception as exc:
            return {
                "schema": SIDECAR_RESPONSE_SCHEMA,
                "status": "REFUSED",
                "phase": "PRE_DISPATCH",
                "request_id": request_id,
                "reason": f"malformed_request:{type(exc).__name__}",
            }

        try:
            result = self._dispatch(envelope)
        except _SidecarDispatchFailure as exc:
            return {
                "schema": SIDECAR_RESPONSE_SCHEMA,
                "status": ("INDETERMINATE" if exc.execution_possible else "REFUSED"),
                "phase": (
                    "POST_DISPATCH" if exc.execution_possible else "PRE_DISPATCH"
                ),
                "request_id": request_id,
                "reason": exc.reason,
            }

        try:
            response = _encode_success(envelope.binding.request_id, result)
            _canonical_json(response)
            return response
        except Exception as exc:
            return {
                "schema": SIDECAR_RESPONSE_SCHEMA,
                "status": "INDETERMINATE",
                "phase": "POST_DISPATCH",
                "request_id": request_id,
                "reason": f"response_unavailable:{type(exc).__name__}",
            }

    def _dispatch(self, envelope: _SidecarEnvelope) -> DispatchResult:
        prepared = envelope.binding.prepared_dispatch
        dispatch = prepared.dispatch
        channel: TargetChannel | None = None
        send_started = False

        try:
            sidecar_observed_at = self._clock()
            try:
                wire_content_type = _safe_header_value(
                    _required_text(dispatch, "wire_content_type"),
                    "wire_content_type",
                )
                wire_content_encoding = _safe_header_value(
                    _required_text(dispatch, "wire_content_encoding"),
                    "wire_content_encoding",
                )
                credential = self._credential_provider.resolve(
                    _required_text(dispatch, "credential_audience")
                )
                channel = self._connector.open(dispatch)
                if not _same_bytes(
                    channel.peer_identity_bytes,
                    prepared.peer_identity_bytes,
                ):
                    raise SidecarError("observed TLS peer does not match permit")

                self._claimant.verify_and_claim(
                    authorization=envelope.authorization,
                    trust_bundle=envelope.trust_bundle,
                    binding=envelope.binding,
                    claimed_at=sidecar_observed_at,
                )
            except Exception as exc:
                raise _SidecarDispatchFailure(
                    f"pre_dispatch_refused:{type(exc).__name__}",
                    execution_possible=False,
                ) from exc

            send_started = True
            response = channel.send_exact(
                wire_bytes=prepared.wire_bytes,
                wire_content_type=wire_content_type,
                wire_content_encoding=wire_content_encoding,
                credential=credential,
            )

            if not isinstance(response.response_bytes, bytes):
                raise SidecarProtocolError("target response bytes are invalid")

            metadata = {
                "dispatch_witness_id": envelope.witness_id,
                "observed_at": sidecar_observed_at,
                "attempt": envelope.attempt,
                "witness_component_id": self._identity.witness_component_id,
                "wire_content_type": dispatch["wire_content_type"],
                "wire_content_encoding": dispatch["wire_content_encoding"],
                "signer_key_id": self._identity.signer_key_id,
            }
            witness = _required_mapping(
                self._witness_signer(
                    metadata,
                    authorization=envelope.authorization,
                    trust_bundle=envelope.trust_bundle,
                    observed_action=envelope.binding.action,
                    observed_dispatch=dispatch,
                    wire_bytes=prepared.wire_bytes,
                    peer_identity_bytes=channel.peer_identity_bytes,
                    signing_key=self._witness_signing_key,
                ),
                "dispatch witness",
            )
            closure = _required_mapping(
                self._closure_writer.seal(
                    authorization=envelope.authorization,
                    witness=witness,
                    trust_bundle=envelope.trust_bundle,
                    dispatch_outcome=response.dispatch_outcome,
                    response_status=response.response_status,
                    response_bytes=response.response_bytes,
                    effect_state=response.effect_state,
                    idempotency_key_digest=_required_text(
                        dispatch, "idempotency_key_digest"
                    ),
                ),
                "closure",
            )
            return DispatchResult(
                witness=witness,
                dispatch_outcome=response.dispatch_outcome,
                response_status=response.response_status,
                response_bytes=response.response_bytes,
                effect_state=response.effect_state,
                output=response.output,
                closure=closure,
            )
        except _SidecarDispatchFailure:
            raise
        except Exception as exc:
            raise _SidecarDispatchFailure(
                f"post_dispatch_unproven:{type(exc).__name__}",
                execution_possible=send_started,
            ) from exc
        finally:
            if channel is not None:
                with suppress(Exception):
                    channel.close()


@dataclass(frozen=True, slots=True)
class _SidecarEnvelope:
    authorization: Mapping[str, Any]
    trust_bundle: Mapping[str, Any]
    binding: ExecutionAuthorizationBinding
    witness_id: str
    attempt: int


class SidecarDispatchTransport:
    """Agent-side transport for an independent dispatcher sidecar."""

    def __init__(self, *, rpc: SidecarRPC, identity: WitnessIdentity) -> None:
        if not identity.independent:
            raise DispatchTransportError(
                "sidecar transport must declare an independent witness"
            )
        self._rpc = rpc
        self.identity = identity

    def dispatch(
        self,
        *,
        authorization: Mapping[str, Any],
        trust_bundle: Mapping[str, Any],
        action: Mapping[str, Any],
        prepared: PreparedDispatch,
        witness_id: str,
        observed_at: str,
        attempt: int,
        binding: ExecutionAuthorizationBinding,
    ) -> DispatchResult:
        if binding.action != dict(action) or binding.prepared_dispatch != prepared:
            raise HandlerNotInvoked("sidecar binding does not match planned dispatch")

        payload = _encode_request(
            authorization=authorization,
            trust_bundle=trust_bundle,
            binding=binding,
            witness_id=witness_id,
            observed_at=observed_at,
            attempt=attempt,
        )
        try:
            response = self._rpc.call(payload)
        except Exception as exc:
            raise DispatchTransportError(
                f"sidecar response unavailable:{type(exc).__name__}"
            ) from exc

        return _decode_response(response, binding.request_id)


class HTTPSidecarRPC:
    """Strict HTTPS/mTLS JSON transport for the sidecar process."""

    def __init__(
        self,
        *,
        endpoint: str,
        ssl_context: ssl.SSLContext,
        timeout_seconds: float = 5.0,
        max_response_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        parsed = urlsplit(endpoint)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("sidecar endpoint must be an absolute HTTPS URL")
        if parsed.query or parsed.fragment:
            raise ValueError("sidecar endpoint must not contain query or fragment")
        if (
            ssl_context.verify_mode != ssl.CERT_REQUIRED
            or not ssl_context.check_hostname
        ):
            raise ValueError(
                "sidecar TLS context must verify certificates and hostnames"
            )
        if timeout_seconds <= 0 or max_response_bytes <= 0:
            raise ValueError("sidecar transport limits must be positive")

        self._endpoint = endpoint
        self._ssl_context = ssl_context
        self._timeout_seconds = timeout_seconds
        self._max_response_bytes = max_response_bytes

    def call(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        body = _canonical_json(payload)
        request = urllib_request.Request(
            self._endpoint,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib_request.urlopen(
                request,
                timeout=self._timeout_seconds,
                context=self._ssl_context,
            ) as response:
                response_body = response.read(self._max_response_bytes + 1)
        except Exception as exc:
            raise SidecarRPCError(
                f"sidecar HTTPS request failed:{type(exc).__name__}"
            ) from exc

        if len(response_body) > self._max_response_bytes:
            raise SidecarRPCError("sidecar response exceeds configured limit")
        try:
            decoded = json.loads(response_body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SidecarRPCError("sidecar response is not valid JSON") from exc
        return _required_mapping(decoded, "sidecar response")


@dataclass(frozen=True, slots=True)
class HTTPSRoute:
    destination: str
    host: str
    port: int
    operations: frozenset[str]
    credential_audience: str

    def __post_init__(self) -> None:
        if not self.destination.strip() or not self.host.strip():
            raise ValueError("route destination and host must be non-empty")
        if self.port < 1 or self.port > 65535:
            raise ValueError("route port is invalid")
        if not self.operations or any(not item.strip() for item in self.operations):
            raise ValueError("route operations must be non-empty")
        if not self.credential_audience.strip():
            raise ValueError("route credential audience must be non-empty")


class PinnedHTTPSConnector:
    """Allow-listed HTTPS connector; model-provided destinations are never used."""

    def __init__(
        self,
        *,
        routes: Mapping[str, HTTPSRoute],
        ssl_context: ssl.SSLContext,
        timeout_seconds: float = 10.0,
        max_response_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        if (
            ssl_context.verify_mode != ssl.CERT_REQUIRED
            or not ssl_context.check_hostname
        ):
            raise ValueError(
                "target TLS context must verify certificates and hostnames"
            )
        if timeout_seconds <= 0 or max_response_bytes <= 0:
            raise ValueError("target transport limits must be positive")
        self._routes = dict(routes)
        self._ssl_context = ssl_context
        self._timeout_seconds = timeout_seconds
        self._max_response_bytes = max_response_bytes

    def open(self, dispatch: Mapping[str, Any]) -> TargetChannel:
        destination = _required_text(dispatch, "destination")
        route = self._routes.get(destination)
        if route is None or route.destination != destination:
            raise SidecarError("destination is not allow-listed")
        if _required_text(dispatch, "transport") != "https":
            raise SidecarError("allow-listed route requires HTTPS")
        operation = _required_text(dispatch, "operation")
        if operation not in route.operations:
            raise SidecarError("operation is not allow-listed")
        if _required_text(dispatch, "credential_audience") != route.credential_audience:
            raise SidecarError("credential audience does not match route")

        return _PinnedHTTPSChannel(
            route=route,
            operation=operation,
            ssl_context=self._ssl_context,
            timeout_seconds=self._timeout_seconds,
            max_response_bytes=self._max_response_bytes,
        )


class _PinnedHTTPSChannel:
    def __init__(
        self,
        *,
        route: HTTPSRoute,
        operation: str,
        ssl_context: ssl.SSLContext,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> None:
        self._route = route
        self._operation = operation
        self._max_response_bytes = max_response_bytes
        self._connection = http.client.HTTPSConnection(
            route.host,
            route.port,
            timeout=timeout_seconds,
            context=ssl_context,
        )
        self._connection.connect()
        socket = self._connection.sock
        if socket is None:
            raise SidecarError("target TLS connection has no peer socket")
        certificate = socket.getpeercert(binary_form=True)
        if not certificate:
            raise SidecarError("target TLS peer certificate is unavailable")
        self.peer_identity_bytes = (
            f"tls-cert-sha256:{hashlib.sha256(certificate).hexdigest()}".encode()
        )

    def send_exact(
        self,
        *,
        wire_bytes: bytes,
        wire_content_type: str,
        wire_content_encoding: str,
        credential: CredentialHeader,
    ) -> TargetResponse:
        method, path = _parse_operation(self._operation)
        self._connection.putrequest(method, path)
        self._connection.putheader("Content-Type", wire_content_type)
        self._connection.putheader("Content-Encoding", wire_content_encoding)
        self._connection.putheader("Content-Length", str(len(wire_bytes)))
        self._connection.putheader(credential.name, credential.value)
        self._connection.endheaders()
        self._connection.send(wire_bytes)
        response = self._connection.getresponse()
        response_bytes = response.read(self._max_response_bytes + 1)
        if len(response_bytes) > self._max_response_bytes:
            raise SidecarError("target response exceeds configured limit")

        return TargetResponse(
            dispatch_outcome="ACKNOWLEDGED",
            response_status=str(response.status),
            response_bytes=response_bytes,
            effect_state=("CONFIRMED" if 200 <= response.status < 300 else "UNKNOWN"),
            output=_decode_target_output(response_bytes),
        )

    def close(self) -> None:
        self._connection.close()


def serve_sidecar(
    *,
    service: SidecarDispatchService,
    host: str,
    port: int,
    ssl_context: ssl.SSLContext,
    max_request_bytes: int = 8 * 1024 * 1024,
) -> None:
    """Serve the dispatcher over TLS; configure mTLS on ``ssl_context``."""

    if ssl_context.verify_mode != ssl.CERT_REQUIRED:
        raise ValueError("sidecar server must require a verified client certificate")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/dispatch":
                self.send_error(404)
                return
            length = _content_length(self.headers.get("Content-Length"))
            if length > max_request_bytes:
                self.send_error(413)
                return
            body = self.rfile.read(length)
            try:
                payload = _required_mapping(json.loads(body), "sidecar request")
                response = service.handle(payload)
                encoded = _canonical_json(response)
            except Exception:
                self.send_error(400)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            # Request bodies can carry authorization evidence. Never let the
            # stdlib request logger print values derived from them.
            return

    server = ThreadingHTTPServer((host, port), Handler)
    server.socket = ssl_context.wrap_socket(server.socket, server_side=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _encode_request(
    *,
    authorization: Mapping[str, Any],
    trust_bundle: Mapping[str, Any],
    binding: ExecutionAuthorizationBinding,
    witness_id: str,
    observed_at: str,
    attempt: int,
) -> dict[str, Any]:
    prepared = binding.prepared_dispatch
    return {
        "schema": SIDECAR_REQUEST_SCHEMA,
        "authorization": dict(authorization),
        "trust_bundle": dict(trust_bundle),
        "action": binding.action,
        "prepared": {
            "request_id": prepared.request_id,
            "dispatch": prepared.dispatch,
            "wire_bytes": _b64(prepared.wire_bytes),
            "peer_identity_bytes": _b64(prepared.peer_identity_bytes),
        },
        "binding": {
            "decision_receipt_digest": binding.decision_receipt_digest,
            "authority_receipt_digest": binding.authority_receipt_digest,
            "approval_artifact_digest": binding.approval_artifact_digest,
            "state_snapshot_digest": binding.state_snapshot_digest,
            "policy_bundle_digest": binding.policy_bundle_digest,
            "obligations_digest": binding.obligations_digest,
            "at_time": binding.at_time,
        },
        "witness": {
            "witness_id": witness_id,
            "attempt": attempt,
        },
    }


def _decode_request(payload: Mapping[str, Any]) -> _SidecarEnvelope:
    if payload.get("schema") != SIDECAR_REQUEST_SCHEMA:
        raise SidecarProtocolError("sidecar request schema is invalid")
    authorization = _required_mapping(payload.get("authorization"), "authorization")
    trust_bundle = _required_mapping(payload.get("trust_bundle"), "trust_bundle")
    action = _required_mapping(payload.get("action"), "action")
    prepared_value = _required_mapping(payload.get("prepared"), "prepared")
    binding_value = _required_mapping(payload.get("binding"), "binding")
    witness = _required_mapping(payload.get("witness"), "witness")

    request_id = _required_text(prepared_value, "request_id")
    prepared = PreparedDispatch.capture(
        request_id=request_id,
        dispatch=_required_mapping(prepared_value.get("dispatch"), "dispatch"),
        wire_bytes=_unb64(prepared_value.get("wire_bytes"), "wire_bytes"),
        peer_identity_bytes=_unb64(
            prepared_value.get("peer_identity_bytes"), "peer_identity_bytes"
        ),
    )
    approval = binding_value.get("approval_artifact_digest")
    if approval is not None and not isinstance(approval, str):
        raise SidecarProtocolError("approval_artifact_digest is invalid")
    binding = ExecutionAuthorizationBinding.capture(
        request_id=request_id,
        action=action,
        prepared_dispatch=prepared,
        decision_receipt_digest=_required_text(
            binding_value, "decision_receipt_digest"
        ),
        authority_receipt_digest=_required_text(
            binding_value, "authority_receipt_digest"
        ),
        approval_artifact_digest=approval,
        state_snapshot_digest=_required_text(binding_value, "state_snapshot_digest"),
        policy_bundle_digest=_required_text(binding_value, "policy_bundle_digest"),
        obligations_digest=_required_text(binding_value, "obligations_digest"),
        at_time=_required_text(binding_value, "at_time"),
    )
    attempt = witness.get("attempt")
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        raise SidecarProtocolError("witness attempt is invalid")
    return _SidecarEnvelope(
        authorization=authorization,
        trust_bundle=trust_bundle,
        binding=binding,
        witness_id=_required_text(witness, "witness_id"),
        attempt=attempt,
    )


def _encode_success(request_id: str, result: DispatchResult) -> dict[str, Any]:
    if result.closure is None:
        raise SidecarProtocolError("independent dispatch produced no closure")
    return {
        "schema": SIDECAR_RESPONSE_SCHEMA,
        "status": "EXECUTED",
        "phase": "CLOSED",
        "request_id": request_id,
        "witness": dict(result.witness),
        "closure": dict(result.closure),
        "dispatch_outcome": result.dispatch_outcome,
        "response_status": result.response_status,
        "response_bytes": _b64(result.response_bytes),
        "effect_state": result.effect_state,
        "output": result.output,
    }


def _decode_response(
    response: Mapping[str, Any], expected_request_id: str
) -> DispatchResult:
    if response.get("schema") != SIDECAR_RESPONSE_SCHEMA:
        raise DispatchTransportError("sidecar response schema is invalid")
    request_id = _required_text(response, "request_id")
    if request_id != expected_request_id:
        raise DispatchTransportError("sidecar response request_id mismatch")
    status = _required_text(response, "status")
    phase = _required_text(response, "phase")
    if status == "REFUSED" and phase == "PRE_DISPATCH":
        raise HandlerNotInvoked(_safe_reason(response))
    if status != "EXECUTED" or phase != "CLOSED":
        raise DispatchTransportError(_safe_reason(response))

    return DispatchResult(
        witness=_required_mapping(response.get("witness"), "witness"),
        closure=_required_mapping(response.get("closure"), "closure"),
        dispatch_outcome=_required_text(response, "dispatch_outcome"),
        response_status=_required_text(response, "response_status"),
        response_bytes=_unb64(response.get("response_bytes"), "response_bytes"),
        effect_state=_required_text(response, "effect_state"),
        output=response.get("output"),
    )


def _required_mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise SidecarProtocolError(f"{path} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _required_text(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise SidecarProtocolError(f"{key} must be non-empty text")
    return item


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _unb64(value: object, path: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise SidecarProtocolError(f"{path} must be non-empty base64 text")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise SidecarProtocolError(f"{path} is invalid base64") from exc
    if not decoded:
        raise SidecarProtocolError(f"{path} must decode to non-empty bytes")
    return decoded


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SidecarProtocolError("sidecar payload is not finite JSON") from exc


def _same_bytes(left: bytes, right: bytes) -> bool:
    from hmac import compare_digest

    return compare_digest(left, right)


def _possible_request_id(payload: Mapping[str, Any]) -> str:
    prepared = payload.get("prepared")
    if isinstance(prepared, Mapping):
        value = prepared.get("request_id")
        if isinstance(value, str):
            return value
    return "unknown"


def _safe_reason(response: Mapping[str, Any]) -> str:
    reason = response.get("reason")
    return reason if isinstance(reason, str) and reason else "sidecar_refused"


def _parse_operation(operation: str) -> tuple[str, str]:
    try:
        method, path = operation.split(" ", 1)
    except ValueError as exc:
        raise SidecarError("operation must contain method and path") from exc
    if not method.isalpha() or method.upper() != method:
        raise SidecarError("operation method is invalid")
    if not path.startswith("/") or "\r" in path or "\n" in path:
        raise SidecarError("operation path is invalid")
    return method, path


def _safe_header_value(value: str, path: str) -> str:
    if "\r" in value or "\n" in value:
        raise SidecarProtocolError(f"{path} contains forbidden control characters")
    return value


def _decode_target_output(response_bytes: bytes) -> Any:
    try:
        return json.loads(response_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"content_base64": _b64(response_bytes)}


def _content_length(value: str | None) -> int:
    if value is None or not value.isdigit():
        raise SidecarProtocolError("Content-Length is required")
    return int(value)


__all__ = [
    "CredentialHeader",
    "CredentialProvider",
    "EnvironmentCredentialBinding",
    "EnvironmentCredentialProvider",
    "HTTPSRoute",
    "HTTPSidecarRPC",
    "PinnedHTTPSConnector",
    "SIDECAR_REQUEST_SCHEMA",
    "SIDECAR_RESPONSE_SCHEMA",
    "SidecarDispatchService",
    "SidecarDispatchTransport",
    "SidecarError",
    "SidecarProtocolError",
    "SidecarRPC",
    "SidecarRPCError",
    "TargetChannel",
    "TargetConnector",
    "TargetResponse",
    "serve_sidecar",
]
