"""The dispatch boundary.

`DispatchTransport` is the seam between a co-located dispatcher and an
independent one. `InProcessDispatchTransport` is **dev/test only**: it runs
the tool handler inside the agent process and cannot produce production
EXECUTED against real Agent DNA. Production assemblies must use an
independent sidecar (`identity.independent is True`); see
`require_production_dispatch_transport`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol

from cbrain.dispatch import PreparedDispatch

if TYPE_CHECKING:
    from cbrain.adapters.privatevault_execution import (
        ExecutionAuthorizationBinding,
    )

WitnessSigner = Callable[..., Mapping[str, Any]]

# Marker type alias for static review: in-process is never a prod transport.
DEV_ONLY_DISPATCH: Final = True


class DispatchTransportError(RuntimeError):
    """The dispatch boundary could not produce witnessed evidence."""


class HandlerNotInvoked(DispatchTransportError):
    """Failure is proven to have occurred before the tool handler ran."""


class ProductionDispatchRequired(DispatchTransportError):
    """Production refused a non-independent / in-process dispatch path."""


@dataclass(frozen=True, slots=True)
class DispatchResult:
    """What the dispatch boundary observed and signed."""

    witness: Mapping[str, Any]
    dispatch_outcome: str
    response_status: str
    response_bytes: bytes
    effect_state: str
    output: Any
    closure: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class WitnessIdentity:
    """Signing identity of the component that owns the dispatch boundary."""

    witness_component_id: str
    signer_key_id: str
    independent: bool


class DispatchTransport(Protocol):
    identity: WitnessIdentity

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
        """Transmit the exact bytes and return signed dispatch evidence."""


def require_production_dispatch_transport(transport: DispatchTransport) -> None:
    """Fail closed if the assembly tries to ship in-process as production."""

    if isinstance(transport, InProcessDispatchTransport):
        raise ProductionDispatchRequired(
            "InProcessDispatchTransport is DEV_ONLY; production requires an "
            "independent sidecar sole-egress path"
        )
    if not transport.identity.independent:
        raise ProductionDispatchRequired(
            "production dispatch transport must declare an independent witness"
        )


class InProcessDispatchTransport:
    """DEV_ONLY unit-test double: dispatch and witness in the agent process.

    It cannot produce EXECUTED against real Agent DNA. The witness signature
    is real and the digests are computed from the same byte string the permit
    commits to, but this transport returns no closure, and production requires
    an independent dispatch boundary to seal one. `identity.independent` is
    False; nothing here prevents the process from signing a witness for bytes
    it did not actually transmit.

    Do not wire this into `DeploymentConfig` / production startup. Call
    `require_production_dispatch_transport` at assembly time.
    """

    # Class attribute for grep/type-system audits.
    DEV_ONLY: Final[bool] = True

    def __init__(
        self,
        *,
        handler_runner: Callable[[Mapping[str, Any]], Any],
        witness_signer: WitnessSigner,
        signing_key: Any,
        identity: WitnessIdentity,
        response_encoder: Callable[[Any], bytes] | None = None,
    ) -> None:
        if identity.independent:
            raise DispatchTransportError(
                "in-process transport cannot claim an independent witness"
            )

        self._handler_runner = handler_runner
        self._witness_signer = witness_signer
        self._signing_key = signing_key
        self.identity = identity
        self._response_encoder = response_encoder or _default_response_encoder

    def with_handler(
        self,
        handler_runner: Callable[[Mapping[str, Any]], Any],
    ) -> InProcessDispatchTransport:
        """Return a copy bound to one request's single-entry handler."""

        return InProcessDispatchTransport(
            handler_runner=handler_runner,
            witness_signer=self._witness_signer,
            signing_key=self._signing_key,
            identity=self.identity,
            response_encoder=self._response_encoder,
        )

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
        dispatch_document = prepared.dispatch

        try:
            arguments = action.get("parameters")
            if not isinstance(arguments, Mapping):
                raise HandlerNotInvoked("action.parameters must be a mapping")
        except HandlerNotInvoked:
            raise
        except Exception as exc:
            raise HandlerNotInvoked(
                f"dispatch preparation failed:{type(exc).__name__}"
            ) from exc

        # Everything above this line is proven pre-execution. Everything below
        # may have caused a real effect and is therefore never retryable.
        try:
            output = self._handler_runner(arguments)
            dispatch_outcome = "ACKNOWLEDGED"
            response_status = "200"
            effect_state = "CONFIRMED"
        except Exception as exc:
            raise DispatchTransportError(
                f"tool handler failed:{type(exc).__name__}"
            ) from exc

        try:
            response_bytes = self._response_encoder(output)
        except Exception as exc:
            raise DispatchTransportError(
                f"response encoding failed:{type(exc).__name__}"
            ) from exc

        metadata = {
            "dispatch_witness_id": witness_id,
            "observed_at": observed_at,
            "attempt": attempt,
            "witness_component_id": self.identity.witness_component_id,
            "wire_content_type": dispatch_document["wire_content_type"],
            "wire_content_encoding": (dispatch_document["wire_content_encoding"]),
            "signer_key_id": self.identity.signer_key_id,
        }

        try:
            witness = self._witness_signer(
                metadata,
                authorization=authorization,
                trust_bundle=trust_bundle,
                observed_action=action,
                observed_dispatch=dispatch_document,
                wire_bytes=prepared.wire_bytes,
                peer_identity_bytes=prepared.peer_identity_bytes,
                signing_key=self._signing_key,
            )
        except Exception as exc:
            raise DispatchTransportError(
                f"witness creation failed:{type(exc).__name__}"
            ) from exc

        return DispatchResult(
            witness=witness,
            dispatch_outcome=dispatch_outcome,
            response_status=response_status,
            response_bytes=response_bytes,
            effect_state=effect_state,
            output=output,
        )


def _default_response_encoder(output: Any) -> bytes:
    import json

    if isinstance(output, bytes):
        return output

    return json.dumps(
        output,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")


__all__ = [
    "DEV_ONLY_DISPATCH",
    "DispatchResult",
    "DispatchTransport",
    "DispatchTransportError",
    "HandlerNotInvoked",
    "InProcessDispatchTransport",
    "ProductionDispatchRequired",
    "WitnessIdentity",
    "require_production_dispatch_transport",
]
