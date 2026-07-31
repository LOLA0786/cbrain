"""Shared contracts for executable target-system simulators.

The simulators model business state and idempotent effects. They deliberately
contain no authorization policy: a request can reach these objects only after
the governed execution path has authorized and dispatched it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

SIMULATOR_EFFECT_SCHEMA = "cbrain-simulator-effect/v1"
SIMULATOR_REQUEST_SCHEMA = "cbrain-simulator-request/v1"


class SimulatorError(RuntimeError):
    """Base error raised by an executable simulator target."""


class SimulatorContractError(SimulatorError):
    """A simulator request violated the target contract."""


class SimulatorNotFound(SimulatorError):
    """A referenced business object does not exist."""


class SimulatorConflict(SimulatorError):
    """A request conflicts with current state or prior idempotent use."""


class SimulatorBusinessError(SimulatorError):
    """A request violates ordinary target-system business constraints."""


JsonObject = dict[str, Any]


@dataclass(frozen=True, slots=True)
class EffectReceipt:
    """Deterministic receipt returned by a mutable simulator."""

    domain: str
    capability: str
    request_id: str
    idempotency_key: str
    effect_id: str
    state_version: int
    mutated: bool
    _result_json: bytes

    def __post_init__(self) -> None:
        for field_name in (
            "domain",
            "capability",
            "request_id",
            "idempotency_key",
            "effect_id",
        ):
            required_text(getattr(self, field_name), field_name)
        if self.state_version < 0:
            raise SimulatorContractError("state_version must be non-negative")
        if not isinstance(self.mutated, bool):
            raise SimulatorContractError("mutated must be boolean")
        if not isinstance(self._result_json, bytes):
            raise SimulatorContractError("_result_json must be immutable bytes")
        restore_object(self._result_json, "result")

    @classmethod
    def capture(
        cls,
        *,
        domain: str,
        capability: str,
        request_id: str,
        idempotency_key: str,
        effect_id: str,
        state_version: int,
        mutated: bool,
        result: Mapping[str, Any],
    ) -> EffectReceipt:
        return cls(
            domain=domain,
            capability=capability,
            request_id=request_id,
            idempotency_key=idempotency_key,
            effect_id=effect_id,
            state_version=state_version,
            mutated=mutated,
            _result_json=canonical_object(result, "result"),
        )

    @property
    def result(self) -> JsonObject:
        return restore_object(self._result_json, "result")

    def to_payload(self) -> JsonObject:
        return {
            "schema": SIMULATOR_EFFECT_SCHEMA,
            "domain": self.domain,
            "capability": self.capability,
            "request_id": self.request_id,
            "idempotency_key": self.idempotency_key,
            "effect_id": self.effect_id,
            "state_version": self.state_version,
            "mutated": self.mutated,
            "result": self.result,
        }


@dataclass(frozen=True, slots=True)
class StoredEffect:
    request_digest: str
    receipt: EffectReceipt


def capture_request(
    *,
    capability: str,
    request_id: str,
    idempotency_key: str,
    arguments: Mapping[str, Any],
) -> tuple[JsonObject, str]:
    required_text(capability, "capability")
    required_text(request_id, "request_id")
    required_text(idempotency_key, "idempotency_key")
    arguments_json = canonical_object(arguments, "arguments")
    arguments_snapshot = restore_object(arguments_json, "arguments")
    request_json = canonical_object(
        {
            "capability": capability,
            "request_id": request_id,
            "idempotency_key": idempotency_key,
            "arguments": arguments_snapshot,
        },
        "simulator request",
    )
    return arguments_snapshot, sha256_digest(request_json)


def effect_identifier(
    *,
    domain: str,
    capability: str,
    idempotency_key: str,
    request_digest: str,
) -> str:
    material = "\x00".join(
        (domain, capability, idempotency_key, request_digest)
    ).encode("utf-8")
    return f"effect-{hashlib.sha256(material).hexdigest()}"


def canonical_object(value: Mapping[str, Any], path: str) -> bytes:
    if not isinstance(value, Mapping):
        raise SimulatorContractError(f"{path} must be a mapping")
    validate_json(value, path, set())
    try:
        return json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SimulatorContractError(f"{path} must contain finite JSON values") from exc


def restore_object(value: bytes, path: str) -> JsonObject:
    try:
        restored = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SimulatorContractError(f"{path} is not valid JSON") from exc
    if not isinstance(restored, dict):
        raise SimulatorContractError(f"{path} must be a JSON object")
    return restored


def validate_json(value: Any, path: str, ancestors: set[int]) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SimulatorContractError(f"{path} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in ancestors:
            raise SimulatorContractError(f"{path} must not contain cycles")
        ancestors.add(identity)
        try:
            for key, child in value.items():
                if not isinstance(key, str):
                    raise SimulatorContractError(f"{path} keys must be strings")
                validate_json(child, f"{path}.*", ancestors)
        finally:
            ancestors.remove(identity)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        identity = id(value)
        if identity in ancestors:
            raise SimulatorContractError(f"{path} must not contain cycles")
        ancestors.add(identity)
        try:
            for child in value:
                validate_json(child, f"{path}[]", ancestors)
        finally:
            ancestors.remove(identity)
        return
    raise SimulatorContractError(f"{path} contains a non-JSON value")


def sha256_digest(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def required_text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SimulatorContractError(f"{path} must be non-empty text")
    return value


def exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    path: str,
) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise SimulatorContractError(
            f"{path} fields are invalid; missing={missing}, unexpected={unexpected}"
        )


__all__ = [
    "EffectReceipt",
    "JsonObject",
    "SIMULATOR_EFFECT_SCHEMA",
    "SIMULATOR_REQUEST_SCHEMA",
    "SimulatorBusinessError",
    "SimulatorConflict",
    "SimulatorContractError",
    "SimulatorError",
    "SimulatorNotFound",
    "StoredEffect",
    "canonical_object",
    "capture_request",
    "effect_identifier",
    "exact_fields",
    "required_text",
    "restore_object",
    "sha256_digest",
]
