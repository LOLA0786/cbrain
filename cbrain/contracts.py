from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ContractError(ValueError):
    """An action or result violated the governed-runtime contract."""


def _required_text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{path} must be a non-empty string")
    return value


def _validate_json(
    value: Any,
    path: str,
    ancestors: set[int],
) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError(f"{path} must contain only finite JSON values")
        return

    if isinstance(value, Mapping):
        identity = id(value)
        if identity in ancestors:
            raise ContractError(f"{path} must not contain cycles")

        ancestors.add(identity)
        try:
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ContractError(f"{path} keys must be strings")
                _validate_json(child, f"{path}.*", ancestors)
        finally:
            ancestors.remove(identity)
        return

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        identity = id(value)
        if identity in ancestors:
            raise ContractError(f"{path} must not contain cycles")

        ancestors.add(identity)
        try:
            for child in value:
                _validate_json(child, f"{path}[]", ancestors)
        finally:
            ancestors.remove(identity)
        return

    raise ContractError(f"{path} must contain only JSON-compatible values")


def _snapshot_object(value: Mapping[str, Any], path: str) -> bytes:
    if not isinstance(value, Mapping):
        raise ContractError(f"{path} must be a mapping")

    _validate_json(value, path, set())

    try:
        return json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{path} must contain only finite JSON values") from exc


def _restore_object(value: bytes) -> dict[str, Any]:
    restored = json.loads(value)

    if not isinstance(restored, dict):
        raise ContractError("stored action value must be a JSON object")

    return restored


@dataclass(frozen=True, slots=True)
class ActionIntent:
    """Immutable action snapshot captured before authorization."""

    request_id: str
    idempotency_key: str
    agent_id: str
    framework: str
    tool_name: str
    capability: str
    timestamp: float
    _arguments_json: bytes
    _context_json: bytes
    _evidence_json: bytes

    def __post_init__(self) -> None:
        for field_name in (
            "request_id",
            "idempotency_key",
            "agent_id",
            "framework",
            "tool_name",
            "capability",
        ):
            _required_text(getattr(self, field_name), field_name)

        if (
            isinstance(self.timestamp, bool)
            or not isinstance(self.timestamp, (int, float))
            or not math.isfinite(float(self.timestamp))
        ):
            raise ContractError("timestamp must be finite")

        for field_name in (
            "_arguments_json",
            "_context_json",
            "_evidence_json",
        ):
            stored = getattr(self, field_name)
            if not isinstance(stored, bytes):
                raise ContractError(f"{field_name} must be bytes")

            try:
                _restore_object(stored)
            except (
                TypeError,
                ValueError,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as exc:
                raise ContractError(f"{field_name} must contain a JSON object") from exc

    @classmethod
    def capture(
        cls,
        *,
        agent_id: str,
        framework: str,
        tool_name: str,
        capability: str,
        arguments: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
        evidence: Mapping[str, Any] | None = None,
        request_id: str | None = None,
        idempotency_key: str | None = None,
        timestamp: float | None = None,
    ) -> ActionIntent:
        resolved_request_id = request_id or str(uuid.uuid4())

        return cls(
            request_id=resolved_request_id,
            idempotency_key=idempotency_key or resolved_request_id,
            agent_id=agent_id,
            framework=framework,
            tool_name=tool_name,
            capability=capability,
            timestamp=time.time() if timestamp is None else timestamp,
            _arguments_json=_snapshot_object(arguments, "arguments"),
            _context_json=_snapshot_object(context or {}, "context"),
            _evidence_json=_snapshot_object(evidence or {}, "evidence"),
        )

    @property
    def arguments(self) -> dict[str, Any]:
        return _restore_object(self._arguments_json)

    @property
    def context(self) -> dict[str, Any]:
        return _restore_object(self._context_json)

    @property
    def evidence(self) -> dict[str, Any]:
        return _restore_object(self._evidence_json)

    def privatevault_decide_payload(self) -> dict[str, Any]:
        """Build the pinned PrivateVault `/v1/decide` request shape."""

        context = self.context
        context["cbrain"] = {
            "framework": self.framework,
            "tool_name": self.tool_name,
            "idempotency_key": self.idempotency_key,
        }

        return {
            "agent_id": self.agent_id,
            "capability": self.capability,
            "timestamp": self.timestamp,
            "arguments": self.arguments,
            "context": context,
            "evidence": self.evidence,
            "request_id": self.request_id,
        }


class ExecutionStatus(StrEnum):
    EXECUTED = "EXECUTED"
    BLOCKED = "BLOCKED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    CONTROL_FAILURE = "CONTROL_FAILURE"
    INDETERMINATE = "INDETERMINATE"


@dataclass(frozen=True, slots=True)
class GovernedExecution:
    status: ExecutionStatus
    request_id: str
    tool_executed: bool | None
    reason: str
    decision_id: str | None = None
    output: Any = None
    retryable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.status, ExecutionStatus):
            raise ContractError("status must be an ExecutionStatus")

        _required_text(self.request_id, "request_id")
        _required_text(self.reason, "reason")

        if self.status is ExecutionStatus.EXECUTED and self.tool_executed is not True:
            raise ContractError("EXECUTED requires tool_executed=True")

        if (
            self.status
            in {
                ExecutionStatus.BLOCKED,
                ExecutionStatus.REVIEW_REQUIRED,
                ExecutionStatus.CONTROL_FAILURE,
            }
            and self.tool_executed is not False
        ):
            raise ContractError(f"{self.status.value} requires tool_executed=False")

        if (
            self.status is ExecutionStatus.INDETERMINATE
            and self.tool_executed is not None
        ):
            raise ContractError(
                "INDETERMINATE requires tool_executed=None; execution may have occurred"
            )

        if self.retryable and self.status is not ExecutionStatus.CONTROL_FAILURE:
            raise ContractError(
                "only a proven pre-execution CONTROL_FAILURE may be retryable"
            )

        if self.retryable and self.tool_executed is not False:
            raise ContractError("only a proven pre-execution failure may be retryable")
