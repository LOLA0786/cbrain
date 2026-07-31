from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .contracts import (
    ContractError,
    _restore_object,
    _snapshot_object,
)

_REQUIRED_DISPATCH_FIELDS = frozenset(
    {
        "transport",
        "destination",
        "operation",
        "wire_content_type",
        "wire_content_encoding",
        "tool_id",
        "tool_schema_digest",
        "tool_artifact_digest",
        "credential_audience",
        "idempotency_key_digest",
        "retry_policy_digest",
    }
)

_TEXT_FIELDS = frozenset(
    {
        "transport",
        "destination",
        "operation",
        "wire_content_type",
        "wire_content_encoding",
        "tool_id",
        "credential_audience",
    }
)

_DIGEST_FIELDS = frozenset(
    {
        "tool_schema_digest",
        "tool_artifact_digest",
        "idempotency_key_digest",
        "retry_policy_digest",
    }
)

_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class PreparedDispatch:
    """Immutable exact-byte dispatch prepared before authorization."""

    request_id: str
    _dispatch_json: bytes
    wire_bytes: bytes
    peer_identity_bytes: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ContractError("request_id must be a non-empty string")

        if not isinstance(self._dispatch_json, bytes):
            raise ContractError("_dispatch_json must be bytes")

        dispatch = _restore_object(self._dispatch_json)
        self._validate_dispatch(dispatch)

        if not isinstance(self.wire_bytes, bytes) or not self.wire_bytes:
            raise ContractError("wire_bytes must be non-empty immutable bytes")

        if (
            not isinstance(self.peer_identity_bytes, bytes)
            or not self.peer_identity_bytes
        ):
            raise ContractError("peer_identity_bytes must be non-empty immutable bytes")

    @classmethod
    def capture(
        cls,
        *,
        request_id: str,
        dispatch: Mapping[str, Any],
        wire_bytes: bytes,
        peer_identity_bytes: bytes,
    ) -> PreparedDispatch:
        snapshot = _snapshot_object(
            dispatch,
            "dispatch",
        )

        return cls(
            request_id=request_id,
            _dispatch_json=snapshot,
            wire_bytes=bytes(wire_bytes),
            peer_identity_bytes=bytes(peer_identity_bytes),
        )

    @property
    def dispatch(self) -> dict[str, Any]:
        return _restore_object(self._dispatch_json)

    @staticmethod
    def _validate_dispatch(
        dispatch: Mapping[str, Any],
    ) -> None:
        actual_fields = frozenset(dispatch)

        if actual_fields != _REQUIRED_DISPATCH_FIELDS:
            missing = sorted(_REQUIRED_DISPATCH_FIELDS - actual_fields)
            unexpected = sorted(actual_fields - _REQUIRED_DISPATCH_FIELDS)
            raise ContractError(
                "dispatch fields do not match "
                f"PrivateVault contract; missing={missing}, "
                f"unexpected={unexpected}"
            )

        for field_name in _TEXT_FIELDS:
            value = dispatch[field_name]
            if not isinstance(value, str) or not value.strip():
                raise ContractError(f"dispatch.{field_name} must be non-empty text")

        for field_name in _DIGEST_FIELDS:
            value = dispatch[field_name]
            if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
                raise ContractError(f"dispatch.{field_name} must be a sha256 digest")
