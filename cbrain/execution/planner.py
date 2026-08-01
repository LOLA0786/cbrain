"""Turn a governed intent into exact outbound bytes.

The planner is the component that decides what will physically go on the wire.
Two properties matter more than anything else here:

1. **Byte determinism.** The same intent must produce the same bytes every
   time, or the permit's digest cannot bind them. Canonical JSON with sorted
   keys and fixed separators; no timestamps, no random ids, no dict ordering
   dependence.

2. **No credentials.** The planner never sees or emits a secret. It emits a
   `credential_audience` naming which credential the dispatch boundary should
   attach. The model proposed the arguments; the model must never be able to
   name, read, or influence the credential.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cbrain.contracts import ActionIntent
from cbrain.dispatch import PreparedDispatch
from cbrain.execution.gateway import PlannedDispatch


class DispatchPlanningError(RuntimeError):
    """The intent could not be turned into a permitted outbound request."""


@dataclass(frozen=True, slots=True)
class ToolRoute:
    """Static, operator-authored description of one callable tool.

    This is configuration, not model output. A tool the operator has not
    routed cannot be dispatched, regardless of what the agent proposes.
    """

    tool_id: str
    capability: str
    destination: str
    operation: str
    credential_audience: str
    peer_identity: str
    allowed_parameters: tuple[str, ...]
    required_parameters: tuple[str, ...] = ()
    transport: str = "https"
    wire_content_type: str = "application/json"
    wire_content_encoding: str = "identity"
    tool_schema_digest: str = ""
    tool_artifact_digest: str = ""
    retry_policy_digest: str = ""

    def __post_init__(self) -> None:
        for name in (
            "tool_id",
            "capability",
            "destination",
            "operation",
            "credential_audience",
            "peer_identity",
        ):
            if not getattr(self, name):
                raise ValueError(f"ToolRoute.{name} must be non-empty")

        missing = set(self.required_parameters) - set(self.allowed_parameters)
        if missing:
            raise ValueError(
                f"required parameters not in allowlist: {sorted(missing)}"
            )


class HttpDispatchPlanner:
    """Plans HTTPS dispatches from a static route table."""

    def __init__(
        self,
        routes: Mapping[str, ToolRoute],
        *,
        subject_principal: str,
        subject_key_id: str,
    ) -> None:
        self._routes = dict(routes)
        self._subject_principal = subject_principal
        self._subject_key_id = subject_key_id

    def plan(self, action: ActionIntent) -> PlannedDispatch:
        route = self._routes.get(action.tool_name)

        if route is None:
            raise DispatchPlanningError(
                f"no route configured for tool {action.tool_name!r}"
            )

        if route.capability != action.capability:
            raise DispatchPlanningError(
                "route capability does not match the governed capability"
            )

        arguments = self._filter_arguments(action, route)

        action_document = {
            "subject_principal": self._subject_principal,
            "subject_key_id": self._subject_key_id,
            "action": action.capability,
            "resource": route.destination,
            "parameters": arguments,
        }

        wire_bytes = _canonical_bytes(arguments)
        peer_identity_bytes = route.peer_identity.encode("utf-8")

        dispatch = {
            "transport": route.transport,
            "destination": route.destination,
            "operation": route.operation,
            "wire_content_type": route.wire_content_type,
            "wire_content_encoding": route.wire_content_encoding,
            "tool_id": route.tool_id,
            "tool_schema_digest": route.tool_schema_digest
            or _digest(route.tool_id.encode("utf-8")),
            "tool_artifact_digest": route.tool_artifact_digest
            or _digest(route.operation.encode("utf-8")),
            "credential_audience": route.credential_audience,
            "idempotency_key_digest": _digest(
                action.idempotency_key.encode("utf-8")
            ),
            "retry_policy_digest": route.retry_policy_digest
            or _digest(b"no-retry"),
        }

        return PlannedDispatch(
            action=action_document,
            prepared=PreparedDispatch.capture(
                request_id=action.request_id,
                dispatch=dispatch,
                wire_bytes=wire_bytes,
                peer_identity_bytes=peer_identity_bytes,
            ),
        )

    def _filter_arguments(
        self,
        action: ActionIntent,
        route: ToolRoute,
    ) -> dict[str, Any]:
        """Drop anything not on the allowlist; require what must be present.

        An argument the operator did not authorise never reaches the wire,
        even if the model produced it and policy allowed the capability.
        """
        raw = action.arguments

        if not isinstance(raw, Mapping):
            raise DispatchPlanningError("action arguments must be a mapping")

        allowed = set(route.allowed_parameters)
        filtered = {
            key: value for key, value in raw.items() if key in allowed
        }

        missing = set(route.required_parameters) - set(filtered)
        if missing:
            raise DispatchPlanningError(
                f"missing required parameters: {sorted(missing)}"
            )

        for key, value in filtered.items():
            if not _is_json_safe(value):
                raise DispatchPlanningError(
                    f"parameter {key!r} is not deterministically encodable"
                )

        return filtered


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    """Deterministic JSON encoding. The permit binds exactly these bytes."""
    try:
        return json.dumps(
            dict(payload),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DispatchPlanningError(
            "arguments are not canonically encodable"
        ) from exc


def _is_json_safe(value: object) -> bool:
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        # NaN and infinity have no canonical JSON form
        return value == value and value not in (float("inf"), float("-inf"))
    if isinstance(value, str):
        return True
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _is_json_safe(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(_is_json_safe(item) for item in value)
    return False


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


__all__ = [
    "DispatchPlanningError",
    "HttpDispatchPlanner",
    "ToolRoute",
]
