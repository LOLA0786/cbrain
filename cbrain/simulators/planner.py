"""Concrete exact-byte dispatch planner for simulator capabilities."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from cbrain.contracts import ActionIntent
from cbrain.dispatch import PreparedDispatch
from cbrain.execution.gateway import PlannedDispatch

from .contracts import required_text, sha256_digest
from .http import encode_simulator_request

_SHA256_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


class SimulatorPlanningError(RuntimeError):
    """An intent cannot be mapped to an immutable simulator route."""


class CapabilityDescription(Protocol):
    @property
    def capability(self) -> str: ...

    @property
    def domain(self) -> str: ...

    @property
    def operation(self) -> str: ...

    @property
    def destination(self) -> str: ...

    @property
    def credential_audience(self) -> str: ...


class CapabilityCatalog(Protocol):
    @property
    def capabilities(self) -> Mapping[str, CapabilityDescription]:
        """Return immutable capability-to-route descriptions."""


@dataclass(frozen=True, slots=True)
class SimulatorTargetBinding:
    """Deployment-owned evidence for one configured simulator target."""

    peer_identity_bytes: bytes
    tool_artifact_digest: str
    retry_policy_digest: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.peer_identity_bytes, bytes)
            or not self.peer_identity_bytes
        ):
            raise ValueError("peer_identity_bytes must be non-empty bytes")
        _digest(self.tool_artifact_digest, "tool_artifact_digest")
        _digest(self.retry_policy_digest, "retry_policy_digest")


class SimulatorDispatchPlanner:
    """Translate known domain intents into sidecar-ready exact bytes.

    Routes and credentials come only from the immutable capability catalog.
    Deployment evidence supplies the pinned TLS peer and artifact digests.
    Nothing in ``ActionIntent.arguments`` can select a destination, credential,
    transport, operation, or tool artifact.
    """

    def __init__(
        self,
        *,
        catalog: CapabilityCatalog,
        targets: Mapping[str, SimulatorTargetBinding],
        tool_schema_digests: Mapping[str, str],
    ) -> None:
        target_map = dict(targets)
        schema_map = dict(tool_schema_digests)
        missing_targets = sorted(
            {
                spec.destination
                for spec in catalog.capabilities.values()
                if spec.destination not in target_map
            }
        )
        if missing_targets:
            raise ValueError(f"missing simulator targets: {missing_targets}")
        missing_schemas = sorted(set(catalog.capabilities) - set(schema_map))
        unexpected_schemas = sorted(set(schema_map) - set(catalog.capabilities))
        if missing_schemas or unexpected_schemas:
            raise ValueError(
                "tool schema coverage is invalid; "
                f"missing={missing_schemas}, unexpected={unexpected_schemas}"
            )
        for capability, digest in schema_map.items():
            _digest(digest, f"tool_schema_digests[{capability!r}]")
        self._catalog = catalog
        self._targets = target_map
        self._tool_schema_digests = schema_map

    def plan(self, action: ActionIntent) -> PlannedDispatch:
        try:
            spec = self._catalog.capabilities[action.capability]
        except KeyError as exc:
            raise SimulatorPlanningError(
                "capability is not in the domain catalog"
            ) from exc
        if action.tool_name != action.capability:
            raise SimulatorPlanningError("tool_name does not match capability")
        target = self._targets[spec.destination]
        wire_bytes = encode_simulator_request(
            request_id=action.request_id,
            idempotency_key=action.idempotency_key,
            arguments=action.arguments,
        )
        prepared = PreparedDispatch.capture(
            request_id=action.request_id,
            dispatch={
                "transport": "https",
                "destination": spec.destination,
                "operation": spec.operation,
                "wire_content_type": "application/json",
                "wire_content_encoding": "identity",
                "tool_id": f"{spec.capability}.v1",
                "tool_schema_digest": self._tool_schema_digests[spec.capability],
                "tool_artifact_digest": target.tool_artifact_digest,
                "credential_audience": spec.credential_audience,
                "idempotency_key_digest": sha256_digest(
                    action.idempotency_key.encode("utf-8")
                ),
                "retry_policy_digest": target.retry_policy_digest,
            },
            wire_bytes=wire_bytes,
            peer_identity_bytes=target.peer_identity_bytes,
        )
        action_document = {
            "subject_principal": action.agent_id,
            "action": action.capability,
            "resource": f"{spec.domain}:{spec.destination}",
            "parameters": action.arguments,
        }
        return PlannedDispatch(action=action_document, prepared=prepared)


def _digest(value: str, path: str) -> str:
    required_text(value, path)
    if _SHA256_DIGEST.fullmatch(value) is None:
        raise ValueError(f"{path} must be a sha256 digest")
    return value


__all__ = [
    "SimulatorDispatchPlanner",
    "SimulatorPlanningError",
    "SimulatorTargetBinding",
]
