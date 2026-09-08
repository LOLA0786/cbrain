"""Trusted security-event evidence for PrivateVault authorize loop gating.

When ``PV_LOOP_EVENTS_REQUIRED`` / ``PV_SECURE_PROFILE`` is enabled, mint
refuses without ``security_events``. Those events must come from a
deployment-owned correlation source bound to the sealed decision and planned
action — never from model output or caller-authored placeholders.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from cbrain.adapters.privatevault import PrivateVaultDecision
from cbrain.contracts import ActionIntent
from cbrain.execution.gateway import PlannedDispatch

LOOP_EVENT_SPEC = "pv-agent-security-event/1.0"


class SecurityEventEvidenceError(RuntimeError):
    """Trusted security-event evidence cannot be produced."""


class SecurityEventEvidenceProvider(Protocol):
    """Resolve loop-discovery events for one authorize attempt."""

    def events_for_authorize(
        self,
        *,
        action: ActionIntent,
        decision: PrivateVaultDecision,
        planned: PlannedDispatch,
    ) -> Sequence[Mapping[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class SealedDecisionInvokeEvidence:
    """Emit one VERIFIED INVOKES edge bound to the sealed ALLOW.

    The ``authorization_id`` is the PrivateVault ``decision_id`` (not a mint
    permit — this edge records that the governed runtime is invoking under a
    sealed decision). ``action_digest`` is derived from the planned EA action
    bytes. Source is the deployment-owned runtime identity; target is the
    authenticated agent.
    """

    runtime_agent_id: str = "governed-runtime"
    clock: Any = None

    def events_for_authorize(
        self,
        *,
        action: ActionIntent,
        decision: PrivateVaultDecision,
        planned: PlannedDispatch,
    ) -> Sequence[Mapping[str, Any]]:
        record = decision.record
        decision_id = record.get("decision_id")
        if not isinstance(decision_id, str) or not decision_id.strip():
            raise SecurityEventEvidenceError(
                "sealed decision_id is required for security-event evidence"
            )
        if decision_id != record.get("decision_id"):
            raise SecurityEventEvidenceError("decision_id mismatch")

        action_digest = record.get("action_digest")
        if not isinstance(action_digest, str) or not action_digest.startswith(
            "sha256:"
        ):
            raise SecurityEventEvidenceError(
                "sealed action_digest is required for security-event evidence"
            )

        # The planned action must still match what was sealed; otherwise the
        # evidence would cover a different semantic action.
        if planned.action.get("action") != action.capability:
            raise SecurityEventEvidenceError(
                "planned action capability does not match the intent"
            )
        if dict(planned.action.get("parameters") or {}) != action.arguments:
            raise SecurityEventEvidenceError(
                "planned parameters do not match the intent"
            )

        occurred_at = self._now()
        return (
            {
                "spec": LOOP_EVENT_SPEC,
                "event_id": f"invoke-{action.request_id}",
                "trace_id": f"trace-{action.request_id}",
                "occurred_at": occurred_at,
                "source_agent_id": self.runtime_agent_id,
                "target_agent_id": action.agent_id,
                "relation": "INVOKES",
                "action_digest": action_digest,
                "authorization_id": decision_id,
                "authorization_state": "VERIFIED",
                "parent_event_id": None,
            },
        )

    def _now(self) -> str:
        if self.clock is not None:
            value = self.clock()
            if not isinstance(value, str) or not value.endswith("Z"):
                raise SecurityEventEvidenceError("clock must return RFC3339 UTC")
            return value
        return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "LOOP_EVENT_SPEC",
    "SealedDecisionInvokeEvidence",
    "SecurityEventEvidenceError",
    "SecurityEventEvidenceProvider",
]
