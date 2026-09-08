"""Business-idempotent purchase reconciliation.

HTTP 202 (or a dropped response after a possible write) is pending acceptance,
not confirmed completion. Recovery looks up the effect by business idempotency
key. Blind resubmission with a new key is refused.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, cast

from cbrain.simulators.contracts import EffectReceipt, SimulatorNotFound
from cbrain.simulators.orders import PROCUREMENT_PO_CREATE, PROCUREMENT_PO_READ


class PurchaseOutcome(StrEnum):
    CONFIRMED = "CONFIRMED"
    PENDING_ACCEPTANCE = "PENDING_ACCEPTANCE"
    INDETERMINATE = "INDETERMINATE"
    CONFLICT = "CONFLICT"
    NOT_FOUND = "NOT_FOUND"


@dataclass(frozen=True, slots=True)
class PurchaseAttemptResult:
    outcome: PurchaseOutcome
    idempotency_key: str
    http_status: int | None
    effect_id: str | None
    po_number: str | None
    detail: str


class PurchaseTarget(Protocol):
    def create_po(
        self,
        *,
        request_id: str,
        idempotency_key: str,
        arguments: Mapping[str, Any],
    ) -> tuple[int, EffectReceipt | None]:
        """Return (http_status, receipt). Receipt may be None on drop/202."""

    def read_po_by_idempotency(
        self,
        *,
        idempotency_key: str,
    ) -> EffectReceipt | None: ...


class OrderSimulatorPurchaseTarget:
    """Reference-target adapter over OrderSimulator (not a live ERP)."""

    def __init__(self, simulator: Any, *, pending_on_create: bool = False) -> None:
        self._simulator = simulator
        self._pending_on_create = pending_on_create
        self._effects_by_key: dict[str, EffectReceipt] = {}

    def create_po(
        self,
        *,
        request_id: str,
        idempotency_key: str,
        arguments: Mapping[str, Any],
    ) -> tuple[int, EffectReceipt | None]:
        receipt = self._simulator.execute(
            PROCUREMENT_PO_CREATE,
            request_id=request_id,
            idempotency_key=idempotency_key,
            arguments=arguments,
        )
        self._effects_by_key[idempotency_key] = receipt
        if self._pending_on_create:
            # Effect committed; transport reports acceptance pending.
            return 202, None
        return 200, receipt

    def read_po_by_idempotency(self, *, idempotency_key: str) -> EffectReceipt | None:
        stored = self._effects_by_key.get(idempotency_key)
        if stored is not None:
            return stored
        # Fall back to reading by po_number if the simulator exposes effects.
        effects = getattr(self._simulator, "_effects", {})
        item = effects.get(idempotency_key)
        if item is None:
            return None
        receipt = getattr(item, "receipt", None)
        return cast(EffectReceipt | None, receipt)


def interpret_create_response(
    *,
    http_status: int | None,
    receipt: EffectReceipt | None,
    idempotency_key: str,
    transport_error: bool = False,
) -> PurchaseAttemptResult:
    if transport_error:
        return PurchaseAttemptResult(
            outcome=PurchaseOutcome.INDETERMINATE,
            idempotency_key=idempotency_key,
            http_status=http_status,
            effect_id=None,
            po_number=None,
            detail="possible write without verified closure; do not blind resubmit",
        )
    if http_status == 202:
        return PurchaseAttemptResult(
            outcome=PurchaseOutcome.PENDING_ACCEPTANCE,
            idempotency_key=idempotency_key,
            http_status=202,
            effect_id=None if receipt is None else receipt.effect_id,
            po_number=None,
            detail="HTTP 202 is pending acceptance, not confirmed completion",
        )
    if http_status == 200 and receipt is not None:
        po_number = None
        payload = receipt.result
        if isinstance(payload, Mapping):
            value = payload.get("po_number")
            if isinstance(value, str):
                po_number = value
        return PurchaseAttemptResult(
            outcome=PurchaseOutcome.CONFIRMED,
            idempotency_key=idempotency_key,
            http_status=200,
            effect_id=receipt.effect_id,
            po_number=po_number,
            detail="create confirmed by target contract",
        )
    if http_status == 409:
        return PurchaseAttemptResult(
            outcome=PurchaseOutcome.CONFLICT,
            idempotency_key=idempotency_key,
            http_status=409,
            effect_id=None,
            po_number=None,
            detail="idempotency conflict; reconcile by key, do not mint a new key",
        )
    return PurchaseAttemptResult(
        outcome=PurchaseOutcome.INDETERMINATE,
        idempotency_key=idempotency_key,
        http_status=http_status,
        effect_id=None,
        po_number=None,
        detail="unverified create outcome",
    )


def reconcile_purchase(
    target: PurchaseTarget,
    *,
    idempotency_key: str,
) -> PurchaseAttemptResult:
    """Resolve an uncertain create without submitting a new business key."""
    receipt = target.read_po_by_idempotency(idempotency_key=idempotency_key)
    if receipt is None:
        return PurchaseAttemptResult(
            outcome=PurchaseOutcome.NOT_FOUND,
            idempotency_key=idempotency_key,
            http_status=None,
            effect_id=None,
            po_number=None,
            detail="no effect for idempotency key; operator escalation required",
        )
    po_number = None
    if isinstance(receipt.result, Mapping):
        value = receipt.result.get("po_number")
        if isinstance(value, str):
            po_number = value
    return PurchaseAttemptResult(
        outcome=PurchaseOutcome.CONFIRMED,
        idempotency_key=idempotency_key,
        http_status=200,
        effect_id=receipt.effect_id,
        po_number=po_number,
        detail="reconciled existing effect for the same business idempotency key",
    )


def refuse_blind_resubmit(*, previous_key: str, new_key: str) -> None:
    if previous_key != new_key:
        raise RuntimeError(
            "refusing blind resubmit with a new idempotency key; "
            f"reconcile previous_key={previous_key!r} instead of minting {new_key!r}"
        )


__all__ = [
    "OrderSimulatorPurchaseTarget",
    "PROCUREMENT_PO_CREATE",
    "PROCUREMENT_PO_READ",
    "PurchaseAttemptResult",
    "PurchaseOutcome",
    "PurchaseTarget",
    "interpret_create_response",
    "reconcile_purchase",
    "refuse_blind_resubmit",
    "SimulatorNotFound",
]
