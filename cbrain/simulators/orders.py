"""Mutable purchase-order and shipment target for governed-execution scenarios."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from threading import RLock
from typing import Any

from .contracts import (
    EffectReceipt,
    JsonObject,
    SimulatorBusinessError,
    SimulatorConflict,
    SimulatorContractError,
    SimulatorNotFound,
    StoredEffect,
    canonical_object,
    capture_request,
    effect_identifier,
    exact_fields,
    required_text,
    restore_object,
)

PROCUREMENT_PO_CREATE = "procurement.po.create"
PROCUREMENT_PO_READ = "procurement.po.read"
PROCUREMENT_PO_AMEND = "procurement.po.amend"
PROCUREMENT_PO_CANCEL = "procurement.po.cancel"
PROCUREMENT_SHIPMENT_READ = "procurement.shipment.read"
ORDER_CAPABILITIES = frozenset(
    {
        PROCUREMENT_PO_CREATE,
        PROCUREMENT_PO_READ,
        PROCUREMENT_PO_AMEND,
        PROCUREMENT_PO_CANCEL,
        PROCUREMENT_SHIPMENT_READ,
    }
)
PAYMENT_TERMS = frozenset({"ADVANCE", "NET_15", "NET_30", "NET_45", "LC_SIGHT"})
SHIPMENT_STATUSES = frozenset(
    {
        "PENDING",
        "DISPATCHED",
        "IN_TRANSIT",
        "DELIVERED",
        "DELAYED",
        "SHORT_DELIVERED",
    }
)
DEFAULT_CLOCK_STAMP = "2026-03-01T09:00:00Z"


@dataclass(frozen=True, slots=True)
class PurchaseOrder:
    po_number: str
    vendor_id: str
    quote_id: str
    tonnes: Decimal
    rate_per_tonne: Decimal
    payment_terms: str
    comparison_digest: str
    status: str
    created_at: str

    def to_payload(self) -> JsonObject:
        return {
            "po_number": self.po_number,
            "vendor_id": self.vendor_id,
            "quote_id": self.quote_id,
            "tonnes": _money_text(self.tonnes),
            "rate_per_tonne": _money_text(self.rate_per_tonne),
            "payment_terms": self.payment_terms,
            "comparison_digest": self.comparison_digest,
            "status": self.status,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class Shipment:
    shipment_id: str
    po_number: str
    status: str

    def to_payload(self) -> JsonObject:
        return {
            "shipment_id": self.shipment_id,
            "po_number": self.po_number,
            "status": self.status,
        }


class OrderSimulator:
    """Thread-safe purchase-order book with ordinary delivery-state rules."""

    domain = "procurement"

    def __init__(self, *, clock: Callable[[], str] | None = None) -> None:
        self._lock = RLock()
        self._clock = clock if clock is not None else _default_clock
        self._orders: dict[str, PurchaseOrder] = {}
        self._shipments: dict[str, Shipment] = {}
        self._state_version = 0
        self._effects: dict[str, StoredEffect] = {}

    @property
    def state_version(self) -> int:
        with self._lock:
            return self._state_version

    def snapshot(self) -> JsonObject:
        with self._lock:
            value = {
                "domain": self.domain,
                "state_version": self._state_version,
                "purchase_orders": [
                    self._orders[key].to_payload() for key in sorted(self._orders)
                ],
                "shipments": [
                    self._shipments[key].to_payload() for key in sorted(self._shipments)
                ],
            }
            return restore_object(
                canonical_object(value, "order snapshot"), "order snapshot"
            )

    def set_shipment_status(self, po_number: str, status: str) -> None:
        required_text(po_number, "po_number")
        required_text(status, "status")
        if status not in SHIPMENT_STATUSES:
            raise SimulatorContractError("shipment status is not supported")
        with self._lock:
            shipment = self._shipments.get(po_number)
            if shipment is None:
                raise SimulatorNotFound("shipment does not exist")
            self._shipments[po_number] = Shipment(
                shipment_id=shipment.shipment_id,
                po_number=shipment.po_number,
                status=status,
            )
            self._state_version += 1

    def execute(
        self,
        capability: str,
        *,
        request_id: str,
        idempotency_key: str,
        arguments: Mapping[str, Any],
    ) -> EffectReceipt:
        if capability not in ORDER_CAPABILITIES:
            raise SimulatorContractError("unknown order capability")
        captured, request_digest = capture_request(
            capability=capability,
            request_id=request_id,
            idempotency_key=idempotency_key,
            arguments=arguments,
        )
        with self._lock:
            stored = self._effects.get(idempotency_key)
            if stored is not None:
                if stored.request_digest != request_digest:
                    raise SimulatorConflict(
                        "idempotency key was already used for another request"
                    )
                return stored.receipt

            result, mutated = self._apply(capability, captured)
            if mutated:
                self._state_version += 1
            receipt = EffectReceipt.capture(
                domain=self.domain,
                capability=capability,
                request_id=request_id,
                idempotency_key=idempotency_key,
                effect_id=effect_identifier(
                    domain=self.domain,
                    capability=capability,
                    idempotency_key=idempotency_key,
                    request_digest=request_digest,
                ),
                state_version=self._state_version,
                mutated=mutated,
                result=result,
            )
            self._effects[idempotency_key] = StoredEffect(request_digest, receipt)
            return receipt

    def _apply(self, capability: str, arguments: JsonObject) -> tuple[JsonObject, bool]:
        if capability == PROCUREMENT_PO_CREATE:
            return self._create(arguments), True
        if capability == PROCUREMENT_PO_READ:
            return self._read(arguments), False
        if capability == PROCUREMENT_PO_AMEND:
            return self._amend(arguments), True
        if capability == PROCUREMENT_PO_CANCEL:
            return self._cancel(arguments), True
        if capability == PROCUREMENT_SHIPMENT_READ:
            return self._read_shipment(arguments), False
        raise SimulatorContractError("unknown order capability")

    def _create(self, arguments: JsonObject) -> JsonObject:
        exact_fields(
            arguments,
            frozenset(
                {
                    "vendor_id",
                    "quote_id",
                    "tonnes",
                    "rate_per_tonne",
                    "payment_terms",
                    "comparison_digest",
                }
            ),
            "arguments",
        )
        payment_terms = required_text(
            arguments["payment_terms"], "arguments.payment_terms"
        )
        if payment_terms not in PAYMENT_TERMS:
            raise SimulatorContractError("payment_terms are not supported")
        created_at = self._clock()
        po_number = f"PO-{created_at[:4]}-{len(self._orders) + 1:04d}"
        order = PurchaseOrder(
            po_number=po_number,
            vendor_id=required_text(arguments["vendor_id"], "arguments.vendor_id"),
            quote_id=required_text(arguments["quote_id"], "arguments.quote_id"),
            tonnes=_positive_money(arguments["tonnes"], "arguments.tonnes"),
            rate_per_tonne=_positive_money(
                arguments["rate_per_tonne"], "arguments.rate_per_tonne"
            ),
            payment_terms=payment_terms,
            comparison_digest=required_text(
                arguments["comparison_digest"], "arguments.comparison_digest"
            ),
            status="OPEN",
            created_at=created_at,
        )
        self._orders[po_number] = order
        self._shipments[po_number] = Shipment(
            shipment_id=f"shp-{po_number}",
            po_number=po_number,
            status="PENDING",
        )
        return order.to_payload()

    def _read(self, arguments: JsonObject) -> JsonObject:
        exact_fields(arguments, frozenset({"po_number"}), "arguments")
        return self._required_order(
            required_text(arguments["po_number"], "arguments.po_number")
        ).to_payload()

    def _amend(self, arguments: JsonObject) -> JsonObject:
        exact_fields(
            arguments,
            frozenset({"po_number", "tonnes", "rate_per_tonne"}),
            "arguments",
        )
        order = self._required_order(
            required_text(arguments["po_number"], "arguments.po_number")
        )
        if order.status == "CANCELLED":
            raise SimulatorBusinessError("cancelled purchase orders cannot be amended")
        updated = PurchaseOrder(
            po_number=order.po_number,
            vendor_id=order.vendor_id,
            quote_id=order.quote_id,
            tonnes=_positive_money(arguments["tonnes"], "arguments.tonnes"),
            rate_per_tonne=_positive_money(
                arguments["rate_per_tonne"], "arguments.rate_per_tonne"
            ),
            payment_terms=order.payment_terms,
            comparison_digest=order.comparison_digest,
            status="AMENDED",
            created_at=order.created_at,
        )
        self._orders[order.po_number] = updated
        return updated.to_payload()

    def _cancel(self, arguments: JsonObject) -> JsonObject:
        exact_fields(arguments, frozenset({"po_number"}), "arguments")
        order = self._required_order(
            required_text(arguments["po_number"], "arguments.po_number")
        )
        if order.status == "CANCELLED":
            raise SimulatorBusinessError("purchase order is already cancelled")
        shipment = self._shipments[order.po_number]
        if shipment.status == "DELIVERED":
            raise SimulatorBusinessError(
                "delivered purchase orders cannot be cancelled"
            )
        cancelled = PurchaseOrder(
            po_number=order.po_number,
            vendor_id=order.vendor_id,
            quote_id=order.quote_id,
            tonnes=order.tonnes,
            rate_per_tonne=order.rate_per_tonne,
            payment_terms=order.payment_terms,
            comparison_digest=order.comparison_digest,
            status="CANCELLED",
            created_at=order.created_at,
        )
        self._orders[order.po_number] = cancelled
        return cancelled.to_payload()

    def _read_shipment(self, arguments: JsonObject) -> JsonObject:
        exact_fields(arguments, frozenset({"po_number"}), "arguments")
        po_number = required_text(arguments["po_number"], "arguments.po_number")
        shipment = self._shipments.get(po_number)
        if shipment is None:
            raise SimulatorNotFound("shipment does not exist")
        return shipment.to_payload()

    def _required_order(self, po_number: str) -> PurchaseOrder:
        order = self._orders.get(po_number)
        if order is None:
            raise SimulatorNotFound("purchase order does not exist")
        return order


def _default_clock() -> str:
    return DEFAULT_CLOCK_STAMP


def _positive_money(value: object, path: str) -> Decimal:
    if isinstance(value, (bool, float)) or not isinstance(value, (str, int, Decimal)):
        raise SimulatorContractError(f"{path} must be a decimal amount")
    try:
        amount = value if isinstance(value, Decimal) else Decimal(str(value))
    except InvalidOperation as exc:
        raise SimulatorContractError(f"{path} must be a decimal amount") from exc
    exponent = amount.as_tuple().exponent
    if (
        not amount.is_finite()
        or amount <= 0
        or not isinstance(exponent, int)
        or exponent < -2
    ):
        raise SimulatorContractError(
            f"{path} must be positive with at most two decimal places"
        )
    return amount.quantize(Decimal("0.01"))


def _money_text(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.01")), "f")


__all__ = [
    "ORDER_CAPABILITIES",
    "PROCUREMENT_PO_AMEND",
    "PROCUREMENT_PO_CANCEL",
    "PROCUREMENT_PO_CREATE",
    "PROCUREMENT_PO_READ",
    "PROCUREMENT_SHIPMENT_READ",
    "OrderSimulator",
    "PurchaseOrder",
    "Shipment",
]
