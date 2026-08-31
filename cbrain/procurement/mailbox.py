"""Registered-vendor RFQ delivery and instant quotation ranking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from types import MappingProxyType

from .erp import ProcurementError, VendorRecord


@dataclass(frozen=True, slots=True)
class Quotation:
    quote_id: str
    rfq_id: str
    vendor_id: str
    material_id: str
    amount_minor: str
    currency: str

    def amount(self) -> Decimal:
        try:
            value = Decimal(self.amount_minor)
        except (InvalidOperation, ValueError) as exc:
            raise ProcurementError("quotation amount_minor is invalid") from exc
        if not value.is_finite() or value <= 0:
            raise ProcurementError(
                "quotation amount_minor must be a positive finite integer"
            )
        if value != value.to_integral_value():
            raise ProcurementError("quotation amount_minor must be integer minor units")
        return value


def require_registered_vendor(
    vendor_id: str, vendors: Mapping[str, VendorRecord]
) -> VendorRecord:
    vendor = vendors.get(vendor_id)
    if vendor is None:
        raise ProcurementError("unknown vendor")
    if not vendor.registered:
        raise ProcurementError("vendor is not registered")
    return vendor


def rank_quotations(quotes: Sequence[Quotation]) -> tuple[Quotation, ...]:
    return tuple(sorted(quotes, key=lambda item: (item.amount(), item.vendor_id)))


def quotation_board_payload(quotes: Sequence[Quotation]) -> Mapping[str, object]:
    ranked = rank_quotations(quotes)
    return MappingProxyType(
        {
            "instant": True,
            "count": len(ranked),
            "quotations": [
                {
                    "rank": index,
                    "quote_id": item.quote_id,
                    "rfq_id": item.rfq_id,
                    "vendor_id": item.vendor_id,
                    "material_id": item.material_id,
                    "amount_minor": item.amount_minor,
                    "currency": item.currency,
                }
                for index, item in enumerate(ranked, start=1)
            ],
        }
    )


__all__ = [
    "Quotation",
    "quotation_board_payload",
    "rank_quotations",
    "require_registered_vendor",
]
