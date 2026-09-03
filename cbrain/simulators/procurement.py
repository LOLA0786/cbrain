"""Mutable steel-vendor and quote target for governed-execution scenarios."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from threading import RLock
from types import MappingProxyType
from typing import Any

from .catalog import vendor_catalog_bytes
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
    sha256_digest,
)

PROCUREMENT_VENDOR_LIST = "procurement.vendor.list"
PROCUREMENT_VENDOR_RELIABILITY_READ = "procurement.vendor.reliability.read"
PROCUREMENT_QUOTE_REQUEST = "procurement.quote.request"
PROCUREMENT_QUOTE_READ = "procurement.quote.read"
PROCUREMENT_CAPABILITIES = frozenset(
    {
        PROCUREMENT_VENDOR_LIST,
        PROCUREMENT_VENDOR_RELIABILITY_READ,
        PROCUREMENT_QUOTE_REQUEST,
        PROCUREMENT_QUOTE_READ,
    }
)
GRADES = frozenset({"Fe500D", "Fe550D", "HRC"})
PAYMENT_TERMS = frozenset({"ADVANCE", "NET_15", "NET_30", "NET_45", "LC_SIGHT"})
DEFAULT_CLOCK_STAMP = "2026-03-01T09:00:00Z"
_CLOCK_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass(frozen=True, slots=True)
class ReliabilityRecord:
    orders_total: int
    on_time_pct: Decimal
    short_delivery_pct: Decimal
    quality_rejection_pct: Decimal
    avg_delay_days: Decimal
    last_dispute_at: str | None

    def to_payload(self, vendor_id: str) -> JsonObject:
        return {
            "vendor_id": vendor_id,
            "orders_total": self.orders_total,
            "on_time_pct": _decimal_text(self.on_time_pct),
            "short_delivery_pct": _decimal_text(self.short_delivery_pct),
            "quality_rejection_pct": _decimal_text(self.quality_rejection_pct),
            "avg_delay_days": _decimal_text(self.avg_delay_days),
            "last_dispute_at": self.last_dispute_at,
        }


@dataclass(frozen=True, slots=True)
class Vendor:
    vendor_id: str
    legal_name: str
    gstin: str
    origin_city: str
    approved: bool
    onboarded_at: str
    bank_beneficiary_id: str | None
    moq_tonnes: Decimal
    freight_per_tonne: Decimal
    rates: Mapping[str, Decimal]
    default_payment_terms: str
    reliability: ReliabilityRecord

    def public_payload(self) -> JsonObject:
        return {
            "vendor_id": self.vendor_id,
            "legal_name": self.legal_name,
            "gstin": self.gstin,
            "origin_city": self.origin_city,
            "approved": self.approved,
            "onboarded_at": self.onboarded_at,
            "bank_beneficiary_id": self.bank_beneficiary_id,
        }


@dataclass(frozen=True, slots=True)
class Quote:
    quote_id: str
    vendor_id: str
    grade: str
    tonnes: Decimal
    rate_per_tonne: Decimal
    freight_per_tonne: Decimal
    gst_rate: Decimal
    payment_terms: str
    quoted_at: str
    valid_until: str
    quote_digest: str

    def to_payload(self, *, now: str) -> JsonObject:
        return {
            "quote_id": self.quote_id,
            "vendor_id": self.vendor_id,
            "grade": self.grade,
            "tonnes": _money_text(self.tonnes),
            "rate_per_tonne": _money_text(self.rate_per_tonne),
            "freight_per_tonne": _money_text(self.freight_per_tonne),
            "gst_rate": _decimal_text(self.gst_rate),
            "payment_terms": self.payment_terms,
            "quoted_at": self.quoted_at,
            "valid_until": self.valid_until,
            "quote_digest": self.quote_digest,
            "status": "EXPIRED" if now > self.valid_until else "OPEN",
        }


class ProcurementSimulator:
    """Thread-safe steel catalog whose ordinary business rules exclude policy."""

    domain = "procurement"

    def __init__(
        self,
        vendors: Sequence[Vendor],
        *,
        gst_rate: Decimal,
        quote_validity_hours: int,
        clock: Callable[[], str] | None = None,
    ) -> None:
        if quote_validity_hours <= 0:
            raise SimulatorContractError("quote_validity_hours must be positive")
        self._lock = RLock()
        self._vendors: dict[str, Vendor] = {}
        for vendor in vendors:
            if vendor.vendor_id in self._vendors:
                raise SimulatorConflict("duplicate vendor_id")
            self._vendors[vendor.vendor_id] = vendor
        if len(self._vendors) != 8:
            raise SimulatorContractError("catalog must contain exactly eight vendors")
        self._gst_rate = gst_rate
        self._quote_validity_hours = quote_validity_hours
        self._clock = clock if clock is not None else _default_clock
        self._quotes: dict[str, Quote] = {}
        self._state_version = 0
        self._effects: dict[str, StoredEffect] = {}

    @classmethod
    def from_catalog(
        cls,
        *,
        clock: Callable[[], str] | None = None,
        catalog: bytes | None = None,
    ) -> ProcurementSimulator:
        loaded = _load_catalog(
            catalog if catalog is not None else vendor_catalog_bytes()
        )
        return cls(
            loaded.vendors,
            gst_rate=loaded.gst_rate,
            quote_validity_hours=loaded.quote_validity_hours,
            clock=clock,
        )

    @property
    def state_version(self) -> int:
        with self._lock:
            return self._state_version

    def snapshot(self) -> JsonObject:
        with self._lock:
            now = self._now()
            value = {
                "domain": self.domain,
                "state_version": self._state_version,
                "vendors": [
                    self._vendors[key].public_payload() for key in sorted(self._vendors)
                ],
                "quotes": [
                    self._quotes[key].to_payload(now=now)
                    for key in sorted(self._quotes)
                ],
            }
            return restore_object(
                canonical_object(value, "procurement snapshot"),
                "procurement snapshot",
            )

    def vendor_rate(self, vendor_id: str, grade: str) -> Decimal:
        with self._lock:
            vendor = self._required_vendor(vendor_id)
            rate = vendor.rates.get(grade)
            if rate is None:
                raise SimulatorContractError("grade is not quoted by vendor")
            return rate

    def quote_rate(self, quote_id: str) -> Decimal:
        with self._lock:
            quote = self._quotes.get(quote_id)
            if quote is None:
                raise SimulatorNotFound("quote does not exist")
            return quote.rate_per_tonne

    def execute(
        self,
        capability: str,
        *,
        request_id: str,
        idempotency_key: str,
        arguments: Mapping[str, Any],
    ) -> EffectReceipt:
        if capability not in PROCUREMENT_CAPABILITIES:
            raise SimulatorContractError("unknown procurement capability")
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
        if capability == PROCUREMENT_VENDOR_LIST:
            return self._list_vendors(arguments), False
        if capability == PROCUREMENT_VENDOR_RELIABILITY_READ:
            return self._reliability(arguments), False
        if capability == PROCUREMENT_QUOTE_REQUEST:
            return self._request_quote(arguments), True
        if capability == PROCUREMENT_QUOTE_READ:
            return self._read_quote(arguments), False
        raise SimulatorContractError("unknown procurement capability")

    def _list_vendors(self, arguments: JsonObject) -> JsonObject:
        exact_fields(arguments, frozenset(), "arguments")
        return {
            "vendors": [
                self._vendors[key].public_payload() for key in sorted(self._vendors)
            ]
        }

    def _reliability(self, arguments: JsonObject) -> JsonObject:
        exact_fields(arguments, frozenset({"vendor_id"}), "arguments")
        vendor = self._required_vendor(
            required_text(arguments["vendor_id"], "arguments.vendor_id")
        )
        return vendor.reliability.to_payload(vendor.vendor_id)

    def _request_quote(self, arguments: JsonObject) -> JsonObject:
        exact_fields(
            arguments,
            frozenset({"vendor_id", "grade", "tonnes", "payment_terms"}),
            "arguments",
        )
        vendor = self._required_vendor(
            required_text(arguments["vendor_id"], "arguments.vendor_id")
        )
        grade = required_text(arguments["grade"], "arguments.grade")
        if grade not in GRADES:
            raise SimulatorContractError("grade is not supported")
        rate = vendor.rates.get(grade)
        if rate is None:
            raise SimulatorContractError("grade is not quoted by vendor")
        payment_terms = required_text(
            arguments["payment_terms"], "arguments.payment_terms"
        )
        if payment_terms not in PAYMENT_TERMS:
            raise SimulatorContractError("payment_terms are not supported")
        tonnes = _positive_money(arguments["tonnes"], "arguments.tonnes")
        if tonnes < vendor.moq_tonnes:
            raise SimulatorBusinessError("tonnage is below vendor MOQ")
        quoted_at = self._now()
        valid_until = _add_hours(quoted_at, self._quote_validity_hours)
        quote_id = f"quote-{len(self._quotes) + 1:04d}"
        digest_material = {
            "freight_per_tonne": _money_text(vendor.freight_per_tonne),
            "grade": grade,
            "gst_rate": _decimal_text(self._gst_rate),
            "payment_terms": payment_terms,
            "quote_id": quote_id,
            "quoted_at": quoted_at,
            "rate_per_tonne": _money_text(rate),
            "tonnes": _money_text(tonnes),
            "valid_until": valid_until,
            "vendor_id": vendor.vendor_id,
        }
        quote = Quote(
            quote_id=quote_id,
            vendor_id=vendor.vendor_id,
            grade=grade,
            tonnes=tonnes,
            rate_per_tonne=rate,
            freight_per_tonne=vendor.freight_per_tonne,
            gst_rate=self._gst_rate,
            payment_terms=payment_terms,
            quoted_at=quoted_at,
            valid_until=valid_until,
            quote_digest=sha256_digest(canonical_object(digest_material, "quote")),
        )
        self._quotes[quote.quote_id] = quote
        return quote.to_payload(now=quoted_at)

    def _read_quote(self, arguments: JsonObject) -> JsonObject:
        exact_fields(arguments, frozenset({"quote_id"}), "arguments")
        quote_id = required_text(arguments["quote_id"], "arguments.quote_id")
        quote = self._quotes.get(quote_id)
        if quote is None:
            raise SimulatorNotFound("quote does not exist")
        return quote.to_payload(now=self._now())

    def _required_vendor(self, vendor_id: str) -> Vendor:
        vendor = self._vendors.get(vendor_id)
        if vendor is None:
            raise SimulatorNotFound("vendor does not exist")
        return vendor

    def _now(self) -> str:
        stamp = self._clock()
        _parse_clock(stamp)
        return stamp


def _default_clock() -> str:
    return DEFAULT_CLOCK_STAMP


def _parse_clock(stamp: str) -> datetime:
    if not isinstance(stamp, str) or not stamp.strip():
        raise SimulatorContractError("clock must return UTC Zulu text")
    try:
        return datetime.strptime(stamp, _CLOCK_FORMAT).replace(tzinfo=UTC)
    except ValueError as exc:
        raise SimulatorContractError("clock must return UTC Zulu text") from exc


def _add_hours(stamp: str, hours: int) -> str:
    moment = _parse_clock(stamp)
    return (moment + timedelta(hours=hours)).strftime(_CLOCK_FORMAT)


@dataclass(frozen=True, slots=True)
class _LoadedCatalog:
    vendors: tuple[Vendor, ...]
    gst_rate: Decimal
    quote_validity_hours: int


def _load_catalog(raw: bytes) -> _LoadedCatalog:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SimulatorContractError("vendor catalog is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise SimulatorContractError("vendor catalog must be an object")
    _reject_floats(payload, "catalog")
    exact_fields(
        payload,
        frozenset(
            {
                "quote_validity_hours",
                "gst_rate",
                "grades",
                "payment_terms",
                "vendors",
            }
        ),
        "catalog",
    )
    grades = payload["grades"]
    terms = payload["payment_terms"]
    if grades != ["Fe500D", "Fe550D", "HRC"]:
        raise SimulatorContractError("catalog grades are invalid")
    if terms != ["ADVANCE", "NET_15", "NET_30", "NET_45", "LC_SIGHT"]:
        raise SimulatorContractError("catalog payment_terms are invalid")
    if not isinstance(payload["quote_validity_hours"], int) or isinstance(
        payload["quote_validity_hours"], bool
    ):
        raise SimulatorContractError("quote_validity_hours must be an integer")
    vendors_payload = payload["vendors"]
    if not isinstance(vendors_payload, list):
        raise SimulatorContractError("catalog vendors must be a list")
    vendors = tuple(
        _vendor_from_payload(item, index) for index, item in enumerate(vendors_payload)
    )
    gst_rate = _non_negative_decimal(payload["gst_rate"], "catalog.gst_rate", places=2)
    return _LoadedCatalog(
        vendors=vendors,
        gst_rate=gst_rate,
        quote_validity_hours=payload["quote_validity_hours"],
    )


def _vendor_from_payload(value: object, index: int) -> Vendor:
    path = f"catalog.vendors[{index}]"
    if not isinstance(value, Mapping):
        raise SimulatorContractError(f"{path} must be an object")
    exact_fields(
        value,
        frozenset(
            {
                "vendor_id",
                "legal_name",
                "gstin",
                "origin_city",
                "approved",
                "onboarded_at",
                "bank_beneficiary_id",
                "moq_tonnes",
                "freight_per_tonne",
                "rates",
                "default_payment_terms",
                "reliability",
            }
        ),
        path,
    )
    if not isinstance(value["approved"], bool):
        raise SimulatorContractError(f"{path}.approved must be boolean")
    beneficiary = value["bank_beneficiary_id"]
    if beneficiary is not None:
        required_text(beneficiary, f"{path}.bank_beneficiary_id")
    rates_payload = value["rates"]
    if not isinstance(rates_payload, Mapping):
        raise SimulatorContractError(f"{path}.rates must be an object")
    rates: dict[str, Decimal] = {}
    for grade in ("Fe500D", "Fe550D", "HRC"):
        if grade not in rates_payload:
            raise SimulatorContractError(f"{path}.rates.{grade} is required")
        rates[grade] = _positive_money(rates_payload[grade], f"{path}.rates.{grade}")
    if frozenset(rates_payload) != GRADES:
        raise SimulatorContractError(f"{path}.rates grades are invalid")
    terms = required_text(
        value["default_payment_terms"], f"{path}.default_payment_terms"
    )
    if terms not in PAYMENT_TERMS:
        raise SimulatorContractError(f"{path}.default_payment_terms are invalid")
    return Vendor(
        vendor_id=required_text(value["vendor_id"], f"{path}.vendor_id"),
        legal_name=required_text(value["legal_name"], f"{path}.legal_name"),
        gstin=required_text(value["gstin"], f"{path}.gstin"),
        origin_city=required_text(value["origin_city"], f"{path}.origin_city"),
        approved=value["approved"],
        onboarded_at=required_text(value["onboarded_at"], f"{path}.onboarded_at"),
        bank_beneficiary_id=beneficiary,
        moq_tonnes=_positive_money(value["moq_tonnes"], f"{path}.moq_tonnes"),
        freight_per_tonne=_positive_money(
            value["freight_per_tonne"], f"{path}.freight_per_tonne"
        ),
        rates=MappingProxyType(rates),
        default_payment_terms=terms,
        reliability=_reliability_from_payload(
            value["reliability"], f"{path}.reliability"
        ),
    )


def _reliability_from_payload(value: object, path: str) -> ReliabilityRecord:
    if not isinstance(value, Mapping):
        raise SimulatorContractError(f"{path} must be an object")
    exact_fields(
        value,
        frozenset(
            {
                "orders_total",
                "on_time_pct",
                "short_delivery_pct",
                "quality_rejection_pct",
                "avg_delay_days",
                "last_dispute_at",
            }
        ),
        path,
    )
    orders_total = value["orders_total"]
    if (
        not isinstance(orders_total, int)
        or isinstance(orders_total, bool)
        or orders_total < 0
    ):
        raise SimulatorContractError(
            f"{path}.orders_total must be a non-negative integer"
        )
    last_dispute = value["last_dispute_at"]
    if last_dispute is not None:
        required_text(last_dispute, f"{path}.last_dispute_at")
    return ReliabilityRecord(
        orders_total=orders_total,
        on_time_pct=_non_negative_decimal(
            value["on_time_pct"], f"{path}.on_time_pct", places=2
        ),
        short_delivery_pct=_non_negative_decimal(
            value["short_delivery_pct"], f"{path}.short_delivery_pct", places=2
        ),
        quality_rejection_pct=_non_negative_decimal(
            value["quality_rejection_pct"],
            f"{path}.quality_rejection_pct",
            places=2,
        ),
        avg_delay_days=_non_negative_decimal(
            value["avg_delay_days"], f"{path}.avg_delay_days", places=2
        ),
        last_dispute_at=last_dispute,
    )


def _reject_floats(value: object, path: str) -> None:
    if isinstance(value, float):
        raise SimulatorContractError(f"{path} must not use floating-point numbers")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_floats(child, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _reject_floats(child, f"{path}[{index}]")


def _positive_money(value: object, path: str) -> Decimal:
    amount = _decimal_amount(value, path, places=2)
    if amount <= 0:
        raise SimulatorContractError(f"{path} must be a positive decimal amount")
    return amount


def _non_negative_decimal(value: object, path: str, *, places: int) -> Decimal:
    amount = _decimal_amount(value, path, places=places)
    if amount < 0:
        raise SimulatorContractError(f"{path} must be a non-negative decimal")
    return amount


def _decimal_amount(value: object, path: str, *, places: int) -> Decimal:
    if isinstance(value, (bool, float)):
        raise SimulatorContractError(f"{path} must be a decimal amount")
    if not isinstance(value, (str, int, Decimal)):
        raise SimulatorContractError(f"{path} must be a decimal amount")
    try:
        amount = value if isinstance(value, Decimal) else Decimal(str(value))
    except InvalidOperation as exc:
        raise SimulatorContractError(f"{path} must be a decimal amount") from exc
    exponent = amount.as_tuple().exponent
    if not amount.is_finite() or not isinstance(exponent, int) or exponent < -places:
        raise SimulatorContractError(
            f"{path} must be a finite decimal with at most {places} decimal places"
        )
    return amount.quantize(Decimal("1").scaleb(-places))


def _money_text(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.01")), "f")


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


__all__ = [
    "PROCUREMENT_CAPABILITIES",
    "PROCUREMENT_QUOTE_READ",
    "PROCUREMENT_QUOTE_REQUEST",
    "PROCUREMENT_VENDOR_LIST",
    "PROCUREMENT_VENDOR_RELIABILITY_READ",
    "ProcurementSimulator",
    "Quote",
    "ReliabilityRecord",
    "Vendor",
]
