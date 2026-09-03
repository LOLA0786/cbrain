"""Deterministic landed-cost comparison for steel quotes.

Pure arithmetic. No model, no I/O, and no authorization policy. Cost of capital
and scoring weights are deployment-owned inputs, never inferred from a quote.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

MONEY = Decimal("0.01")
SCORE = Decimal("0.00000001")
_DEFAULT_WEIGHT_COST = Decimal("0.7")
_DEFAULT_WEIGHT_RELIABILITY = Decimal("0.3")
_TERM_DAYS: Mapping[str, int] = {
    "ADVANCE": 0,
    "LC_SIGHT": 0,
    "NET_15": 15,
    "NET_30": 30,
    "NET_45": 45,
}


def _require_decimal(value: object, name: str) -> Decimal:
    if isinstance(value, (bool, float)) or not isinstance(value, Decimal):
        raise ValueError(f"{name} must be Decimal")
    if not value.is_finite():
        raise ValueError(f"{name} must be a finite Decimal")
    return value


def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _money_text(value: Decimal) -> str:
    return format(value.quantize(MONEY), "f")


def _score_text(value: Decimal) -> str:
    return format(value.quantize(SCORE), "f")


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


@dataclass(frozen=True, slots=True)
class SourcingPolicy:
    annual_cost_of_capital: Decimal
    input_credit_eligible: bool
    weight_cost: Decimal = _DEFAULT_WEIGHT_COST
    weight_reliability: Decimal = _DEFAULT_WEIGHT_RELIABILITY

    def __post_init__(self) -> None:
        _require_decimal(self.annual_cost_of_capital, "annual_cost_of_capital")
        _require_bool(self.input_credit_eligible, "input_credit_eligible")
        _require_decimal(self.weight_cost, "weight_cost")
        _require_decimal(self.weight_reliability, "weight_reliability")
        if self.annual_cost_of_capital < 0:
            raise ValueError("annual_cost_of_capital must be non-negative")
        if self.weight_cost < 0 or self.weight_reliability < 0:
            raise ValueError("sourcing weights must be non-negative")
        if self.weight_cost + self.weight_reliability != Decimal("1"):
            raise ValueError("sourcing weights must sum to 1")

    def to_payload(self) -> dict[str, Any]:
        return {
            "annual_cost_of_capital": _decimal_text(self.annual_cost_of_capital),
            "input_credit_eligible": self.input_credit_eligible,
            "weight_cost": _decimal_text(self.weight_cost),
            "weight_reliability": _decimal_text(self.weight_reliability),
        }

    def digest(self) -> str:
        return _digest(_canonical(self.to_payload(), "sourcing policy"))


@dataclass(frozen=True, slots=True)
class QuoteInputs:
    quote_id: str
    vendor_id: str
    grade: str
    tonnes: Decimal
    rate_per_tonne: Decimal
    freight_per_tonne: Decimal
    gst_rate: Decimal
    payment_terms: str
    quote_digest: str

    def __post_init__(self) -> None:
        for field_name in (
            "quote_id",
            "vendor_id",
            "grade",
            "payment_terms",
            "quote_digest",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be non-empty text")
        if self.payment_terms not in _TERM_DAYS:
            raise ValueError("payment_terms are not supported")
        for field_name in (
            "tonnes",
            "rate_per_tonne",
            "freight_per_tonne",
            "gst_rate",
        ):
            amount = _require_decimal(getattr(self, field_name), field_name)
            if amount < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.tonnes <= 0:
            raise ValueError("tonnes must be positive")


@dataclass(frozen=True, slots=True)
class ReliabilityInputs:
    vendor_id: str
    orders_total: int
    on_time_pct: Decimal
    short_delivery_pct: Decimal
    quality_rejection_pct: Decimal
    avg_delay_days: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.vendor_id, str) or not self.vendor_id.strip():
            raise ValueError("vendor_id must be non-empty text")
        if (
            not isinstance(self.orders_total, int)
            or isinstance(self.orders_total, bool)
            or self.orders_total < 0
        ):
            raise ValueError("orders_total must be a non-negative integer")
        for field_name in (
            "on_time_pct",
            "short_delivery_pct",
            "quality_rejection_pct",
            "avg_delay_days",
        ):
            amount = _require_decimal(getattr(self, field_name), field_name)
            if amount < 0:
                raise ValueError(f"{field_name} must be non-negative")


@dataclass(frozen=True, slots=True)
class LandedCostBreakdown:
    vendor_id: str
    quote_id: str
    rate_per_tonne: Decimal
    freight_per_tonne: Decimal
    gst_effect_per_tonne: Decimal
    credit_cost_per_tonne: Decimal
    landed_cost_per_tonne: Decimal
    normalized_landed_cost: Decimal
    reliability_index: Decimal
    score: Decimal
    quote_digest: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "credit_cost_per_tonne": _money_text(self.credit_cost_per_tonne),
            "freight_per_tonne": _money_text(self.freight_per_tonne),
            "gst_effect_per_tonne": _money_text(self.gst_effect_per_tonne),
            "landed_cost_per_tonne": _money_text(self.landed_cost_per_tonne),
            "normalized_landed_cost": _score_text(self.normalized_landed_cost),
            "quote_digest": self.quote_digest,
            "quote_id": self.quote_id,
            "rate_per_tonne": _money_text(self.rate_per_tonne),
            "reliability_index": _score_text(self.reliability_index),
            "score": _score_text(self.score),
            "vendor_id": self.vendor_id,
        }


@dataclass(frozen=True, slots=True)
class ComparisonReceipt:
    ranked: tuple[LandedCostBreakdown, ...]
    policy_digest: str
    quote_digests: tuple[str, ...]
    computed_at: str
    comparison_digest: str
    _payload_json: bytes

    @property
    def breakdowns(self) -> tuple[LandedCostBreakdown, ...]:
        return tuple(sorted(self.ranked, key=lambda item: item.vendor_id))

    @property
    def ranked_vendor_ids(self) -> tuple[str, ...]:
        return tuple(item.vendor_id for item in self.ranked)

    def to_payload(self) -> dict[str, Any]:
        restored = json.loads(self._payload_json)
        if not isinstance(restored, dict):
            raise ValueError("comparison receipt payload is invalid")
        return restored

    def to_canonical_bytes(self) -> bytes:
        return self._payload_json


def gst_effect(
    rate: Decimal,
    freight: Decimal,
    gst_rate: Decimal,
    input_credit_eligible: bool,
) -> Decimal:
    _require_decimal(rate, "rate")
    _require_decimal(freight, "freight")
    _require_decimal(gst_rate, "gst_rate")
    _require_bool(input_credit_eligible, "input_credit_eligible")
    if input_credit_eligible:
        return Decimal("0.00")
    return ((rate + freight) * gst_rate).quantize(MONEY)


def credit_cost(
    taxable: Decimal,
    payment_terms: str,
    annual_cost_of_capital: Decimal,
) -> Decimal:
    _require_decimal(taxable, "taxable")
    _require_decimal(annual_cost_of_capital, "annual_cost_of_capital")
    if payment_terms not in _TERM_DAYS:
        raise ValueError("payment_terms are not supported")
    days = Decimal(_TERM_DAYS[payment_terms])
    return (-(taxable * annual_cost_of_capital * days) / Decimal(365)).quantize(MONEY)


def reliability_index(on_time_pct: Decimal) -> Decimal:
    _require_decimal(on_time_pct, "on_time_pct")
    return ((Decimal("100") - on_time_pct) / Decimal("100")).quantize(SCORE)


@dataclass(frozen=True, slots=True)
class _Draft:
    quote: QuoteInputs
    gst: Decimal
    credit: Decimal
    landed: Decimal
    reliability_index: Decimal


def compare_quotes(
    quotes: Sequence[QuoteInputs],
    reliability: Sequence[ReliabilityInputs],
    policy: SourcingPolicy,
    *,
    computed_at: str,
) -> ComparisonReceipt:
    if not isinstance(policy, SourcingPolicy):
        raise ValueError("policy must be a SourcingPolicy")
    if not isinstance(computed_at, str) or not computed_at.strip():
        raise ValueError("computed_at must be non-empty text")
    if not quotes:
        raise ValueError("at least one quote is required")
    quote_list = tuple(quotes)
    reliability_by_vendor = _reliability_map(reliability)
    vendors = [item.vendor_id for item in quote_list]
    if len(set(vendors)) != len(vendors):
        raise ValueError("quotes must be unique per vendor")
    missing = [
        vendor_id for vendor_id in vendors if vendor_id not in reliability_by_vendor
    ]
    if missing:
        raise ValueError("reliability is missing for one or more vendors")

    draft: list[_Draft] = []
    for quote in quote_list:
        record = reliability_by_vendor[quote.vendor_id]
        taxable = quote.rate_per_tonne + quote.freight_per_tonne
        gst = gst_effect(
            quote.rate_per_tonne,
            quote.freight_per_tonne,
            quote.gst_rate,
            policy.input_credit_eligible,
        )
        credit = credit_cost(
            taxable, quote.payment_terms, policy.annual_cost_of_capital
        )
        landed = (taxable + gst + credit).quantize(MONEY)
        draft.append(
            _Draft(
                quote=quote,
                gst=gst,
                credit=credit,
                landed=landed,
                reliability_index=reliability_index(record.on_time_pct),
            )
        )

    peak = max(item.landed for item in draft)
    if peak <= 0:
        raise ValueError("landed cost must be positive")

    ranked_rows: list[LandedCostBreakdown] = []
    for item in draft:
        normalized = (item.landed / peak).quantize(SCORE)
        score = (
            policy.weight_cost * normalized
            + policy.weight_reliability * item.reliability_index
        ).quantize(SCORE)
        quote = item.quote
        ranked_rows.append(
            LandedCostBreakdown(
                vendor_id=quote.vendor_id,
                quote_id=quote.quote_id,
                rate_per_tonne=quote.rate_per_tonne.quantize(MONEY),
                freight_per_tonne=quote.freight_per_tonne.quantize(MONEY),
                gst_effect_per_tonne=item.gst,
                credit_cost_per_tonne=item.credit,
                landed_cost_per_tonne=item.landed,
                normalized_landed_cost=normalized,
                reliability_index=item.reliability_index,
                score=score,
                quote_digest=quote.quote_digest,
            )
        )

    ranked = tuple(sorted(ranked_rows, key=lambda row: (row.score, row.vendor_id)))
    policy_digest = policy.digest()
    quote_digests = tuple(sorted(quote.quote_digest for quote in quote_list))
    body = {
        "computed_at": computed_at,
        "policy": policy.to_payload(),
        "policy_digest": policy_digest,
        "quote_digests": list(quote_digests),
        "ranked": [
            {"rank": index, **row.to_payload()}
            for index, row in enumerate(ranked, start=1)
        ],
    }
    payload_without_digest = json.loads(
        _canonical({**body, "comparison_digest": ""}, "comparison").decode("utf-8")
    )
    if not isinstance(payload_without_digest, dict):
        raise ValueError("comparison payload is invalid")
    digest = _digest(
        _canonical({**payload_without_digest, "comparison_digest": ""}, "comparison")
    )
    payload = {**payload_without_digest, "comparison_digest": digest}
    encoded = _canonical(payload, "comparison receipt")
    return ComparisonReceipt(
        ranked=ranked,
        policy_digest=policy_digest,
        quote_digests=quote_digests,
        computed_at=computed_at,
        comparison_digest=digest,
        _payload_json=encoded,
    )


def _reliability_map(
    reliability: Sequence[ReliabilityInputs],
) -> dict[str, ReliabilityInputs]:
    mapped: dict[str, ReliabilityInputs] = {}
    for record in reliability:
        if not isinstance(record, ReliabilityInputs):
            raise ValueError("reliability records are invalid")
        if record.vendor_id in mapped:
            raise ValueError("reliability records must be unique per vendor")
        mapped[record.vendor_id] = record
    return mapped


def _canonical(value: Mapping[str, Any], path: str) -> bytes:
    try:
        return json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must contain finite JSON values") from exc


def _digest(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


__all__ = [
    "ComparisonReceipt",
    "LandedCostBreakdown",
    "QuoteInputs",
    "ReliabilityInputs",
    "SourcingPolicy",
    "compare_quotes",
    "credit_cost",
    "gst_effect",
    "reliability_index",
]
