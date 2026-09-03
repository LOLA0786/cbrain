from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from cbrain.procurement.comparator import (
    ComparisonReceipt,
    QuoteInputs,
    ReliabilityInputs,
    SourcingPolicy,
    compare_quotes,
)
from cbrain.simulators import (
    PROCUREMENT_QUOTE_REQUEST,
    PROCUREMENT_VENDOR_LIST,
    PROCUREMENT_VENDOR_RELIABILITY_READ,
    ProcurementSimulator,
)


def _walk(value: Any) -> None:
    if isinstance(value, float):
        raise AssertionError(f"float leaked into comparison JSON: {value!r}")
    if isinstance(value, dict):
        for child in value.values():
            _walk(child)
        return
    if isinstance(value, list):
        for child in value:
            _walk(child)


def _quotes_and_reliability(
    *,
    payment_terms: str | None = None,
) -> tuple[tuple[QuoteInputs, ...], tuple[ReliabilityInputs, ...]]:
    target = ProcurementSimulator.from_catalog()
    vendors = target.execute(
        PROCUREMENT_VENDOR_LIST,
        request_id="list",
        idempotency_key="list",
        arguments={},
    ).result["vendors"]
    quotes: list[QuoteInputs] = []
    reliability: list[ReliabilityInputs] = []
    for vendor in vendors:
        terms = payment_terms
        if terms is None:
            terms = {
                "v-kalika": "ADVANCE",
                "v-jsw": "NET_15",
                "v-jindal": "NET_30",
                "v-bhiwandi": "NET_45",
                "v-raigad": "LC_SIGHT",
                "v-hazira": "NET_30",
                "v-bhilai": "ADVANCE",
                "v-grey": "ADVANCE",
            }[vendor["vendor_id"]]
        quote = target.execute(
            PROCUREMENT_QUOTE_REQUEST,
            request_id=f"q-{vendor['vendor_id']}",
            idempotency_key=f"q-{vendor['vendor_id']}",
            arguments={
                "vendor_id": vendor["vendor_id"],
                "grade": "Fe500D",
                "tonnes": "25",
                "payment_terms": terms,
            },
        ).result
        quotes.append(
            QuoteInputs(
                quote_id=quote["quote_id"],
                vendor_id=quote["vendor_id"],
                grade=quote["grade"],
                tonnes=Decimal(quote["tonnes"]),
                rate_per_tonne=Decimal(quote["rate_per_tonne"]),
                freight_per_tonne=Decimal(quote["freight_per_tonne"]),
                gst_rate=Decimal(quote["gst_rate"]),
                payment_terms=quote["payment_terms"],
                quote_digest=quote["quote_digest"],
            )
        )
        record = target.execute(
            PROCUREMENT_VENDOR_RELIABILITY_READ,
            request_id=f"r-{vendor['vendor_id']}",
            idempotency_key=f"r-{vendor['vendor_id']}",
            arguments={"vendor_id": vendor["vendor_id"]},
        ).result
        reliability.append(
            ReliabilityInputs(
                vendor_id=record["vendor_id"],
                orders_total=record["orders_total"],
                on_time_pct=Decimal(record["on_time_pct"]),
                short_delivery_pct=Decimal(record["short_delivery_pct"]),
                quality_rejection_pct=Decimal(record["quality_rejection_pct"]),
                avg_delay_days=Decimal(record["avg_delay_days"]),
            )
        )
    return tuple(quotes), tuple(reliability)


def _policy() -> SourcingPolicy:
    return SourcingPolicy(
        weight_cost=Decimal("0.7"),
        weight_reliability=Decimal("0.3"),
        annual_cost_of_capital=Decimal("0.12"),
        input_credit_eligible=False,
    )


def test_headline_cheapest_rate_loses_after_freight_and_terms() -> None:
    quotes, reliability = _quotes_and_reliability()
    receipt = compare_quotes(
        quotes,
        reliability,
        _policy(),
        computed_at="2026-03-01T09:00:00Z",
    )
    cheapest_rate = min(quotes, key=lambda item: item.rate_per_tonne)
    cheapest_landed = min(
        receipt.breakdowns, key=lambda item: item.landed_cost_per_tonne
    )
    assert cheapest_rate.vendor_id == "v-bhilai"
    assert cheapest_landed.vendor_id != "v-bhilai"
    assert cheapest_landed.vendor_id == "v-kalika"


def test_landed_cheapest_loses_once_reliability_weighting_applies() -> None:
    quotes, reliability = _quotes_and_reliability()
    receipt = compare_quotes(
        quotes,
        reliability,
        _policy(),
        computed_at="2026-03-01T09:00:00Z",
    )
    cheapest_landed = min(
        receipt.breakdowns, key=lambda item: item.landed_cost_per_tonne
    )
    ranked = receipt.ranked_vendor_ids
    assert cheapest_landed.vendor_id == "v-kalika"
    assert ranked[0] != cheapest_landed.vendor_id


def test_credit_cost_is_zero_for_advance_and_negative_for_net_45() -> None:
    quotes, reliability = _quotes_and_reliability()
    receipt = compare_quotes(
        quotes,
        reliability,
        _policy(),
        computed_at="2026-03-01T09:00:00Z",
    )
    by_vendor = {item.vendor_id: item for item in receipt.breakdowns}
    assert by_vendor["v-kalika"].credit_cost_per_tonne == Decimal("0.00")
    assert by_vendor["v-bhiwandi"].credit_cost_per_tonne < Decimal("0")
    net_45 = QuoteInputs(
        quote_id="q-same",
        vendor_id="v-same",
        grade="Fe500D",
        tonnes=Decimal("25.00"),
        rate_per_tonne=Decimal("50000.00"),
        freight_per_tonne=Decimal("400.00"),
        gst_rate=Decimal("0.18"),
        payment_terms="NET_45",
        quote_digest="sha256:" + ("ab" * 32),
    )
    advance = QuoteInputs(
        quote_id="q-adv",
        vendor_id="v-adv",
        grade="Fe500D",
        tonnes=Decimal("25.00"),
        rate_per_tonne=Decimal("50000.00"),
        freight_per_tonne=Decimal("400.00"),
        gst_rate=Decimal("0.18"),
        payment_terms="ADVANCE",
        quote_digest="sha256:" + ("cd" * 32),
    )
    reliability_pair = (
        ReliabilityInputs(
            vendor_id="v-same",
            orders_total=10,
            on_time_pct=Decimal("90.00"),
            short_delivery_pct=Decimal("1.00"),
            quality_rejection_pct=Decimal("1.00"),
            avg_delay_days=Decimal("1.00"),
        ),
        ReliabilityInputs(
            vendor_id="v-adv",
            orders_total=10,
            on_time_pct=Decimal("90.00"),
            short_delivery_pct=Decimal("1.00"),
            quality_rejection_pct=Decimal("1.00"),
            avg_delay_days=Decimal("1.00"),
        ),
    )
    compared = compare_quotes(
        (net_45, advance),
        reliability_pair,
        _policy(),
        computed_at="2026-03-01T09:00:00Z",
    )
    by_id = {item.vendor_id: item for item in compared.breakdowns}
    assert by_id["v-same"].landed_cost_per_tonne < by_id["v-adv"].landed_cost_per_tonne


def test_gst_effect_is_zero_when_input_credit_is_eligible() -> None:
    quotes, reliability = _quotes_and_reliability(payment_terms="ADVANCE")
    credited = compare_quotes(
        quotes,
        reliability,
        SourcingPolicy(
            weight_cost=Decimal("0.7"),
            weight_reliability=Decimal("0.3"),
            annual_cost_of_capital=Decimal("0.12"),
            input_credit_eligible=True,
        ),
        computed_at="2026-03-01T09:00:00Z",
    )
    assert all(
        item.gst_effect_per_tonne == Decimal("0.00") for item in credited.breakdowns
    )


def test_comparison_receipt_is_byte_stable() -> None:
    quotes, reliability = _quotes_and_reliability()
    first = compare_quotes(
        quotes,
        reliability,
        _policy(),
        computed_at="2026-03-01T09:00:00Z",
    )
    second = compare_quotes(
        quotes,
        reliability,
        _policy(),
        computed_at="2026-03-01T09:00:00Z",
    )
    assert isinstance(first, ComparisonReceipt)
    assert first.to_canonical_bytes() == second.to_canonical_bytes()
    assert first.comparison_digest == second.comparison_digest
    assert first.comparison_digest.startswith("sha256:")
    _walk(first.to_payload())
    assert first.policy_digest.startswith("sha256:")
    assert len(first.quote_digests) == 8


def test_default_weights_are_seven_tenths_cost() -> None:
    policy = SourcingPolicy(
        annual_cost_of_capital=Decimal("0.12"),
        input_credit_eligible=False,
    )
    assert policy.weight_cost == Decimal("0.7")
    assert policy.weight_reliability == Decimal("0.3")


def test_float_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        SourcingPolicy(
            weight_cost=0.7,  # type: ignore[arg-type]
            weight_reliability=Decimal("0.3"),
            annual_cost_of_capital=Decimal("0.12"),
            input_credit_eligible=False,
        )
    with pytest.raises(ValueError):
        QuoteInputs(
            quote_id="q1",
            vendor_id="v1",
            grade="Fe500D",
            tonnes=25.0,  # type: ignore[arg-type]
            rate_per_tonne=Decimal("1.00"),
            freight_per_tonne=Decimal("1.00"),
            gst_rate=Decimal("0.18"),
            payment_terms="ADVANCE",
            quote_digest="sha256:" + ("ee" * 32),
        )
