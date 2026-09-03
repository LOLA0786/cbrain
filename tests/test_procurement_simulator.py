from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from cbrain.simulators import (
    PROCUREMENT_CAPABILITIES,
    PROCUREMENT_QUOTE_READ,
    PROCUREMENT_QUOTE_REQUEST,
    PROCUREMENT_VENDOR_LIST,
    PROCUREMENT_VENDOR_RELIABILITY_READ,
    EffectReceipt,
    ProcurementSimulator,
    SimulatorBusinessError,
    SimulatorConflict,
    SimulatorContractError,
    SimulatorNotFound,
)
from cbrain.simulators.catalog import vendor_catalog_bytes

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CATALOG = ROOT / "tests" / "fixtures" / "vendors.json"

QUOTE_REQUEST = {
    "vendor_id": "v-kalika",
    "grade": "Fe500D",
    "tonnes": "25",
    "payment_terms": "ADVANCE",
}

GST = Decimal("0.18")
DEFAULT_CLOCK = "2026-03-01T09:00:00Z"


def _walk(value: Any) -> None:
    if isinstance(value, float):
        raise AssertionError(f"float leaked into simulator JSON: {value!r}")
    if isinstance(value, dict):
        for child in value.values():
            _walk(child)
        return
    if isinstance(value, list):
        for child in value:
            _walk(child)


def _landed(rate: Decimal, freight: Decimal) -> Decimal:
    return (rate + freight) * (Decimal("1") + GST)


def test_fixture_catalog_is_mirrored_into_the_package() -> None:
    assert FIXTURE_CATALOG.read_bytes() == vendor_catalog_bytes()


def test_seeded_catalog_has_eight_vendors_and_no_floats() -> None:
    payload = json.loads(vendor_catalog_bytes())
    _walk(payload)
    assert len(payload["vendors"]) == 8
    ids = [vendor["vendor_id"] for vendor in payload["vendors"]]
    assert len(set(ids)) == 8


def test_vendor_list_returns_seeded_state_without_policy() -> None:
    target = ProcurementSimulator.from_catalog()
    receipt = target.execute(
        PROCUREMENT_VENDOR_LIST,
        request_id="list-1",
        idempotency_key="list-1",
        arguments={},
    )
    assert isinstance(receipt, EffectReceipt)
    assert receipt.mutated is False
    vendors = receipt.result["vendors"]
    assert len(vendors) == 8
    _walk(receipt.result)
    assert {vendor["vendor_id"] for vendor in vendors} >= {
        "v-kalika",
        "v-grey",
        "v-bhiwandi",
    }
    grey = next(vendor for vendor in vendors if vendor["vendor_id"] == "v-grey")
    assert grey["approved"] is False
    assert grey["bank_beneficiary_id"] == "bnf-grey-new"
    bhilai = next(vendor for vendor in vendors if vendor["vendor_id"] == "v-bhilai")
    assert bhilai["bank_beneficiary_id"] is None
    assert PROCUREMENT_VENDOR_LIST in PROCUREMENT_CAPABILITIES


def test_reliability_is_recorded_history_not_model_opinion() -> None:
    target = ProcurementSimulator.from_catalog()
    receipt = target.execute(
        PROCUREMENT_VENDOR_RELIABILITY_READ,
        request_id="rel-kalika",
        idempotency_key="rel-kalika",
        arguments={"vendor_id": "v-kalika"},
    )
    record = receipt.result
    _walk(record)
    assert record["vendor_id"] == "v-kalika"
    assert record["orders_total"] == 140
    assert record["on_time_pct"] == "51.20"
    assert isinstance(Decimal(record["on_time_pct"]), Decimal)
    assert record["last_dispute_at"] == "2026-02-20T00:00:00Z"


def test_landed_cheapest_vendor_has_the_worst_on_time_record() -> None:
    target = ProcurementSimulator.from_catalog()
    vendors = target.execute(
        PROCUREMENT_VENDOR_LIST,
        request_id="list-spread",
        idempotency_key="list-spread",
        arguments={},
    ).result["vendors"]

    landed: dict[str, Decimal] = {}
    on_time: dict[str, Decimal] = {}
    for vendor in vendors:
        quote = target.execute(
            PROCUREMENT_QUOTE_REQUEST,
            request_id=f"q-{vendor['vendor_id']}",
            idempotency_key=f"q-{vendor['vendor_id']}",
            arguments={
                "vendor_id": vendor["vendor_id"],
                "grade": "Fe500D",
                "tonnes": "25",
                "payment_terms": "ADVANCE",
            },
        ).result
        landed[vendor["vendor_id"]] = _landed(
            Decimal(quote["rate_per_tonne"]),
            Decimal(quote["freight_per_tonne"]),
        )
        reliability = target.execute(
            PROCUREMENT_VENDOR_RELIABILITY_READ,
            request_id=f"r-{vendor['vendor_id']}",
            idempotency_key=f"r-{vendor['vendor_id']}",
            arguments={"vendor_id": vendor["vendor_id"]},
        ).result
        on_time[vendor["vendor_id"]] = Decimal(reliability["on_time_pct"])

    cheapest = min(landed, key=landed.__getitem__)
    worst_on_time = min(on_time, key=on_time.__getitem__)
    assert cheapest == "v-kalika"
    assert worst_on_time == cheapest
    assert on_time[cheapest] < min(
        value for vendor_id, value in on_time.items() if vendor_id != cheapest
    )


def test_quote_receipt_uses_decimal_strings_never_float() -> None:
    target = ProcurementSimulator.from_catalog()
    receipt = target.execute(
        PROCUREMENT_QUOTE_REQUEST,
        request_id="quote-1",
        idempotency_key="quote-1",
        arguments=QUOTE_REQUEST,
    )
    result = receipt.result
    _walk(result)
    assert receipt.mutated is True
    assert result["vendor_id"] == "v-kalika"
    assert result["grade"] == "Fe500D"
    assert result["payment_terms"] == "ADVANCE"
    assert result["status"] == "OPEN"
    assert result["quoted_at"] == DEFAULT_CLOCK
    assert result["valid_until"] == "2026-03-08T09:00:00Z"
    assert result["rate_per_tonne"] == "48500.00"
    assert result["freight_per_tonne"] == "320.00"
    assert result["gst_rate"] == "0.18"
    assert result["tonnes"] == "25.00"
    assert result["quote_digest"].startswith("sha256:")
    for field in (
        "rate_per_tonne",
        "freight_per_tonne",
        "gst_rate",
        "tonnes",
    ):
        parsed = Decimal(result[field])
        assert parsed == parsed  # finite
        assert not isinstance(result[field], float)


def test_expired_quote_is_still_readable() -> None:
    now = {"stamp": DEFAULT_CLOCK}
    target = ProcurementSimulator.from_catalog(clock=lambda: now["stamp"])
    created = target.execute(
        PROCUREMENT_QUOTE_REQUEST,
        request_id="quote-exp",
        idempotency_key="quote-exp",
        arguments=QUOTE_REQUEST,
    ).result
    now["stamp"] = "2026-03-09T09:00:00Z"
    read = target.execute(
        PROCUREMENT_QUOTE_READ,
        request_id="read-exp",
        idempotency_key="read-exp",
        arguments={"quote_id": created["quote_id"]},
    ).result
    assert read["status"] == "EXPIRED"
    assert read["quote_id"] == created["quote_id"]
    assert read["quote_digest"] == created["quote_digest"]


def test_tonnage_below_moq_is_a_business_error() -> None:
    target = ProcurementSimulator.from_catalog()
    with pytest.raises(SimulatorBusinessError, match="MOQ"):
        target.execute(
            PROCUREMENT_QUOTE_REQUEST,
            request_id="below-moq",
            idempotency_key="below-moq",
            arguments={
                "vendor_id": "v-kalika",
                "grade": "Fe500D",
                "tonnes": "19.99",
                "payment_terms": "NET_30",
            },
        )
    assert target.state_version == 0
    assert target.snapshot()["quotes"] == []


def test_quote_request_replay_is_atomic_under_concurrency() -> None:
    target = ProcurementSimulator.from_catalog()

    def request(_: int) -> EffectReceipt:
        return target.execute(
            PROCUREMENT_QUOTE_REQUEST,
            request_id="quote-once",
            idempotency_key="quote-once",
            arguments=QUOTE_REQUEST,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        receipts = list(pool.map(request, range(32)))

    assert len({receipt.effect_id for receipt in receipts}) == 1
    assert len({receipt.result["quote_id"] for receipt in receipts}) == 1
    assert len(target.snapshot()["quotes"]) == 1
    assert target.state_version == 1


def test_quote_idempotency_key_cannot_change_arguments() -> None:
    target = ProcurementSimulator.from_catalog()
    target.execute(
        PROCUREMENT_QUOTE_REQUEST,
        request_id="quote-a",
        idempotency_key="shared-quote",
        arguments=QUOTE_REQUEST,
    )
    with pytest.raises(SimulatorConflict, match="another request"):
        target.execute(
            PROCUREMENT_QUOTE_REQUEST,
            request_id="quote-b",
            idempotency_key="shared-quote",
            arguments={**QUOTE_REQUEST, "payment_terms": "NET_45"},
        )


def test_unknown_vendor_and_quote_are_not_found() -> None:
    target = ProcurementSimulator.from_catalog()
    with pytest.raises(SimulatorNotFound, match="vendor"):
        target.execute(
            PROCUREMENT_VENDOR_RELIABILITY_READ,
            request_id="missing-vendor",
            idempotency_key="missing-vendor",
            arguments={"vendor_id": "v-absent"},
        )
    with pytest.raises(SimulatorNotFound, match="quote"):
        target.execute(
            PROCUREMENT_QUOTE_READ,
            request_id="missing-quote",
            idempotency_key="missing-quote",
            arguments={"quote_id": "quote-absent"},
        )


def test_float_money_and_tonnage_are_rejected() -> None:
    target = ProcurementSimulator.from_catalog()
    with pytest.raises(SimulatorContractError):
        target.execute(
            PROCUREMENT_QUOTE_REQUEST,
            request_id="float-tonnes",
            idempotency_key="float-tonnes",
            arguments={**QUOTE_REQUEST, "tonnes": 25.0},
        )
    snapshot = target.snapshot()
    _walk(snapshot)


def test_snapshot_and_internal_money_are_decimal_not_float() -> None:
    target = ProcurementSimulator.from_catalog()
    target.execute(
        PROCUREMENT_QUOTE_REQUEST,
        request_id="snap",
        idempotency_key="snap",
        arguments=QUOTE_REQUEST,
    )
    snapshot = target.snapshot()
    _walk(snapshot)
    quote = snapshot["quotes"][0]
    assert isinstance(target.quote_rate(quote["quote_id"]), Decimal)
    assert isinstance(target.vendor_rate("v-kalika", "Fe500D"), Decimal)
    assert not isinstance(target.quote_rate(quote["quote_id"]), float)
