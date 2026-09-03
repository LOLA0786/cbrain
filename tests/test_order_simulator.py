from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from procurement_faults import DroppedSimulatorResponse, inject_drop_po_create_response

from cbrain.simulators import (
    PROCUREMENT_PO_AMEND,
    PROCUREMENT_PO_CANCEL,
    PROCUREMENT_PO_CREATE,
    PROCUREMENT_PO_READ,
    PROCUREMENT_SHIPMENT_READ,
    EffectReceipt,
    OrderSimulator,
    SimulatorBusinessError,
    SimulatorConflict,
    SimulatorContractError,
    SimulatorNotFound,
)

CREATE = {
    "vendor_id": "v-kalika",
    "quote_id": "quote-0001",
    "tonnes": "25.00",
    "rate_per_tonne": "48500.00",
    "payment_terms": "ADVANCE",
    "comparison_digest": "sha256:" + ("aa" * 32),
}


def test_po_create_returns_po_number_and_pending_shipment() -> None:
    target = OrderSimulator()
    created = target.execute(
        PROCUREMENT_PO_CREATE,
        request_id="po-1",
        idempotency_key="po-1",
        arguments=CREATE,
    )
    assert created.mutated is True
    assert created.result["po_number"] == "PO-2026-0001"
    assert created.result["status"] == "OPEN"
    assert created.result["rate_per_tonne"] == "48500.00"
    assert not isinstance(created.result["rate_per_tonne"], float)
    shipment = target.execute(
        PROCUREMENT_SHIPMENT_READ,
        request_id="ship-1",
        idempotency_key="ship-1",
        arguments={"po_number": "PO-2026-0001"},
    )
    assert shipment.mutated is False
    assert shipment.result["status"] == "PENDING"


def test_po_create_replay_is_atomic_and_does_not_duplicate() -> None:
    target = OrderSimulator()

    def create(_: int) -> EffectReceipt:
        return target.execute(
            PROCUREMENT_PO_CREATE,
            request_id="po-once",
            idempotency_key="po-once",
            arguments=CREATE,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        receipts = list(pool.map(create, range(32)))

    assert len({receipt.result["po_number"] for receipt in receipts}) == 1
    assert len(target.snapshot()["purchase_orders"]) == 1


def test_drop_after_create_leaves_exactly_one_po() -> None:
    target = OrderSimulator()
    inject_drop_po_create_response(target)
    with pytest.raises(DroppedSimulatorResponse):
        target.execute(
            PROCUREMENT_PO_CREATE,
            request_id="po-drop",
            idempotency_key="po-drop",
            arguments=CREATE,
        )
    assert len(target.snapshot()["purchase_orders"]) == 1
    assert target.snapshot()["purchase_orders"][0]["po_number"] == "PO-2026-0001"


def test_amend_and_cancel_are_ordinary_business_mutations() -> None:
    target = OrderSimulator()
    target.execute(
        PROCUREMENT_PO_CREATE,
        request_id="po-1",
        idempotency_key="po-1",
        arguments=CREATE,
    )
    amended = target.execute(
        PROCUREMENT_PO_AMEND,
        request_id="po-amend",
        idempotency_key="po-amend",
        arguments={
            "po_number": "PO-2026-0001",
            "tonnes": "30.00",
            "rate_per_tonne": "48600.00",
        },
    )
    assert amended.result["status"] == "AMENDED"
    assert amended.result["tonnes"] == "30.00"
    cancelled = target.execute(
        PROCUREMENT_PO_CANCEL,
        request_id="po-cancel",
        idempotency_key="po-cancel",
        arguments={"po_number": "PO-2026-0001"},
    )
    assert cancelled.result["status"] == "CANCELLED"
    with pytest.raises(SimulatorBusinessError, match="cancelled"):
        target.execute(
            PROCUREMENT_PO_AMEND,
            request_id="po-amend-2",
            idempotency_key="po-amend-2",
            arguments={
                "po_number": "PO-2026-0001",
                "tonnes": "31.00",
                "rate_per_tonne": "48700.00",
            },
        )


def test_cannot_cancel_a_delivered_shipment_po() -> None:
    target = OrderSimulator()
    target.execute(
        PROCUREMENT_PO_CREATE,
        request_id="po-1",
        idempotency_key="po-1",
        arguments=CREATE,
    )
    target.set_shipment_status("PO-2026-0001", "DELIVERED")
    with pytest.raises(SimulatorBusinessError, match="delivered"):
        target.execute(
            PROCUREMENT_PO_CANCEL,
            request_id="po-cancel",
            idempotency_key="po-cancel",
            arguments={"po_number": "PO-2026-0001"},
        )


def test_unknown_po_is_not_found() -> None:
    target = OrderSimulator()
    with pytest.raises(SimulatorNotFound, match="purchase order"):
        target.execute(
            PROCUREMENT_PO_READ,
            request_id="missing",
            idempotency_key="missing",
            arguments={"po_number": "PO-absent"},
        )


def test_idempotency_key_cannot_change_create_arguments() -> None:
    target = OrderSimulator()
    target.execute(
        PROCUREMENT_PO_CREATE,
        request_id="po-1",
        idempotency_key="shared",
        arguments=CREATE,
    )
    with pytest.raises(SimulatorConflict, match="another request"):
        target.execute(
            PROCUREMENT_PO_CREATE,
            request_id="po-2",
            idempotency_key="shared",
            arguments={**CREATE, "tonnes": "40.00"},
        )


def test_float_rate_is_rejected() -> None:
    target = OrderSimulator()
    with pytest.raises(SimulatorContractError):
        target.execute(
            PROCUREMENT_PO_CREATE,
            request_id="float",
            idempotency_key="float",
            arguments={**CREATE, "rate_per_tonne": 48500.0},
        )
    assert target.snapshot()["purchase_orders"] == []
