"""Governed purchase: HTTP 202 ≠ done; reconcile without blind resubmit.

Evidence level: unit / reference-target. This is not a live ERP sandbox.
"""

from __future__ import annotations

import pytest
from procurement_faults import DroppedSimulatorResponse, inject_drop_po_create_response

from cbrain.procurement.reconcile import (
    OrderSimulatorPurchaseTarget,
    PurchaseOutcome,
    interpret_create_response,
    reconcile_purchase,
    refuse_blind_resubmit,
)
from cbrain.simulators.orders import OrderSimulator

CREATE = {
    "vendor_id": "v-kalika",
    "quote_id": "quote-0001",
    "tonnes": "25.00",
    "rate_per_tonne": "48500.00",
    "payment_terms": "ADVANCE",
    "comparison_digest": "sha256:" + ("aa" * 32),
}


def test_http_202_is_pending_acceptance_not_confirmed() -> None:
    simulator = OrderSimulator()
    target = OrderSimulatorPurchaseTarget(simulator, pending_on_create=True)
    status, receipt = target.create_po(
        request_id="po-202",
        idempotency_key="biz-po-202",
        arguments=CREATE,
    )
    attempt = interpret_create_response(
        http_status=status,
        receipt=receipt,
        idempotency_key="biz-po-202",
    )
    assert attempt.outcome is PurchaseOutcome.PENDING_ACCEPTANCE
    assert attempt.po_number is None

    reconciled = reconcile_purchase(target, idempotency_key="biz-po-202")
    assert reconciled.outcome is PurchaseOutcome.CONFIRMED
    assert reconciled.po_number is not None


def test_dropped_response_is_indeterminate_then_reconciles_same_key() -> None:
    simulator = OrderSimulator()
    inject_drop_po_create_response(simulator)
    target = OrderSimulatorPurchaseTarget(simulator)

    with pytest.raises(DroppedSimulatorResponse):
        target.create_po(
            request_id="po-drop",
            idempotency_key="biz-po-drop",
            arguments=CREATE,
        )

    # The fault injects after OrderSimulator.execute commits; rebuild adapter
    # state from the simulator's durable effect map.
    target = OrderSimulatorPurchaseTarget(simulator)
    attempt = interpret_create_response(
        http_status=None,
        receipt=None,
        idempotency_key="biz-po-drop",
        transport_error=True,
    )
    assert attempt.outcome is PurchaseOutcome.INDETERMINATE

    reconciled = reconcile_purchase(target, idempotency_key="biz-po-drop")
    assert reconciled.outcome is PurchaseOutcome.CONFIRMED
    assert reconciled.po_number.startswith("PO-")


def test_blind_resubmit_with_new_key_is_refused() -> None:
    with pytest.raises(RuntimeError, match="blind resubmit"):
        refuse_blind_resubmit(previous_key="biz-po-1", new_key="biz-po-2")


def test_same_business_key_replay_returns_same_po() -> None:
    simulator = OrderSimulator()
    target = OrderSimulatorPurchaseTarget(simulator)
    first_status, first = target.create_po(
        request_id="po-replay-1",
        idempotency_key="biz-po-stable",
        arguments=CREATE,
    )
    second_status, second = target.create_po(
        request_id="po-replay-1",
        idempotency_key="biz-po-stable",
        arguments=CREATE,
    )
    assert first_status == 200 and second_status == 200
    assert first is not None and second is not None
    assert first.result["po_number"] == second.result["po_number"]
    assert first.effect_id == second.effect_id
