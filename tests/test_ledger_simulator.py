from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import pytest

from cbrain.simulators import (
    PAYMENTS_BENEFICIARY_ADD,
    PAYMENTS_LIMIT_MODIFY,
    PAYMENTS_TRANSFER_INITIATE,
    Beneficiary,
    EffectReceipt,
    LedgerAccount,
    LedgerSimulator,
    SimulatorBusinessError,
    SimulatorConflict,
    SimulatorContractError,
    SimulatorNotFound,
)


def ledger(*, beneficiary: bool = False) -> LedgerSimulator:
    beneficiaries = (
        (Beneficiary("beneficiary-17", "Vendor", "external-account-17"),)
        if beneficiary
        else ()
    )
    return LedgerSimulator(
        (
            LedgerAccount(
                "treasury-primary",
                "USD",
                Decimal("100000.00"),
                Decimal("10000.00"),
            ),
        ),
        beneficiaries,
    )


TRANSFER = {
    "account_id": "treasury-primary",
    "beneficiary_id": "beneficiary-17",
    "amount": "50000.00",
    "currency": "USD",
}


def test_app_fraud_chain_is_business_valid_without_governance() -> None:
    target = ledger()

    with pytest.raises(SimulatorNotFound, match="beneficiary"):
        target.execute(
            PAYMENTS_TRANSFER_INITIATE,
            request_id="before-beneficiary",
            idempotency_key="before-beneficiary",
            arguments=TRANSFER,
        )

    target.execute(
        PAYMENTS_LIMIT_MODIFY,
        request_id="raise-limit",
        idempotency_key="raise-limit",
        arguments={
            "account_id": "treasury-primary",
            "currency": "USD",
            "new_limit": "50000.00",
        },
    )
    target.execute(
        PAYMENTS_BENEFICIARY_ADD,
        request_id="add-beneficiary",
        idempotency_key="add-beneficiary",
        arguments={
            "beneficiary_id": "beneficiary-17",
            "name": "New Settlement Vendor",
            "destination_account_ref": "external-account-17",
        },
    )
    transfer = target.execute(
        PAYMENTS_TRANSFER_INITIATE,
        request_id="transfer",
        idempotency_key="transfer",
        arguments=TRANSFER,
    )

    assert transfer.result["status"] == "SETTLED"
    assert transfer.result["remaining_balance"] == "50000.00"
    assert target.state_version == 3
    assert len(target.snapshot()["transfers"]) == 1


def test_transfer_requires_current_business_limit() -> None:
    target = ledger(beneficiary=True)

    with pytest.raises(SimulatorBusinessError, match="transfer limit"):
        target.execute(
            PAYMENTS_TRANSFER_INITIATE,
            request_id="large-transfer",
            idempotency_key="large-transfer",
            arguments=TRANSFER,
        )

    assert target.state_version == 0
    assert target.snapshot()["accounts"][0]["balance"] == "100000.00"


def test_transfer_replay_cannot_debit_twice_under_concurrency() -> None:
    target = ledger(beneficiary=True)
    target.execute(
        PAYMENTS_LIMIT_MODIFY,
        request_id="raise-limit",
        idempotency_key="raise-limit",
        arguments={
            "account_id": "treasury-primary",
            "currency": "USD",
            "new_limit": "50000.00",
        },
    )

    def transfer(_: int) -> EffectReceipt:
        return target.execute(
            PAYMENTS_TRANSFER_INITIATE,
            request_id="transfer-once",
            idempotency_key="transfer-once",
            arguments=TRANSFER,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        receipts = list(pool.map(transfer, range(32)))

    assert len({receipt.effect_id for receipt in receipts}) == 1
    snapshot = target.snapshot()
    assert snapshot["accounts"][0]["balance"] == "50000.00"
    assert len(snapshot["transfers"]) == 1
    assert snapshot["state_version"] == 2


def test_transfer_idempotency_key_cannot_change_amount() -> None:
    target = ledger(beneficiary=True)
    small = {**TRANSFER, "amount": "5000.00"}
    target.execute(
        PAYMENTS_TRANSFER_INITIATE,
        request_id="transfer-1",
        idempotency_key="transfer-key",
        arguments=small,
    )

    with pytest.raises(SimulatorConflict, match="another request"):
        target.execute(
            PAYMENTS_TRANSFER_INITIATE,
            request_id="transfer-2",
            idempotency_key="transfer-key",
            arguments={**small, "amount": "6000.00"},
        )


@pytest.mark.parametrize("amount", [True, "0", "-1", "1.001", "NaN"])
def test_money_contract_rejects_ambiguous_amounts(amount: object) -> None:
    target = ledger(beneficiary=True)

    with pytest.raises(SimulatorContractError):
        target.execute(
            PAYMENTS_TRANSFER_INITIATE,
            request_id=f"bad-{amount}",
            idempotency_key=f"bad-{amount}",
            arguments={**TRANSFER, "amount": amount},
        )
