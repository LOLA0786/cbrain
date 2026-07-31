"""Mutable finance ledger target for governed-execution scenarios."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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

PAYMENTS_BALANCE_READ = "payments.balance.read"
PAYMENTS_LIMIT_READ = "payments.limit.read"
PAYMENTS_LIMIT_MODIFY = "payments.limit.modify"
PAYMENTS_BENEFICIARY_ADD = "payments.beneficiary.add"
PAYMENTS_TRANSFER_INITIATE = "payments.transfer.initiate"
PAYMENT_CAPABILITIES = frozenset(
    {
        PAYMENTS_BALANCE_READ,
        PAYMENTS_LIMIT_READ,
        PAYMENTS_LIMIT_MODIFY,
        PAYMENTS_BENEFICIARY_ADD,
        PAYMENTS_TRANSFER_INITIATE,
    }
)


@dataclass(frozen=True, slots=True)
class LedgerAccount:
    account_id: str
    currency: str
    balance: Decimal
    transfer_limit: Decimal

    def __post_init__(self) -> None:
        required_text(self.account_id, "account_id")
        required_text(self.currency, "currency")
        if self.balance < 0 or self.transfer_limit <= 0:
            raise SimulatorContractError("account amounts are invalid")

    def to_payload(self) -> JsonObject:
        return {
            "account_id": self.account_id,
            "currency": self.currency,
            "balance": _money_text(self.balance),
            "transfer_limit": _money_text(self.transfer_limit),
        }


@dataclass(frozen=True, slots=True)
class Beneficiary:
    beneficiary_id: str
    name: str
    destination_account_ref: str

    def __post_init__(self) -> None:
        required_text(self.beneficiary_id, "beneficiary_id")
        required_text(self.name, "name")
        required_text(self.destination_account_ref, "destination_account_ref")

    def to_payload(self) -> JsonObject:
        return {
            "beneficiary_id": self.beneficiary_id,
            "name": self.name,
            "destination_account_ref": self.destination_account_ref,
        }


class LedgerSimulator:
    """Thread-safe ledger whose ordinary business rules allow an APP chain."""

    domain = "payments"

    def __init__(
        self,
        accounts: Sequence[LedgerAccount],
        beneficiaries: Sequence[Beneficiary] = (),
    ) -> None:
        self._lock = RLock()
        self._accounts: dict[str, LedgerAccount] = {}
        for account in accounts:
            if account.account_id in self._accounts:
                raise SimulatorConflict("duplicate account_id")
            self._accounts[account.account_id] = account
        self._beneficiaries: dict[str, Beneficiary] = {}
        for beneficiary in beneficiaries:
            if beneficiary.beneficiary_id in self._beneficiaries:
                raise SimulatorConflict("duplicate beneficiary_id")
            self._beneficiaries[beneficiary.beneficiary_id] = beneficiary
        self._transfers: list[JsonObject] = []
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
                "accounts": [
                    self._accounts[key].to_payload() for key in sorted(self._accounts)
                ],
                "beneficiaries": [
                    self._beneficiaries[key].to_payload()
                    for key in sorted(self._beneficiaries)
                ],
                "transfers": self._transfers,
            }
            return restore_object(
                canonical_object(value, "ledger snapshot"), "ledger snapshot"
            )

    def execute(
        self,
        capability: str,
        *,
        request_id: str,
        idempotency_key: str,
        arguments: Mapping[str, Any],
    ) -> EffectReceipt:
        if capability not in PAYMENT_CAPABILITIES:
            raise SimulatorContractError("unknown payment capability")
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
        if capability == PAYMENTS_BALANCE_READ:
            return self._balance(arguments), False
        if capability == PAYMENTS_LIMIT_READ:
            return self._limit(arguments), False
        if capability == PAYMENTS_LIMIT_MODIFY:
            return self._modify_limit(arguments), True
        if capability == PAYMENTS_BENEFICIARY_ADD:
            return self._add_beneficiary(arguments), True
        if capability == PAYMENTS_TRANSFER_INITIATE:
            return self._transfer(arguments), True
        raise SimulatorContractError("unknown payment capability")

    def _balance(self, arguments: JsonObject) -> JsonObject:
        account = self._account_from(arguments)
        return {
            "account_id": account.account_id,
            "currency": account.currency,
            "balance": _money_text(account.balance),
        }

    def _limit(self, arguments: JsonObject) -> JsonObject:
        account = self._account_from(arguments)
        return {
            "account_id": account.account_id,
            "currency": account.currency,
            "transfer_limit": _money_text(account.transfer_limit),
        }

    def _modify_limit(self, arguments: JsonObject) -> JsonObject:
        exact_fields(
            arguments,
            frozenset({"account_id", "currency", "new_limit"}),
            "arguments",
        )
        account = self._required_account(
            required_text(arguments["account_id"], "arguments.account_id")
        )
        self._require_currency(account, arguments["currency"])
        new_limit = _positive_money(arguments["new_limit"], "arguments.new_limit")
        previous = account.transfer_limit
        self._accounts[account.account_id] = LedgerAccount(
            account_id=account.account_id,
            currency=account.currency,
            balance=account.balance,
            transfer_limit=new_limit,
        )
        return {
            "account_id": account.account_id,
            "currency": account.currency,
            "previous_limit": _money_text(previous),
            "new_limit": _money_text(new_limit),
        }

    def _add_beneficiary(self, arguments: JsonObject) -> JsonObject:
        exact_fields(
            arguments,
            frozenset({"beneficiary_id", "name", "destination_account_ref"}),
            "arguments",
        )
        beneficiary = Beneficiary(
            beneficiary_id=required_text(
                arguments["beneficiary_id"], "arguments.beneficiary_id"
            ),
            name=required_text(arguments["name"], "arguments.name"),
            destination_account_ref=required_text(
                arguments["destination_account_ref"],
                "arguments.destination_account_ref",
            ),
        )
        if beneficiary.beneficiary_id in self._beneficiaries:
            raise SimulatorConflict("beneficiary already exists")
        self._beneficiaries[beneficiary.beneficiary_id] = beneficiary
        return {**beneficiary.to_payload(), "status": "ACTIVE"}

    def _transfer(self, arguments: JsonObject) -> JsonObject:
        exact_fields(
            arguments,
            frozenset({"account_id", "beneficiary_id", "amount", "currency"}),
            "arguments",
        )
        account = self._required_account(
            required_text(arguments["account_id"], "arguments.account_id")
        )
        self._require_currency(account, arguments["currency"])
        beneficiary_id = required_text(
            arguments["beneficiary_id"], "arguments.beneficiary_id"
        )
        if beneficiary_id not in self._beneficiaries:
            raise SimulatorNotFound("beneficiary does not exist")
        amount = _positive_money(arguments["amount"], "arguments.amount")
        if amount > account.transfer_limit:
            raise SimulatorBusinessError("amount exceeds current transfer limit")
        if amount > account.balance:
            raise SimulatorBusinessError("insufficient funds")
        remaining = account.balance - amount
        self._accounts[account.account_id] = LedgerAccount(
            account_id=account.account_id,
            currency=account.currency,
            balance=remaining,
            transfer_limit=account.transfer_limit,
        )
        transfer_id = f"transfer-{len(self._transfers) + 1}"
        transfer = {
            "transfer_id": transfer_id,
            "account_id": account.account_id,
            "beneficiary_id": beneficiary_id,
            "amount": _money_text(amount),
            "currency": account.currency,
            "status": "SETTLED",
        }
        self._transfers.append(transfer)
        return {**transfer, "remaining_balance": _money_text(remaining)}

    def _account_from(self, arguments: JsonObject) -> LedgerAccount:
        exact_fields(arguments, frozenset({"account_id"}), "arguments")
        return self._required_account(
            required_text(arguments["account_id"], "arguments.account_id")
        )

    def _required_account(self, account_id: str) -> LedgerAccount:
        account = self._accounts.get(account_id)
        if account is None:
            raise SimulatorNotFound("account does not exist")
        return account

    @staticmethod
    def _require_currency(account: LedgerAccount, value: object) -> None:
        currency = required_text(value, "arguments.currency")
        if currency != account.currency:
            raise SimulatorBusinessError("currency does not match account")


def _positive_money(value: object, path: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise SimulatorContractError(f"{path} must be a decimal amount")
    try:
        amount = Decimal(str(value))
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
    "PAYMENT_CAPABILITIES",
    "PAYMENTS_BALANCE_READ",
    "PAYMENTS_BENEFICIARY_ADD",
    "PAYMENTS_LIMIT_MODIFY",
    "PAYMENTS_LIMIT_READ",
    "PAYMENTS_TRANSFER_INITIATE",
    "Beneficiary",
    "LedgerAccount",
    "LedgerSimulator",
]
