"""Typed argument validation before company-agent governance decisions."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from cbrain.contracts import ActionIntent

from .authority import CompanyExecutionContext
from .kinds import CompanyAgentKind
from .reasons import (
    ACCOUNTS_AMOUNT_INVALID,
    ACCOUNTS_AMOUNT_NON_FINITE,
    ACCOUNTS_AMOUNT_NON_POSITIVE,
    ACCOUNTS_AMOUNT_OVER_LIMIT,
    ACCOUNTS_BENEFICIARY_MISMATCH,
    ACCOUNTS_CURRENCY_UNSUPPORTED,
    ACCOUNTS_INVOICE_MISMATCH,
    LEGAL_CONTRACT_OUTSIDE_SCOPE,
    LEGAL_CROSS_MATTER_AUTHORITY_REQUIRED,
    LEGAL_MATTER_SCOPE_MISMATCH,
    LEGAL_MATTER_SCOPE_REQUIRED,
    PROCUREMENT_MATERIAL_UNKNOWN,
    PROCUREMENT_QUOTE_MISMATCH,
    PROCUREMENT_QUOTE_MISSING,
    PROCUREMENT_REQUISITION_MISSING,
    PROCUREMENT_RFQ_IDEMPOTENCY_CONFLICT,
    PROCUREMENT_RFQ_VENDORS_INVALID,
    PROCUREMENT_VENDOR_UNKNOWN,
    PROCUREMENT_VENDOR_UNREGISTERED,
)
from .simulators import CompanySimulatorBundle

LEGAL_SCOPED_TOOLS = frozenset(
    {
        "search_contracts",
        "extract_clause",
        "compare_contracts",
        "identify_deviations",
        "prepare_redline",
        "draft_legal_summary",
        "send_commitment",
        "sign_contract",
        "file_document",
        "provide_legal_advice",
    }
)
LEGAL_CONTRACT_TOOLS = frozenset(
    {
        "extract_clause",
        "identify_deviations",
        "prepare_redline",
        "draft_legal_summary",
        "sign_contract",
    }
)
MONEY_TOOLS = frozenset({"prepare_payment", "prepare_refund", "execute_payment"})
PROCUREMENT_MONEY_TOOLS = frozenset(
    {"award_quote", "create_purchase_requisition", "release_purchase_order"}
)
SUPPORTED_CURRENCIES = frozenset({"USD", "EUR"})
AMOUNT_CEILING = Decimal("100000000")
_INTEGER_MINOR = re.compile(r"^-?\d+$")
_BOOL_TEXT = frozenset({"true", "false"})


@dataclass(frozen=True, slots=True)
class ArgumentValidationResult:
    ok: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class _ParsedAmount:
    amount: Decimal | None
    reason: str | None


def validate_company_action(
    *,
    kind: CompanyAgentKind,
    action: ActionIntent,
    context: CompanyExecutionContext | None,
    bundle: CompanySimulatorBundle | None,
) -> ArgumentValidationResult:
    if kind is CompanyAgentKind.LEGAL and action.tool_name in LEGAL_SCOPED_TOOLS:
        return _validate_legal(action, context=context, bundle=bundle)
    if kind is CompanyAgentKind.ACCOUNTS and action.tool_name in MONEY_TOOLS:
        return _validate_money(action, bundle=bundle)
    if kind is CompanyAgentKind.PROCUREMENT:
        return _validate_procurement(action, bundle=bundle)
    return ArgumentValidationResult(ok=True)


def _validate_legal(
    action: ActionIntent,
    *,
    context: CompanyExecutionContext | None,
    bundle: CompanySimulatorBundle | None,
) -> ArgumentValidationResult:
    if context is None:
        return ArgumentValidationResult(ok=False, reason=LEGAL_MATTER_SCOPE_REQUIRED)
    permitted = context.permitted_matter_ids
    arguments = action.arguments
    tool = action.tool_name
    if tool in {"search_contracts", "send_commitment", "provide_legal_advice"}:
        matter_id = arguments.get("matter_id")
        if not isinstance(matter_id, str) or matter_id not in permitted:
            return ArgumentValidationResult(
                ok=False, reason=LEGAL_MATTER_SCOPE_MISMATCH
            )
        return ArgumentValidationResult(ok=True)
    if tool == "compare_contracts":
        return _validate_compare(arguments, permitted=permitted, bundle=bundle)
    if tool in LEGAL_CONTRACT_TOOLS or tool == "file_document":
        contract_id = str(
            arguments.get("contract_id") or arguments.get("document_id") or ""
        )
        if contract_id not in _authorized_contract_ids(permitted, bundle):
            return ArgumentValidationResult(
                ok=False, reason=LEGAL_CONTRACT_OUTSIDE_SCOPE
            )
        return ArgumentValidationResult(ok=True)
    return ArgumentValidationResult(ok=True)


def _validate_compare(
    arguments: Mapping[str, Any],
    *,
    permitted: frozenset[str],
    bundle: CompanySimulatorBundle | None,
) -> ArgumentValidationResult:
    left_id = str(arguments.get("left_id") or "")
    right_id = str(arguments.get("right_id") or "")
    authorized = _authorized_contract_ids(permitted, bundle)
    left_matter = _contract_matter(left_id, bundle)
    right_matter = _contract_matter(right_id, bundle)
    if (
        left_matter is not None
        and right_matter is not None
        and left_matter != right_matter
        and not {left_matter, right_matter} <= set(permitted)
    ):
        return ArgumentValidationResult(
            ok=False, reason=LEGAL_CROSS_MATTER_AUTHORITY_REQUIRED
        )
    if left_id not in authorized or right_id not in authorized:
        return ArgumentValidationResult(ok=False, reason=LEGAL_CONTRACT_OUTSIDE_SCOPE)
    return ArgumentValidationResult(ok=True)


def _authorized_contract_ids(
    permitted: frozenset[str], bundle: CompanySimulatorBundle | None
) -> set[str]:
    if bundle is None:
        return set()
    ids: set[str] = set()
    for matter_id in permitted:
        ids.update(bundle.legal.matters.get(matter_id, set()))
    return ids


def _contract_matter(
    contract_id: str, bundle: CompanySimulatorBundle | None
) -> str | None:
    if bundle is None:
        return None
    contract = bundle.legal.contracts.get(contract_id)
    if not isinstance(contract, dict):
        return None
    matter = contract.get("matter_id")
    return str(matter) if matter is not None else None


def _validate_money(
    action: ActionIntent, *, bundle: CompanySimulatorBundle | None
) -> ArgumentValidationResult:
    arguments = action.arguments
    parsed = _parse_amount(arguments.get("amount_minor"))
    if parsed.reason is not None:
        return ArgumentValidationResult(ok=False, reason=parsed.reason)
    amount = parsed.amount
    if amount is None:
        return ArgumentValidationResult(ok=False, reason=ACCOUNTS_AMOUNT_INVALID)
    currency = arguments.get("currency")
    if not isinstance(currency, str) or currency not in SUPPORTED_CURRENCIES:
        return ArgumentValidationResult(ok=False, reason=ACCOUNTS_CURRENCY_UNSUPPORTED)
    invoice_id = arguments.get("invoice_id")
    if not isinstance(invoice_id, str) or bundle is None:
        return ArgumentValidationResult(ok=False, reason=ACCOUNTS_INVOICE_MISMATCH)
    invoice = bundle.accounts.invoices.get(invoice_id)
    if invoice is None:
        return ArgumentValidationResult(ok=False, reason=ACCOUNTS_INVOICE_MISMATCH)
    invoice_currency = str(invoice.get("currency") or "")
    invoice_amount = Decimal(str(invoice.get("amount_minor") or "0"))
    if currency != invoice_currency:
        return ArgumentValidationResult(ok=False, reason=ACCOUNTS_INVOICE_MISMATCH)
    if action.tool_name == "prepare_refund":
        if amount > invoice_amount:
            return ArgumentValidationResult(ok=False, reason=ACCOUNTS_INVOICE_MISMATCH)
    elif amount != invoice_amount:
        return ArgumentValidationResult(ok=False, reason=ACCOUNTS_INVOICE_MISMATCH)
    beneficiary = arguments.get("beneficiary_id")
    if action.tool_name == "execute_payment" or beneficiary is not None:
        vendor_id = str(invoice.get("vendor_id") or "")
        if not isinstance(beneficiary, str) or beneficiary != vendor_id:
            return ArgumentValidationResult(
                ok=False, reason=ACCOUNTS_BENEFICIARY_MISMATCH
            )
    return ArgumentValidationResult(ok=True)


def _parse_amount(value: object) -> _ParsedAmount:
    if value is None:
        return _ParsedAmount(None, ACCOUNTS_AMOUNT_INVALID)
    if isinstance(value, (bool, float)):
        return _ParsedAmount(None, ACCOUNTS_AMOUNT_INVALID)
    if isinstance(value, int):
        text = str(value)
    elif isinstance(value, str):
        text = value.strip()
    else:
        return _ParsedAmount(None, ACCOUNTS_AMOUNT_INVALID)
    if not text or text.lower() in _BOOL_TEXT:
        return _ParsedAmount(None, ACCOUNTS_AMOUNT_INVALID)
    lowered = text.lower()
    if lowered in {"nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
        return _ParsedAmount(None, ACCOUNTS_AMOUNT_NON_FINITE)
    if "e" in lowered:
        return _ParsedAmount(None, ACCOUNTS_AMOUNT_INVALID)
    if not _INTEGER_MINOR.fullmatch(text):
        return _ParsedAmount(None, ACCOUNTS_AMOUNT_INVALID)
    try:
        amount = Decimal(text)
    except (InvalidOperation, ValueError):
        return _ParsedAmount(None, ACCOUNTS_AMOUNT_INVALID)
    if not amount.is_finite():
        return _ParsedAmount(None, ACCOUNTS_AMOUNT_NON_FINITE)
    if amount <= 0:
        return _ParsedAmount(None, ACCOUNTS_AMOUNT_NON_POSITIVE)
    if amount > AMOUNT_CEILING:
        return _ParsedAmount(None, ACCOUNTS_AMOUNT_OVER_LIMIT)
    return _ParsedAmount(amount, None)


def _validate_procurement(
    action: ActionIntent, *, bundle: CompanySimulatorBundle | None
) -> ArgumentValidationResult:
    if bundle is None:
        return ArgumentValidationResult(ok=False, reason=PROCUREMENT_VENDOR_UNKNOWN)
    arguments = action.arguments
    tool = action.tool_name
    vendors = bundle.procurement.vendors
    if tool in {"lookup_vendor", "change_vendor_bank", "post_erp_payment"}:
        vendor_id = arguments.get("vendor_id")
        if not isinstance(vendor_id, str) or vendor_id not in vendors:
            return ArgumentValidationResult(ok=False, reason=PROCUREMENT_VENDOR_UNKNOWN)
        return ArgumentValidationResult(ok=True)
    if tool == "search_catalog":
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return ArgumentValidationResult(
                ok=False, reason=PROCUREMENT_MATERIAL_UNKNOWN
            )
        return ArgumentValidationResult(ok=True)
    if tool == "send_rfq_email":
        return _validate_rfq(arguments, bundle=bundle)
    if tool == "show_quotations":
        rfq_id = arguments.get("rfq_id")
        if not isinstance(rfq_id, str) or not rfq_id.strip():
            return ArgumentValidationResult(ok=False, reason=PROCUREMENT_QUOTE_MISSING)
        return ArgumentValidationResult(ok=True)
    if tool in PROCUREMENT_MONEY_TOOLS:
        return _validate_procurement_money(action, bundle=bundle)
    return ArgumentValidationResult(ok=True)


def _validate_rfq(
    arguments: Mapping[str, Any], *, bundle: CompanySimulatorBundle
) -> ArgumentValidationResult:
    material_id = arguments.get("material_id")
    if not isinstance(material_id, str) or (
        material_id not in bundle.procurement.catalog
    ):
        return ArgumentValidationResult(ok=False, reason=PROCUREMENT_MATERIAL_UNKNOWN)
    vendor_ids = arguments.get("vendor_ids")
    if not isinstance(vendor_ids, list) or not vendor_ids:
        return ArgumentValidationResult(
            ok=False, reason=PROCUREMENT_RFQ_VENDORS_INVALID
        )
    seen: set[str] = set()
    for item in vendor_ids:
        if not isinstance(item, str) or not item.strip() or item in seen:
            return ArgumentValidationResult(
                ok=False, reason=PROCUREMENT_RFQ_VENDORS_INVALID
            )
        seen.add(item)
        vendor = bundle.procurement.vendors.get(item)
        if vendor is None:
            return ArgumentValidationResult(ok=False, reason=PROCUREMENT_VENDOR_UNKNOWN)
        if not vendor.registered:
            return ArgumentValidationResult(
                ok=False, reason=PROCUREMENT_VENDOR_UNREGISTERED
            )
    rfq_id = arguments.get("rfq_id")
    if not isinstance(rfq_id, str) or not rfq_id.strip():
        return ArgumentValidationResult(
            ok=False, reason=PROCUREMENT_RFQ_VENDORS_INVALID
        )
    if bundle.procurement.rfq_idempotency_conflict(arguments):
        return ArgumentValidationResult(
            ok=False, reason=PROCUREMENT_RFQ_IDEMPOTENCY_CONFLICT
        )
    return ArgumentValidationResult(ok=True)


def _validate_procurement_money(
    action: ActionIntent, *, bundle: CompanySimulatorBundle
) -> ArgumentValidationResult:
    parsed = _parse_amount(action.arguments.get("amount_minor"))
    if parsed.reason is not None:
        return ArgumentValidationResult(ok=False, reason=parsed.reason)
    amount = parsed.amount
    if amount is None:
        return ArgumentValidationResult(ok=False, reason=ACCOUNTS_AMOUNT_INVALID)
    currency = action.arguments.get("currency")
    if not isinstance(currency, str) or currency not in SUPPORTED_CURRENCIES:
        return ArgumentValidationResult(ok=False, reason=ACCOUNTS_CURRENCY_UNSUPPORTED)
    if action.tool_name == "release_purchase_order":
        pr_id = action.arguments.get("pr_id")
        if not isinstance(pr_id, str):
            return ArgumentValidationResult(
                ok=False, reason=PROCUREMENT_REQUISITION_MISSING
            )
        requisition = bundle.procurement.requisitions.get(pr_id)
        if requisition is None:
            return ArgumentValidationResult(
                ok=False, reason=PROCUREMENT_REQUISITION_MISSING
            )
        if (
            str(requisition.get("amount_minor")) != str(amount)
            or str(requisition.get("currency")) != currency
        ):
            return ArgumentValidationResult(ok=False, reason=PROCUREMENT_QUOTE_MISMATCH)
        return ArgumentValidationResult(ok=True)
    rfq_id = action.arguments.get("rfq_id")
    quote_id = action.arguments.get("quote_id")
    if not isinstance(rfq_id, str) or not isinstance(quote_id, str):
        return ArgumentValidationResult(ok=False, reason=PROCUREMENT_QUOTE_MISSING)
    quote = bundle.procurement.quotations.get(quote_id)
    if quote is None or quote.rfq_id != rfq_id:
        return ArgumentValidationResult(ok=False, reason=PROCUREMENT_QUOTE_MISSING)
    vendor = bundle.procurement.vendors.get(quote.vendor_id)
    if vendor is None or not vendor.registered:
        return ArgumentValidationResult(
            ok=False, reason=PROCUREMENT_VENDOR_UNREGISTERED
        )
    if quote.amount() != amount or quote.currency != currency:
        return ArgumentValidationResult(ok=False, reason=PROCUREMENT_QUOTE_MISMATCH)
    if action.tool_name == "create_purchase_requisition":
        material_id = action.arguments.get("material_id")
        quantity = action.arguments.get("quantity")
        if material_id != quote.material_id:
            return ArgumentValidationResult(ok=False, reason=PROCUREMENT_QUOTE_MISMATCH)
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 1:
            return ArgumentValidationResult(ok=False, reason=PROCUREMENT_QUOTE_MISMATCH)
    return ArgumentValidationResult(ok=True)


__all__ = [
    "AMOUNT_CEILING",
    "ArgumentValidationResult",
    "LEGAL_SCOPED_TOOLS",
    "MONEY_TOOLS",
    "PROCUREMENT_MONEY_TOOLS",
    "SUPPORTED_CURRENCIES",
    "validate_company_action",
]
