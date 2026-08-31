"""Purchasing proof pack: quote board plus verified PrivateVault evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from cbrain.adapters.privatevault_execution import (
    PrivateVaultAgentDNAVerifier,
    VerifiedClosure,
)
from cbrain.contracts import ActionIntent, ExecutionStatus, GovernedExecution

from .erp import ProcurementError
from .mailbox import quote_board_digest

PROCUREMENT_PROOF_SCHEMA = "cbrain-procurement-proof/v1"
_OFFLINE_AUTHORITY = "company_test_gateway"
_PRIVATEVAULT_AUTHORITY = "privatevault"
_SUCCESS_PROOF_TOOLS = frozenset(
    {
        "award_quote",
        "create_purchase_requisition",
        "release_purchase_order",
    }
)
_FORBIDDEN = frozenset(
    {
        "api_key",
        "arguments",
        "authorization",
        "connection",
        "credential",
        "credentials",
        "database_url",
        "dsn",
        "password",
        "private_key",
        "secret",
        "secrets",
        "token",
    }
)


@dataclass(frozen=True, slots=True)
class ProcurementProof:
    schema: str
    rfq_id: str
    quote_id: str | None
    awarded_vendor_id: str | None
    action_intent_digest: str
    execution_status: str
    tool_executed: bool | None
    decision_authority: str
    decision_id: str | None
    decision_receipt_digest: str | None
    authority_receipt_digest: str | None
    quote_board_digest: str
    ranked_quote_ids: tuple[str, ...]

    def to_payload(self) -> Mapping[str, Any]:
        payload = {
            "schema": self.schema,
            "rfq_id": self.rfq_id,
            "quote_id": self.quote_id,
            "awarded_vendor_id": self.awarded_vendor_id,
            "action_intent_digest": self.action_intent_digest,
            "execution_status": self.execution_status,
            "tool_executed": self.tool_executed,
            "decision_authority": self.decision_authority,
            "decision_id": self.decision_id,
            "decision_receipt_digest": self.decision_receipt_digest,
            "authority_receipt_digest": self.authority_receipt_digest,
            "quote_board_digest": self.quote_board_digest,
            "ranked_quote_ids": list(self.ranked_quote_ids),
        }
        rendered = json.dumps(payload)
        for token in _FORBIDDEN:
            if token in rendered.casefold():
                raise ProcurementError(
                    "procurement proof must not contain secret fields"
                )
        return MappingProxyType(payload)


def build_procurement_proof(
    *,
    action: ActionIntent,
    execution: GovernedExecution,
    quote_board: Mapping[str, Any],
) -> ProcurementProof:
    """Offline/eval award proof. Never claims PrivateVault authorization."""

    bound = _bind_success_proof(action, execution, quote_board)
    return ProcurementProof(
        schema=PROCUREMENT_PROOF_SCHEMA,
        rfq_id=bound.rfq_id,
        quote_id=bound.quote_id,
        awarded_vendor_id=bound.awarded_vendor_id,
        action_intent_digest=_action_intent_digest(action),
        execution_status=execution.status.value,
        tool_executed=execution.tool_executed,
        decision_authority=_OFFLINE_AUTHORITY,
        decision_id=execution.decision_id,
        decision_receipt_digest=None,
        authority_receipt_digest=None,
        quote_board_digest=bound.quote_board_digest,
        ranked_quote_ids=bound.ranked_quote_ids,
    )


def build_privatevault_procurement_proof(
    *,
    action: ActionIntent,
    execution: GovernedExecution,
    quote_board: Mapping[str, Any],
    verified_closure: VerifiedClosure,
    verifier: PrivateVaultAgentDNAVerifier | None = None,
) -> ProcurementProof:
    """Award proof from a PrivateVault adapter VerifiedClosure only."""

    if not isinstance(verified_closure, VerifiedClosure):
        raise ProcurementError(
            "PrivateVault proofs require a VerifiedClosure from the adapter"
        )
    bound = _bind_success_proof(action, execution, quote_board)
    verified = _reverify_privatevault_closure(verified_closure, verifier=verifier)
    binding = verified.dispatch.authorization.binding
    if (
        binding.request_id != action.request_id
        or binding.request_id != execution.request_id
    ):
        raise ProcurementError("PrivateVault request_id does not match the action")
    if binding.action != action.privatevault_decide_payload():
        raise ProcurementError(
            "PrivateVault action digest does not match the ActionIntent"
        )
    authorization = verified.dispatch.authorization.authorization
    decision_id = authorization.get("decision_id")
    if not isinstance(decision_id, str) or not decision_id.strip():
        raise ProcurementError("PrivateVault authorization is missing decision_id")
    if execution.decision_id != decision_id:
        raise ProcurementError("PrivateVault decision_id does not match execution")
    _require_unexpired_authorization(authorization, at_time=binding.at_time)
    return ProcurementProof(
        schema=PROCUREMENT_PROOF_SCHEMA,
        rfq_id=bound.rfq_id,
        quote_id=bound.quote_id,
        awarded_vendor_id=bound.awarded_vendor_id,
        action_intent_digest=_action_intent_digest(action),
        execution_status=execution.status.value,
        tool_executed=execution.tool_executed,
        decision_authority=_PRIVATEVAULT_AUTHORITY,
        decision_id=decision_id,
        decision_receipt_digest=binding.decision_receipt_digest,
        authority_receipt_digest=binding.authority_receipt_digest,
        quote_board_digest=bound.quote_board_digest,
        ranked_quote_ids=bound.ranked_quote_ids,
    )


@dataclass(frozen=True, slots=True)
class _BoundSuccess:
    rfq_id: str
    quote_id: str
    awarded_vendor_id: str
    quote_board_digest: str
    ranked_quote_ids: tuple[str, ...]


def _action_intent_digest(action: ActionIntent) -> str:
    from cbrain.evaluation.company_harness import canonical_action_intent_digest

    return canonical_action_intent_digest(action)


def _bind_success_proof(
    action: ActionIntent,
    execution: GovernedExecution,
    quote_board: Mapping[str, Any],
) -> _BoundSuccess:
    if action.tool_name not in _SUCCESS_PROOF_TOOLS:
        raise ProcurementError(
            "procurement success proofs are limited to award, PR, and PO"
        )
    if (
        execution.status is not ExecutionStatus.EXECUTED
        or execution.tool_executed is not True
    ):
        raise ProcurementError("award proof requires EXECUTED tool execution")
    if action.request_id != execution.request_id:
        raise ProcurementError("proof action and execution request_id must match")
    output = execution.output
    if not isinstance(output, Mapping):
        raise ProcurementError("execution output is required for an award proof")
    ranked_ids = _ranked_quote_ids(quote_board)
    board_digest = quote_board_digest(quote_board)
    output_digest = output.get("quote_board_digest")
    if not isinstance(output_digest, str) or output_digest != board_digest:
        raise ProcurementError(
            "quote-board digest does not match verified execution output"
        )
    if action.tool_name == "release_purchase_order":
        pr_id = action.arguments.get("pr_id")
        if not isinstance(pr_id, str) or pr_id != output.get("pr_id"):
            raise ProcurementError("pr_id does not match verified execution output")
    rfq_id = _bound_text(action, output, quote_board, field_name="rfq_id")
    quote_id = _bound_text(action, output, quote_board, field_name="quote_id")
    awarded_vendor_id = _bound_vendor(action, output, quote_board)
    _bound_money(action, output, quote_board, quote_id=quote_id)
    if quote_id not in ranked_ids:
        raise ProcurementError("awarded quote_id is not on the verified quote board")
    for item in _quote_entries(quote_board):
        if str(item.get("rfq_id")) != rfq_id:
            raise ProcurementError("quote board rfq_id does not match the ActionIntent")
    return _BoundSuccess(
        rfq_id=rfq_id,
        quote_id=quote_id,
        awarded_vendor_id=awarded_vendor_id,
        quote_board_digest=board_digest,
        ranked_quote_ids=ranked_ids,
    )


def _bound_text(
    action: ActionIntent,
    output: Mapping[str, Any],
    quote_board: Mapping[str, Any],
    *,
    field_name: str,
) -> str:
    from_action = action.arguments.get(field_name)
    from_output = output.get(field_name)
    if action.tool_name == "release_purchase_order" and field_name in {
        "rfq_id",
        "quote_id",
    }:
        from_action = from_output
    if not isinstance(from_action, str) or not from_action.strip():
        raise ProcurementError(f"{field_name} is missing from the ActionIntent")
    if from_action != from_output:
        raise ProcurementError(f"{field_name} does not match verified execution output")
    if field_name == "rfq_id":
        board_rfq = quote_board.get("rfq_id")
        if board_rfq is not None and board_rfq != from_action:
            raise ProcurementError("quote board rfq_id does not match the ActionIntent")
    return from_action


def _bound_vendor(
    action: ActionIntent,
    output: Mapping[str, Any],
    quote_board: Mapping[str, Any],
) -> str:
    vendor_id = output.get("vendor_id")
    if not isinstance(vendor_id, str) or not vendor_id.strip():
        raise ProcurementError(
            "awarded_vendor_id is missing from verified execution output"
        )
    argument_vendor = action.arguments.get("vendor_id")
    if argument_vendor is not None and argument_vendor != vendor_id:
        raise ProcurementError("awarded_vendor_id does not match the ActionIntent")
    quote_id = output.get("quote_id")
    for item in _quote_entries(quote_board):
        if item.get("quote_id") == quote_id and item.get("vendor_id") != vendor_id:
            raise ProcurementError("awarded_vendor_id does not match the quote board")
    return vendor_id


def _bound_money(
    action: ActionIntent,
    output: Mapping[str, Any],
    quote_board: Mapping[str, Any],
    *,
    quote_id: str,
) -> None:
    amount = action.arguments.get("amount_minor")
    currency = action.arguments.get("currency")
    if amount != output.get("amount_minor") or currency != output.get("currency"):
        raise ProcurementError(
            "amount or currency does not match verified execution output"
        )
    for item in _quote_entries(quote_board):
        if item.get("quote_id") != quote_id:
            continue
        if item.get("amount_minor") != amount or item.get("currency") != currency:
            raise ProcurementError("amount or currency does not match the quote board")
        return
    raise ProcurementError("awarded quote_id is not on the verified quote board")


def _ranked_quote_ids(quote_board: Mapping[str, Any]) -> tuple[str, ...]:
    ranked: list[str] = []
    for item in _quote_entries(quote_board):
        quote_id = item.get("quote_id")
        if not isinstance(quote_id, str) or not quote_id.strip():
            raise ProcurementError("quote board entries must include quote_id")
        ranked.append(quote_id)
    return tuple(ranked)


def _quote_entries(quote_board: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    quotations = quote_board.get("quotations")
    if not isinstance(quotations, list) or not quotations:
        raise ProcurementError("quote board must include quotations")
    entries: list[Mapping[str, Any]] = []
    for item in quotations:
        if not isinstance(item, Mapping):
            raise ProcurementError("quote board entries must include quote_id")
        entries.append(item)
    return entries


def _reverify_privatevault_closure(
    verified_closure: VerifiedClosure,
    *,
    verifier: PrivateVaultAgentDNAVerifier | None,
) -> VerifiedClosure:
    adapter = verifier or PrivateVaultAgentDNAVerifier()
    authorization = verified_closure.dispatch.authorization
    if not authorization.trust_bundle:
        raise ProcurementError("PrivateVault trust bundle is required")
    rebound = adapter.verify_authorization(
        authorization=authorization.authorization,
        trust_bundle=authorization.trust_bundle,
        binding=authorization.binding,
        already_consumed=True,
    )
    redispatched = adapter.verify_dispatch(
        verified_authorization=rebound,
        witness=verified_closure.dispatch.witness,
    )
    return adapter.verify_closure(
        verified_dispatch=redispatched,
        closure=verified_closure.closure,
    )


def _require_unexpired_authorization(
    authorization: Mapping[str, Any], *, at_time: str
) -> None:
    expiry = authorization.get("expires_at")
    if expiry is None:
        expiry = authorization.get("not_after")
    if expiry is None:
        raise ProcurementError("PrivateVault authorization is missing expiry")
    if not isinstance(expiry, str) or not expiry.strip():
        raise ProcurementError("PrivateVault authorization expiry is invalid")
    if expiry <= at_time:
        raise ProcurementError("PrivateVault authorization has expired")


__all__ = [
    "PROCUREMENT_PROOF_SCHEMA",
    "ProcurementProof",
    "build_privatevault_procurement_proof",
    "build_procurement_proof",
]
