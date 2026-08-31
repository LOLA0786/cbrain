"""Purchasing proof pack: quote board plus optional PrivateVault receipts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from cbrain.contracts import ActionIntent, GovernedExecution

from .erp import ProcurementError

PROCUREMENT_PROOF_SCHEMA = "cbrain-procurement-proof/v1"
_RECEIPT = re.compile(r"^sha256:[0-9a-f]{64}$")
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


def _canonical(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _optional_receipt(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _RECEIPT.fullmatch(value):
        raise ProcurementError(f"{field_name} must be a sha256 receipt digest")
    return value


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
    rfq_id: str,
    quote_board: Mapping[str, Any],
    quote_id: str | None = None,
    awarded_vendor_id: str | None = None,
    decision_authority: str,
    action_intent_digest: str,
    decision_receipt_digest: str | None = None,
    authority_receipt_digest: str | None = None,
) -> ProcurementProof:
    if not rfq_id.strip() or not action_intent_digest.strip():
        raise ProcurementError("rfq_id and action_intent_digest are required")
    if not decision_authority.strip():
        raise ProcurementError("decision_authority is required")
    if decision_authority == "privatevault" and (
        decision_receipt_digest is None or authority_receipt_digest is None
    ):
        raise ProcurementError("privatevault proofs require both receipt digests")
    if decision_authority == "company_test_gateway" and (
        decision_receipt_digest is not None or authority_receipt_digest is not None
    ):
        raise ProcurementError(
            "company_test_gateway must not claim PrivateVault receipts"
        )
    quotations = quote_board.get("quotations")
    if not isinstance(quotations, list):
        raise ProcurementError("quote board must include quotations")
    ranked_ids: list[str] = []
    for item in quotations:
        if not isinstance(item, Mapping) or not isinstance(item.get("quote_id"), str):
            raise ProcurementError("quote board entries must include quote_id")
        ranked_ids.append(str(item["quote_id"]))
    if action.request_id != execution.request_id:
        raise ProcurementError("proof action and execution request_id must match")
    return ProcurementProof(
        schema=PROCUREMENT_PROOF_SCHEMA,
        rfq_id=rfq_id,
        quote_id=quote_id,
        awarded_vendor_id=awarded_vendor_id,
        action_intent_digest=action_intent_digest,
        execution_status=execution.status.value,
        tool_executed=execution.tool_executed,
        decision_authority=decision_authority,
        decision_id=execution.decision_id,
        decision_receipt_digest=_optional_receipt(
            decision_receipt_digest, field_name="decision_receipt_digest"
        ),
        authority_receipt_digest=_optional_receipt(
            authority_receipt_digest, field_name="authority_receipt_digest"
        ),
        quote_board_digest=_digest(quote_board),
        ranked_quote_ids=tuple(ranked_ids),
    )


__all__ = [
    "PROCUREMENT_PROOF_SCHEMA",
    "ProcurementProof",
    "build_procurement_proof",
]
