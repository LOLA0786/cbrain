"""Tool risk classification for company agents."""

from __future__ import annotations

from enum import StrEnum

from .kinds import CompanyAgentKind


class ToolRiskLevel(StrEnum):
    ALLOW = "allow"
    REVIEW = "review"
    BLOCK = "block"


_GTM_RISKS: dict[str, ToolRiskLevel] = {
    "search_crm": ToolRiskLevel.ALLOW,
    "research_account": ToolRiskLevel.ALLOW,
    "qualify_lead": ToolRiskLevel.ALLOW,
    "detect_duplicate_leads": ToolRiskLevel.ALLOW,
    "draft_outreach": ToolRiskLevel.ALLOW,
    "update_crm_record": ToolRiskLevel.ALLOW,
    "prepare_email": ToolRiskLevel.REVIEW,
    "send_email": ToolRiskLevel.REVIEW,
    "bulk_campaign": ToolRiskLevel.BLOCK,
    "delete_lead": ToolRiskLevel.BLOCK,
}

_OPERATIONS_RISKS: dict[str, ToolRiskLevel] = {
    "check_service_health": ToolRiskLevel.ALLOW,
    "search_runbooks": ToolRiskLevel.ALLOW,
    "triage_ticket": ToolRiskLevel.ALLOW,
    "assign_severity": ToolRiskLevel.ALLOW,
    "create_incident_draft": ToolRiskLevel.ALLOW,
    "prepare_escalation": ToolRiskLevel.ALLOW,
    "propose_restart": ToolRiskLevel.REVIEW,
    "propose_infra_change": ToolRiskLevel.REVIEW,
    "delete_data": ToolRiskLevel.BLOCK,
}

_LEGAL_RISKS: dict[str, ToolRiskLevel] = {
    "search_contracts": ToolRiskLevel.ALLOW,
    "extract_clause": ToolRiskLevel.ALLOW,
    "compare_contracts": ToolRiskLevel.ALLOW,
    "identify_deviations": ToolRiskLevel.ALLOW,
    "prepare_redline": ToolRiskLevel.ALLOW,
    "draft_legal_summary": ToolRiskLevel.ALLOW,
    "sign_contract": ToolRiskLevel.BLOCK,
    "file_document": ToolRiskLevel.BLOCK,
    "send_commitment": ToolRiskLevel.REVIEW,
    "provide_legal_advice": ToolRiskLevel.BLOCK,
}

_CODING_RISKS: dict[str, ToolRiskLevel] = {
    "search_repo": ToolRiskLevel.ALLOW,
    "read_file": ToolRiskLevel.ALLOW,
    "run_tests": ToolRiskLevel.ALLOW,
    "propose_patch": ToolRiskLevel.REVIEW,
    "apply_patch": ToolRiskLevel.REVIEW,
    "open_pull_request": ToolRiskLevel.REVIEW,
    "force_push": ToolRiskLevel.BLOCK,
    "write_secret": ToolRiskLevel.BLOCK,
}

_PROCUREMENT_RISKS: dict[str, ToolRiskLevel] = {
    "lookup_vendor": ToolRiskLevel.ALLOW,
    "search_catalog": ToolRiskLevel.ALLOW,
    "list_registered_vendors": ToolRiskLevel.ALLOW,
    "list_open_requisitions": ToolRiskLevel.ALLOW,
    "send_rfq_email": ToolRiskLevel.ALLOW,
    "show_quotations": ToolRiskLevel.ALLOW,
    "create_purchase_requisition": ToolRiskLevel.REVIEW,
    "award_quote": ToolRiskLevel.REVIEW,
    "release_purchase_order": ToolRiskLevel.REVIEW,
    "change_vendor_bank": ToolRiskLevel.BLOCK,
    "post_erp_payment": ToolRiskLevel.BLOCK,
}

_ACCOUNTS_RISKS: dict[str, ToolRiskLevel] = {
    "read_invoices": ToolRiskLevel.ALLOW,
    "extract_invoice_data": ToolRiskLevel.ALLOW,
    "reconcile_records": ToolRiskLevel.ALLOW,
    "detect_duplicate_invoices": ToolRiskLevel.ALLOW,
    "prepare_aging_report": ToolRiskLevel.ALLOW,
    "inspect_ledger": ToolRiskLevel.ALLOW,
    "prepare_payment": ToolRiskLevel.REVIEW,
    "prepare_refund": ToolRiskLevel.REVIEW,
    "execute_payment": ToolRiskLevel.REVIEW,
    "change_bank_details": ToolRiskLevel.BLOCK,
    "change_vendor": ToolRiskLevel.BLOCK,
}


def risk_for_tool(kind: CompanyAgentKind, tool_name: str) -> ToolRiskLevel:
    table = _RISK_TABLE[kind]
    try:
        return table[tool_name]
    except KeyError as exc:
        raise ValueError(f"unknown tool {tool_name!r} for {kind.value}") from exc


_RISK_TABLE: dict[CompanyAgentKind, dict[str, ToolRiskLevel]] = {
    CompanyAgentKind.GTM: _GTM_RISKS,
    CompanyAgentKind.OPERATIONS: _OPERATIONS_RISKS,
    CompanyAgentKind.LEGAL: _LEGAL_RISKS,
    CompanyAgentKind.ACCOUNTS: _ACCOUNTS_RISKS,
    CompanyAgentKind.CODING: _CODING_RISKS,
    CompanyAgentKind.PROCUREMENT: _PROCUREMENT_RISKS,
}


__all__ = ["ToolRiskLevel", "risk_for_tool"]
