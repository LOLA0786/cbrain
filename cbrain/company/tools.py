"""Governed tool definitions for company agents."""

from __future__ import annotations

from collections.abc import Iterable

from cbrain.agent import GovernedTool, ToolRegistry, ToolRegistryError

from .kinds import CompanyAgentKind

_STRING = {"type": "string"}
_OBJECT = {"type": "object"}


def _tool(
    name: str,
    capability: str,
    description: str,
    *,
    properties: dict[str, object],
    required: list[str],
) -> GovernedTool:
    return GovernedTool(
        name=name,
        capability=capability,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    )


_GTM_TOOLS: tuple[GovernedTool, ...] = (
    _tool(
        "search_crm",
        "company.gtm.crm.search",
        "Search simulated CRM records",
        properties={"query": _STRING, "limit": {"type": "integer"}},
        required=["query"],
    ),
    _tool(
        "research_account",
        "company.gtm.account.research",
        "Research supplied account fixture evidence",
        properties={"account_id": _STRING},
        required=["account_id"],
    ),
    _tool(
        "qualify_lead",
        "company.gtm.lead.qualify",
        "Qualify a lead using supplied evidence",
        properties={"lead_id": _STRING, "score": {"type": "integer"}},
        required=["lead_id", "score"],
    ),
    _tool(
        "detect_duplicate_leads",
        "company.gtm.lead.duplicate_detect",
        "Detect duplicate leads in simulated CRM",
        properties={"email": _STRING},
        required=["email"],
    ),
    _tool(
        "draft_outreach",
        "company.gtm.outreach.draft",
        "Draft personalized outreach from evidence",
        properties={"lead_id": _STRING, "template_id": _STRING},
        required=["lead_id", "template_id"],
    ),
    _tool(
        "update_crm_record",
        "company.gtm.crm.update",
        "Create or update a simulated CRM record",
        properties={"record_id": _STRING, "fields": _OBJECT},
        required=["record_id", "fields"],
    ),
    _tool(
        "prepare_email",
        "company.gtm.email.prepare",
        "Prepare an email draft for human review",
        properties={
            "recipient": _STRING,
            "subject": _STRING,
            "body": _STRING,
        },
        required=["recipient", "subject", "body"],
    ),
    _tool(
        "send_email",
        "company.gtm.email.send",
        "Send an external email",
        properties={
            "recipient": _STRING,
            "subject": _STRING,
            "body": _STRING,
        },
        required=["recipient", "subject", "body"],
    ),
    _tool(
        "bulk_campaign",
        "company.gtm.campaign.bulk",
        "Launch a bulk email campaign",
        properties={"campaign_id": _STRING, "recipients": {"type": "array"}},
        required=["campaign_id", "recipients"],
    ),
    _tool(
        "delete_lead",
        "company.gtm.lead.delete",
        "Delete a CRM lead",
        properties={"lead_id": _STRING},
        required=["lead_id"],
    ),
)

_OPERATIONS_TOOLS: tuple[GovernedTool, ...] = (
    _tool(
        "check_service_health",
        "company.ops.health.check",
        "Inspect simulated service health",
        properties={"service_id": _STRING},
        required=["service_id"],
    ),
    _tool(
        "search_runbooks",
        "company.ops.runbook.search",
        "Search supplied runbooks",
        properties={"query": _STRING},
        required=["query"],
    ),
    _tool(
        "triage_ticket",
        "company.ops.ticket.triage",
        "Triage a support ticket",
        properties={"ticket_id": _STRING, "summary": _STRING},
        required=["ticket_id", "summary"],
    ),
    _tool(
        "assign_severity",
        "company.ops.incident.severity",
        "Assign incident severity",
        properties={"ticket_id": _STRING, "severity": _STRING},
        required=["ticket_id", "severity"],
    ),
    _tool(
        "create_incident_draft",
        "company.ops.incident.draft",
        "Create an incident draft",
        properties={"title": _STRING, "severity": _STRING},
        required=["title", "severity"],
    ),
    _tool(
        "prepare_escalation",
        "company.ops.escalation.prepare",
        "Prepare an escalation notice",
        properties={"ticket_id": _STRING, "channel": _STRING},
        required=["ticket_id", "channel"],
    ),
    _tool(
        "propose_restart",
        "company.ops.service.restart",
        "Propose restarting a service",
        properties={"service_id": _STRING},
        required=["service_id"],
    ),
    _tool(
        "propose_infra_change",
        "company.ops.infra.change",
        "Propose an infrastructure change",
        properties={"change_id": _STRING, "description": _STRING},
        required=["change_id", "description"],
    ),
    _tool(
        "delete_data",
        "company.ops.data.delete",
        "Delete operational data",
        properties={"resource_id": _STRING},
        required=["resource_id"],
    ),
)

_LEGAL_TOOLS: tuple[GovernedTool, ...] = (
    _tool(
        "search_contracts",
        "company.legal.contract.search",
        "Search supplied contract fixtures",
        properties={"matter_id": _STRING, "query": _STRING},
        required=["matter_id", "query"],
    ),
    _tool(
        "extract_clause",
        "company.legal.clause.extract",
        "Extract a clause with citation",
        properties={"contract_id": _STRING, "clause_id": _STRING},
        required=["contract_id", "clause_id"],
    ),
    _tool(
        "compare_contracts",
        "company.legal.contract.compare",
        "Compare contract versions",
        properties={"left_id": _STRING, "right_id": _STRING},
        required=["left_id", "right_id"],
    ),
    _tool(
        "identify_deviations",
        "company.legal.playbook.deviations",
        "Identify deviations from approved playbook",
        properties={"contract_id": _STRING, "playbook_id": _STRING},
        required=["contract_id", "playbook_id"],
    ),
    _tool(
        "prepare_redline",
        "company.legal.redline.prepare",
        "Prepare a source-grounded redline",
        properties={"contract_id": _STRING, "changes": {"type": "array"}},
        required=["contract_id", "changes"],
    ),
    _tool(
        "draft_legal_summary",
        "company.legal.summary.draft",
        "Draft a source-grounded legal summary",
        properties={"contract_id": _STRING, "sections": {"type": "array"}},
        required=["contract_id"],
    ),
    _tool(
        "sign_contract",
        "company.legal.contract.sign",
        "Sign a contract",
        properties={"contract_id": _STRING},
        required=["contract_id"],
    ),
    _tool(
        "file_document",
        "company.legal.document.file",
        "File a legal document",
        properties={"document_id": _STRING, "court": _STRING},
        required=["document_id", "court"],
    ),
    _tool(
        "send_commitment",
        "company.legal.commitment.send",
        "Send a legal commitment externally",
        properties={"matter_id": _STRING, "text": _STRING},
        required=["matter_id", "text"],
    ),
    _tool(
        "provide_legal_advice",
        "company.legal.advice.provide",
        "Provide final legal advice",
        properties={"matter_id": _STRING, "conclusion": _STRING},
        required=["matter_id", "conclusion"],
    ),
)

_ACCOUNTS_TOOLS: tuple[GovernedTool, ...] = (
    _tool(
        "read_invoices",
        "company.accounts.invoice.read",
        "Read supplied invoice fixtures",
        properties={"vendor_id": _STRING},
        required=["vendor_id"],
    ),
    _tool(
        "extract_invoice_data",
        "company.accounts.invoice.extract",
        "Extract structured invoice data",
        properties={"invoice_id": _STRING},
        required=["invoice_id"],
    ),
    _tool(
        "reconcile_records",
        "company.accounts.reconcile",
        "Reconcile invoice and purchase-order records",
        properties={"invoice_id": _STRING, "po_id": _STRING},
        required=["invoice_id", "po_id"],
    ),
    _tool(
        "detect_duplicate_invoices",
        "company.accounts.invoice.duplicate_detect",
        "Detect duplicate invoices",
        properties={"invoice_number": _STRING},
        required=["invoice_number"],
    ),
    _tool(
        "prepare_aging_report",
        "company.accounts.aging.prepare",
        "Prepare an aging report",
        properties={"as_of_date": _STRING},
        required=["as_of_date"],
    ),
    _tool(
        "inspect_ledger",
        "company.accounts.ledger.inspect",
        "Inspect simulated ledger state",
        properties={"account_id": _STRING},
        required=["account_id"],
    ),
    _tool(
        "prepare_payment",
        "company.accounts.payment.prepare",
        "Prepare a payment proposal",
        properties={
            "invoice_id": _STRING,
            "amount_minor": _STRING,
            "currency": _STRING,
        },
        required=["invoice_id", "amount_minor", "currency"],
    ),
    _tool(
        "prepare_refund",
        "company.accounts.refund.prepare",
        "Prepare a refund proposal",
        properties={
            "invoice_id": _STRING,
            "amount_minor": _STRING,
            "currency": _STRING,
        },
        required=["invoice_id", "amount_minor", "currency"],
    ),
    _tool(
        "execute_payment",
        "company.accounts.payment.execute",
        "Execute a payment against simulated ledger",
        properties={
            "invoice_id": _STRING,
            "amount_minor": _STRING,
            "currency": _STRING,
            "beneficiary_id": _STRING,
        },
        required=["invoice_id", "amount_minor", "currency", "beneficiary_id"],
    ),
    _tool(
        "change_bank_details",
        "company.accounts.bank.change",
        "Change vendor bank details",
        properties={"vendor_id": _STRING, "account_ref": _STRING},
        required=["vendor_id", "account_ref"],
    ),
    _tool(
        "change_vendor",
        "company.accounts.vendor.change",
        "Change vendor identity",
        properties={"vendor_id": _STRING, "new_name": _STRING},
        required=["vendor_id", "new_name"],
    ),
)

_CODING_TOOLS: tuple[GovernedTool, ...] = (
    _tool(
        "search_repo",
        "company.coding.repo.search",
        "Search the simulated repository index",
        properties={"query": _STRING},
        required=["query"],
    ),
    _tool(
        "read_file",
        "company.coding.file.read",
        "Read a simulated source file",
        properties={"path": _STRING},
        required=["path"],
    ),
    _tool(
        "run_tests",
        "company.coding.tests.run",
        "Run the simulated test suite",
        properties={"suite": _STRING},
        required=["suite"],
    ),
    _tool(
        "propose_patch",
        "company.coding.patch.propose",
        "Propose a source patch for human review",
        properties={"path": _STRING, "diff": _STRING},
        required=["path", "diff"],
    ),
    _tool(
        "apply_patch",
        "company.coding.patch.apply",
        "Apply an approved patch to the simulated workspace",
        properties={"path": _STRING, "diff": _STRING},
        required=["path", "diff"],
    ),
    _tool(
        "open_pull_request",
        "company.coding.pr.open",
        "Open a pull request after review",
        properties={"title": _STRING, "body": _STRING, "head": _STRING},
        required=["title", "body", "head"],
    ),
    _tool(
        "force_push",
        "company.coding.git.force_push",
        "Force-push to a protected branch",
        properties={"ref": _STRING},
        required=["ref"],
    ),
    _tool(
        "write_secret",
        "company.coding.secret.write",
        "Write a credential into the workspace",
        properties={"path": _STRING, "name": _STRING},
        required=["path", "name"],
    ),
)

_PROCUREMENT_TOOLS: tuple[GovernedTool, ...] = (
    _tool(
        "lookup_vendor",
        "company.procurement.vendor.lookup",
        "Look up a vendor from the ERP replica extract",
        properties={"vendor_id": _STRING},
        required=["vendor_id"],
    ),
    _tool(
        "search_catalog",
        "company.procurement.catalog.search",
        "Search replica catalog materials",
        properties={"query": _STRING},
        required=["query"],
    ),
    _tool(
        "list_registered_vendors",
        "company.procurement.vendor.list_registered",
        "List vendors registered to receive RFQs",
        properties={"material_id": _STRING},
        required=[],
    ),
    _tool(
        "list_open_requisitions",
        "company.procurement.requisition.list",
        "List open purchase requisitions and replica POs",
        properties={"vendor_id": _STRING},
        required=[],
    ),
    _tool(
        "send_rfq_email",
        "company.procurement.rfq.send",
        "Send an RFQ email to registered vendor IDs and return quotations instantly",
        properties={
            "rfq_id": _STRING,
            "material_id": _STRING,
            "vendor_ids": {"type": "array", "items": {"type": "string"}},
        },
        required=["rfq_id", "material_id", "vendor_ids"],
    ),
    _tool(
        "show_quotations",
        "company.procurement.quote.show",
        "Show the ranked quotation board for an RFQ",
        properties={"rfq_id": _STRING},
        required=["rfq_id"],
    ),
    _tool(
        "create_purchase_requisition",
        "company.procurement.requisition.create",
        "Create a purchase requisition from a quoted award",
        properties={
            "rfq_id": _STRING,
            "quote_id": _STRING,
            "material_id": _STRING,
            "quantity": {"type": "integer"},
            "amount_minor": _STRING,
            "currency": _STRING,
        },
        required=[
            "rfq_id",
            "quote_id",
            "material_id",
            "quantity",
            "amount_minor",
            "currency",
        ],
    ),
    _tool(
        "award_quote",
        "company.procurement.quote.award",
        "Award a quotation after human buyer-lead approval",
        properties={
            "rfq_id": _STRING,
            "quote_id": _STRING,
            "amount_minor": _STRING,
            "currency": _STRING,
        },
        required=["rfq_id", "quote_id", "amount_minor", "currency"],
    ),
    _tool(
        "release_purchase_order",
        "company.procurement.po.release",
        "Release a purchase order to the ERP after human approval",
        properties={
            "pr_id": _STRING,
            "amount_minor": _STRING,
            "currency": _STRING,
        },
        required=["pr_id", "amount_minor", "currency"],
    ),
    _tool(
        "change_vendor_bank",
        "company.procurement.vendor.bank.change",
        "Change vendor bank details in ERP",
        properties={"vendor_id": _STRING, "account_ref": _STRING},
        required=["vendor_id", "account_ref"],
    ),
    _tool(
        "post_erp_payment",
        "company.procurement.payment.post",
        "Post a payment in Oracle, SAP, or SQL Server",
        properties={
            "vendor_id": _STRING,
            "amount_minor": _STRING,
            "currency": _STRING,
        },
        required=["vendor_id", "amount_minor", "currency"],
    ),
)

_ALL_TOOLS: tuple[GovernedTool, ...] = (
    _GTM_TOOLS
    + _OPERATIONS_TOOLS
    + _LEGAL_TOOLS
    + _ACCOUNTS_TOOLS
    + _CODING_TOOLS
    + _PROCUREMENT_TOOLS
)

_KIND_TOOLS: dict[CompanyAgentKind, tuple[GovernedTool, ...]] = {
    CompanyAgentKind.GTM: _GTM_TOOLS,
    CompanyAgentKind.OPERATIONS: _OPERATIONS_TOOLS,
    CompanyAgentKind.LEGAL: _LEGAL_TOOLS,
    CompanyAgentKind.ACCOUNTS: _ACCOUNTS_TOOLS,
    CompanyAgentKind.CODING: _CODING_TOOLS,
    CompanyAgentKind.PROCUREMENT: _PROCUREMENT_TOOLS,
}


def tools_for_kind(kind: CompanyAgentKind) -> Iterable[GovernedTool]:
    return _KIND_TOOLS[kind]


def company_tool_registry() -> _CompanyToolRegistry:
    return _CompanyToolRegistry(_ALL_TOOLS)


class _CompanyToolRegistry:
    def __init__(self, tools: Iterable[GovernedTool]) -> None:
        self._registry = ToolRegistry(tools)

    def validate_permitted(self, permitted: frozenset[str]) -> None:
        self._registry.definitions_for(permitted)

    def registry_for(self, permitted: frozenset[str]) -> ToolRegistry:
        missing = permitted - set(self._registry._tools)
        if missing:
            raise ToolRegistryError(f"unknown tools: {sorted(missing)}")
        return ToolRegistry(self._registry._tools[name] for name in sorted(permitted))


__all__ = ["company_tool_registry", "tools_for_kind"]
