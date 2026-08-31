"""Simulated tool handlers for offline company agents."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from decimal import Decimal
from typing import Any

from cbrain.procurement.erp import ProcurementError
from cbrain.procurement.mailbox import (
    Quotation,
    quotation_board_payload,
    quote_board_digest,
)

from .authority import CompanyExecutionContext
from .kinds import CompanyAgentKind
from .simulators import (
    AccountsSimulator,
    CodingSimulator,
    CompanySimulatorBundle,
    GTMSimulator,
    LegalSimulator,
    OperationsSimulator,
    ProcurementSimulator,
    SimulatorValidationError,
    parse_decimal_minor,
)


def build_handlers(
    *,
    kind: CompanyAgentKind,
    bundle: CompanySimulatorBundle,
    context: CompanyExecutionContext | None = None,
) -> dict[str, Callable[[Mapping[str, Any]], Any]]:
    if kind is CompanyAgentKind.GTM:
        return _gtm_handlers(bundle.gtm)
    if kind is CompanyAgentKind.OPERATIONS:
        return _operations_handlers(bundle.operations)
    if kind is CompanyAgentKind.LEGAL:
        return _legal_handlers(bundle.legal, context=context)
    if kind is CompanyAgentKind.CODING:
        return _coding_handlers(bundle.coding)
    if kind is CompanyAgentKind.PROCUREMENT:
        return _procurement_handlers(bundle.procurement)
    return _accounts_handlers(bundle.accounts)


def _gtm_handlers(sim: GTMSimulator) -> dict[str, Callable[[Mapping[str, Any]], Any]]:
    def search_crm(arguments: Mapping[str, Any]) -> dict[str, Any]:
        query = str(arguments["query"]).lower()
        matches = [
            lead
            for lead in sim.leads.values()
            if query in str(lead.get("email", "")).lower()
            or query in str(lead.get("company", "")).lower()
        ]
        return {"matches": matches[: int(arguments.get("limit", 10))]}

    def research_account(arguments: Mapping[str, Any]) -> dict[str, Any]:
        account_id = str(arguments["account_id"])
        lead = sim.leads.get(account_id)
        if lead is None:
            return {"found": False, "evidence": []}
        return {"found": True, "evidence": [lead.get("evidence", "")]}

    def qualify_lead(arguments: Mapping[str, Any]) -> dict[str, Any]:
        lead_id = str(arguments["lead_id"])
        if lead_id not in sim.leads:
            raise SimulatorValidationError("unknown lead")
        return {"lead_id": lead_id, "score": int(arguments["score"]), "qualified": True}

    def detect_duplicate_leads(arguments: Mapping[str, Any]) -> dict[str, Any]:
        email = str(arguments["email"]).lower()
        dupes = [
            lead["lead_id"]
            for lead in sim.leads.values()
            if str(lead.get("email", "")).lower() == email
        ]
        return {"email": email, "duplicate": len(dupes) > 1, "lead_ids": dupes}

    def draft_outreach(arguments: Mapping[str, Any]) -> dict[str, Any]:
        lead_id = str(arguments["lead_id"])
        lead = sim.leads.get(lead_id)
        if lead is None:
            raise SimulatorValidationError("unknown lead")
        return {
            "lead_id": lead_id,
            "template_id": str(arguments["template_id"]),
            "draft": f"Hello {lead.get('company', 'there')}",
            "grounded": True,
        }

    def update_crm_record(arguments: Mapping[str, Any]) -> dict[str, Any]:
        record_id = str(arguments["record_id"])
        fields = dict(arguments["fields"])
        with sim._lock:
            sim.crm_updates.append({"record_id": record_id, "fields": fields})
            sim.leads.setdefault(record_id, {"lead_id": record_id})
            sim.leads[record_id].update(fields)
        return {"record_id": record_id, "updated": True}

    def prepare_email(arguments: Mapping[str, Any]) -> dict[str, Any]:
        recipient = str(arguments["recipient"])
        if recipient in sim.opt_outs:
            raise SimulatorValidationError("recipient opted out")
        payload = {
            "recipient": recipient,
            "subject": str(arguments["subject"]),
            "body": str(arguments["body"]),
            "prepared": True,
        }
        with sim._lock:
            sim.emails_prepared.append(payload)
        return payload

    def send_email(arguments: Mapping[str, Any]) -> dict[str, Any]:
        recipient = str(arguments["recipient"])
        if recipient in sim.opt_outs:
            raise SimulatorValidationError("recipient opted out")
        payload = {
            "recipient": recipient,
            "subject": str(arguments["subject"]),
            "body": str(arguments["body"]),
            "sent": True,
        }
        with sim._lock:
            sim.emails_sent.append(payload)
        return payload

    def bulk_campaign(arguments: Mapping[str, Any]) -> dict[str, Any]:
        return {"campaign_id": str(arguments["campaign_id"]), "launched": True}

    def delete_lead(arguments: Mapping[str, Any]) -> dict[str, Any]:
        lead_id = str(arguments["lead_id"])
        with sim._lock:
            sim.leads.pop(lead_id, None)
        return {"lead_id": lead_id, "deleted": True}

    return {
        "search_crm": search_crm,
        "research_account": research_account,
        "qualify_lead": qualify_lead,
        "detect_duplicate_leads": detect_duplicate_leads,
        "draft_outreach": draft_outreach,
        "update_crm_record": update_crm_record,
        "prepare_email": prepare_email,
        "send_email": send_email,
        "bulk_campaign": bulk_campaign,
        "delete_lead": delete_lead,
    }


def _operations_handlers(
    sim: OperationsSimulator,
) -> dict[str, Callable[[Mapping[str, Any]], Any]]:
    def check_service_health(arguments: Mapping[str, Any]) -> dict[str, Any]:
        service_id = str(arguments["service_id"])
        return {
            "service_id": service_id,
            "status": sim.services.get(service_id, "unknown"),
        }

    def search_runbooks(arguments: Mapping[str, Any]) -> dict[str, Any]:
        query = str(arguments["query"]).lower()
        matches = [
            rb
            for rb in sim.runbooks.values()
            if query in str(rb.get("title", "")).lower()
        ]
        return {"matches": matches}

    def triage_ticket(arguments: Mapping[str, Any]) -> dict[str, Any]:
        ticket_id = str(arguments["ticket_id"])
        return {"ticket_id": ticket_id, "summary": str(arguments["summary"])}

    def assign_severity(arguments: Mapping[str, Any]) -> dict[str, Any]:
        ticket_id = str(arguments["ticket_id"])
        severity = str(arguments["severity"])
        if ticket_id in sim.tickets:
            sim.tickets[ticket_id]["severity"] = severity
        return {"ticket_id": ticket_id, "severity": severity}

    def create_incident_draft(arguments: Mapping[str, Any]) -> dict[str, Any]:
        draft = {
            "title": str(arguments["title"]),
            "severity": str(arguments["severity"]),
        }
        with sim._lock:
            sim.incidents.append(draft)
        return draft

    def prepare_escalation(arguments: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "ticket_id": str(arguments["ticket_id"]),
            "channel": str(arguments["channel"]),
            "prepared": True,
        }

    def propose_restart(arguments: Mapping[str, Any]) -> dict[str, Any]:
        service_id = str(arguments["service_id"])
        with sim._lock:
            sim.restarts.append(service_id)
        return {"service_id": service_id, "restarted": True}

    def propose_infra_change(arguments: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "change_id": str(arguments["change_id"]),
            "description": str(arguments["description"]),
            "proposed": True,
        }

    def delete_data(arguments: Mapping[str, Any]) -> dict[str, Any]:
        return {"resource_id": str(arguments["resource_id"]), "deleted": True}

    return {
        "check_service_health": check_service_health,
        "search_runbooks": search_runbooks,
        "triage_ticket": triage_ticket,
        "assign_severity": assign_severity,
        "create_incident_draft": create_incident_draft,
        "prepare_escalation": prepare_escalation,
        "propose_restart": propose_restart,
        "propose_infra_change": propose_infra_change,
        "delete_data": delete_data,
    }


def _legal_handlers(
    sim: LegalSimulator,
    *,
    context: CompanyExecutionContext | None,
) -> dict[str, Callable[[Mapping[str, Any]], Any]]:
    def _permitted() -> frozenset[str]:
        if context is None:
            raise SimulatorValidationError("legal matter scope required")
        return context.permitted_matter_ids

    def _authorized_contracts() -> set[str]:
        ids: set[str] = set()
        for matter_id in _permitted():
            ids.update(sim.matters.get(matter_id, set()))
        return ids

    def _require_contract(contract_id: str) -> dict[str, Any]:
        if contract_id not in _authorized_contracts():
            raise SimulatorValidationError("contract outside authorized matter scope")
        contract = sim.contracts.get(contract_id)
        if contract is None:
            raise SimulatorValidationError("contract outside authorized matter scope")
        with sim._lock:
            sim.access_log.append(
                {
                    "matter_id": str(contract["matter_id"]),
                    "contract_id": contract_id,
                }
            )
        return contract

    def search_contracts(arguments: Mapping[str, Any]) -> dict[str, Any]:
        matter_id = str(arguments["matter_id"])
        if matter_id not in _permitted():
            raise SimulatorValidationError("matter outside authorized scope")
        query = str(arguments["query"]).lower()
        allowed = sim.matters.get(matter_id, set()) & _authorized_contracts()
        matches = [
            sim.contracts[cid]
            for cid in allowed
            if cid in sim.contracts and query in str(sim.contracts[cid]).lower()
        ]
        return {"matter_id": matter_id, "matches": matches}

    def extract_clause(arguments: Mapping[str, Any]) -> dict[str, Any]:
        contract_id = str(arguments["contract_id"])
        clause_id = str(arguments["clause_id"])
        contract = _require_contract(contract_id)
        clauses = contract.get("clauses", {})
        if clause_id not in clauses:
            raise SimulatorValidationError("unknown clause")
        citation = f"{contract_id}:{clause_id}"
        return {
            "contract_id": contract_id,
            "clause_id": clause_id,
            "text": clauses[clause_id],
            "citation": citation,
        }

    def compare_contracts(arguments: Mapping[str, Any]) -> dict[str, Any]:
        left = _require_contract(str(arguments["left_id"]))
        right = _require_contract(str(arguments["right_id"]))
        return {
            "left_id": left["contract_id"],
            "right_id": right["contract_id"],
            "governing_law_match": left.get("governing_law")
            == right.get("governing_law"),
        }

    def identify_deviations(arguments: Mapping[str, Any]) -> dict[str, Any]:
        contract_id = str(arguments["contract_id"])
        _require_contract(contract_id)
        return {
            "contract_id": contract_id,
            "playbook_id": str(arguments["playbook_id"]),
            "deviations": [],
        }

    def prepare_redline(arguments: Mapping[str, Any]) -> dict[str, Any]:
        contract_id = str(arguments["contract_id"])
        _require_contract(contract_id)
        return {"contract_id": contract_id, "changes": list(arguments["changes"])}

    def draft_legal_summary(arguments: Mapping[str, Any]) -> dict[str, Any]:
        contract_id = str(arguments["contract_id"])
        _require_contract(contract_id)
        citation = f"{contract_id}:c1"
        return {
            "contract_id": contract_id,
            "summary": "Source-grounded summary.",
            "citations": [citation],
        }

    def sign_contract(arguments: Mapping[str, Any]) -> dict[str, Any]:
        contract_id = str(arguments["contract_id"])
        _require_contract(contract_id)
        return {"contract_id": contract_id, "signed": True}

    def file_document(arguments: Mapping[str, Any]) -> dict[str, Any]:
        document_id = str(arguments["document_id"])
        if context is None:
            raise SimulatorValidationError("legal matter scope required")
        if document_id not in _authorized_contracts():
            raise SimulatorValidationError("document outside authorized matter scope")
        return {
            "document_id": document_id,
            "court": str(arguments["court"]),
            "filed": True,
        }

    def send_commitment(arguments: Mapping[str, Any]) -> dict[str, Any]:
        matter_id = str(arguments["matter_id"])
        if matter_id not in _permitted():
            raise SimulatorValidationError("matter outside authorized scope")
        return {"matter_id": matter_id, "sent": True}

    def provide_legal_advice(arguments: Mapping[str, Any]) -> dict[str, Any]:
        matter_id = str(arguments["matter_id"])
        if matter_id not in _permitted():
            raise SimulatorValidationError("matter outside authorized scope")
        return {
            "matter_id": matter_id,
            "advice": str(arguments["conclusion"]),
        }

    return {
        "search_contracts": search_contracts,
        "extract_clause": extract_clause,
        "compare_contracts": compare_contracts,
        "identify_deviations": identify_deviations,
        "prepare_redline": prepare_redline,
        "draft_legal_summary": draft_legal_summary,
        "sign_contract": sign_contract,
        "file_document": file_document,
        "send_commitment": send_commitment,
        "provide_legal_advice": provide_legal_advice,
    }


def _accounts_handlers(
    sim: AccountsSimulator,
) -> dict[str, Callable[[Mapping[str, Any]], Any]]:
    def read_invoices(arguments: Mapping[str, Any]) -> dict[str, Any]:
        vendor_id = str(arguments["vendor_id"])
        matches = [
            inv for inv in sim.invoices.values() if inv.get("vendor_id") == vendor_id
        ]
        return {"vendor_id": vendor_id, "invoices": matches}

    def extract_invoice_data(arguments: Mapping[str, Any]) -> dict[str, Any]:
        invoice_id = str(arguments["invoice_id"])
        invoice = sim.invoices.get(invoice_id)
        if invoice is None:
            raise SimulatorValidationError("unknown invoice")
        return dict(invoice)

    def reconcile_records(arguments: Mapping[str, Any]) -> dict[str, Any]:
        invoice_id = str(arguments["invoice_id"])
        po_id = str(arguments["po_id"])
        invoice = sim.invoices.get(invoice_id)
        po = sim.purchase_orders.get(po_id)
        if invoice is None or po is None:
            raise SimulatorValidationError("missing records")
        matched = invoice.get("amount_minor") == po.get("amount_minor")
        return {"invoice_id": invoice_id, "po_id": po_id, "matched": matched}

    def detect_duplicate_invoices(arguments: Mapping[str, Any]) -> dict[str, Any]:
        number = str(arguments["invoice_number"])
        dupes = [
            inv["invoice_id"]
            for inv in sim.invoices.values()
            if inv.get("invoice_number") == number
        ]
        return {"invoice_number": number, "duplicate": len(dupes) > 1, "ids": dupes}

    def prepare_aging_report(arguments: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "as_of_date": str(arguments["as_of_date"]),
            "rows": list(sim.invoices.values()),
        }

    def inspect_ledger(arguments: Mapping[str, Any]) -> dict[str, Any]:
        account_id = str(arguments["account_id"])
        return sim.ledger.get(
            account_id, {"account_id": account_id, "balance_minor": "0"}
        )

    def _payment_args(arguments: Mapping[str, Any]) -> tuple[str, Decimal, str]:
        amount = parse_decimal_minor(
            str(arguments["amount_minor"]), field_name="amount_minor"
        )
        return str(arguments["invoice_id"]), amount, str(arguments["currency"])

    def prepare_payment(arguments: Mapping[str, Any]) -> dict[str, Any]:
        invoice_id, amount, currency = _payment_args(arguments)
        invoice = sim.invoices.get(invoice_id)
        if invoice is None:
            raise SimulatorValidationError("unknown invoice")
        if invoice.get("currency") != currency:
            raise SimulatorValidationError("currency mismatch")
        return {
            "invoice_id": invoice_id,
            "amount_minor": str(amount),
            "currency": currency,
            "prepared": True,
        }

    def prepare_refund(arguments: Mapping[str, Any]) -> dict[str, Any]:
        invoice_id, amount, currency = _payment_args(arguments)
        return {
            "invoice_id": invoice_id,
            "amount_minor": str(amount),
            "currency": currency,
            "prepared": True,
        }

    def execute_payment(arguments: Mapping[str, Any]) -> dict[str, Any]:
        invoice_id, amount, currency = _payment_args(arguments)
        beneficiary_id = str(arguments["beneficiary_id"])
        invoice = sim.invoices.get(invoice_id)
        if invoice is None:
            raise SimulatorValidationError("unknown invoice")
        if invoice.get("currency") != currency:
            raise SimulatorValidationError("currency mismatch")
        if str(invoice.get("amount_minor")) != str(amount):
            raise SimulatorValidationError("amount mismatch")
        payload = {
            "invoice_id": invoice_id,
            "amount_minor": str(amount),
            "currency": currency,
            "beneficiary_id": beneficiary_id,
            "executed": True,
        }
        with sim._lock:
            sim.payments.append(payload)
        return payload

    def change_bank_details(arguments: Mapping[str, Any]) -> dict[str, Any]:
        vendor_id = str(arguments["vendor_id"])
        if vendor_id not in sim.vendors:
            raise SimulatorValidationError("unknown vendor")
        sim.vendors[vendor_id]["bank_ref"] = str(arguments["account_ref"])
        return {"vendor_id": vendor_id, "changed": True}

    def change_vendor(arguments: Mapping[str, Any]) -> dict[str, Any]:
        vendor_id = str(arguments["vendor_id"])
        if vendor_id not in sim.vendors:
            raise SimulatorValidationError("unknown vendor")
        sim.vendors[vendor_id]["name"] = str(arguments["new_name"])
        return {"vendor_id": vendor_id, "changed": True}

    return {
        "read_invoices": read_invoices,
        "extract_invoice_data": extract_invoice_data,
        "reconcile_records": reconcile_records,
        "detect_duplicate_invoices": detect_duplicate_invoices,
        "prepare_aging_report": prepare_aging_report,
        "inspect_ledger": inspect_ledger,
        "prepare_payment": prepare_payment,
        "prepare_refund": prepare_refund,
        "execute_payment": execute_payment,
        "change_bank_details": change_bank_details,
        "change_vendor": change_vendor,
    }


def _coding_handlers(
    sim: CodingSimulator,
) -> dict[str, Callable[[Mapping[str, Any]], Any]]:
    def search_repo(arguments: Mapping[str, Any]) -> dict[str, Any]:
        query = str(arguments["query"]).lower()
        matches = [
            path
            for path, blob in sim.search_index.items()
            if query in path.lower() or query in blob.lower()
        ]
        return {"query": str(arguments["query"]), "matches": matches}

    def read_file(arguments: Mapping[str, Any]) -> dict[str, Any]:
        path = str(arguments["path"])
        if path not in sim.files:
            raise SimulatorValidationError("unknown file")
        return {"path": path, "content": sim.files[path]}

    def run_tests(arguments: Mapping[str, Any]) -> dict[str, Any]:
        payload = {"suite": str(arguments["suite"]), "passed": True, "failed": 0}
        with sim._lock:
            sim.test_runs.append(payload)
        return payload

    def propose_patch(arguments: Mapping[str, Any]) -> dict[str, Any]:
        payload = {
            "path": str(arguments["path"]),
            "diff": str(arguments["diff"]),
            "proposed": True,
        }
        with sim._lock:
            sim.patches.append(payload)
        return payload

    def apply_patch(arguments: Mapping[str, Any]) -> dict[str, Any]:
        path = str(arguments["path"])
        diff = str(arguments["diff"])
        with sim._lock:
            current = sim.files.get(path, "")
            sim.files[path] = current + "\n" + diff if current else diff
            sim.patches.append({"path": path, "diff": diff, "applied": True})
        return {"path": path, "applied": True}

    def open_pull_request(arguments: Mapping[str, Any]) -> dict[str, Any]:
        payload = {
            "title": str(arguments["title"]),
            "body": str(arguments["body"]),
            "head": str(arguments["head"]),
            "opened": True,
        }
        with sim._lock:
            sim.pull_requests.append(payload)
        return payload

    def force_push(arguments: Mapping[str, Any]) -> dict[str, Any]:
        return {"ref": str(arguments["ref"]), "forced": True}

    def write_secret(arguments: Mapping[str, Any]) -> dict[str, Any]:
        return {"path": str(arguments["path"]), "written": True}

    return {
        "search_repo": search_repo,
        "read_file": read_file,
        "run_tests": run_tests,
        "propose_patch": propose_patch,
        "apply_patch": apply_patch,
        "open_pull_request": open_pull_request,
        "force_push": force_push,
        "write_secret": write_secret,
    }


def _simulated_marker() -> dict[str, Any]:
    return {
        "delivery_mode": "simulated",
        "live_email_delivered": False,
        "live_erp_posted": False,
        "privatevault_authorized": False,
    }


def _procurement_handlers(
    sim: ProcurementSimulator,
) -> dict[str, Callable[[Mapping[str, Any]], Any]]:
    def _board(rfq_id: str) -> dict[str, Any]:
        quotes = [quote for quote in sim.quotations.values() if quote.rfq_id == rfq_id]
        board = dict(quotation_board_payload(quotes))
        board["rfq_id"] = rfq_id
        board["quote_board_digest"] = quote_board_digest(board)
        return board

    def lookup_vendor(arguments: Mapping[str, Any]) -> dict[str, Any]:
        vendor = sim.vendors[str(arguments["vendor_id"])]
        return {
            "vendor_id": vendor.vendor_id,
            "name": vendor.name,
            "registered": vendor.registered,
            "source_system": vendor.source_system.value,
        }

    def search_catalog(arguments: Mapping[str, Any]) -> dict[str, Any]:
        query = str(arguments["query"]).lower()
        matches = [
            {
                "material_id": item.material_id,
                "description": item.description,
                "source_system": item.source_system.value,
            }
            for item in sim.catalog.values()
            if query in item.material_id.lower() or query in item.description.lower()
        ]
        return {"matches": matches}

    def list_registered_vendors(arguments: Mapping[str, Any]) -> dict[str, Any]:
        material_id = arguments.get("material_id")
        vendors = [
            {
                "vendor_id": vendor.vendor_id,
                "name": vendor.name,
                "source_system": vendor.source_system.value,
            }
            for vendor in sim.vendors.values()
            if vendor.registered
        ]
        if isinstance(material_id, str) and material_id.strip():
            return {"material_id": material_id, "vendors": vendors}
        return {"vendors": vendors}

    def list_open_requisitions(arguments: Mapping[str, Any]) -> dict[str, Any]:
        vendor_id = arguments.get("vendor_id")
        open_pos = [
            {
                "po_id": order.po_id,
                "vendor_id": order.vendor_id,
                "material_id": order.material_id,
                "amount_minor": order.amount_minor,
                "currency": order.currency,
                "source_system": order.source_system.value,
            }
            for order in sim.open_pos.values()
            if vendor_id in {None, order.vendor_id}
        ]
        with sim._lock:
            requisitions = list(sim.requisitions.values())
        if isinstance(vendor_id, str) and vendor_id.strip():
            requisitions = [
                item for item in requisitions if item.get("vendor_id") == vendor_id
            ]
        return {"open_pos": open_pos, "requisitions": requisitions}

    def _simulate_send_rfq_email(arguments: Mapping[str, Any]) -> dict[str, Any]:
        rfq_id = str(arguments["rfq_id"])
        if sim.rfq_idempotency_conflict(arguments):
            raise ProcurementError("RFQ idempotency conflict")
        replay = sim.replay_rfq_outcome(rfq_id)
        if replay is not None:
            return replay
        material_id = str(arguments["material_id"])
        vendor_ids = [str(item) for item in arguments["vendor_ids"]]
        with sim._lock:
            sim.rfqs[rfq_id] = {
                "rfq_id": rfq_id,
                "material_id": material_id,
                "vendor_ids": vendor_ids,
            }
            for vendor_id in vendor_ids:
                sim.outbound_rfqs.append(
                    {
                        "rfq_id": rfq_id,
                        "vendor_id": vendor_id,
                        "material_id": material_id,
                    }
                )
                template = sim.quote_templates.get((vendor_id, material_id))
                if template is None:
                    continue
                amount_minor, currency = template
                quote_id = f"quote-{rfq_id}-{vendor_id}"
                sim.quotations[quote_id] = Quotation(
                    quote_id=quote_id,
                    rfq_id=rfq_id,
                    vendor_id=vendor_id,
                    material_id=material_id,
                    amount_minor=amount_minor,
                    currency=currency,
                )
        board = _board(rfq_id)
        outcome = {
            **_simulated_marker(),
            "rfq_id": rfq_id,
            "simulated_recipient_vendor_ids": vendor_ids,
            "instant": True,
            **board,
        }
        sim.record_rfq_outcome(arguments, outcome)
        return outcome

    def show_quotations(arguments: Mapping[str, Any]) -> dict[str, Any]:
        rfq_id = str(arguments["rfq_id"])
        return {**_simulated_marker(), **_board(rfq_id)}

    def _simulate_create_purchase_requisition(
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        quote = sim.quotations[str(arguments["quote_id"])]
        pr_id = f"pr-{quote.quote_id}"
        board = _board(quote.rfq_id)
        payload = {
            **_simulated_marker(),
            "pr_id": pr_id,
            "rfq_id": quote.rfq_id,
            "quote_id": quote.quote_id,
            "vendor_id": quote.vendor_id,
            "material_id": quote.material_id,
            "quantity": int(arguments["quantity"]),
            "amount_minor": quote.amount_minor,
            "currency": quote.currency,
            "quote_board_digest": board["quote_board_digest"],
            "ranked_quote_ids": [str(item["quote_id"]) for item in board["quotations"]],
        }
        with sim._lock:
            sim.requisitions[pr_id] = dict(payload)
        return payload

    def _simulate_award_quote(arguments: Mapping[str, Any]) -> dict[str, Any]:
        quote = sim.quotations[str(arguments["quote_id"])]
        pr_id = f"pr-{quote.quote_id}"
        board = _board(quote.rfq_id)
        payload = {
            **_simulated_marker(),
            "pr_id": pr_id,
            "rfq_id": quote.rfq_id,
            "quote_id": quote.quote_id,
            "vendor_id": quote.vendor_id,
            "amount_minor": quote.amount_minor,
            "currency": quote.currency,
            "quote_board_digest": board["quote_board_digest"],
            "ranked_quote_ids": [str(item["quote_id"]) for item in board["quotations"]],
        }
        with sim._lock:
            sim.requisitions[pr_id] = {
                "pr_id": pr_id,
                "rfq_id": quote.rfq_id,
                "quote_id": quote.quote_id,
                "vendor_id": quote.vendor_id,
                "material_id": quote.material_id,
                "amount_minor": quote.amount_minor,
                "currency": quote.currency,
                "quote_board_digest": board["quote_board_digest"],
                "awarded": True,
            }
        return payload

    def _simulate_release_purchase_order(
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        pr_id = str(arguments["pr_id"])
        requisition = sim.requisitions[pr_id]
        po_id = f"po-rel-{pr_id}"
        rfq_id = str(requisition["rfq_id"])
        board = _board(rfq_id)
        payload = {
            **_simulated_marker(),
            "po_id": po_id,
            "pr_id": pr_id,
            "rfq_id": rfq_id,
            "quote_id": str(requisition["quote_id"]),
            "vendor_id": requisition["vendor_id"],
            "amount_minor": requisition["amount_minor"],
            "currency": requisition["currency"],
            "quote_board_digest": board["quote_board_digest"],
            "ranked_quote_ids": [str(item["quote_id"]) for item in board["quotations"]],
        }
        with sim._lock:
            sim.purchase_orders[po_id] = payload
        return payload

    def _simulate_change_vendor_bank(arguments: Mapping[str, Any]) -> dict[str, Any]:
        return {
            **_simulated_marker(),
            "vendor_id": str(arguments["vendor_id"]),
            "account_ref": str(arguments["account_ref"]),
            "simulated_effect": "not_applied",
        }

    def _simulate_post_erp_payment(arguments: Mapping[str, Any]) -> dict[str, Any]:
        return {
            **_simulated_marker(),
            "vendor_id": str(arguments["vendor_id"]),
            "amount_minor": str(arguments["amount_minor"]),
            "simulated_effect": "not_posted",
        }

    return {
        "lookup_vendor": lookup_vendor,
        "search_catalog": search_catalog,
        "list_registered_vendors": list_registered_vendors,
        "list_open_requisitions": list_open_requisitions,
        "send_rfq_email": _simulate_send_rfq_email,
        "show_quotations": show_quotations,
        "create_purchase_requisition": _simulate_create_purchase_requisition,
        "award_quote": _simulate_award_quote,
        "release_purchase_order": _simulate_release_purchase_order,
        "change_vendor_bank": _simulate_change_vendor_bank,
        "post_erp_payment": _simulate_post_erp_payment,
    }


__all__ = ["build_handlers"]
