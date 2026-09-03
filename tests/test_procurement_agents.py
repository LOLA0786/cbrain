"""Procurement bots: ERP replicas, registered-vendor RFQs, HITL award proofs."""

from __future__ import annotations

from pathlib import Path

import pytest
from knowledge_fakes import DeterministicEmbeddingProvider, RuleBasedExtractor
from operator_fixtures import APPROVER_DIRECTORY, BUYER_LEAD_PRINCIPAL

from cbrain import ExecutionStatus, GovernedRuntime
from cbrain.agent import FoundationAgent, FoundationAgentError, ToolRegistry
from cbrain.company.approval import (
    ApprovalBoundedGateway,
    ApprovalInbox,
    ApprovalRole,
)
from cbrain.company.governance import CompanyRiskGateway
from cbrain.company.handlers import build_handlers
from cbrain.company.kinds import CompanyAgentKind
from cbrain.company.profiles import spec_for_kind
from cbrain.company.risk import ToolRiskLevel
from cbrain.company.simulators import load_fixture_bundle
from cbrain.company.spec import CompanyAgentSpec
from cbrain.company.tools import tools_for_kind
from cbrain.contracts import ActionIntent
from cbrain.evaluation.company_harness import canonical_action_intent_digest
from cbrain.knowledge import KnowledgeRuntime, RetrievalQuery
from cbrain.models import CompletionRequest, ModelRouter, TextOutput
from cbrain.procurement import (
    BUYER_PROFILE,
    CATEGORY_MANAGER_PROFILE,
    VENDOR_ONBOARDING_PROFILE,
    ErpSystem,
    ProcurementError,
    build_procurement_proof,
    documents_from_replica,
    independent_procurement_profiles,
    load_json_extract,
)
from cbrain.procurement.demo import demo_replica_sources
from cbrain.procurement.erp import records_from_mapping
from cbrain.procurement.extracts import COLLECTION_VENDORS

NOW = 1_700_000_000.0
RECEIPT = "sha256:" + ("ab" * 32)


def _action(
    spec: CompanyAgentSpec, tool: str, arguments: dict[str, object]
) -> ActionIntent:
    return ActionIntent.capture(
        agent_id=spec.agent_id,
        framework="test",
        tool_name=tool,
        capability=spec.tools.get(tool).capability,
        arguments=arguments,
    )


def test_independent_bots_share_buyer_runtime_but_not_write_tools() -> None:
    profiles = independent_procurement_profiles()
    assert {profile.agent_id for profile in profiles} == {
        BUYER_PROFILE.agent_id,
        CATEGORY_MANAGER_PROFILE.agent_id,
        VENDOR_ONBOARDING_PROFILE.agent_id,
    }
    assert BUYER_PROFILE.knowledge_required_for_tools is True
    assert "send_rfq_email" in BUYER_PROFILE.permitted_tools
    assert "award_quote" in BUYER_PROFILE.permitted_tools
    assert "send_rfq_email" not in CATEGORY_MANAGER_PROFILE.permitted_tools
    assert "award_quote" not in CATEGORY_MANAGER_PROFILE.permitted_tools
    assert "award_quote" not in VENDOR_ONBOARDING_PROFILE.permitted_tools
    assert "change_vendor_bank" not in BUYER_PROFILE.permitted_tools
    assert "post_erp_payment" not in BUYER_PROFILE.permitted_tools
    assert "change_vendor_bank" not in VENDOR_ONBOARDING_PROFILE.permitted_tools
    assert "post_erp_payment" not in VENDOR_ONBOARDING_PROFILE.permitted_tools


def test_replica_extracts_are_tagged_by_oracle_sap_and_sql_server() -> None:
    systems = {source.system() for source in demo_replica_sources()}
    assert systems == {ErpSystem.ORACLE, ErpSystem.SAP, ErpSystem.SQL_SERVER}


def test_json_extract_rejects_dsn_and_password_fields(tmp_path: Path) -> None:
    path = tmp_path / "oracle.json"
    path.write_text('{"dsn": "host=secret", "vendors": []}', encoding="utf-8")
    with pytest.raises(ProcurementError, match="secret"):
        load_json_extract(path, system=ErpSystem.ORACLE)
    with pytest.raises(ProcurementError, match="connection material"):
        records_from_mapping(
            {
                "vendors": [
                    {
                        "vendor_id": "v1",
                        "name": "Bad",
                        "email": "jdbc:oracle:thin:@//host/orcl",
                        "registered": True,
                    }
                ]
            },
            system=ErpSystem.ORACLE,
        )


def test_replica_documents_ingest_as_untrusted_knowledge() -> None:
    runtime = KnowledgeRuntime(
        clock=lambda: NOW,
        embeddings=DeterministicEmbeddingProvider(),
        extractor=RuleBasedExtractor(),
    )
    oracle = demo_replica_sources()[0]
    documents = documents_from_replica(
        oracle,
        tenant_id="tenant-a",
        principal_id="buyer-1",
        created_at=NOW,
    )
    for document in documents:
        result = runtime.ingest(document)
        assert result.published is True
    retrieved = runtime.retrieve(
        RetrievalQuery(
            tenant_id="tenant-a",
            collection_id=COLLECTION_VENDORS,
            principal_id="buyer-1",
            text="vendor-oracle-1",
            top_k=4,
            created_at=NOW,
        )
    )
    assert retrieved.hits
    assert "UNTRUSTED_SOURCE_CONTEXT" in retrieved.quoted_evidence
    assert "dsn" not in retrieved.quoted_evidence.casefold()
    assert "password" not in retrieved.quoted_evidence.casefold()


def test_rfq_to_registered_vendors_shows_quotations_instantly() -> None:
    spec = spec_for_kind(CompanyAgentKind.PROCUREMENT)
    bundle = load_fixture_bundle("default")
    gateway = CompanyRiskGateway(spec, bundle=bundle)
    handlers = build_handlers(kind=CompanyAgentKind.PROCUREMENT, bundle=bundle)
    execution = gateway.decide_and_execute(
        _action(
            spec,
            "send_rfq_email",
            {
                "rfq_id": "rfq-fast-1",
                "material_id": "mat-steel-rod",
                "vendor_ids": ["vendor-oracle-1", "vendor-sap-1", "vendor-sql-1"],
            },
        ),
        handlers["send_rfq_email"],
    )
    assert execution.status is ExecutionStatus.EXECUTED
    output = execution.output
    assert output["instant"] is True
    assert output["count"] == 3
    assert output["delivery_mode"] == "simulated"
    assert output["live_email_delivered"] is False
    assert output["live_erp_posted"] is False
    assert output["privatevault_authorized"] is False
    assert output["quotations"][0]["vendor_id"] == "vendor-oracle-1"
    assert output["quotations"][0]["amount_minor"] == "10000"
    shown = handlers["show_quotations"]({"rfq_id": "rfq-fast-1"})
    assert shown["quotations"][0]["quote_id"] == output["quotations"][0]["quote_id"]


def test_unregistered_vendor_rfq_is_blocked_before_handler() -> None:
    spec = spec_for_kind(CompanyAgentKind.PROCUREMENT)
    bundle = load_fixture_bundle("default")
    gateway = CompanyRiskGateway(spec, bundle=bundle)
    handlers = build_handlers(kind=CompanyAgentKind.PROCUREMENT, bundle=bundle)
    execution = gateway.decide_and_execute(
        _action(
            spec,
            "send_rfq_email",
            {
                "rfq_id": "rfq-blocked-1",
                "material_id": "mat-steel-rod",
                "vendor_ids": ["vendor-ghost"],
            },
        ),
        handlers["send_rfq_email"],
    )
    assert execution.status is ExecutionStatus.BLOCKED
    assert execution.tool_executed is False
    assert execution.reason == "PROCUREMENT_VENDOR_UNREGISTERED"
    assert bundle.procurement.outbound_rfqs == []


def test_award_requires_buyer_lead_and_emits_offline_proof() -> None:
    spec = spec_for_kind(CompanyAgentKind.PROCUREMENT)
    bundle = load_fixture_bundle("default")
    inner = CompanyRiskGateway(spec, bundle=bundle)
    inbox = ApprovalInbox(
        allowed_approvers=APPROVER_DIRECTORY,
        max_ttl_seconds=3600.0,
    )
    gateway = ApprovalBoundedGateway(
        inner,
        inbox,
        digest_for=canonical_action_intent_digest,
        required_role_for=lambda _: ApprovalRole.BUYER_LEAD,
    )
    handlers = build_handlers(kind=CompanyAgentKind.PROCUREMENT, bundle=bundle)
    action = _action(
        spec,
        "award_quote",
        {
            "rfq_id": "rfq-steel-1",
            "quote_id": "quote-oracle-1",
            "amount_minor": "10000",
            "currency": "USD",
        },
    )
    digest = canonical_action_intent_digest(action)
    first = GovernedRuntime(gateway).execute(action, handlers["award_quote"])
    assert first.status is ExecutionStatus.REVIEW_REQUIRED
    assert first.tool_executed is False
    inbox.approve(action.request_id, principal=BUYER_LEAD_PRINCIPAL)
    final = GovernedRuntime(gateway).execute(action, handlers["award_quote"])
    assert final.status is ExecutionStatus.EXECUTED
    assert final.output["vendor_id"] == "vendor-oracle-1"
    assert final.output["delivery_mode"] == "simulated"
    assert final.output["privatevault_authorized"] is False
    board = handlers["show_quotations"]({"rfq_id": "rfq-steel-1"})
    offline = build_procurement_proof(
        action=action,
        execution=final,
        quote_board=board,
    )
    payload = offline.to_payload()
    assert payload["decision_authority"] == "company_test_gateway"
    assert payload["decision_receipt_digest"] is None
    assert payload["authority_receipt_digest"] is None
    assert payload["action_intent_digest"] == digest
    assert payload["quote_id"] == "quote-oracle-1"
    assert payload["awarded_vendor_id"] == "vendor-oracle-1"
    with pytest.raises(TypeError):
        build_procurement_proof(
            action=action,
            execution=final,
            quote_board=board,
            decision_authority="privatevault",
            action_intent_digest=digest,
            decision_receipt_digest=RECEIPT,
            authority_receipt_digest=RECEIPT,
        )
    rendered = str(payload)
    assert "dsn" not in rendered
    assert "password" not in rendered
    assert "oracle-steel@example.com" not in rendered


def test_review_and_block_procurement_tools_never_run_handlers() -> None:
    spec = spec_for_kind(CompanyAgentKind.PROCUREMENT)
    bundle = load_fixture_bundle("default")
    gateway = CompanyRiskGateway(spec, bundle=bundle)
    handlers = build_handlers(kind=CompanyAgentKind.PROCUREMENT, bundle=bundle)
    samples: dict[str, dict[str, object]] = {
        "create_purchase_requisition": {
            "rfq_id": "rfq-steel-1",
            "quote_id": "quote-oracle-1",
            "material_id": "mat-steel-rod",
            "quantity": 10,
            "amount_minor": "10000",
            "currency": "USD",
        },
        "award_quote": {
            "rfq_id": "rfq-steel-1",
            "quote_id": "quote-oracle-1",
            "amount_minor": "10000",
            "currency": "USD",
        },
        "release_purchase_order": {
            "pr_id": "missing",
            "amount_minor": "10000",
            "currency": "USD",
        },
        "change_vendor_bank": {
            "vendor_id": "vendor-oracle-1",
            "account_ref": "bank",
        },
        "post_erp_payment": {
            "vendor_id": "vendor-oracle-1",
            "amount_minor": "10000",
            "currency": "USD",
        },
    }
    for tool, risk in spec.tool_risks.items():
        if risk is ToolRiskLevel.ALLOW:
            continue
        execution = gateway.decide_and_execute(
            _action(spec, tool, samples[tool]),
            handlers[tool],
        )
        assert execution.tool_executed is False
        assert tool not in gateway.handler_calls


def test_category_manager_cannot_register_award_or_rfq_handlers() -> None:
    bundle = load_fixture_bundle("default")
    all_handlers = build_handlers(kind=CompanyAgentKind.PROCUREMENT, bundle=bundle)
    handlers = {
        name: all_handlers[name] for name in CATEGORY_MANAGER_PROFILE.permitted_tools
    }
    runtime = GovernedRuntime(
        CompanyRiskGateway(spec_for_kind(CompanyAgentKind.PROCUREMENT), bundle=bundle)
    )
    models = ModelRouter({"offline": _TextOnly()})
    registry = ToolRegistry(tools_for_kind(CompanyAgentKind.PROCUREMENT))
    FoundationAgent(
        profile=CATEGORY_MANAGER_PROFILE,
        runtime=runtime,
        model_router=models,
        tools=registry,
        handlers=handlers,
    )
    with pytest.raises(FoundationAgentError, match="non-permitted"):
        FoundationAgent(
            profile=CATEGORY_MANAGER_PROFILE,
            runtime=runtime,
            model_router=models,
            tools=registry,
            handlers={**handlers, "award_quote": all_handlers["award_quote"]},
        )
    with pytest.raises(FoundationAgentError, match="non-permitted"):
        FoundationAgent(
            profile=CATEGORY_MANAGER_PROFILE,
            runtime=runtime,
            model_router=models,
            tools=registry,
            handlers={**handlers, "send_rfq_email": all_handlers["send_rfq_email"]},
        )


class _TextOnly:
    @property
    def provider(self) -> str:
        return "offline"

    @property
    def model(self) -> str:
        return "scripted"

    def complete(self, request: CompletionRequest) -> TextOutput:
        return TextOutput(text="ok")
