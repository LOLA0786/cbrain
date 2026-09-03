"""Offline demo: registered-vendor RFQ, instant quotes, HITL award proof."""

from __future__ import annotations

from types import MappingProxyType

from cbrain import GovernedRuntime
from cbrain.company.approval import (
    ApprovalBoundedGateway,
    ApprovalInbox,
    ApprovalPrincipal,
    ApprovalRole,
)
from cbrain.evaluation.company_gateway import CompanyRiskGateway
from cbrain.company.handlers import build_handlers
from cbrain.company.kinds import CompanyAgentKind
from cbrain.company.profiles import spec_for_kind
from cbrain.company.simulators import load_fixture_bundle
from cbrain.contracts import ActionIntent
from cbrain.evaluation.company_harness import canonical_action_intent_digest
from cbrain.procurement import build_procurement_proof

_BUYER_LEAD = ApprovalPrincipal(
    actor_id="demo-buyer-lead", role=ApprovalRole.BUYER_LEAD
)
_DIRECTORY = MappingProxyType(
    {ApprovalRole.BUYER_LEAD: frozenset({_BUYER_LEAD.actor_id})}
)


def main() -> int:
    spec = spec_for_kind(CompanyAgentKind.PROCUREMENT)
    bundle = load_fixture_bundle("default")
    handlers = build_handlers(kind=CompanyAgentKind.PROCUREMENT, bundle=bundle)
    gateway = CompanyRiskGateway(spec, bundle=bundle)
    runtime = GovernedRuntime(gateway)

    rfq = ActionIntent.capture(
        agent_id=spec.agent_id,
        framework="cbrain-procurement-demo",
        tool_name="send_rfq_email",
        capability=spec.tools.get("send_rfq_email").capability,
        arguments={
            "rfq_id": "rfq-demo-1",
            "material_id": "mat-steel-rod",
            "vendor_ids": ["vendor-oracle-1", "vendor-sap-1", "vendor-sql-1"],
        },
    )
    sent = runtime.execute(rfq, handlers["send_rfq_email"])
    print("rfq status:", sent.status.value)
    print("instant quotes:", sent.output["count"] if sent.output else 0)
    print("delivery_mode:", sent.output.get("delivery_mode") if sent.output else None)

    inbox = ApprovalInbox(
        allowed_approvers=_DIRECTORY,
        max_ttl_seconds=3600.0,
    )
    bounded = ApprovalBoundedGateway(
        CompanyRiskGateway(spec, bundle=bundle),
        inbox,
        digest_for=canonical_action_intent_digest,
        required_role_for=lambda _: ApprovalRole.BUYER_LEAD,
    )
    award_runtime = GovernedRuntime(bounded)
    award = ActionIntent.capture(
        agent_id=spec.agent_id,
        framework="cbrain-procurement-demo",
        tool_name="award_quote",
        capability=spec.tools.get("award_quote").capability,
        arguments={
            "rfq_id": "rfq-demo-1",
            "quote_id": sent.output["quotations"][0]["quote_id"],
            "amount_minor": sent.output["quotations"][0]["amount_minor"],
            "currency": sent.output["quotations"][0]["currency"],
        },
    )
    parked = award_runtime.execute(award, handlers["award_quote"])
    print("award parked:", parked.status.value)
    inbox.approve(award.request_id, principal=_BUYER_LEAD)
    done = award_runtime.execute(award, handlers["award_quote"])
    board = handlers["show_quotations"]({"rfq_id": "rfq-demo-1"})
    proof = build_procurement_proof(
        action=award,
        execution=done,
        quote_board=board,
    )
    print("award status:", done.status.value)
    print("proof authority:", proof.decision_authority)
    print("proof digest:", proof.action_intent_digest)
    return 0 if done.status.value == "EXECUTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
