"""Offline-eval approver identities. Not a production directory."""

from __future__ import annotations

from types import MappingProxyType

from cbrain.company.approval import ApprovalPrincipal, ApprovalRole
from cbrain.company.kinds import CompanyAgentKind

FIXTURE_MAX_TTL_SECONDS = 3600.0

CONTROLLER_PRINCIPAL = ApprovalPrincipal(
    actor_id="operator-controller-1",
    role=ApprovalRole.CONTROLLER,
)
COUNSEL_PRINCIPAL = ApprovalPrincipal(
    actor_id="operator-counsel-1",
    role=ApprovalRole.COUNSEL,
)
CODE_REVIEWER_PRINCIPAL = ApprovalPrincipal(
    actor_id="operator-code-reviewer-1",
    role=ApprovalRole.CODE_REVIEWER,
)

APPROVER_DIRECTORY = MappingProxyType(
    {
        ApprovalRole.CONTROLLER: frozenset({CONTROLLER_PRINCIPAL.actor_id}),
        ApprovalRole.COUNSEL: frozenset({COUNSEL_PRINCIPAL.actor_id}),
        ApprovalRole.CODE_REVIEWER: frozenset({CODE_REVIEWER_PRINCIPAL.actor_id}),
    }
)

PRINCIPAL_BY_AGENT = MappingProxyType(
    {
        CompanyAgentKind.ACCOUNTS: CONTROLLER_PRINCIPAL,
        CompanyAgentKind.LEGAL: COUNSEL_PRINCIPAL,
        CompanyAgentKind.CODING: CODE_REVIEWER_PRINCIPAL,
    }
)
