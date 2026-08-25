"""Company agent identifiers."""

from __future__ import annotations

from enum import StrEnum


class CompanyAgentKind(StrEnum):
    GTM = "gtm"
    OPERATIONS = "operations"
    LEGAL = "legal"
    ACCOUNTS = "accounts"
    CODING = "coding"


MATRIX_AGENT_KINDS: tuple[CompanyAgentKind, ...] = (
    CompanyAgentKind.GTM,
    CompanyAgentKind.OPERATIONS,
    CompanyAgentKind.LEGAL,
    CompanyAgentKind.ACCOUNTS,
)

OPERATOR_AGENT_KINDS: tuple[CompanyAgentKind, ...] = (
    CompanyAgentKind.ACCOUNTS,
    CompanyAgentKind.LEGAL,
    CompanyAgentKind.CODING,
)


__all__ = ["CompanyAgentKind", "MATRIX_AGENT_KINDS", "OPERATOR_AGENT_KINDS"]
