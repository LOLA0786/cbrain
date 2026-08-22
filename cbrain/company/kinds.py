"""Company agent identifiers."""

from __future__ import annotations

from enum import StrEnum


class CompanyAgentKind(StrEnum):
    GTM = "gtm"
    OPERATIONS = "operations"
    LEGAL = "legal"
    ACCOUNTS = "accounts"


__all__ = ["CompanyAgentKind"]
