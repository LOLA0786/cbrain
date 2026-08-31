"""Configuration-driven company agents over the shared FoundationAgent runtime."""

from .authority import CompanyExecutionContext
from .kinds import CompanyAgentKind
from .profiles import (
    ACCOUNTS_PROFILE,
    CODING_PROFILE,
    GTM_PROFILE,
    LEGAL_PROFILE,
    OPERATIONS_PROFILE,
    PROCUREMENT_PROFILE,
    all_company_profiles,
    profile_for_kind,
)
from .spec import (
    COMPANY_AGENT_VERSION,
    SCENARIO_SUITE_VERSION,
    CompanyAgentSpec,
    EvaluationThresholds,
    build_company_spec,
)
from .tools import company_tool_registry

__all__ = [
    "ACCOUNTS_PROFILE",
    "CODING_PROFILE",
    "COMPANY_AGENT_VERSION",
    "SCENARIO_SUITE_VERSION",
    "CompanyAgentKind",
    "CompanyExecutionContext",
    "CompanyAgentSpec",
    "EvaluationThresholds",
    "GTM_PROFILE",
    "LEGAL_PROFILE",
    "OPERATIONS_PROFILE",
    "PROCUREMENT_PROFILE",
    "all_company_profiles",
    "build_company_spec",
    "company_tool_registry",
    "profile_for_kind",
]
