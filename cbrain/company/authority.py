"""Deployment-owned execution authority for company agents."""

from __future__ import annotations

from dataclasses import dataclass

from .kinds import CompanyAgentKind


@dataclass(frozen=True, slots=True)
class CompanyExecutionContext:
    """Non-model-controlled authorization scope for one company-agent run."""

    principal_id: str
    agent_kind: CompanyAgentKind
    permitted_matter_ids: frozenset[str]
    authorization_scope_id: str

    def __post_init__(self) -> None:
        if not self.principal_id.strip():
            raise ValueError("principal_id must be non-empty")
        if not self.authorization_scope_id.strip():
            raise ValueError("authorization_scope_id must be non-empty")
        if not isinstance(self.permitted_matter_ids, frozenset):
            raise ValueError("permitted_matter_ids must be a frozenset")
        if any(
            not isinstance(item, str) or not item.strip()
            for item in self.permitted_matter_ids
        ):
            raise ValueError("permitted matter IDs must be non-empty strings")


__all__ = ["CompanyExecutionContext"]
