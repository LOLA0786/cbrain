"""Fail-closed errors for the knowledge runtime."""

from __future__ import annotations


class KnowledgeError(ValueError):
    """Knowledge input, configuration, or publication is invalid."""


class KnowledgeUnavailable(KnowledgeError):
    """Durable knowledge store cannot serve the request."""

    code = "KNOWLEDGE_UNAVAILABLE"


class KnowledgeConfigError(KnowledgeError):
    """Deployment-owned knowledge configuration is invalid."""


__all__ = ["KnowledgeConfigError", "KnowledgeError", "KnowledgeUnavailable"]
