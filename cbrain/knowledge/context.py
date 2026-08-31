"""Foundation-agent context adapter. Retrieved text is quoted evidence only."""

from __future__ import annotations

from .contracts import RetrievalQuery, RetrievedContext
from .retrieval import HybridRetriever


class RetrievedKnowledgeProvider:
    """Optional FoundationAgent dependency. Never creates an ActionIntent."""

    def __init__(self, retriever: HybridRetriever) -> None:
        self._retriever = retriever

    def retrieve(self, query: RetrievalQuery) -> RetrievedContext:
        return self._retriever.retrieve(query)


__all__ = ["RetrievedKnowledgeProvider"]
