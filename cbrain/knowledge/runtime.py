"""Facade over ingestion, retrieval, and graph for reusable agents."""

from __future__ import annotations

import time
from collections.abc import Callable

from .configuration import KnowledgeConfig, default_knowledge_config
from .context import RetrievedKnowledgeProvider
from .contracts import (
    GraphPath,
    IngestionResult,
    RetrievalQuery,
    RetrievedContext,
    SourceDocument,
)
from .graph import graph_paths
from .ingestion import IngestionPipeline
from .ports import EmbeddingProvider, EntityRelationExtractor
from .retrieval import HybridRetriever
from .stores.memory import (
    DeterministicEmbeddingProvider,
    InMemoryKnowledgeStore,
    InMemoryKVCache,
    RuleBasedExtractor,
)


class KnowledgeRuntime:
    def __init__(
        self,
        *,
        config: KnowledgeConfig | None = None,
        store: InMemoryKnowledgeStore | None = None,
        cache: InMemoryKVCache | None = None,
        embeddings: EmbeddingProvider | None = None,
        extractor: EntityRelationExtractor | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.config = config or default_knowledge_config()
        self.store = store or InMemoryKnowledgeStore()
        self.cache = cache or InMemoryKVCache()
        self.embeddings = embeddings or DeterministicEmbeddingProvider()
        self.extractor = extractor or RuleBasedExtractor()
        self._clock = clock or time.time
        self.pipeline = IngestionPipeline(
            store=self.store,
            config=self.config,
            embeddings=self.embeddings,
            extractor=self.extractor,
            clock=self._clock,
        )
        self.retriever = HybridRetriever(
            store=self.store,
            cache=self.cache,
            config=self.config,
            embeddings=self.embeddings,
        )
        self.provider = RetrievedKnowledgeProvider(self.retriever)

    def ingest(self, document: SourceDocument) -> IngestionResult:
        return self.pipeline.ingest(document)

    def tombstone(
        self, tenant_id: str, collection_id: str, source_id: str
    ) -> IngestionResult:
        return self.pipeline.tombstone(tenant_id, collection_id, source_id)

    def retrieve(self, query: RetrievalQuery) -> RetrievedContext:
        return self.provider.retrieve(query)

    def graph_path(
        self,
        *,
        tenant_id: str,
        principal_id: str,
        seed_node_ids: tuple[str, ...],
    ) -> tuple[GraphPath, ...]:
        return graph_paths(
            self.store,
            tenant_id=tenant_id,
            principal_id=principal_id,
            seed_node_ids=seed_node_ids,
            max_depth=self.config.graph_max_depth,
            max_nodes=self.config.graph_max_nodes,
        )

    def collection_status(
        self, tenant_id: str, collection_id: str
    ) -> dict[str, object]:
        return {
            "tenant_id": tenant_id,
            "collection_id": collection_id,
            "collection_revision": self.store.collection_revision(
                tenant_id, collection_id
            ),
            "embedding_profile": (
                None
                if self.store.embedding_profile(tenant_id, collection_id) is None
                else self.config.embedding_profile.profile_id
            ),
        }


__all__ = ["KnowledgeRuntime"]
