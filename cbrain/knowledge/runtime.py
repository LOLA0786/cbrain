"""Facade over ingestion, retrieval, and graph for reusable agents."""

from __future__ import annotations

import time
from collections.abc import Callable

from .configuration import (
    KnowledgeConfig,
    default_knowledge_config,
    knowledge_config_from_env,
)
from .context import RetrievedKnowledgeProvider
from .contracts import (
    GraphPath,
    IngestionResult,
    RetrievalQuery,
    RetrievedContext,
    SourceDocument,
)
from .errors import KnowledgeConfigError
from .graph import graph_paths
from .ingestion import IngestionPipeline
from .ports import EmbeddingProvider, EntityRelationExtractor, KnowledgeStore, KVCache
from .retrieval import HybridRetriever
from .stores.memory import (
    InMemoryKnowledgeStore,
    InMemoryKVCache,
    NullKVCache,
)


class KnowledgeRuntime:
    def __init__(
        self,
        *,
        config: KnowledgeConfig | None = None,
        store: KnowledgeStore | None = None,
        cache: KVCache | None = None,
        embeddings: EmbeddingProvider,
        extractor: EntityRelationExtractor,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.config = config or default_knowledge_config()
        self.store = store or InMemoryKnowledgeStore()
        self.cache = cache or InMemoryKVCache()
        self.embeddings = embeddings
        self.extractor = extractor
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

    @classmethod
    def in_memory(
        cls,
        *,
        config: KnowledgeConfig | None = None,
        embeddings: EmbeddingProvider,
        extractor: EntityRelationExtractor,
        clock: Callable[[], float] | None = None,
        store: KnowledgeStore | None = None,
        cache: KVCache | None = None,
    ) -> KnowledgeRuntime:
        return cls(
            config=config,
            store=store or InMemoryKnowledgeStore(),
            cache=cache or InMemoryKVCache(),
            embeddings=embeddings,
            extractor=extractor,
            clock=clock,
        )

    @classmethod
    def from_config(
        cls,
        config: KnowledgeConfig | None = None,
        *,
        production: bool = False,
        embeddings: EmbeddingProvider,
        extractor: EntityRelationExtractor,
        clock: Callable[[], float] | None = None,
        store: KnowledgeStore | None = None,
        cache: KVCache | None = None,
    ) -> KnowledgeRuntime:
        resolved = config or knowledge_config_from_env()
        use_postgres = production or bool(resolved.postgres_dsn)
        if production and not resolved.postgres_dsn:
            raise KnowledgeConfigError("production knowledge requires postgres_dsn")
        if use_postgres:
            if not resolved.postgres_dsn:
                raise KnowledgeConfigError("production knowledge requires postgres_dsn")
            from .stores.postgres import PostgresKnowledgeStore
            from .stores.redis_cache import RedisKVCache

            durable_store = store or PostgresKnowledgeStore(
                resolved.postgres_dsn,
                timeout_seconds=resolved.retrieval_timeout_seconds,
            )
            if cache is not None:
                durable_cache = cache
            elif resolved.redis_dsn:
                durable_cache = RedisKVCache(resolved.redis_dsn)
            else:
                durable_cache = NullKVCache()
            return cls(
                config=resolved,
                store=durable_store,
                cache=durable_cache,
                embeddings=embeddings,
                extractor=extractor,
                clock=clock,
            )
        return cls.in_memory(
            config=resolved,
            embeddings=embeddings,
            extractor=extractor,
            clock=clock,
            store=store,
            cache=cache,
        )

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
        profile = self.store.embedding_profile(tenant_id, collection_id)
        return {
            "tenant_id": tenant_id,
            "collection_id": collection_id,
            "collection_revision": self.store.collection_revision(
                tenant_id, collection_id
            ),
            "embedding_profile": None if profile is None else profile.profile_id,
        }


__all__ = ["KnowledgeRuntime"]
