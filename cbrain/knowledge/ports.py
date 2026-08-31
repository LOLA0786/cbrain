"""Provider-neutral knowledge ports. Selection is deployment-owned."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from .contracts import (
    CacheKey,
    Citation,
    DocumentRevision,
    EmbeddingProfile,
    GraphEdge,
    GraphNode,
    GraphPath,
    KnowledgeChunk,
    RetrievalQuery,
    RetrievedContext,
    SourceDocument,
)


class DocumentRepository(Protocol):
    def get_revision(
        self, tenant_id: str, collection_id: str, source_id: str
    ) -> DocumentRevision | None: ...

    def list_live_chunks(
        self,
        *,
        tenant_id: str,
        collection_id: str,
        principal_id: str,
    ) -> tuple[KnowledgeChunk, ...]: ...

    def load_chunks(self, chunk_ids: Sequence[str]) -> tuple[KnowledgeChunk, ...]: ...

    def collection_revision(self, tenant_id: str, collection_id: str) -> int: ...

    def embedding_profile(
        self, tenant_id: str, collection_id: str
    ) -> EmbeddingProfile | None: ...


class VectorIndex(Protocol):
    def search(
        self,
        *,
        tenant_id: str,
        collection_id: str,
        principal_id: str,
        vector: tuple[float, ...],
        limit: int,
    ) -> tuple[tuple[str, float], ...]: ...


class KeywordIndex(Protocol):
    def search(
        self,
        *,
        tenant_id: str,
        collection_id: str,
        principal_id: str,
        query: str,
        limit: int,
    ) -> tuple[tuple[str, float], ...]: ...


class GraphRepository(Protocol):
    def upsert_nodes(self, nodes: Sequence[GraphNode]) -> None: ...

    def upsert_edges(self, edges: Sequence[GraphEdge]) -> None: ...

    def traverse(
        self,
        *,
        tenant_id: str,
        principal_id: str,
        seed_node_ids: Sequence[str],
        max_depth: int,
        max_nodes: int,
    ) -> tuple[GraphPath, ...]: ...

    def nodes_for_chunks(self, chunk_ids: Sequence[str]) -> tuple[GraphNode, ...]: ...


class KVCache(Protocol):
    def get(self, key: CacheKey) -> tuple[str, ...] | None: ...

    def set(
        self, key: CacheKey, chunk_ids: Sequence[str], ttl_seconds: float
    ) -> None: ...


class EmbeddingProvider(Protocol):
    def embed(
        self, texts: Sequence[str], profile: EmbeddingProfile
    ) -> tuple[tuple[float, ...], ...]: ...


class EntityRelationExtractor(Protocol):
    def extract(
        self, chunk: KnowledgeChunk
    ) -> tuple[tuple[GraphNode, ...], tuple[GraphEdge, ...]]: ...


class Reranker(Protocol):
    def rerank(self, hits: Sequence[object]) -> tuple[object, ...]: ...


class KnowledgeContextProvider(Protocol):
    def retrieve(self, query: RetrievalQuery) -> RetrievedContext: ...


class KnowledgePublisher(Protocol):
    def publish_revision(
        self,
        *,
        document: SourceDocument,
        revision: DocumentRevision,
        chunks: Sequence[KnowledgeChunk],
        embeddings: Mapping[str, tuple[float, ...]],
        nodes: Sequence[GraphNode],
        edges: Sequence[GraphEdge],
        profile: EmbeddingProfile,
    ) -> int: ...

    def tombstone(
        self, tenant_id: str, collection_id: str, source_id: str, created_at: float
    ) -> int: ...


class CitationLoader(Protocol):
    def citations_for(
        self, chunks: Sequence[KnowledgeChunk]
    ) -> tuple[Citation, ...]: ...


__all__ = [
    "CitationLoader",
    "DocumentRepository",
    "EmbeddingProvider",
    "EntityRelationExtractor",
    "GraphRepository",
    "KVCache",
    "KeywordIndex",
    "KnowledgeContextProvider",
    "KnowledgePublisher",
    "Reranker",
    "VectorIndex",
]
