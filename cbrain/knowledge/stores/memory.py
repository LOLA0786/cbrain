"""Deterministic in-memory knowledge adapters for unit tests."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Mapping, Sequence

from ..chunking import authorized
from ..contracts import (
    KNOWLEDGE_SCHEMA_VERSION,
    CacheKey,
    Citation,
    DocumentRevision,
    EmbeddingProfile,
    GraphEdge,
    GraphNode,
    GraphPath,
    KnowledgeChunk,
    SourceDocument,
    finite_positive_ttl,
)
from ..errors import KnowledgeError, KnowledgeUnavailable
from ..graph import bounded_walk


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm_l = math.sqrt(sum(a * a for a in left))
    norm_r = math.sqrt(sum(b * b for b in right))
    if norm_l == 0.0 or norm_r == 0.0:
        return 0.0
    return dot / (norm_l * norm_r)


class NullKVCache:
    """Always misses. Used when Redis is not configured."""

    def get(self, key: CacheKey) -> tuple[str, ...] | None:
        del key
        return None

    def set(self, key: CacheKey, chunk_ids: Sequence[str], ttl_seconds: float) -> None:
        del key, chunk_ids
        finite_positive_ttl(ttl_seconds)


class InMemoryKVCache:
    def __init__(self) -> None:
        self._entries: dict[str, tuple[float, tuple[str, ...]]] = {}
        self._lock = threading.Lock()
        self.fail_closed = False

    def get(self, key: CacheKey) -> tuple[str, ...] | None:
        if self.fail_closed:
            raise KnowledgeError("cache unavailable")
        encoded = key.encoded()
        now = time.time()
        with self._lock:
            item = self._entries.get(encoded)
            if item is None:
                return None
            expires_at, value = item
            if now >= expires_at:
                del self._entries[encoded]
                return None
            return value

    def set(self, key: CacheKey, chunk_ids: Sequence[str], ttl_seconds: float) -> None:
        if self.fail_closed:
            raise KnowledgeError("cache unavailable")
        ttl = finite_positive_ttl(ttl_seconds)
        with self._lock:
            self._entries[key.encoded()] = (time.time() + ttl, tuple(chunk_ids))


class InMemoryKnowledgeStore:
    """Transactional in-memory document, vector, keyword, and graph store."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._revisions: dict[tuple[str, str, str], DocumentRevision] = {}
        self._chunks: dict[str, KnowledgeChunk] = {}
        self._live_chunks: dict[tuple[str, str, str], tuple[str, ...]] = {}
        self._embeddings: dict[str, tuple[float, ...]] = {}
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, GraphEdge] = {}
        self._collection_revision: dict[tuple[str, str], int] = {}
        self._profiles: dict[tuple[str, str], EmbeddingProfile] = {}
        self.fail_reads = False

    def _guard(self) -> None:
        if self.fail_reads:
            raise KnowledgeUnavailable("KNOWLEDGE_UNAVAILABLE")

    def get_revision(
        self, tenant_id: str, collection_id: str, source_id: str
    ) -> DocumentRevision | None:
        self._guard()
        with self._lock:
            return self._revisions.get((tenant_id, collection_id, source_id))

    def list_live_chunks(
        self,
        *,
        tenant_id: str,
        collection_id: str,
        principal_id: str,
    ) -> tuple[KnowledgeChunk, ...]:
        self._guard()
        with self._lock:
            selected = [
                chunk
                for chunk in self._chunks.values()
                if chunk.tenant_id == tenant_id
                and chunk.collection_id == collection_id
                and authorized(chunk, principal_id)
                and not self._is_tombstoned(chunk)
            ]
        return tuple(sorted(selected, key=lambda item: item.chunk_id))

    def load_chunks(self, chunk_ids: Sequence[str]) -> tuple[KnowledgeChunk, ...]:
        self._guard()
        with self._lock:
            loaded = []
            for chunk_id in chunk_ids:
                chunk = self._chunks.get(chunk_id)
                if chunk is None or self._is_tombstoned(chunk):
                    continue
                loaded.append(chunk)
        return tuple(loaded)

    def collection_revision(self, tenant_id: str, collection_id: str) -> int:
        self._guard()
        with self._lock:
            return self._collection_revision.get((tenant_id, collection_id), 0)

    def embedding_profile(
        self, tenant_id: str, collection_id: str
    ) -> EmbeddingProfile | None:
        self._guard()
        with self._lock:
            return self._profiles.get((tenant_id, collection_id))

    def search_vector(
        self,
        *,
        tenant_id: str,
        collection_id: str,
        principal_id: str,
        vector: tuple[float, ...],
        limit: int,
    ) -> tuple[tuple[str, float], ...]:
        self._guard()
        scored: list[tuple[str, float]] = []
        for chunk in self.list_live_chunks(
            tenant_id=tenant_id,
            collection_id=collection_id,
            principal_id=principal_id,
        ):
            embedding = self._embeddings.get(chunk.chunk_id)
            if embedding is None:
                continue
            scored.append((chunk.chunk_id, _cosine(vector, embedding)))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return tuple(scored[:limit])

    def search(
        self,
        *,
        tenant_id: str,
        collection_id: str,
        principal_id: str,
        vector: tuple[float, ...] | None = None,
        query: str | None = None,
        limit: int,
    ) -> tuple[tuple[str, float], ...]:
        if vector is not None:
            return self.search_vector(
                tenant_id=tenant_id,
                collection_id=collection_id,
                principal_id=principal_id,
                vector=vector,
                limit=limit,
            )
        return self.search_keyword(
            tenant_id=tenant_id,
            collection_id=collection_id,
            principal_id=principal_id,
            query=query or "",
            limit=limit,
        )

    def search_keyword(
        self,
        *,
        tenant_id: str,
        collection_id: str,
        principal_id: str,
        query: str,
        limit: int,
    ) -> tuple[tuple[str, float], ...]:
        self._guard()
        terms = set(" ".join(query.split()).casefold().split())
        scored: list[tuple[str, float]] = []
        for chunk in self.list_live_chunks(
            tenant_id=tenant_id,
            collection_id=collection_id,
            principal_id=principal_id,
        ):
            words = set(chunk.text.casefold().split())
            overlap = len(terms & words)
            if overlap == 0:
                continue
            scored.append((chunk.chunk_id, float(overlap)))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return tuple(scored[:limit])

    def upsert_nodes(self, nodes: Sequence[GraphNode]) -> None:
        with self._lock:
            for node in nodes:
                self._nodes[node.node_id] = node

    def upsert_edges(self, edges: Sequence[GraphEdge]) -> None:
        with self._lock:
            for edge in edges:
                self._edges[edge.edge_id] = edge

    def nodes_for_chunks(self, chunk_ids: Sequence[str]) -> tuple[GraphNode, ...]:
        wanted = set(chunk_ids)
        return tuple(
            node
            for node in self._nodes.values()
            if wanted.intersection(node.provenance_chunk_ids)
        )

    def traverse(
        self,
        *,
        tenant_id: str,
        principal_id: str,
        seed_node_ids: Sequence[str],
        max_depth: int,
        max_nodes: int,
    ) -> tuple[GraphPath, ...]:
        self._guard()
        allowed_chunks = {
            chunk.chunk_id
            for chunk in self.list_live_chunks(
                tenant_id=tenant_id,
                collection_id=next(
                    (
                        node.source_id
                        for node in self._nodes.values()
                        if node.tenant_id == tenant_id
                    ),
                    "",
                )
                if False
                else "",
                principal_id=principal_id,
            )
        }
        # ACL: a node is visible if any provenance chunk is authorized for this tenant.
        visible_nodes = {
            node.node_id: node
            for node in self._nodes.values()
            if node.tenant_id == tenant_id
            and any(
                chunk_id in self._authorized_chunk_ids(tenant_id, principal_id)
                for chunk_id in node.provenance_chunk_ids
            )
        }
        del allowed_chunks
        adjacency: dict[str, list[GraphEdge]] = {}
        for edge in self._edges.values():
            if edge.tenant_id != tenant_id:
                continue
            if edge.source_node_id not in visible_nodes:
                continue
            if edge.target_node_id not in visible_nodes:
                continue
            if not any(
                chunk_id in self._authorized_chunk_ids(tenant_id, principal_id)
                for chunk_id in edge.provenance_chunk_ids
            ):
                continue
            adjacency.setdefault(edge.source_node_id, []).append(edge)

        return bounded_walk(
            visible_nodes=visible_nodes,
            adjacency=adjacency,
            seed_node_ids=seed_node_ids,
            max_depth=max_depth,
            max_nodes=max_nodes,
            citations_for=lambda nodes: self._citations_for_nodes(
                nodes, tenant_id, principal_id
            ),
        )

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
    ) -> int:
        key = (document.tenant_id, document.collection_id)
        with self._lock:
            existing = self._profiles.get(key)
            if existing is not None and existing != profile:
                raise KnowledgeError(
                    "embedding profile is immutable after first revision"
                )
            for vector in embeddings.values():
                if len(vector) != profile.dimension:
                    raise KnowledgeError("embedding dimension mismatch")
            previous = self._live_chunks.get(
                (document.tenant_id, document.collection_id, document.source_id), ()
            )
            for chunk_id in previous:
                self._chunks.pop(chunk_id, None)
                self._embeddings.pop(chunk_id, None)
            self._revisions[
                (document.tenant_id, document.collection_id, document.source_id)
            ] = revision
            live_ids: list[str] = []
            for chunk in chunks:
                self._chunks[chunk.chunk_id] = chunk
                live_ids.append(chunk.chunk_id)
            for chunk_id, vector in embeddings.items():
                self._embeddings[chunk_id] = vector
            for node in nodes:
                self._nodes[node.node_id] = node
            for edge in edges:
                self._edges[edge.edge_id] = edge
            self._live_chunks[
                (document.tenant_id, document.collection_id, document.source_id)
            ] = tuple(live_ids)
            self._profiles[key] = profile
            next_revision = self._collection_revision.get(key, 0) + 1
            self._collection_revision[key] = next_revision
            return next_revision

    def tombstone(
        self, tenant_id: str, collection_id: str, source_id: str, created_at: float
    ) -> int:
        with self._lock:
            current = self._revisions.get((tenant_id, collection_id, source_id))
            if current is None:
                raise KnowledgeError("source does not exist")
            self._revisions[(tenant_id, collection_id, source_id)] = DocumentRevision(
                tenant_id=tenant_id,
                collection_id=collection_id,
                source_id=source_id,
                source_revision=current.source_revision + 1,
                content_digest=current.content_digest,
                provenance=current.provenance,
                acl_principals=current.acl_principals,
                created_at=created_at,
                schema_version=KNOWLEDGE_SCHEMA_VERSION,
                tombstoned=True,
            )
            key = (tenant_id, collection_id)
            next_revision = self._collection_revision.get(key, 0) + 1
            self._collection_revision[key] = next_revision
            return next_revision

    def citations_for(self, chunks: Sequence[KnowledgeChunk]) -> tuple[Citation, ...]:
        return tuple(
            Citation(
                tenant_id=chunk.tenant_id,
                source_id=chunk.source_id,
                source_revision=chunk.source_revision,
                content_digest=chunk.content_digest,
                provenance=chunk.provenance,
                acl_principals=chunk.acl_principals,
                created_at=chunk.created_at,
                schema_version=chunk.schema_version,
                chunk_id=chunk.chunk_id,
            )
            for chunk in chunks
        )

    def _is_tombstoned(self, chunk: KnowledgeChunk) -> bool:
        revision = self._revisions.get(
            (chunk.tenant_id, chunk.collection_id, chunk.source_id)
        )
        return revision is None or revision.tombstoned

    def _authorized_chunk_ids(self, tenant_id: str, principal_id: str) -> set[str]:
        return {
            chunk.chunk_id
            for chunk in self._chunks.values()
            if chunk.tenant_id == tenant_id
            and authorized(chunk, principal_id)
            and not self._is_tombstoned(chunk)
        }

    def _citations_for_nodes(
        self,
        nodes: Sequence[GraphNode],
        tenant_id: str,
        principal_id: str,
    ) -> tuple[Citation, ...]:
        allowed = self._authorized_chunk_ids(tenant_id, principal_id)
        chunks: list[KnowledgeChunk] = []
        seen: set[str] = set()
        for node in nodes:
            for chunk_id in node.provenance_chunk_ids:
                if chunk_id not in allowed or chunk_id in seen:
                    continue
                chunk = self._chunks.get(chunk_id)
                if chunk is None:
                    continue
                chunks.append(chunk)
                seen.add(chunk_id)
        return self.citations_for(chunks)


class VectorAdapter:
    def __init__(self, store: InMemoryKnowledgeStore) -> None:
        self._store = store

    def search(
        self,
        *,
        tenant_id: str,
        collection_id: str,
        principal_id: str,
        vector: tuple[float, ...],
        limit: int,
    ) -> tuple[tuple[str, float], ...]:
        return self._store.search_vector(
            tenant_id=tenant_id,
            collection_id=collection_id,
            principal_id=principal_id,
            vector=vector,
            limit=limit,
        )


class KeywordAdapter:
    def __init__(self, store: InMemoryKnowledgeStore) -> None:
        self._store = store

    def search(
        self,
        *,
        tenant_id: str,
        collection_id: str,
        principal_id: str,
        query: str,
        limit: int,
    ) -> tuple[tuple[str, float], ...]:
        return self._store.search_keyword(
            tenant_id=tenant_id,
            collection_id=collection_id,
            principal_id=principal_id,
            query=query,
            limit=limit,
        )


__all__ = [
    "InMemoryKVCache",
    "InMemoryKnowledgeStore",
    "KeywordAdapter",
    "NullKVCache",
    "VectorAdapter",
]
