"""Deterministic ingestion: normalize → chunk → embed → transactional publish."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .chunking import build_chunks
from .configuration import KnowledgeConfig
from .contracts import (
    KNOWLEDGE_SCHEMA_VERSION,
    DocumentRevision,
    EmbeddingProfile,
    IngestionResult,
    SourceDocument,
)
from .errors import KnowledgeError
from .ports import EmbeddingProvider, EntityRelationExtractor, KnowledgeStore


class IngestionPipeline:
    def __init__(
        self,
        *,
        store: KnowledgeStore,
        config: KnowledgeConfig,
        embeddings: EmbeddingProvider,
        extractor: EntityRelationExtractor,
        clock: Callable[[], float],
    ) -> None:
        self._store = store
        self._config = config
        self._embeddings = embeddings
        self._extractor = extractor
        self._clock = clock

    def ingest(self, document: SourceDocument) -> IngestionResult:
        if len(document.text.encode("utf-8")) > self._config.max_document_bytes:
            raise KnowledgeError("document exceeds max_document_bytes")
        existing = self._store.get_revision(
            document.tenant_id, document.collection_id, document.source_id
        )
        digest = document.content_digest()
        if (
            existing is not None
            and not existing.tombstoned
            and existing.content_digest == digest
            and existing.acl_principals == document.acl_principals
            and existing.provenance == document.provenance
        ):
            return IngestionResult(
                tenant_id=document.tenant_id,
                collection_id=document.collection_id,
                source_id=document.source_id,
                source_revision=existing.source_revision,
                collection_revision=self._store.collection_revision(
                    document.tenant_id, document.collection_id
                ),
                chunk_ids=tuple(
                    chunk.chunk_id
                    for chunk in self._store.list_live_chunks(
                        tenant_id=document.tenant_id,
                        collection_id=document.collection_id,
                        principal_id=next(iter(document.acl_principals)),
                    )
                    if chunk.source_id == document.source_id
                ),
                content_digest=digest,
                idempotent=True,
                published=True,
            )
        published_profile = self._store.embedding_profile(
            document.tenant_id, document.collection_id
        )
        profile = self._require_profile(published_profile)
        source_revision = 1 if existing is None else existing.source_revision + 1
        created_at = self._clock()
        chunks = build_chunks(
            document,
            source_revision=source_revision,
            config=self._config,
            created_at=created_at,
        )
        vectors = self._embeddings.embed([chunk.text for chunk in chunks], profile)
        if any(len(vector) != profile.dimension for vector in vectors):
            raise KnowledgeError("embedding dimension mismatch")
        embeddings = {
            chunk.chunk_id: vector
            for chunk, vector in zip(chunks, vectors, strict=True)
        }
        nodes: list[Any] = []
        edges: list[Any] = []
        for chunk in chunks:
            extracted_nodes, extracted_edges = self._extractor.extract(chunk)
            nodes.extend(extracted_nodes)
            edges.extend(extracted_edges)
        revision = DocumentRevision(
            tenant_id=document.tenant_id,
            collection_id=document.collection_id,
            source_id=document.source_id,
            source_revision=source_revision,
            content_digest=digest,
            provenance=document.provenance,
            acl_principals=document.acl_principals,
            created_at=created_at,
            schema_version=KNOWLEDGE_SCHEMA_VERSION,
        )
        collection_revision = self._store.publish_revision(
            document=document,
            revision=revision,
            chunks=chunks,
            embeddings=embeddings,
            nodes=nodes,
            edges=edges,
            profile=profile,
        )
        return IngestionResult(
            tenant_id=document.tenant_id,
            collection_id=document.collection_id,
            source_id=document.source_id,
            source_revision=source_revision,
            collection_revision=collection_revision,
            chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
            content_digest=digest,
            idempotent=False,
            published=True,
        )

    def tombstone(
        self, tenant_id: str, collection_id: str, source_id: str
    ) -> IngestionResult:
        created_at = self._clock()
        current = self._store.get_revision(tenant_id, collection_id, source_id)
        if current is None:
            raise KnowledgeError("source does not exist")
        collection_revision = self._store.tombstone(
            tenant_id, collection_id, source_id, created_at
        )
        return IngestionResult(
            tenant_id=tenant_id,
            collection_id=collection_id,
            source_id=source_id,
            source_revision=current.source_revision + 1,
            collection_revision=collection_revision,
            chunk_ids=(),
            content_digest=current.content_digest,
            idempotent=False,
            published=True,
        )

    def _require_profile(self, published: EmbeddingProfile | None) -> EmbeddingProfile:
        requested = self._config.embedding_profile
        if published is None:
            return requested
        if published != requested:
            raise KnowledgeError("embedding profile is immutable after first revision")
        return published


__all__ = ["IngestionPipeline"]
