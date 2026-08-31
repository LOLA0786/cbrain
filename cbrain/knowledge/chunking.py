"""Deterministic document normalization and chunking."""

from __future__ import annotations

from collections.abc import Sequence

from .configuration import KnowledgeConfig
from .contracts import (
    KNOWLEDGE_SCHEMA_VERSION,
    KnowledgeChunk,
    SourceDocument,
    require_positive_int,
    sha256_hex,
)
from .errors import KnowledgeError


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def text_digest(text: str) -> str:
    return sha256_hex(normalize_text(text))


def chunk_id_for(
    *,
    tenant_id: str,
    source_id: str,
    source_revision: int,
    chunk_index: int,
    normalized_text: str,
) -> str:
    return sha256_hex(
        {
            "tenant_id": tenant_id,
            "source_id": source_id,
            "source_revision": source_revision,
            "chunk_index": chunk_index,
            "normalized_text_digest": text_digest(normalized_text),
        }
    )


def split_chunks(text: str, *, size: int, overlap: int) -> tuple[str, ...]:
    require_positive_int(size, "chunk_size")
    normalized = normalize_text(text)
    if not normalized:
        raise KnowledgeError("document text is empty after normalization")
    if overlap < 0 or overlap >= size:
        raise KnowledgeError("chunk_overlap must be < chunk_size")
    if len(normalized) <= size:
        return (normalized,)
    parts: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + size)
        parts.append(normalized[start:end])
        if end >= len(normalized):
            break
        start = end - overlap
    return tuple(parts)


def build_chunks(
    document: SourceDocument,
    *,
    source_revision: int,
    config: KnowledgeConfig,
    created_at: float,
) -> tuple[KnowledgeChunk, ...]:
    texts = split_chunks(
        document.text, size=config.chunk_size, overlap=config.chunk_overlap
    )
    chunks: list[KnowledgeChunk] = []
    for index, text in enumerate(texts):
        digest = text_digest(text)
        chunks.append(
            KnowledgeChunk(
                chunk_id=chunk_id_for(
                    tenant_id=document.tenant_id,
                    source_id=document.source_id,
                    source_revision=source_revision,
                    chunk_index=index,
                    normalized_text=text,
                ),
                tenant_id=document.tenant_id,
                collection_id=document.collection_id,
                source_id=document.source_id,
                source_revision=source_revision,
                chunk_index=index,
                text=text,
                content_digest=digest,
                provenance=document.provenance,
                acl_principals=document.acl_principals,
                created_at=created_at,
                schema_version=KNOWLEDGE_SCHEMA_VERSION,
            )
        )
    return tuple(chunks)


def authorized(chunk: KnowledgeChunk, principal_id: str) -> bool:
    return principal_id in chunk.acl_principals or "*" in chunk.acl_principals


def select_diverse(
    chunks: Sequence[KnowledgeChunk], limit: int
) -> tuple[KnowledgeChunk, ...]:
    chosen: list[KnowledgeChunk] = []
    seen_sources: set[str] = set()
    for chunk in chunks:
        if chunk.source_id in seen_sources and len(chosen) < limit:
            continue
        chosen.append(chunk)
        seen_sources.add(chunk.source_id)
        if len(chosen) >= limit:
            return tuple(chosen)
    for chunk in chunks:
        if chunk in chosen:
            continue
        chosen.append(chunk)
        if len(chosen) >= limit:
            break
    return tuple(chosen)


__all__ = [
    "authorized",
    "build_chunks",
    "chunk_id_for",
    "normalize_text",
    "select_diverse",
    "split_chunks",
    "text_digest",
]
