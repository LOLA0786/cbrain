"""Immutable knowledge contracts. Retrieved data is untrusted context only."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from .errors import KnowledgeError

KNOWLEDGE_SCHEMA_VERSION = 1
UNTRUSTED_SOURCE_CONTEXT = "UNTRUSTED_SOURCE_CONTEXT"
FORBIDDEN_METADATA_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "connection",
        "credential",
        "credentials",
        "database_url",
        "dsn",
        "password",
        "private_key",
        "redis_url",
        "secret",
        "secrets",
        "token",
        "url",
        "urls",
    }
)


class DistanceMetric(StrEnum):
    COSINE = "cosine"


class KnowledgeRequirement(StrEnum):
    OPTIONAL = "optional"
    REQUIRED_FOR_TOOLS = "required_for_tools"


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def sha256_hex(payload: Mapping[str, Any] | str) -> str:
    if isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raw = canonical_json(payload).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise KnowledgeError(f"{field_name} must be a non-empty string")
    if "\x00" in value:
        raise KnowledgeError(f"{field_name} must not contain null bytes")
    return value.strip()


def require_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise KnowledgeError(f"{field_name} must be a positive integer")
    return value


def finite_timestamp(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise KnowledgeError(f"{field_name} must be a finite number")
    timestamp = float(value)
    if not math.isfinite(timestamp):
        raise KnowledgeError(f"{field_name} must be a finite number")
    return timestamp


def finite_positive_ttl(value: object, field_name: str = "ttl_seconds") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise KnowledgeError(f"{field_name} must be a finite positive number")
    ttl = float(value)
    if not math.isfinite(ttl) or ttl <= 0:
        raise KnowledgeError(f"{field_name} must be a finite positive number")
    return ttl


def reject_forbidden_metadata(metadata: Mapping[str, Any]) -> None:
    for key in metadata:
        folded = str(key).casefold()
        if folded in FORBIDDEN_METADATA_KEYS:
            raise KnowledgeError(f"forbidden metadata field {key!r}")


def _frozen_text_set(
    values: Sequence[str] | frozenset[str], field_name: str
) -> frozenset[str]:
    items = values if isinstance(values, frozenset) else frozenset(values)
    cleaned: set[str] = set()
    for item in items:
        cleaned.add(require_text(item, field_name))
    return frozenset(cleaned)


@dataclass(frozen=True, slots=True)
class EmbeddingProfile:
    profile_id: str
    model_id: str
    dimension: int
    distance_metric: DistanceMetric
    normalization_version: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "profile_id", require_text(self.profile_id, "profile_id")
        )
        object.__setattr__(self, "model_id", require_text(self.model_id, "model_id"))
        object.__setattr__(
            self, "dimension", require_positive_int(self.dimension, "dimension")
        )
        if not isinstance(self.distance_metric, DistanceMetric):
            raise KnowledgeError("distance_metric must be a defined DistanceMetric")
        object.__setattr__(
            self,
            "normalization_version",
            require_text(self.normalization_version, "normalization_version"),
        )

    def digest(self) -> str:
        return sha256_hex(
            {
                "profile_id": self.profile_id,
                "model_id": self.model_id,
                "dimension": self.dimension,
                "distance_metric": self.distance_metric.value,
                "normalization_version": self.normalization_version,
            }
        )


@dataclass(frozen=True, slots=True)
class SourceDocument:
    tenant_id: str
    collection_id: str
    source_id: str
    media_type: str
    text: str
    acl_principals: frozenset[str]
    created_at: float
    provenance: str
    metadata: Mapping[str, str] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", require_text(self.tenant_id, "tenant_id"))
        object.__setattr__(
            self, "collection_id", require_text(self.collection_id, "collection_id")
        )
        object.__setattr__(self, "source_id", require_text(self.source_id, "source_id"))
        object.__setattr__(
            self, "media_type", require_text(self.media_type, "media_type")
        )
        if self.media_type not in {"text/plain", "text/markdown"}:
            raise KnowledgeError("unsupported document media type")
        if not isinstance(self.text, str) or not self.text:
            raise KnowledgeError("document text must be a non-empty string")
        if "\x00" in self.text:
            raise KnowledgeError("document text must not contain null bytes")
        object.__setattr__(
            self,
            "acl_principals",
            _frozen_text_set(self.acl_principals, "acl_principals"),
        )
        object.__setattr__(
            self, "created_at", finite_timestamp(self.created_at, "created_at")
        )
        object.__setattr__(
            self, "provenance", require_text(self.provenance, "provenance")
        )
        if not isinstance(self.metadata, Mapping):
            raise KnowledgeError("metadata must be a mapping")
        reject_forbidden_metadata(self.metadata)
        cleaned: dict[str, str] = {}
        for key, value in self.metadata.items():
            cleaned[require_text(key, "metadata key")] = require_text(
                value, "metadata value"
            )
        object.__setattr__(self, "metadata", MappingProxyType(cleaned))

    def content_digest(self) -> str:
        return sha256_hex(self.text)


@dataclass(frozen=True, slots=True)
class DocumentRevision:
    tenant_id: str
    collection_id: str
    source_id: str
    source_revision: int
    content_digest: str
    provenance: str
    acl_principals: frozenset[str]
    created_at: float
    schema_version: int
    tombstoned: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", require_text(self.tenant_id, "tenant_id"))
        object.__setattr__(
            self, "collection_id", require_text(self.collection_id, "collection_id")
        )
        object.__setattr__(self, "source_id", require_text(self.source_id, "source_id"))
        object.__setattr__(
            self,
            "source_revision",
            require_positive_int(self.source_revision, "source_revision"),
        )
        object.__setattr__(
            self, "content_digest", require_text(self.content_digest, "content_digest")
        )
        object.__setattr__(
            self, "provenance", require_text(self.provenance, "provenance")
        )
        object.__setattr__(
            self,
            "acl_principals",
            _frozen_text_set(self.acl_principals, "acl_principals"),
        )
        object.__setattr__(
            self, "created_at", finite_timestamp(self.created_at, "created_at")
        )
        object.__setattr__(
            self,
            "schema_version",
            require_positive_int(self.schema_version, "schema_version"),
        )
        if not isinstance(self.tombstoned, bool):
            raise KnowledgeError("tombstoned must be a bool")


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    chunk_id: str
    tenant_id: str
    collection_id: str
    source_id: str
    source_revision: int
    chunk_index: int
    text: str
    content_digest: str
    provenance: str
    acl_principals: frozenset[str]
    created_at: float
    schema_version: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "chunk_id", require_text(self.chunk_id, "chunk_id"))
        object.__setattr__(self, "tenant_id", require_text(self.tenant_id, "tenant_id"))
        object.__setattr__(
            self, "collection_id", require_text(self.collection_id, "collection_id")
        )
        object.__setattr__(self, "source_id", require_text(self.source_id, "source_id"))
        object.__setattr__(
            self,
            "source_revision",
            require_positive_int(self.source_revision, "source_revision"),
        )
        if isinstance(self.chunk_index, bool) or not isinstance(self.chunk_index, int):
            raise KnowledgeError("chunk_index must be an integer")
        if self.chunk_index < 0:
            raise KnowledgeError("chunk_index must be non-negative")
        if not isinstance(self.text, str) or not self.text:
            raise KnowledgeError("chunk text must be a non-empty string")
        object.__setattr__(
            self, "content_digest", require_text(self.content_digest, "content_digest")
        )
        object.__setattr__(
            self, "provenance", require_text(self.provenance, "provenance")
        )
        object.__setattr__(
            self,
            "acl_principals",
            _frozen_text_set(self.acl_principals, "acl_principals"),
        )
        object.__setattr__(
            self, "created_at", finite_timestamp(self.created_at, "created_at")
        )
        object.__setattr__(
            self,
            "schema_version",
            require_positive_int(self.schema_version, "schema_version"),
        )


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    tenant_id: str
    collection_id: str
    principal_id: str
    text: str
    top_k: int
    created_at: float
    policy_version: str = "v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", require_text(self.tenant_id, "tenant_id"))
        object.__setattr__(
            self, "collection_id", require_text(self.collection_id, "collection_id")
        )
        object.__setattr__(
            self, "principal_id", require_text(self.principal_id, "principal_id")
        )
        object.__setattr__(self, "text", require_text(self.text, "query text"))
        object.__setattr__(self, "top_k", require_positive_int(self.top_k, "top_k"))
        object.__setattr__(
            self, "created_at", finite_timestamp(self.created_at, "created_at")
        )
        object.__setattr__(
            self, "policy_version", require_text(self.policy_version, "policy_version")
        )

    def digest(self) -> str:
        return sha256_hex(
            {
                "tenant_id": self.tenant_id,
                "collection_id": self.collection_id,
                "principal_id": self.principal_id,
                "text": " ".join(self.text.split()).casefold(),
                "top_k": self.top_k,
                "policy_version": self.policy_version,
            }
        )


@dataclass(frozen=True, slots=True)
class Citation:
    tenant_id: str
    source_id: str
    source_revision: int
    content_digest: str
    provenance: str
    acl_principals: frozenset[str]
    created_at: float
    schema_version: int
    chunk_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", require_text(self.tenant_id, "tenant_id"))
        object.__setattr__(self, "source_id", require_text(self.source_id, "source_id"))
        object.__setattr__(
            self,
            "source_revision",
            require_positive_int(self.source_revision, "source_revision"),
        )
        object.__setattr__(
            self, "content_digest", require_text(self.content_digest, "content_digest")
        )
        object.__setattr__(
            self, "provenance", require_text(self.provenance, "provenance")
        )
        object.__setattr__(
            self,
            "acl_principals",
            _frozen_text_set(self.acl_principals, "acl_principals"),
        )
        object.__setattr__(
            self, "created_at", finite_timestamp(self.created_at, "created_at")
        )
        object.__setattr__(
            self,
            "schema_version",
            require_positive_int(self.schema_version, "schema_version"),
        )
        object.__setattr__(self, "chunk_id", require_text(self.chunk_id, "chunk_id"))


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    chunk: KnowledgeChunk
    score: float
    citation: Citation
    rank: int

    def __post_init__(self) -> None:
        if not isinstance(self.chunk, KnowledgeChunk):
            raise KnowledgeError("hit chunk is invalid")
        if not isinstance(self.citation, Citation):
            raise KnowledgeError("hit citation is invalid")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise KnowledgeError("hit score must be finite")
        if not math.isfinite(float(self.score)):
            raise KnowledgeError("hit score must be finite")
        object.__setattr__(self, "rank", require_positive_int(self.rank, "rank"))
        if (
            self.citation.chunk_id != self.chunk.chunk_id
            or self.citation.content_digest != self.chunk.content_digest
            or self.citation.source_id != self.chunk.source_id
            or self.citation.tenant_id != self.chunk.tenant_id
        ):
            raise KnowledgeError("citation does not match chunk")


@dataclass(frozen=True, slots=True)
class RetrievalDiagnostics:
    vector_candidate_count: int
    keyword_candidate_count: int
    graph_expanded_count: int
    cache_hit: bool
    latency_ms: Mapping[str, float]
    final_hit_count: int
    truncated: bool

    def __post_init__(self) -> None:
        for field_name in (
            "vector_candidate_count",
            "keyword_candidate_count",
            "graph_expanded_count",
            "final_hit_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise KnowledgeError(f"{field_name} must be a non-negative integer")
        if not isinstance(self.cache_hit, bool):
            raise KnowledgeError("cache_hit must be a bool")
        if not isinstance(self.truncated, bool):
            raise KnowledgeError("truncated must be a bool")
        cleaned: dict[str, float] = {}
        for key, value in dict(self.latency_ms).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise KnowledgeError("latency values must be finite")
            number = float(value)
            if not math.isfinite(number):
                raise KnowledgeError("latency values must be finite")
            cleaned[require_text(key, "latency stage")] = number
        object.__setattr__(self, "latency_ms", MappingProxyType(cleaned))


@dataclass(frozen=True, slots=True)
class RetrievedContext:
    trust: str
    query_digest: str
    hits: tuple[RetrievalHit, ...]
    diagnostics: RetrievalDiagnostics
    quoted_evidence: str

    def __post_init__(self) -> None:
        if self.trust != UNTRUSTED_SOURCE_CONTEXT:
            raise KnowledgeError("retrieved context must be marked untrusted")
        object.__setattr__(
            self, "query_digest", require_text(self.query_digest, "query_digest")
        )
        if not isinstance(self.hits, tuple):
            raise KnowledgeError("hits must be a tuple")
        if not isinstance(self.diagnostics, RetrievalDiagnostics):
            raise KnowledgeError("diagnostics are required")
        if not isinstance(self.quoted_evidence, str):
            raise KnowledgeError("quoted_evidence must be a string")


@dataclass(frozen=True, slots=True)
class GraphNode:
    node_id: str
    tenant_id: str
    canonical_type: str
    canonical_name: str
    attributes: Mapping[str, str]
    provenance_chunk_ids: tuple[str, ...]
    confidence: float
    valid_from: float
    valid_to: float | None
    source_revision: int
    source_id: str
    schema_version: int
    merge_score: float | None = None
    merge_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", require_text(self.node_id, "node_id"))
        object.__setattr__(self, "tenant_id", require_text(self.tenant_id, "tenant_id"))
        object.__setattr__(
            self, "canonical_type", require_text(self.canonical_type, "canonical_type")
        )
        object.__setattr__(
            self, "canonical_name", require_text(self.canonical_name, "canonical_name")
        )
        if not self.provenance_chunk_ids:
            raise KnowledgeError("graph node requires source provenance")
        object.__setattr__(
            self,
            "provenance_chunk_ids",
            tuple(
                require_text(item, "provenance chunk id")
                for item in self.provenance_chunk_ids
            ),
        )
        object.__setattr__(self, "source_id", require_text(self.source_id, "source_id"))
        object.__setattr__(
            self,
            "source_revision",
            require_positive_int(self.source_revision, "source_revision"),
        )
        if isinstance(self.confidence, bool) or not isinstance(
            self.confidence, (int, float)
        ):
            raise KnowledgeError("confidence must be finite")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise KnowledgeError("confidence must be between 0 and 1")
        object.__setattr__(
            self, "valid_from", finite_timestamp(self.valid_from, "valid_from")
        )
        if self.valid_to is not None:
            object.__setattr__(
                self, "valid_to", finite_timestamp(self.valid_to, "valid_to")
            )
        object.__setattr__(
            self,
            "schema_version",
            require_positive_int(self.schema_version, "schema_version"),
        )
        reject_forbidden_metadata(self.attributes)
        object.__setattr__(
            self,
            "attributes",
            MappingProxyType(
                {
                    require_text(key, "attribute"): require_text(value, "attribute")
                    for key, value in dict(self.attributes).items()
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class GraphEdge:
    edge_id: str
    tenant_id: str
    source_node_id: str
    relation_type: str
    target_node_id: str
    provenance_chunk_ids: tuple[str, ...]
    confidence: float
    valid_from: float
    valid_to: float | None
    source_revision: int
    source_id: str
    schema_version: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "edge_id", require_text(self.edge_id, "edge_id"))
        object.__setattr__(self, "tenant_id", require_text(self.tenant_id, "tenant_id"))
        object.__setattr__(
            self, "source_node_id", require_text(self.source_node_id, "source_node_id")
        )
        object.__setattr__(
            self, "relation_type", require_text(self.relation_type, "relation_type")
        )
        object.__setattr__(
            self, "target_node_id", require_text(self.target_node_id, "target_node_id")
        )
        if not self.provenance_chunk_ids:
            raise KnowledgeError("graph edge requires source provenance")
        object.__setattr__(
            self,
            "provenance_chunk_ids",
            tuple(
                require_text(item, "provenance chunk id")
                for item in self.provenance_chunk_ids
            ),
        )
        object.__setattr__(self, "source_id", require_text(self.source_id, "source_id"))
        object.__setattr__(
            self,
            "source_revision",
            require_positive_int(self.source_revision, "source_revision"),
        )
        if isinstance(self.confidence, bool) or not isinstance(
            self.confidence, (int, float)
        ):
            raise KnowledgeError("confidence must be finite")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise KnowledgeError("confidence must be between 0 and 1")
        object.__setattr__(
            self, "valid_from", finite_timestamp(self.valid_from, "valid_from")
        )
        if self.valid_to is not None:
            object.__setattr__(
                self, "valid_to", finite_timestamp(self.valid_to, "valid_to")
            )
        object.__setattr__(
            self,
            "schema_version",
            require_positive_int(self.schema_version, "schema_version"),
        )


@dataclass(frozen=True, slots=True)
class GraphPath:
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    citations: tuple[Citation, ...]

    def __post_init__(self) -> None:
        if not self.nodes:
            raise KnowledgeError("graph path requires nodes")
        if not self.citations:
            raise KnowledgeError("graph path requires citations")


@dataclass(frozen=True, slots=True)
class CacheKey:
    schema_version: int
    tenant_id: str
    principal_acl_digest: str
    collection_id: str
    collection_revision: int
    query_digest: str
    embedding_profile_digest: str
    retrieval_config_digest: str
    policy_version: str
    top_k: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            require_positive_int(self.schema_version, "schema_version"),
        )
        object.__setattr__(self, "tenant_id", require_text(self.tenant_id, "tenant_id"))
        object.__setattr__(
            self,
            "principal_acl_digest",
            require_text(self.principal_acl_digest, "principal_acl_digest"),
        )
        object.__setattr__(
            self, "collection_id", require_text(self.collection_id, "collection_id")
        )
        if isinstance(self.collection_revision, bool) or not isinstance(
            self.collection_revision, int
        ):
            raise KnowledgeError("collection_revision must be an integer")
        if self.collection_revision < 0:
            raise KnowledgeError("collection_revision must be non-negative")
        object.__setattr__(
            self, "query_digest", require_text(self.query_digest, "query_digest")
        )
        object.__setattr__(
            self,
            "embedding_profile_digest",
            require_text(self.embedding_profile_digest, "embedding_profile_digest"),
        )
        object.__setattr__(
            self,
            "retrieval_config_digest",
            require_text(self.retrieval_config_digest, "retrieval_config_digest"),
        )
        object.__setattr__(
            self, "policy_version", require_text(self.policy_version, "policy_version")
        )
        object.__setattr__(self, "top_k", require_positive_int(self.top_k, "top_k"))

    def encoded(self) -> str:
        return sha256_hex(
            {
                "schema_version": self.schema_version,
                "tenant_id": self.tenant_id,
                "principal_acl_digest": self.principal_acl_digest,
                "collection_id": self.collection_id,
                "collection_revision": self.collection_revision,
                "query_digest": self.query_digest,
                "embedding_profile_digest": self.embedding_profile_digest,
                "retrieval_config_digest": self.retrieval_config_digest,
                "policy_version": self.policy_version,
                "top_k": self.top_k,
            }
        )


@dataclass(frozen=True, slots=True)
class IngestionResult:
    tenant_id: str
    collection_id: str
    source_id: str
    source_revision: int
    collection_revision: int
    chunk_ids: tuple[str, ...]
    content_digest: str
    idempotent: bool
    published: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", require_text(self.tenant_id, "tenant_id"))
        object.__setattr__(
            self, "collection_id", require_text(self.collection_id, "collection_id")
        )
        object.__setattr__(self, "source_id", require_text(self.source_id, "source_id"))
        object.__setattr__(
            self,
            "source_revision",
            require_positive_int(self.source_revision, "source_revision"),
        )
        if isinstance(self.collection_revision, bool) or not isinstance(
            self.collection_revision, int
        ):
            raise KnowledgeError("collection_revision must be an integer")
        if self.collection_revision < 0:
            raise KnowledgeError("collection_revision must be non-negative")
        object.__setattr__(
            self, "content_digest", require_text(self.content_digest, "content_digest")
        )
        if not isinstance(self.idempotent, bool) or not isinstance(
            self.published, bool
        ):
            raise KnowledgeError("ingestion flags must be bool")


__all__ = [
    "FORBIDDEN_METADATA_KEYS",
    "KNOWLEDGE_SCHEMA_VERSION",
    "UNTRUSTED_SOURCE_CONTEXT",
    "CacheKey",
    "Citation",
    "DistanceMetric",
    "DocumentRevision",
    "EmbeddingProfile",
    "GraphEdge",
    "GraphNode",
    "GraphPath",
    "IngestionResult",
    "KnowledgeChunk",
    "KnowledgeRequirement",
    "RetrievedContext",
    "RetrievalDiagnostics",
    "RetrievalHit",
    "RetrievalQuery",
    "SourceDocument",
    "canonical_json",
    "finite_positive_ttl",
    "finite_timestamp",
    "require_positive_int",
    "require_text",
    "sha256_hex",
]
