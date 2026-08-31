"""Deployment-owned knowledge configuration. Not model-tool arguments."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import (
    DistanceMetric,
    EmbeddingProfile,
    finite_positive_ttl,
    require_positive_int,
    require_text,
)
from .errors import KnowledgeConfigError


def _optional_dsn(value: str | None, field_name: str) -> str | None:
    if value is None or value == "":
        return None
    text = require_text(value, field_name)
    if any(token in text.casefold() for token in ("password=", "secret=", "api_key=")):
        # Accept operator DSNs but never echo them later.
        return text
    return text


@dataclass(frozen=True, slots=True)
class KnowledgeConfig:
    postgres_dsn: str | None
    redis_dsn: str | None
    cache_ttl_seconds: float
    max_document_bytes: int
    chunk_size: int
    chunk_overlap: int
    top_k: int
    vector_candidate_count: int
    keyword_candidate_count: int
    graph_max_depth: int
    graph_max_nodes: int
    retrieval_timeout_seconds: float
    embedding_profile: EmbeddingProfile
    max_context_chunks: int
    max_context_bytes: int
    max_context_tokens: int
    policy_version: str = "v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "postgres_dsn", _optional_dsn(self.postgres_dsn, "postgres_dsn")
        )
        object.__setattr__(
            self, "redis_dsn", _optional_dsn(self.redis_dsn, "redis_dsn")
        )
        try:
            object.__setattr__(
                self,
                "cache_ttl_seconds",
                finite_positive_ttl(self.cache_ttl_seconds, "cache_ttl_seconds"),
            )
            object.__setattr__(
                self,
                "max_document_bytes",
                require_positive_int(self.max_document_bytes, "max_document_bytes"),
            )
            object.__setattr__(
                self, "chunk_size", require_positive_int(self.chunk_size, "chunk_size")
            )
            if isinstance(self.chunk_overlap, bool) or not isinstance(
                self.chunk_overlap, int
            ):
                raise KnowledgeConfigError(
                    "chunk_overlap must be a non-negative integer"
                )
            if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
                raise KnowledgeConfigError("chunk_overlap must be < chunk_size")
            object.__setattr__(self, "top_k", require_positive_int(self.top_k, "top_k"))
            object.__setattr__(
                self,
                "vector_candidate_count",
                require_positive_int(
                    self.vector_candidate_count, "vector_candidate_count"
                ),
            )
            object.__setattr__(
                self,
                "keyword_candidate_count",
                require_positive_int(
                    self.keyword_candidate_count, "keyword_candidate_count"
                ),
            )
            object.__setattr__(
                self,
                "graph_max_depth",
                require_positive_int(self.graph_max_depth, "graph_max_depth"),
            )
            object.__setattr__(
                self,
                "graph_max_nodes",
                require_positive_int(self.graph_max_nodes, "graph_max_nodes"),
            )
            object.__setattr__(
                self,
                "retrieval_timeout_seconds",
                finite_positive_ttl(
                    self.retrieval_timeout_seconds, "retrieval_timeout_seconds"
                ),
            )
            object.__setattr__(
                self,
                "max_context_chunks",
                require_positive_int(self.max_context_chunks, "max_context_chunks"),
            )
            object.__setattr__(
                self,
                "max_context_bytes",
                require_positive_int(self.max_context_bytes, "max_context_bytes"),
            )
            object.__setattr__(
                self,
                "max_context_tokens",
                require_positive_int(self.max_context_tokens, "max_context_tokens"),
            )
            object.__setattr__(
                self,
                "policy_version",
                require_text(self.policy_version, "policy_version"),
            )
        except KnowledgeConfigError:
            raise
        except Exception as exc:
            raise KnowledgeConfigError(str(exc)) from exc
        if not isinstance(self.embedding_profile, EmbeddingProfile):
            raise KnowledgeConfigError("embedding_profile is required")

    def retrieval_digest(self) -> str:
        from .contracts import sha256_hex

        return sha256_hex(
            {
                "top_k": self.top_k,
                "vector_candidate_count": self.vector_candidate_count,
                "keyword_candidate_count": self.keyword_candidate_count,
                "graph_max_depth": self.graph_max_depth,
                "graph_max_nodes": self.graph_max_nodes,
                "max_context_chunks": self.max_context_chunks,
                "max_context_bytes": self.max_context_bytes,
                "max_context_tokens": self.max_context_tokens,
                "policy_version": self.policy_version,
            }
        )


def default_knowledge_config() -> KnowledgeConfig:
    return KnowledgeConfig(
        postgres_dsn=None,
        redis_dsn=None,
        cache_ttl_seconds=30.0,
        max_document_bytes=65_536,
        chunk_size=160,
        chunk_overlap=32,
        top_k=4,
        vector_candidate_count=8,
        keyword_candidate_count=8,
        graph_max_depth=2,
        graph_max_nodes=16,
        retrieval_timeout_seconds=2.0,
        embedding_profile=EmbeddingProfile(
            profile_id="deterministic-8",
            model_id="cbrain-deterministic",
            dimension=8,
            distance_metric=DistanceMetric.COSINE,
            normalization_version="v1",
        ),
        max_context_chunks=4,
        max_context_bytes=4_096,
        max_context_tokens=512,
    )


__all__ = ["KnowledgeConfig", "default_knowledge_config"]
