"""Deployment-owned knowledge configuration. Not model-tool arguments."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

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
    return require_text(value, field_name)


def _redact_dsn(value: str | None) -> str | None:
    return "set" if value else None


@dataclass(frozen=True, slots=True)
class KnowledgeConfig:
    postgres_dsn: str | None = field(repr=False)
    redis_dsn: str | None = field(repr=False)
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

    def __repr__(self) -> str:
        return (
            "KnowledgeConfig("
            f"postgres_dsn={_redact_dsn(self.postgres_dsn)!r}, "
            f"redis_dsn={_redact_dsn(self.redis_dsn)!r}, "
            f"cache_ttl_seconds={self.cache_ttl_seconds!r}, "
            f"max_document_bytes={self.max_document_bytes!r}, "
            f"chunk_size={self.chunk_size!r}, "
            f"chunk_overlap={self.chunk_overlap!r}, "
            f"top_k={self.top_k!r}, "
            f"vector_candidate_count={self.vector_candidate_count!r}, "
            f"keyword_candidate_count={self.keyword_candidate_count!r}, "
            f"graph_max_depth={self.graph_max_depth!r}, "
            f"graph_max_nodes={self.graph_max_nodes!r}, "
            f"retrieval_timeout_seconds={self.retrieval_timeout_seconds!r}, "
            f"embedding_profile={self.embedding_profile!r}, "
            f"max_context_chunks={self.max_context_chunks!r}, "
            f"max_context_bytes={self.max_context_bytes!r}, "
            f"max_context_tokens={self.max_context_tokens!r}, "
            f"policy_version={self.policy_version!r})"
        )

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


def knowledge_config_from_env() -> KnowledgeConfig:
    return replace(
        default_knowledge_config(),
        postgres_dsn=os.environ.get("CBRAIN_KNOWLEDGE_PG_DSN") or None,
        redis_dsn=os.environ.get("CBRAIN_KNOWLEDGE_REDIS_DSN") or None,
    )


def production_mode_requested() -> bool:
    return (
        os.environ.get("CBRAIN_KNOWLEDGE_MODE", "").strip().casefold() == "production"
    )


__all__ = [
    "KnowledgeConfig",
    "default_knowledge_config",
    "knowledge_config_from_env",
    "production_mode_requested",
]
