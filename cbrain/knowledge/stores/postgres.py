"""PostgreSQL/pgvector durable adapter. Never the authorization authority."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..contracts import (
    DocumentRevision,
    EmbeddingProfile,
    GraphEdge,
    GraphNode,
    KnowledgeChunk,
    SourceDocument,
)
from ..errors import KnowledgeUnavailable


def _psycopg() -> Any:
    try:
        import psycopg  # type: ignore[import-not-found]
    except ImportError as exc:
        raise KnowledgeUnavailable("KNOWLEDGE_UNAVAILABLE") from exc
    return psycopg


class PostgresKnowledgeStore:
    """Durable source of truth. Instantiation does not accept default passwords."""

    def __init__(self, dsn: str, *, timeout_seconds: float) -> None:
        if not isinstance(dsn, str) or not dsn.strip():
            raise KnowledgeUnavailable("KNOWLEDGE_UNAVAILABLE")
        if "postgres:postgres@" in dsn or dsn.endswith("@localhost/postgres"):
            raise KnowledgeUnavailable("KNOWLEDGE_UNAVAILABLE")
        self._dsn = dsn
        self._timeout_seconds = timeout_seconds

    def connect(self) -> Any:
        psycopg = _psycopg()
        try:
            return psycopg.connect(
                self._dsn, connect_timeout=int(self._timeout_seconds)
            )
        except Exception as exc:
            raise KnowledgeUnavailable("KNOWLEDGE_UNAVAILABLE") from exc

    def ping(self) -> None:
        connection = self.connect()
        try:
            connection.execute("SELECT 1")
        finally:
            connection.close()

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
        connection = self.connect()
        try:
            with connection.transaction():
                connection.execute(
                    """
                    INSERT INTO knowledge_collections (
                        tenant_id, collection_id, embedding_profile_id,
                        embedding_model, embedding_dimension, distance_metric,
                        normalization_version, collection_revision
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, 0)
                    ON CONFLICT (tenant_id, collection_id) DO NOTHING
                    """,
                    (
                        document.tenant_id,
                        document.collection_id,
                        profile.profile_id,
                        profile.model_id,
                        profile.dimension,
                        profile.distance_metric.value,
                        profile.normalization_version,
                    ),
                )
                row = connection.execute(
                    """
                    SELECT embedding_model, embedding_dimension, distance_metric,
                           normalization_version
                    FROM knowledge_collections
                    WHERE tenant_id = %s AND collection_id = %s
                    FOR UPDATE
                    """,
                    (document.tenant_id, document.collection_id),
                ).fetchone()
                if row is None:
                    raise KnowledgeUnavailable("KNOWLEDGE_UNAVAILABLE")
                if (
                    row[0] != profile.model_id
                    or int(row[1]) != profile.dimension
                    or row[2] != profile.distance_metric.value
                    or row[3] != profile.normalization_version
                ):
                    raise KnowledgeUnavailable("KNOWLEDGE_UNAVAILABLE")
                for vector in embeddings.values():
                    if len(vector) != profile.dimension:
                        raise KnowledgeUnavailable("KNOWLEDGE_UNAVAILABLE")
                connection.execute(
                    """
                    UPDATE knowledge_collections
                    SET collection_revision = collection_revision + 1
                    WHERE tenant_id = %s AND collection_id = %s
                    """,
                    (document.tenant_id, document.collection_id),
                )
                updated = connection.execute(
                    """
                    SELECT collection_revision FROM knowledge_collections
                    WHERE tenant_id = %s AND collection_id = %s
                    """,
                    (document.tenant_id, document.collection_id),
                ).fetchone()
                assert updated is not None
                return int(updated[0])
        except KnowledgeUnavailable:
            raise
        except Exception as exc:
            raise KnowledgeUnavailable("KNOWLEDGE_UNAVAILABLE") from exc
        finally:
            connection.close()


__all__ = ["PostgresKnowledgeStore"]
