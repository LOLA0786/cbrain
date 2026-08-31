"""PostgreSQL/pgvector durable adapter. Never the authorization authority."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..contracts import (
    INDEXED_EMBEDDING_DIMENSION,
    Citation,
    DistanceMetric,
    DocumentRevision,
    EmbeddingProfile,
    GraphEdge,
    GraphNode,
    GraphPath,
    KnowledgeChunk,
    SourceDocument,
)
from ..errors import KnowledgeError, KnowledgeUnavailable
from ..graph import bounded_walk

_LIVE_FROM = """
FROM knowledge_chunks c
JOIN knowledge_documents d
  ON d.tenant_id = c.tenant_id
 AND d.collection_id = c.collection_id
 AND d.source_id = c.source_id
 AND d.source_revision = c.source_revision
JOIN (
  SELECT tenant_id, collection_id, source_id, MAX(source_revision) AS source_revision
  FROM knowledge_documents
  WHERE tenant_id = %s AND collection_id = %s
  GROUP BY tenant_id, collection_id, source_id
) latest
  ON latest.tenant_id = d.tenant_id
 AND latest.collection_id = d.collection_id
 AND latest.source_id = d.source_id
 AND latest.source_revision = d.source_revision
"""

_LIVE_ACL = """
 AND d.tombstoned = FALSE
 AND (%s = ANY(c.acl_principals) OR '*' = ANY(c.acl_principals))
"""

_COUNTED_TABLES = (
    "knowledge_documents",
    "knowledge_chunks",
    "knowledge_embeddings",
    "knowledge_revisions",
    "ingestion_jobs",
)

_MIGRATION_FILES = (
    "002_knowledge_runtime.sql",
    "003_knowledge_runtime_vector_index.sql",
)


def _psycopg() -> Any:
    try:
        import psycopg
    except ImportError as exc:
        raise KnowledgeUnavailable("KNOWLEDGE_UNAVAILABLE") from exc
    return psycopg


def _unavailable(exc: BaseException) -> KnowledgeUnavailable:
    del exc
    return KnowledgeUnavailable("KNOWLEDGE_UNAVAILABLE")


def _connect_timeout(timeout_seconds: float) -> int:
    return max(1, int(math.ceil(timeout_seconds)))


def _vector_literal(values: Sequence[float]) -> str:
    if len(values) != INDEXED_EMBEDDING_DIMENSION:
        raise KnowledgeError("embedding dimension mismatch")
    return "[" + ",".join(repr(float(item)) for item in values) + "]"


def _timestamp(value: float) -> datetime:
    return datetime.fromtimestamp(value, tz=UTC)


def _as_float(value: Any) -> float:
    if isinstance(value, datetime):
        return value.timestamp()
    return float(value)


def _acl(values: Any) -> frozenset[str]:
    if values is None:
        return frozenset()
    return frozenset(str(item) for item in values)


def knowledge_migrations_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "migrations" / "postgres"


def statements_from_sql(script: str) -> tuple[str, ...]:
    statements: list[str] = []
    buffer: list[str] = []
    for line in script.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        buffer.append(line)
        if stripped.endswith(";"):
            statement = "\n".join(buffer).strip()
            if statement:
                statements.append(statement)
            buffer = []
    tail = "\n".join(buffer).strip()
    if tail:
        statements.append(tail)
    return tuple(statements)


def apply_knowledge_migrations(dsn: str, *, timeout_seconds: float) -> None:
    store = PostgresKnowledgeStore(dsn, timeout_seconds=timeout_seconds)
    connection = store.connect()
    try:
        directory = knowledge_migrations_dir()
        for name in _MIGRATION_FILES:
            script = (directory / name).read_text(encoding="utf-8")
            for statement in statements_from_sql(script):
                try:
                    connection.execute(statement)
                except Exception as exc:
                    code = getattr(exc, "sqlstate", None)
                    if name.startswith("003") and code in {"42710", "42P07", "42701"}:
                        continue
                    raise _unavailable(exc) from exc
        connection.commit()
    except KnowledgeUnavailable:
        raise
    except Exception as exc:
        raise _unavailable(exc) from exc
    finally:
        connection.close()


class PostgresKnowledgeStore:
    """Durable source of truth. Instantiation does not accept default passwords."""

    def __init__(self, dsn: str, *, timeout_seconds: float) -> None:
        if not isinstance(dsn, str) or not dsn.strip():
            raise KnowledgeUnavailable("KNOWLEDGE_UNAVAILABLE")
        if "postgres:postgres@" in dsn or dsn.endswith("@localhost/postgres"):
            raise KnowledgeUnavailable("KNOWLEDGE_UNAVAILABLE")
        self._dsn = dsn
        self._timeout_seconds = timeout_seconds
        self.fail_after: str | None = None

    def __repr__(self) -> str:
        return "PostgresKnowledgeStore(dsn=redacted)"

    def connect(self) -> Any:
        psycopg = _psycopg()
        try:
            milliseconds = max(1, int(self._timeout_seconds * 1000))
            return psycopg.connect(
                self._dsn,
                connect_timeout=_connect_timeout(self._timeout_seconds),
                autocommit=True,
                options=f"-c statement_timeout={milliseconds}",
            )
        except KnowledgeUnavailable:
            raise
        except Exception as exc:
            raise _unavailable(exc) from exc

    def ping(self) -> None:
        connection = self.connect()
        try:
            connection.execute("SELECT 1")
        except KnowledgeUnavailable:
            raise
        except Exception as exc:
            raise _unavailable(exc) from exc
        finally:
            connection.close()

    def require_pgvector(self) -> None:
        connection = self.connect()
        try:
            row = connection.execute(
                "SELECT extname FROM pg_extension WHERE extname = %s",
                ("vector",),
            ).fetchone()
            if row is None:
                raise KnowledgeUnavailable("KNOWLEDGE_UNAVAILABLE")
        except KnowledgeUnavailable:
            raise
        except Exception as exc:
            raise _unavailable(exc) from exc
        finally:
            connection.close()

    def require_schema(self) -> None:
        connection = self.connect()
        try:
            for table in (
                "knowledge_collections",
                "knowledge_documents",
                "knowledge_chunks",
                "knowledge_embeddings",
                "knowledge_graph_nodes",
                "knowledge_graph_edges",
                "knowledge_revisions",
                "ingestion_jobs",
            ):
                row = connection.execute("SELECT to_regclass(%s)", (table,)).fetchone()
                if row is None or row[0] is None:
                    raise KnowledgeUnavailable("KNOWLEDGE_UNAVAILABLE")
            embedding_type = connection.execute(
                """
                SELECT format_type(a.atttypid, a.atttypmod)
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = %s
                  AND a.attname = %s
                  AND n.nspname = current_schema()
                  AND NOT a.attisdropped
                """,
                ("knowledge_embeddings", "embedding"),
            ).fetchone()
            expected = f"vector({INDEXED_EMBEDDING_DIMENSION})"
            if embedding_type is None or embedding_type[0] != expected:
                raise KnowledgeUnavailable("KNOWLEDGE_UNAVAILABLE")
            index = connection.execute(
                """
                SELECT 1 FROM pg_indexes
                WHERE tablename = %s AND indexdef ILIKE %s
                """,
                ("knowledge_embeddings", "%hnsw%"),
            ).fetchone()
            if index is None:
                raise KnowledgeUnavailable("KNOWLEDGE_UNAVAILABLE")
        except KnowledgeUnavailable:
            raise
        except Exception as exc:
            raise _unavailable(exc) from exc
        finally:
            connection.close()

    def get_revision(
        self, tenant_id: str, collection_id: str, source_id: str
    ) -> DocumentRevision | None:
        connection = self.connect()
        try:
            row = connection.execute(
                """
                SELECT tenant_id, collection_id, source_id, source_revision,
                       content_digest, provenance, acl_principals, created_at,
                       schema_version, tombstoned
                FROM knowledge_documents
                WHERE tenant_id = %s AND collection_id = %s AND source_id = %s
                ORDER BY source_revision DESC
                LIMIT 1
                """,
                (tenant_id, collection_id, source_id),
            ).fetchone()
            if row is None:
                return None
            return self._document_revision(row)
        except KnowledgeError:
            raise
        except Exception as exc:
            raise _unavailable(exc) from exc
        finally:
            connection.close()

    def list_live_chunks(
        self,
        *,
        tenant_id: str,
        collection_id: str,
        principal_id: str,
    ) -> tuple[KnowledgeChunk, ...]:
        connection = self.connect()
        try:
            sql = (
                "SELECT c.chunk_id, c.tenant_id, c.collection_id, c.source_id, "
                "c.source_revision, c.chunk_index, c.text, c.content_digest, "
                "c.provenance, c.acl_principals, c.created_at, c.schema_version "
                + _LIVE_FROM
                + " WHERE c.tenant_id = %s AND c.collection_id = %s "
                + _LIVE_ACL
                + " ORDER BY c.chunk_id"
            )
            rows = connection.execute(
                sql,
                (tenant_id, collection_id, tenant_id, collection_id, principal_id),
            ).fetchall()
            return tuple(self._chunk(row) for row in rows)
        except KnowledgeError:
            raise
        except Exception as exc:
            raise _unavailable(exc) from exc
        finally:
            connection.close()

    def load_chunks(self, chunk_ids: Sequence[str]) -> tuple[KnowledgeChunk, ...]:
        if not chunk_ids:
            return ()
        connection = self.connect()
        try:
            rows = connection.execute(
                """
                SELECT c.chunk_id, c.tenant_id, c.collection_id, c.source_id,
                       c.source_revision, c.chunk_index, c.text, c.content_digest,
                       c.provenance, c.acl_principals, c.created_at, c.schema_version
                FROM knowledge_chunks c
                JOIN knowledge_documents d
                  ON d.tenant_id = c.tenant_id
                 AND d.collection_id = c.collection_id
                 AND d.source_id = c.source_id
                 AND d.source_revision = c.source_revision
                JOIN (
                  SELECT tenant_id, collection_id, source_id,
                         MAX(source_revision) AS source_revision
                  FROM knowledge_documents
                  GROUP BY tenant_id, collection_id, source_id
                ) latest
                  ON latest.tenant_id = d.tenant_id
                 AND latest.collection_id = d.collection_id
                 AND latest.source_id = d.source_id
                 AND latest.source_revision = d.source_revision
                WHERE c.chunk_id = ANY(%s) AND d.tombstoned = FALSE
                """,
                (list(chunk_ids),),
            ).fetchall()
            by_id = {row[0]: self._chunk(row) for row in rows}
            return tuple(by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id)
        except KnowledgeError:
            raise
        except Exception as exc:
            raise _unavailable(exc) from exc
        finally:
            connection.close()

    def collection_revision(self, tenant_id: str, collection_id: str) -> int:
        connection = self.connect()
        try:
            row = connection.execute(
                """
                SELECT collection_revision FROM knowledge_collections
                WHERE tenant_id = %s AND collection_id = %s
                """,
                (tenant_id, collection_id),
            ).fetchone()
            return 0 if row is None else int(row[0])
        except KnowledgeError:
            raise
        except Exception as exc:
            raise _unavailable(exc) from exc
        finally:
            connection.close()

    def embedding_profile(
        self, tenant_id: str, collection_id: str
    ) -> EmbeddingProfile | None:
        connection = self.connect()
        try:
            row = connection.execute(
                """
                SELECT embedding_profile_id, embedding_model, embedding_dimension,
                       distance_metric, normalization_version
                FROM knowledge_collections
                WHERE tenant_id = %s AND collection_id = %s
                """,
                (tenant_id, collection_id),
            ).fetchone()
            if row is None:
                return None
            return EmbeddingProfile(
                profile_id=str(row[0]),
                model_id=str(row[1]),
                dimension=int(row[2]),
                distance_metric=DistanceMetric(str(row[3])),
                normalization_version=str(row[4]),
            )
        except KnowledgeError:
            raise
        except Exception as exc:
            raise _unavailable(exc) from exc
        finally:
            connection.close()

    def search_vector(
        self,
        *,
        tenant_id: str,
        collection_id: str,
        principal_id: str,
        vector: tuple[float, ...],
        limit: int,
    ) -> tuple[tuple[str, float], ...]:
        literal = _vector_literal(vector)
        connection = self.connect()
        try:
            sql = (
                "SELECT e.chunk_id, 1 - (e.embedding <=> %s::vector) "
                "FROM knowledge_embeddings e "
                "JOIN knowledge_chunks c ON c.chunk_id = e.chunk_id "
                "JOIN knowledge_documents d "
                "  ON d.tenant_id = c.tenant_id "
                " AND d.collection_id = c.collection_id "
                " AND d.source_id = c.source_id "
                " AND d.source_revision = c.source_revision "
                "JOIN ( "
                "  SELECT tenant_id, collection_id, source_id, "
                "         MAX(source_revision) AS source_revision "
                "  FROM knowledge_documents "
                "  WHERE tenant_id = %s AND collection_id = %s "
                "  GROUP BY tenant_id, collection_id, source_id "
                ") latest "
                "  ON latest.tenant_id = d.tenant_id "
                " AND latest.collection_id = d.collection_id "
                " AND latest.source_id = d.source_id "
                " AND latest.source_revision = d.source_revision "
                "WHERE e.tenant_id = %s AND e.collection_id = %s "
                + _LIVE_ACL
                + " ORDER BY e.embedding <=> %s::vector, e.chunk_id LIMIT %s"
            )
            rows = connection.execute(
                sql,
                (
                    literal,
                    tenant_id,
                    collection_id,
                    tenant_id,
                    collection_id,
                    principal_id,
                    literal,
                    limit,
                ),
            ).fetchall()
            return tuple((str(row[0]), float(row[1])) for row in rows)
        except KnowledgeError:
            raise
        except Exception as exc:
            raise _unavailable(exc) from exc
        finally:
            connection.close()

    def search_keyword(
        self,
        *,
        tenant_id: str,
        collection_id: str,
        principal_id: str,
        query: str,
        limit: int,
    ) -> tuple[tuple[str, float], ...]:
        connection = self.connect()
        try:
            sql = (
                "SELECT c.chunk_id, ts_rank_cd(to_tsvector('simple', c.text), "
                "plainto_tsquery('simple', %s)) "
                + _LIVE_FROM
                + " WHERE c.tenant_id = %s AND c.collection_id = %s "
                + _LIVE_ACL
                + " AND to_tsvector('simple', c.text) @@ "
                "plainto_tsquery('simple', %s) "
                " ORDER BY ts_rank_cd(to_tsvector('simple', c.text), "
                "plainto_tsquery('simple', %s)) DESC, c.chunk_id LIMIT %s"
            )
            rows = connection.execute(
                sql,
                (
                    query,
                    tenant_id,
                    collection_id,
                    tenant_id,
                    collection_id,
                    principal_id,
                    query,
                    query,
                    limit,
                ),
            ).fetchall()
            return tuple((str(row[0]), float(row[1])) for row in rows)
        except KnowledgeError:
            raise
        except Exception as exc:
            raise _unavailable(exc) from exc
        finally:
            connection.close()

    def upsert_nodes(self, nodes: Sequence[GraphNode]) -> None:
        if not nodes:
            return
        connection = self.connect()
        try:
            with connection.transaction():
                for node in nodes:
                    self._insert_node(connection, node)
        except KnowledgeError:
            raise
        except Exception as exc:
            raise _unavailable(exc) from exc
        finally:
            connection.close()

    def upsert_edges(self, edges: Sequence[GraphEdge]) -> None:
        if not edges:
            return
        connection = self.connect()
        try:
            with connection.transaction():
                for edge in edges:
                    self._insert_edge(connection, edge)
        except KnowledgeError:
            raise
        except Exception as exc:
            raise _unavailable(exc) from exc
        finally:
            connection.close()

    def nodes_for_chunks(self, chunk_ids: Sequence[str]) -> tuple[GraphNode, ...]:
        if not chunk_ids:
            return ()
        connection = self.connect()
        try:
            rows = connection.execute(
                """
                SELECT node_id, tenant_id, canonical_type, canonical_name, attributes,
                       provenance_chunk_ids, confidence, valid_from, valid_to,
                       source_revision, source_id, schema_version
                FROM knowledge_graph_nodes
                WHERE provenance_chunk_ids && %s::text[]
                """,
                (list(chunk_ids),),
            ).fetchall()
            return tuple(self._node(row) for row in rows)
        except KnowledgeError:
            raise
        except Exception as exc:
            raise _unavailable(exc) from exc
        finally:
            connection.close()

    def traverse(
        self,
        *,
        tenant_id: str,
        principal_id: str,
        seed_node_ids: Sequence[str],
        max_depth: int,
        max_nodes: int,
    ) -> tuple[GraphPath, ...]:
        connection = self.connect()
        try:
            node_rows = connection.execute(
                """
                SELECT n.node_id, n.tenant_id, n.canonical_type, n.canonical_name,
                       n.attributes, n.provenance_chunk_ids, n.confidence,
                       n.valid_from, n.valid_to, n.source_revision, n.source_id,
                       n.schema_version
                FROM knowledge_graph_nodes n
                WHERE n.tenant_id = %s
                  AND EXISTS (
                    SELECT 1 FROM knowledge_chunks c
                    JOIN knowledge_documents d
                      ON d.tenant_id = c.tenant_id
                     AND d.collection_id = c.collection_id
                     AND d.source_id = c.source_id
                     AND d.source_revision = c.source_revision
                    JOIN (
                      SELECT tenant_id, collection_id, source_id,
                             MAX(source_revision) AS source_revision
                      FROM knowledge_documents
                      WHERE tenant_id = %s
                      GROUP BY tenant_id, collection_id, source_id
                    ) latest
                      ON latest.tenant_id = d.tenant_id
                     AND latest.collection_id = d.collection_id
                     AND latest.source_id = d.source_id
                     AND latest.source_revision = d.source_revision
                    WHERE c.chunk_id = ANY(n.provenance_chunk_ids)
                      AND c.tenant_id = %s
                      AND d.tombstoned = FALSE
                      AND (%s = ANY(c.acl_principals) OR '*' = ANY(c.acl_principals))
                  )
                """,
                (tenant_id, tenant_id, tenant_id, principal_id),
            ).fetchall()
            visible_nodes = {row[0]: self._node(row) for row in node_rows}
            edge_rows = connection.execute(
                """
                SELECT e.edge_id, e.tenant_id, e.source_node_id, e.relation_type,
                       e.target_node_id, e.provenance_chunk_ids, e.confidence,
                       e.valid_from, e.valid_to, e.source_revision, e.source_id,
                       e.schema_version
                FROM knowledge_graph_edges e
                WHERE e.tenant_id = %s
                  AND EXISTS (
                    SELECT 1 FROM knowledge_chunks c
                    JOIN knowledge_documents d
                      ON d.tenant_id = c.tenant_id
                     AND d.collection_id = c.collection_id
                     AND d.source_id = c.source_id
                     AND d.source_revision = c.source_revision
                    JOIN (
                      SELECT tenant_id, collection_id, source_id,
                             MAX(source_revision) AS source_revision
                      FROM knowledge_documents
                      WHERE tenant_id = %s
                      GROUP BY tenant_id, collection_id, source_id
                    ) latest
                      ON latest.tenant_id = d.tenant_id
                     AND latest.collection_id = d.collection_id
                     AND latest.source_id = d.source_id
                     AND latest.source_revision = d.source_revision
                    WHERE c.chunk_id = ANY(e.provenance_chunk_ids)
                      AND c.tenant_id = %s
                      AND d.tombstoned = FALSE
                      AND (%s = ANY(c.acl_principals) OR '*' = ANY(c.acl_principals))
                  )
                """,
                (tenant_id, tenant_id, tenant_id, principal_id),
            ).fetchall()
            adjacency: dict[str, list[GraphEdge]] = {}
            for row in edge_rows:
                edge = self._edge(row)
                if (
                    edge.source_node_id not in visible_nodes
                    or edge.target_node_id not in visible_nodes
                ):
                    continue
                adjacency.setdefault(edge.source_node_id, []).append(edge)
            chunk_ids = [
                chunk_id
                for node in visible_nodes.values()
                for chunk_id in node.provenance_chunk_ids
            ]
            chunks = {chunk.chunk_id: chunk for chunk in self.load_chunks(chunk_ids)}

            def _citations(nodes: Sequence[GraphNode]) -> tuple[Citation, ...]:
                selected: list[KnowledgeChunk] = []
                seen: set[str] = set()
                for node in nodes:
                    for chunk_id in node.provenance_chunk_ids:
                        chunk = chunks.get(chunk_id)
                        if chunk is None or chunk_id in seen:
                            continue
                        selected.append(chunk)
                        seen.add(chunk_id)
                return self.citations_for(selected)

            return bounded_walk(
                visible_nodes=visible_nodes,
                adjacency=adjacency,
                seed_node_ids=seed_node_ids,
                max_depth=max_depth,
                max_nodes=max_nodes,
                citations_for=_citations,
            )
        except KnowledgeError:
            raise
        except Exception as exc:
            raise _unavailable(exc) from exc
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
        if profile.dimension != INDEXED_EMBEDDING_DIMENSION:
            raise KnowledgeError("embedding dimension mismatch")
        for vector in embeddings.values():
            if len(vector) != INDEXED_EMBEDDING_DIMENSION:
                raise KnowledgeError("embedding dimension mismatch")
        connection = self.connect()
        try:
            with connection.transaction():
                return self._publish_in_transaction(
                    connection,
                    document=document,
                    revision=revision,
                    chunks=chunks,
                    embeddings=embeddings,
                    nodes=nodes,
                    edges=edges,
                    profile=profile,
                )
        except KnowledgeError:
            raise
        except Exception as exc:
            raise _unavailable(exc) from exc
        finally:
            connection.close()

    def tombstone(
        self, tenant_id: str, collection_id: str, source_id: str, created_at: float
    ) -> int:
        connection = self.connect()
        try:
            with connection.transaction():
                locked = connection.execute(
                    """
                    SELECT collection_revision FROM knowledge_collections
                    WHERE tenant_id = %s AND collection_id = %s
                    FOR UPDATE
                    """,
                    (tenant_id, collection_id),
                ).fetchone()
                if locked is None:
                    raise KnowledgeError("source does not exist")
                current = connection.execute(
                    """
                    SELECT content_digest, provenance, acl_principals, schema_version,
                           source_revision
                    FROM knowledge_documents
                    WHERE tenant_id = %s AND collection_id = %s AND source_id = %s
                    ORDER BY source_revision DESC
                    LIMIT 1
                    """,
                    (tenant_id, collection_id, source_id),
                ).fetchone()
                if current is None:
                    raise KnowledgeError("source does not exist")
                connection.execute(
                    """
                    INSERT INTO knowledge_documents (
                        tenant_id, collection_id, source_id, source_revision,
                        content_digest, provenance, acl_principals, created_at,
                        schema_version, tombstoned
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE)
                    """,
                    (
                        tenant_id,
                        collection_id,
                        source_id,
                        int(current[4]) + 1,
                        current[0],
                        current[1],
                        current[2],
                        _timestamp(created_at),
                        current[3],
                    ),
                )
                updated = connection.execute(
                    """
                    UPDATE knowledge_collections
                    SET collection_revision = collection_revision + 1
                    WHERE tenant_id = %s AND collection_id = %s
                    RETURNING collection_revision
                    """,
                    (tenant_id, collection_id),
                ).fetchone()
                assert updated is not None
                collection_revision = int(updated[0])
                connection.execute(
                    """
                    INSERT INTO knowledge_revisions (
                        tenant_id, collection_id, collection_revision
                    ) VALUES (%s, %s, %s)
                    """,
                    (tenant_id, collection_id, collection_revision),
                )
                return collection_revision
        except KnowledgeError:
            raise
        except Exception as exc:
            raise _unavailable(exc) from exc
        finally:
            connection.close()

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

    def row_counts(self, tenant_id: str, collection_id: str) -> dict[str, int]:
        connection = self.connect()
        try:
            counts: dict[str, int] = {}
            for table in _COUNTED_TABLES:
                row = connection.execute(
                    "SELECT COUNT(*) FROM "
                    + table
                    + " WHERE tenant_id = %s AND collection_id = %s",
                    (tenant_id, collection_id),
                ).fetchone()
                counts[table] = 0 if row is None else int(row[0])
            nodes = connection.execute(
                """
                SELECT COUNT(*) FROM knowledge_graph_nodes n
                WHERE n.tenant_id = %s
                  AND EXISTS (
                    SELECT 1 FROM knowledge_chunks c
                    WHERE c.chunk_id = ANY(n.provenance_chunk_ids)
                      AND c.tenant_id = %s AND c.collection_id = %s
                  )
                """,
                (tenant_id, tenant_id, collection_id),
            ).fetchone()
            edges = connection.execute(
                """
                SELECT COUNT(*) FROM knowledge_graph_edges e
                WHERE e.tenant_id = %s
                  AND EXISTS (
                    SELECT 1 FROM knowledge_chunks c
                    WHERE c.chunk_id = ANY(e.provenance_chunk_ids)
                      AND c.tenant_id = %s AND c.collection_id = %s
                  )
                """,
                (tenant_id, tenant_id, collection_id),
            ).fetchone()
            counts["knowledge_graph_nodes"] = 0 if nodes is None else int(nodes[0])
            counts["knowledge_graph_edges"] = 0 if edges is None else int(edges[0])
            return counts
        except KnowledgeError:
            raise
        except Exception as exc:
            raise _unavailable(exc) from exc
        finally:
            connection.close()

    def list_nodes(self, tenant_id: str) -> tuple[GraphNode, ...]:
        connection = self.connect()
        try:
            rows = connection.execute(
                """
                SELECT node_id, tenant_id, canonical_type, canonical_name, attributes,
                       provenance_chunk_ids, confidence, valid_from, valid_to,
                       source_revision, source_id, schema_version
                FROM knowledge_graph_nodes
                WHERE tenant_id = %s
                ORDER BY node_id
                """,
                (tenant_id,),
            ).fetchall()
            return tuple(self._node(row) for row in rows)
        except KnowledgeError:
            raise
        except Exception as exc:
            raise _unavailable(exc) from exc
        finally:
            connection.close()

    def _publish_in_transaction(
        self,
        connection: Any,
        *,
        document: SourceDocument,
        revision: DocumentRevision,
        chunks: Sequence[KnowledgeChunk],
        embeddings: Mapping[str, tuple[float, ...]],
        nodes: Sequence[GraphNode],
        edges: Sequence[GraphEdge],
        profile: EmbeddingProfile,
    ) -> int:
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
            SELECT embedding_profile_id, embedding_model, embedding_dimension,
                   distance_metric, normalization_version, collection_revision
            FROM knowledge_collections
            WHERE tenant_id = %s AND collection_id = %s
            FOR UPDATE
            """,
            (document.tenant_id, document.collection_id),
        ).fetchone()
        if row is None:
            raise KnowledgeUnavailable("KNOWLEDGE_UNAVAILABLE")
        published = EmbeddingProfile(
            profile_id=str(row[0]),
            model_id=str(row[1]),
            dimension=int(row[2]),
            distance_metric=DistanceMetric(str(row[3])),
            normalization_version=str(row[4]),
        )
        if published != profile:
            raise KnowledgeError("embedding profile is immutable after first revision")
        existing = connection.execute(
            """
            SELECT content_digest, tombstoned, source_revision
            FROM knowledge_documents
            WHERE tenant_id = %s AND collection_id = %s AND source_id = %s
            ORDER BY source_revision DESC
            LIMIT 1
            """,
            (document.tenant_id, document.collection_id, document.source_id),
        ).fetchone()
        if (
            existing is not None
            and not bool(existing[1])
            and str(existing[0]) == revision.content_digest
        ):
            return int(row[5])
        if self.fail_after == "collection":
            raise RuntimeError("injected failure")
        connection.execute(
            """
            INSERT INTO knowledge_documents (
                tenant_id, collection_id, source_id, source_revision,
                content_digest, provenance, acl_principals, created_at,
                schema_version, tombstoned
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                revision.tenant_id,
                revision.collection_id,
                revision.source_id,
                revision.source_revision,
                revision.content_digest,
                revision.provenance,
                list(revision.acl_principals),
                _timestamp(revision.created_at),
                revision.schema_version,
                revision.tombstoned,
            ),
        )
        if self.fail_after == "document":
            raise RuntimeError("injected failure")
        for chunk in chunks:
            connection.execute(
                """
                INSERT INTO knowledge_chunks (
                    chunk_id, tenant_id, collection_id, source_id,
                    source_revision, chunk_index, content_digest, provenance,
                    acl_principals, created_at, schema_version, text
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    chunk.chunk_id,
                    chunk.tenant_id,
                    chunk.collection_id,
                    chunk.source_id,
                    chunk.source_revision,
                    chunk.chunk_index,
                    chunk.content_digest,
                    chunk.provenance,
                    list(chunk.acl_principals),
                    _timestamp(chunk.created_at),
                    chunk.schema_version,
                    chunk.text,
                ),
            )
        if self.fail_after == "chunks":
            raise RuntimeError("injected failure")
        for chunk_id, vector in embeddings.items():
            connection.execute(
                """
                INSERT INTO knowledge_embeddings (
                    chunk_id, tenant_id, collection_id, embedding, profile_id
                ) VALUES (%s, %s, %s, %s::vector, %s)
                """,
                (
                    chunk_id,
                    document.tenant_id,
                    document.collection_id,
                    _vector_literal(vector),
                    profile.profile_id,
                ),
            )
        if self.fail_after == "embeddings":
            raise RuntimeError("injected failure")
        for node in nodes:
            self._insert_node(connection, node)
        for edge in edges:
            self._insert_edge(connection, edge)
        if self.fail_after == "graph":
            raise RuntimeError("injected failure")
        updated = connection.execute(
            """
            UPDATE knowledge_collections
            SET collection_revision = collection_revision + 1
            WHERE tenant_id = %s AND collection_id = %s
            RETURNING collection_revision
            """,
            (document.tenant_id, document.collection_id),
        ).fetchone()
        assert updated is not None
        collection_revision = int(updated[0])
        connection.execute(
            """
            INSERT INTO knowledge_revisions (
                tenant_id, collection_id, collection_revision
            ) VALUES (%s, %s, %s)
            """,
            (document.tenant_id, document.collection_id, collection_revision),
        )
        if self.fail_after == "revision":
            raise RuntimeError("injected failure")
        connection.execute(
            """
            INSERT INTO ingestion_jobs (
                job_id, tenant_id, collection_id, source_id, status
            ) VALUES (%s, %s, %s, %s, %s)
            """,
            (
                f"{document.tenant_id}:{document.collection_id}:"
                f"{document.source_id}:{revision.source_revision}",
                document.tenant_id,
                document.collection_id,
                document.source_id,
                "completed",
            ),
        )
        if self.fail_after == "job":
            raise RuntimeError("injected failure")
        return collection_revision

    def _insert_node(self, connection: Any, node: GraphNode) -> None:
        connection.execute(
            """
            INSERT INTO knowledge_graph_nodes (
                node_id, tenant_id, canonical_type, canonical_name, attributes,
                provenance_chunk_ids, confidence, valid_from, valid_to,
                source_revision, source_id, schema_version
            ) VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (node_id) DO UPDATE SET
                attributes = EXCLUDED.attributes,
                provenance_chunk_ids = EXCLUDED.provenance_chunk_ids,
                confidence = EXCLUDED.confidence
            """,
            (
                node.node_id,
                node.tenant_id,
                node.canonical_type,
                node.canonical_name,
                json.dumps(dict(node.attributes), sort_keys=True),
                list(node.provenance_chunk_ids),
                node.confidence,
                _timestamp(node.valid_from),
                None if node.valid_to is None else _timestamp(node.valid_to),
                node.source_revision,
                node.source_id,
                node.schema_version,
            ),
        )

    def _insert_edge(self, connection: Any, edge: GraphEdge) -> None:
        connection.execute(
            """
            INSERT INTO knowledge_graph_edges (
                edge_id, tenant_id, source_node_id, relation_type, target_node_id,
                provenance_chunk_ids, confidence, valid_from, valid_to,
                source_revision, source_id, schema_version
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (edge_id) DO UPDATE SET
                provenance_chunk_ids = EXCLUDED.provenance_chunk_ids,
                confidence = EXCLUDED.confidence
            """,
            (
                edge.edge_id,
                edge.tenant_id,
                edge.source_node_id,
                edge.relation_type,
                edge.target_node_id,
                list(edge.provenance_chunk_ids),
                edge.confidence,
                _timestamp(edge.valid_from),
                None if edge.valid_to is None else _timestamp(edge.valid_to),
                edge.source_revision,
                edge.source_id,
                edge.schema_version,
            ),
        )

    def _document_revision(self, row: Any) -> DocumentRevision:
        return DocumentRevision(
            tenant_id=str(row[0]),
            collection_id=str(row[1]),
            source_id=str(row[2]),
            source_revision=int(row[3]),
            content_digest=str(row[4]),
            provenance=str(row[5]),
            acl_principals=_acl(row[6]),
            created_at=_as_float(row[7]),
            schema_version=int(row[8]),
            tombstoned=bool(row[9]),
        )

    def _chunk(self, row: Any) -> KnowledgeChunk:
        return KnowledgeChunk(
            chunk_id=str(row[0]),
            tenant_id=str(row[1]),
            collection_id=str(row[2]),
            source_id=str(row[3]),
            source_revision=int(row[4]),
            chunk_index=int(row[5]),
            text=str(row[6]),
            content_digest=str(row[7]),
            provenance=str(row[8]),
            acl_principals=_acl(row[9]),
            created_at=_as_float(row[10]),
            schema_version=int(row[11]),
        )

    def _node(self, row: Any) -> GraphNode:
        attributes = row[4]
        if isinstance(attributes, str):
            attributes = json.loads(attributes)
        return GraphNode(
            node_id=str(row[0]),
            tenant_id=str(row[1]),
            canonical_type=str(row[2]),
            canonical_name=str(row[3]),
            attributes=dict(attributes or {}),
            provenance_chunk_ids=tuple(str(item) for item in row[5]),
            confidence=float(row[6]),
            valid_from=_as_float(row[7]),
            valid_to=None if row[8] is None else _as_float(row[8]),
            source_revision=int(row[9]),
            source_id=str(row[10]),
            schema_version=int(row[11]),
        )

    def _edge(self, row: Any) -> GraphEdge:
        return GraphEdge(
            edge_id=str(row[0]),
            tenant_id=str(row[1]),
            source_node_id=str(row[2]),
            relation_type=str(row[3]),
            target_node_id=str(row[4]),
            provenance_chunk_ids=tuple(str(item) for item in row[5]),
            confidence=float(row[6]),
            valid_from=_as_float(row[7]),
            valid_to=None if row[8] is None else _as_float(row[8]),
            source_revision=int(row[9]),
            source_id=str(row[10]),
            schema_version=int(row[11]),
        )


__all__ = [
    "PostgresKnowledgeStore",
    "apply_knowledge_migrations",
    "knowledge_migrations_dir",
]
