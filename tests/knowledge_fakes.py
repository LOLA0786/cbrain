"""Test doubles for knowledge embeddings and extraction.

These providers must never be imported from production CBrain modules.
A deployment that needs retrieval must inject real implementations.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence

from cbrain.knowledge.contracts import (
    KNOWLEDGE_SCHEMA_VERSION,
    EmbeddingProfile,
    GraphEdge,
    GraphNode,
    KnowledgeChunk,
    sha256_hex,
)


class DeterministicEmbeddingProvider:
    """Hash-derived embeddings. Never calls a live provider."""

    def embed(
        self, texts: Sequence[str], profile: EmbeddingProfile
    ) -> tuple[tuple[float, ...], ...]:
        vectors: list[tuple[float, ...]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            values: list[float] = []
            for index in range(profile.dimension):
                byte = digest[index % len(digest)]
                values.append(((byte / 255.0) * 2.0) - 1.0)
            vectors.append(tuple(values))
        return tuple(vectors)


class RuleBasedExtractor:
    """Deterministic extractor for tests: 'Name works at Org' / 'Name is a Role'."""

    _WORKS_AT = re.compile(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+works at\s+([A-Z][A-Za-z0-9& ]+)\b"
    )
    _IS_A = re.compile(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is an?\s+([a-z][a-z\s]+)\b"
    )

    def extract(
        self, chunk: KnowledgeChunk
    ) -> tuple[tuple[GraphNode, ...], tuple[GraphEdge, ...]]:
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        seen: set[str] = set()
        for person, org in self._WORKS_AT.findall(chunk.text):
            person_node = self._node(chunk, "person", person, {"role": "unknown"})
            org_node = self._node(chunk, "organization", org.strip(), {})
            if person_node.node_id not in seen:
                nodes.append(person_node)
                seen.add(person_node.node_id)
            if org_node.node_id not in seen:
                nodes.append(org_node)
                seen.add(org_node.node_id)
            edges.append(self._edge(chunk, person_node, "works_at", org_node))
        for person, role in self._IS_A.findall(chunk.text):
            person_node = self._node(
                chunk, "person", person, {"role": " ".join(role.split())}
            )
            if person_node.node_id not in seen:
                nodes.append(person_node)
                seen.add(person_node.node_id)
        return tuple(nodes), tuple(edges)

    def _node(
        self,
        chunk: KnowledgeChunk,
        canonical_type: str,
        name: str,
        attributes: Mapping[str, str],
    ) -> GraphNode:
        normalized = " ".join(name.split()).casefold()
        node_id = sha256_hex(
            {
                "tenant_id": chunk.tenant_id,
                "type": canonical_type,
                "name": normalized,
                "source_id": chunk.source_id,
                "source_revision": chunk.source_revision,
                "attributes": dict(attributes),
            }
        )
        return GraphNode(
            node_id=node_id,
            tenant_id=chunk.tenant_id,
            canonical_type=canonical_type,
            canonical_name=normalized,
            attributes=attributes,
            provenance_chunk_ids=(chunk.chunk_id,),
            confidence=0.6,
            valid_from=chunk.created_at,
            valid_to=None,
            source_revision=chunk.source_revision,
            source_id=chunk.source_id,
            schema_version=KNOWLEDGE_SCHEMA_VERSION,
            merge_score=None,
            merge_reason="exact-source-identity",
        )

    def _edge(
        self,
        chunk: KnowledgeChunk,
        source: GraphNode,
        relation: str,
        target: GraphNode,
    ) -> GraphEdge:
        edge_id = sha256_hex(
            {
                "tenant_id": chunk.tenant_id,
                "source": source.node_id,
                "relation": relation,
                "target": target.node_id,
                "source_id": chunk.source_id,
                "source_revision": chunk.source_revision,
            }
        )
        return GraphEdge(
            edge_id=edge_id,
            tenant_id=chunk.tenant_id,
            source_node_id=source.node_id,
            relation_type=relation,
            target_node_id=target.node_id,
            provenance_chunk_ids=(chunk.chunk_id,),
            confidence=0.6,
            valid_from=chunk.created_at,
            valid_to=None,
            source_revision=chunk.source_revision,
            source_id=chunk.source_id,
            schema_version=KNOWLEDGE_SCHEMA_VERSION,
        )
