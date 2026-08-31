"""Provenance-backed graph helpers. Extracted relations are not facts."""

from __future__ import annotations

from collections.abc import Sequence

from .contracts import GraphEdge, GraphNode, GraphPath
from .errors import KnowledgeError
from .stores.memory import InMemoryKnowledgeStore


def require_provenance(node: GraphNode | GraphEdge) -> None:
    if not node.provenance_chunk_ids:
        raise KnowledgeError("graph record requires source provenance")


def resolve_entities(
    existing: Sequence[GraphNode], candidate: GraphNode
) -> tuple[GraphNode, float, str]:
    """Expose merge score and reason. Incompatible entities stay distinct."""

    for node in existing:
        if node.tenant_id != candidate.tenant_id:
            continue
        if node.canonical_type != candidate.canonical_type:
            continue
        if node.canonical_name != candidate.canonical_name:
            continue
        if dict(node.attributes) != dict(candidate.attributes):
            return (
                candidate,
                0.4,
                "same-name-conflicting-attributes",
            )
        if node.source_id != candidate.source_id:
            return (
                candidate,
                0.7,
                "compatible-name-different-source",
            )
        return node, 1.0, "exact-source-identity"
    return candidate, 0.0, "no-match"


def graph_paths(
    store: InMemoryKnowledgeStore,
    *,
    tenant_id: str,
    principal_id: str,
    seed_node_ids: Sequence[str],
    max_depth: int,
    max_nodes: int,
) -> tuple[GraphPath, ...]:
    return store.traverse(
        tenant_id=tenant_id,
        principal_id=principal_id,
        seed_node_ids=seed_node_ids,
        max_depth=max_depth,
        max_nodes=max_nodes,
    )


__all__ = ["graph_paths", "require_provenance", "resolve_entities"]
