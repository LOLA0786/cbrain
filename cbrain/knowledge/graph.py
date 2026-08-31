"""Provenance-backed graph helpers. Extracted relations are not facts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from .contracts import Citation, GraphEdge, GraphNode, GraphPath
from .errors import KnowledgeError
from .ports import KnowledgeStore


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


def bounded_walk(
    *,
    visible_nodes: Mapping[str, GraphNode],
    adjacency: Mapping[str, Sequence[GraphEdge]],
    seed_node_ids: Sequence[str],
    max_depth: int,
    max_nodes: int,
    citations_for: Callable[[Sequence[GraphNode]], tuple[Citation, ...]],
) -> tuple[GraphPath, ...]:
    paths: list[GraphPath] = []
    for seed in seed_node_ids:
        if seed not in visible_nodes:
            continue
        visited: set[str] = set()
        stack: list[tuple[str, int, tuple[GraphNode, ...], tuple[GraphEdge, ...]]] = [
            (seed, 0, (visible_nodes[seed],), ())
        ]
        while stack and len(visited) < max_nodes:
            node_id, depth, nodes, edges = stack.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)
            citations = citations_for(nodes)
            if citations:
                paths.append(GraphPath(nodes=nodes, edges=edges, citations=citations))
            if depth >= max_depth:
                continue
            for edge in sorted(
                adjacency.get(node_id, ()), key=lambda item: item.edge_id
            ):
                if edge.target_node_id in visited:
                    continue
                if len(visited) >= max_nodes:
                    break
                target = visible_nodes.get(edge.target_node_id)
                if target is None:
                    continue
                stack.append(
                    (
                        edge.target_node_id,
                        depth + 1,
                        (*nodes, target),
                        (*edges, edge),
                    )
                )
    return tuple(paths)


def graph_paths(
    store: KnowledgeStore,
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


__all__ = [
    "bounded_walk",
    "graph_paths",
    "require_provenance",
    "resolve_entities",
]
