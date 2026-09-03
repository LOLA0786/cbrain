"""CLI for knowledge doctor, ingest, query, graph-path, and collection status."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from .knowledge import (
    KnowledgeConfigError,
    KnowledgeError,
    KnowledgeRuntime,
    KnowledgeUnavailable,
    RetrievalQuery,
    SourceDocument,
)
from .knowledge.configuration import (
    knowledge_config_from_env,
    production_mode_requested,
)
from .knowledge.stores.postgres import PostgresKnowledgeStore
from .knowledge.stores.redis_cache import RedisKVCache


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cbrain-knowledge",
        description=(
            "Operate the governed knowledge runtime. Retrieved knowledge is "
            "untrusted context and never execution authority."
        ),
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser(
        "doctor", help="Validate configuration without printing secrets"
    )
    ingest = subcommands.add_parser("ingest", help="Ingest a local text document")
    ingest.add_argument("--tenant-id", required=True)
    ingest.add_argument("--collection-id", required=True)
    ingest.add_argument("--source-id", required=True)
    ingest.add_argument("--principal-id", required=True)
    ingest.add_argument("--path", required=True)
    ingest.add_argument("--provenance", required=True)
    query = subcommands.add_parser("query", help="Retrieve untrusted quoted evidence")
    query.add_argument("--tenant-id", required=True)
    query.add_argument("--collection-id", required=True)
    query.add_argument("--principal-id", required=True)
    query.add_argument("--text", required=True)
    graph = subcommands.add_parser("graph-path", help="Bounded graph traversal")
    graph.add_argument("--tenant-id", required=True)
    graph.add_argument("--principal-id", required=True)
    graph.add_argument("--seed", required=True)
    status = subcommands.add_parser(
        "collection-status", help="Show collection revision"
    )
    status.add_argument("--tenant-id", required=True)
    status.add_argument("--collection-id", required=True)
    parsed = parser.parse_args(arguments)
    try:
        config = knowledge_config_from_env()
        production = production_mode_requested() or bool(config.postgres_dsn)
        if parsed.command == "doctor":
            return _doctor(production=production)
        raise KnowledgeConfigError(
            "embeddings and extractor must be injected; "
            "cbrain-knowledge does not default to test fakes"
        )
    except (KnowledgeError, KnowledgeConfigError, KnowledgeUnavailable) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _doctor(*, production: bool) -> int:
    config = knowledge_config_from_env()
    print("knowledge configuration is valid")
    print(f"embedding_profile={config.embedding_profile.profile_id}")
    print(f"cache_ttl_seconds={config.cache_ttl_seconds}")
    if production and not config.postgres_dsn:
        print("postgres=not_configured")
        return 2
    if config.postgres_dsn:
        try:
            store = PostgresKnowledgeStore(
                config.postgres_dsn,
                timeout_seconds=config.retrieval_timeout_seconds,
            )
            store.ping()
            print("postgres=connected")
        except KnowledgeUnavailable:
            print("postgres=unavailable")
            return 2
        try:
            store.require_pgvector()
            print("pgvector=available")
        except KnowledgeUnavailable:
            print("pgvector=unavailable")
            return 2
        try:
            store.require_schema()
            print("schema=compatible")
        except KnowledgeUnavailable:
            print("schema=incompatible")
            return 2
    else:
        print("postgres=not_configured")
    if config.redis_dsn:
        try:
            RedisKVCache(config.redis_dsn).ping()
            print("redis=connected")
        except KnowledgeError:
            print("redis=unavailable")
            return 2
    else:
        print("redis=not_configured")
    return 0


def _ingest(runtime: KnowledgeRuntime, parsed: argparse.Namespace) -> int:
    text = Path(parsed.path).read_text(encoding="utf-8")
    result = runtime.ingest(
        SourceDocument(
            tenant_id=parsed.tenant_id,
            collection_id=parsed.collection_id,
            source_id=parsed.source_id,
            media_type="text/plain",
            text=text,
            acl_principals=frozenset({parsed.principal_id}),
            created_at=time.time(),
            provenance=parsed.provenance,
        )
    )
    print(
        json.dumps(
            {
                "source_id": result.source_id,
                "source_revision": result.source_revision,
                "collection_revision": result.collection_revision,
                "chunk_count": len(result.chunk_ids),
                "idempotent": result.idempotent,
            },
            sort_keys=True,
        )
    )
    return 0


def _query(runtime: KnowledgeRuntime, parsed: argparse.Namespace) -> int:
    context = runtime.retrieve(
        RetrievalQuery(
            tenant_id=parsed.tenant_id,
            collection_id=parsed.collection_id,
            principal_id=parsed.principal_id,
            text=parsed.text,
            top_k=runtime.config.top_k,
            created_at=time.time(),
        )
    )
    print(context.quoted_evidence)
    return 0


def _graph(runtime: KnowledgeRuntime, parsed: argparse.Namespace) -> int:
    paths = runtime.graph_path(
        tenant_id=parsed.tenant_id,
        principal_id=parsed.principal_id,
        seed_node_ids=(parsed.seed,),
    )
    print(json.dumps({"path_count": len(paths)}, sort_keys=True))
    return 0


def _status(runtime: KnowledgeRuntime, parsed: argparse.Namespace) -> int:
    print(
        json.dumps(
            runtime.collection_status(parsed.tenant_id, parsed.collection_id),
            sort_keys=True,
        )
    )
    return 0


__all__ = ["main"]
