"""Live PostgreSQL/pgvector and Redis knowledge integration tests.

When CBRAIN_KNOWLEDGE_LIVE is unset, these tests skip with that exact reason.
When CBRAIN_KNOWLEDGE_LIVE=1, missing DSNs or skipped round-trips fail the job.
"""

from __future__ import annotations

import inspect
import os
import threading
import uuid
from dataclasses import replace
from typing import Any

import pytest

from cbrain import GovernedRuntime
from cbrain.agent import (
    AgentProfile,
    FoundationAgent,
    GovernedTool,
    RunInput,
    RunStatus,
    ToolRegistry,
)
from cbrain.knowledge import (
    EmbeddingProfile,
    GraphEdge,
    InMemoryKnowledgeStore,
    KnowledgeConfig,
    KnowledgeError,
    KnowledgeRuntime,
    KnowledgeUnavailable,
    RetrievalQuery,
    SourceDocument,
)
from cbrain.knowledge.configuration import default_knowledge_config
from cbrain.knowledge.contracts import INDEXED_EMBEDDING_DIMENSION, DistanceMetric
from cbrain.knowledge.errors import KnowledgeConfigError
from cbrain.knowledge.graph import resolve_entities
from cbrain.knowledge.stores.postgres import (
    PostgresKnowledgeStore,
    apply_knowledge_migrations,
)
from cbrain.knowledge.stores.redis_cache import RedisKVCache
from cbrain.knowledge_cli import main as knowledge_main
from cbrain.models import ModelRouter, ToolCall

LIVE_REASON = "CBRAIN_KNOWLEDGE_LIVE is not set"
NOW = 1_700_000_000.0
COLLECTION = "ops-docs"
ALICE = "alice"
BOB = "bob"


def _live_requested() -> bool:
    return os.environ.get("CBRAIN_KNOWLEDGE_LIVE") == "1"


def _pg_dsn() -> str | None:
    value = os.environ.get("CBRAIN_KNOWLEDGE_PG_DSN")
    return value if value and value.strip() else None


def _redis_dsn() -> str | None:
    value = os.environ.get("CBRAIN_KNOWLEDGE_REDIS_DSN")
    return value if value and value.strip() else None


@pytest.fixture(scope="session")
def live_env() -> tuple[str, str]:
    if not _live_requested():
        pytest.skip(LIVE_REASON)
    postgres = _pg_dsn()
    redis_dsn = _redis_dsn()
    if postgres is None:
        pytest.fail("CBRAIN_KNOWLEDGE_LIVE=1 but CBRAIN_KNOWLEDGE_PG_DSN is missing")
    if redis_dsn is None:
        pytest.fail("CBRAIN_KNOWLEDGE_LIVE=1 but CBRAIN_KNOWLEDGE_REDIS_DSN is missing")
    apply_knowledge_migrations(postgres, timeout_seconds=5.0)
    store = PostgresKnowledgeStore(postgres, timeout_seconds=5.0)
    store.ping()
    store.require_pgvector()
    store.require_schema()
    RedisKVCache(redis_dsn).ping()
    return postgres, redis_dsn


def _tenant() -> str:
    return f"tenant-{uuid.uuid4().hex[:16]}"


def _config(postgres: str, redis_dsn: str) -> KnowledgeConfig:
    return replace(
        default_knowledge_config(),
        postgres_dsn=postgres,
        redis_dsn=redis_dsn,
    )


def _runtime(postgres: str, redis_dsn: str, **overrides: Any) -> KnowledgeRuntime:
    payload = {
        "config": _config(postgres, redis_dsn),
        "clock": lambda: NOW,
    }
    payload.update(overrides)
    return KnowledgeRuntime.from_config(**payload)


def _document(
    tenant_id: str,
    *,
    text: str,
    source_id: str = "src-1",
    principals: frozenset[str] = frozenset({ALICE}),
) -> SourceDocument:
    return SourceDocument(
        tenant_id=tenant_id,
        collection_id=COLLECTION,
        source_id=source_id,
        media_type="text/plain",
        text=text,
        acl_principals=principals,
        created_at=NOW,
        provenance="operator-upload",
    )


def _query(
    tenant_id: str, text: str = "Alice works", *, principal: str = ALICE
) -> RetrievalQuery:
    return RetrievalQuery(
        tenant_id=tenant_id,
        collection_id=COLLECTION,
        principal_id=principal,
        text=text,
        top_k=4,
        created_at=NOW,
    )


def _pg_store(runtime: KnowledgeRuntime) -> PostgresKnowledgeStore:
    store = runtime.store
    assert isinstance(store, PostgresKnowledgeStore)
    return store


def test_postgres_blank_dsn_is_unavailable() -> None:
    with pytest.raises(KnowledgeUnavailable):
        PostgresKnowledgeStore("   ", timeout_seconds=1.0).ping()


def test_live_round_trip_persists_and_retrieves(live_env: tuple[str, str]) -> None:
    postgres, redis_dsn = live_env
    tenant = _tenant()
    runtime = _runtime(postgres, redis_dsn)
    result = runtime.ingest(_document(tenant, text="Alice works at Acme."))
    assert result.published is True
    store = runtime.store
    assert isinstance(store, PostgresKnowledgeStore)
    counts = store.row_counts(tenant, COLLECTION)
    assert counts["knowledge_documents"] >= 1
    assert counts["knowledge_chunks"] >= 1
    assert counts["knowledge_embeddings"] >= 1
    assert counts["knowledge_revisions"] >= 1
    assert counts["knowledge_graph_nodes"] >= 1
    assert counts["knowledge_graph_edges"] >= 1
    context = runtime.retrieve(_query(tenant))
    assert context.hits
    assert all(
        hit.citation.chunk_id == hit.chunk.chunk_id
        and hit.citation.content_digest == hit.chunk.content_digest
        for hit in context.hits
    )
    vector_hits = store.search_vector(
        tenant_id=tenant,
        collection_id=COLLECTION,
        principal_id=ALICE,
        vector=runtime.embeddings.embed(
            ["Alice works"], runtime.config.embedding_profile
        )[0],
        limit=4,
    )
    assert vector_hits
    keyword_hits = store.search_keyword(
        tenant_id=tenant,
        collection_id=COLLECTION,
        principal_id=ALICE,
        query="Alice works at Acme",
        limit=4,
    )
    assert keyword_hits


def test_live_vector_and_keyword_are_tenant_authorized(
    live_env: tuple[str, str],
) -> None:
    postgres, redis_dsn = live_env
    tenant = _tenant()
    other = _tenant()
    runtime = _runtime(postgres, redis_dsn)
    runtime.ingest(_document(tenant, text="Alice works at Acme secret runbook."))
    vector = runtime.embeddings.embed(
        ["Alice works"], runtime.config.embedding_profile
    )[0]
    assert (
        runtime.store.search_vector(
            tenant_id=other,
            collection_id=COLLECTION,
            principal_id=ALICE,
            vector=vector,
            limit=8,
        )
        == ()
    )
    assert (
        runtime.store.search_keyword(
            tenant_id=other,
            collection_id=COLLECTION,
            principal_id=ALICE,
            query="Alice works at Acme secret runbook",
            limit=8,
        )
        == ()
    )
    assert runtime.retrieve(_query(other)).hits == ()
    assert runtime.retrieve(_query(tenant, principal=BOB)).hits == ()


def test_live_hybrid_ordering_is_deterministic(live_env: tuple[str, str]) -> None:
    postgres, redis_dsn = live_env
    tenant = _tenant()
    runtime = _runtime(postgres, redis_dsn)
    runtime.ingest(
        _document(
            tenant,
            text="Alice works at Acme and reviews tickets daily.",
            source_id="a",
        )
    )
    runtime.ingest(
        _document(
            tenant,
            text="Bob works at Acme and reviews the same tickets.",
            source_id="b",
        )
    )
    first = runtime.retrieve(_query(tenant, "reviews tickets"))
    second = runtime.retrieve(_query(tenant, "reviews tickets"))
    assert [hit.chunk.chunk_id for hit in first.hits] == [
        hit.chunk.chunk_id for hit in second.hits
    ]


def test_live_redis_hit_miss_outage_malformed_and_invalidation(
    live_env: tuple[str, str],
) -> None:
    postgres, redis_dsn = live_env
    tenant = _tenant()
    runtime = _runtime(postgres, redis_dsn)
    runtime.ingest(_document(tenant, text="Alice works at Acme.", source_id="one"))
    miss = runtime.retrieve(_query(tenant))
    hit = runtime.retrieve(_query(tenant))
    assert miss.diagnostics.cache_hit is False
    assert hit.diagnostics.cache_hit is True

    import redis

    client = redis.Redis.from_url(redis_dsn)
    keys = list(client.scan_iter(match="cbrain:knowledge:v1:*"))
    assert keys
    for key in keys:
        client.set(key, b"not-json")
    malformed = runtime.retrieve(_query(tenant))
    assert malformed.hits
    assert malformed.diagnostics.cache_hit is False

    runtime.ingest(
        _document(tenant, text="Alice works at Globex now.", source_id="two")
    )
    refreshed = runtime.retrieve(_query(tenant))
    assert refreshed.diagnostics.cache_hit is False

    recovered = KnowledgeRuntime.from_config(
        config=_config(postgres, "redis://127.0.0.1:1/0"),
        clock=lambda: NOW,
    ).retrieve(_query(tenant))
    assert recovered.hits
    assert recovered.diagnostics.cache_hit is False
    blob = recovered.quoted_evidence + str(dict(recovered.diagnostics.latency_ms))
    assert "postgres://" not in blob
    assert "redis://" not in blob


def test_live_tombstone_and_idempotent_and_new_revision(
    live_env: tuple[str, str],
) -> None:
    postgres, redis_dsn = live_env
    tenant = _tenant()
    runtime = _runtime(postgres, redis_dsn)
    document = _document(tenant, text="Alice works at Acme.")
    first = runtime.ingest(document)
    second = runtime.ingest(document)
    assert first.idempotent is False
    assert second.idempotent is True
    assert first.source_revision == second.source_revision == 1
    changed = runtime.ingest(_document(tenant, text="Alice works at Globex."))
    assert changed.source_revision == 2
    runtime.tombstone(tenant, COLLECTION, "src-1")
    assert runtime.retrieve(_query(tenant)).hits == ()


def test_live_dimension_and_profile_mismatch_fail_before_mutation(
    live_env: tuple[str, str],
) -> None:
    postgres, redis_dsn = live_env
    tenant = _tenant()
    runtime = _runtime(postgres, redis_dsn)

    class WrongDimension:
        def embed(self, texts: Any, profile: EmbeddingProfile) -> Any:
            return tuple((0.0,) * (profile.dimension + 1) for _ in texts)

    with pytest.raises(KnowledgeError, match="dimension"):
        _runtime(postgres, redis_dsn, embeddings=WrongDimension()).ingest(
            _document(tenant, text="Alice works at Acme.")
        )
    assert _pg_store(runtime).row_counts(tenant, COLLECTION)["knowledge_documents"] == 0

    runtime.ingest(_document(tenant, text="Alice works at Acme."))
    other_profile = replace(
        default_knowledge_config(),
        postgres_dsn=postgres,
        redis_dsn=redis_dsn,
        embedding_profile=EmbeddingProfile(
            profile_id="other-8",
            model_id="other-model",
            dimension=INDEXED_EMBEDDING_DIMENSION,
            distance_metric=DistanceMetric.COSINE,
            normalization_version="v1",
        ),
    )
    before = _pg_store(runtime).row_counts(tenant, COLLECTION)["knowledge_documents"]
    with pytest.raises(KnowledgeError, match="immutable"):
        KnowledgeRuntime.from_config(
            config=other_profile,
            clock=lambda: NOW,
        ).ingest(_document(tenant, text="Alice works at Globex.", source_id="other"))
    assert (
        _pg_store(runtime).row_counts(tenant, COLLECTION)["knowledge_documents"]
        == before
    )


def test_live_concurrent_ingestion_is_monotonic(live_env: tuple[str, str]) -> None:
    postgres, redis_dsn = live_env
    tenant = _tenant()
    errors: list[Exception] = []

    def _ingest(source_id: str, text: str) -> None:
        try:
            _runtime(postgres, redis_dsn).ingest(
                _document(tenant, text=text, source_id=source_id)
            )
        except Exception as exc:
            errors.append(exc)

    workers = [
        threading.Thread(
            target=_ingest, args=("src-a", "Alice works at Acme engineering.")
        ),
        threading.Thread(
            target=_ingest, args=("src-b", "Bob works at Acme operations.")
        ),
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert errors == []
    status = _runtime(postgres, redis_dsn).collection_status(tenant, COLLECTION)
    assert status["collection_revision"] == 2


def test_live_mid_transaction_failure_leaves_zero_partial_state(
    live_env: tuple[str, str],
) -> None:
    postgres, redis_dsn = live_env
    tenant = _tenant()
    runtime = _runtime(postgres, redis_dsn)
    store = _pg_store(runtime)
    store.fail_after = "embeddings"
    with pytest.raises(KnowledgeUnavailable):
        runtime.ingest(_document(tenant, text="Alice works at Acme."))
    assert store.row_counts(tenant, COLLECTION) == {
        "knowledge_documents": 0,
        "knowledge_chunks": 0,
        "knowledge_embeddings": 0,
        "knowledge_revisions": 0,
        "knowledge_graph_nodes": 0,
        "knowledge_graph_edges": 0,
        "ingestion_jobs": 0,
    }


def test_live_graph_bounds_conflicts_and_isolation(
    live_env: tuple[str, str],
) -> None:
    postgres, redis_dsn = live_env
    tenant = _tenant()
    other = _tenant()
    runtime = _runtime(postgres, redis_dsn)
    runtime.ingest(_document(tenant, text="Alice works at Acme.", source_id="acme"))
    runtime.ingest(_document(tenant, text="Alice works at Globex.", source_id="globex"))
    nodes = _pg_store(runtime).list_nodes(tenant)
    people = [
        node
        for node in nodes
        if node.canonical_type == "person" and node.canonical_name == "alice"
    ]
    assert len(people) == 2
    score = resolve_entities((people[0],), people[1])[1]
    assert score < 1.0
    if len(nodes) >= 2:
        left, right = nodes[0], nodes[1]
        runtime.store.upsert_edges(
            [
                GraphEdge(
                    edge_id=f"cycle-{tenant}",
                    tenant_id=tenant,
                    source_node_id=right.node_id,
                    relation_type="employs",
                    target_node_id=left.node_id,
                    provenance_chunk_ids=left.provenance_chunk_ids,
                    confidence=0.5,
                    valid_from=NOW,
                    valid_to=None,
                    source_revision=1,
                    source_id="src-1",
                    schema_version=1,
                )
            ]
        )
    paths = runtime.graph_path(
        tenant_id=tenant,
        principal_id=ALICE,
        seed_node_ids=tuple(node.node_id for node in nodes),
    )
    assert paths
    assert all(len(path.nodes) <= runtime.config.graph_max_nodes for path in paths)
    assert (
        runtime.graph_path(
            tenant_id=other,
            principal_id=ALICE,
            seed_node_ids=tuple(node.node_id for node in nodes),
        )
        == ()
    )


def test_live_sql_injection_shaped_values_cannot_alter_queries(
    live_env: tuple[str, str],
) -> None:
    postgres, redis_dsn = live_env
    tenant = "t1'; DROP TABLE knowledge_chunks;--"
    runtime = _runtime(postgres, redis_dsn)
    runtime.ingest(
        _document(
            tenant,
            text="Alice works at Acme. '; DROP TABLE knowledge_chunks;--",
            source_id="src'; DELETE FROM knowledge_documents;--",
        )
    )
    context = runtime.retrieve(
        _query(tenant, "Alice works at Acme. '; DROP TABLE knowledge_chunks;--")
    )
    assert context.hits
    _pg_store(runtime).require_schema()


def test_live_secrets_never_appear_in_output(
    live_env: tuple[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    postgres, redis_dsn = live_env
    tenant = _tenant()
    runtime = _runtime(postgres, redis_dsn)
    runtime.ingest(_document(tenant, text="Alice works at Acme."))
    context = runtime.retrieve(_query(tenant))
    blob = (
        repr(runtime.config)
        + context.quoted_evidence
        + str(dict(context.diagnostics.latency_ms))
        + str(runtime.collection_status(tenant, COLLECTION))
    )
    assert postgres not in blob
    assert redis_dsn not in blob
    assert "password" not in blob
    assert knowledge_main(("doctor",)) == 0
    output = capsys.readouterr().out + capsys.readouterr().err
    assert "password=" not in output
    assert "postgresql://" not in output


def test_live_retrieval_cannot_create_action_intent(
    live_env: tuple[str, str],
) -> None:
    postgres, redis_dsn = live_env
    tenant = _tenant()
    runtime = _runtime(postgres, redis_dsn)
    runtime.ingest(_document(tenant, text="Alice works at Acme."))
    runtime.retrieve(_query(tenant))
    import cbrain.knowledge.retrieval as retrieval
    import cbrain.knowledge.stores.postgres as postgres_mod

    assert "ActionIntent" not in inspect.getsource(retrieval)
    assert "ActionIntent" not in inspect.getsource(postgres_mod)


def test_live_required_knowledge_failure_prevents_tool_execution() -> None:
    class AllowGateway:
        def __init__(self) -> None:
            self.actions: list[Any] = []

        def decide_and_execute(self, action: Any, handler: Any) -> Any:
            from cbrain import ExecutionStatus, GovernedExecution

            self.actions.append(action)
            output = handler(action.arguments)
            return GovernedExecution(
                status=ExecutionStatus.EXECUTED,
                request_id=action.request_id,
                tool_executed=True,
                reason="test_allow",
                output=output,
            )

    store = InMemoryKnowledgeStore()
    store.fail_reads = True
    gateway = AllowGateway()
    profile = AgentProfile(
        agent_id="research-agent",
        instructions="Answer with concise facts.",
        model_route="local",
        permitted_tools=frozenset({"lookup"}),
        max_model_turns=4,
        max_tool_calls=2,
        timeout_seconds=30.0,
        knowledge_required_for_tools=True,
    )

    class SequenceModel:
        def __init__(self, outputs: list[Any]) -> None:
            self._outputs = list(outputs)

        @property
        def provider(self) -> str:
            return "sequence"

        @property
        def model(self) -> str:
            return "sequence-v1"

        def complete(self, request: Any) -> Any:
            return self._outputs.pop(0)

    agent = FoundationAgent(
        profile=profile,
        runtime=GovernedRuntime(gateway),
        model_router=ModelRouter(
            {
                profile.model_route: SequenceModel(
                    [
                        ToolCall.capture(
                            call_id="call-1", name="lookup", arguments={"id": "1"}
                        )
                    ]
                )
            }
        ),
        tools=ToolRegistry(
            [
                GovernedTool(
                    name="lookup",
                    capability="data.lookup",
                    description="Look up a record by id",
                    input_schema={
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                    },
                )
            ]
        ),
        handlers={"lookup": lambda _args: {"ok": True}},
        clock=lambda: NOW,
        run_id_factory=lambda: "run-knowledge-live",
        request_id_factory=lambda run_id, step: f"{run_id}-step-{step}",
        knowledge=KnowledgeRuntime(store=store, clock=lambda: NOW).provider,
    )
    result = agent.run(
        RunInput(
            task="Look up Alice",
            metadata={
                "tenant_id": "tenant-a",
                "collection_id": COLLECTION,
                "principal_id": ALICE,
            },
        )
    )
    assert result.status is RunStatus.TOOL_FAILURE
    assert gateway.actions == []


def test_production_from_config_without_postgres_fails_closed() -> None:
    with pytest.raises((KnowledgeConfigError, KnowledgeError, KnowledgeUnavailable)):
        KnowledgeRuntime.from_config(
            config=replace(
                default_knowledge_config(), postgres_dsn=None, redis_dsn=None
            ),
            production=True,
        )
