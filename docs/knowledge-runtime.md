# CBrain Knowledge Runtime v0.6.1

Retrieved knowledge is context, not authority. All consequential actions remain
subject to CBrain invariants, PrivateVault authorization, atomic consumption,
exact-byte sidecar dispatch, and verified closure.

This document distinguishes three evidence classes:

- **Unit tests** (`tests/test_knowledge_runtime.py`): in-memory adapters only.
  They are network-free and do not prove PostgreSQL, pgvector, or Redis behavior.
- **Live integration tests** (`tests/test_knowledge_runtime_integration.py`):
  require `CBRAIN_KNOWLEDGE_LIVE=1` plus PostgreSQL/pgvector and Redis DSNs.
  When that flag is unset they skip with `CBRAIN_KNOWLEDGE_LIVE is not set`.
  When the flag is set, missing DSNs fail rather than skip.
- **Production evidence**: a successful mandatory `knowledge-live` CI job
  against pinned PostgreSQL+pgvector and Redis services. Do not treat local
  unit results as production adapter proof.

## Architecture and authority boundary

The knowledge runtime supplies **untrusted source context** to reusable
foundation agents. It never creates, approves, signs, consumes, or dispatches
an `ActionIntent`. Memory, RAG, cache, embeddings, and graph records have no
execution authority.

```text
source → normalize → chunk → digest → embed → transactional publish
query  → vector + keyword → RRF → optional graph → ACL → budget → citations
```

PrivateVault remains the only authorization and evidence authority. Instructions
found inside retrieved documents are quoted as untrusted data.

When `knowledge_required_for_tools=True`, missing, malformed, timed-out, or
unavailable durable knowledge prevents consequential tool execution before an
`ActionIntent` is captured.

## Store abstractions

`KnowledgeRuntime`, ingestion, retrieval, and graph traversal depend on the
`KnowledgeStore` and `KVCache` ports. Constructors:

- `KnowledgeRuntime.in_memory(...)` for deterministic unit tests
- `KnowledgeRuntime.from_config(...)` for deployment-owned adapters

External services are not contacted during module import. `from_config` uses
PostgreSQL when `postgres_dsn` is set or `CBRAIN_KNOWLEDGE_MODE=production`.
Production mode without a PostgreSQL DSN fails closed.

## Ingestion lifecycle

1. Validate media type and size.
2. Normalize whitespace.
3. Deterministic chunking with configured size/overlap.
4. Content digest and deterministic chunk IDs
   (`tenant_id + source_id + source_revision + chunk_index + normalized_text_digest`).
5. Idempotent re-ingestion of the same revision.
6. Injected `EmbeddingProvider` (unit tests use `DeterministicEmbeddingProvider`).
7. Rule-based entity/relation extraction (not verified facts).
8. One PostgreSQL transaction publishes collection lock/profile, document,
   chunks, embeddings, graph nodes/edges, collection revision, and ingestion-job
   completion. Any failure rolls back all of that state.
9. Cache keys include the collection revision, so previous results miss.

A changed source creates a new revision. Deleted sources are tombstoned and
excluded from retrieval SQL. Partial failure does not publish.

After the first published revision, the collection embedding profile
(model, dimension, distance metric, normalization version) is immutable.

## Vector index strategy

Indexed retrieval uses a **fixed deployment-owned dimension of 8**, matching
`INDEXED_EMBEDDING_DIMENSION` and the default `deterministic-8` profile.

- Column type: `vector(8)`
- Index: HNSW with `vector_cosine_ops`
- Dimension is validated before any embedding insert
- Changing dimension requires a new collection and a later schema migration.
  This runtime does not claim indexed retrieval for untyped `vector` columns
  or for profiles whose dimension is not 8.

## Retrieval algorithm

1. Validate tenant, principal, query, and limits.
2. PostgreSQL pgvector candidates, filtered by tenant, collection, live
   revision, tombstone, and ACL **inside SQL**.
3. PostgreSQL FTS candidates via `to_tsvector('simple', text)` and
   `plainto_tsquery('simple', %s)`, with the same SQL filters.
4. Deterministic Reciprocal Rank Fusion (`k=60`).
5. Optional bounded graph expansion with tenant/ACL, cycle termination, depth
   and node limits.
6. ACL post-filter.
7. Deterministic rerank by fused score, then source id, then chunk index.
8. Token/byte/chunk budget with source diversity.
9. Citations and content digests on every hit.
10. `RetrievedContext` marked `UNTRUSTED_SOURCE_CONTEXT`.

Diagnostics contain counts, cache hit/miss, per-stage latency, and truncation.
They never contain document bodies, credentials, or connection strings.

## Redis cache

Redis is optional and never the source of truth.

- Outage or a malformed/oversized value is a cache miss; retrieval falls back
  to PostgreSQL.
- PostgreSQL outage remains `KNOWLEDGE_UNAVAILABLE`.
- Values are versioned canonical JSON `{"schema":1,"chunk_ids":[...]}`.
- TTL is stored as milliseconds (`PX`). A positive TTL that would truncate to
  zero milliseconds is rejected.
- Keys include schema version, tenant, collection, revision, principal/ACL
  digest, query digest, embedding profile, retrieval configuration, policy
  version, and `top_k`.
- Collection revision changes cause previous keys to miss.
- Raw documents, credentials, approvals, authorization envelopes, and
  `ActionIntent`s are not cached.

## Configuration and secrets

Deployment-owned (not model tool arguments):

- `CBRAIN_KNOWLEDGE_PG_DSN`
- `CBRAIN_KNOWLEDGE_REDIS_DSN`
- `CBRAIN_KNOWLEDGE_MODE=production` to require PostgreSQL
- cache TTL, chunk limits, top_k, graph bounds, retrieval timeout
- embedding profile
- context token/byte/chunk budgets

DSN fields are omitted from dataclass `repr` and must not appear in exceptions,
logs, CLI output, or diagnostics. There are no default passwords.

Install the locked extra:

```text
uv sync --locked --extra knowledge
```

Keep `EmbeddingProvider` injectable. This tree does not add an external
embedding API or credential-bearing model tool.

## Local development and provisioning

Unit tests:

```text
uv run python -m pytest -q tests/test_knowledge_runtime.py
```

Live PostgreSQL/pgvector and Redis:

1. Provision PostgreSQL 16 with the `vector` extension and Redis.
2. Apply `migrations/postgres/002_knowledge_runtime.sql` then
   `migrations/postgres/003_knowledge_runtime_vector_index.sql`
   (or `apply_knowledge_migrations(dsn, timeout_seconds=...)`).
3. Export:

```text
CBRAIN_KNOWLEDGE_LIVE=1
CBRAIN_KNOWLEDGE_PG_DSN=...
CBRAIN_KNOWLEDGE_REDIS_DSN=...
```

4. Run `uv run python -m pytest -q tests/test_knowledge_runtime_integration.py`

CLI commands `doctor`, `ingest`, `query`, `graph-path`, and
`collection-status` use `KnowledgeRuntime.from_config`. Doctor checks
configuration, PostgreSQL connectivity, pgvector, schema compatibility, and
Redis without printing secrets. Production mode with unavailable PostgreSQL or
an incompatible schema exits nonzero.

```text
cbrain-knowledge doctor
cbrain-knowledge ingest --tenant-id T --collection-id C --source-id S \
    --principal-id P --path FILE --provenance LABEL
cbrain-knowledge query --tenant-id T --collection-id C --principal-id P --text Q
cbrain-knowledge graph-path --tenant-id T --principal-id P --seed NODE
cbrain-knowledge collection-status --tenant-id T --collection-id C
```

## FoundationAgent integration

`KnowledgeContextProvider` is optional. Retrieved evidence is appended as a
quoted user message, never as system instructions. If
`knowledge_required_for_tools` is true and retrieval is unavailable, the agent
must not capture or execute an `ActionIntent`.

## Remaining limitations

- The default embedding provider is deterministic hash-derived vectors, not a
  semantic model. A production embedding service is still an injected
  deployment concern.
- Graph storage is PostgreSQL, not Neo4j/Memgraph.
- Keyword search in the in-memory unit adapter is term overlap, not `tsvector`.
- Entity extraction is rule-based and low-confidence by design.
- Indexed vector search is only defined for dimension 8. Other dimensions are
  rejected rather than silently stored.
- Databases created from the original untyped v0.6 `002` schema receive typed
  `vector(8)` and HNSW from `003`; foreign keys added in the hardened `002`
  are not retrofitted onto tables that already existed without them.
- Local live tests are not production evidence unless the mandatory
  `knowledge-live` CI job also succeeded.
