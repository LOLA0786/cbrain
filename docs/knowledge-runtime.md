# CBrain Knowledge Runtime v0.6

Retrieved knowledge is context, not authority. All consequential actions remain
subject to CBrain invariants, PrivateVault authorization, atomic consumption,
exact-byte sidecar dispatch, and verified closure.

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

## Ingestion lifecycle

1. Validate media type and size.
2. Normalize whitespace.
3. Deterministic chunking with configured size/overlap.
4. Content digest and deterministic chunk IDs
   (`tenant_id + source_id + source_revision + chunk_index + normalized_text_digest`).
5. Idempotent re-ingestion of the same revision.
6. Fake or deployment-owned embeddings (unit tests never call a live provider).
7. Rule-based entity/relation extraction (not verified facts).
8. One transaction publishes document, vector, and graph state, then increments
   the collection revision.
9. Cache keys include that revision, so old results expire automatically.

A changed source creates a new revision. Deleted sources are tombstoned and
excluded from retrieval. Partial failure does not publish.

After the first published revision, the collection embedding profile
(model, dimension, distance metric, normalization version) is immutable.

## Retrieval algorithm

1. Validate tenant, principal, query, and limits.
2. Vector candidates (tenant + ACL filtered).
3. Keyword candidates (PostgreSQL FTS in production; in-memory overlap in tests).
4. Deterministic Reciprocal Rank Fusion (`k=60`).
5. Optional bounded graph expansion.
6. ACL post-filter.
7. Deterministic rerank by fused score, then source id, then chunk index.
8. Token/byte/chunk budget with source diversity.
9. Citations and content digests on every hit.
10. `RetrievedContext` marked `UNTRUSTED_SOURCE_CONTEXT`.

Diagnostics contain counts, cache hit/miss, per-stage latency, and truncation.
They never contain document bodies, credentials, or connection strings.

## Cache semantics

Redis is cache only. Entries store authorized hit identifiers, query embedding
material when policy permits, graph-neighborhood identifiers, and revision
metadata. They never store credentials, approvals, authorization envelopes,
`ActionIntent`s, sidecar evidence, unrestricted raw documents, or cross-tenant
results.

Every cache key includes schema version, tenant, principal/ACL digest,
collection id and revision, query digest, embedding profile, retrieval
configuration digest, policy version, and `top_k`. TTL must be finite and
positive. Cache failure falls back to the durable store. Durable-store failure
returns `KNOWLEDGE_UNAVAILABLE`. A cache hit is never authorization.

## Graph provenance

Nodes and edges require provenance chunk IDs. Extracted relations are not
authority or verified facts. Conflicting claims from different sources remain
distinct; entity resolution exposes a merge score and reason. Traversal is
bounded by depth, node count, tenant, and ACL, and terminates on cycles.

A future Neo4j/Memgraph adapter must implement `GraphRepository` without
changing `FoundationAgent`.

## Tenant isolation

Tenant and ACL filters apply when selecting vector/keyword candidates and again
after fusion and graph expansion. Cross-tenant retrieval returns zero hits.

## Configuration

Deployment-owned (not model tool arguments):

- PostgreSQL DSN and Redis DSN
- cache TTL
- max document size, chunk size/overlap
- top_k and candidate counts
- graph max depth/nodes
- retrieval timeout
- embedding profile
- context token/byte/chunk budgets

Validate at startup. Doctor never prints secrets.

## Local development

Unit tests use `InMemoryKnowledgeStore`, `DeterministicEmbeddingProvider`,
`RuleBasedExtractor`, and `InMemoryKVCache`. They do not require Docker or
network access.

Production adapters import `psycopg[binary]>=3.2` and `redis>=5.0` only when
those packages are installed in the deployment image. They are not default
CBrain dependencies and are not pinned in `uv.lock` here because the local
workspace also contains a sibling PrivateVault tree that would rewrite the
locked PrivateVault pin.

```text
cbrain-knowledge doctor
cbrain-knowledge ingest --tenant-id T --collection-id C --source-id S \
    --principal-id P --path FILE --provenance LABEL
cbrain-knowledge query --tenant-id T --collection-id C --principal-id P --text Q
cbrain-knowledge graph-path --tenant-id T --principal-id P --seed NODE
cbrain-knowledge collection-status --tenant-id T --collection-id C
```

## Migrations

`migrations/postgres/002_knowledge_runtime.sql` enables `vector` and creates
collections, documents, chunks, embeddings, graph nodes/edges, revisions, and
ingestion jobs with tenant/ACL indexes and full-text search.

## FoundationAgent integration

`KnowledgeContextProvider` is optional. Retrieved evidence is appended as a
quoted user message, never as system instructions. If
`knowledge_required_for_tools` is true and retrieval is unavailable, the agent
must not capture or execute an `ActionIntent`.

## Known limitations

- Production PostgreSQL/Redis adapters are present but not fully exercised
  unless `CBRAIN_KNOWLEDGE_PG_DSN` / `CBRAIN_KNOWLEDGE_REDIS_DSN` are set.
- The default embedding provider is deterministic and not a semantic model.
- Graph storage for v0.6 is PostgreSQL, not Neo4j.
- Keyword search in the unit adapter is term overlap, not `tsvector`.
- Entity extraction is rule-based and low-confidence by design.

## Production-readiness checklist

- [ ] Pin embedding profile before first published revision
- [ ] Provision PostgreSQL with `pgvector` and apply `002_knowledge_runtime.sql`
- [ ] Provision Redis with finite TTLs only
- [ ] Keep DSNs and credentials out of prompts, chunks, logs, and evidence
- [ ] Confirm `required_for_tools` fail-closed behavior in the target profile
- [ ] Confirm PrivateVault still authorizes every consequential tool call
