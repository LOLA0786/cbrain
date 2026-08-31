CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS knowledge_collections (
    tenant_id TEXT NOT NULL,
    collection_id TEXT NOT NULL,
    embedding_profile_id TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    distance_metric TEXT NOT NULL,
    normalization_version TEXT NOT NULL,
    collection_revision INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tenant_id, collection_id),
    CHECK (length(tenant_id) > 0),
    CHECK (length(collection_id) > 0),
    CHECK (embedding_dimension > 0),
    CHECK (collection_revision >= 0)
);

CREATE TABLE IF NOT EXISTS knowledge_documents (
    tenant_id TEXT NOT NULL,
    collection_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_revision INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    provenance TEXT NOT NULL,
    acl_principals TEXT[] NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    schema_version INTEGER NOT NULL,
    tombstoned BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (tenant_id, collection_id, source_id, source_revision),
    CHECK (source_revision > 0)
);

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    chunk_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    collection_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_revision INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    provenance TEXT NOT NULL,
    acl_principals TEXT[] NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    schema_version INTEGER NOT NULL,
    text TEXT NOT NULL,
    CHECK (chunk_index >= 0)
);

CREATE TABLE IF NOT EXISTS knowledge_embeddings (
    chunk_id TEXT PRIMARY KEY REFERENCES knowledge_chunks(chunk_id),
    tenant_id TEXT NOT NULL,
    collection_id TEXT NOT NULL,
    embedding vector NOT NULL,
    profile_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge_graph_nodes (
    node_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    canonical_type TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    attributes JSONB NOT NULL,
    provenance_chunk_ids TEXT[] NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    valid_from TIMESTAMPTZ NOT NULL,
    valid_to TIMESTAMPTZ,
    source_revision INTEGER NOT NULL,
    source_id TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    CHECK (cardinality(provenance_chunk_ids) > 0)
);

CREATE TABLE IF NOT EXISTS knowledge_graph_edges (
    edge_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    source_node_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    target_node_id TEXT NOT NULL,
    provenance_chunk_ids TEXT[] NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    valid_from TIMESTAMPTZ NOT NULL,
    valid_to TIMESTAMPTZ,
    source_revision INTEGER NOT NULL,
    source_id TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    CHECK (cardinality(provenance_chunk_ids) > 0)
);

CREATE TABLE IF NOT EXISTS knowledge_revisions (
    tenant_id TEXT NOT NULL,
    collection_id TEXT NOT NULL,
    collection_revision INTEGER NOT NULL,
    published_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tenant_id, collection_id, collection_revision)
);

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    job_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    collection_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_tenant_acl
    ON knowledge_chunks (tenant_id, collection_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_acl
    ON knowledge_chunks USING GIN (acl_principals);
CREATE INDEX IF NOT EXISTS idx_knowledge_documents_tenant
    ON knowledge_documents (tenant_id, collection_id, tombstoned);
CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_tenant
    ON knowledge_graph_nodes (tenant_id, canonical_name);
CREATE INDEX IF NOT EXISTS idx_knowledge_edges_tenant
    ON knowledge_graph_edges (tenant_id, relation_type);
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_fts
    ON knowledge_chunks USING GIN (to_tsvector('simple', text));
