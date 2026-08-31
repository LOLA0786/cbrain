-- Forward migration for v0.6.1 indexed pgvector retrieval.
-- Repeatable. Does not drop or rewrite existing document bodies.

ALTER TABLE knowledge_embeddings
    ALTER COLUMN embedding TYPE vector(8);

CREATE INDEX IF NOT EXISTS idx_knowledge_embeddings_hnsw
    ON knowledge_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_knowledge_embeddings_tenant
    ON knowledge_embeddings (tenant_id, collection_id);

CREATE INDEX IF NOT EXISTS idx_knowledge_revisions_tenant
    ON knowledge_revisions (tenant_id, collection_id);

CREATE INDEX IF NOT EXISTS idx_knowledge_jobs_tenant
    ON ingestion_jobs (tenant_id, collection_id, source_id);
