"""Opt-in PostgreSQL/pgvector and Redis integration tests.

These tests skip only when the explicit integration environment is absent.
Unit tests never skip.
"""

from __future__ import annotations

import os

import pytest

from cbrain.knowledge.errors import KnowledgeUnavailable
from cbrain.knowledge.stores.postgres import PostgresKnowledgeStore
from cbrain.knowledge.stores.redis_cache import RedisKVCache

_PG = os.environ.get("CBRAIN_KNOWLEDGE_PG_DSN")
_REDIS = os.environ.get("CBRAIN_KNOWLEDGE_REDIS_DSN")


@pytest.mark.skipif(not _PG, reason="CBRAIN_KNOWLEDGE_PG_DSN is not set")
def test_postgres_ping_when_configured() -> None:
    store = PostgresKnowledgeStore(_PG or "", timeout_seconds=2.0)
    store.ping()


@pytest.mark.skipif(not _REDIS, reason="CBRAIN_KNOWLEDGE_REDIS_DSN is not set")
def test_redis_ping_when_configured() -> None:
    RedisKVCache(_REDIS or "").ping()


def test_postgres_blank_dsn_is_unavailable() -> None:
    with pytest.raises(KnowledgeUnavailable):
        PostgresKnowledgeStore("   ", timeout_seconds=1.0).ping()
