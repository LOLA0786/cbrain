"""Redis query cache. Never the source of truth or an authorization signal."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..contracts import CacheKey, finite_positive_ttl
from ..errors import KnowledgeError


def _redis() -> Any:
    try:
        import redis  # type: ignore[import-not-found]
    except ImportError as exc:
        raise KnowledgeError("cache unavailable") from exc
    return redis


class RedisKVCache:
    def __init__(self, dsn: str) -> None:
        if not isinstance(dsn, str) or not dsn.strip():
            raise KnowledgeError("cache unavailable")
        self._dsn = dsn

    def _client(self) -> Any:
        redis = _redis()
        try:
            return redis.Redis.from_url(self._dsn, socket_connect_timeout=1.0)
        except Exception as exc:
            raise KnowledgeError("cache unavailable") from exc

    def get(self, key: CacheKey) -> tuple[str, ...] | None:
        client = self._client()
        try:
            raw = client.get(key.encoded())
        except Exception as exc:
            raise KnowledgeError("cache unavailable") from exc
        if raw is None:
            return None
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        if not text:
            return None
        return tuple(part for part in text.split("\n") if part)

    def set(self, key: CacheKey, chunk_ids: Sequence[str], ttl_seconds: float) -> None:
        ttl = finite_positive_ttl(ttl_seconds)
        client = self._client()
        try:
            client.set(key.encoded(), "\n".join(chunk_ids), ex=int(ttl))
        except Exception as exc:
            raise KnowledgeError("cache unavailable") from exc

    def ping(self) -> None:
        client = self._client()
        try:
            client.ping()
        except Exception as exc:
            raise KnowledgeError("cache unavailable") from exc


__all__ = ["RedisKVCache"]
