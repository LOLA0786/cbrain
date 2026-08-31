"""Redis query cache. Never the source of truth or an authorization signal."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from ..contracts import CacheKey, canonical_json, finite_positive_ttl
from ..errors import KnowledgeError

CACHE_VALUE_SCHEMA = 1
MAX_CACHE_VALUE_BYTES = 65_536
CACHE_KEY_PREFIX = "cbrain:knowledge:v1:"


def ttl_to_milliseconds(ttl_seconds: float) -> int:
    ttl = finite_positive_ttl(ttl_seconds)
    milliseconds = int(ttl * 1000)
    if milliseconds < 1:
        raise KnowledgeError(
            "ttl_seconds is too small to preserve without truncating to zero"
        )
    return milliseconds


def encode_cache_value(chunk_ids: Sequence[str]) -> str:
    cleaned: list[str] = []
    for chunk_id in chunk_ids:
        if not isinstance(chunk_id, str) or not chunk_id or "\x00" in chunk_id:
            raise KnowledgeError("cache value is invalid")
        cleaned.append(chunk_id)
    payload = canonical_json({"schema": CACHE_VALUE_SCHEMA, "chunk_ids": cleaned})
    if len(payload.encode("utf-8")) > MAX_CACHE_VALUE_BYTES:
        raise KnowledgeError("cache value exceeds max size")
    return payload


def decode_cache_value(raw: str) -> tuple[str, ...] | None:
    if not raw or len(raw.encode("utf-8")) > MAX_CACHE_VALUE_BYTES:
        return None
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema") != CACHE_VALUE_SCHEMA:
        return None
    chunk_ids = payload.get("chunk_ids")
    if not isinstance(chunk_ids, list):
        return None
    cleaned: list[str] = []
    for chunk_id in chunk_ids:
        if not isinstance(chunk_id, str) or not chunk_id or "\x00" in chunk_id:
            return None
        cleaned.append(chunk_id)
    return tuple(cleaned)


def _redis() -> Any:
    try:
        import redis
    except ImportError as exc:
        raise KnowledgeError("cache unavailable") from exc
    return redis


class RedisKVCache:
    def __init__(self, dsn: str) -> None:
        if not isinstance(dsn, str) or not dsn.strip():
            raise KnowledgeError("cache unavailable")
        self._dsn = dsn

    def __repr__(self) -> str:
        return "RedisKVCache(dsn=redacted)"

    def _client(self) -> Any:
        redis = _redis()
        try:
            return redis.Redis.from_url(
                self._dsn,
                socket_connect_timeout=1.0,
                socket_timeout=1.0,
                decode_responses=False,
            )
        except Exception as exc:
            raise KnowledgeError("cache unavailable") from exc

    def _namespaced(self, key: CacheKey) -> str:
        return CACHE_KEY_PREFIX + key.encoded()

    def get(self, key: CacheKey) -> tuple[str, ...] | None:
        try:
            raw = self._client().get(self._namespaced(key))
        except Exception:
            return None
        if raw is None:
            return None
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        return decode_cache_value(text)

    def set(self, key: CacheKey, chunk_ids: Sequence[str], ttl_seconds: float) -> None:
        encoded = encode_cache_value(chunk_ids)
        milliseconds = ttl_to_milliseconds(ttl_seconds)
        try:
            self._client().set(self._namespaced(key), encoded, px=milliseconds)
        except Exception as exc:
            raise KnowledgeError("cache unavailable") from exc

    def ping(self) -> None:
        try:
            self._client().ping()
        except Exception as exc:
            raise KnowledgeError("cache unavailable") from exc


__all__ = [
    "CACHE_VALUE_SCHEMA",
    "RedisKVCache",
    "decode_cache_value",
    "encode_cache_value",
    "ttl_to_milliseconds",
]
