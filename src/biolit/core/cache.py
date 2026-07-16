"""Redis cache helpers with TTL."""

from __future__ import annotations

from typing import Any

import redis.asyncio as redis

from biolit.core.config import get_settings
from biolit.core.logging import get_logger

logger = get_logger(__name__)

_client: redis.Redis | None = None


async def get_redis() -> redis.Redis:
    """Return a shared async Redis client."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        logger.info("Redis client created", extra={"url": settings.redis_url})
    return _client


async def cache_get(key: str) -> str | None:
    """Get a string value from Redis, or None on miss / error."""
    try:
        client = await get_redis()
        value = await client.get(key)
        if value is None:
            return None
        return str(value)
    except Exception:
        logger.exception("cache_get failed", extra={"key": key})
        return None


async def cache_set(key: str, value: str, ttl_seconds: int | None = None) -> bool:
    """Set a string value in Redis with TTL. Returns False on error."""
    settings = get_settings()
    ttl = ttl_seconds if ttl_seconds is not None else settings.cache_ttl_seconds
    try:
        client = await get_redis()
        await client.set(key, value, ex=ttl)
        return True
    except Exception:
        logger.exception("cache_set failed", extra={"key": key})
        return False


async def cache_delete(key: str) -> bool:
    """Delete a cache key. Returns False on error."""
    try:
        client = await get_redis()
        await client.delete(key)
        return True
    except Exception:
        logger.exception("cache_delete failed", extra={"key": key})
        return False


async def close_redis() -> None:
    """Close the shared Redis client."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def ping_redis() -> dict[str, Any]:
    """Health-check Redis connectivity."""
    try:
        client = await get_redis()
        pong = await client.ping()
        return {"ok": bool(pong)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
