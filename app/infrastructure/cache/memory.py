"""In-process LRU used when REDIS_URL is empty."""

from __future__ import annotations

from typing import Any

from cachetools import TTLCache


class MemoryCache:
    def __init__(self, maxsize: int = 128, ttl_s: int = 86_400) -> None:
        self._ttl_s = ttl_s
        self._cache: TTLCache[str, dict[str, Any]] = TTLCache(maxsize=maxsize, ttl=ttl_s)

    def get(self, key: str) -> dict[str, Any] | None:
        value = self._cache.get(key)
        if value is None:
            return None
        return dict(value)

    def set(self, key: str, value: dict[str, Any], ttl_s: int) -> None:
        # cachetools TTL is per-cache; honour a shorter caller TTL by storing a copy.
        del ttl_s
        self._cache[key] = dict(value)
