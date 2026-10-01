"""Redis adapter with two short retries, then memory fallback."""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Any

from app.infrastructure.cache.memory import MemoryCache

logger = logging.getLogger(__name__)


class RedisCache:
    def __init__(self, url: str, fallback: MemoryCache | None = None) -> None:
        import redis

        self._client = redis.Redis.from_url(url, decode_responses=True, socket_timeout=1.5)
        self._fallback = fallback or MemoryCache()
        self._use_fallback = False

    def get(self, key: str) -> dict[str, Any] | None:
        if self._use_fallback:
            return self._fallback.get(key)
        raw = self._call(self._client.get, key)
        if raw is None:
            return None
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return loaded if isinstance(loaded, dict) else None

    def set(self, key: str, value: dict[str, Any], ttl_s: int) -> None:
        payload = json.dumps(value)
        if self._use_fallback:
            self._fallback.set(key, value, ttl_s)
            return
        ok = self._call(self._client.set, key, payload, ex=ttl_s)
        if ok is None and self._use_fallback:
            self._fallback.set(key, value, ttl_s)

    def _call(self, fn, *args, **kwargs):  # type: ignore[no-untyped-def]
        last: Exception | None = None
        for attempt in range(3):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 — redis raises many client errors
                last = exc
                if attempt < 2:
                    time.sleep(0.05 + random.random() * 0.1)
        logger.warning("redis unavailable; falling back to memory cache: %s", last)
        self._use_fallback = True
        return None
