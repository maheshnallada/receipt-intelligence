"""One GPU job at a time. A second concurrent forward pass OOMs a 16 GB T4."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.domain.errors import GpuBusy, QueueFull


class SingleFlightGpuSlot:
    """Bounded wait queue in front of a mutex. Depth includes the holder."""

    def __init__(self, depth: int = 8, timeout_s: float = 60.0) -> None:
        if depth < 1:
            raise ValueError("queue depth must be at least 1")
        self._depth = depth
        self._timeout_s = timeout_s
        self._mutex = asyncio.Lock()
        self._occupied = 0
        self._gate = asyncio.Lock()

    @property
    def occupied(self) -> int:
        return self._occupied

    @asynccontextmanager
    async def hold(self) -> AsyncIterator[float]:
        async with self._gate:
            if self._occupied >= self._depth:
                raise QueueFull("GPU queue is full")
            self._occupied += 1
        started = time.perf_counter()
        try:
            try:
                await asyncio.wait_for(self._mutex.acquire(), timeout=self._timeout_s)
            except TimeoutError as exc:
                raise GpuBusy("queue wait exceeded QUEUE_TIMEOUT_S") from exc
            wait_ms = (time.perf_counter() - started) * 1000.0
            try:
                yield wait_ms
            finally:
                self._mutex.release()
        finally:
            async with self._gate:
                self._occupied -= 1
