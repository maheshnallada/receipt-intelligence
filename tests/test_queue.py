from __future__ import annotations

import asyncio

import pytest

from app.domain.errors import QueueFull
from app.infrastructure.gpu.single_flight import SingleFlightGpuSlot


@pytest.mark.asyncio
async def test_five_callers_run_one_at_a_time() -> None:
    slot = SingleFlightGpuSlot(depth=8, timeout_s=5)
    current = 0
    peak = 0
    lock = asyncio.Lock()

    async def worker() -> None:
        nonlocal current, peak
        async with slot.hold():
            async with lock:
                current += 1
                peak = max(peak, current)
            await asyncio.sleep(0.05)
            async with lock:
                current -= 1

    await asyncio.gather(*[worker() for _ in range(5)])
    assert peak == 1


@pytest.mark.asyncio
async def test_overflow_raises_queue_full() -> None:
    slot = SingleFlightGpuSlot(depth=2, timeout_s=5)
    started = asyncio.Event()
    release = asyncio.Event()

    async def holder() -> None:
        async with slot.hold():
            started.set()
            await release.wait()

    first = asyncio.create_task(holder())
    await started.wait()
    second = asyncio.create_task(holder())
    await asyncio.sleep(0.02)

    with pytest.raises(QueueFull):
        async with slot.hold():
            pass

    release.set()
    await asyncio.gather(first, second)
