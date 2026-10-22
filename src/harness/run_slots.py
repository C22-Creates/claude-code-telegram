"""RunSlots: the global cap on concurrent agent runs.

Runs in different topics are parallel up to ``cap`` (settings
``harness_concurrency_cap``). When every slot is busy, acquirers wait; a freed
slot is handed directly to the next waiter -- interactive waiters before
scheduled ones, FIFO within a class -- so a fresh acquirer can never jump the
queue and scheduled jobs never pre-empt Carl's messages.
"""

import asyncio
from collections import deque
from typing import Literal

import structlog

logger = structlog.get_logger()

Priority = Literal["interactive", "scheduled"]
PRIORITIES: tuple[Priority, ...] = ("interactive", "scheduled")  # serve order


class _Slot:
    """``async with`` handle for one slot; acquired on enter, freed on exit."""

    def __init__(self, slots: "RunSlots", priority: Priority) -> None:
        self._slots = slots
        self._priority = priority

    async def __aenter__(self) -> None:
        await self._slots._take(self._priority)

    async def __aexit__(self, *exc: object) -> None:
        self._slots._release()


class RunSlots:
    """Counting semaphore with two priority classes of waiters."""

    def __init__(self, cap: int) -> None:
        if cap < 1:
            raise ValueError(f"concurrency cap must be >= 1, got {cap}")
        self._cap = cap
        self._in_use = 0
        self._waiters: dict[str, deque[asyncio.Future[None]]] = {
            p: deque() for p in PRIORITIES
        }

    @property
    def in_use(self) -> int:
        return self._in_use

    def would_wait(self) -> bool:
        """True when no slot is free right now (an acquire would queue)."""
        return self._in_use >= self._cap

    def acquire(self, priority: Priority) -> _Slot:
        """Return an async context manager holding one slot while inside it.

        Usage: ``async with slots.acquire("interactive"): ...``. Waiting happens
        on enter; cancelling a waiter never leaks a slot.
        """
        if priority not in self._waiters:
            raise ValueError(f"unknown priority {priority!r}")
        return _Slot(self, priority)

    async def _take(self, priority: Priority) -> None:
        if self._in_use < self._cap and not any(self._waiters.values()):
            self._in_use += 1
            return
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        queue = self._waiters[priority]
        queue.append(fut)
        logger.info("harness.slot_wait", priority=priority, in_use=self._in_use)
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                # Granted, then cancelled before resuming: pass the slot on.
                self._release()
            else:
                try:
                    queue.remove(fut)
                except ValueError:
                    pass
            raise

    def _release(self) -> None:
        """Hand the slot to the next live waiter, or free it."""
        for priority in PRIORITIES:
            queue = self._waiters[priority]
            while queue:
                fut = queue.popleft()
                if not fut.done():
                    fut.set_result(None)  # the slot transfers; in_use unchanged
                    return
        self._in_use -= 1
