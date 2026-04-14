"""Selective-concurrency update processor for PTB.

Regular updates process sequentially *within a single topic* (one chat +
message_thread_id pair). Different topics — different chats or different
forum topics in the same supergroup — run concurrently.

Priority callbacks (``stop:*``) bypass all locks and run immediately so they
can interrupt the currently-running handler.
"""

import asyncio
from typing import Any, Awaitable, Dict, Optional, Tuple

from telegram import Update
from telegram.ext._baseupdateprocessor import BaseUpdateProcessor

# Fallback lock key for updates we can't associate with a (chat, thread).
_FALLBACK_KEY: Tuple[int, int] = (0, 0)


class StopAwareUpdateProcessor(BaseUpdateProcessor):
    """Update processor with per-topic locks and a stop-callback fast path.

    PTB calls ``process_update(update, coroutine)`` for every incoming update.
    The base class holds a semaphore (max 256) then calls
    ``do_process_update()``.

    For priority callbacks (``stop:*``): we ``await coroutine`` directly —
    runs immediately, no lock.

    For everything else: we acquire the lock for the update's
    ``(chat_id, message_thread_id or 0)``. Two updates in the same topic
    queue; updates in different topics run in parallel.

    A stop callback arrives while a text handler holds a topic lock -> stop
    callback runs concurrently -> fires the ``asyncio.Event`` -> the watcher
    task inside ``execute_command()`` calls ``client.interrupt()`` -> Claude
    stops -> ``run_command()`` returns -> handler finishes -> lock released.
    """

    _PRIORITY_PREFIXES = ("stop:",)

    def __init__(self) -> None:
        # High limit so priority callbacks are never blocked by semaphore
        super().__init__(max_concurrent_updates=256)
        self._topic_locks: Dict[Tuple[int, int], asyncio.Lock] = {}
        # Guards insertion into _topic_locks so two concurrent first-uses of
        # the same key don't race on dict setdefault under asyncio.
        self._registry_lock = asyncio.Lock()

    @classmethod
    def _is_priority_callback(cls, update: object) -> bool:
        """Return True if the update is a priority callback query."""
        if not isinstance(update, Update):
            return False
        cb = update.callback_query
        return (
            cb is not None
            and cb.data is not None
            and cb.data.startswith(cls._PRIORITY_PREFIXES)
        )

    @staticmethod
    def _topic_key(update: object) -> Tuple[int, int]:
        """Extract a (chat_id, message_thread_id or 0) key from an update.

        Falls back to ``_FALLBACK_KEY`` if neither a chat nor a message can
        be resolved (should be rare — status updates, etc.).
        """
        if not isinstance(update, Update):
            return _FALLBACK_KEY

        # Prefer effective_chat + effective_message (covers messages,
        # edited messages, callback queries, channel posts).
        chat = update.effective_chat
        chat_id = chat.id if chat is not None else None

        message = update.effective_message
        thread_id: Optional[int] = (
            getattr(message, "message_thread_id", None) if message is not None else None
        )

        if chat_id is None:
            return _FALLBACK_KEY
        return (chat_id, thread_id or 0)

    async def _get_lock(self, key: Tuple[int, int]) -> asyncio.Lock:
        """Return (creating if needed) the lock for the given topic key."""
        lock = self._topic_locks.get(key)
        if lock is not None:
            return lock
        async with self._registry_lock:
            # Re-check under the registry lock to avoid a TOCTOU race.
            lock = self._topic_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._topic_locks[key] = lock
            return lock

    async def do_process_update(
        self,
        update: object,
        coroutine: Awaitable[Any],
    ) -> None:
        """Process an update.

        Priority callbacks run immediately. All other updates acquire the
        per-topic lock derived from the update's (chat_id, thread_id) before
        running.
        """
        if self._is_priority_callback(update):
            await coroutine
            return

        lock = await self._get_lock(self._topic_key(update))
        async with lock:
            await coroutine

    async def initialize(self) -> None:
        """Initialize the processor (no-op)."""

    async def shutdown(self) -> None:
        """Shutdown the processor (no-op)."""
