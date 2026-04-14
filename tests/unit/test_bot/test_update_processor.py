"""Tests for StopAwareUpdateProcessor.

Covers:
- Stop callbacks bypass all locks (run immediately)
- Regular updates in the SAME topic are serialized
- Regular updates in DIFFERENT topics run concurrently
- Non-stop callbacks (e.g. cd:) share the per-topic lock
- Updates without a chat fall back to a shared lock
"""

import asyncio
from unittest.mock import MagicMock

from telegram import CallbackQuery, Chat, Message, Update

from src.bot.update_processor import StopAwareUpdateProcessor


def _make_update(
    callback_data: str | None = None,
    chat_id: int | None = 100,
    thread_id: int | None = None,
) -> Update:
    """Build a minimal Update mock with chat + optional callback_query data."""
    update = MagicMock(spec=Update)

    if chat_id is None:
        update.effective_chat = None
        update.effective_message = None
    else:
        chat = MagicMock(spec=Chat)
        chat.id = chat_id
        update.effective_chat = chat

        msg = MagicMock(spec=Message)
        msg.message_thread_id = thread_id
        update.effective_message = msg

    if callback_data is not None:
        cb = MagicMock(spec=CallbackQuery)
        cb.data = callback_data
        update.callback_query = cb
    else:
        update.callback_query = None

    return update


class TestIsPriorityCallback:
    def test_stop_callback_detected(self):
        update = _make_update("stop:123")
        assert StopAwareUpdateProcessor._is_priority_callback(update) is True

    def test_cd_callback_not_priority(self):
        update = _make_update("cd:my_project")
        assert StopAwareUpdateProcessor._is_priority_callback(update) is False

    def test_no_callback_query(self):
        update = _make_update(None)
        assert StopAwareUpdateProcessor._is_priority_callback(update) is False

    def test_non_update_object(self):
        assert StopAwareUpdateProcessor._is_priority_callback("not an update") is False

    def test_callback_with_none_data(self):
        update = MagicMock(spec=Update)
        cb = MagicMock(spec=CallbackQuery)
        cb.data = None
        update.callback_query = cb
        assert StopAwareUpdateProcessor._is_priority_callback(update) is False


class TestTopicKey:
    def test_chat_without_thread_returns_chat_zero(self):
        update = _make_update(chat_id=42, thread_id=None)
        assert StopAwareUpdateProcessor._topic_key(update) == (42, 0)

    def test_chat_with_thread_returns_chat_thread(self):
        update = _make_update(chat_id=42, thread_id=7)
        assert StopAwareUpdateProcessor._topic_key(update) == (42, 7)

    def test_update_without_chat_falls_back(self):
        update = _make_update(chat_id=None)
        assert StopAwareUpdateProcessor._topic_key(update) == (0, 0)

    def test_non_update_object_falls_back(self):
        assert StopAwareUpdateProcessor._topic_key("not an update") == (0, 0)


class TestStopCallbackBypassesLock:
    async def test_stop_callback_runs_while_lock_held(self):
        """A stop callback runs immediately even when a topic lock is held."""
        processor = StopAwareUpdateProcessor()

        execution_order: list[str] = []
        lock_acquired = asyncio.Event()
        stop_done = asyncio.Event()

        async def slow_coroutine():
            execution_order.append("regular_start")
            lock_acquired.set()
            await stop_done.wait()
            execution_order.append("regular_end")

        async def stop_coroutine():
            execution_order.append("stop_start")
            execution_order.append("stop_end")
            stop_done.set()

        regular_update = _make_update(None, chat_id=1)
        stop_update = _make_update("stop:42", chat_id=1)

        regular_task = asyncio.create_task(
            processor.do_process_update(regular_update, slow_coroutine())
        )
        await lock_acquired.wait()

        stop_task = asyncio.create_task(
            processor.do_process_update(stop_update, stop_coroutine())
        )

        await asyncio.gather(regular_task, stop_task)

        assert execution_order == [
            "regular_start",
            "stop_start",
            "stop_end",
            "regular_end",
        ]


class TestSameTopicSerializes:
    async def test_two_updates_same_topic_do_not_overlap(self):
        """Two regular updates in the same topic are serialized."""
        processor = StopAwareUpdateProcessor()

        execution_log: list[str] = []

        async def coroutine_a():
            execution_log.append("a_start")
            await asyncio.sleep(0.05)
            execution_log.append("a_end")

        async def coroutine_b():
            execution_log.append("b_start")
            await asyncio.sleep(0.05)
            execution_log.append("b_end")

        update_a = _make_update(None, chat_id=1, thread_id=10)
        update_b = _make_update(None, chat_id=1, thread_id=10)

        task_a = asyncio.create_task(
            processor.do_process_update(update_a, coroutine_a())
        )
        await asyncio.sleep(0)
        task_b = asyncio.create_task(
            processor.do_process_update(update_b, coroutine_b())
        )

        await asyncio.gather(task_a, task_b)

        assert execution_log == ["a_start", "a_end", "b_start", "b_end"]


class TestDifferentTopicsConcurrent:
    async def test_two_updates_different_threads_run_concurrently(self):
        """Updates in different forum topics of the same chat run in parallel."""
        processor = StopAwareUpdateProcessor()

        execution_log: list[str] = []
        a_started = asyncio.Event()

        async def coroutine_a():
            execution_log.append("a_start")
            a_started.set()
            # Wait for b to also start, proving they overlap
            await asyncio.sleep(0.05)
            execution_log.append("a_end")

        async def coroutine_b():
            # Only start after we know a has started and is holding its lock
            await a_started.wait()
            execution_log.append("b_start")
            execution_log.append("b_end")

        # Same chat, different threads = different topic keys
        update_a = _make_update(None, chat_id=1, thread_id=10)
        update_b = _make_update(None, chat_id=1, thread_id=20)

        task_a = asyncio.create_task(
            processor.do_process_update(update_a, coroutine_a())
        )
        task_b = asyncio.create_task(
            processor.do_process_update(update_b, coroutine_b())
        )

        await asyncio.gather(task_a, task_b)

        # b ran while a was still in progress — a_end comes last
        assert execution_log == ["a_start", "b_start", "b_end", "a_end"]

    async def test_two_updates_different_chats_run_concurrently(self):
        """Updates in different chats (DM vs group) run in parallel."""
        processor = StopAwareUpdateProcessor()

        execution_log: list[str] = []
        a_started = asyncio.Event()

        async def coroutine_a():
            execution_log.append("a_start")
            a_started.set()
            await asyncio.sleep(0.05)
            execution_log.append("a_end")

        async def coroutine_b():
            await a_started.wait()
            execution_log.append("b_start")
            execution_log.append("b_end")

        update_a = _make_update(None, chat_id=1)
        update_b = _make_update(None, chat_id=2)

        task_a = asyncio.create_task(
            processor.do_process_update(update_a, coroutine_a())
        )
        task_b = asyncio.create_task(
            processor.do_process_update(update_b, coroutine_b())
        )

        await asyncio.gather(task_a, task_b)

        assert execution_log == ["a_start", "b_start", "b_end", "a_end"]


class TestNonStopCallbackSharesTopicLock:
    async def test_cd_callback_same_topic_serializes(self):
        """Non-stop callbacks (cd:*) share the per-topic lock with messages."""
        processor = StopAwareUpdateProcessor()

        execution_log: list[str] = []

        async def regular_coroutine():
            execution_log.append("regular_start")
            await asyncio.sleep(0.05)
            execution_log.append("regular_end")

        async def cd_coroutine():
            execution_log.append("cd_start")
            execution_log.append("cd_end")

        regular_update = _make_update(None, chat_id=1, thread_id=10)
        cd_update = _make_update("cd:my_project", chat_id=1, thread_id=10)

        task_regular = asyncio.create_task(
            processor.do_process_update(regular_update, regular_coroutine())
        )
        await asyncio.sleep(0)
        task_cd = asyncio.create_task(
            processor.do_process_update(cd_update, cd_coroutine())
        )

        await asyncio.gather(task_regular, task_cd)

        assert execution_log == [
            "regular_start",
            "regular_end",
            "cd_start",
            "cd_end",
        ]


class TestFallbackKey:
    async def test_updates_without_chat_share_fallback_lock(self):
        """Updates that can't be keyed to a chat queue on a shared fallback lock."""
        processor = StopAwareUpdateProcessor()

        execution_log: list[str] = []

        async def coroutine_a():
            execution_log.append("a_start")
            await asyncio.sleep(0.05)
            execution_log.append("a_end")

        async def coroutine_b():
            execution_log.append("b_start")
            execution_log.append("b_end")

        update_a = _make_update(None, chat_id=None)
        update_b = _make_update(None, chat_id=None)

        task_a = asyncio.create_task(
            processor.do_process_update(update_a, coroutine_a())
        )
        await asyncio.sleep(0)
        task_b = asyncio.create_task(
            processor.do_process_update(update_b, coroutine_b())
        )

        await asyncio.gather(task_a, task_b)

        assert execution_log == ["a_start", "a_end", "b_start", "b_end"]


class TestInitializeShutdown:
    async def test_initialize_and_shutdown_are_noop(self):
        """initialize() and shutdown() should not raise."""
        processor = StopAwareUpdateProcessor()
        await processor.initialize()
        await processor.shutdown()
