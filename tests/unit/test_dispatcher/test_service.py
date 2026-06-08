"""Tests for the Hermes dispatcher service.

Hermetic: dispatcher logic is tested against an in-memory FakeBoard and fakes
for the runner + event bus — no Claude, no Telegram, no Hermes import path. The
board itself is covered by hermes/test_board.py in the c22os repo.

One guarded integration test exercises the REAL Hermes board via board_client
when the c22os checkout is present.

Run:  .venv/bin/python -m unittest tests.unit.test_dispatcher.test_service -v
"""

import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.dispatcher.service import (
    BLOCKED,
    CLAIMED,
    DONE,
    FAILED,
    PENDING,
    DispatcherService,
    RunOutcome,
)


# --------------------------------------------------------------------------- fakes
class FakeBoard:
    """Minimal in-memory board implementing the surface the dispatcher uses."""

    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self._seq = 0
        self.reclaim_calls = 0
        self.heartbeats: List[str] = []

    def add(self, *, title="t", assignee=None, chat_id=None, payload=None):
        self._seq += 1
        tid = f"task{self._seq}"
        self.tasks[tid] = {
            "id": tid, "title": title, "status": PENDING, "assignee": assignee,
            "chat_id": chat_id, "payload": payload, "result": None,
            "error": None, "block_reason": None,
        }
        return tid

    def reclaim_stale(self, *, stale_seconds=600):
        self.reclaim_calls += 1
        return []

    def claim(self, *, worker=None):
        for task in self.tasks.values():
            if task["status"] == PENDING:
                task["status"] = CLAIMED
                task["claimed_by"] = worker
                return dict(task)
        return None

    def heartbeat(self, task_id, *, now=None):
        task = self.tasks[task_id]
        if task["status"] != CLAIMED:
            raise RuntimeError("not claimed")
        self.heartbeats.append(task_id)
        return dict(task)

    def get(self, task_id):
        task = self.tasks.get(task_id)
        return dict(task) if task else None

    def complete(self, task_id, *, result=None, now=None):
        self.tasks[task_id].update(status=DONE, result=result)
        return dict(self.tasks[task_id])

    def block(self, task_id, reason, *, now=None):
        self.tasks[task_id].update(status=BLOCKED, block_reason=reason)
        return dict(self.tasks[task_id])

    def fail(self, task_id, error, *, now=None):
        self.tasks[task_id].update(status=FAILED, error=str(error))
        return dict(self.tasks[task_id])


class FakeRunner:
    """Runner with programmable behavior per call."""

    def __init__(self, behavior=None):
        # behavior(task) -> RunOutcome | raises | mutates board
        self._behavior = behavior or (lambda task: RunOutcome(content="done"))
        self.calls: List[str] = []

    async def run(self, task):
        self.calls.append(task["id"])
        result = self._behavior(task)
        if isinstance(result, Exception):
            raise result
        return result


class FakeEventBus:
    def __init__(self):
        self.published: List[Any] = []

    async def publish(self, event):
        self.published.append(event)


def make_dispatcher(board, runner, bus, **kw):
    defaults = dict(
        worker_id="test", tick_seconds=60, stale_seconds=600,
        max_per_tick=10, heartbeat_seconds=120,
    )
    defaults.update(kw)
    return DispatcherService(board=board, runner=runner, event_bus=bus, **defaults)


# --------------------------------------------------------------------------- tests
class DispatcherTest(unittest.IsolatedAsyncioTestCase):
    async def test_tick_claims_runs_and_completes(self):
        board = FakeBoard()
        board.add(title="job", chat_id="555")
        runner = FakeRunner(lambda t: RunOutcome(content="all done"))
        bus = FakeEventBus()
        d = make_dispatcher(board, runner, bus)

        ran = await d.tick()

        self.assertEqual(ran, 1)
        self.assertEqual(list(board.tasks.values())[0]["status"], DONE)
        self.assertEqual(len(bus.published), 1)
        self.assertEqual(bus.published[0].chat_id, 555)
        self.assertEqual(bus.published[0].text, "all done")

    async def test_reclaim_runs_every_tick(self):
        board = FakeBoard()
        d = make_dispatcher(board, FakeRunner(), FakeEventBus())
        await d.tick()
        await d.tick()
        self.assertEqual(board.reclaim_calls, 2)

    async def test_drains_up_to_max_per_tick(self):
        board = FakeBoard()
        for _ in range(5):
            board.add()
        runner = FakeRunner()
        d = make_dispatcher(board, runner, FakeEventBus(), max_per_tick=3)
        ran = await d.tick()
        self.assertEqual(ran, 3)
        self.assertEqual(len(runner.calls), 3)

    async def test_drains_until_empty(self):
        board = FakeBoard()
        for _ in range(2):
            board.add()
        d = make_dispatcher(board, FakeRunner(), FakeEventBus(), max_per_tick=10)
        ran = await d.tick()
        self.assertEqual(ran, 2)
        self.assertTrue(all(t["status"] == DONE for t in board.tasks.values()))

    async def test_outcome_blocked_blocks_task(self):
        board = FakeBoard()
        board.add(title="ambiguous", chat_id="42")
        runner = FakeRunner(
            lambda t: RunOutcome(blocked=True, block_reason="Which Chris?"))
        bus = FakeEventBus()
        d = make_dispatcher(board, runner, bus)

        await d.tick()

        task = list(board.tasks.values())[0]
        self.assertEqual(task["status"], BLOCKED)
        self.assertEqual(task["block_reason"], "Which Chris?")
        self.assertEqual(len(bus.published), 1)
        self.assertIn("Which Chris?", bus.published[0].text)
        self.assertEqual(bus.published[0].chat_id, 42)

    async def test_worker_self_block_is_respected(self):
        """If the worker blocked itself via the CLI mid-run, the dispatcher must
        NOT complete it — it reconciles against live board state."""
        board = FakeBoard()
        tid = board.add(title="self-block", chat_id="7")

        def behavior(task):
            # Simulate the worker running `cli.py block ...` during its run.
            board.block(tid, "need a decision")
            return RunOutcome(content="(worker stopped after blocking)")

        bus = FakeEventBus()
        d = make_dispatcher(board, FakeRunner(behavior), bus)

        await d.tick()

        self.assertEqual(board.tasks[tid]["status"], BLOCKED)
        self.assertEqual(board.tasks[tid]["block_reason"], "need a decision")
        # Notified with the board's reason, not the outcome content.
        self.assertIn("need a decision", bus.published[0].text)

    async def test_runner_exception_fails_task(self):
        board = FakeBoard()
        board.add(title="boom", chat_id="9")
        runner = FakeRunner(lambda t: RuntimeError("kaboom"))
        bus = FakeEventBus()
        d = make_dispatcher(board, runner, bus)

        await d.tick()

        task = list(board.tasks.values())[0]
        self.assertEqual(task["status"], FAILED)
        self.assertIn("kaboom", task["error"])
        self.assertEqual(len(bus.published), 1)
        self.assertIn("failed", bus.published[0].text.lower())

    async def test_done_without_chat_id_does_not_notify(self):
        board = FakeBoard()
        board.add(title="internal", chat_id=None)
        bus = FakeEventBus()
        d = make_dispatcher(board, FakeRunner(), bus)
        await d.tick()
        self.assertEqual(list(board.tasks.values())[0]["status"], DONE)
        self.assertEqual(bus.published, [])  # silent internal task

    async def test_blocked_without_chat_id_warns_no_notify(self):
        board = FakeBoard()
        board.add(title="orphan", chat_id=None)
        runner = FakeRunner(lambda t: RunOutcome(blocked=True, block_reason="?"))
        bus = FakeEventBus()
        d = make_dispatcher(board, runner, bus)
        await d.tick()
        self.assertEqual(list(board.tasks.values())[0]["status"], BLOCKED)
        self.assertEqual(bus.published, [])  # can't ask a human with no chat

    async def test_thread_id_from_payload(self):
        board = FakeBoard()
        board.add(title="topic-job", chat_id="100", payload={"_thread_id": 88})
        bus = FakeEventBus()
        d = make_dispatcher(board, FakeRunner(), bus)
        await d.tick()
        self.assertEqual(bus.published[0].message_thread_id, 88)

    async def test_heartbeat_fires_during_long_run(self):
        import asyncio

        board = FakeBoard()
        tid = board.add(title="slow")

        async def slow(task):
            await asyncio.sleep(0.06)
            return RunOutcome(content="ok")

        runner = type("R", (), {"run": staticmethod(slow), "calls": []})()
        d = make_dispatcher(board, runner, FakeEventBus(), heartbeat_seconds=0.02)
        await d.tick()
        self.assertGreaterEqual(len(board.heartbeats), 1)
        self.assertEqual(board.tasks[tid]["status"], DONE)

    async def test_start_stop_runs_loop(self):
        import asyncio

        board = FakeBoard()
        board.add(title="looped")
        d = make_dispatcher(board, FakeRunner(), FakeEventBus(), tick_seconds=600)
        await d.start()
        await asyncio.sleep(0.05)  # let the first tick run
        await d.stop()
        self.assertEqual(list(board.tasks.values())[0]["status"], DONE)


# ----------------------------------------------------- integration (real board)
C22OS_HERMES = Path("/home/c22bot/Projects/c22os/hermes")


@unittest.skipUnless(
    (C22OS_HERMES / "board.py").is_file(),
    "real Hermes board not present at the default c22os path",
)
class RealBoardIntegrationTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from src.dispatcher.board_client import load_task_board

        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.board = load_task_board(C22OS_HERMES, self.db_path)

    def tearDown(self):
        self.board.close()
        for suffix in ("", "-wal", "-shm"):
            p = self.db_path + suffix
            if os.path.exists(p):
                os.unlink(p)

    async def test_full_lifecycle_against_real_board(self):
        self.board.create("real task", chat_id="321", payload={"k": "v"})
        runner = FakeRunner(lambda t: RunOutcome(content="processed"))
        bus = FakeEventBus()
        d = make_dispatcher(self.board, runner, bus)

        ran = await d.tick()

        self.assertEqual(ran, 1)
        stats = self.board.stats()
        self.assertEqual(stats["done"], 1)
        self.assertEqual(bus.published[0].text, "processed")
        self.assertEqual(bus.published[0].chat_id, 321)

    async def test_real_board_block_then_resume(self):
        tid = self.board.create("needs-human", chat_id="1")["id"]

        def behavior(task):
            self.board.block(tid, "confirm X?")
            return RunOutcome()

        d = make_dispatcher(self.board, FakeRunner(behavior), FakeEventBus())
        await d.tick()
        self.assertEqual(self.board.get(tid)["status"], BLOCKED)

        # Human answers -> resume re-queues -> next tick completes it.
        self.board.resume(tid, note="X is confirmed")
        d2 = make_dispatcher(
            self.board, FakeRunner(lambda t: RunOutcome(content="finished")),
            FakeEventBus())
        await d2.tick()
        self.assertEqual(self.board.get(tid)["status"], DONE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
