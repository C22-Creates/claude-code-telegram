"""Tests for the Telegram reply -> board.resume bridge.

Hermetic (fake board + fake PTB message) plus one guarded integration test
against the real Hermes board.

Run:  .venv/bin/python -m unittest tests.unit.test_dispatcher.test_resume_bridge -v
"""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.dispatcher.resume_bridge import ResumeBridge


# --------------------------------------------------------------------------- fakes
class FakeBoard:
    def __init__(self):
        self.tasks = []
        self.resumed = []

    def add_blocked(self, task_id, chat_id, thread_id=None, title="job"):
        payload = {"_thread_id": thread_id} if thread_id is not None else None
        self.tasks.append({
            "id": task_id, "title": title, "status": "blocked",
            "chat_id": str(chat_id), "payload": payload,
        })

    def list_tasks(self, *, status=None, assignee=None, limit=None):
        return [dict(t) for t in self.tasks if status is None or t["status"] == status]

    def resume(self, task_id, *, note=None):
        self.resumed.append((task_id, note))
        for t in self.tasks:
            if t["id"] == task_id:
                t["status"] = "pending"
        return {"id": task_id, "status": "pending"}


class RaisingBoard(FakeBoard):
    def resume(self, task_id, *, note=None):
        raise RuntimeError("db locked")


class FakeMessage:
    def __init__(self, *, text, chat_id, thread_id=None, quoted=None):
        self.text = text
        self.chat_id = chat_id
        self.message_thread_id = thread_id
        self.reply_to_message = (
            SimpleNamespace(text=quoted) if quoted is not None else None
        )
        self.replies = []

    async def reply_text(self, text):
        self.replies.append(text)


def update_with(message):
    return SimpleNamespace(message=message)


# --------------------------------------------------------------------------- tests
class ResumeBridgeTest(unittest.IsolatedAsyncioTestCase):
    async def test_single_blocked_task_resumes(self):
        board = FakeBoard()
        board.add_blocked("abc12345def", chat_id=555)
        bridge = ResumeBridge(board)

        resumed = await bridge.try_resume(
            update_with(FakeMessage(text="It's Jane Doe", chat_id=555)))

        self.assertTrue(resumed)
        self.assertEqual(board.resumed, [("abc12345def", "It's Jane Doe")])

    async def test_reply_to_marker_selects_specific_task(self):
        board = FakeBoard()
        board.add_blocked("aaaa1111zzzz", chat_id=555, title="A")
        board.add_blocked("bbbb2222yyyy", chat_id=555, title="B")
        bridge = ResumeBridge(board)

        # User replied to B's notification, which contained "(task bbbb2222)".
        msg = FakeMessage(
            text="use the second one", chat_id=555,
            quoted="🔔 Paused — I need your input:\n\nB\n\n(task bbbb2222)")
        resumed = await bridge.try_resume(update_with(msg))

        self.assertTrue(resumed)
        self.assertEqual(board.resumed, [("bbbb2222yyyy", "use the second one")])

    async def test_marker_in_reply_text_selects_task(self):
        board = FakeBoard()
        board.add_blocked("aaaa1111", chat_id=9)
        board.add_blocked("bbbb2222", chat_id=9)
        bridge = ResumeBridge(board)

        msg = FakeMessage(text="task aaaa1111 yes go ahead", chat_id=9)
        resumed = await bridge.try_resume(update_with(msg))

        self.assertTrue(resumed)
        self.assertEqual(board.resumed[0][0], "aaaa1111")

    async def test_ambiguous_without_marker_does_not_resume(self):
        board = FakeBoard()
        board.add_blocked("aaaa1111", chat_id=9)
        board.add_blocked("bbbb2222", chat_id=9)
        bridge = ResumeBridge(board)

        resumed = await bridge.try_resume(
            update_with(FakeMessage(text="yes", chat_id=9)))

        self.assertFalse(resumed)
        self.assertEqual(board.resumed, [])  # never guesses

    async def test_no_blocked_tasks(self):
        bridge = ResumeBridge(FakeBoard())
        resumed = await bridge.try_resume(
            update_with(FakeMessage(text="hello", chat_id=1)))
        self.assertFalse(resumed)

    async def test_chat_mismatch(self):
        board = FakeBoard()
        board.add_blocked("abc123", chat_id=555)
        bridge = ResumeBridge(board)
        resumed = await bridge.try_resume(
            update_with(FakeMessage(text="answer", chat_id=999)))
        self.assertFalse(resumed)
        self.assertEqual(board.resumed, [])

    async def test_thread_must_match(self):
        board = FakeBoard()
        board.add_blocked("abc123", chat_id=555, thread_id=88)
        bridge = ResumeBridge(board)
        # Same chat, wrong topic -> no resume.
        resumed = await bridge.try_resume(
            update_with(FakeMessage(text="answer", chat_id=555, thread_id=99)))
        self.assertFalse(resumed)

    async def test_thread_match_resumes(self):
        board = FakeBoard()
        board.add_blocked("abc123", chat_id=555, thread_id=88)
        bridge = ResumeBridge(board)
        resumed = await bridge.try_resume(
            update_with(FakeMessage(text="answer", chat_id=555, thread_id=88)))
        self.assertTrue(resumed)
        self.assertEqual(board.resumed[0][0], "abc123")

    async def test_non_text_message_ignored(self):
        board = FakeBoard()
        board.add_blocked("abc123", chat_id=555)
        bridge = ResumeBridge(board)
        resumed = await bridge.try_resume(
            update_with(FakeMessage(text=None, chat_id=555)))
        self.assertFalse(resumed)

    async def test_board_error_is_swallowed(self):
        board = RaisingBoard()
        board.add_blocked("abc123", chat_id=555)
        bridge = ResumeBridge(board)
        # Must not raise into the handler.
        resumed = await bridge.try_resume(
            update_with(FakeMessage(text="answer", chat_id=555)))
        self.assertFalse(resumed)

    async def test_confirmation_reply_sent(self):
        board = FakeBoard()
        board.add_blocked("abc123", chat_id=555, title="Process meeting X")
        bridge = ResumeBridge(board)
        msg = FakeMessage(text="answer", chat_id=555)
        await bridge.try_resume(update_with(msg))
        self.assertEqual(len(msg.replies), 1)
        self.assertIn("Process meeting X", msg.replies[0])

    async def test_marker_no_match_falls_back_to_single(self):
        board = FakeBoard()
        board.add_blocked("realid99", chat_id=7)
        bridge = ResumeBridge(board)
        # Marker points at a non-existent task, but there's only one candidate.
        msg = FakeMessage(text="task deadbeef whatever", chat_id=7)
        resumed = await bridge.try_resume(update_with(msg))
        self.assertTrue(resumed)
        self.assertEqual(board.resumed[0][0], "realid99")


# ----------------------------------------------------- integration (real board)
C22OS_HERMES = Path("/home/c22bot/Projects/c22os/hermes")


@unittest.skipUnless(
    (C22OS_HERMES / "board.py").is_file(),
    "real Hermes board not present at the default c22os path",
)
class RealBoardResumeTest(unittest.IsolatedAsyncioTestCase):
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

    async def test_reply_resumes_real_blocked_task(self):
        task = self.board.create("real recap", chat_id="555")
        self.board.claim(worker="d")
        self.board.block(task["id"], "Which Chris?")

        bridge = ResumeBridge(self.board)
        msg = FakeMessage(text="It's Chris Mortenson", chat_id=555)
        resumed = await bridge.try_resume(update_with(msg))

        self.assertTrue(resumed)
        refreshed = self.board.get(task["id"])
        self.assertEqual(refreshed["status"], "pending")
        notes = refreshed["payload"]["_resume_notes"]
        self.assertEqual(notes[0]["note"], "It's Chris Mortenson")


if __name__ == "__main__":
    unittest.main(verbosity=2)
