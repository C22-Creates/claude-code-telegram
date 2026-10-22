"""Tests for the SDK lifecycle helpers ported from src/claude/sdk_integration.py.

The incident tests (await-cancelled, reap) are copies of the originals,
retargeted at module-level functions; the originals stay until src/claude/ is
deleted.
"""

import asyncio
import os
import subprocess
import time
import types

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    CLIConnectionError,
    TextBlock,
    ToolUseBlock,
)
from claude_agent_sdk.types import StreamEvent

from src.harness.sdk_lifecycle import (
    CANCELLED_TASK_CLEANUP_TIMEOUT,
    MAX_REAPED_DESCENDANTS,
    await_cancelled,
    descendant_pids,
    is_retryable_error,
    last_tool_name,
    load_mcp_config,
    reap_wedged_child,
    turn_signal,
)


def _text(text="hi"):
    return AssistantMessage(
        content=[TextBlock(text=text)], model="claude-sonnet-4-20250514"
    )


def _event(event, parent=None):
    return StreamEvent(uuid="u", session_id="s", event=event, parent_tool_use_id=parent)


def _delta(stop_reason, parent=None):
    return _event(
        {"type": "message_delta", "delta": {"stop_reason": stop_reason}}, parent
    )


def test_constants_keep_incident_values():
    assert CANCELLED_TASK_CLEANUP_TIMEOUT == 10.0
    assert MAX_REAPED_DESCENDANTS == 64


class TestTurnSignal:
    """The live stream never sets stop_reason; the turn end is message_delta."""

    def test_end_turn_delta_means_turn_ended(self):
        assert turn_signal(_delta("end_turn")) is True

    def test_tool_use_delta_means_turn_continues(self):
        assert turn_signal(_delta("tool_use")) is False

    def test_message_start_means_turn_continues(self):
        assert turn_signal(_event({"type": "message_start"})) is False

    def test_other_stream_events_say_nothing(self):
        assert turn_signal(_event({"type": "message_stop"})) is None

    def test_subagent_end_turn_is_ignored(self):
        assert turn_signal(_delta("end_turn", parent="toolu_task")) is None

    def test_delta_without_delta_payload_is_not_end_turn(self):
        assert turn_signal(_event({"type": "message_delta", "delta": None})) is False

    def test_assistant_message_with_tool_use_continues(self):
        msg = AssistantMessage(
            content=[ToolUseBlock(id="t", name="Read", input={})],
            model="claude-sonnet-4-20250514",
        )
        assert turn_signal(msg) is False

    def test_text_only_assistant_message_says_nothing(self):
        assert turn_signal(_text()) is None

    def test_unrelated_object_says_nothing(self):
        assert turn_signal(object()) is None


class TestIsRetryableError:
    def test_transient_connection_error(self):
        assert is_retryable_error(CLIConnectionError("connection reset")) is True

    def test_mcp_connection_error(self):
        assert is_retryable_error(CLIConnectionError("mcp server failed")) is False

    def test_timeout_not_retried(self):
        assert is_retryable_error(asyncio.TimeoutError()) is False

    def test_control_request_timeout_retried(self):
        exc = Exception("Control request timeout: initialize")
        assert is_retryable_error(exc) is True

    def test_other_bare_exception_not_retried(self):
        assert is_retryable_error(Exception("boom")) is False

    def test_subclass_with_control_message_not_retried(self):
        assert is_retryable_error(ValueError("control request timeout")) is False


class TestAwaitCancelledBounded:
    """Cancelling an SDK task must never hang the caller.

    Regression guard for the 2026-07-29 incident: a wedged child process
    swallowed cancellation, the bare `await run_task` in the cleanup path
    never returned, and the orchestrator's `finally: heartbeat.cancel()`
    therefore never ran — leaving the Telegram typing indicator on forever
    and the child orphaned for 15.5 hours.
    """

    @staticmethod
    async def _wedged_task(stop: asyncio.Event) -> None:
        """A task that swallows cancellation, like a child stuck on a pipe read."""
        while not stop.is_set():
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                pass  # refuse to die — this is the failure mode under test

    async def test_returns_false_and_does_not_hang_when_task_ignores_cancel(self):
        stop = asyncio.Event()
        task = asyncio.create_task(self._wedged_task(stop))
        await asyncio.sleep(0)  # let it start

        task.cancel()
        loop = asyncio.get_running_loop()
        started = loop.time()
        settled = await await_cancelled(task, reason="test", timeout=0.05)
        elapsed = loop.time() - started

        assert settled is False, "a wedged task must be reported as unsettled"
        # The point of the fix: control returns on a bound, not never.
        assert elapsed < 2.0, f"cleanup await was not bounded (took {elapsed:.2f}s)"

        # Let the fixture task exit so it does not leak into other tests.
        stop.set()
        await asyncio.wait({task}, timeout=1.0)

    async def test_returns_true_when_task_honors_cancellation(self):
        async def _cooperative() -> None:
            await asyncio.sleep(60)

        task = asyncio.create_task(_cooperative())
        await asyncio.sleep(0)
        task.cancel()

        settled = await await_cancelled(task, reason="test", timeout=1.0)

        assert settled is True
        assert task.cancelled()


class TestLastToolName:
    def test_returns_most_recent_tool_use(self):
        older = AssistantMessage(
            content=[ToolUseBlock(id="t1", name="Read", input={})],
            model="claude-sonnet-4-20250514",
        )
        newer = AssistantMessage(
            content=[
                ToolUseBlock(id="t2", name="mcp__notion__notion-update-page", input={})
            ],
            model="claude-sonnet-4-20250514",
        )
        assert last_tool_name([older, newer]) == "mcp__notion__notion-update-page"

    def test_none_when_no_tool_use(self):
        assert last_tool_name([_text("just talking")]) is None

    def test_empty_messages(self):
        assert last_tool_name([]) is None


class TestLoadMcpConfig:
    def test_returns_mcp_servers_dict(self, tmp_path):
        path = tmp_path / "mcp.json"
        path.write_text('{"mcpServers": {"s": {"command": "echo", "args": ["hi"]}}}')
        assert load_mcp_config(path) == {"s": {"command": "echo", "args": ["hi"]}}

    def test_missing_key_gives_empty_dict(self, tmp_path):
        path = tmp_path / "mcp.json"
        path.write_text("{}")
        assert load_mcp_config(path) == {}

    def test_invalid_json_gives_empty_dict(self, tmp_path):
        path = tmp_path / "mcp.json"
        path.write_text("{not json")
        assert load_mcp_config(path) == {}

    def test_missing_file_gives_empty_dict(self, tmp_path):
        assert load_mcp_config(tmp_path / "nope.json") == {}


def _process_state(pid: int) -> str:
    """Return the Linux process state letter, or "gone" if the pid is absent.

    "Z" (zombie) counts as dead for reaping purposes — the process no longer
    runs, it is merely waiting to be reaped by its parent.
    """
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("State:"):
                    return line.split()[1]
    except OSError:
        return "gone"
    return "gone"


def _is_running(pid: int) -> bool:
    return _process_state(pid) not in ("gone", "Z", "X")


def _wait_until_dead(pid: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while _is_running(pid) and time.monotonic() < deadline:
        time.sleep(0.02)


@pytest.mark.skipif(
    not os.path.isdir("/proc"), reason="requires Linux /proc for pid discovery"
)
class TestReapWedgedChild:
    """A wedged SDK child and its MCP subprocesses must not outlive the request.

    Regression guard for the 2026-08-21 recurrence: after cancellation failed
    to settle, the `claude` child survived 2h28m holding 660MB plus its own
    node/bun MCP subprocesses, because the transport's cooperative shutdown
    was itself blocked.
    """

    @staticmethod
    def _spawn_tree():
        """Spawn `sh` owning two `sleep` children; returns the Popen."""
        # Trailing `true` prevents the shell from exec'ing into the last sleep,
        # so `sh` survives as the parent of both children.
        return subprocess.Popen(["sh", "-c", "sleep 300 & sleep 300; true"])

    @staticmethod
    def _fake_client(pid: int, returncode=None):
        process = types.SimpleNamespace(pid=pid, returncode=returncode)
        transport = types.SimpleNamespace(_process=process)
        return types.SimpleNamespace(_transport=transport)

    def test_kills_child_and_its_descendants(self):
        proc = self._spawn_tree()
        try:
            # Give the shell a moment to fork both sleeps.
            deadline = time.monotonic() + 5.0
            descendants = []
            while time.monotonic() < deadline:
                descendants = descendant_pids(proc.pid)
                if len(descendants) >= 2:
                    break
                time.sleep(0.05)

            assert (
                len(descendants) >= 2
            ), f"expected to discover 2 sleep children, found {descendants}"

            reap_wedged_child(self._fake_client(proc.pid), reason="test")

            assert proc.wait(timeout=5) != 0, "child should have been killed"
            # SIGKILL delivery is asynchronous: a descendant can still read as
            # running for a few ms after the parent has been waited on. Poll
            # with a deadline instead of asserting on a single instant.
            for pid in descendants:
                _wait_until_dead(pid)
                assert not _is_running(pid), f"descendant {pid} survived the reap"
        finally:
            proc.kill()
            proc.wait(timeout=5)

    def test_skips_process_that_already_exited(self):
        proc = self._spawn_tree()
        proc.kill()
        proc.wait(timeout=5)

        # returncode set => transport already shut down cleanly; nothing to do.
        reap_wedged_child(self._fake_client(proc.pid, returncode=0), reason="test")

    def test_refuses_to_kill_own_process(self):
        # The guard matters: SIGKILL on our own pid would take down the bot.
        reap_wedged_child(self._fake_client(os.getpid()), reason="test")
        assert _is_running(os.getpid())

    @pytest.mark.parametrize("client", [None, object(), "not-a-client"])
    def test_never_raises_on_unexpected_client(self, client):
        # Runs on an already-failing path; it must not mask the original error.
        reap_wedged_child(client, reason="test")

    def test_descendant_pids_of_leaf_process_is_empty(self):
        proc = subprocess.Popen(["sleep", "300"])
        try:
            assert descendant_pids(proc.pid) == []
        finally:
            proc.kill()
            proc.wait(timeout=5)
