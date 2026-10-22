"""Tests for AgentRun: completion, watchdog, cap, injection, bounded teardown.

Time is fake: AgentRun's supervisor ``sleep`` advances a ``FakeClock`` and the
fake SDK client's step delays wait on that same clock, so a 60-minute run
finishes in milliseconds. The bounded real-time waits (interrupt, cleanup) are
shrunk via constructor arguments.
"""

import asyncio
import time

import pytest
from claude_agent_sdk import ClaudeAgentOptions, CLIConnectionError

from src.config.settings import Settings
from src.harness import agent_run as agent_run_module
from src.harness.agent_run import AgentRun
from src.harness.types import Outcome, ProgressEvent
from tests.harness import fakes
from tests.harness.fakes import FakeSDKClient


class FakeClock:
    """Monotonic clock that only moves when AgentRun's supervisor sleeps."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def tick(self, seconds: float) -> None:
        """AgentRun's ``sleep``: drives time forward."""
        self.now += seconds
        await asyncio.sleep(0)

    async def wait(self, seconds: float) -> None:
        """The fake client's ``sleep``: follows time, never drives it."""
        target = self.now + seconds
        while self.now < target:
            await asyncio.sleep(0)


@pytest.fixture
def settings(tmp_path):
    return Settings(
        telegram_bot_token="t",
        telegram_bot_username="b",
        approved_directory=str(tmp_path),
    )


class Harness:
    """One AgentRun wired to a scripted fake client and a fake clock."""

    def __init__(self, script, settings, on_progress=None, **client_kw):
        self.clock = FakeClock()
        self.factory = FakeSDKClient.factory(script, sleep=self.clock.wait, **client_kw)
        self.client = self.factory.client
        self.events: list[ProgressEvent] = []

        async def record(event: ProgressEvent) -> None:
            self.events.append(event)
            if on_progress is not None:
                await on_progress(event)

        self.run = AgentRun(
            ClaudeAgentOptions(),
            record,
            settings,
            self.factory,
            clock=self.clock,
            sleep=self.clock.tick,
            interrupt_timeout=0.05,
            cleanup_timeout=0.05,
        )

    def kinds(self, kind: str) -> list[ProgressEvent]:
        return [e for e in self.events if e.kind == kind]


def every_4m(n: int, label: str = "t") -> list:
    """``n`` text messages, 4 minutes apart (keeps the stall watchdog quiet)."""
    return [(240, fakes.text(f"{label}{i}")) for i in range(n)]


async def test_happy_path_done(settings):
    h = Harness(
        [
            (0, fakes.system_init(["missive"])),
            (0, fakes.message_start()),
            (0, fakes.tool_use("mcp__notion__notion-search")),
            (0, fakes.message_start()),
            (0, fakes.text("Here is the answer.")),
            (0, fakes.result(session_id="sess-9")),
        ],
        settings,
    )
    res = await h.run.run("hello")
    assert res.outcome is Outcome.DONE
    assert res.text == "Here is the answer."
    assert res.tool_calls == 1
    assert res.reason == ""
    assert res.session_id == "sess-9" == h.run.session_id
    assert [e.tool for e in h.kinds("tool")] == ["mcp__notion__notion-search"]
    assert [e.turn for e in h.kinds("turn")] == [1, 2]
    assert h.client.queries == ["hello"]
    assert h.client.disconnected
    assert not h.run.is_active


async def test_stall_interrupts_after_5m_keeps_session(settings):
    h = Harness(
        [
            (0, fakes.message_start()),
            (0, fakes.text("Looking now.")),
            (0, fakes.tool_use("mcp__missive__list_conversations")),
        ],
        settings,
    )
    res = await h.run.run("inbox?")
    assert res.outcome is Outcome.STOPPED
    assert res.reason == "stalled on mcp__missive__list_conversations"
    assert h.client.interrupted == 1
    assert res.session_id == "sess-1"  # from the post-interrupt ResultMessage
    assert res.text == "Looking now."
    assert 300 <= res.elapsed_s <= 302


async def test_heartbeats_at_20_and_40m(settings):
    h = Harness(
        [(0, fakes.message_start()), *every_4m(11), (0, fakes.result())], settings
    )
    res = await h.run.run("long job")
    assert res.outcome is Outcome.DONE
    beats = h.kinds("heartbeat")
    assert [e.detail for e in beats] == ["20m", "40m"]
    assert [round(e.elapsed_s) for e in beats] == [1200, 2400]


async def test_cap_at_60m_returns_partial_text(settings):
    h = Harness([(0, fakes.message_start()), *every_4m(30)], settings)
    res = await h.run.run("forever")
    assert res.outcome is Outcome.STOPPED
    assert res.reason == "hit 60m cap"
    assert h.client.interrupted == 1
    assert 3600 <= res.elapsed_s <= 3602
    # 15 texts landed before the cap (at 4m, 8m, ... 60m minus the race).
    assert res.text.startswith("t0\n\nt1\n\n")
    assert "t13" in res.text


async def test_end_turn_grace_completes_without_result(settings):
    h = Harness(
        [
            (0, fakes.message_start()),
            (0, fakes.text("All done.")),
            (0, fakes.message_delta_end_turn()),
        ],
        settings,
    )
    res = await h.run.run("hi")
    assert res.outcome is Outcome.DONE
    assert res.text == "All done."
    assert 15 <= res.elapsed_s <= 17
    assert h.client.interrupted == 0


async def test_subagent_end_turn_ignored(settings):
    h = Harness(
        [
            (0, fakes.message_start()),
            (0, fakes.tool_use("Task")),
            (0, fakes.message_delta_end_turn(parent_tool_use_id="tu-Task")),
            (40, fakes.message_start()),
            (0, fakes.text("Subagent finished; summary.")),
            (0, fakes.result()),
        ],
        settings,
    )
    res = await h.run.run("delegate")
    assert res.outcome is Outcome.DONE
    assert res.elapsed_s >= 40  # no grace completion at ~15s
    assert res.text == "Subagent finished; summary."


async def test_inject_mid_turn_merged_single_result(settings):
    holder: dict = {}
    h = Harness(
        [
            (0, fakes.message_start()),
            (0, fakes.tool_use("Bash")),
            ("call", lambda: holder["run"].inject("Also: what is 2+2?")),
            (5, fakes.message_start()),
            (0, fakes.text("Counted to 30. 2 + 2 = 4.")),
            (0, fakes.message_delta_end_turn()),
            (0, fakes.result()),
        ],
        settings,
    )
    holder["run"] = h.run
    res = await h.run.run("count to 30")
    assert res.outcome is Outcome.DONE
    assert h.client.queries == ["count to 30", "Also: what is 2+2?"]
    assert res.text == "Counted to 30. 2 + 2 = 4."


async def test_inject_after_end_turn_waits_for_second_result(settings):
    holder: dict = {}
    h = Harness(
        [
            (0, fakes.message_start()),
            (0, fakes.text("First answer.")),
            (0, fakes.message_delta_end_turn()),
            ("call", lambda: holder["run"].inject("one more thing")),
            # Longer than the end-turn grace: grace must not complete the run
            # while the injection is unconsumed.
            (30, fakes.result(text="First answer.")),
            (0, fakes.message_start()),
            (0, fakes.text("Second answer.")),
            (0, fakes.message_delta_end_turn()),
            (0, fakes.result(session_id="sess-2", text="Second answer.")),
        ],
        settings,
    )
    holder["run"] = h.run
    res = await h.run.run("question")
    assert res.outcome is Outcome.DONE
    assert h.client.queries == ["question", "one more thing"]
    assert res.text == "First answer.\n\nSecond answer."
    assert res.session_id == "sess-2"
    assert res.elapsed_s >= 30


async def test_unconsumed_injection_still_bounded_by_stall(settings):
    holder: dict = {}
    h = Harness(
        [
            (0, fakes.message_start()),
            (0, fakes.text("Done.")),
            ("call", lambda: holder["run"].inject("and?")),
            (0, fakes.result()),
        ],
        settings,
    )
    holder["run"] = h.run
    res = await h.run.run("q")
    assert res.outcome is Outcome.STOPPED
    assert res.reason == "stalled on unknown"


async def test_interrupt_hang_reaps_child(settings, monkeypatch):
    reaped: list = []
    monkeypatch.setattr(
        agent_run_module,
        "reap_wedged_child",
        lambda client, reason: reaped.append((client, reason)),
    )
    seen_tool = asyncio.Event()

    async def on_progress(event: ProgressEvent) -> None:
        if event.kind == "tool":
            seen_tool.set()

    h = Harness(
        [(0, fakes.message_start()), (0, fakes.tool_use("Bash"))],
        settings,
        on_progress=on_progress,
        hang_interrupt=True,
    )
    # Real-time supervisor: the fake clock must not race to a stall here.
    h.run._sleep = asyncio.sleep
    task = asyncio.create_task(h.run.run("run forever"))
    await asyncio.wait_for(seen_tool.wait(), 1)
    started = time.monotonic()
    await h.run.interrupt("cancelled")
    res = await asyncio.wait_for(task, 1)
    assert time.monotonic() - started < 0.5  # 0.05 interrupt + 0.05 cleanup
    assert res.outcome is Outcome.STOPPED
    assert res.reason == "cancelled"
    assert reaped == [(h.client, "cancelled")]
    assert h.client.disconnected


async def test_external_interrupt_is_stopped_not_error(settings):
    holder: dict = {}
    h = Harness(
        [
            (0, fakes.message_start()),
            (0, fakes.text("Partial.")),
            ("call", lambda: holder["run"].interrupt("cancelled")),
            (10, fakes.text("never")),
        ],
        settings,
    )
    holder["run"] = h.run
    res = await h.run.run("go")
    assert res.outcome is Outcome.STOPPED
    assert res.reason == "cancelled"
    assert res.text == "Partial."
    assert h.client.interrupted == 1


async def test_cli_crash_is_error_one_line(settings):
    h = Harness(
        [
            (0, fakes.message_start()),
            (0, fakes.text("Working.")),
            ("raise", RuntimeError("Command failed with exit code 1\nTraceback...")),
        ],
        settings,
    )
    res = await h.run.run("x")
    assert res.outcome is Outcome.ERROR
    assert res.reason == "Command failed with exit code 1"
    assert res.text == "Working."
    assert h.client.queries == ["x"]  # an error after the first event: no retry
    assert h.client.disconnected


async def test_retryable_error_before_first_event_retried_once(settings):
    h = Harness([], settings, raise_on_receive=CLIConnectionError("pipe closed"))
    res = await h.run.run("x")
    assert res.outcome is Outcome.ERROR
    assert res.reason == "pipe closed"
    assert h.client.queries == ["x", "x"]


async def test_non_retryable_error_not_retried(settings):
    h = Harness([], settings, raise_on_receive=ValueError("bad config"))
    res = await h.run.run("x")
    assert res.outcome is Outcome.ERROR
    assert res.reason == "bad config"
    assert h.client.queries == ["x"]


async def test_error_result_maps_to_error_or_max_turns(settings):
    h = Harness(
        [(0, fakes.result(is_error=True, text="API Error: 529 overloaded\nmore"))],
        settings,
    )
    res = await h.run.run("x")
    assert (res.outcome, res.reason) == (Outcome.ERROR, "API Error: 529 overloaded")

    h = Harness(
        [(0, fakes.result(is_error=True, text=None, subtype="error_max_turns"))],
        settings,
    )
    res = await h.run.run("x")
    assert (res.outcome, res.reason) == (Outcome.STOPPED, "hit max turns")

    h = Harness([(0, fakes.result(is_error=True, text=None))], settings)
    res = await h.run.run("x")
    assert (res.outcome, res.reason) == (Outcome.ERROR, "error_during_execution")


async def test_mcp_failed_emits_unavailable_and_continues(settings):
    h = Harness(
        [
            (
                0,
                fakes.system_init([{"name": "notion", "status": "failed"}, "missive"]),
            ),
            (0, fakes.message_start()),
            (0, fakes.text("Answered without Notion.")),
            (0, fakes.result()),
        ],
        settings,
    )
    res = await h.run.run("x")
    assert res.outcome is Outcome.DONE
    assert [e.detail for e in h.kinds("mcp_unavailable")] == ["notion"]


async def test_progress_callback_errors_are_swallowed(settings):
    async def boom(event: ProgressEvent) -> None:
        raise RuntimeError("telegram hiccup")

    h = Harness(
        [(0, fakes.message_start()), (0, fakes.text("ok")), (0, fakes.result())],
        settings,
        on_progress=boom,
    )
    res = await h.run.run("x")
    assert res.outcome is Outcome.DONE
    assert res.text == "ok"


async def test_watchdog_paused_while_waiting_then_restarts(settings):
    holder: dict = {}

    async def ask() -> None:
        holder["run"].pause_watchdog()
        await holder["clock"].wait(720)  # a 12-minute pending question
        holder["run"].resume_watchdog()

    h = Harness(
        [
            (0, fakes.message_start()),
            (0, fakes.tool_use("AskUserQuestion")),
            ("call", ask),
        ],
        settings,
    )
    holder.update(run=h.run, clock=h.clock)
    res = await h.run.run("x")
    assert res.outcome is Outcome.STOPPED
    assert res.reason == "stalled on AskUserQuestion"
    # Stall clock restarted from the answer at 12m, not from the tool call.
    assert 1020 <= res.elapsed_s <= 1023


async def test_multimodal_prompt_and_inject_send_user_message(settings):
    blocks = [
        {"type": "image", "source": {"type": "base64", "media_type": "x", "data": ""}},
        {"type": "text", "text": "what is this?"},
    ]
    h = Harness([(0, fakes.message_start()), (0, fakes.result())], settings)
    res = await h.run.run(blocks)
    assert res.outcome is Outcome.DONE
    sent = [m async for m in h.client.queries[0]]
    assert sent == [{"type": "user", "message": {"role": "user", "content": blocks}}]


async def test_inject_when_idle_raises(settings):
    h = Harness([], settings)
    with pytest.raises(RuntimeError):
        await h.run.inject("hello")


async def test_caller_cancel_tears_down_and_propagates(settings):
    seen = asyncio.Event()

    async def on_progress(event: ProgressEvent) -> None:
        seen.set()

    h = Harness(
        [(0, fakes.message_start()), (0, fakes.text("hi"))],
        settings,
        on_progress=on_progress,
    )
    h.run._sleep = asyncio.sleep
    task = asyncio.create_task(h.run.run("x"))
    await asyncio.wait_for(seen.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert h.client.disconnected
    assert not h.run.is_active
