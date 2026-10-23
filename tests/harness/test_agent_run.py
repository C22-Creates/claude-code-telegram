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

    def __init__(
        self, script, settings, on_progress=None, client_cls=FakeSDKClient, **client_kw
    ):
        self.clock = FakeClock()
        self.factory = client_cls.factory(script, sleep=self.clock.wait, **client_kw)
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
            progress_timeout=0.05,
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
            (0, fakes.result(session_id="sess-9", text="Here is the answer.")),
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
    assert res.reason == "stalled on mcp__missive__list_conversations for 5m"
    assert h.client.interrupted == 1
    assert res.session_id == "sess-1"  # from the post-interrupt ResultMessage
    assert res.text == "Looking now."
    assert 300 <= res.elapsed_s <= 310  # fake time drifts a few ticks


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
    assert 3600 <= res.elapsed_s <= 3610
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
    assert 15 <= res.elapsed_s <= 25  # not 300 (stall), not 0
    assert h.client.interrupted == 0


async def test_subagent_end_turn_ignored(settings):
    h = Harness(
        [
            (0, fakes.message_start()),
            (0, fakes.tool_use("Task")),
            (0, fakes.message_delta_end_turn(parent_tool_use_id="tu-Task")),
            (40, fakes.message_start()),
            (0, fakes.text("Subagent finished; summary.")),
            (0, fakes.result(text=None)),
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
            (0, fakes.result(text=None)),
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
    assert res.reason == "stalled on unknown for 5m"


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
    assert res.reason == "stalled on AskUserQuestion for 5m"
    # Stall clock restarted from the answer at 12m, not from the tool call.
    assert 1020 <= res.elapsed_s <= 1035


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


# ------------------------------------------------------- review fix round 1


class SlowInjectClient(FakeSDKClient):
    """``query()`` for injections yields a few times before the write lands."""

    async def query(self, message, session_id="default"):
        if self.queries:
            for _ in range(5):
                await asyncio.sleep(0)
        await super().query(message, session_id)


async def real_tick(_seconds: float) -> None:
    """A supervisor/connect-watch sleep that does not move fake time."""
    await asyncio.sleep(0.001)


class GatedConnectClient(FakeSDKClient):
    """``connect()`` blocks until the test opens ``gate``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate = asyncio.Event()

    async def connect(self, prompt=None):
        await self.gate.wait()
        await super().connect(prompt)


async def test_narration_excluded_from_answer(settings):
    narrated = [
        (0, fakes.message_start()),
        (0, fakes.text("Let me check…")),
        (0, fakes.tool_use("Bash")),
        (0, fakes.message_start()),
        (0, fakes.text("Answer")),
    ]
    h = Harness([*narrated, (0, fakes.result(text=None))], settings)
    assert (await h.run.run("x")).text == "Answer"
    h = Harness([*narrated, (0, fakes.result(text="Answer"))], settings)
    assert (await h.run.run("x")).text == "Answer"
    h = Harness([*narrated, (0, fakes.message_delta_end_turn())], settings)
    res = await h.run.run("x")  # grace completion
    assert (res.outcome, res.text) == (Outcome.DONE, "Answer")


async def test_inject_in_flight_during_result_is_not_dropped(settings):
    holder: dict = {}

    async def start_inject() -> None:
        holder["task"] = asyncio.create_task(holder["run"].inject("more"))
        await asyncio.sleep(0)  # the write is now in flight

    h = Harness(
        [
            (0, fakes.message_start()),
            (0, fakes.text("A")),
            ("call", start_inject),
            (0, fakes.result(text="A")),
            (10, fakes.message_start()),
            (0, fakes.text("B")),
            (0, fakes.result(text="B")),
        ],
        settings,
        client_cls=SlowInjectClient,
    )
    holder["run"] = h.run
    res = await h.run.run("q")
    await holder["task"]  # accepted, so it must have been answered
    assert h.client.queries == ["q", "more"]
    assert (res.outcome, res.text) == (Outcome.DONE, "A\n\nB")


async def test_inject_raises_if_run_finished_during_write(settings):
    holder: dict = {}

    async def start_inject() -> None:
        holder["task"] = asyncio.create_task(holder["run"].inject("more"))
        await asyncio.sleep(0)

    h = Harness(
        [
            (0, fakes.message_start()),
            ("call", start_inject),
            (0, fakes.result(is_error=True, text="boom")),
        ],
        settings,
        client_cls=SlowInjectClient,
    )
    holder["run"] = h.run
    res = await h.run.run("q")
    assert res.outcome is Outcome.ERROR
    with pytest.raises(RuntimeError):
        await holder["task"]  # the router can re-queue it


async def test_stream_error_after_stop_is_stopped(settings):
    holder: dict = {}
    h = Harness(
        [
            (0, fakes.message_start()),
            (0, fakes.tool_use("Bash")),
            ("call", lambda: holder["run"].interrupt("cancelled")),
            ("raise", RuntimeError("pipe broke")),
        ],
        settings,
    )
    holder["run"] = h.run
    res = await h.run.run("q")
    assert (res.outcome, res.reason) == (Outcome.STOPPED, "cancelled")


async def test_hung_progress_callback_does_not_freeze_watchdog(settings):
    async def hang(event: ProgressEvent) -> None:
        await asyncio.Event().wait()

    s = settings.model_copy(update={"harness_heartbeat_minutes": [1]})
    h = Harness(
        [(0, fakes.message_start()), (0, fakes.tool_use("X"))], s, on_progress=hang
    )
    # The heartbeat at 1m hangs in the supervisor; the stall must still fire.
    res = await asyncio.wait_for(h.run.run("q"), 5)
    assert (res.outcome, res.reason) == (Outcome.STOPPED, "stalled on X for 5m")


async def test_inject_during_connect_is_buffered_after_prompt(settings):
    h = Harness(
        [(0, fakes.message_start()), (0, fakes.text("A")), (0, fakes.result(text="A"))],
        settings,
        client_cls=GatedConnectClient,
    )
    h.run._sleep = real_tick  # the gate opens in real time; hold fake time
    task = asyncio.create_task(h.run.run("go"))
    await asyncio.sleep(0.01)
    await h.run.inject("more")  # accepted while still connecting
    assert h.client.queries == []
    h.client.gate.set()
    res = await asyncio.wait_for(task, 1)
    assert h.client.queries == ["go", "more"]
    assert (res.outcome, res.text) == (Outcome.DONE, "A")


async def test_interrupt_during_connect_returns_promptly(settings):
    h = Harness([], settings, client_cls=GatedConnectClient)
    h.run._sleep = real_tick
    task = asyncio.create_task(h.run.run("go"))
    await asyncio.sleep(0.01)
    await h.run.interrupt("cancelled")
    res = await asyncio.wait_for(task, 1)
    assert (res.outcome, res.reason) == (Outcome.STOPPED, "cancelled")
    assert h.client.queries == []
    assert h.client.disconnected


async def test_supervisor_failure_ends_run_with_error(settings):
    async def broken_sleep(_s: float) -> None:
        raise RuntimeError("tick broke")

    h = Harness([(0, fakes.message_start()), (0, fakes.text("x"))], settings)
    h.run._sleep = broken_sleep
    res = await asyncio.wait_for(h.run.run("q"), 1)
    assert (res.outcome, res.reason) == (Outcome.ERROR, "tick broke")


async def test_elapsed_s_counts_while_running_and_is_zero_after(settings):
    seen: list[float] = []

    async def peek(event: ProgressEvent) -> None:
        seen.append(h.run.elapsed_s)

    h = Harness(
        [(0, fakes.message_start()), (240, fakes.text("a")), (0, fakes.result())],
        settings,
        on_progress=peek,
    )
    assert h.run.elapsed_s == 0.0
    await h.run.run("hi")
    assert seen and max(seen) >= 240
    assert h.run.elapsed_s == 0.0


# ------------------------------------------------- final-review fixes (I1-I5, M2)


def delta(at: float = 0) -> tuple:
    """An invisible stream event (token delta): resets the stall clock only."""
    from claude_agent_sdk.types import StreamEvent

    return (
        at,
        StreamEvent(
            uuid="u-d",
            session_id="sess-1",
            event={"type": "content_block_delta"},
            parent_tool_use_id=None,
        ),
    )


class StatusBot(fakes.FakeBot):
    """FakeBot that stamps each status edit with fake time and client state."""

    def __init__(self, clock, client_ref) -> None:
        super().__init__()
        self.clock = clock
        self.client_ref = client_ref
        self.edits: list[tuple[float, str, bool]] = []

    async def edit_message_text(self, **kwargs):
        self.edits.append(
            (self.clock.now, kwargs["text"], self.client_ref().disconnected)
        )
        return await super().edit_message_text(**kwargs)


def with_status(h: "Harness", settings):
    """Feed the run's progress into a real StatusMessage on a StatusBot."""
    from src.harness.status_message import StatusMessage

    bot = StatusBot(h.clock, lambda: h.client)
    status = StatusMessage(bot, 1, None, 9, settings, h.clock, sleep=h.clock.wait)

    async def both(event: ProgressEvent) -> None:
        h.events.append(event)
        await status.update(event)

    h.run._on_progress = both
    return status, bot


async def test_supervisor_emits_tick_every_60s(settings):
    h = Harness(
        [(0, fakes.message_start()), delta(240), delta(240), (0, fakes.result())],
        settings,
    )
    res = await h.run.run("q")
    assert res.outcome is Outcome.DONE
    ticks = [round(e.elapsed_s) for e in h.kinds("tick")]
    assert len(ticks) == 8 and all(b - a == 60 for a, b in zip(ticks, ticks[1:]))


async def test_invisible_events_still_change_status_every_5m(settings):
    h = Harness(
        [(0, fakes.message_start()), *[delta(150) for _ in range(4)]]
        + [(0, fakes.result(text="fin"))],
        settings,
    )
    status, bot = with_status(h, settings)
    await status.open("⚙️ starting")
    res = await h.run.run("q")
    assert res.outcome is Outcome.DONE and res.elapsed_s >= 600
    times = [0.0, *[t for t, _, _ in bot.edits], res.elapsed_s]
    assert max(b - a for a, b in zip(times, times[1:])) <= 300


async def test_stall_shows_stalling_note_before_teardown(settings):
    h = Harness([(0, fakes.message_start()), (0, fakes.tool_use("Bash"))], settings)
    status, bot = with_status(h, settings)
    await status.open("⚙️ starting")
    res = await h.run.run("q")
    assert res.reason == "stalled on Bash for 5m"
    notes = [e for e in bot.edits if "stalled on Bash for 5m, stopping…" in e[1]]
    assert notes, bot.edits
    at, _, disconnected = notes[0]
    last_event = h.kinds("tool")[-1].elapsed_s  # fake time drifts a few ticks
    assert at - last_event <= 301
    assert not disconnected  # visible before teardown completes
    await status.finish(res)
    assert bot.edits[-1][1] == ("⚠️ stalled on Bash for 5m, stopped. Reply to retry.")


async def test_late_result_after_grace_done_not_appended_twice(settings):
    holder: dict = {}

    async def wait_finished() -> None:
        while not holder["run"]._finished.is_set():
            await asyncio.sleep(0)

    h = Harness(
        [
            (0, fakes.message_start()),
            (0, fakes.text("Answer.")),
            (0, fakes.message_delta_end_turn()),
            ("call", wait_finished),
            (0, fakes.result(text="Answer.")),
        ],
        settings,
    )
    holder["run"] = h.run
    original = h.run._on_result
    late: list = []

    def spy(message):
        late.append(h.run._finished.is_set())
        original(message)

    h.run._on_result = spy
    res = await h.run.run("q")
    assert res.outcome is Outcome.DONE
    assert res.text == "Answer."
    assert late == [True]  # the late ResultMessage really was processed


class HangDisconnectClient(FakeSDKClient):
    async def disconnect(self) -> None:
        self.disconnected = True
        self._closed_evt.set()
        await asyncio.Event().wait()


async def test_disconnect_timeout_reaps_child(settings, monkeypatch):
    reaped: list = []
    monkeypatch.setattr(
        agent_run_module,
        "reap_wedged_child",
        lambda client, reason: reaped.append((client, reason)),
    )
    h = Harness(
        [(0, fakes.message_start()), (0, fakes.result())],
        settings,
        client_cls=HangDisconnectClient,
    )
    res = await asyncio.wait_for(h.run.run("q"), 1)
    assert res.outcome is Outcome.DONE
    assert reaped == [(h.client, "disconnect_timeout")]


async def test_error_outcome_logs_stderr_tail(settings):
    from structlog.testing import capture_logs

    seen: list[str] = []
    holder: dict = {}

    def spew() -> None:
        for i in range(25):
            holder["client"].options.stderr(f"line {i}")

    h = Harness(
        [
            ("call", spew),
            ("raise", CLIConnectionError("CLI exited 1")),
        ],
        settings,
    )
    holder["client"] = h.client
    h.run._options = ClaudeAgentOptions(stderr=seen.append)
    with capture_logs() as logs:
        res = await h.run.run("q")
    assert res.outcome is Outcome.ERROR
    assert seen[-1] == "line 24"  # the caller's own callback still gets lines
    end = [e for e in logs if e["event"] == "harness.run_end"]
    assert end[-1]["log_level"] == "error"
    assert end[-1]["stderr_tail"] == "\n".join(f"line {i}" for i in range(5, 25))


async def test_done_outcome_run_end_has_no_stderr_tail(settings):
    from structlog.testing import capture_logs

    h = Harness([(0, fakes.message_start()), (0, fakes.result())], settings)
    with capture_logs() as logs:
        await h.run.run("q")
    end = [e for e in logs if e["event"] == "harness.run_end"][-1]
    assert end["log_level"] == "info" and "stderr_tail" not in end


async def test_result_reports_events_seen(settings):
    ok = Harness([(0, fakes.message_start()), (0, fakes.result())], settings)
    assert (await ok.run.run("q")).events_seen == 2
    dead = Harness([("raise", CLIConnectionError("No conversation found"))], settings)
    res = await dead.run.run("q")
    assert (res.outcome, res.events_seen) == (Outcome.ERROR, 0)


# ------------------------------------- resume-retry marker, connect-phase ticks


class SlowConnectClient(FakeSDKClient):
    """``connect()`` takes ``connect_s`` of fake time (forever when None)."""

    connect_s: float | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.connects = 0

    async def connect(self, prompt=None):
        self.connects += 1
        if self.connect_s is None:
            await asyncio.Event().wait()
        await self._sleep(self.connect_s)
        await super().connect(prompt)


class ExitOnConnectClient(FakeSDKClient):
    """``connect()`` fails the way the real CLI does on an unknown --resume."""

    async def connect(self, prompt=None):
        from claude_agent_sdk import ProcessError

        raise ProcessError(
            "Claude Code returned an error result: No conversation found with "
            "session ID: gone",
            exit_code=1,
        )


async def test_hung_connect_ticks_then_times_out_without_cli_exit(settings):
    h = Harness([], settings, client_cls=SlowConnectClient)
    status, bot = with_status(h, settings)
    await status.open("⚙️ starting")
    res = await asyncio.wait_for(h.run.run("q"), 5)
    assert (res.outcome, res.reason) == (Outcome.ERROR, "agent did not start in time")
    assert not res.cli_exited
    assert h.client.connects == 1  # a start timeout is not retried
    assert [round(e.elapsed_s) for e in h.kinds("tick")] == [60, 120, 180, 240]
    await status.finish(res)
    assert bot.edits[-1][1].startswith("❌")
    times = [0.0, *[t for t, _, _ in bot.edits]]
    assert max(b - a for a, b in zip(times, times[1:])) <= 300


async def test_slow_connect_emits_ticks_then_runs(settings):
    SlowClient = type("SlowClient", (SlowConnectClient,), {"connect_s": 150.0})
    h = Harness(
        [(0, fakes.message_start()), (0, fakes.result(text="ok"))],
        settings,
        client_cls=SlowClient,
    )
    res = await asyncio.wait_for(h.run.run("q"), 5)
    assert (res.outcome, res.text) == (Outcome.DONE, "ok")
    assert [round(e.elapsed_s) for e in h.kinds("tick")] == [60, 120]


async def test_cli_exit_before_any_event_is_marked(settings):
    from claude_agent_sdk import ProcessError

    on_connect = Harness([], settings, client_cls=ExitOnConnectClient)
    res = await on_connect.run.run("q")
    assert (res.outcome, res.cli_exited) == (Outcome.ERROR, True)
    on_read = Harness([("raise", ProcessError("exit", exit_code=1))], settings)
    assert (await on_read.run.run("q")).cli_exited
    late = Harness(
        [(0, fakes.message_start()), ("raise", ProcessError("exit", exit_code=1))],
        settings,
    )
    assert not (await late.run.run("q")).cli_exited  # it had already started


async def test_non_cli_failures_are_not_marked_cli_exit(settings):
    from claude_agent_sdk import CLINotFoundError

    crash = Harness([("raise", RuntimeError("harness bug"))], settings)
    assert not (await crash.run.run("q")).cli_exited
    missing = Harness([("raise", CLINotFoundError("no claude"))], settings)
    assert not (await missing.run.run("q")).cli_exited
    ok = Harness([(0, fakes.message_start()), (0, fakes.result())], settings)
    assert not (await ok.run.run("q")).cli_exited
