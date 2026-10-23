"""TopicRouter: idle/busy routing, slots, rotation, /new, /cancel, crashes.

Happy, inject, cancel and rotation paths run the real ``AgentRun`` against a
scripted ``FakeSDKClient`` (one per topic, matched by ``options.cwd``), a real
``SessionStore`` on a temp SQLite DB and a real ``StatusMessage`` on
``FakeBot``. Runs are held open with ``("call", gate.wait)`` script steps.
"""

import asyncio
import time
from pathlib import Path

import pytest
from claude_agent_sdk import PermissionResultAllow, SystemMessage

from src.config.settings import Settings
from src.harness.agent_run import AgentRun
from src.harness.run_slots import RunSlots
from src.harness.session_store import SessionStore
from src.harness.topic_router import (
    CARRYOVER_HEADER,
    EYES,
    QUEUED_BEHIND,
    QUEUED_TEXT,
    RESTARTING,
    STARTING_TEXT,
    TopicContext,
    TopicRouter,
)
from src.harness.types import Inbound, Outcome, RunResult
from src.security.validators import SecurityValidator
from src.storage.database import DatabaseManager
from tests.harness.fakes import FakeBot, FakeSDKClient, message_start, result, text

HOUR = 3600.0


async def settle(n: int = 20) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


async def until(pred, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.001)


@pytest.fixture
def settings(tmp_path):
    return Settings(
        telegram_bot_token="t",
        telegram_bot_username="b",
        approved_directory=str(tmp_path),
    )


@pytest.fixture
async def store(tmp_path):
    db = DatabaseManager(f"sqlite:///{tmp_path / 'bot.db'}")
    await db.initialize()
    yield SessionStore(db, projects_root=tmp_path / "projects")
    await db.close()


class World:
    """Router plus its fakes; ``script(topic, steps)`` arms that topic's client."""

    def __init__(self, tmp_path, settings, store, cap=3, **router_kw):
        self.tmp = tmp_path
        self.settings = settings
        self.store = store
        self.bot = FakeBot()
        self.slots = RunSlots(cap)
        self.scripts: dict[str, list[list]] = {}
        self.clients: dict[str, list[FakeSDKClient]] = {}
        self.summaries: list[str] = []
        self.summary = "SUMMARY"
        router_kw.setdefault("security_validator", SecurityValidator(tmp_path))
        router_kw.setdefault("run_factory", self._make_run)
        router_kw.setdefault("summarize_fn", self._summarize)
        self.router = TopicRouter(
            self.bot, store, self.slots, settings, self._resolve, **router_kw
        )

    def cwd(self, topic: str) -> Path:
        path = self.tmp / topic
        path.mkdir(exist_ok=True)
        return path

    def _resolve(self, inbound: Inbound) -> TopicContext:
        return TopicContext(inbound.topic_key, self.cwd(inbound.topic_key), None)

    def script(self, topic: str, steps: list) -> None:
        self.scripts.setdefault(topic, []).append(steps)

    def _make_run(self, options, on_progress):
        topic = Path(options.cwd).name
        factory = FakeSDKClient.factory(self.scripts[topic].pop(0))
        self.clients.setdefault(topic, []).append(factory.client)
        return AgentRun(
            options,
            on_progress,
            self.settings,
            factory,
            interrupt_timeout=0.05,
            cleanup_timeout=0.05,
            progress_timeout=0.05,
        )

    async def _summarize(self, session_id, cwd, settings):
        self.summaries.append(session_id)
        return self.summary

    def client(self, topic: str, i: int = -1) -> FakeSDKClient:
        return self.clients[topic][i]

    def sent_texts(self) -> list[str]:
        return [kw["text"] for kw in self.bot.calls_for("send_message")]

    def edited_texts(self) -> list[str]:
        return [kw["text"] for kw in self.bot.calls_for("edit_message_text")]


def msg(topic="t", text_="hello", message_id=100) -> Inbound:
    return Inbound(
        topic_key=topic,
        chat_id=-1001,
        thread_id=7,
        message_id=message_id,
        user_id=1,
        text=text_,
    )


def simple(session="sess-1", answer="answer"):
    return [
        (0, message_start()),
        (0, text(answer)),
        (0, result(session_id=session, text=answer)),
    ]


def init(session: str) -> SystemMessage:
    return SystemMessage(subtype="init", data={"session_id": session})


def gated(gate: asyncio.Event, session="sess-1", answer="answer", reached=None):
    reached = reached or asyncio.Event()
    return [
        (0, init(session)),
        (0, message_start()),
        ("call", reached.set),
        ("call", gate.wait),
        (0, text(answer)),
        (0, result(session_id=session, text=answer)),
    ]


async def run_rows(store):
    async with store._db.get_connection() as conn:
        cur = await conn.execute("SELECT topic_key, status FROM harness_runs")
        return [tuple(r) for r in await cur.fetchall()]


# ----------------------------------------------------------------- happy path


async def test_idle_topic_runs_and_persists_session(tmp_path, settings, store):
    w = World(tmp_path, settings, store)
    w.script("t", simple())
    await w.router.handle(msg())

    client = w.client("t")
    assert client.queries == ["hello"]
    assert client.options.resume is None
    assert w.sent_texts()[0] == STARTING_TEXT
    assert "answer" in w.sent_texts()[1]
    assert w.edited_texts()[-1].startswith("✅ done")
    assert (await store.get("t")).session_id == "sess-1"
    assert await run_rows(store) == [("t", "done")]
    assert w.router.active_run("t") is None
    assert w.slots.in_use == 0


async def test_next_run_resumes_stored_session(tmp_path, settings, store):
    w = World(tmp_path, settings, store)
    w.script("t", simple())
    w.script("t", simple())
    await w.router.handle(msg())
    await w.router.handle(msg(text_="again"))
    assert w.client("t", 1).options.resume == "sess-1"
    assert w.summaries == []


async def test_opening_text_override(tmp_path, settings, store):
    w = World(tmp_path, settings, store)
    w.script("t", simple())
    await w.router.handle(msg(), opening="🎙️ heard: 'hello' · starting")
    assert w.sent_texts()[0] == "🎙️ heard: 'hello' · starting"


async def test_default_can_use_tool_allows(tmp_path, settings, store):
    w = World(tmp_path, settings, store)
    w.script("t", simple())
    await w.router.handle(msg())
    allow = await w.client("t").options.can_use_tool("Bash", {}, None)
    assert isinstance(allow, PermissionResultAllow)


# ---------------------------------------------------------------- concurrency


async def test_two_topics_run_in_parallel(tmp_path, settings, store):
    w = World(tmp_path, settings, store)
    gate = asyncio.Event()
    w.script("a", gated(gate))
    w.script("b", gated(gate))
    tasks = [
        asyncio.create_task(w.router.handle(msg("a"))),
        asyncio.create_task(w.router.handle(msg("b"))),
    ]
    await until(
        lambda: len(w.clients) == 2 and all(c[0].queries for c in w.clients.values())
    )
    assert w.router.active_run("a") is not None
    assert w.router.active_run("b") is not None
    gate.set()
    await asyncio.gather(*tasks)


async def test_fourth_run_is_queued_until_a_slot_frees(tmp_path, settings, store):
    w = World(tmp_path, settings, store, cap=3)
    gates = {t: asyncio.Event() for t in "abcd"}
    for t in "abcd":
        w.script(t, gated(gates[t], session=f"s-{t}"))
    tasks = {}
    for t in "abc":
        tasks[t] = asyncio.create_task(w.router.handle(msg(t)))
    await until(lambda: len(w.clients) == 3)
    tasks["d"] = asyncio.create_task(w.router.handle(msg("d")))
    await until(lambda: QUEUED_TEXT in w.sent_texts())
    await settle()
    assert "d" not in w.clients

    gates["a"].set()
    await tasks["a"]
    await until(lambda: "d" in w.clients and w.client("d").queries)
    for g in gates.values():
        g.set()
    await asyncio.gather(*tasks.values())
    assert (await store.get("d")).session_id == "s-d"
    assert w.slots.in_use == 0


# ------------------------------------------------------------------ injection


async def test_busy_topic_injects_and_reacts_without_new_status(
    tmp_path, settings, store
):
    w = World(tmp_path, settings, store)
    gate, reached = asyncio.Event(), asyncio.Event()
    w.script(
        "t",
        [
            (0, message_start()),
            ("call", reached.set),
            ("call", gate.wait),
            (0, text("first")),
            (0, result(text="first")),
            (0, message_start()),
            (0, text("second")),
            (0, result(text="second")),
        ],
    )
    first = asyncio.create_task(w.router.handle(msg()))
    await reached.wait()  # the first turn is underway (not still connecting)
    await w.router.handle(msg(text_="also this", message_id=101))

    assert w.client("t").queries == ["hello", "also this"]
    reaction = w.bot.calls_for("set_message_reaction")
    assert reaction == [{"chat_id": -1001, "message_id": 101, "reaction": EYES}]
    status_opens = [t for t in w.sent_texts() if t == STARTING_TEXT]
    assert len(status_opens) == 1
    gate.set()
    await first
    assert "first" in w.sent_texts()[-1] and "second" in w.sent_texts()[-1]
    assert len(w.clients["t"]) == 1


class FakeRun:
    """Minimal AgentRun stand-in for paths the real one cannot time reliably."""

    def __init__(self, *, inject_error=None, run_error=None, session="fs"):
        self.gate = asyncio.Event()
        self.prompts: list = []
        self.injected: list = []
        self.inject_error = inject_error
        self.run_error = run_error
        self.session = session
        self.is_active = False
        self.paused = 0

    async def run(self, prompt):
        self.prompts.append(prompt)
        self.is_active = True
        try:
            await self.gate.wait()
            if self.run_error:
                raise self.run_error
            return RunResult(
                outcome=Outcome.DONE, text="ok", elapsed_s=1, session_id=self.session
            )
        finally:
            self.is_active = False

    async def inject(self, content):
        if self.inject_error:
            raise self.inject_error
        self.injected.append(content)

    async def interrupt(self, reason):
        self.stop_reason = reason

    def pause_watchdog(self):
        self.paused += 1

    def resume_watchdog(self):
        self.paused -= 1


def fake_runs(runs: list[FakeRun]):
    made: list[FakeRun] = []

    def factory(options, on_progress):
        run = runs.pop(0)
        run.options = options
        made.append(run)
        return run

    factory.made = made
    return factory


async def test_inject_into_finishing_run_starts_a_new_run(tmp_path, settings, store):
    first = FakeRun(inject_error=RuntimeError("run already finished"))
    second = FakeRun()
    second.gate.set()
    w = World(tmp_path, settings, store, run_factory=fake_runs([first, second]))
    t1 = asyncio.create_task(w.router.handle(msg()))
    await until(lambda: first.prompts)
    t2 = asyncio.create_task(w.router.handle(msg(text_="late", message_id=101)))
    await settle()
    assert not t2.done()  # waits for the first flow, then runs its own
    first.gate.set()
    await asyncio.gather(t1, t2)
    assert second.prompts == ["late"]
    assert second.options.resume == "fs"


async def test_message_while_topic_queued_waits_behind(tmp_path, settings, store):
    blocker = FakeRun()
    first = FakeRun()
    second = FakeRun()
    w = World(
        tmp_path,
        settings,
        store,
        cap=1,
        run_factory=fake_runs([blocker, first, second]),
    )
    tb = asyncio.create_task(w.router.handle(msg("other")))
    await until(lambda: blocker.prompts)
    t1 = asyncio.create_task(w.router.handle(msg("t")))
    await until(lambda: QUEUED_TEXT in w.sent_texts())
    t2 = asyncio.create_task(w.router.handle(msg("t", "follow-up", message_id=101)))
    await until(lambda: w.bot.calls_for("set_message_reaction"))
    assert w.bot.calls_for("set_message_reaction")[0]["reaction"] == QUEUED_BEHIND
    for r in (blocker, first, second):
        r.gate.set()
    await asyncio.gather(tb, t1, t2)
    assert first.prompts == ["hello"]
    assert second.prompts == ["follow-up"]


# ------------------------------------------------------------------ ask bridge


class FakeBridge:
    def __init__(self, answers: bool):
        self.answers = answers
        self.texts: list[tuple[str, str]] = []
        self.made: list[tuple] = []

    async def try_answer_with_text(self, topic_key: str, text_: str) -> bool:
        self.texts.append((topic_key, text_))
        return self.answers

    def make_can_use_tool(self, topic_key, chat_id, thread_id, run):
        self.made.append((topic_key, chat_id, thread_id, run))

        async def can_use_tool(tool_name, tool_input, context):
            run.pause_watchdog()
            return PermissionResultAllow()

        return can_use_tool


@pytest.mark.parametrize("answers", [True, False])
async def test_bridge_gets_text_before_injection(tmp_path, settings, store, answers):
    bridge = FakeBridge(answers)
    run = FakeRun()
    w = World(
        tmp_path, settings, store, ask_bridge=bridge, run_factory=fake_runs([run])
    )
    task = asyncio.create_task(w.router.handle(msg()))
    await until(lambda: run.prompts)
    await w.router.handle(msg(text_="Option B", message_id=101))
    assert bridge.texts == [("t", "Option B")]
    assert run.injected == ([] if answers else ["Option B"])
    run.gate.set()
    await task


async def test_bridge_callback_is_late_bound_to_the_run(tmp_path, settings, store):
    bridge = FakeBridge(False)
    run = FakeRun()
    w = World(
        tmp_path, settings, store, ask_bridge=bridge, run_factory=fake_runs([run])
    )
    task = asyncio.create_task(w.router.handle(msg()))
    await until(lambda: run.prompts)
    topic, chat, thread, ref = bridge.made[0]
    assert (topic, chat, thread) == ("t", -1001, 7)
    await run.options.can_use_tool("AskUserQuestion", {}, None)
    assert run.paused == 1  # the ref forwarded to the live run
    run.gate.set()
    await task


# ------------------------------------------------------------- /cancel, /new


async def test_cancel_stops_run_and_keeps_session(tmp_path, settings, store):
    w = World(tmp_path, settings, store)
    gate, reached = asyncio.Event(), asyncio.Event()
    w.script("t", gated(gate, reached=reached))
    task = asyncio.create_task(w.router.handle(msg()))
    await reached.wait()
    assert await w.router.cancel("t") is True
    await task
    assert w.edited_texts()[-1].startswith("⚠️ stopped (cancelled) · ")
    assert (await store.get("t")).session_id == "sess-1"
    assert await run_rows(store) == [("t", "stopped")]


async def test_cancel_without_run_is_noop(tmp_path, settings, store):
    w = World(tmp_path, settings, store)
    assert await w.router.cancel("t") is False
    assert w.bot.calls == []


async def test_cancel_while_queued_never_starts(tmp_path, settings, store):
    blocker = FakeRun()
    w = World(tmp_path, settings, store, cap=1, run_factory=fake_runs([blocker]))
    tb = asyncio.create_task(w.router.handle(msg("other")))
    await until(lambda: blocker.prompts)
    t1 = asyncio.create_task(w.router.handle(msg("t")))
    await until(lambda: QUEUED_TEXT in w.sent_texts())
    assert await w.router.cancel("t") is True
    blocker.gate.set()
    await asyncio.gather(tb, t1)
    assert w.edited_texts()[-1].startswith("⚠️ stopped (cancelled)")
    assert w.slots.in_use == 0


async def test_new_clears_session_no_summary(tmp_path, settings, store):
    w = World(tmp_path, settings, store)
    await store.save("t", "old", w.cwd("t"))
    await w.router.new_session("t")
    assert await store.get("t") is None
    w.script("t", simple())
    await w.router.handle(msg())
    assert w.client("t").options.resume is None
    assert w.client("t").queries == ["hello"]
    assert w.summaries == []


async def test_new_during_run_stops_it_and_does_not_resave(tmp_path, settings, store):
    w = World(tmp_path, settings, store)
    gate = asyncio.Event()
    w.script("t", gated(gate))
    task = asyncio.create_task(w.router.handle(msg()))
    await until(lambda: w.router.active_run("t") is not None)
    await w.router.new_session("t")
    await task
    assert await store.get("t") is None
    assert w.edited_texts()[-1].startswith("⚠️ stopped (new session) · ")


# ------------------------------------------------------------------- rotation


async def test_stale_session_rotates_with_summary(tmp_path, settings, store):
    w = World(tmp_path, settings, store)
    await store.save("t", "old", w.cwd("t"), now=time.time() - 2 * HOUR - 60)
    w.script("t", simple(session="new"))
    await w.router.handle(msg())
    assert w.summaries == ["old"]
    assert w.client("t").options.resume is None
    assert w.client("t").queries == [f"{CARRYOVER_HEADER}\nSUMMARY\n\nhello"]
    assert (await store.get("t")).session_id == "new"


async def test_summary_failure_still_starts_fresh(tmp_path, settings, store):
    w = World(tmp_path, settings, store)
    w.summary = ""
    await store.save("t", "old", w.cwd("t"), now=time.time() - 3 * HOUR)
    w.script("t", simple(session="new"))
    await w.router.handle(msg())
    assert w.client("t").options.resume is None
    assert w.client("t").queries == ["hello"]


async def test_summary_exception_still_starts_fresh(tmp_path, settings, store):
    async def boom(session_id, cwd, settings):
        raise RuntimeError("haiku down")

    w = World(tmp_path, settings, store, summarize_fn=boom)
    await store.save("t", "old", w.cwd("t"), now=time.time() - 3 * HOUR)
    w.script("t", simple(session="new"))
    await w.router.handle(msg())
    assert w.client("t").options.resume is None
    assert w.client("t").queries == ["hello"]


# ---------------------------------------------------------------------- crash


async def test_run_crash_errors_and_next_message_gets_fresh_run(
    tmp_path, settings, store
):
    crashing = FakeRun(run_error=ValueError("boom\nsecond line"))
    crashing.gate.set()
    healthy = FakeRun(session="fresh")
    healthy.gate.set()
    w = World(tmp_path, settings, store, run_factory=fake_runs([crashing, healthy]))
    await w.router.handle(msg())
    assert w.edited_texts()[-1] == "❌ error: boom"
    assert w.router.active_run("t") is None
    assert await run_rows(store) == [("t", "error")]
    assert w.slots.in_use == 0

    await w.router.handle(msg(text_="retry"))
    assert healthy.prompts == ["retry"]
    assert (await store.get("t")).session_id == "fresh"


async def test_crash_while_message_steers_never_runs_two_flows(
    tmp_path, settings, store
):
    """B waits on the topic lock while C (holding it) is in the bridge; the
    flow crashes meanwhile. B and C must not both start flows."""
    crashing = FakeRun(
        inject_error=RuntimeError("no active run"), run_error=ValueError("boom")
    )
    # Later runs stay open until released, so an overlap would be visible.
    runs = [crashing, FakeRun(session="s2"), FakeRun(session="s3")]
    active = {"now": 0, "max": 0}
    made: list[FakeRun] = []

    def factory(options, on_progress):
        run = runs.pop(0)
        made.append(run)
        run.options = options
        original = run.run

        async def counted(prompt):
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
            try:
                return await original(prompt)
            finally:
                active["now"] -= 1

        run.run = counted
        return run

    in_bridge, release_bridge = asyncio.Event(), asyncio.Event()

    class BlockingBridge(FakeBridge):
        async def try_answer_with_text(self, topic_key, text_):
            in_bridge.set()
            await release_bridge.wait()
            return False

    w = World(
        tmp_path, settings, store, run_factory=factory, ask_bridge=BlockingBridge(False)
    )
    a = asyncio.create_task(w.router.handle(msg()))
    await until(lambda: crashing.prompts)
    c = asyncio.create_task(w.router.handle(msg(text_="C", message_id=102)))
    await in_bridge.wait()  # C holds the topic lock
    b = asyncio.create_task(w.router.handle(msg(text_="B", message_id=101)))
    await settle()
    crashing.gate.set()
    await a  # the flow crashed and closed while C and B are mid-decision
    release_bridge.set()
    await until(lambda: active["now"] >= 1)
    await asyncio.sleep(0.2)  # real DB I/O: give a second flow time to appear
    assert active["now"] == 1  # exactly one flow runs in the topic
    while not (b.done() and c.done()):
        for r in made:
            r.gate.set()
        await settle()
    await asyncio.gather(b, c)
    assert active["max"] == 1
    handled = [p for r in made[1:] for p in r.prompts + r.injected]
    assert sorted(handled) == ["B", "C"]
    assert len(w.router._workers) == 1


async def test_new_while_queued_starts_fresh(tmp_path, settings, store):
    blocker = FakeRun()
    target = FakeRun(session="fresh")
    target.gate.set()
    w = World(
        tmp_path, settings, store, cap=1, run_factory=fake_runs([blocker, target])
    )
    await store.save("t", "old", w.cwd("t"))
    tb = asyncio.create_task(w.router.handle(msg("other")))
    await until(lambda: blocker.prompts)
    t1 = asyncio.create_task(w.router.handle(msg("t")))
    await until(lambda: QUEUED_TEXT in w.sent_texts())
    await w.router.new_session("t")
    blocker.gate.set()
    await asyncio.gather(tb, t1)
    assert target.options.resume is None
    assert (await store.get("t")).session_id == "fresh"


async def test_new_during_setup_after_row_read_starts_fresh(tmp_path, settings, store):
    """/new lands while the carry-over summary runs (old row already read)."""
    in_summary, release = asyncio.Event(), asyncio.Event()

    async def slow_summary(session_id, cwd, settings):
        in_summary.set()
        await release.wait()
        return "OLD CONTEXT"

    run = FakeRun(session="fresh")
    run.gate.set()
    w = World(
        tmp_path,
        settings,
        store,
        run_factory=fake_runs([run]),
        summarize_fn=slow_summary,
    )
    await store.save("t", "old", w.cwd("t"), now=time.time() - 3 * HOUR)
    task = asyncio.create_task(w.router.handle(msg()))
    await in_summary.wait()
    await w.router.new_session("t")
    release.set()
    await task
    assert run.options.resume is None
    assert run.prompts == ["hello"]  # no carry-over after /new
    assert (await store.get("t")).session_id == "fresh"


async def test_new_during_setup_with_resumable_session(tmp_path, settings, store):
    """/new after the (fresh) row was read for resume: no resume of the old id."""
    in_start, release = asyncio.Event(), asyncio.Event()
    real_start_run = store.start_run

    async def slow_start_run(*args, **kwargs):
        in_start.set()
        await release.wait()
        return await real_start_run(*args, **kwargs)

    store.start_run = slow_start_run
    run = FakeRun(session="fresh")
    run.gate.set()
    w = World(tmp_path, settings, store, run_factory=fake_runs([run]))
    await store.save("t", "old", w.cwd("t"))
    task = asyncio.create_task(w.router.handle(msg()))
    await in_start.wait()
    await w.router.new_session("t")
    release.set()
    await task
    assert run.options.resume is None
    assert (await store.get("t")).session_id == "fresh"


@pytest.mark.parametrize("method", ["get", "start_run"])
async def test_store_failure_finishes_with_error_and_frees_slot(
    tmp_path, settings, store, method
):
    async def broken(*args, **kwargs):
        raise OSError("database is locked")

    setattr(store, method, broken)
    run = FakeRun()
    w = World(tmp_path, settings, store, cap=1, run_factory=fake_runs([run]))
    await w.router.handle(msg())  # must not raise
    assert w.edited_texts()[-1] == "❌ error: database is locked"
    assert w.slots.in_use == 0
    assert run.prompts == []
    assert w.router._workers["t"].flow_done is None


async def test_cancel_after_run_returned_is_noop(tmp_path, settings, store):
    run = FakeRun()
    w = World(tmp_path, settings, store, run_factory=fake_runs([run]))
    task = asyncio.create_task(w.router.handle(msg()))
    await until(lambda: run.prompts)
    run.is_active = False  # returned; the flow is still finishing
    assert await w.router.cancel("t") is False
    run.gate.set()
    await task


class BlockingFinishStatus:
    """Status double whose finish() blocks until released."""

    def __init__(self):
        self.in_finish = asyncio.Event()
        self.release = asyncio.Event()
        self.results: list[RunResult] = []

    async def open(self, text_):
        return 1

    async def update(self, ev):
        pass

    async def note(self, text_):
        pass

    async def finish(self, result_):
        self.results.append(result_)
        self.in_finish.set()
        await self.release.wait()


async def test_new_after_run_returned_before_save_sticks(tmp_path, settings, store):
    """/new while status.finish is in flight (run returned, save pending)."""
    first, second = FakeRun(session="pre-new"), FakeRun(session="post-new")
    first.gate.set()
    second.gate.set()
    statuses = [BlockingFinishStatus(), BlockingFinishStatus()]
    statuses[1].release.set()
    w = World(
        tmp_path,
        settings,
        store,
        run_factory=fake_runs([first, second]),
        status_factory=lambda inbound: statuses.pop(0),
    )
    blocked = statuses[0]
    await store.save("t", "old", w.cwd("t"))
    task = asyncio.create_task(w.router.handle(msg()))
    await blocked.in_finish.wait()
    assert first.options.resume == "old"
    await w.router.new_session("t")
    blocked.release.set()
    await task
    assert await store.get("t") is None  # neither "old" nor "pre-new" survives

    await w.router.handle(msg(text_="next"))
    assert second.options.resume is None
    assert (await store.get("t")).session_id == "post-new"


# ------------------------------------------------------------------- shutdown


async def test_shutdown_interrupts_live_run_and_leaves_row_running(
    tmp_path, settings, store
):
    """A process stop interrupts the run but keeps its row 'running' so the
    next boot posts the restart notice (and only then marks it abandoned)."""
    w = World(tmp_path, settings, store)
    gate, reached = asyncio.Event(), asyncio.Event()
    w.script("t", gated(gate, reached=reached))
    task = asyncio.create_task(w.router.handle(msg()))
    await reached.wait()
    assert await w.router.shutdown() == 1
    await task
    assert w.client("t").interrupted == 1
    assert w.edited_texts()[-1].startswith(f"⚠️ stopped ({RESTARTING}) · ")
    assert (await store.get("t")).session_id == "sess-1"  # resumable
    assert await run_rows(store) == [("t", "running")]


async def test_shutdown_stops_queued_flow_before_it_runs(tmp_path, settings, store):
    blocker = FakeRun()
    w = World(tmp_path, settings, store, cap=1, run_factory=fake_runs([blocker]))
    tb = asyncio.create_task(w.router.handle(msg("other")))
    await until(lambda: blocker.prompts)
    t1 = asyncio.create_task(w.router.handle(msg("t")))
    await until(lambda: QUEUED_TEXT in w.sent_texts())
    await w.router.shutdown()
    blocker.gate.set()  # FakeRun ignores interrupt; let it return
    await asyncio.gather(tb, t1)
    assert w.edited_texts()[-1].startswith(f"⚠️ stopped ({RESTARTING})")
    assert await run_rows(store) == [("other", "running")]


async def test_message_after_shutdown_never_starts_a_run(tmp_path, settings, store):
    w = World(tmp_path, settings, store, run_factory=fake_runs([]))
    await w.router.shutdown()
    await w.router.handle(msg())
    assert w.edited_texts()[-1].startswith(f"⚠️ stopped ({RESTARTING})")
    assert await run_rows(store) == []


async def test_run_finishing_on_its_own_during_shutdown_is_recorded(
    tmp_path, settings, store
):
    run = FakeRun()
    w = World(tmp_path, settings, store, run_factory=fake_runs([run]))
    task = asyncio.create_task(w.router.handle(msg()))
    await until(lambda: run.prompts)
    run.is_active = False  # already returned; nothing left to interrupt
    assert await w.router.shutdown() == 0
    run.gate.set()
    await task
    assert await run_rows(store) == [("t", "done")]


# ------------------------------------------- final-review fixes (I2, I5, M8)


def dead_cli():
    from claude_agent_sdk import ProcessError

    return [("raise", ProcessError("Command failed with exit code 1", exit_code=1))]


async def test_unresumable_session_retries_fresh_once(tmp_path, settings, store):
    w = World(tmp_path, settings, store)
    await store.save("t", "gone", w.cwd("t"))
    w.script("t", dead_cli())
    w.script("t", simple(session="new"))
    await w.router.handle(msg())
    assert w.client("t", 0).options.resume == "gone"
    assert w.client("t", 1).options.resume is None
    assert w.client("t", 1).queries == ["hello"]  # no carry-over summary
    assert w.summaries == []
    sent = w.sent_texts()
    assert sent[0] == STARTING_TEXT and len(sent) == 2  # one status message
    assert w.edited_texts()[-1].startswith("✅ done")
    assert (await store.get("t")).session_id == "new"
    assert await run_rows(store) == [("t", "done")]
    assert w.slots.in_use == 0


async def test_resume_failure_after_events_is_not_retried(tmp_path, settings, store):
    w = World(tmp_path, settings, store)
    await store.save("t", "live", w.cwd("t"))
    w.script("t", [(0, message_start()), *dead_cli()])
    await w.router.handle(msg())
    assert len(w.clients["t"]) == 1
    assert w.edited_texts()[-1].startswith("❌ error")
    assert (await store.get("t")).session_id == "live"


async def test_fresh_run_error_is_not_retried(tmp_path, settings, store):
    w = World(tmp_path, settings, store)
    w.script("t", dead_cli())
    await w.router.handle(msg())
    assert len(w.clients["t"]) == 1
    assert w.edited_texts()[-1].startswith("❌ error")


async def test_fresh_retry_happens_only_once(tmp_path, settings, store):
    w = World(tmp_path, settings, store)
    await store.save("t", "gone", w.cwd("t"))
    w.script("t", dead_cli())
    w.script("t", dead_cli())
    await w.router.handle(msg())
    assert len(w.clients["t"]) == 2
    assert w.edited_texts()[-1].startswith("❌ error")
    assert await store.get("t") is None  # the dead session was dropped
    assert await run_rows(store) == [("t", "error")]


async def test_raising_status_factory_does_not_wedge_topic(tmp_path, settings, store):
    calls = {"n": 0}
    w = World(tmp_path, settings, store)

    def flaky(inbound):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("status broke")
        return w.router._default_status(inbound)

    w.router._status_factory = flaky
    w.script("t", simple())
    with pytest.raises(RuntimeError):
        await w.router.handle(msg())
    assert w.router._workers["t"].flow_done is None
    await asyncio.wait_for(w.router.handle(msg(text_="again")), 2)
    assert w.client("t").queries == ["again"]


async def test_hung_carryover_still_starts_the_run(tmp_path, settings, store):
    import functools

    from src.harness import carryover

    release = asyncio.Event()

    async def stubborn(*, prompt, options, transport=None):
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await release.wait()  # a wedged CLI swallowing the cancel
        yield None

    summarize = functools.partial(
        carryover.summarize_for_carryover,
        query_fn=stubborn,
        timeout=0.05,
        close_timeout=0.05,
        transport_factory=lambda prompt, options: object(),
    )
    w = World(tmp_path, settings, store, summarize_fn=summarize)
    await store.save("t", "old", w.cwd("t"), now=time.time() - 3 * HOUR)
    w.script("t", simple(session="new"))
    await asyncio.wait_for(w.router.handle(msg()), 2)
    assert w.client("t").options.resume is None
    assert w.client("t").queries == ["hello"]
    assert (await store.get("t")).session_id == "new"
    assert w.slots.in_use == 0
    release.set()


# ------------------------- fresh retry only on a CLI exit; ticks during connect


class FakeClock:
    """Fake monotonic time: AgentRun's supervisor sleep drives it."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def tick(self, seconds: float) -> None:
        self.now += seconds
        await asyncio.sleep(0)

    async def wait(self, seconds: float) -> None:
        target = self.now + seconds
        while self.now < target:
            await asyncio.sleep(0)


class TimedBot(FakeBot):
    """Stamps every status send/edit with fake time."""

    def __init__(self, clock: FakeClock) -> None:
        super().__init__()
        self.clock = clock
        self.stamps: list[float] = []

    async def send_message(self, **kwargs):
        self.stamps.append(self.clock())
        return await super().send_message(**kwargs)

    async def edit_message_text(self, **kwargs):
        self.stamps.append(self.clock())
        return await super().edit_message_text(**kwargs)


class HangConnectClient(FakeSDKClient):
    async def connect(self, prompt=None):
        await asyncio.Event().wait()


class RejectResumeClient(FakeSDKClient):
    """What the real CLI does with an unknown --resume: connect() raises."""

    async def connect(self, prompt=None):
        from claude_agent_sdk import ProcessError

        raise ProcessError(
            "Claude Code returned an error result: No conversation found",
            exit_code=1,
        )


def timed_world(tmp_path, settings, store, client_classes):
    """A World on fake time whose n-th run uses ``client_classes[n]``."""
    from src.harness.status_message import StatusMessage

    clock = FakeClock()
    w = World(tmp_path, settings, store, clock=clock)
    w.bot = TimedBot(clock)
    w.router._bot = w.bot
    classes = list(client_classes)

    def make_run(options, on_progress):
        topic = Path(options.cwd).name
        steps = w.scripts.get(topic, [[]]).pop(0) if w.scripts.get(topic) else []
        factory = classes.pop(0).factory(steps, sleep=clock.wait)
        w.clients.setdefault(topic, []).append(factory.client)
        return AgentRun(
            options,
            on_progress,
            settings,
            factory,
            clock=clock,
            sleep=clock.tick,
            interrupt_timeout=0.05,
            cleanup_timeout=0.05,
            progress_timeout=0.05,
        )

    def status(inbound):
        return StatusMessage(
            w.bot,
            inbound.chat_id,
            inbound.thread_id,
            inbound.message_id,
            settings,
            clock,
            sleep=clock.wait,
        )

    w.router._run_factory = make_run
    w.router._status_factory = status
    return w, clock


async def test_hung_connect_on_resume_is_not_retried_fresh(tmp_path, settings, store):
    w, clock = timed_world(tmp_path, settings, store, [HangConnectClient])
    await store.save("t", "live", w.cwd("t"))
    await asyncio.wait_for(w.router.handle(msg()), 5)
    assert len(w.clients["t"]) == 1  # exactly one attempt
    assert w.client("t").options.resume == "live"
    assert w.edited_texts()[-1].startswith("❌")
    assert (await store.get("t")).session_id == "live"  # pointer kept
    assert await run_rows(store) == [("t", "error")]
    gaps = [b - a for a, b in zip(w.bot.stamps, w.bot.stamps[1:])]
    assert max(gaps) <= 300, w.bot.stamps


async def test_rejected_resume_on_connect_retries_fresh(tmp_path, settings, store):
    w, _ = timed_world(tmp_path, settings, store, [RejectResumeClient, FakeSDKClient])
    await store.save("t", "gone", w.cwd("t"))
    w.script("t", [])
    w.script("t", simple(session="new"))
    await asyncio.wait_for(w.router.handle(msg()), 5)
    assert [c.options.resume for c in w.clients["t"]] == ["gone", None]
    assert w.edited_texts()[-1].startswith("✅ done")
    assert (await store.get("t")).session_id == "new"


async def test_harness_crash_while_resuming_is_not_retried(tmp_path, settings, store):
    calls: list = []

    def crashing_run(options, on_progress):
        calls.append(options.resume)
        raise RuntimeError("harness bug")

    w = World(tmp_path, settings, store, run_factory=crashing_run)
    await store.save("t", "live", w.cwd("t"))
    await w.router.handle(msg())
    assert calls == ["live"]
    assert w.edited_texts()[-1].startswith("❌ error")
    assert (await store.get("t")).session_id == "live"
