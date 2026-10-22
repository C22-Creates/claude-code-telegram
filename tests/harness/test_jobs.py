"""Scheduled jobs and dispatcher tasks through AgentRun.

Real ``AgentRun`` + ``FakeSDKClient`` throughout; no network, no CLI.
"""

import asyncio
import time
from pathlib import Path

import pytest
from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

from src.config.settings import Settings
from src.dispatcher.runner import JUDGMENT_MODEL, ClaudeTaskRunner
from src.events.bus import EventBus
from src.events.types import AgentResponseEvent, ScheduledEvent
from src.harness.agent_run import AgentRun
from src.harness.jobs import (
    NO_ONE_TO_ASK,
    ScheduledJobHandler,
    outcome_text,
    run_job,
    unattended_can_use_tool,
)
from src.harness.run_slots import RunSlots
from src.harness.types import Inbound, Outcome, RunResult
from tests.harness import fakes
from tests.harness.fakes import FakeSDKClient, message_start, result, text
from tests.harness.test_agent_run import FakeClock

FAST = dict(interrupt_timeout=0.05, cleanup_timeout=0.05, progress_timeout=0.05)


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


def simple(answer="digest ready", session="sess-j"):
    return [(0, message_start()), (0, text(answer)), (0, result(session, text=answer))]


# ------------------------------------------------------------------- run_job


async def test_run_job_is_fresh_session_with_model_override(tmp_path, settings):
    factory = FakeSDKClient.factory(simple())
    res = await run_job(
        "hi", tmp_path, None, RunSlots(3), settings, "claude-x",
        client_factory=factory, run_kwargs=FAST,
    )  # fmt: skip
    assert res.outcome is Outcome.DONE and res.text == "digest ready"
    opts = factory.client.options
    assert opts.model == "claude-x"
    assert not opts.resume


async def test_run_job_never_raises_on_options_error(tmp_path, settings):
    def boom(*a, **k):
        raise ValueError("bad config")

    import src.harness.jobs as jobs

    orig = jobs.build_options
    jobs.build_options = boom
    try:
        res = await run_job("hi", tmp_path, None, RunSlots(3), settings)
    finally:
        jobs.build_options = orig
    assert res.outcome is Outcome.ERROR and res.reason == "bad config"


async def test_run_job_applies_60m_cap(tmp_path, settings):
    clock = FakeClock()
    script = [(0, message_start())] + [(240, text(f"t{i}")) for i in range(30)]
    factory = FakeSDKClient.factory(script, sleep=clock.wait)
    res = await run_job(
        "long", tmp_path, None, RunSlots(3), settings,
        client_factory=factory, run_kwargs=dict(clock=clock, sleep=clock.tick, **FAST),
    )  # fmt: skip
    assert res.outcome is Outcome.STOPPED and res.reason == "hit 60m cap"
    assert "⚠️ stopped (hit 60m cap)" in outcome_text(res)


async def test_unattended_callback_denies_only_ask_question():
    deny = await unattended_can_use_tool("AskUserQuestion", {}, None)
    assert isinstance(deny, PermissionResultDeny)
    assert deny.message == NO_ONE_TO_ASK and deny.interrupt is False
    assert isinstance(
        await unattended_can_use_tool("Bash", {}, None), PermissionResultAllow
    )


async def test_outcome_text_variants():
    def r(outcome, body, reason=""):
        return RunResult(outcome=outcome, text=body, elapsed_s=5, reason=reason)

    assert outcome_text(r(Outcome.DONE, "all good")) == "all good"
    assert outcome_text(r(Outcome.STOPPED, "part", "stalled on X")) == (
        "part\n\n⚠️ stopped (stalled on X) · 5s"
    )
    assert outcome_text(r(Outcome.ERROR, "", "boom")) == "❌ error: boom"


# ------------------------------------------------------------------- handler


async def collect(bus):
    got = []

    async def sink(event):
        got.append(event)

    bus.subscribe(AgentResponseEvent, sink)
    await bus.start()
    return got


async def test_daily_digest_event_yields_one_response_with_original_ids(
    tmp_path, settings
):
    bus = EventBus()
    got = await collect(bus)
    seen = {}

    async def fake_job(prompt, cwd, agent, slots, settings_, model=None):
        seen.update(prompt=prompt, cwd=cwd)
        return RunResult(outcome=Outcome.DONE, text="digest!", elapsed_s=1)

    handler = ScheduledJobHandler(bus, RunSlots(3), settings, run_job_fn=fake_job)
    handler.register()
    event = ScheduledEvent(
        job_id="j1", job_name="daily-digest", prompt="go", skill_name="daily-digest",
        working_directory=tmp_path, target_chat_ids=[42], message_thread_id=7,
    )  # fmt: skip
    await bus.publish(event)
    await until(lambda: len(got) == 1)
    await handler.wait_idle()
    await bus.stop()
    assert len(got) == 1
    assert (got[0].chat_id, got[0].message_thread_id, got[0].text) == (42, 7, "digest!")
    assert got[0].originating_event_id == event.id
    assert seen == {"prompt": "/daily-digest\n\ngo", "cwd": tmp_path}


async def test_stopped_job_posts_text_with_footer_and_broadcasts_without_targets(
    tmp_path, settings
):
    bus = EventBus()
    got = await collect(bus)

    async def fake_job(*a, **k):
        return RunResult(
            outcome=Outcome.STOPPED, text="half", elapsed_s=61, reason="hit 60m cap"
        )

    handler = ScheduledJobHandler(bus, RunSlots(3), settings, run_job_fn=fake_job)
    handler.register()
    await bus.publish(
        ScheduledEvent(job_name="x", prompt="p", working_directory=tmp_path)
    )
    await until(lambda: len(got) == 1)
    await bus.stop()
    assert got[0].chat_id == 0
    assert got[0].text == "half\n\n⚠️ stopped (hit 60m cap) · 1m1s"


async def test_handler_does_not_block_bus_so_jobs_run_concurrently(tmp_path, settings):
    bus = EventBus()
    got = await collect(bus)
    gate = asyncio.Event()
    started = []

    async def fake_job(prompt, *a, **k):
        started.append(prompt)
        await gate.wait()
        return RunResult(outcome=Outcome.DONE, text=prompt, elapsed_s=1)

    handler = ScheduledJobHandler(bus, RunSlots(3), settings, run_job_fn=fake_job)
    handler.register()
    for p in ("one", "two"):
        await bus.publish(
            ScheduledEvent(prompt=p, working_directory=tmp_path, target_chat_ids=[1])
        )
    await until(lambda: len(started) == 2)
    gate.set()
    await until(lambda: len(got) == 2)
    await bus.stop()


async def test_handler_survives_a_failing_job(tmp_path, settings):
    bus = EventBus()
    got = await collect(bus)

    async def fake_job(prompt, *a, **k):
        if prompt == "bad":
            raise RuntimeError("kaboom")
        return RunResult(outcome=Outcome.DONE, text="fine", elapsed_s=1)

    handler = ScheduledJobHandler(bus, RunSlots(3), settings, run_job_fn=fake_job)
    handler.register()
    for p in ("bad", "good"):
        await bus.publish(
            ScheduledEvent(prompt=p, working_directory=tmp_path, target_chat_ids=[1])
        )
    await until(lambda: len(got) == 1)
    await bus.stop()
    assert got[0].text == "fine"


# ------------------------------------------------------- Review Focus: no inject


async def test_scheduled_job_never_injects_into_interactive_run(tmp_path, settings):
    from src.harness.session_store import SessionStore
    from src.harness.topic_router import TopicContext, TopicRouter
    from src.storage.database import DatabaseManager

    db = DatabaseManager(f"sqlite:///{tmp_path / 'bot.db'}")
    await db.initialize()
    store = SessionStore(db, projects_root=tmp_path / "projects")
    topic_dir = tmp_path / "topic"
    topic_dir.mkdir()
    slots = RunSlots(2)
    gate = asyncio.Event()
    interactive = FakeSDKClient.factory(
        [(0, message_start()), ("call", gate.wait), (0, result("s-int", text="int"))]
    )
    runs: list[AgentRun] = []

    def make_run(options, on_progress):
        run = AgentRun(options, on_progress, settings, interactive, **FAST)
        runs.append(run)
        return run

    router = TopicRouter(
        fakes.FakeBot(), store, slots, settings,
        lambda inb: TopicContext(inb.topic_key, topic_dir, None),
        run_factory=make_run,
    )  # fmt: skip
    inbound = Inbound(
        topic_key="topic", chat_id=1, thread_id=None, message_id=1, user_id=1, text="hi"
    )
    interactive_task = asyncio.create_task(router.handle(inbound))
    await until(lambda: interactive.client.queries)
    live = router.active_run("topic")
    injected = []
    orig_inject = live.inject

    async def spy(content):
        injected.append(content)
        await orig_inject(content)

    live.inject = spy

    # Cap 2: one slot is held by the interactive run, so the job gets the other.
    job_factory = FakeSDKClient.factory(simple("job done"))
    job_res = await run_job(
        "cron", topic_dir, None, slots, settings,
        client_factory=job_factory, run_kwargs=FAST,
    )  # fmt: skip
    assert job_res.text == "job done"
    assert job_factory.client is not interactive.client
    assert injected == []
    assert len(interactive.client.queries) == 1  # nothing was injected

    # Fill the cap (interactive + one more), then a job must wait and a later
    # interactive acquirer is served before the already-waiting job.
    order = []
    holder = slots.acquire("interactive")
    await holder.__aenter__()
    assert slots.would_wait()

    async def job():
        await run_job(
            "cron2", topic_dir, None, slots, settings,
            client_factory=FakeSDKClient.factory(simple()), run_kwargs=FAST,
        )  # fmt: skip
        order.append("job")

    async def late_interactive():
        async with slots.acquire("interactive"):
            order.append("interactive")

    jt = asyncio.create_task(job())
    await asyncio.sleep(0.02)
    it = asyncio.create_task(late_interactive())
    await asyncio.sleep(0.02)
    assert order == []  # both waiting
    await holder.__aexit__(None, None, None)
    await until(lambda: order == ["interactive"])
    gate.set()  # interactive run ends, freeing a slot for the job
    await asyncio.gather(interactive_task, jt, it)
    assert order == ["interactive", "job"]
    assert not injected
    await db.close()


# ------------------------------------------------------- dispatcher -> options


async def test_dispatcher_model_override_reaches_options(tmp_path, settings):
    factory = FakeSDKClient.factory(simple("recap done"))

    async def job(prompt, cwd, agent, slots, settings_, model=None):
        return await run_job(
            prompt, cwd, agent, slots, settings_, model,
            client_factory=factory, run_kwargs=FAST,
        )  # fmt: skip

    runner = ClaudeTaskRunner(
        default_working_directory=tmp_path,
        board_db_path=Path("/tmp/board.db"),
        hermes_cli_path=Path("/tmp/cli.py"),
        slots=RunSlots(3),
        settings=settings,
        run_job_fn=job,
    )
    task = {
        "id": "t9",
        "title": "Recap",
        "payload": {"role": "recap", "runner_doc": "workflows/hermes-recap-runner.md"},
    }
    outcome = await runner.run(task)
    assert outcome.content == "recap done"
    assert factory.client.options.model == JUDGMENT_MODEL
