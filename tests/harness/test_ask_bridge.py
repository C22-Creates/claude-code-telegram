"""Tests for AskBridge: buttons, typed answers, expiry, watchdog, router wiring."""

import asyncio
import time
from types import SimpleNamespace

import pytest
from claude_agent_sdk import (
    ClaudeAgentOptions,
    PermissionResultAllow,
    PermissionResultDeny,
)

from src.config.settings import Settings
from src.harness.agent_run import AgentRun
from src.harness.ask_bridge import AskBridge
from src.harness.run_slots import RunSlots
from src.harness.session_store import SessionStore
from src.harness.topic_router import TopicContext, TopicRouter
from src.harness.types import Inbound, Outcome
from src.storage.database import DatabaseManager
from tests.harness import fakes
from tests.harness.fakes import FakeBot, FakeSDKClient
from tests.harness.test_agent_run import FakeClock

CHAT, THREAD, TOPIC = -1001, 7, "t"
CTX = SimpleNamespace()
QUESTION = {
    "question": "Which color?",
    "header": "Color",
    "options": [
        {"label": "Red", "description": "warm"},
        {"label": "Blue", "description": "cool"},
    ],
    "multiSelect": False,
}
ASK_INPUT = {"questions": [QUESTION]}


@pytest.fixture
def settings(tmp_path):
    return Settings(
        telegram_bot_token="t",
        telegram_bot_username="b",
        approved_directory=str(tmp_path),
    )


class StubRun:
    """Records watchdog/interrupt calls."""

    def __init__(self) -> None:
        self.log: list[str] = []

    def pause_watchdog(self) -> None:
        self.log.append("pause")

    def resume_watchdog(self) -> None:
        self.log.append("resume")

    async def interrupt(self, reason: str) -> None:
        self.log.append(f"interrupt:{reason}")


def tap(data: str, qid: str = "cq1") -> SimpleNamespace:
    return SimpleNamespace(id=qid, data=data)


async def until(pred, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.001)


def buttons(bot: FakeBot, i: int = -1) -> list[tuple[str, str]]:
    markup = bot.calls_for("send_message")[i]["reply_markup"]
    return [(b.text, b.callback_data) for row in markup.inline_keyboard for b in row]


@pytest.fixture
def bot():
    return FakeBot()


@pytest.fixture
def bridge(bot, settings):
    return AskBridge(bot, settings, expiry_seconds=60)


async def start_ask(bridge, run=None, tool_input=ASK_INPUT):
    run = run or StubRun()
    cb = bridge.make_can_use_tool(TOPIC, CHAT, THREAD, run)
    task = asyncio.create_task(cb("AskUserQuestion", tool_input, CTX))
    return run, task


async def test_other_tools_allowed_immediately(bridge, bot):
    cb = bridge.make_can_use_tool(TOPIC, CHAT, THREAD, StubRun())
    res = await cb("Bash", {"command": "ls"}, CTX)
    assert isinstance(res, PermissionResultAllow)
    assert res.updated_input is None
    assert bot.calls == []


async def test_question_sent_with_option_buttons_and_other(bridge, bot):
    run, task = await start_ask(bridge)
    await until(lambda: bot.calls_for("send_message"))
    sent = bot.calls_for("send_message")[0]
    assert sent["chat_id"] == CHAT and sent["message_thread_id"] == THREAD
    assert "Which color?" in sent["text"]
    texts = [t for t, _ in buttons(bot)]
    assert texts == ["Red", "Blue", "Other…"]
    for _, data in buttons(bot):
        assert data.startswith("ask:") and len(data.encode()) <= 64
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def test_button_tap_answers_and_removes_keyboard(bridge, bot):
    run, task = await start_ask(bridge)
    await until(lambda: bot.calls_for("send_message"))
    await bridge.on_callback(tap(buttons(bot)[1][1]))  # Blue
    res = await task
    assert isinstance(res, PermissionResultAllow)
    assert res.updated_input == {**ASK_INPUT, "answers": {"Which color?": "Blue"}}
    edit = bot.calls_for("edit_message_text")[-1]
    assert edit["text"] == "✅ Blue" and "reply_markup" not in edit
    assert bot.calls_for("answer_callback_query")[0]["callback_query_id"] == "cq1"
    assert run.log == ["pause", "resume"]


async def test_text_reply_answers_pending_question(bridge, bot):
    run, task = await start_ask(bridge)
    await until(lambda: bot.calls_for("send_message"))
    assert await bridge.try_answer_with_text(TOPIC, "blue") is True
    res = await task
    assert res.updated_input["answers"] == {"Which color?": "blue"}
    assert bot.calls_for("edit_message_text")[-1]["text"] == "✅ blue"
    # nothing pending any more: the router would inject this one
    assert await bridge.try_answer_with_text(TOPIC, "again") is False


async def test_text_reply_without_pending_question_is_not_consumed(bridge):
    assert await bridge.try_answer_with_text(TOPIC, "hi") is False
    assert await bridge.try_answer_with_text("other-topic", "hi") is False


async def test_other_button_prompts_then_takes_next_text(bridge, bot):
    run, task = await start_ask(bridge)
    await until(lambda: bot.calls_for("send_message"))
    await bridge.on_callback(tap(buttons(bot)[2][1]))  # Other…
    assert "Type your answer" in bot.calls_for("edit_message_text")[-1]["text"]
    assert not task.done()
    assert await bridge.try_answer_with_text(TOPIC, "green") is True
    res = await task
    assert res.updated_input["answers"] == {"Which color?": "green"}


async def test_stale_button_answers_expired_and_changes_nothing(bridge, bot):
    run, task = await start_ask(bridge)
    await until(lambda: bot.calls_for("send_message"))
    await bridge.on_callback(tap("ask:deadbeef:0", "cq9"))
    await bridge.on_callback(tap("garbage", "cq10"))
    answers = bot.calls_for("answer_callback_query")
    assert [(a["callback_query_id"], a["text"]) for a in answers] == [
        ("cq9", "expired"),
        ("cq10", "expired"),
    ]
    assert bot.calls_for("edit_message_text") == []
    assert not task.done()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def test_tap_after_answer_is_expired(bridge, bot):
    run, task = await start_ask(bridge)
    await until(lambda: bot.calls_for("send_message"))
    data = buttons(bot)[0][1]
    await bridge.on_callback(tap(data))
    await task
    await bridge.on_callback(tap(data, "cq2"))
    assert bot.calls_for("answer_callback_query")[-1]["text"] == "expired"


async def test_expiry_denies_interrupts_and_edits(bot, settings):
    bridge = AskBridge(bot, settings, expiry_seconds=0.02)
    run, task = await start_ask(bridge)
    res = await task
    assert isinstance(res, PermissionResultDeny)
    assert (res.message, res.interrupt) == ("no answer", True)
    assert bot.calls_for("edit_message_text")[-1]["text"] == "no answer, stopped"
    assert run.log == ["pause", "interrupt:no answer", "resume"]
    assert await bridge.try_answer_with_text(TOPIC, "late") is False


async def test_default_expiry_is_settings_value(bot, settings):
    assert settings.harness_ask_expiry_seconds == 1800
    assert AskBridge(bot, settings)._expiry == 1800


async def test_multiple_questions_asked_in_sequence(bridge, bot):
    q2 = {**QUESTION, "question": "Which size?", "options": [{"label": "S"}]}
    run, task = await start_ask(bridge, tool_input={"questions": [QUESTION, q2]})
    await until(lambda: len(bot.calls_for("send_message")) == 1)
    await bridge.on_callback(tap(buttons(bot, 0)[0][1]))  # Red
    await until(lambda: len(bot.calls_for("send_message")) == 2)
    await bridge.on_callback(tap(buttons(bot, 1)[0][1]))  # S
    res = await task
    assert res.updated_input["answers"] == {"Which color?": "Red", "Which size?": "S"}
    assert run.log == ["pause", "resume"]


async def test_second_ask_in_topic_waits_behind_first(bridge, bot):
    run1, t1 = await start_ask(bridge)
    await until(lambda: len(bot.calls_for("send_message")) == 1)
    run2, t2 = await start_ask(bridge)
    await asyncio.sleep(0.02)
    assert len(bot.calls_for("send_message")) == 1  # second not sent yet
    await bridge.try_answer_with_text(TOPIC, "one")
    await until(lambda: len(bot.calls_for("send_message")) == 2)
    await bridge.try_answer_with_text(TOPIC, "two")
    assert (await t1).updated_input["answers"] == {"Which color?": "one"}
    assert (await t2).updated_input["answers"] == {"Which color?": "two"}


async def test_send_failure_denies_without_interrupt(bridge, bot):
    bot.fail_next("send_message", RuntimeError("telegram down"))
    run, task = await start_ask(bridge)
    res = await task
    assert isinstance(res, PermissionResultDeny) and res.interrupt is False
    assert run.log == ["pause", "resume"]


async def test_cancel_while_pending_resumes_watchdog_and_clears(bridge, bot):
    run, task = await start_ask(bridge)
    await until(lambda: bot.calls_for("send_message"))
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert run.log == ["pause", "resume"]
    assert bot.calls_for("edit_message_text")[-1]["text"] == "cancelled"
    assert await bridge.try_answer_with_text(TOPIC, "x") is False


# ------------------------------------------------------- Review Focus 1: watchdog


async def test_watchdog_paused_while_question_pending(settings, bot):
    """A 12-minute pending ask is not a stall; the stall clock restarts at answer."""
    clock = FakeClock()
    bridge = AskBridge(bot, settings, expiry_seconds=3600)
    holder: dict = {}

    async def ask() -> None:
        cb = bridge.make_can_use_tool(TOPIC, CHAT, THREAD, holder["run"])
        task = asyncio.create_task(cb("AskUserQuestion", ASK_INPUT, CTX))
        await until(lambda: bot.calls_for("send_message"))
        await clock.wait(720)  # twelve minutes pass with the question open
        assert holder["run"].is_active  # the 5-min stall did not trip
        holder["answered_at"] = clock.now
        await bridge.on_callback(tap(buttons(bot)[0][1]))
        await task

    factory = FakeSDKClient.factory(
        [
            (0, fakes.message_start()),
            (0, fakes.tool_use("AskUserQuestion")),
            ("call", ask),
        ],
        sleep=clock.wait,
    )
    run = AgentRun(
        ClaudeAgentOptions(),
        lambda event: asyncio.sleep(0),
        settings,
        factory,
        clock=clock,
        sleep=clock.tick,
        interrupt_timeout=0.05,
        cleanup_timeout=0.05,
        progress_timeout=0.05,
    )
    holder["run"] = run
    res = await run.run("x")
    assert res.outcome is Outcome.STOPPED
    assert res.reason == "stalled on AskUserQuestion"
    # Stalled 5 minutes after the answer, not 5 minutes after the ask.
    assert holder["answered_at"] >= 720
    assert 300 <= res.elapsed_s - holder["answered_at"] <= 315


# ------------------------------------------------------------ router wiring


async def test_router_routes_typed_message_to_pending_question(tmp_path, settings, bot):
    db = DatabaseManager(f"sqlite:///{tmp_path / 'bot.db'}")
    await db.initialize()
    store = SessionStore(db, projects_root=tmp_path / "projects")
    bridge = AskBridge(bot, settings, expiry_seconds=60)
    captured: dict = {}

    async def ask() -> None:
        captured["task"] = asyncio.create_task(
            captured["cb"]("AskUserQuestion", ASK_INPUT, CTX)
        )
        captured["answer"] = await captured["task"]

    factory = FakeSDKClient.factory(
        [
            (0, fakes.message_start()),
            ("call", ask),
            (0, fakes.text("You chose blue.")),
            (0, fakes.result(text="You chose blue.")),
        ]
    )

    def make_run(options, on_progress):
        captured["cb"] = options.can_use_tool
        return AgentRun(
            options,
            on_progress,
            settings,
            factory,
            interrupt_timeout=0.05,
            cleanup_timeout=0.05,
            progress_timeout=0.05,
        )

    cwd = tmp_path / TOPIC
    cwd.mkdir()
    router = TopicRouter(
        bot,
        store,
        RunSlots(3),
        settings,
        lambda inbound: TopicContext(inbound.topic_key, cwd, None),
        ask_bridge=bridge,
        run_factory=make_run,
    )

    def inbound(text: str, mid: int) -> Inbound:
        return Inbound(TOPIC, CHAT, THREAD, mid, 1, text)

    first = asyncio.create_task(router.handle(inbound("pick a color", 100)))
    await until(
        lambda: any("reply_markup" in kw for kw in bot.calls_for("send_message"))
    )
    await router.handle(inbound("blue", 101))
    await asyncio.wait_for(first, 5)

    assert captured["answer"].updated_input["answers"] == {"Which color?": "blue"}
    assert factory.client.queries == ["pick a color"]  # "blue" was not injected
    await db.close()


# ---------------------------------------------------------- run-end cleanup


async def test_cancel_topic_cleans_ask_whose_task_is_not_cancelled(bridge, bot):
    run1, old = await start_ask(bridge)
    await until(lambda: len(bot.calls_for("send_message")) == 1)
    run1b, queued = await start_ask(bridge)  # queued behind the dead run's ask
    await asyncio.sleep(0)
    await bridge.cancel_topic(TOPIC)  # the SDK never cancelled `old`
    assert bot.calls_for("edit_message_text")[-1]["text"] == "stopped"
    assert await bridge.try_answer_with_text(TOPIC, "swallowed?") is False
    run2, new = await start_ask(bridge)  # the next run asks immediately
    await until(lambda: len(bot.calls_for("send_message")) == 2)
    assert isinstance(await old, PermissionResultDeny)
    assert isinstance(await queued, PermissionResultDeny)  # dead run's queue drops
    assert len(bot.calls_for("send_message")) == 2
    assert await bridge.try_answer_with_text(TOPIC, "blue") is True
    assert (await new).updated_input["answers"] == {"Which color?": "blue"}


async def test_cancel_topic_without_pending_is_noop(bridge, bot):
    await bridge.cancel_topic(TOPIC)
    assert bot.calls == []


async def test_out_of_range_button_index_is_expired(bridge, bot):
    run, task = await start_ask(bridge)
    await until(lambda: bot.calls_for("send_message"))
    ask_id = buttons(bot)[0][1].split(":")[1]
    await bridge.on_callback(tap(f"ask:{ask_id}:99"))
    assert bot.calls_for("answer_callback_query")[-1]["text"] == "expired"
    assert not task.done()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def test_router_cancel_with_pending_question_frees_next_message(
    tmp_path, settings, bot
):
    db = DatabaseManager(f"sqlite:///{tmp_path / 'bot.db'}")
    await db.initialize()
    store = SessionStore(db, projects_root=tmp_path / "projects")
    bridge = AskBridge(bot, settings, expiry_seconds=60)
    captured: dict = {}

    def ask() -> None:  # fire and forget: nothing ever cancels this task
        captured["task"] = asyncio.create_task(
            captured["cb"]("AskUserQuestion", ASK_INPUT, CTX)
        )

    factories = [
        FakeSDKClient.factory([(0, fakes.message_start()), ("call", ask)]),
        FakeSDKClient.factory(
            [(0, fakes.message_start()), (0, fakes.result(text="fresh run"))]
        ),
    ]

    def make_run(options, on_progress):
        captured["cb"] = options.can_use_tool
        return AgentRun(
            options,
            on_progress,
            settings,
            factories.pop(0),
            interrupt_timeout=0.05,
            cleanup_timeout=0.05,
            progress_timeout=0.05,
        )

    cwd = tmp_path / TOPIC
    cwd.mkdir()
    router = TopicRouter(
        bot,
        store,
        RunSlots(3),
        settings,
        lambda inbound: TopicContext(inbound.topic_key, cwd, None),
        ask_bridge=bridge,
        run_factory=make_run,
    )

    def inbound(text: str, mid: int) -> Inbound:
        return Inbound(TOPIC, CHAT, THREAD, mid, 1, text)

    first = asyncio.create_task(router.handle(inbound("go", 100)))
    await until(
        lambda: any("reply_markup" in kw for kw in bot.calls_for("send_message"))
    )
    assert await router.cancel(TOPIC) is True
    await asyncio.wait_for(first, 5)
    assert bot.calls_for("edit_message_text")[-1]["text"] != "✅ next"

    await asyncio.wait_for(router.handle(inbound("next", 101)), 5)
    assert factories == []  # "next" started a second run instead of being eaten
    assert isinstance(await asyncio.wait_for(captured["task"], 5), PermissionResultDeny)
    await db.close()
