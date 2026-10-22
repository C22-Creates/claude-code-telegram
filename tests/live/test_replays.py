"""Live replays of the two incidents that motivated the harness rewrite.

These drive the real stack -- ``TopicRouter`` -> ``AgentRun`` -> real
``ClaudeSDKClient`` -> the configured MCP servers -- with a ``FakeBot`` standing
in for Telegram, so they record what Carl would have seen. They cost real
model calls, so they are marked ``live`` and skipped unless ``HARNESS_LIVE=1``.

Both replays are READ-ONLY lookups. A prompt that could write externally
(email, Notion, ClickUp, calendar) is a test-design bug.

Settings come from the live bot's env file (never printed); the MCP config is
read from the live checkout (read-only reference, not copied into the repo).
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import AssistantMessage, ClaudeSDKClient, ToolUseBlock

from src.config.settings import Settings
from src.harness.agent_run import AgentRun
from src.harness.run_slots import RunSlots
from src.harness.session_store import SessionStore
from src.harness.topic_router import TopicContext, TopicRouter
from src.harness.types import Inbound, Outcome, RunResult
from src.security.validators import SecurityValidator
from src.storage.database import DatabaseManager
from tests.harness.fakes import FakeBot

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("HARNESS_LIVE") != "1",
        reason="live replays call the real CLI; set HARNESS_LIVE=1",
    ),
]

LIVE_ENV = Path("/home/c22bot/c22os-telegram/.env")
LIVE_MCP = Path("/home/c22bot/c22os-telegram/config/mcp.json")
C22OS = Path("/home/c22bot/Projects/c22os")
APPROVED = Path("/home/c22bot/Projects")
CHAT_ID = 1
FOOTERS = ("✅ done", "⚠️ stopped", "❌ error")
SKILL_TOOLS = {"Skill", "SlashCommand"}
# Spec §2 targets.
LOOKUP_MAX_S = 60.0
LOOKUP_MAX_TOOLS = 8
MAX_STATUS_GAP_S = 300.0


class RecordingClient(ClaudeSDKClient):
    """Real client that also records every tool use (name + input)."""

    tool_uses: list[tuple[str, dict[str, Any]]] = []

    async def receive_messages(self):  # type: ignore[override]
        async for message in super().receive_messages():
            if isinstance(message, AssistantMessage):
                for block in message.content or []:
                    if isinstance(block, ToolUseBlock):
                        self.tool_uses.append((block.name, dict(block.input or {})))
            yield message


class TimedBot(FakeBot):
    """FakeBot that timestamps every call (monotonic seconds)."""

    def __init__(self) -> None:
        super().__init__()
        self.times: list[float] = []

    def _record(self, method: str, kwargs: dict[str, Any]) -> None:
        super()._record(method, kwargs)
        self.times.append(time.monotonic())


class RecordingRun(AgentRun):
    """AgentRun that keeps its ``RunResult`` (the router discards it)."""

    last_result: RunResult | None = None

    async def run(self, prompt):  # type: ignore[override]
        RecordingRun.last_result = await super().run(prompt)
        return RecordingRun.last_result


@pytest.fixture
async def world(tmp_path):
    """Router wired like ``harness.app._wire`` but with FakeBot + temp DB."""
    settings = Settings(
        _env_file=str(LIVE_ENV),
        approved_directory=APPROVED,
        database_url=f"sqlite:///{tmp_path / 'live.db'}",
        mcp_config_path=LIVE_MCP,
        enable_project_threads=False,
        projects_config_path=None,
    )
    db = DatabaseManager(settings.database_url)
    await db.initialize()
    store = SessionStore.from_settings(db, settings)
    bot = TimedBot()
    RecordingClient.tool_uses = []
    RecordingRun.last_result = None

    def make_run(options, on_progress):
        return RecordingRun(options, on_progress, settings, RecordingClient)

    router = TopicRouter(
        bot,
        store,
        RunSlots(settings.harness_concurrency_cap),
        settings,
        lambda inbound: TopicContext(inbound.topic_key, C22OS, None),
        None,  # no AskBridge: allow-all inner, guard still on
        security_validator=SecurityValidator(
            APPROVED, disable_security_patterns=settings.disable_security_patterns
        ),
        run_factory=make_run,
    )
    yield router, bot
    await db.close()


def _inbound(topic: str, text: str, message_id: int) -> Inbound:
    return Inbound(
        topic_key=topic,
        chat_id=CHAT_ID,
        thread_id=None,
        message_id=message_id,
        user_id=1,
        text=text,
    )


def _contact_lookup_used() -> tuple[bool, bool]:
    """(strict, loose): a Skill/SlashCommand tool use naming contact-lookup;
    any tool input mentioning it (e.g. a Read of the command file)."""
    uses = RecordingClient.tool_uses
    strict = any(
        n in SKILL_TOOLS and "contact-lookup" in json.dumps(i) for n, i in uses
    )
    loose = any("contact-lookup" in json.dumps(i) for _, i in uses)
    return strict, loose


def _max_gap(bot: TimedBot, started: float, ended: float) -> float:
    """Longest silence between consecutive bot sends/edits over the run."""
    marks = [started, *[t for t in bot.times if started <= t <= ended], ended]
    return max(b - a for a, b in zip(marks, marks[1:]))


def _footer(bot: TimedBot) -> str:
    edits = bot.calls_for("edit_message_text")
    return edits[-1]["text"] if edits else ""


def _report(name: str, started: float, ended: float, bot: TimedBot) -> dict:
    res = RecordingRun.last_result
    strict, loose = _contact_lookup_used()
    row = {
        "replay": name,
        "outcome": res.outcome.value if res else None,
        "elapsed_s": round(res.elapsed_s, 1) if res else None,
        "tool_calls": res.tool_calls if res else None,
        "contact_lookup_skill": strict,
        "contact_lookup_mentioned": loose,
        "max_status_gap_s": round(_max_gap(bot, started, ended), 1),
        "footer": _footer(bot),
        "tools": [n for n, _ in RecordingClient.tool_uses],
    }
    print("LIVE_ROW " + json.dumps(row, ensure_ascii=False))
    return row


@pytest.mark.parametrize("run_no", [1, 2])
async def test_replay_person_lookup_1017(world, run_no):
    """10/17: 'Do I have Steve McGovern at Crystal Ventures' email?' thrashed."""
    router, bot = world
    started = time.monotonic()
    await router.handle(
        _inbound(
            f"live-1017-{run_no}",
            "Do I have Steve McGovern at Crystal Ventures' email?",
            10,
        )
    )
    ended = time.monotonic()
    row = _report("1017", started, ended, bot)
    res = RecordingRun.last_result
    assert res is not None and res.outcome is Outcome.DONE, row
    assert res.elapsed_s < LOOKUP_MAX_S, row
    assert res.tool_calls <= LOOKUP_MAX_TOOLS, row
    assert row["contact_lookup_skill"] or row["contact_lookup_mentioned"], row


@pytest.mark.parametrize("run_no", [1, 2])
async def test_replay_resumed_multistep_1014(world, run_no):
    """10/14: a resumed session plus a multi-step ask went silent for ages."""
    router, bot = world
    topic = f"live-1014-{run_no}"
    await router.handle(_inbound(topic, "Reply with just the word: ready", 20))
    warm = RecordingRun.last_result
    assert warm is not None and warm.session_id, "warm-up produced no session"
    bot.calls.clear()
    bot.times.clear()
    RecordingClient.tool_uses = []

    started = time.monotonic()
    await router.handle(
        _inbound(
            topic,
            "What meetings does Carl have tomorrow across his calendars, "
            "and who is the first one with? Read-only; do not book or change "
            "anything.",
            21,
        )
    )
    ended = time.monotonic()
    row = _report("1014", started, ended, bot)
    assert row["max_status_gap_s"] <= MAX_STATUS_GAP_S, row
    assert row["footer"].startswith(FOOTERS), row
