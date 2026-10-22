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
import re
import time
from pathlib import Path
from typing import Any

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeSDKClient,
    PermissionResultAllow,
    PermissionResultDeny,
    ToolUseBlock,
)

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


READ_ONLY_DENIAL = "live replay is read-only"
SAFE_TOOLS = {
    "Skill",
    "Read",
    "Glob",
    "Grep",
    "ToolSearch",
    "TodoWrite",
    "WebSearch",
    "WebFetch",
}
MCP_READ_WORDS = ("search", "query", "list", "get", "fetch", "read")
MCP_WRITE_WORDS = (
    "create",
    "send",
    "update",
    "delete",
    "draft",
    "post",
    "move",
    "add",
    "remove",
    "set",
    "write",
)
# Read-only shell: every ;/&&/| segment must be a gog read, the contacts index
# lookup, or a plain inspection command. No redirects (bar 2>&1), no
# substitution.
BASH_SEGMENT = re.compile(
    r"^\s*(?:"
    r"gog\b.*\b(?:calendar\s+(?:events|list|search)|drive\s+(?:search|ls|get)"
    r"|docs\s+export|sheets\s+(?:get|read))\b"
    r"|python3\s+scripts/contacts-lookup\.py\b"
    r"|(?:cat|ls|jq|head|tail|grep|date|pwd|echo)\b"
    r")"
)
BASH_UNSAFE = re.compile(r"`|\$\(|>|<|\bsudo\b")
BASH_SPLIT = re.compile(r"&&|\|\||;|\|")
INPUT_TRUNC = 200


def is_read_only(tool: str, tool_input: dict[str, Any]) -> bool:
    """Allowlist for the live replays; everything else is denied."""
    if tool in SAFE_TOOLS:
        return True
    if tool.startswith("mcp__"):
        name = tool.lower()
        return any(w in name for w in MCP_READ_WORDS) and not any(
            w in name.split("__", 2)[-1] for w in MCP_WRITE_WORDS
        )
    if tool == "Bash":
        cmd = str(tool_input.get("command", ""))
        cmd = cmd.replace("2>&1", "")
        if BASH_UNSAFE.search(cmd):
            return False
        return all(
            BASH_SEGMENT.match(seg) for seg in BASH_SPLIT.split(cmd) if seg.strip()
        )
    return False


def _make_tap(seen: list[dict[str, Any]]):
    """Client class recording every tool use (incl. auto-approved ones such as
    Skill, which never reach can_use_tool) into ``seen``."""

    class TapClient(ClaudeSDKClient):
        async def receive_messages(self):  # type: ignore[override]
            async for message in super().receive_messages():
                if isinstance(message, AssistantMessage):
                    for block in message.content or []:
                        if isinstance(block, ToolUseBlock):
                            seen.append(
                                {
                                    "tool": block.name,
                                    "input": json.dumps(
                                        block.input or {}, ensure_ascii=False
                                    )[:INPUT_TRUNC],
                                }
                            )
                yield message

    return TapClient


class ReadOnlyBridge:
    """Stands in for the AskBridge: supplies a read-only inner can_use_tool
    (the boundary guard still wraps it) and records every tool use."""

    def __init__(self) -> None:
        self.tool_uses: list[dict[str, Any]] = []  # every tool use (tap)
        self.denied: list[dict[str, Any]] = []

    async def try_answer_with_text(self, topic_key: str, text: str) -> bool:
        return False

    async def cancel_topic(self, topic_key: str) -> None:
        return None

    def make_can_use_tool(self, topic_key, chat_id, thread_id, run):
        async def can_use_tool(tool_name, tool_input, context):
            allowed = is_read_only(tool_name, tool_input)
            if allowed:
                return PermissionResultAllow()
            self.denied.append(
                {
                    "tool": tool_name,
                    "input": json.dumps(tool_input, ensure_ascii=False)[:INPUT_TRUNC],
                }
            )
            return PermissionResultDeny(message=READ_ONLY_DENIAL)

        return can_use_tool


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
    bridge = ReadOnlyBridge()
    RecordingRun.last_result = None

    def make_run(options, on_progress):
        return RecordingRun(options, on_progress, settings, _make_tap(bridge.tool_uses))

    router = TopicRouter(
        bot,
        store,
        RunSlots(settings.harness_concurrency_cap),
        settings,
        lambda inbound: TopicContext(inbound.topic_key, C22OS, None),
        bridge,  # read-only inner can_use_tool; the guard still wraps it
        security_validator=SecurityValidator(
            APPROVED, disable_security_patterns=settings.disable_security_patterns
        ),
        run_factory=make_run,
    )
    yield router, bot, bridge
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


def _contact_lookup_used(bridge: ReadOnlyBridge) -> tuple[bool, bool]:
    """(strict, loose). Strict: a Skill tool use naming contact-lookup.
    Loose (diagnostic only): any tool input mentioning it."""
    uses = bridge.tool_uses
    strict = any(
        u["tool"] in SKILL_TOOLS and "contact-lookup" in u["input"] for u in uses
    )
    loose = any("contact-lookup" in u["input"] for u in uses)
    return strict, loose


def _max_gap(bot: TimedBot, started: float, ended: float) -> float:
    """Longest silence between consecutive bot sends/edits over the run."""
    marks = [started, *[t for t in bot.times if started <= t <= ended], ended]
    return max(b - a for a, b in zip(marks, marks[1:]))


def _footer(bot: TimedBot) -> str:
    edits = bot.calls_for("edit_message_text")
    return edits[-1]["text"].splitlines()[0] if edits else ""


def _report(
    name: str, started: float, ended: float, bot: TimedBot, bridge: ReadOnlyBridge
) -> dict:
    res = RecordingRun.last_result
    strict, loose = _contact_lookup_used(bridge)
    row = {
        "replay": name,
        "outcome": res.outcome.value if res else None,
        "elapsed_s": round(res.elapsed_s, 1) if res else None,
        "wall_s": round(ended - started, 1),
        "tool_calls": res.tool_calls if res else None,
        "contact_lookup_skill": strict,
        "contact_lookup_mentioned": loose,  # diagnostic only
        "max_status_gap_s": round(_max_gap(bot, started, ended), 1),
        "footer": _footer(bot),
        "tools": [u["tool"] for u in bridge.tool_uses],
        "bash": [u["input"] for u in bridge.tool_uses if u["tool"] == "Bash"],
    }
    print("LIVE_ROW " + json.dumps(row, ensure_ascii=False))
    return row


def _assert_read_only(bridge: ReadOnlyBridge) -> None:
    denied = [(u["tool"], u["input"]) for u in bridge.denied]
    assert not denied, f"write/unlisted tool attempted: {denied}"


@pytest.mark.parametrize("run_no", [1, 2])
async def test_replay_person_lookup_1017(world, run_no):
    """10/17: 'Do I have Steve McGovern at Crystal Ventures' email?' thrashed."""
    router, bot, bridge = world
    started = time.monotonic()
    await router.handle(
        _inbound(
            f"live-1017-{run_no}",
            "Do I have Steve McGovern at Crystal Ventures' email?",
            10,
        )
    )
    ended = time.monotonic()
    row = _report("1017", started, ended, bot, bridge)
    _assert_read_only(bridge)
    res = RecordingRun.last_result
    assert res is not None and res.outcome is Outcome.DONE, row
    assert res.elapsed_s < LOOKUP_MAX_S, row
    assert ended - started < LOOKUP_MAX_S, row
    assert res.tool_calls <= LOOKUP_MAX_TOOLS, row
    assert row["contact_lookup_skill"], row
    assert row["max_status_gap_s"] <= MAX_STATUS_GAP_S, row
    assert row["footer"].startswith(FOOTERS), row


@pytest.mark.parametrize("run_no", [1, 2])
async def test_replay_resumed_multistep_1014(world, run_no):
    """10/14: a resumed session plus a multi-step ask went silent for ages."""
    router, bot, bridge = world
    topic = f"live-1014-{run_no}"
    await router.handle(_inbound(topic, "Reply with just the word: ready", 20))
    warm = RecordingRun.last_result
    assert warm is not None and warm.session_id, "warm-up produced no session"
    _assert_read_only(bridge)
    bot.calls.clear()
    bot.times.clear()
    bridge.tool_uses.clear()
    bridge.denied.clear()

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
    row = _report("1014", started, ended, bot, bridge)
    _assert_read_only(bridge)
    res = RecordingRun.last_result
    assert res is not None and res.outcome is Outcome.DONE, row
    assert row["max_status_gap_s"] <= MAX_STATUS_GAP_S, row
    assert row["footer"].startswith(FOOTERS), row
