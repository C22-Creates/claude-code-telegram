"""StatusMessage: coalesced status edits and answer delivery (no network)."""

import asyncio
from html.parser import HTMLParser
from types import SimpleNamespace

import pytest
from telegram.error import BadRequest, NetworkError, RetryAfter, TimedOut

from src.harness.format import split_message
from src.harness.status_message import NO_TEXT_REPLY, StatusMessage
from src.harness.types import Outcome, ProgressEvent, RunResult
from tests.harness.fakes import FakeBot

SETTINGS = SimpleNamespace(harness_status_edit_interval_seconds=5)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay
        await asyncio.sleep(0)


async def settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


def ev(kind="tool", tool="Bash", turn=1, elapsed=3.0, detail=""):
    return ProgressEvent(kind, tool, turn, elapsed, detail)


def make(bot=None):
    bot = bot or FakeBot()
    clock = Clock()
    sm = StatusMessage(bot, 10, 77, 55, SETTINGS, clock, sleep=clock.sleep)
    return sm, bot, clock


def done(text="hi", elapsed=252.0):
    return RunResult(outcome=Outcome.DONE, text=text, elapsed_s=elapsed)


async def test_open_replies_in_thread():
    sm, bot, _ = make()
    mid = await sm.open("starting")
    kw = bot.calls_for("send_message")[0]
    assert mid == 1
    assert kw["reply_parameters"].message_id == 55
    assert kw["message_thread_id"] == 77
    assert "parse_mode" not in kw


async def test_updates_coalesce_and_last_wins():
    sm, bot, clock = make()
    await sm.open("start")
    for i in range(20):
        clock.now = i * 0.15
        await sm.update(ev(tool=f"T{i}", turn=i, elapsed=i))
    await settle()
    edits = bot.calls_for("edit_message_text")
    assert len(edits) == 1
    assert edits[0]["text"] == "🔧 T19 · turn 19 · 19s"
    assert clock.sleeps  # waited out the remainder of the interval


async def test_edit_after_interval_is_immediate():
    sm, bot, clock = make()
    await sm.open("start")
    clock.now = 6
    await sm.update(ev())
    await settle()
    assert clock.sleeps == []
    assert len(bot.calls_for("edit_message_text")) == 1


async def test_heartbeat_mcp_and_note_render():
    sm, bot, clock = make()
    await sm.open("start")
    await sm.note("⏳ queued")
    await sm.update(ev("mcp_unavailable", None, detail="notion"))
    await sm.update(ev("heartbeat", None, turn=2, elapsed=1200, detail="20m"))
    await settle()
    assert bot.calls_for("edit_message_text")[-1]["text"] == (
        "⏳ queued\nworking · turn 2 · 20m0s\n⚠️ notion unavailable\n"
        "still working · 20m"
    )


async def test_tool_event_clears_note():
    sm, bot, clock = make()
    await sm.open("start")
    await sm.note("⏳ queued")
    clock.now = 10
    await sm.update(ev())
    await settle()
    assert bot.calls_for("edit_message_text")[-1]["text"] == "🔧 Bash · turn 1 · 3s"


@pytest.mark.parametrize(
    "exc", [RetryAfter(3), BadRequest("Message to edit not found"), NetworkError("x")]
)
async def test_edit_errors_swallowed_not_retried(exc):
    sm, bot, clock = make()
    await sm.open("start")
    bot.fail_next("edit_message_text", exc)
    clock.now = 10
    await sm.update(ev())
    await settle()
    assert len(bot.calls_for("edit_message_text")) == 1
    await sm.note("later")  # still usable


async def test_not_modified_is_success_and_equal_text_skipped():
    sm, bot, _ = make()
    await sm.open("start")
    bot.fail_next("edit_message_text", BadRequest("message is not modified"))
    await sm.note("same")
    await sm.note("same")
    assert len(bot.calls_for("edit_message_text")) == 1


async def test_finish_posts_answer_then_footer_and_no_late_edit():
    sm, bot, clock = make()
    await sm.open("start")
    await sm.update(ev())  # timer armed, sleeping
    await sm.finish(done("hello"))
    await settle()
    names = [m for m, _ in bot.calls]
    assert names == ["send_message", "send_message", "edit_message_text"]
    answer = bot.calls_for("send_message")[1]
    assert answer["parse_mode"] == "HTML"
    assert answer["reply_parameters"].message_id == 55
    assert answer["message_thread_id"] == 77
    assert bot.calls_for("edit_message_text")[0]["text"] == "✅ done · 4m12s"
    await sm.update(ev())
    await settle()
    assert len(bot.calls_for("edit_message_text")) == 1


class _TagChecker(HTMLParser):
    """Records tag balance problems in Telegram-style HTML."""

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.text = ""

    def handle_starttag(self, tag, attrs):
        self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack.pop() != tag:
            self.errors.append(tag)

    def handle_data(self, data):
        self.text += data


def parsed(html: str) -> _TagChecker:
    checker = _TagChecker()
    checker.feed(html)
    checker.close()
    assert checker.errors == [] and checker.stack == []
    return checker


async def test_long_answer_split_and_plaintext_fallback():
    sm, bot, _ = make()
    await sm.open("start")
    text = "\n".join("line %04d " % i + "x" * 80 for i in range(100))[:9000]
    bot.fail_next("send_message", BadRequest("Can't parse entities: bad"))
    await sm.finish(done(text))
    sends = bot.calls_for("send_message")[1:]
    assert len(sends) == 4  # failed HTML try + plain resend + 2 more chunks
    assert sends[0]["parse_mode"] == "HTML"
    assert "parse_mode" not in sends[1]
    assert sends[1]["text"] == text[: len(sends[1]["text"])]
    assert all(len(s["text"]) <= 4096 for s in sends)


async def test_special_chars_long_answer_chunks_are_well_formed():
    sm, bot, _ = make()
    await sm.open("start")
    text = " ".join(f"a<b>&c{i} x&y <tag> " for i in range(450))[:9000]
    await sm.finish(done(text))
    sends = bot.calls_for("send_message")[1:]
    assert 4 <= len(sends) <= 8  # escapes inflate the HTML, so more than 3
    assert all(len(s["text"]) <= 4096 for s in sends)
    delivered = "".join(parsed(s["text"]).text for s in sends)
    assert delivered.replace(" ", "").replace("\n", "") == text.replace(
        " ", ""
    ).replace("\n", "")


async def test_fence_spanning_chunk_boundary_is_closed_and_reopened():
    sm, bot, _ = make()
    await sm.open("start")
    code = "\n".join(f"print({i})  # <&>" for i in range(400))
    text = f"intro\n```python\n{code}\n```\noutro"
    assert len(text) > 3800
    await sm.finish(done(text))
    sends = bot.calls_for("send_message")[1:]
    assert len(sends) >= 2
    for s in sends:
        assert parsed(s["text"]).stack == []
        assert "```" not in s["text"]  # every fence converted to <pre>
    assert sends[0]["text"].count("<pre>") + sends[0]["text"].count("<pre") >= 1
    assert 'class="language-python"' in sends[1]["text"]


async def test_non_parse_bad_request_is_delivery_failure_not_retried():
    sm, bot, clock = make()
    await sm.open("start")
    bot.fail_next("send_message", BadRequest("Message is too long"))
    await sm.finish(done("a"))
    assert clock.sleeps == []
    assert len(bot.calls_for("send_message")) == 2
    assert "couldn't deliver" in bot.calls_for("edit_message_text")[0]["text"]


async def test_retry_after_sleep_is_capped():
    sm, bot, clock = make()
    await sm.open("start")
    bot.fail_next("send_message", RetryAfter(3600))
    await sm.finish(done("a"))
    assert clock.sleeps == [60]


async def test_update_during_inflight_edit_is_edited_after_interval():
    sm, bot, clock = make()
    await sm.open("start")
    gate, entered = asyncio.Event(), asyncio.Event()
    real_edit = bot.edit_message_text

    async def gated_edit(**kw):
        if not entered.is_set():
            entered.set()
            await gate.wait()
        return await real_edit(**kw)

    bot.edit_message_text = gated_edit
    clock.now = 10
    await sm.update(ev(tool="First", turn=1))
    await entered.wait()
    await sm.update(ev(tool="Second", turn=2))  # lands while edit is in flight
    gate.set()
    await settle()
    texts = [c["text"] for c in bot.calls_for("edit_message_text")]
    assert texts == ["🔧 First · turn 1 · 3s", "🔧 Second · turn 2 · 3s"]
    assert clock.sleeps == [5]  # second edit waited out the interval


async def test_nine_thousand_chars_three_sends():
    sm, bot, _ = make()
    await sm.open("start")
    await sm.finish(done("y" * 9000))
    assert len(bot.calls_for("send_message")) == 1 + 3


@pytest.mark.parametrize("outcome", [Outcome.STOPPED, Outcome.ERROR])
async def test_empty_stopped_or_error_posts_footer_only(outcome):
    sm, bot, _ = make()
    await sm.open("start")
    await sm.finish(RunResult(outcome=outcome, text="", elapsed_s=5, reason="stop"))
    assert len(bot.calls_for("send_message")) == 1
    assert bot.calls_for("edit_message_text")[0]["text"].startswith(
        "⚠️ stopped (stop)" if outcome is Outcome.STOPPED else "❌ error: stop"
    )


async def test_done_with_empty_text_posts_placeholder():
    sm, bot, _ = make()
    await sm.open("start")
    await sm.finish(done(""))
    assert bot.calls_for("send_message")[1]["text"] == NO_TEXT_REPLY


async def test_footer_edit_failure_sends_new_message():
    sm, bot, _ = make()
    await sm.open("start")
    bot.fail_next("edit_message_text", BadRequest("message to edit not found"))
    await sm.finish(done("a"))
    assert bot.calls_for("send_message")[-1]["text"] == "✅ done · 4m12s"


async def test_answer_send_retries_with_backoff():
    sm, bot, clock = make()
    await sm.open("start")
    bot.fail_next("send_message", NetworkError("n"))
    bot.fail_next("send_message", TimedOut())
    bot.fail_next("send_message", RetryAfter(7))
    await sm.finish(done("a"))
    assert clock.sleeps == [1, 2, 7]
    assert len(bot.calls_for("send_message")) == 1 + 4


async def test_answer_send_gives_up_after_three_retries():
    sm, bot, clock = make()
    await sm.open("start")
    for _ in range(4):
        bot.fail_next("send_message", NetworkError("n"))
    await sm.finish(done("a"))
    assert clock.sleeps == [1, 2, 4]
    footer_text = bot.calls_for("edit_message_text")[0]["text"]
    assert footer_text.startswith("✅ done") and "couldn't deliver" in footer_text


def test_split_message_prefers_newlines():
    parts = split_message("aaa\nbbb\nccc", limit=8)
    assert parts == ["aaa\nbbb", "ccc"]
    assert split_message("x" * 10, limit=4) == ["xxxx", "xxxx", "xx"]
