"""Pin the behavior of the test fakes so later tasks can rely on them."""

import asyncio

import pytest
from claude_agent_sdk.types import ResultMessage

from tests.harness import fakes
from tests.harness.fakes import FakeBot, FakeSDKClient


async def _no_sleep(_s: float) -> None:
    await asyncio.sleep(0)


async def _take(client: FakeSDKClient, n: int, timeout: float = 1.0) -> list:
    out: list = []

    async def run() -> None:
        async for m in client.receive_messages():
            out.append(m)
            if len(out) >= n:
                return

    await asyncio.wait_for(run(), timeout)
    return out


async def test_script_replays_in_order_and_stays_open():
    c = FakeSDKClient(
        [(0, fakes.tool_use("Bash")), (0.01, fakes.text("hi")), (0, fakes.result())],
        sleep=_no_sleep,
    )
    msgs = await _take(c, 3)
    assert [type(m).__name__ for m in msgs] == [
        "AssistantMessage",
        "AssistantMessage",
        "ResultMessage",
    ]
    # Stream stays open after the script is exhausted.
    with pytest.raises(asyncio.TimeoutError):
        await _take(c, 4, timeout=0.05)


async def test_disconnect_ends_stream():
    c = FakeSDKClient([], sleep=_no_sleep)
    task = asyncio.create_task(_take(c, 1, timeout=1))
    await asyncio.sleep(0.01)
    await c.disconnect()
    assert await task == []  # iterator ended cleanly
    assert c.disconnected


async def test_await_query_blocks_until_query():
    c = FakeSDKClient(
        [(0, fakes.text("a")), ("await_query",), (0, fakes.text("b"))],
        sleep=_no_sleep,
    )
    got: list = []

    async def run() -> None:
        async for m in c.receive_messages():
            got.append(m)

    task = asyncio.create_task(run())
    await asyncio.sleep(0.05)
    assert len(got) == 1
    await c.query("more")
    await asyncio.sleep(0.05)
    assert len(got) == 2
    assert c.queries == ["more"]
    await c.disconnect()
    await task


async def test_interrupt_ends_turn_with_error_result_and_skips_rest():
    gate = asyncio.Event()

    async def slow(_s: float) -> None:
        await gate.wait()

    c = FakeSDKClient(
        [
            (0, fakes.text("a")),
            (30, fakes.text("never")),
            (0, fakes.result()),
            (0, fakes.text("next turn")),
        ],
        sleep=lambda s: slow(s) if s >= 30 else _no_sleep(s),
    )
    out: list = []

    async def run() -> None:
        async for m in c.receive_messages():
            out.append(m)

    task = asyncio.create_task(run())
    await asyncio.sleep(0.05)
    await c.interrupt()
    await asyncio.sleep(0.05)
    assert c.interrupted == 1
    assert isinstance(out[1], ResultMessage)
    assert out[1].is_error is True and out[1].result is None
    assert out[2].content[0].text == "next turn"
    await c.disconnect()
    await task


async def test_hung_interrupt_never_returns_or_ends_turn():
    c = FakeSDKClient([(30, fakes.text("x"))], sleep=lambda s: asyncio.sleep(30))
    c.hang_interrupt = True
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(c.interrupt(), 0.05)
    assert c.interrupted == 1


async def test_receive_raises_for_crash():
    c = FakeSDKClient(raise_on_receive=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        await _take(c, 1)
    c2 = FakeSDKClient(
        [(0, fakes.text("a")), ("raise", ValueError("mid"))], sleep=_no_sleep
    )
    with pytest.raises(ValueError, match="mid"):
        await _take(c2, 5)


def test_factory_returns_same_client_and_records_options():
    build = FakeSDKClient.factory([])
    c = build("opts")
    assert c is build.client and c.options == "opts"


def test_builders_shape():
    assert fakes.tool_use("Read").content[0].name == "Read"
    assert fakes.message_delta_end_turn().event["delta"]["stop_reason"] == "end_turn"
    assert fakes.message_start("p").parent_tool_use_id == "p"
    assert fakes.result(session_id="s9").session_id == "s9"
    init = fakes.system_init(["notion", {"name": "x", "status": "failed"}])
    assert init.subtype == "init"
    assert init.data["mcp_servers"] == [
        {"name": "notion", "status": "connected"},
        {"name": "x", "status": "failed"},
    ]


async def test_bot_records_calls_and_increments_ids():
    bot = FakeBot()
    m1 = await bot.send_message(chat_id=1, text="a", message_thread_id=7)
    m2 = await bot.send_message(chat_id=1, text="b", parse_mode="HTML")
    assert (m1.message_id, m2.message_id) == (1, 2)
    e = await bot.edit_message_text(chat_id=1, message_id=1, text="a2")
    assert e.message_id == 1
    assert await bot.set_message_reaction(chat_id=1, message_id=1, reaction=None)
    assert await bot.answer_callback_query(callback_query_id="q")
    assert [m for m, _ in bot.calls] == [
        "send_message",
        "send_message",
        "edit_message_text",
        "set_message_reaction",
        "answer_callback_query",
    ]
    assert bot.calls[0][1] == {"chat_id": 1, "text": "a", "message_thread_id": 7}


async def test_bot_fail_next_raises_once_then_recovers():
    bot = FakeBot()
    bot.fail_next("edit_message_text", RuntimeError("flood"))
    with pytest.raises(RuntimeError, match="flood"):
        await bot.edit_message_text(chat_id=1, message_id=1, text="x")
    await bot.edit_message_text(chat_id=1, message_id=1, text="x")
    assert len(bot.calls_for("edit_message_text")) == 2


async def test_bot_rejects_missing_required_args():
    with pytest.raises(TypeError):
        await FakeBot().send_message(chat_id=1)


async def test_bot_rejects_unknown_kwargs():
    with pytest.raises(TypeError, match="unexpected"):
        await FakeBot().send_message(chat_id=1, text="a", bogus=True)


async def test_bot_edit_requires_message_target():
    bot = FakeBot()
    with pytest.raises(TypeError):
        await bot.edit_message_text(text="x")
    with pytest.raises(TypeError):
        await bot.edit_message_text(chat_id=1, text="x")
    await bot.edit_message_text(inline_message_id="i", text="x")
    await bot.edit_message_text(chat_id=1, message_id=2, text="x")


async def test_await_query_ignores_earlier_query_mid_script():
    c = FakeSDKClient(
        [
            (0, fakes.text("a")),
            (0, fakes.result()),
            ("await_query",),
            (0, fakes.text("b")),
        ],
        sleep=_no_sleep,
    )
    await c.query("initial")
    got: list = []

    async def run() -> None:
        async for m in c.receive_messages():
            got.append(m)

    task = asyncio.create_task(run())
    await asyncio.sleep(0.05)
    assert len(got) == 2  # blocked despite the initial query
    await c.query("second")
    await asyncio.sleep(0.05)
    assert len(got) == 3
    await c.disconnect()
    await task


async def test_leading_await_query_consumes_initial_query():
    c = FakeSDKClient([("await_query",), (0, fakes.text("a"))], sleep=_no_sleep)
    await c.query("initial")
    assert len(await _take(c, 1)) == 1


async def test_close_beats_interrupt_and_step_when_simultaneous():
    c = FakeSDKClient([(1, fakes.text("a"))], sleep=_no_sleep)
    await c.interrupt()
    await c.disconnect()
    out = [m async for m in c.receive_messages()]
    assert out == []


def test_tool_use_ids_unique():
    assert fakes.tool_use("Bash").content[0].id != fakes.tool_use("Bash").content[0].id


async def test_call_step_runs_between_messages_sync_or_async():
    seen: list = []

    async def acall() -> None:
        seen.append(("async", len(got)))

    got: list = []
    c = FakeSDKClient(
        [
            (0, fakes.text("a")),
            ("call", lambda: seen.append(("sync", len(got)))),
            ("call", acall),
            (0, fakes.text("b")),
        ],
        sleep=_no_sleep,
    )
    async for m in c.receive_messages():
        got.append(m)
        if len(got) == 2:
            break
    # Both calls ran after "a" was consumed and before "b" was yielded.
    assert seen == [("sync", 1), ("async", 1)]
