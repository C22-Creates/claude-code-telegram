"""AskBridge: the agent's AskUserQuestion becomes Telegram inline buttons.

The SDK awaits ``can_use_tool`` for AskUserQuestion, so the callback can simply
block until Carl taps a button (or types a reply) and then return the answer.
While it blocks, the run's stall watchdog is paused: a person thinking for ten
minutes is not a hung tool. The 60-minute run cap still applies.

State is in memory, one pending question per topic. A second concurrent ask in
the same topic queues behind the first. Wiring (Task 8): the callback-query
handler for ``ask:`` data calls ``on_callback``; ``TopicRouter.handle`` calls
``try_answer_with_text`` before injecting a typed message.

multiSelect questions are v1-simplified: a single tap picks one option.
"""

import asyncio
import secrets
from dataclasses import dataclass, field
from typing import Any

import structlog
from claude_agent_sdk import (
    CanUseTool,
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

logger = structlog.get_logger()

ASK_TOOL = "AskUserQuestion"
CALLBACK_PREFIX = "ask"
OTHER_LABEL = "Other…"
EXPIRED_NOTICE = "expired"
EXPIRY_TEXT = "no answer, stopped"
DENY_MESSAGE = "no answer"
TYPE_PROMPT = "✍️ Type your answer"
STOPPED_TEXT = "stopped"
_STOPPED = object()  # future result: the run ended while the question was open


@dataclass
class _Pending:
    """One open question awaiting an answer."""

    ask_id: str
    topic_key: str
    chat_id: int
    message_id: int
    question: str
    options: list[str]
    future: "asyncio.Future[Any]" = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )


class AskBridge:
    """Sends agent questions to Telegram and resolves them from taps or text.

    Args:
        bot: object with PTB's async ``send_message``/``edit_message_text``/
            ``answer_callback_query``.
        settings: provides ``harness_ask_expiry_seconds``.
        expiry_seconds: override of the settings value (tests use tiny values).
    """

    def __init__(
        self, bot: Any, settings: Any, *, expiry_seconds: float | None = None
    ) -> None:
        self._bot = bot
        self._expiry = (
            expiry_seconds
            if expiry_seconds is not None
            else settings.harness_ask_expiry_seconds
        )
        self._by_id: dict[str, _Pending] = {}
        self._by_topic: dict[str, _Pending] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._generation: dict[str, int] = {}

    # ----------------------------------------------------------- public API

    def make_can_use_tool(
        self, topic_key: str, chat_id: int, thread_id: int | None, run: Any
    ) -> CanUseTool:
        """Build the run's permission callback (``run`` is a ``RunRef``/run)."""

        async def can_use_tool(
            tool_name: str, tool_input: dict[str, Any], context: ToolPermissionContext
        ) -> PermissionResultAllow | PermissionResultDeny:
            if tool_name != ASK_TOOL:
                return PermissionResultAllow()
            return await self._ask_all(topic_key, chat_id, thread_id, run, tool_input)

        return can_use_tool

    async def try_answer_with_text(self, topic_key: str, text: str) -> bool:
        """Use *text* as the answer to the topic's pending question, if any."""
        pending = self._by_topic.get(topic_key)
        if pending is None or pending.future.done():
            return False
        pending.future.set_result(text)
        return True

    async def cancel_topic(self, topic_key: str) -> None:
        """Drop the topic's pending question because its run ended.

        Does not rely on the SDK cancelling the in-flight callback: the pending
        state is removed here, the callback is woken with a stop marker, and a
        generation bump makes asks queued behind it deny instead of sending.
        """
        self._generation[topic_key] = self._generation.get(topic_key, 0) + 1
        pending = self._by_topic.pop(topic_key, None)
        if pending is None:
            return
        self._by_id.pop(pending.ask_id, None)
        if not pending.future.done():
            pending.future.set_result(_STOPPED)
        await self._edit(pending, STOPPED_TEXT)

    async def on_callback(self, callback_query: Any) -> None:
        """Handle a tap on an ``ask:{ask_id}:{idx}`` button."""
        pending, idx = self._parse(callback_query.data)
        if (
            pending is None
            or idx is None
            or idx > len(pending.options)
            or pending.future.done()
        ):
            await self._answer_query(callback_query.id, EXPIRED_NOTICE)
            return
        await self._answer_query(callback_query.id)
        if idx == len(pending.options):  # the "Other…" button
            await self._edit(pending, f"{TYPE_PROMPT}\n\n{pending.question}")
            return
        pending.future.set_result(pending.options[idx])

    # ------------------------------------------------------------- the flow

    async def _ask_all(
        self,
        topic_key: str,
        chat_id: int,
        thread_id: int | None,
        run: Any,
        tool_input: dict[str, Any],
    ) -> PermissionResultAllow | PermissionResultDeny:
        lock = self._locks.setdefault(topic_key, asyncio.Lock())
        generation = self._generation.get(topic_key, 0)
        async with lock:  # a second ask in this topic waits its turn
            if self._generation.get(topic_key, 0) != generation:
                return PermissionResultDeny(message=STOPPED_TEXT)  # run ended
            run.pause_watchdog()
            try:
                answers: dict[str, str] = {}
                for q in tool_input.get("questions") or []:
                    answer = await self._ask_one(topic_key, chat_id, thread_id, q)
                    if answer is _STOPPED:
                        return PermissionResultDeny(message=STOPPED_TEXT)
                    if answer is None:
                        await run.interrupt(DENY_MESSAGE)  # defensive
                        return PermissionResultDeny(
                            message=DENY_MESSAGE, interrupt=True
                        )
                    answers[q.get("question", "")] = answer
                return PermissionResultAllow(
                    updated_input={**tool_input, "answers": answers}
                )
            except Exception as exc:  # noqa: BLE001 - tell the agent, don't crash
                logger.exception("harness.ask_failed", topic=topic_key)
                return PermissionResultDeny(message=f"could not ask: {exc}")
            finally:
                run.resume_watchdog()

    async def _ask_one(
        self, topic_key: str, chat_id: int, thread_id: int | None, q: dict[str, Any]
    ) -> Any:
        """Ask one question; the answer, None on expiry, or _STOPPED."""
        question = q.get("question", "")
        labels = [str(o.get("label", "")) for o in q.get("options") or []]
        ask_id = secrets.token_hex(4)
        sent = await self._send(chat_id, thread_id, ask_id, q, labels)
        pending = _Pending(
            ask_id, topic_key, chat_id, sent.message_id, question, labels
        )
        self._by_id[ask_id] = pending
        self._by_topic[topic_key] = pending
        try:
            answer = await asyncio.wait_for(pending.future, self._expiry)
        except asyncio.TimeoutError:
            await self._edit(pending, EXPIRY_TEXT)
            return None
        except asyncio.CancelledError:
            await self._edit(pending, "cancelled")
            raise
        finally:
            self._by_id.pop(ask_id, None)
            if self._by_topic.get(topic_key) is pending:
                self._by_topic.pop(topic_key, None)
        if answer is _STOPPED:  # cancel_topic already edited the message
            return _STOPPED
        await self._edit(pending, f"✅ {answer}")
        return answer

    # -------------------------------------------------------------- helpers

    def _parse(self, data: str | None) -> tuple[_Pending | None, int | None]:
        parts = (data or "").split(":")
        if len(parts) != 3 or parts[0] != CALLBACK_PREFIX or not parts[2].isdigit():
            return None, None
        return self._by_id.get(parts[1]), int(parts[2])

    async def _send(
        self,
        chat_id: int,
        thread_id: int | None,
        ask_id: str,
        q: dict[str, Any],
        labels: list[str],
    ) -> Any:
        buttons = [
            [
                InlineKeyboardButton(
                    label, callback_data=f"{CALLBACK_PREFIX}:{ask_id}:{i}"
                )
            ]
            for i, label in enumerate([*labels, OTHER_LABEL])
        ]
        header = q.get("header")
        text = (
            f"❓ {header}\n{q.get('question', '')}"
            if header
            else f"❓ {q.get('question', '')}"
        )
        kwargs: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "reply_markup": InlineKeyboardMarkup(buttons),
        }
        if thread_id is not None:
            kwargs["message_thread_id"] = thread_id
        return await self._bot.send_message(**kwargs)

    async def _edit(self, pending: _Pending, text: str) -> None:
        """Replace the question text (dropping the keyboard); never raises."""
        try:
            await self._bot.edit_message_text(
                chat_id=pending.chat_id, message_id=pending.message_id, text=text
            )
        except Exception:  # noqa: BLE001 - cosmetic; the answer already landed
            logger.warning("harness.ask_edit_failed", ask_id=pending.ask_id)

    async def _answer_query(self, query_id: str, text: str | None = None) -> None:
        kwargs: dict[str, Any] = {"callback_query_id": query_id}
        if text is not None:
            kwargs["text"] = text
        try:
            await self._bot.answer_callback_query(**kwargs)
        except Exception:  # noqa: BLE001 - the spinner is cosmetic
            logger.warning("harness.ask_callback_answer_failed")
