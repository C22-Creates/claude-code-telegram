"""The one Telegram message per run: live progress, then the outcome footer.

Status text (plain, no parse_mode), top to bottom, empty parts omitted::

    {note}                                  last note(), e.g. "⏳ queued"
    🔧 {tool} · turn {n} · {elapsed}        latest tool event
    working · turn {n} · {elapsed}          instead, before any tool event
    ⚠️ {server} unavailable                 one line per mcp_unavailable
    still working · {n}m                    latest heartbeat

A note is shown until the next "tool" event (real progress) or the next note.

Edits are coalesced: ``update()`` only records state and arms a timer, so it
returns at once (the run bounds each progress callback at 10s). At most one
edit goes out per ``harness_status_edit_interval_seconds``; the last state wins.
Status edits are best-effort and never retried. ``finish()`` closes the status
(no edit can land after the footer), posts the answer as new message(s), then
edits the status into the outcome footer.
"""

import asyncio
import contextlib
import re
import warnings
from datetime import timedelta
from typing import Any, Awaitable, Callable

import structlog
from telegram import ReplyParameters
from telegram.error import BadRequest, NetworkError, RetryAfter, TelegramError, TimedOut

from src.harness.format import markdown_to_telegram_html, split_message
from src.harness.types import Outcome, ProgressEvent, RunResult
from src.harness.util import fmt_elapsed, footer

logger = structlog.get_logger()

# Telegram counts its 4096 limit on the text *after* entity parsing, so HTML escapes
# and tags would not strictly count. We stay conservative anyway: the converted HTML
# string of every chunk is kept <= 4096 (oversized chunks are re-split smaller), and
# 3800 is the first-pass size so that rarely needs to happen.
ANSWER_CHUNK_CHARS = 3800
MAX_MESSAGE_CHARS = 4096
MAX_RETRY_AFTER_SECONDS = 60
_FENCE_RE = re.compile(r"^```(\w*)[ \t]*$", re.MULTILINE)
RETRY_DELAYS = (1, 2, 4)
NO_TEXT_REPLY = "(done — no text reply)"
DELIVERY_FAILED = "⚠️ couldn't deliver the reply"


def _balance_fences(chunks: list[str]) -> list[str]:
    """Close a ``` fence left open at a chunk's end and reopen it in the next.

    Without this the converter sees an unpaired fence, leaves the backticks as
    text, and the code loses its formatting for the rest of the answer.
    """
    out: list[str] = []
    open_lang: str | None = None
    for chunk in chunks:
        if open_lang is not None:
            chunk = f"```{open_lang}\n{chunk}"
            open_lang = None  # the reopening fence below is counted as an opener
        for m in _FENCE_RE.finditer(chunk):
            open_lang = None if open_lang is not None else m.group(1)
        if open_lang is not None:
            chunk = f"{chunk}\n```"
        out.append(chunk)
    return out


def _render_chunks(text: str, limit: int) -> list[tuple[str, str]]:
    """Split markdown into (markdown, html) chunks whose HTML fits one message."""
    out: list[tuple[str, str]] = []
    for chunk in _balance_fences(split_message(text, limit)):
        html = markdown_to_telegram_html(chunk)
        if len(html) > MAX_MESSAGE_CHARS and len(chunk) > 1:
            shrunk = int(len(chunk) * MAX_MESSAGE_CHARS / len(html) * 0.9)
            out.extend(_render_chunks(chunk, max(1, min(shrunk, len(chunk) - 1))))
        else:
            out.append((chunk, html))
    return out


class StatusMessage:
    """Throttled status message plus final answer delivery for one run."""

    def __init__(
        self,
        bot: Any,
        chat_id: int,
        thread_id: int | None,
        reply_to: int,
        settings: Any,
        clock: Callable[[], float],
        *,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        self._bot = bot
        self._chat_id = chat_id
        self._thread_id = thread_id
        self._reply_to = reply_to
        self._interval = settings.harness_status_edit_interval_seconds
        self._clock = clock
        self._sleep = sleep
        self._message_id: int | None = None
        self._last_sent: str | None = None
        self._last_edit_at = float("-inf")
        self._note = ""
        self._tool = ""
        self._turn = 0
        self._elapsed = 0.0
        self._unavailable: list[str] = []
        self._heartbeat = ""
        self._timer: asyncio.Task | None = None
        self._edit_lock = asyncio.Lock()
        self._closed = False
        self._version = 0  # bumped by every update(); drives the flusher loop

    # ------------------------------------------------------------- public API

    async def open(self, text: str) -> int:
        """Send the status message (reply to the user's message); return its id."""
        msg = await self._send(text, parse_mode=None)
        self._message_id = msg.message_id
        self._last_sent = text
        self._last_edit_at = self._clock()
        return msg.message_id

    async def update(self, ev: ProgressEvent) -> None:
        """Record ``ev`` and arm a coalesced edit; never touches the network."""
        if self._closed:
            return
        self._turn, self._elapsed = ev.turn, ev.elapsed_s
        if ev.kind == "tool":
            self._tool = ev.tool or "tool"
            self._note = ""
        elif ev.kind == "mcp_unavailable":
            if ev.detail not in self._unavailable:
                self._unavailable = [*self._unavailable, ev.detail]
        elif ev.kind == "heartbeat":
            self._heartbeat = f"still working · {ev.detail}"
        self._version += 1
        self._arm_timer()

    async def note(self, text: str) -> None:
        """Immediate (uncoalesced) edit; also the base of the next edit."""
        if self._closed:
            return
        self._note = text
        await self._edit(text)

    async def finish(self, result: RunResult) -> None:
        """Close the status, post the answer, edit the status to the footer."""
        self._closed = True
        await self._cancel_timer()
        delivered = await self._deliver_answer(result)
        line = footer(result) if delivered else f"{footer(result)}\n{DELIVERY_FAILED}"
        if not await self._edit(line, force=True):
            try:
                await self._send(line, parse_mode=None)
            except TelegramError as exc:
                logger.error("footer_send_failed", error=str(exc))

    # --------------------------------------------------------------- status

    def _render(self) -> str:
        head = f"turn {self._turn} · {fmt_elapsed(self._elapsed)}"
        lines = [
            self._note,
            f"🔧 {self._tool} · {head}" if self._tool else f"working · {head}",
            *[f"⚠️ {name} unavailable" for name in self._unavailable],
            self._heartbeat,
        ]
        return "\n".join(line for line in lines if line)

    def _arm_timer(self) -> None:
        if self._timer is None or self._timer.done():
            self._timer = asyncio.create_task(self._flush_later())

    async def _flush_later(self) -> None:
        """Edit once per interval until no update() landed during the last edit.

        An update() arriving while the edit is in flight finds this task still
        running and arms nothing, so the loop must re-check ``_version`` itself.
        """
        while not self._closed:
            wait = self._interval - (self._clock() - self._last_edit_at)
            if wait > 0:
                await self._sleep(wait)
            if self._closed:
                return
            seen = self._version  # absorb updates that arrived while sleeping
            await self._edit(self._render())
            if self._version == seen:
                return

    async def _cancel_timer(self) -> None:
        timer, self._timer = self._timer, None
        if timer is None or timer is asyncio.current_task():
            return
        timer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await timer

    async def _edit(self, text: str, *, force: bool = False) -> bool:
        """Best-effort edit; True if the message now shows ``text``."""
        if self._message_id is None:
            return False
        async with self._edit_lock:
            if self._closed and not force:
                return False
            if text == self._last_sent:
                return True
            self._last_edit_at = self._clock()
            try:
                await self._bot.edit_message_text(
                    chat_id=self._chat_id, message_id=self._message_id, text=text
                )
            except BadRequest as exc:
                if "not modified" in str(exc).lower():
                    self._last_sent = text
                    return True
                logger.warning("status_edit_failed", error=str(exc))
                return False
            except TelegramError as exc:  # RetryAfter, NetworkError, TimedOut, ...
                logger.warning("status_edit_failed", error=str(exc))
                return False
            self._last_sent = text
            return True

    # --------------------------------------------------------------- answer

    async def _deliver_answer(self, result: RunResult) -> bool:
        text = result.text.strip()
        if not text:
            if result.outcome is not Outcome.DONE:
                return True
            text = NO_TEXT_REPLY
        ok = True
        for chunk, html in _render_chunks(text, ANSWER_CHUNK_CHARS):
            ok = await self._send_chunk(chunk, html) and ok
        return ok

    async def _send_chunk(self, chunk: str, html: str) -> bool:
        try:
            try:
                await self._send_retrying(html, "HTML")
            except BadRequest as exc:
                if "can't parse entities" not in str(exc).lower():
                    raise
                await self._send_retrying(chunk, None)
        except TelegramError as exc:
            logger.error("answer_send_failed", error=str(exc))
            return False
        return True

    async def _send_retrying(self, text: str, parse_mode: str | None) -> Any:
        """Send, retrying network errors (1s, 2s, 4s) and RetryAfter flood waits."""
        for delay in RETRY_DELAYS:
            try:
                return await self._send(text, parse_mode)
            except RetryAfter as exc:
                with warnings.catch_warnings():  # PTB 22.2+ deprecation notice
                    warnings.simplefilter("ignore")
                    wait = exc.retry_after
                secs = wait.total_seconds() if isinstance(wait, timedelta) else wait
                await self._sleep(min(secs, MAX_RETRY_AFTER_SECONDS))
            except BadRequest:
                raise  # a NetworkError subclass, but retrying cannot fix it
            except (NetworkError, TimedOut) as exc:
                logger.warning("answer_send_retry", delay=delay, error=str(exc))
                await self._sleep(delay)
        return await self._send(text, parse_mode)

    async def _send(self, text: str, parse_mode: str | None) -> Any:
        kwargs: dict[str, Any] = {
            "chat_id": self._chat_id,
            "text": text,
            "reply_parameters": ReplyParameters(
                message_id=self._reply_to, allow_sending_without_reply=True
            ),
        }
        if parse_mode:
            kwargs["parse_mode"] = parse_mode
        if self._thread_id is not None:
            kwargs["message_thread_id"] = self._thread_id
        return await self._bot.send_message(**kwargs)
