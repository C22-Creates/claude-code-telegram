"""TelegramAdapter: the front door from Telegram updates to the TopicRouter.

Every handler first authenticates the update: only ``ALLOWED_USERS`` count,
group messages only from ``PROJECT_THREADS_CHAT_ID``, and a DM from an allowed
user routes to the ``c22os`` project. Anything else is dropped silently (a
debug log, no reply) so strangers learn nothing about the bot.

``TopicRouter.handle`` awaits a whole run, so text, voice, photo and document
updates are processed in a task per update (held in ``_tasks`` until done,
failures logged): the PTB handler returns at once and polling never waits on
Claude. Commands and button taps are short and run inline.

Topic resolution needs the ``project_threads`` table (async), but the router's
``resolve_topic`` hook is synchronous. The adapter resolves before it calls
the router and caches the result per topic key; ``resolve_topic`` reads it.
"""

import asyncio
import os
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from telegram import Message, MessageEntity, ReplyParameters, Update
from telegram.ext import (
    AIORateLimiter,
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    TypeHandler,
    filters,
)

from src.harness.attachments import inbox_path
from src.harness.topic_router import TopicContext
from src.harness.types import Attachment, Inbound
from src.harness.util import fmt_elapsed, redact_proxy_url

logger = structlog.get_logger()

DM_PROJECT = "c22os"
INBOX_ROOT = Path("data/inbox")  # tests/standalone only; the app passes inbox_root
# Production inbox (ruling R17): inside APPROVED_DIRECTORY, so the agent's
# Read of "Attached file: <path>" passes the tool guard's boundary check.
INBOX_DIRNAME = ".telegram-inbox"
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024  # the Bot API's getFile limit
GENERAL_TOPIC_ID = 1  # forum "General" topic; Telegram omits its thread id
ASK_PREFIX = "ask:"
COMMAND_PASSTHROUGH_GROUP = 10
UPDATE_LOG_GROUP = -1  # runs before every handler group
HEARD_PREVIEW_CHARS = 80

START_TEXT = "👋 c22os is listening — message any project topic."
TRANSCRIBING_TEXT = "🎙️ transcribing…"
VOICE_FAILED_TEXT = "❌ error: couldn't transcribe voice note"
VOICE_DISABLED_TEXT = "❌ error: voice notes are disabled"
TOO_LARGE_TEXT = (
    f"❌ error: file too large (max {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB)"
)
DOWNLOAD_FAILED_TEXT = "❌ error: couldn't download the attachment"
FRESH_SESSION_TEXT = "🆕 fresh session"
NOTHING_TO_CANCEL = "nothing to cancel"
NO_ACTIVE_RUNS = "no active runs"
ACTIVE_RUNS_HEADER = "⚙️ active runs"


class _TooLarge(Exception):
    """The attachment exceeds ``MAX_DOWNLOAD_BYTES``."""


@dataclass(frozen=True)
class _Origin:
    """Where an authorised update came from."""

    chat_id: int
    thread_id: int | None  # for sending; None in DMs and the General topic
    user_id: int
    is_dm: bool

    @property
    def topic_key(self) -> str:
        return f"{self.chat_id}:{self.thread_id or 0}"


def _command_name(message: Message) -> str:
    """The leading command's name, lowercased, without ``/`` or ``@bot``."""
    for entity in message.entities or ():
        if entity.type == MessageEntity.BOT_COMMAND and entity.offset == 0:
            raw = message.parse_entity(entity)
            return raw[1:].split("@")[0].lower()
    return ""


def heard_text(transcript: str, *, starting: bool) -> str:
    """The voice status line: what was heard (first 80 chars)."""
    line = f"🎙️ heard: '{transcript[:HEARD_PREVIEW_CHARS]}'"
    return f"{line} · starting" if starting else line


class TelegramAdapter:
    """Turns Telegram updates into router, ask-bridge and resume-bridge calls.

    Args:
        settings: ``allowed_users``, ``project_threads_chat_id``,
            ``enable_voice_messages``, ``telegram_token_str``.
        router: the ``TopicRouter`` (``handle``/``new_session``/``cancel``/
            ``active_run``).
        ask_bridge: ``AskBridge``; receives ``ask:`` button taps.
        voice: ``VoiceHandler`` (or None when voice is unavailable).
        thread_manager: ``ProjectThreadManager`` (topic → project + registry).
        resume_bridge: Hermes ``ResumeBridge``; None when the dispatcher is off.
        inbox_root: where attachments are saved; must be inside the approved
            directory or the agent cannot open them (the app passes
            ``approved_directory / INBOX_DIRNAME``).
        now: wall clock for attachment timestamps.
    """

    def __init__(
        self,
        settings: Any,
        router: Any,
        ask_bridge: Any,
        voice: Any,
        thread_manager: Any,
        resume_bridge: Any = None,
        *,
        inbox_root: Path = INBOX_ROOT,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._settings = settings
        self._router = router
        self._ask_bridge = ask_bridge
        self._voice = voice
        self._threads = thread_manager
        self._resume_bridge = resume_bridge
        self._inbox_root = Path(inbox_root).resolve()
        self._now = now
        self._contexts: dict[str, TopicContext] = {}
        self._names: dict[str, str] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------- public API

    def bind(self, router: Any, ask_bridge: Any) -> None:
        """Late-bind the router and ask bridge.

        Both need the application's ``bot``, and the router needs this
        adapter's ``resolve_topic``, so the app builds the adapter first.
        """
        self._router = router
        self._ask_bridge = ask_bridge

    def build_application(self) -> Application:
        """A PTB application with every harness handler registered."""
        builder = (
            Application.builder()
            .token(self._settings.telegram_token_str)
            .rate_limiter(AIORateLimiter(max_retries=1))
            .concurrent_updates(True)
            .connect_timeout(30)
            .read_timeout(30)
            .write_timeout(30)
            .pool_timeout(30)
            # A getUpdates torn down mid-flight can leave the default
            # one-connection pool checked out forever ("Pool timeout"); a
            # small pool lets polling recover on its own (old bot/core.py).
            .get_updates_connection_pool_size(8)
        )
        # PTB ignores HTTP(S)_PROXY; without an explicit proxy the httpx pool
        # can wedge behind one (old bot/core.py).
        proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
        if proxy_url:
            builder = builder.proxy(proxy_url).get_updates_proxy(proxy_url)
            logger.info("harness.proxy_configured", proxy=redact_proxy_url(proxy_url))
        app = builder.build()
        # One line per incoming update, so an update no handler acts on is
        # still visible in the log (a voice note once vanished silently).
        app.add_handler(TypeHandler(Update, self.log_update), group=UPDATE_LOG_GROUP)
        for name, callback in self._commands().items():
            app.add_handler(CommandHandler(name, callback))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.on_text))
        # Any other /command (e.g. /c22:weekly-plan) goes to the agent as text.
        # A later group sees every command, so it skips the known ones that
        # group 0 already handled (mirrors the old orchestrator).
        app.add_handler(
            MessageHandler(filters.COMMAND, self.on_other_command),
            group=COMMAND_PASSTHROUGH_GROUP,
        )
        app.add_handler(MessageHandler(filters.VOICE, self.on_voice))
        app.add_handler(MessageHandler(filters.PHOTO, self.on_photo))
        app.add_handler(MessageHandler(filters.Document.ALL, self.on_document))
        app.add_handler(CallbackQueryHandler(self.on_callback))
        return app

    def resolve_topic(self, inbound: Inbound) -> TopicContext:
        """The router's hook: the context resolved when this update came in."""
        try:
            return self._contexts[inbound.topic_key]
        except KeyError:
            raise LookupError(f"topic {inbound.topic_key} was never resolved")

    @property
    def pending_tasks(self) -> int:
        return len(self._tasks)

    async def drain(self, timeout: float | None = None) -> None:
        """Wait for every in-flight update task (shutdown, tests).

        With ``timeout``, tasks still running when it elapses are cancelled
        and awaited, so shutdown stays bounded.
        """
        try:
            await asyncio.wait_for(self._drain(), timeout)
        except asyncio.TimeoutError:
            stragglers = list(self._tasks)
            logger.warning("harness.drain_timeout", cancelled=len(stragglers))
            for task in stragglers:
                task.cancel()
            await asyncio.gather(*stragglers, return_exceptions=True)

    async def _drain(self) -> None:
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    # ------------------------------------------------------------- handlers

    def _commands(self) -> dict[str, Callable[..., Any]]:
        return {
            "start": self.on_start,
            "new": self.on_new,
            "cancel": self.on_cancel,
            "status": self.on_status,
        }

    async def on_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if (origin := self._authorize(update)) is not None:
            self._spawn(self._text_flow(update, context.bot, origin), "text")

    async def on_other_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Forward an unregistered /command to the agent, text unchanged.

        The command name is read from the leading bot_command entity, as PTB's
        CommandHandler does: Telegram ends it at the first character outside
        ``[A-Za-z0-9_]``, so ``/c22:add-link`` is the unknown command ``c22``
        while ``/new@bot`` is the known ``new``.
        """
        message = update.effective_message
        if message is None or not message.text:
            return
        if _command_name(message) in self._commands():
            return  # its CommandHandler (group 0) already ran
        await self.on_text(update, context)

    async def on_voice(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if (origin := self._authorize(update)) is not None:
            self._spawn(self._voice_flow(update, context.bot, origin), "voice")

    async def on_photo(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if (origin := self._authorize(update)) is not None:
            self._spawn(self._file_flow(update, context.bot, origin), "photo")

    async def on_document(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if (origin := self._authorize(update)) is not None:
            self._spawn(self._file_flow(update, context.bot, origin), "document")

    async def on_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """``ask:`` taps go to the ask bridge; other buttons are just acked."""
        query = update.callback_query
        if query is None or self._authorize(update) is None:
            return
        if (query.data or "").startswith(ASK_PREFIX):
            await self._ask_bridge.on_callback(query)
            return
        try:
            await context.bot.answer_callback_query(callback_query_id=query.id)
        except Exception as exc:  # noqa: BLE001 - the spinner is cosmetic
            logger.warning("harness.callback_ack_failed", error=str(exc))

    async def on_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if (origin := self._authorize(update)) is not None:
            await self._reply(context.bot, origin, update.effective_message, START_TEXT)

    async def on_new(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        origin = self._authorize(update)
        if origin is None or await self._context_for(origin) is None:
            return
        await self._router.new_session(origin.topic_key)
        await self._reply(
            context.bot, origin, update.effective_message, FRESH_SESSION_TEXT
        )

    async def on_cancel(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Stop the topic's run; the router's status message shows the footer."""
        origin = self._authorize(update)
        if origin is None or await self._context_for(origin) is None:
            return
        if not await self._router.cancel(origin.topic_key):
            await self._reply(
                context.bot, origin, update.effective_message, NOTHING_TO_CANCEL
            )

    async def on_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """List this chat's live runs with their elapsed time."""
        origin = self._authorize(update)
        if origin is None:
            return
        prefix = f"{origin.chat_id}:"
        lines = []
        for key in sorted(k for k in self._contexts if k.startswith(prefix)):
            run = self._router.active_run(key)
            if run is None or not run.is_active:
                continue
            name = self._names.get(key, key)
            lines.append(f"• {name} · {fmt_elapsed(run.elapsed_s)}")
        text = "\n".join([ACTIVE_RUNS_HEADER, *lines]) if lines else NO_ACTIVE_RUNS
        await self._reply(context.bot, origin, update.effective_message, text)

    # ---------------------------------------------------------------- flows

    async def _text_flow(self, update: Update, bot: Any, origin: _Origin) -> None:
        if await self._context_for(origin) is None:
            return
        # Hermes: an answer to a dispatcher-blocked task resumes it instead.
        if self._resume_bridge is not None and await self._resume_bridge.try_resume(
            update
        ):
            return
        message = update.effective_message
        await self._router.handle(self._inbound(origin, message, message.text or ""))

    async def _voice_flow(self, update: Update, bot: Any, origin: _Origin) -> None:
        """Transcribe, then start a run (idle topic) or inject (busy topic).

        A ``🎙️ transcribing…`` note shows progress while the provider works.
        For an idle topic it is deleted once the transcript is in, and the
        router opens the run's one status message with the heard line. For a
        busy topic the note is edited to the heard line: the run's existing
        status message carries on and the 👀 reaction acks the inject.
        """
        message = update.effective_message
        if await self._context_for(origin) is None:
            return
        if self._voice is None or not self._settings.enable_voice_messages:
            await self._reply(bot, origin, message, VOICE_DISABLED_TEXT)
            return
        note = await self._reply(bot, origin, message, TRANSCRIBING_TEXT)
        try:
            processed = await self._voice.process_voice_message(
                message.voice, caption=message.caption
            )
        except Exception as exc:  # noqa: BLE001 - any provider failure
            logger.warning("harness.voice_transcription_failed", error=str(exc))
            await self._show(bot, origin, message, note, VOICE_FAILED_TEXT)
            return
        inbound = self._inbound(origin, message, processed.prompt, is_voice=True)
        transcript = processed.transcription
        run = self._router.active_run(origin.topic_key)
        if run is not None and run.is_active:
            await self._show(
                bot, origin, message, note, heard_text(transcript, starting=False)
            )
            await self._router.handle(inbound)
            return
        await self._delete(bot, note)
        await self._router.handle(
            inbound, opening=heard_text(transcript, starting=True)
        )

    async def _file_flow(self, update: Update, bot: Any, origin: _Origin) -> None:
        message = update.effective_message
        if await self._context_for(origin) is None:
            return
        try:
            attachment = await self._save_attachment(bot, origin, message)
        except _TooLarge:
            await self._reply(bot, origin, message, TOO_LARGE_TEXT)
            return
        except Exception as exc:  # noqa: BLE001 - network/disk
            logger.warning("harness.attachment_download_failed", error=str(exc))
            await self._reply(bot, origin, message, DOWNLOAD_FAILED_TEXT)
            return
        await self._router.handle(
            self._inbound(origin, message, message.caption or "", [attachment])
        )

    # ---------------------------------------------------------- attachments

    async def _save_attachment(
        self, bot: Any, origin: _Origin, message: Message
    ) -> Attachment:
        if message.photo:
            photo = max(message.photo, key=lambda p: (p.file_size or 0, p.width))
            path = await self._download(
                bot,
                origin,
                photo.file_id,
                photo.file_size,
                f"photo-{photo.file_unique_id}.jpg",
            )
            return Attachment(kind="photo", path=path, mime="image/jpeg")
        doc = message.document
        path = await self._download(
            bot,
            origin,
            doc.file_id,
            doc.file_size,
            doc.file_name or f"document-{doc.file_unique_id}",
        )
        mime = doc.mime_type or "application/octet-stream"
        return Attachment(kind="document", path=path, mime=mime)

    async def _download(
        self, bot: Any, origin: _Origin, file_id: str, size: int | None, name: str
    ) -> Path:
        if size is not None and size > MAX_DOWNLOAD_BYTES:
            raise _TooLarge()
        file = await bot.get_file(file_id=file_id)
        if file.file_size is not None and file.file_size > MAX_DOWNLOAD_BYTES:
            raise _TooLarge()
        path = inbox_path(self._inbox_root, origin.topic_key, name, self._now())
        await file.download_to_drive(custom_path=path)
        return path

    async def log_update(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Log the shape of an incoming update — never its content."""
        message = update.effective_message
        kinds = [
            kind
            for kind in ("text", "voice", "audio", "video_note", "photo", "document")
            if message is not None and getattr(message, kind, None)
        ]
        logger.info(
            "harness.update_received",
            kinds=kinds or ["callback" if update.callback_query else "other"],
            chat=getattr(update.effective_chat, "id", None),
            thread=getattr(message, "message_thread_id", None),
            user=getattr(update.effective_user, "id", None),
        )

    # ------------------------------------------------------- auth & context

    def _authorize(self, update: Update) -> _Origin | None:
        """The update's origin if it may reach the bot, else None (logged)."""
        user = update.effective_user
        chat = update.effective_chat
        allowed = self._settings.allowed_users or []
        if user is None or chat is None or user.id not in allowed:
            logger.info(
                "harness.update_ignored", reason="user", user=getattr(user, "id", None)
            )
            return None
        if chat.type == "private":
            return _Origin(chat.id, None, user.id, is_dm=True)
        if chat.id != self._settings.project_threads_chat_id:
            logger.info("harness.update_ignored", reason="chat", chat=chat.id)
            return None
        message = update.effective_message
        thread_id = getattr(message, "message_thread_id", None) if message else None
        if not (isinstance(thread_id, int) and thread_id > 0):
            thread_id = None
        return _Origin(chat.id, thread_id, user.id, is_dm=False)

    async def _context_for(self, origin: _Origin) -> TopicContext | None:
        """Resolve (and cache) the topic's project; None if it has none."""
        if origin.is_dm:
            project = self._threads.registry.get_by_slug(DM_PROJECT)
        else:
            lookup = origin.thread_id or GENERAL_TOPIC_ID
            project = await self._threads.resolve_project(origin.chat_id, lookup)
        if project is None or not project.enabled:
            logger.info(
                "harness.update_ignored", reason="topic", topic=origin.topic_key
            )
            return None
        ctx = TopicContext(
            topic_key=origin.topic_key,
            cwd=project.absolute_path,
            agent=project.agent or None,
        )
        self._contexts[origin.topic_key] = ctx
        self._names[origin.topic_key] = project.name
        return ctx

    @staticmethod
    def _inbound(
        origin: _Origin,
        message: Message,
        text: str,
        attachments: list[Attachment] | None = None,
        *,
        is_voice: bool = False,
    ) -> Inbound:
        reply = message.reply_to_message
        return Inbound(
            topic_key=origin.topic_key,
            chat_id=origin.chat_id,
            thread_id=origin.thread_id,
            message_id=message.message_id,
            user_id=origin.user_id,
            text=text,
            attachments=attachments or [],
            reply_to=reply.message_id if reply is not None else None,
            is_voice=is_voice,
        )

    # -------------------------------------------------------------- telegram

    async def _reply(
        self, bot: Any, origin: _Origin, message: Message | None, text: str
    ) -> Any:
        """Reply in the topic; returns the sent message or None (logged)."""
        kwargs: dict[str, Any] = {"chat_id": origin.chat_id, "text": text}
        if message is not None:
            kwargs["reply_parameters"] = ReplyParameters(
                message_id=message.message_id, allow_sending_without_reply=True
            )
        if origin.thread_id is not None:
            kwargs["message_thread_id"] = origin.thread_id
        try:
            return await bot.send_message(**kwargs)
        except Exception as exc:  # noqa: BLE001 - never kill the flow for a reply
            logger.warning("harness.reply_failed", error=str(exc))
            return None

    async def _show(
        self, bot: Any, origin: _Origin, message: Message, note: Any, text: str
    ) -> None:
        """Turn the voice note into *text*, or reply afresh if there is none."""
        if note is not None:
            try:
                await bot.edit_message_text(
                    chat_id=origin.chat_id, message_id=note.message_id, text=text
                )
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("harness.note_edit_failed", error=str(exc))
        await self._reply(bot, origin, message, text)

    async def _delete(self, bot: Any, note: Any) -> None:
        if note is None:
            return
        try:
            await bot.delete_message(chat_id=note.chat_id, message_id=note.message_id)
        except Exception as exc:  # noqa: BLE001 - a stray note is cosmetic
            logger.warning("harness.note_delete_failed", error=str(exc))

    # --------------------------------------------------------------- tasks

    def _spawn(self, coro: Coroutine[Any, Any, None], what: str) -> None:
        task = asyncio.create_task(coro, name=f"harness-{what}")
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: "asyncio.Task[None]") -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "harness.update_task_failed",
                task=task.get_name(),
                error=repr(exc),
                exc_info=exc,
            )
