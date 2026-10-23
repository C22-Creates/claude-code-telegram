"""TelegramAdapter: auth, topic resolution, spawning, voice, files, commands.

Updates are real PTB objects built with ``Update.de_json`` from Bot API
dicts; replies go to ``FakeBot`` (which rejects kwargs the real Bot doesn't
take). The router, voice handler, ask bridge and resume bridge are recording
fakes: the adapter's contract with them is the call it makes.
"""

import asyncio
import re
from datetime import datetime
from itertools import count
from pathlib import Path
from types import SimpleNamespace

import pytest
from structlog.testing import capture_logs
from telegram import Update
from telegram.ext import (
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
)

from src.config.settings import Settings
from src.harness.telegram_adapter import (
    FRESH_SESSION_TEXT,
    NO_ACTIVE_RUNS,
    NOTHING_TO_CANCEL,
    TOO_LARGE_TEXT,
    TRANSCRIBING_TEXT,
    VOICE_FAILED_TEXT,
    TelegramAdapter,
)
from src.harness.voice import ProcessedVoice
from src.projects.registry import ProjectDefinition, ProjectRegistry
from src.projects.thread_manager import ProjectThreadManager
from tests.harness.fakes import FakeBot

CARL, SISI, STRANGER = 111, 222, 999
GROUP, OTHER_GROUP = -100500, -100777
NOW = datetime(2026, 10, 21, 9, 30, 5)
_ids = count(1)


async def settle(n: int = 20) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


# ------------------------------------------------------------------ fakes


class FakeRouter:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.gate: asyncio.Event | None = None
        self.finished = 0
        self.runs: dict[str, object] = {}
        self.new_sessions: list[str] = []
        self.cancels: list[str] = []
        self.cancel_result = True
        self.raise_exc: BaseException | None = None

    async def handle(self, inbound, *, opening=None):
        self.calls.append((inbound, opening))
        if self.raise_exc is not None:
            raise self.raise_exc
        if self.gate is not None:
            await self.gate.wait()
        self.finished += 1

    def active_run(self, topic_key):
        return self.runs.get(topic_key)

    async def new_session(self, topic_key):
        self.new_sessions.append(topic_key)

    async def cancel(self, topic_key):
        self.cancels.append(topic_key)
        return self.cancel_result


class FakeVoice:
    def __init__(self, transcript: str = "please draft the recap", fail=False):
        self.transcript = transcript
        self.fail = fail
        self.calls: list = []

    async def process_voice_message(self, voice, caption=None):
        self.calls.append((voice, caption))
        if self.fail:
            raise RuntimeError("provider down")
        return ProcessedVoice(
            prompt=f"Voice message transcription:\n\n{self.transcript}",
            transcription=self.transcript,
            duration=3,
        )


class FakeAskBridge:
    def __init__(self) -> None:
        self.queries: list = []

    async def on_callback(self, query):
        self.queries.append(query)


class FakeResumeBridge:
    def __init__(self, result: bool) -> None:
        self.result = result
        self.updates: list = []

    async def try_resume(self, update):
        self.updates.append(update)
        return self.result


class FakeThreadRepo:
    def __init__(self, mapping: dict[tuple[int, int], str]) -> None:
        self.mapping = mapping

    async def get_by_chat_thread(self, chat_id, thread_id):
        slug = self.mapping.get((chat_id, thread_id))
        return SimpleNamespace(project_slug=slug) if slug else None


# --------------------------------------------------------------- fixtures


@pytest.fixture
def settings(tmp_path):
    return Settings(
        telegram_bot_token="123:abc",
        telegram_bot_username="b",
        approved_directory=str(tmp_path),
        allowed_users=[CARL, SISI],
        project_threads_chat_id=GROUP,
        enable_voice_messages=True,
    )


def _project(tmp_path, slug, name, agent=""):
    path = tmp_path / slug
    path.mkdir(exist_ok=True)
    return ProjectDefinition(
        slug=slug,
        name=name,
        relative_path=Path(slug),
        absolute_path=path,
        agent=agent,
    )


class World:
    def __init__(self, tmp_path, settings, *, voice=None, resume=None):
        self.bot = FakeBot()
        self.router = FakeRouter()
        self.ask = FakeAskBridge()
        self.voice = voice or FakeVoice()
        registry = ProjectRegistry(
            [
                _project(tmp_path, "c22os", "c22os", agent="milton"),
                _project(tmp_path, "nevado", "Nevado"),
            ]
        )
        repo = FakeThreadRepo({(GROUP, 7): "nevado", (GROUP, 1): "c22os"})
        self.inbox = tmp_path / "inbox"
        self.adapter = TelegramAdapter(
            settings,
            self.router,
            self.ask,
            self.voice,
            ProjectThreadManager(registry, repo),
            resume,
            inbox_root=self.inbox,
            now=lambda: NOW,
        )
        self.context = SimpleNamespace(bot=self.bot)

    async def send(self, handler: str, update: Update) -> None:
        await getattr(self.adapter, handler)(update, self.context)
        await self.adapter.drain()


@pytest.fixture
def world(tmp_path, settings):
    return World(tmp_path, settings)


def message(
    *,
    user=CARL,
    chat=GROUP,
    thread: int | None = 7,
    private=False,
    **extra,
) -> dict:
    chat_obj = (
        {"id": user, "type": "private", "first_name": "C"}
        if private
        else {"id": chat, "type": "supergroup", "title": "g", "is_forum": True}
    )
    data = {
        "message_id": next(_ids) + 1000,
        "date": 0,
        "chat": chat_obj,
        "from": {"id": user, "is_bot": False, "first_name": "C"},
        **extra,
    }
    if thread is not None and not private:
        data["message_thread_id"] = thread
        data["is_topic_message"] = True
    return data


def update(msg: dict | None = None, **extra) -> Update:
    data = {"update_id": next(_ids), **extra}
    if msg is not None:
        data["message"] = msg
    return Update.de_json(data, None)


def text_update(text="hello", **kw) -> Update:
    return update(message(text=text, **kw))


# ---------------------------------------------------------------- auth


async def test_unauthorized_user_is_ignored_silently(world):
    await world.send("on_text", text_update(user=STRANGER))
    await world.send("on_new", text_update("/new", user=STRANGER))
    assert world.router.calls == []
    assert world.router.new_sessions == []
    assert world.bot.calls == []


async def test_foreign_group_is_ignored_silently(world):
    await world.send("on_text", text_update(chat=OTHER_GROUP))
    assert world.router.calls == []
    assert world.bot.calls == []


async def test_unmapped_topic_is_ignored_silently(world):
    await world.send("on_text", text_update(thread=42))
    assert world.router.calls == []
    assert world.bot.calls == []


async def test_every_update_and_every_drop_is_logged_at_info(world):
    voice = {"file_id": "v1", "file_unique_id": "u1", "duration": 3}
    with capture_logs() as logs:
        await world.send("log_update", update(message(voice=voice, thread=42)))
        await world.send("on_text", text_update(user=STRANGER))
        await world.send("on_text", text_update(chat=OTHER_GROUP))
        await world.send("on_text", text_update(thread=42))
    received = [e for e in logs if e["event"] == "harness.update_received"]
    assert received[0]["kinds"] == ["voice"]
    assert received[0]["thread"] == 42
    assert received[0]["log_level"] == "info"
    ignored = [e for e in logs if e["event"] == "harness.update_ignored"]
    assert [e["reason"] for e in ignored] == ["user", "chat", "topic"]
    assert all(e["log_level"] == "info" for e in ignored)


async def test_missing_allowed_users_denies_everyone(tmp_path, settings):
    settings.allowed_users = None
    w = World(tmp_path, settings)
    await w.send("on_text", text_update())
    assert w.router.calls == []


# ------------------------------------------------------------- routing


async def test_text_becomes_an_inbound_for_the_topic(world, tmp_path):
    reply_to = {"message_id": 5, "date": 0, "chat": {"id": GROUP, "type": "supergroup"}}
    upd = text_update("draft it", reply_to_message=reply_to)
    await world.send("on_text", upd)

    ((inbound, opening),) = world.router.calls
    assert opening is None
    assert inbound.topic_key == f"{GROUP}:7"
    assert (inbound.chat_id, inbound.thread_id) == (GROUP, 7)
    assert inbound.message_id == upd.message.message_id
    assert inbound.user_id == CARL
    assert inbound.text == "draft it"
    assert inbound.reply_to == 5
    assert inbound.attachments == [] and not inbound.is_voice

    ctx = world.adapter.resolve_topic(inbound)
    assert ctx.topic_key == f"{GROUP}:7"
    assert ctx.cwd == tmp_path / "nevado"
    assert ctx.agent is None  # empty registry agent -> no agent


async def test_dm_from_allowed_user_routes_to_c22os(world, tmp_path):
    await world.send("on_text", text_update(user=SISI, private=True))
    ((inbound, _),) = world.router.calls
    assert inbound.topic_key == f"{SISI}:0"
    assert inbound.thread_id is None
    ctx = world.adapter.resolve_topic(inbound)
    assert ctx.cwd == tmp_path / "c22os"
    assert ctx.agent == "milton"


async def test_forum_general_topic_resolves_as_thread_1(world, tmp_path):
    await world.send("on_text", text_update(thread=None))
    ((inbound, _),) = world.router.calls
    assert inbound.topic_key == f"{GROUP}:0"
    assert inbound.thread_id is None  # General: never send with a thread id
    assert world.adapter.resolve_topic(inbound).cwd == tmp_path / "c22os"


async def test_handler_returns_before_the_run_finishes(world):
    world.router.gate = asyncio.Event()
    # Bounded: a handler that awaited the run would hang here, not pass.
    await asyncio.wait_for(world.adapter.on_text(text_update(), world.context), 1)
    await settle()
    assert len(world.router.calls) == 1
    assert world.router.finished == 0  # the handler did not wait for the run
    world.router.gate.set()
    await world.adapter.drain()
    assert world.router.finished == 1


async def test_failed_update_task_is_logged_not_raised(world):
    world.router.raise_exc = RuntimeError("boom")
    with capture_logs() as logs:
        await world.send("on_text", text_update())
        await settle()
    assert any(e["event"] == "harness.update_task_failed" for e in logs)
    assert world.adapter.pending_tasks == 0


# ---------------------------------------------------------- dispatcher


async def test_dispatcher_resume_short_circuits(tmp_path, settings):
    resume = FakeResumeBridge(True)
    w = World(tmp_path, settings, resume=resume)
    await w.send("on_text", text_update("go"))
    assert len(resume.updates) == 1
    assert w.router.calls == []


async def test_not_a_resume_falls_through_to_the_router(tmp_path, settings):
    resume = FakeResumeBridge(False)
    w = World(tmp_path, settings, resume=resume)
    await w.send("on_text", text_update("go"))
    assert len(resume.updates) == 1
    assert len(w.router.calls) == 1


# ---------------------------------------------------------- attachments


def _photo(file_id, size, w=10):
    return {
        "file_id": file_id,
        "file_unique_id": f"u{file_id}",
        "width": w,
        "height": w,
        "file_size": size,
    }


async def test_photo_is_saved_and_attached(world):
    world.bot.files["big"] = b"\xff\xd8big-jpeg"
    upd = update(
        message(
            caption="what is this?", photo=[_photo("small", 10), _photo("big", 99, 900)]
        )
    )
    await world.send("on_photo", upd)

    assert [c["file_id"] for c in world.bot.calls_for("get_file")] == ["big"]
    ((inbound, _),) = world.router.calls
    assert inbound.text == "what is this?"
    (att,) = inbound.attachments
    assert att.kind == "photo" and att.mime == "image/jpeg"
    assert att.path.is_absolute()
    assert att.path.parent == world.inbox / f"{GROUP}_7"
    assert att.path.name == "20261021-093005-photo-ubig.jpg"
    assert att.path.read_bytes() == b"\xff\xd8big-jpeg"


async def test_photo_over_the_download_cap_is_refused(world):
    upd = update(message(photo=[_photo("huge", 21 * 1024 * 1024)]))
    await world.send("on_photo", upd)
    assert world.router.calls == []
    assert world.bot.calls_for("get_file") == []
    (sent,) = world.bot.calls_for("send_message")
    assert sent["text"] == TOO_LARGE_TEXT
    assert sent["message_thread_id"] == 7
    assert sent["reply_parameters"].message_id == upd.message.message_id


async def test_document_is_saved_under_a_sanitized_name(world):
    world.bot.files["doc"] = b"%PDF"
    doc = {
        "file_id": "doc",
        "file_unique_id": "udoc",
        "file_name": "../../Q3 plan.pdf",
        "mime_type": "application/pdf",
        "file_size": 4,
    }
    await world.send("on_document", update(message(caption="summarize", document=doc)))
    ((inbound, _),) = world.router.calls
    (att,) = inbound.attachments
    assert att.kind == "document" and att.mime == "application/pdf"
    assert att.path == world.inbox / f"{GROUP}_7" / "20261021-093005-Q3_plan.pdf"
    assert att.path.read_bytes() == b"%PDF"
    assert inbound.text == "summarize"


# ---------------------------------------------------------------- voice


def _voice_update(**kw) -> Update:
    voice = {"file_id": "v", "file_unique_id": "uv", "duration": 3, "file_size": 50}
    return update(message(voice=voice, **kw))


async def test_voice_in_idle_topic_routes_with_the_heard_opening(tmp_path, settings):
    w = World(tmp_path, settings, voice=FakeVoice("x" * 100))
    upd = _voice_update()
    await w.send("on_voice", upd)

    (sent,) = w.bot.calls_for("send_message")
    assert sent["text"] == TRANSCRIBING_TEXT
    assert sent["message_thread_id"] == 7
    assert sent["reply_parameters"].message_id == upd.message.message_id
    # The transcribing note gives way to the router's single status message.
    (deleted,) = w.bot.calls_for("delete_message")
    assert deleted == {"chat_id": GROUP, "message_id": 1}

    ((inbound, opening),) = w.router.calls
    assert opening == f"🎙️ heard: '{'x' * 80}' · starting"
    assert inbound.is_voice
    assert inbound.text == f"Voice message transcription:\n\n{'x' * 100}"


async def test_voice_in_busy_topic_is_injected(world):
    world.router.runs[f"{GROUP}:7"] = SimpleNamespace(is_active=True, elapsed_s=5.0)
    await world.send("on_voice", _voice_update())

    ((inbound, opening),) = world.router.calls
    assert opening is None
    assert "please draft the recap" in inbound.text
    (edit,) = world.bot.calls_for("edit_message_text")
    assert edit["text"] == "🎙️ heard: 'please draft the recap'"
    assert world.bot.calls_for("delete_message") == []


async def test_voice_transcription_failure_replies_with_an_error(tmp_path, settings):
    w = World(tmp_path, settings, voice=FakeVoice(fail=True))
    upd = _voice_update()
    await w.send("on_voice", upd)

    assert w.router.calls == []
    (sent,) = w.bot.calls_for("send_message")
    assert sent["reply_parameters"].message_id == upd.message.message_id
    (edit,) = w.bot.calls_for("edit_message_text")
    assert edit == {"chat_id": GROUP, "message_id": 1, "text": VOICE_FAILED_TEXT}


async def test_voice_failure_without_status_message_still_replies(tmp_path, settings):
    w = World(tmp_path, settings, voice=FakeVoice(fail=True))
    w.bot.fail_next("send_message", RuntimeError("telegram down"))
    await w.send("on_voice", _voice_update())
    texts = [c["text"] for c in w.bot.calls_for("send_message")]
    assert texts == [TRANSCRIBING_TEXT, VOICE_FAILED_TEXT]


# ------------------------------------------------------------ callbacks


def _callback(data: str, user=CARL, chat=GROUP) -> Update:
    msg = message(user=CARL, chat=chat, text="❓ q")
    msg["from"] = {"id": 1, "is_bot": True, "first_name": "bot"}
    query = {
        "id": "q1",
        "from": {"id": user, "is_bot": False, "first_name": "C"},
        "chat_instance": "ci",
        "data": data,
        "message": msg,
    }
    return update(callback_query=query)


async def test_ask_callback_goes_to_the_ask_bridge(world):
    upd = _callback("ask:abc:0")
    await world.send("on_callback", upd)
    assert world.ask.queries == [upd.callback_query]
    assert world.bot.calls == []


async def test_other_callback_is_acknowledged_and_ignored(world):
    await world.send("on_callback", _callback("stop:1:2:3"))
    assert world.ask.queries == []
    assert world.bot.calls_for("answer_callback_query") == [{"callback_query_id": "q1"}]


async def test_unauthorized_callback_is_ignored(world):
    await world.send("on_callback", _callback("ask:abc:0", user=STRANGER))
    await world.send("on_callback", _callback("ask:abc:0", chat=OTHER_GROUP))
    assert world.ask.queries == []
    assert world.bot.calls == []


# ------------------------------------------------------------- commands


async def test_new_rotates_the_session(world):
    await world.send("on_new", text_update("/new"))
    assert world.router.new_sessions == [f"{GROUP}:7"]
    (sent,) = world.bot.calls_for("send_message")
    assert sent["text"] == FRESH_SESSION_TEXT


async def test_cancel_reports_when_idle(world):
    world.router.cancel_result = False
    await world.send("on_cancel", text_update("/cancel"))
    assert world.router.cancels == [f"{GROUP}:7"]
    (sent,) = world.bot.calls_for("send_message")
    assert sent["text"] == NOTHING_TO_CANCEL


async def test_cancel_of_a_live_run_leaves_the_footer_to_the_router(world):
    await world.send("on_cancel", text_update("/cancel"))
    assert world.router.cancels == [f"{GROUP}:7"]
    assert world.bot.calls_for("send_message") == []


async def test_status_lists_active_runs_in_this_chat(world):
    await world.send("on_text", text_update(thread=7))
    await world.send("on_text", text_update(thread=None))
    await world.send("on_text", text_update(user=CARL, private=True))
    world.router.runs = {
        f"{GROUP}:7": SimpleNamespace(is_active=True, elapsed_s=125.0),
        f"{CARL}:0": SimpleNamespace(is_active=True, elapsed_s=9.0),
    }
    await world.send("on_status", text_update("/status"))
    (sent,) = world.bot.calls_for("send_message")
    assert sent["text"] == "⚙️ active runs\n• Nevado · 2m5s"


async def test_status_with_nothing_running(world):
    await world.send("on_status", text_update("/status"))
    (sent,) = world.bot.calls_for("send_message")
    assert sent["text"] == NO_ACTIVE_RUNS


# ---------------------------------------------------------- application


def test_build_application_wires_every_handler(world):
    app = world.adapter.build_application()
    handlers = app.handlers[0]
    commands = {
        next(iter(h.commands)) for h in handlers if isinstance(h, CommandHandler)
    }
    assert {"new", "cancel", "status", "start"} <= commands
    assert any(isinstance(h, CallbackQueryHandler) for h in handlers)
    callbacks = {h.callback for h in handlers if isinstance(h, MessageHandler)}
    assert callbacks == {
        world.adapter.on_text,
        world.adapter.on_voice,
        world.adapter.on_photo,
        world.adapter.on_document,
    }
    assert app.update_processor.max_concurrent_updates > 1


# ------------------------------------------------- review round 1 fixes


def command_update(text: str, **kw) -> Update:
    """A slash command the way Telegram tokenizes it.

    A bot_command entity only spans ``[A-Za-z0-9_]`` after the slash, so
    ``/c22:weekly-plan`` carries an entity for ``/c22`` alone.
    """
    head = text.split()[0]
    length = 1 + len(re.match(r"/(\w*)", head).group(1))
    entities = [{"type": "bot_command", "offset": 0, "length": length}]
    upd = update(message(text=text, entities=entities, **kw))
    # CommandHandler reads the bot's username to match "/cmd@bot" forms.
    upd.message.set_bot(SimpleNamespace(username="c22bot"))
    return upd


async def dispatch(app, upd: Update, context) -> None:
    """PTB's dispatch rule: the first matching handler in each group, in order."""
    for group in sorted(app.handlers):
        for handler in app.handlers[group]:
            check = handler.check_update(upd)
            if check is not None and check is not False:
                await handler.callback(upd, context)
                break


async def test_unknown_command_reaches_the_agent_unchanged(world):
    app = world.adapter.build_application()
    await dispatch(app, command_update("/c22:weekly-plan next week"), world.context)
    await world.adapter.drain()
    ((inbound, _),) = world.router.calls
    assert inbound.text == "/c22:weekly-plan next week"


async def test_known_command_fires_only_its_handler(world):
    app = world.adapter.build_application()
    await dispatch(app, command_update("/new"), world.context)
    await world.adapter.drain()
    assert world.router.new_sessions == [f"{GROUP}:7"]
    assert world.router.calls == []


async def test_unknown_command_from_a_stranger_is_ignored(world):
    app = world.adapter.build_application()
    await dispatch(app, command_update("/c22:add-link x", user=STRANGER), world.context)
    await world.adapter.drain()
    assert world.router.calls == [] and world.bot.calls == []


async def test_unauthorized_text_never_reaches_the_resume_bridge(tmp_path, settings):
    resume = FakeResumeBridge(True)
    w = World(tmp_path, settings, resume=resume)
    await w.send("on_text", text_update("go", user=STRANGER))
    await w.send("on_text", text_update("go", chat=OTHER_GROUP))
    assert resume.updates == []
    assert w.router.calls == []


async def test_unauthorized_media_is_never_downloaded_or_transcribed(world):
    doc = {"file_id": "d", "file_unique_id": "ud", "file_name": "a.pdf"}
    for user, chat in ((STRANGER, GROUP), (CARL, OTHER_GROUP)):
        await world.send("on_voice", _voice_update(user=user, chat=chat))
        await world.send(
            "on_photo", update(message(user=user, chat=chat, photo=[_photo("p", 5)]))
        )
        await world.send(
            "on_document", update(message(user=user, chat=chat, document=doc))
        )
    assert world.voice.calls == []
    assert world.bot.calls == []  # no get_file, no reply
    assert not world.inbox.exists()
    assert world.router.calls == []


async def test_voice_after_the_run_returned_takes_the_idle_path(world):
    # The flow is finishing: the run object is still published but inactive.
    world.router.runs[f"{GROUP}:7"] = SimpleNamespace(is_active=False, elapsed_s=0.0)
    await world.send("on_voice", _voice_update())
    ((_, opening),) = world.router.calls
    assert opening == "🎙️ heard: 'please draft the recap' · starting"
    assert len(world.bot.calls_for("delete_message")) == 1
    assert world.bot.calls_for("edit_message_text") == []


async def test_same_named_documents_do_not_overwrite_each_other(world):
    world.bot.files.update({"d1": b"one", "d2": b"two"})
    for fid in ("d1", "d2"):
        doc = {"file_id": fid, "file_unique_id": f"u{fid}", "file_name": "plan.pdf"}
        await world.send("on_document", update(message(document=doc)))
    paths = [inbound.attachments[0].path for inbound, _ in world.router.calls]
    assert paths[0] != paths[1]
    assert [p.read_bytes() for p in paths] == [b"one", b"two"]


def test_build_application_keeps_production_connection_settings(world, monkeypatch):
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://u:secret@proxy.internal:3128")
    updates_req, req = world.adapter.build_application().bot._request
    assert updates_req._client_kwargs["limits"].max_connections == 8
    assert req._client_kwargs["timeout"].pool == 30
    assert req._client_kwargs["proxy"] == "http://u:secret@proxy.internal:3128"
    assert updates_req._client_kwargs["proxy"] == "http://u:secret@proxy.internal:3128"


def test_build_application_without_proxy(world, monkeypatch):
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    _, req = world.adapter.build_application().bot._request
    assert req._client_kwargs.get("proxy") is None


async def test_drain_with_timeout_cancels_stragglers(world):
    hung = asyncio.Event()
    cancelled = []

    async def forever():
        try:
            await hung.wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    world.adapter._spawn(forever(), "test")
    await asyncio.wait_for(world.adapter.drain(timeout=0.01), 1)
    assert cancelled == [True]
    assert world.adapter.pending_tasks == 0


def test_bind_replaces_router_and_ask_bridge(world):
    router, ask = FakeRouter(), FakeAskBridge()
    world.adapter.bind(router, ask)
    assert world.adapter._router is router
    assert world.adapter._ask_bridge is ask
