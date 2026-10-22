"""App wiring: restart recovery, ordered bounded shutdown, startup smoke test.

The whole app is built for real (DB on a temp file, registry, router, adapter,
event bus, notifications, scheduler, job handler) except the PTB
``Application``, which is replaced by ``FakeApplication`` so nothing touches
the network: its ``bot`` is a ``FakeBot`` and polling is a recorded no-op.
"""

import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from claude_agent_sdk import SystemMessage

from src.config.settings import Settings
from src.exceptions import ConfigurationError
from src.harness.agent_run import AgentRun
from src.harness.app import RESTART_TEXT, HarnessApp, ShutdownBudget, build
from src.harness.session_store import SessionStore
from src.harness.topic_router import RESTARTING, TopicContext
from src.harness.types import Inbound
from src.storage.database import DatabaseManager
from tests.harness.fakes import FakeBot, FakeSDKClient, message_start, result, text

CARL = 111
GROUP = -100500


class AppBot(FakeBot):
    """FakeBot plus the forum/menu calls startup makes."""

    def __init__(self) -> None:
        super().__init__()
        self._next_thread = 500

    async def set_my_commands(self, commands, **kwargs):
        self._record("set_my_commands", {"commands": commands})
        return True

    async def create_forum_topic(self, **kwargs):
        self._record("create_forum_topic", kwargs)
        self._next_thread += 1
        return SimpleNamespace(message_thread_id=self._next_thread, **kwargs)

    async def edit_forum_topic(self, **kwargs):
        self._record("edit_forum_topic", kwargs)
        return True

    async def reopen_forum_topic(self, **kwargs):
        self._record("reopen_forum_topic", kwargs)
        return True

    async def close_forum_topic(self, **kwargs):
        self._record("close_forum_topic", kwargs)
        return True


class FakeUpdater:
    def __init__(self, log: list[str]) -> None:
        self._log = log
        self.running = False
        self.polling_kwargs: dict = {}

    async def start_polling(self, **kwargs):
        self._log.append("polling")
        self.polling_kwargs = kwargs
        self.running = True

    async def stop(self):
        self._log.append("updater.stop")
        self.running = False


class FakeApplication:
    """The PTB Application surface the app uses, without a network."""

    def __init__(self, log: list[str]) -> None:
        self.bot = AppBot()
        self.log = log
        self.updater = FakeUpdater(log)
        self.error_handlers: list = []

    def add_error_handler(self, callback):
        self.error_handlers.append(callback)

    async def initialize(self):
        self.log.append("app.initialize")

    async def start(self):
        self.log.append("app.start")

    async def stop(self):
        self.log.append("app.stop")

    async def shutdown(self):
        self.log.append("app.shutdown")


@pytest.fixture
def env(tmp_path):
    root = tmp_path / "Projects"
    for name in ("c22os", "nevado"):
        (root / name).mkdir(parents=True)
    projects = tmp_path / "projects.yaml"
    projects.write_text(
        "projects:\n"
        "  - {slug: c22os, name: c22os, path: c22os, agent: milton}\n"
        "  - {slug: nevado, name: Nevado, path: nevado}\n"
    )
    return SimpleNamespace(tmp=tmp_path, root=root, projects=projects)


def make_settings(env, **overrides) -> Settings:
    values = dict(
        telegram_bot_token="123:abc",
        telegram_bot_username="c22bot",
        approved_directory=str(env.root),
        allowed_users=[CARL],
        database_url=f"sqlite:///{env.tmp / 'bot.db'}",
        enable_project_threads=True,
        project_threads_mode="group",
        project_threads_chat_id=GROUP,
        projects_config_path=str(env.projects),
        project_threads_sync_action_interval_seconds=0,
        enable_scheduler=True,
        enable_voice_messages=False,
    )
    values.update(overrides)
    return Settings(_env_file=None, **values)


async def make_app(settings, *, router_kwargs=None, budget=None) -> HarnessApp:
    log: list[str] = []
    return await build(
        settings,
        application_factory=lambda adapter: FakeApplication(log),
        router_kwargs=router_kwargs,
        budget=budget,
    )


async def seed_running(settings, rows) -> None:
    db = DatabaseManager(settings.database_url)
    await db.initialize()
    store = SessionStore(db)
    for topic, chat, thread in rows:
        await store.start_run(topic, chat, thread, "do the thing")
    await db.close()


async def run_statuses(app: HarnessApp) -> list[tuple[str, str]]:
    async with app.db.get_connection() as conn:
        cur = await conn.execute(
            "SELECT topic_key, status FROM harness_runs ORDER BY started_at"
        )
        return [tuple(r) for r in await cur.fetchall()]


def restart_notices(app: HarnessApp) -> list[dict]:
    return [
        kw for kw in app.bot.calls_for("send_message") if kw["text"] == RESTART_TEXT
    ]


async def until(pred, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.001)


# ------------------------------------------------------------ restart recovery


async def test_boot_posts_restart_notice_per_running_row_and_abandons_them(env):
    settings = make_settings(env)
    await seed_running(settings, [(f"{GROUP}:7", GROUP, 7), (f"{CARL}:0", CARL, None)])
    app = await make_app(settings)
    await app.start()
    try:
        notices = restart_notices(app)
        assert [(n["chat_id"], n["message_thread_id"]) for n in notices] == [
            (GROUP, 7),
            (CARL, None),
        ]
        assert [s for _, s in await run_statuses(app)] == ["abandoned", "abandoned"]
        # Recovery runs before polling, so no new run's row can be abandoned.
        log = app.application.log
        assert log.index("polling") > log.index("app.initialize")
    finally:
        await app.stop()


async def test_restart_notice_failure_still_abandons_rows(env):
    settings = make_settings(env)
    await seed_running(settings, [(f"{GROUP}:7", GROUP, 7), (f"{GROUP}:9", GROUP, 9)])
    app = await make_app(settings)
    app.bot.fail_next("send_message", RuntimeError("chat gone"))
    await app.start()
    try:
        # The first send fails; the second topic is still notified.
        assert [n["message_thread_id"] for n in restart_notices(app)] == [7, 9]
        assert [s for _, s in await run_statuses(app)] == ["abandoned", "abandoned"]
    finally:
        await app.stop()


async def test_boot_with_no_running_rows_posts_nothing(env):
    app = await make_app(make_settings(env))
    await app.start()
    try:
        assert restart_notices(app) == []
    finally:
        await app.stop()


def held_run_factory(settings, gate: asyncio.Event, clients: list, **client_kw):
    """Router run factory whose run blocks on ``gate`` (a live, interruptible run)."""

    def factory(options, on_progress):
        script = [
            (0, SystemMessage(subtype="init", data={"session_id": "sess-1"})),
            (0, message_start()),
            ("call", gate.wait),
            (0, text("answer")),
            (0, result(session_id="sess-1", text="answer")),
        ]
        client_factory = FakeSDKClient.factory(script, **client_kw)
        clients.append(client_factory.client)
        return AgentRun(
            options,
            on_progress,
            settings,
            client_factory,
            interrupt_timeout=0.05,
            cleanup_timeout=0.05,
            progress_timeout=0.05,
        )

    return factory


def inbound(topic_key: str, thread_id: int | None = 7) -> Inbound:
    return Inbound(
        topic_key=topic_key,
        chat_id=GROUP,
        thread_id=thread_id,
        message_id=42,
        user_id=CARL,
        text="long job",
    )


async def test_stop_interrupts_active_run_and_next_boot_posts_restart(env):
    settings = make_settings(env)
    gate, clients = asyncio.Event(), []
    app = await make_app(
        settings,
        router_kwargs={"run_factory": held_run_factory(settings, gate, clients)},
    )
    await app.start()
    key = f"{GROUP}:7"
    # What the adapter does per update: resolve the topic, then spawn handle().
    app.adapter._contexts[key] = TopicContext(key, env.root / "nevado", None)
    app.adapter._spawn(app.router.handle(inbound(key)), "text")
    await until(lambda: app.router.active_run(key) is not None)

    started = time.monotonic()
    await app.stop()
    assert time.monotonic() - started < 2
    assert clients[0].interrupted == 1
    edits = [kw["text"] for kw in app.bot.calls_for("edit_message_text")]
    assert edits[-1].startswith(f"⚠️ stopped ({RESTARTING}) · ")

    reboot = await make_app(settings)
    await reboot.start()
    try:
        notices = restart_notices(reboot)
        assert [(n["chat_id"], n["message_thread_id"]) for n in notices] == [(GROUP, 7)]
        assert await run_statuses(reboot) == [(key, "abandoned")]
        # The interrupted run saved its session, so "reply to resume" resumes.
        assert (await reboot.store.get(key)).session_id == "sess-1"
    finally:
        await reboot.stop()


async def test_stop_is_bounded_when_a_run_ignores_interrupt(env):
    settings = make_settings(env)
    gate, clients = asyncio.Event(), []
    budget = ShutdownBudget(runs=0.2)
    app = await make_app(
        settings,
        router_kwargs={
            "run_factory": held_run_factory(
                settings, gate, clients, hang_interrupt=True
            )
        },
        budget=budget,
    )
    await app.start()
    key = f"{GROUP}:7"
    app.adapter._contexts[key] = TopicContext(key, env.root / "nevado", None)
    app.adapter._spawn(app.router.handle(inbound(key)), "text")
    await until(lambda: app.router.active_run(key) is not None)

    started = time.monotonic()
    await app.stop()
    assert time.monotonic() - started < 3
    assert app.adapter.pending_tasks == 0

    reboot = await make_app(settings)
    await reboot.start()
    try:
        assert len(restart_notices(reboot)) == 1
    finally:
        await reboot.stop()


# --------------------------------------------------------------- wiring smoke


async def test_full_app_starts_and_stops_in_order(env):
    settings = make_settings(env)
    app = await make_app(settings)
    log = app.application.log
    await app.start()

    # Startup: command menu, forum topics synced, polling with dropped backlog.
    menu = app.bot.calls_for("set_my_commands")[0]["commands"]
    assert {c.command for c in menu} == {"start", "new", "cancel", "status"}
    created = {kw["name"] for kw in app.bot.calls_for("create_forum_topic")}
    assert created == {"c22os", "Nevado"}
    assert app.application.updater.polling_kwargs["drop_pending_updates"] is True
    assert app.application.error_handlers
    assert app.scheduler is not None and app.dispatcher is None
    # One shared slot pool for interactive and scheduled runs.
    assert app.router._slots is app.slots and app.jobs._slots is app.slots
    assert app.router._validator is app.jobs._validator
    assert app.adapter._router is app.router

    await app.stop()
    assert log.index("updater.stop") < log.index("app.stop") < log.index("app.shutdown")
    await app.stop()  # idempotent


async def test_scheduled_event_runs_through_the_job_handler(env):
    from src.events.types import ScheduledEvent

    settings = make_settings(env)
    calls = []

    async def fake_run_job(prompt, cwd, agent, slots, settings_, model=None, **kw):
        calls.append((prompt, cwd, slots))
        from src.harness.types import Outcome, RunResult

        return RunResult(outcome=Outcome.DONE, text="digest", elapsed_s=1)

    app = await make_app(settings)
    app.jobs._run_job = fake_run_job
    await app.start()
    try:
        await app.event_bus.publish(
            ScheduledEvent(
                job_id="j1",
                job_name="digest",
                prompt="go",
                working_directory=env.root / "c22os",
                target_chat_ids=[GROUP],
            )
        )
        await until(lambda: calls)
        await app.jobs.wait_idle()
        assert calls[0][2] is app.slots
    finally:
        await app.stop()


async def test_dispatcher_wiring_when_enabled(env, monkeypatch):
    board = SimpleNamespace(reclaim_stale=lambda stale_seconds: [])
    monkeypatch.setattr("src.harness.app.load_task_board", lambda path, db: board)
    settings = make_settings(env, enable_dispatcher=True, dispatcher_tick_seconds=3600)
    app = await make_app(settings)
    assert app.dispatcher is not None
    assert app.adapter._resume_bridge is not None
    runner = app.dispatcher._runner
    assert runner._slots is app.slots and runner._validator is app.router._validator
    assert runner._hermes_cli_path == settings.hermes_path / "cli.py"
    await app.stop()  # never started: stop still closes the DB


# ------------------------------------------------------------- config errors


async def test_missing_allowed_users_is_a_configuration_error(env):
    with pytest.raises(ConfigurationError, match="ALLOWED_USERS"):
        await make_app(make_settings(env, allowed_users=None))


async def test_project_threads_are_required(env):
    with pytest.raises(ConfigurationError, match="ENABLE_PROJECT_THREADS"):
        await make_app(make_settings(env, enable_project_threads=False))


async def test_failed_build_leaves_no_live_database_threads(env):
    """Regression (old test_main): a startup failure after the DB opened must
    close it, or aiosqlite's non-daemon threads keep the process alive and
    sys.exit() hangs -- which a supervisor reads as healthy."""
    settings = make_settings(env)
    env.projects.write_text("projects: []\n")  # registry load fails after DB init
    before = {t.ident for t in threading.enumerate()}
    with pytest.raises(ValueError):
        await make_app(settings)
    leaked: list = []
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        leaked = [
            t
            for t in threading.enumerate()
            if t.ident not in before and t.is_alive() and not t.daemon
        ]
        if not leaked:
            break
        await asyncio.sleep(0.05)
    assert not leaked, f"non-daemon threads still alive after failure: {leaked}"


async def test_voice_handler_built_only_when_usable(env):
    on = await make_app(
        make_settings(env, enable_voice_messages=True, voice_provider="local")
    )
    off = await make_app(
        make_settings(env, enable_voice_messages=True, voice_provider="openai")
    )
    try:
        assert on.adapter._voice is not None
        assert off.adapter._voice is None  # no OPENAI_API_KEY
    finally:
        await on.stop()
        await off.stop()


def test_main_module_imports():
    import src.main  # noqa: F401

    assert Path(src.main.__file__).name == "main.py"
