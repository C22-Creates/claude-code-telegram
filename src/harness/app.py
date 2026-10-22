"""Builds, starts and stops the whole bot around the harness.

``build(settings)`` wires every component once, with no network calls;
``HarnessApp.start()`` brings the bot online and ``stop()`` takes it down in a
fixed, time-bounded order (systemd allows 30s before SIGKILL).

Restart recovery: a run that was in flight when the process went down left
its ``harness_runs`` row 'running' (a crash, or ``stop()``, which interrupts
live runs but deliberately keeps their rows). The next ``start()`` posts the
restart notice to each such topic, then marks every leftover row abandoned --
before polling starts, so no new run's row can be swept up with them.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

import structlog
from telegram import BotCommand, Update

from src.config.features import FeatureFlags
from src.config.settings import Settings
from src.dispatcher.board_client import load_task_board
from src.dispatcher.resume_bridge import ResumeBridge
from src.dispatcher.runner import ClaudeTaskRunner
from src.dispatcher.service import DispatcherService
from src.events.bus import EventBus
from src.exceptions import ConfigurationError
from src.harness.ask_bridge import AskBridge
from src.harness.jobs import ScheduledJobHandler
from src.harness.run_slots import RunSlots
from src.harness.session_store import SessionStore
from src.harness.telegram_adapter import TelegramAdapter
from src.harness.topic_router import TopicRouter
from src.harness.voice import VoiceHandler
from src.notifications.service import NotificationService
from src.projects import ProjectThreadManager, load_project_registry
from src.scheduler.scheduler import JobScheduler
from src.security.validators import SecurityValidator
from src.storage.database import DatabaseManager
from src.storage.repositories import ProjectThreadRepository

logger = structlog.get_logger()

RESTART_TEXT = "♻️ restarted during your last request — reply to resume"

BOT_COMMANDS = [
    BotCommand("new", "Start a fresh session in this topic"),
    BotCommand("cancel", "Stop this topic's run"),
    BotCommand("status", "List active runs"),
    BotCommand("start", "Say hello"),
]


@dataclass(frozen=True)
class ShutdownBudget:
    """Seconds each shutdown step may take; the sum stays well under 30s."""

    polling: float = 2.0
    dispatcher: float = 4.0
    scheduler: float = 1.0
    jobs: float = 4.0
    runs: float = 8.0  # interrupt live runs + drain in-flight updates
    notifications: float = 1.0
    application: float = 3.0
    storage: float = 2.0


ApplicationFactory = Callable[[TelegramAdapter], Any]


class HarnessApp:
    """Every long-lived component, plus ordered start/stop."""

    def __init__(
        self,
        *,
        settings: Settings,
        db: DatabaseManager,
        store: SessionStore,
        slots: RunSlots,
        thread_manager: ProjectThreadManager,
        application: Any,
        adapter: TelegramAdapter,
        router: TopicRouter,
        event_bus: EventBus,
        notifications: NotificationService,
        jobs: ScheduledJobHandler,
        scheduler: Optional[JobScheduler],
        dispatcher: Optional[DispatcherService],
        budget: ShutdownBudget,
    ) -> None:
        self.settings = settings
        self.db = db
        self.store = store
        self.slots = slots
        self.thread_manager = thread_manager
        self.application = application
        self.bot = application.bot
        self.adapter = adapter
        self.router = router
        self.event_bus = event_bus
        self.notifications = notifications
        self.jobs = jobs
        self.scheduler = scheduler
        self.dispatcher = dispatcher
        self.budget = budget
        self._initialized = False
        self._app_started = False
        self._stopped = False

    # ------------------------------------------------------------------ start

    async def start(self) -> None:
        """Bring the bot online; polling starts last."""
        await self.application.initialize()
        self._initialized = True
        await self._set_commands()
        await self._sync_topics()
        await self.recover_runs()

        await self.event_bus.start()
        self.notifications.register()
        await self.notifications.start()
        self.jobs.register()  # before the scheduler fires catch-up jobs
        if self.scheduler is not None:
            await self.scheduler.start()
            logger.info("Job scheduler enabled")
        if self.dispatcher is not None:
            await self.dispatcher.start()
            logger.info(
                "Hermes dispatcher enabled",
                board_db=str(self.settings.hermes_board_db),
                tick_seconds=self.settings.dispatcher_tick_seconds,
            )

        await self.application.start()
        self._app_started = True
        await self.application.updater.start_polling(
            allowed_updates=Update.ALL_TYPES, drop_pending_updates=True
        )
        logger.info("Harness bot started", mode="polling")

    async def recover_runs(self) -> int:
        """Post the restart notice for leftover 'running' rows; abandon them all."""
        rows = await self.store.running_runs()
        notified: set[tuple[int, Optional[int]]] = set()
        for row in rows:
            if row.chat_id is None or (row.chat_id, row.thread_id) in notified:
                continue
            notified.add((row.chat_id, row.thread_id))
            try:
                await self.bot.send_message(
                    chat_id=row.chat_id,
                    message_thread_id=row.thread_id,
                    text=RESTART_TEXT,
                )
            except Exception as exc:  # noqa: BLE001 - one dead chat must not block
                logger.warning(
                    "harness.restart_notice_failed",
                    topic=row.topic_key,
                    error=str(exc),
                )
        await self.store.abandon_all_running()
        if rows:
            logger.info("harness.restart_recovered", runs=len(rows))
        return len(notified)

    async def _set_commands(self) -> None:
        try:
            await self.bot.set_my_commands(BOT_COMMANDS)
        except Exception as exc:  # noqa: BLE001 - the menu is cosmetic
            logger.warning("harness.set_commands_failed", error=str(exc))

    async def _sync_topics(self) -> None:
        """Group mode: create/reconcile one forum topic per project (as before)."""
        if self.settings.project_threads_mode != "group":
            return
        result = await self.thread_manager.sync_topics(
            self.bot, chat_id=self.settings.project_threads_chat_id
        )
        logger.info(
            "Project thread startup sync complete",
            chat_id=self.settings.project_threads_chat_id,
            created=result.created,
            reused=result.reused,
            renamed=result.renamed,
            failed=result.failed,
            deactivated=result.deactivated,
        )

    # ------------------------------------------------------------------- stop

    async def stop(self) -> None:
        """Ordered, bounded shutdown; safe after a partial start, idempotent.

        Intake stops first, then dispatcher -> scheduler -> scheduled jobs ->
        interactive runs (interrupted, rows kept for the restart notice) ->
        notifications -> Telegram application -> storage.
        """
        if self._stopped:
            return
        self._stopped = True
        b = self.budget
        logger.info("Shutting down application")
        if self._app_started and self.application.updater.running:
            await _step("polling", self.application.updater.stop(), b.polling)
        if self.dispatcher is not None:
            await _step("dispatcher", self.dispatcher.stop(), b.dispatcher)
        if self.scheduler is not None:
            await _step("scheduler", self.scheduler.stop(), b.scheduler)
        await _step("jobs", self.jobs.cancel_all(timeout=b.jobs), b.jobs + 1)
        await _step("runs", self._stop_runs(), b.runs + 1)
        await _step("notifications", self.notifications.stop(), b.notifications)
        await _step("event_bus", self.event_bus.stop(), b.notifications)
        await _step("application", self._stop_application(), b.application)
        await _step("storage", self.db.close(), b.storage)
        logger.info("Application shutdown complete")

    async def _stop_runs(self) -> None:
        """Interrupt live runs, then drain update tasks within ``budget.runs``."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.budget.runs
        try:
            interrupted = await asyncio.wait_for(
                self.router.shutdown(), self.budget.runs / 4
            )
            logger.info("harness.runs_interrupted", count=interrupted)
        except asyncio.TimeoutError:
            logger.error("harness.runs_interrupt_timeout")
        await self.adapter.drain(timeout=max(0.0, deadline - loop.time()))

    async def _stop_application(self) -> None:
        if self._app_started:
            await self.application.stop()
        if self._initialized:
            await self.application.shutdown()

    async def _on_error(self, update: object, context: Any) -> None:
        """PTB error handler: log; the harness replies to users on its own."""
        logger.error(
            "harness.telegram_error",
            error=str(context.error),
            error_type=type(context.error).__name__,
            update_type=type(update).__name__ if update else None,
        )


async def _step(name: str, aw: Awaitable[Any], timeout: float) -> None:
    """Run one shutdown step; log (never raise) failures and overruns."""
    try:
        await asyncio.wait_for(aw, timeout)
    except asyncio.TimeoutError:
        logger.error("harness.shutdown_step_timeout", step=name, timeout=timeout)
    except Exception:  # noqa: BLE001 - later steps (storage close) must still run
        logger.exception("harness.shutdown_step_failed", step=name)


def _check_config(settings: Settings) -> None:
    if not settings.allowed_users:
        raise ConfigurationError(
            "ALLOWED_USERS must list at least one Telegram user id"
        )
    if not settings.enable_project_threads or not settings.projects_config_path:
        raise ConfigurationError(
            "The harness routes by project: ENABLE_PROJECT_THREADS and "
            "PROJECTS_CONFIG_PATH are required"
        )


def _voice_handler(settings: Settings) -> Optional[VoiceHandler]:
    """Only when the provider is usable (local, or its API key is set)."""
    if not FeatureFlags(settings).voice_messages_enabled:
        return None
    return VoiceHandler(settings)


async def build(
    settings: Settings,
    *,
    application_factory: Optional[ApplicationFactory] = None,
    router_kwargs: Optional[dict[str, Any]] = None,
    budget: Optional[ShutdownBudget] = None,
) -> HarnessApp:
    """Wire the app (DB opened and migrated; nothing else touches the network).

    ``application_factory`` and ``router_kwargs`` exist for tests (a fake PTB
    application; scripted runs).
    """
    _check_config(settings)
    db = DatabaseManager(settings.database_url)
    await db.initialize()
    try:
        return _wire(
            settings,
            db,
            application_factory or (lambda adapter: adapter.build_application()),
            router_kwargs or {},
            budget or ShutdownBudget(),
        )
    except BaseException:
        # aiosqlite's threads are non-daemon: left open, they keep the
        # interpreter alive and sys.exit() hangs instead of exiting.
        await db.close()
        raise


def _wire(
    settings: Settings,
    db: DatabaseManager,
    application_factory: ApplicationFactory,
    router_kwargs: dict[str, Any],
    budget: ShutdownBudget,
) -> HarnessApp:
    store = SessionStore.from_settings(db, settings)
    validator = SecurityValidator(
        settings.approved_directory,
        disable_security_patterns=settings.disable_security_patterns,
    )
    # One pool for interactive AND scheduled runs: the cap is global.
    slots = RunSlots(settings.harness_concurrency_cap)
    registry = load_project_registry(
        config_path=settings.projects_config_path,
        approved_directory=settings.approved_directory,
    )
    thread_manager = ProjectThreadManager(
        registry=registry,
        repository=ProjectThreadRepository(db),
        sync_action_interval_seconds=(
            settings.project_threads_sync_action_interval_seconds
        ),
    )
    event_bus = EventBus()
    features = FeatureFlags(settings)

    board = None
    if features.dispatcher_enabled:
        board = load_task_board(settings.hermes_path, settings.hermes_board_db)

    # The router needs adapter.resolve_topic and the bot; the adapter needs
    # the router: build the adapter first, then bind.
    adapter = TelegramAdapter(
        settings,
        None,
        None,
        _voice_handler(settings),
        thread_manager,
        ResumeBridge(board) if board is not None else None,
    )
    application = application_factory(adapter)
    bot = application.bot
    ask_bridge = AskBridge(bot, settings)
    router = TopicRouter(
        bot,
        store,
        slots,
        settings,
        resolve_topic=adapter.resolve_topic,
        ask_bridge=ask_bridge,
        security_validator=validator,
        **router_kwargs,
    )
    adapter.bind(router, ask_bridge)

    jobs = ScheduledJobHandler(event_bus, slots, settings, security_validator=validator)
    notifications = NotificationService(
        event_bus=event_bus,
        bot=bot,
        default_chat_ids=settings.notification_chat_ids or [],
    )
    scheduler = (
        JobScheduler(
            event_bus=event_bus,
            db_manager=db,
            default_working_directory=settings.approved_directory,
        )
        if features.scheduler_enabled
        else None
    )
    dispatcher = None
    if board is not None:
        runner = ClaudeTaskRunner(
            default_working_directory=settings.approved_directory,
            board_db_path=settings.hermes_board_db,
            hermes_cli_path=settings.hermes_path / "cli.py",
            slots=slots,
            settings=settings,
            security_validator=validator,
        )
        dispatcher = DispatcherService(
            board=board,
            runner=runner,
            event_bus=event_bus,
            tick_seconds=settings.dispatcher_tick_seconds,
            stale_seconds=settings.dispatcher_stale_seconds,
            max_per_tick=settings.dispatcher_max_per_tick,
            heartbeat_seconds=settings.dispatcher_heartbeat_seconds,
        )

    app = HarnessApp(
        settings=settings,
        db=db,
        store=store,
        slots=slots,
        thread_manager=thread_manager,
        application=application,
        adapter=adapter,
        router=router,
        event_bus=event_bus,
        notifications=notifications,
        jobs=jobs,
        scheduler=scheduler,
        dispatcher=dispatcher,
        budget=budget,
    )
    application.add_error_handler(app._on_error)
    return app
