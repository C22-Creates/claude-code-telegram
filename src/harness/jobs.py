"""Unattended runs: scheduled (cron) jobs and Hermes board tasks.

Both go through ``AgentRun`` with a fresh session each time (``resume=None``)
and the ``scheduled`` slot priority, so they count toward the global
concurrency cap but never pre-empt an interactive run: a freed slot goes to
interactive waiters first. They share the interactive 60-minute run cap, never
touch a topic's ``TopicRouter`` worker (so nothing is ever injected into
Carl's live run), and post no status message.
"""

import asyncio
import dataclasses
from pathlib import Path
from typing import Any, Awaitable, Callable, Set

import structlog
from claude_agent_sdk import (
    ClaudeSDKClient,
    PermissionResultAllow,
    PermissionResultDeny,
)

from src.config.settings import Settings
from src.events.bus import Event, EventBus
from src.events.types import AgentResponseEvent, ScheduledEvent
from src.harness.agent_run import AgentRun
from src.harness.prompt_builder import ASK_TOOL, build_options
from src.harness.run_slots import RunSlots
from src.harness.types import Outcome, ProgressEvent, RunResult
from src.harness.util import footer

logger = structlog.get_logger()

NO_ONE_TO_ASK = "no one is available to answer; proceed with your best judgment"

RunJobFn = Callable[..., Awaitable[RunResult]]


async def unattended_can_use_tool(
    tool_name: str, tool_input: dict, context: Any
) -> PermissionResultAllow | PermissionResultDeny:
    """Allow every tool except AskUserQuestion.

    A job has no human to answer a question, and an unanswered question would
    hang until the stall watchdog fires. Denying with ``interrupt=False`` tells
    the model to carry on with its best judgment. (``build_options`` is given a
    callback rather than ``None`` so ``AskUserQuestion`` is routed here at all.)
    """
    if tool_name == ASK_TOOL:
        return PermissionResultDeny(message=NO_ONE_TO_ASK, interrupt=False)
    return PermissionResultAllow()


async def _drop_progress(event: ProgressEvent) -> None:
    logger.debug("harness.job_progress", kind=event.kind, tool=event.tool)


async def run_job(
    prompt: str,
    cwd: Path,
    agent: str | None,
    slots: RunSlots,
    settings: Settings,
    model: str | None = None,
    *,
    client_factory: Callable[[Any], Any] = ClaudeSDKClient,
    run_kwargs: dict[str, Any] | None = None,
) -> RunResult:
    """Run one unattended job in a fresh session; never raises.

    Waits for a ``scheduled`` slot, builds options with ``resume=None`` (and
    *model* if given), and returns the ``RunResult``. Setup errors become an
    ERROR result. ``run_kwargs`` (clock/sleep/timeouts) go to ``AgentRun``.
    """
    try:
        options = build_options(
            cwd, agent, settings, resume=None, can_use_tool=unattended_can_use_tool
        )
        if model:
            options = dataclasses.replace(options, model=model)
        async with slots.acquire("scheduled"):
            run = AgentRun(
                options, _drop_progress, settings, client_factory, **(run_kwargs or {})
            )
            return await run.run(prompt)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - a job failure must not escape
        logger.exception("harness.job_failed", cwd=str(cwd))
        reason = (str(exc).splitlines() or [""])[0] or type(exc).__name__
        return RunResult(outcome=Outcome.ERROR, text="", elapsed_s=0.0, reason=reason)


def outcome_text(result: RunResult) -> str:
    """Text to post: the answer alone when DONE, else answer plus footer."""
    if result.outcome is Outcome.DONE:
        return result.text
    prefix = f"{result.text}\n\n" if result.text else ""
    return prefix + footer(result)


class ScheduledJobHandler:
    """Runs ``ScheduledEvent``s through ``run_job`` and posts the answer.

    The bus awaits all handlers of an event together, so each job is spawned as
    a tracked task: the bus is never blocked for the job's duration and jobs
    can run concurrently up to the global cap (``RunSlots`` does the limiting).
    """

    def __init__(
        self,
        event_bus: EventBus,
        slots: RunSlots,
        settings: Settings,
        *,
        run_job_fn: RunJobFn = run_job,
    ) -> None:
        self.event_bus = event_bus
        self._slots = slots
        self._settings = settings
        self._run_job = run_job_fn
        self._tasks: Set[asyncio.Task[Any]] = set()

    def register(self) -> None:
        self.event_bus.subscribe(ScheduledEvent, self.handle_scheduled)

    async def handle_scheduled(self, event: Event) -> None:
        if not isinstance(event, ScheduledEvent):
            return
        logger.info(
            "Processing scheduled event through harness",
            job_id=event.job_id,
            job_name=event.job_name,
        )
        task = asyncio.create_task(
            self._run_scheduled(event),
            name=f"scheduled:{event.job_id or event.job_name or event.id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        await asyncio.sleep(0)

    async def wait_idle(self) -> None:
        """Wait for in-flight jobs (shutdown and tests)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def _run_scheduled(self, event: ScheduledEvent) -> None:
        prompt = event.prompt
        if event.skill_name:
            prompt = (
                f"/{event.skill_name}\n\n{prompt}" if prompt else f"/{event.skill_name}"
            )
        result = await self._run_job(
            prompt, event.working_directory, None, self._slots, self._settings
        )
        text = outcome_text(result)
        if not text:
            return
        # chat_id 0 = broadcast to default chats when no targets were given.
        for chat_id in event.target_chat_ids or [0]:
            await self.event_bus.publish(
                AgentResponseEvent(
                    chat_id=chat_id,
                    text=text,
                    message_thread_id=event.message_thread_id,
                    originating_event_id=event.id,
                )
            )

    def _task_done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("Scheduled job handler failed", task_name=task.get_name())
