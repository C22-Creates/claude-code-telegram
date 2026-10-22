"""TopicRouter: per-topic run lifecycle (start, steer, cancel, rotate).

Each Telegram forum topic has a ``TopicWorker``. A message to an idle topic
starts a *flow*: status message -> run slot -> session rotation -> ``AgentRun``
-> answer + footer -> persist session. A message to a busy topic is offered to
the ask bridge (a typed answer to a pending question), else injected into the
live run and acknowledged with a 👀 reaction -- no new status message. When the
topic is busy but nothing can take the message (queued for a slot, starting,
or the run finished while the write was in flight) the message reacts with
``QUEUED_BEHIND`` and waits for the current flow, then starts its own run.

``handle()`` awaits the whole flow it starts (simple to test). The Telegram
adapter (Task 8) must therefore spawn it as a task per update instead of
awaiting it inline. A handle() that only steers returns as soon as the inject
lands.

One ``asyncio.Lock`` per topic guards only the idle/busy decision, never the
run, so follow-ups stay injectable while a run is in progress.
"""

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

import structlog
from claude_agent_sdk import CanUseTool, ClaudeAgentOptions, PermissionResultAllow

from src.harness.agent_run import AgentRun
from src.harness.carryover import summarize_for_carryover
from src.harness.prompt_builder import build_options
from src.harness.run_slots import RunSlots
from src.harness.run_support import Prompt, first_line
from src.harness.session_store import SessionStore
from src.harness.status_message import StatusMessage
from src.harness.types import Inbound, Outcome, ProgressEvent, RunResult

logger = structlog.get_logger()

STARTING_TEXT = "⚙️ starting"
QUEUED_TEXT = "⏳ queued"
EYES = "👀"
# Telegram only accepts reactions from a fixed emoji set, which has no ⏳.
QUEUED_BEHIND = "🫡"
CANCELLED = "cancelled"  # footer: "⚠️ stopped (cancelled) · {elapsed}"
NEW_SESSION = "new session"
CARRYOVER_HEADER = "Context from previous session:"


@dataclass(frozen=True)
class TopicContext:
    """Where a topic's runs execute and which agent identity they use."""

    topic_key: str
    cwd: Path
    agent: str | None


class AskBridgeLike(Protocol):
    """What the router needs from the ask bridge (implemented in Task 7)."""

    async def try_answer_with_text(self, topic_key: str, text: str) -> bool:
        """Consume ``text`` as the answer to a pending question; True if used."""
        ...

    async def cancel_topic(self, topic_key: str) -> None:
        """Drop any pending question for the topic (its run has ended)."""
        ...

    def make_can_use_tool(
        self, topic_key: str, chat_id: int, thread_id: int | None, run: "RunRef"
    ) -> CanUseTool:
        """Build the run's ``can_use_tool`` callback (``run`` is a ``RunRef``)."""
        ...


class RunRef:
    """Late-bound handle to a topic's ``AgentRun``.

    ``can_use_tool`` needs the run (to pause the watchdog while a question is
    open), but the run needs options that already contain ``can_use_tool``.
    The callback therefore gets this ref, built first; the router binds it to
    the ``AgentRun`` right after constructing it, before ``run()`` starts.
    Calls before binding are no-ops.
    """

    def __init__(self) -> None:
        self._run: Any = None

    def bind(self, run: Any) -> None:
        self._run = run

    @property
    def run(self) -> Any:
        return self._run

    def pause_watchdog(self) -> None:
        if self._run is not None:
            self._run.pause_watchdog()

    def resume_watchdog(self) -> None:
        if self._run is not None:
            self._run.resume_watchdog()

    async def interrupt(self, reason: str) -> None:
        if self._run is not None:
            await self._run.interrupt(reason)


async def allow_all(tool_name: str, tool_input: dict, context: Any) -> Any:
    """``can_use_tool`` used when no ask bridge is wired."""
    return PermissionResultAllow()


class TopicWorker:
    """Mutable per-topic state; owned and mutated only by the router."""

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.flow_done: asyncio.Event | None = None  # set while a flow is live
        self.run: Any = None  # the AgentRun while run() is executing
        self.cancel_requested = False
        self.reset_requested = False  # /new landed before the run started
        self.run_started = False  # the flow's run was published (may have returned)
        self.discard_session = False


RunFactory = Callable[
    [ClaudeAgentOptions, Callable[[ProgressEvent], Awaitable[None]]], Any
]
OptionsFactory = Callable[..., ClaudeAgentOptions]
SummarizeFn = Callable[[str, Path, Any], Awaitable[str]]


class TopicRouter:
    """Routes inbound messages to per-topic runs under the global slot cap."""

    def __init__(
        self,
        bot: Any,
        store: SessionStore,
        slots: RunSlots,
        settings: Any,
        resolve_topic: Callable[[Inbound], TopicContext],
        ask_bridge: AskBridgeLike | None = None,
        *,
        run_factory: RunFactory | None = None,
        options_factory: OptionsFactory = build_options,
        status_factory: Callable[[Inbound], Any] | None = None,
        summarize_fn: SummarizeFn = summarize_for_carryover,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._bot = bot
        self._store = store
        self._slots = slots
        self._settings = settings
        self._resolve_topic = resolve_topic
        self._ask_bridge = ask_bridge
        self._run_factory = run_factory or self._default_run
        self._options_factory = options_factory
        self._status_factory = status_factory or self._default_status
        self._summarize = summarize_fn
        self._clock = clock
        self._wall_clock = wall_clock
        self._workers: dict[str, TopicWorker] = {}

    # ------------------------------------------------------------- public API

    async def handle(self, inbound: Inbound, *, opening: str | None = None) -> None:
        """Start a run for ``inbound`` or steer the live one; see module doc.

        ``opening`` replaces the default status text (e.g. the voice
        "🎙️ heard: '…' · starting" line).
        """
        ctx = self._resolve_topic(inbound)
        key = ctx.topic_key
        queued_behind = False
        while True:
            worker = self._workers.setdefault(key, TopicWorker())
            async with worker.lock:
                if self._workers.get(key) is not worker:
                    continue  # defensive: never run on a worker no longer mapped
                if worker.flow_done is None:
                    worker.flow_done = asyncio.Event()
                    break
                done = worker.flow_done
                if await self._try_steer(worker, key, inbound):
                    return
            if not queued_behind:
                queued_behind = True
                await self._react(inbound, QUEUED_BEHIND)
            await done.wait()
        await self._flow(worker, ctx, inbound, opening)

    async def new_session(self, topic_key: str) -> None:
        """``/new``: forget the stored session; the next run starts fresh.

        Once a flow's run has started (live, or returned but not yet saved)
        its session is not saved, otherwise it would silently undo the reset;
        a live run is also stopped. A flow that is still
        queued or setting up (its run not yet published) is flagged so it
        starts fresh (``resume=None``, no carry-over) even if it already read
        the old row.
        """
        worker = self._workers.get(topic_key)
        if worker is not None and worker.flow_done is not None:
            if worker.run_started:
                # Running or already returned: its session predates /new.
                worker.discard_session = True
                if worker.run is not None:
                    await worker.run.interrupt(NEW_SESSION)
            else:
                worker.reset_requested = True
        # Flags first, then clear: if a flagged flow's fresh session were saved
        # before the clear lands, it is wiped (next run merely starts fresh);
        # the pre-/new session can never survive.
        await self._store.clear(topic_key)

    async def cancel(self, topic_key: str) -> bool:
        """``/cancel``: stop the topic's run (or its queued start); False if idle."""
        worker = self._workers.get(topic_key)
        if worker is None or worker.flow_done is None:
            return False
        if worker.run is not None:
            if not worker.run.is_active:
                return False  # already returned; the flow is just finishing
            await worker.run.interrupt(CANCELLED)
        else:
            worker.cancel_requested = True
        return True

    def active_run(self, topic_key: str) -> Any:
        """The topic's executing ``AgentRun``, or None."""
        worker = self._workers.get(topic_key)
        return worker.run if worker is not None else None

    # ------------------------------------------------------------- steering

    async def _try_steer(self, worker: TopicWorker, key: str, inbound: Inbound) -> bool:
        """Route a busy-topic message into the live run; False if it can't."""
        run = worker.run
        if run is None or not run.is_active:
            return False
        if self._ask_bridge is not None:
            try:
                if await self._ask_bridge.try_answer_with_text(key, inbound.text):
                    return True
            except Exception:  # noqa: BLE001 - a bridge bug must not drop text
                logger.exception("harness.ask_bridge_failed", topic=key)
        try:
            await run.inject(self._build_prompt(inbound))
        except RuntimeError as exc:  # run finished/stopping: start a new one
            logger.info("harness.inject_refused", topic=key, reason=str(exc))
            return False
        except Exception:  # noqa: BLE001 - write failed; run it separately
            logger.exception("harness.inject_failed", topic=key)
            return False
        await self._react(inbound, EYES)
        return True

    # ----------------------------------------------------------------- flow

    async def _flow(
        self,
        worker: TopicWorker,
        ctx: TopicContext,
        inbound: Inbound,
        opening: str | None,
    ) -> None:
        key = ctx.topic_key
        status = self._status_factory(inbound)
        queued = self._slots.would_wait()
        await self._open_status(status, opening, queued)
        run_id: str | None = None
        db_status = "error"
        try:
            async with self._slots.acquire("interactive"):
                if queued:
                    await self._note(status, opening or STARTING_TEXT)
                started = self._clock()
                if worker.cancel_requested:
                    result: RunResult | None = None
                else:
                    try:
                        resume, summary = await self._session_for(ctx)
                        run_id = await self._store.start_run(
                            key,
                            inbound.chat_id,
                            inbound.thread_id,
                            _prompt_head(
                                _with_carryover(self._build_prompt(inbound), summary)
                            ),
                        )
                    except Exception as exc:  # noqa: BLE001 - store/DB failure
                        logger.exception("harness.flow_setup_failed", topic=key)
                        result = self._error_result(exc, started)
                    else:
                        result = await self._execute(
                            worker, ctx, inbound, resume, summary, status
                        )
                if result is None:
                    result = RunResult(
                        outcome=Outcome.STOPPED,
                        text="",
                        elapsed_s=0.0,
                        reason=CANCELLED,
                    )
            await self._finish_status(status, result)
            if result.session_id and not worker.discard_session:
                try:
                    await self._store.save(key, result.session_id, ctx.cwd)
                except Exception:  # noqa: BLE001 - the answer is already out
                    logger.exception("harness.session_save_failed", topic=key)
            db_status = result.outcome.value
        except asyncio.CancelledError:
            db_status = "stopped"
            raise
        finally:
            if run_id is not None:
                await self._end_run(run_id, db_status)
            self._close_flow(worker)

    async def _execute(
        self,
        worker: TopicWorker,
        ctx: TopicContext,
        inbound: Inbound,
        resume: str | None,
        summary: str,
        status: Any,
    ) -> RunResult | None:
        """Build and run the AgentRun; None if cancelled before it started.

        Everything from the reset check to ``run()`` entry is synchronous, so a
        /new or /cancel either lands before (flags honoured here) or after the
        run is published (handled via ``worker.run``).
        """
        started = self._clock()
        try:
            if worker.reset_requested:  # /new during queue/setup: start fresh
                resume, summary = None, ""
            prompt = _with_carryover(self._build_prompt(inbound), summary)
            ref = RunRef()
            can_use_tool: CanUseTool = (
                self._ask_bridge.make_can_use_tool(
                    ctx.topic_key, inbound.chat_id, inbound.thread_id, ref
                )
                if self._ask_bridge is not None
                else allow_all
            )
            options = self._options_factory(
                ctx.cwd, ctx.agent, self._settings, resume, can_use_tool
            )
            run = self._run_factory(options, status.update)
            ref.bind(run)
            if worker.cancel_requested:  # /cancel landed during setup
                return None
            # No await between publishing the run and run() entry, so /cancel
            # and inject() always see an active run once it is visible.
            worker.run_started = True
            worker.run = run
            return await run.run(prompt)
        except Exception as exc:  # noqa: BLE001 - AgentRun.run should not raise
            logger.exception("harness.run_crashed", topic=ctx.topic_key)
            return self._error_result(exc, started)
        finally:
            worker.run = None
            await self._clear_asks(ctx.topic_key)

    async def _clear_asks(self, topic_key: str) -> None:
        """Best effort: a dead run must not leave a question pending."""
        if self._ask_bridge is None:
            return
        try:
            await self._ask_bridge.cancel_topic(topic_key)
        except Exception:  # noqa: BLE001 - cleanup must not mask the result
            logger.exception("harness.ask_cancel_failed", topic=topic_key)

    def _error_result(self, exc: BaseException, started: float) -> RunResult:
        return RunResult(
            outcome=Outcome.ERROR,
            text="",
            elapsed_s=self._clock() - started,
            reason=first_line(str(exc)) or type(exc).__name__,
        )

    def _close_flow(self, worker: TopicWorker) -> None:
        """Mark the topic idle and wake messages queued behind the flow.

        The worker is kept even after a crash: every flow builds a fresh
        AgentRun (hence a fresh client), and keeping one worker per topic means
        no message can ever hold a stale, unmapped worker (two flows at once).
        """
        done = worker.flow_done
        worker.flow_done = None
        worker.run = None
        worker.cancel_requested = False
        worker.reset_requested = False
        worker.run_started = False
        worker.discard_session = False
        if done is not None:
            done.set()

    async def _session_for(self, ctx: TopicContext) -> tuple[str | None, str]:
        """(resume id, carry-over summary). Rotation drops resume."""
        row = await self._store.get(ctx.topic_key)
        if row is None:
            return None, ""
        stale = row.cwd != str(ctx.cwd) or self._store.needs_rotation(
            row, self._wall_clock()
        )
        if not stale:
            return row.session_id, ""
        logger.info("harness.rotate_session", topic=ctx.topic_key, old=row.session_id)
        try:
            summary = await self._summarize(
                row.session_id, Path(row.cwd), self._settings
            )
        except Exception:  # noqa: BLE001 - rotation proceeds without a summary
            logger.exception("harness.carryover_failed", topic=ctx.topic_key)
            summary = ""
        return None, (summary or "").strip()

    def _build_prompt(self, inbound: Inbound) -> Prompt:
        """The user's message as a prompt. Task 8 extends this (image blocks).

        Attachments are listed by path so the agent can open them.
        """
        lines = [inbound.text]
        lines += [f"[attached {a.kind}: {a.path}]" for a in inbound.attachments]
        return "\n\n".join(line for line in lines if line)

    # ------------------------------------------------------- telegram helpers

    async def _open_status(
        self, status: Any, opening: str | None, queued: bool
    ) -> None:
        if queued:
            text = f"{opening}\n{QUEUED_TEXT}" if opening else QUEUED_TEXT
        else:
            text = opening or STARTING_TEXT
        try:
            await status.open(text)
        except Exception:  # noqa: BLE001 - the run proceeds; footer is re-sent
            logger.exception("harness.status_open_failed")

    async def _note(self, status: Any, text: str) -> None:
        try:
            await status.note(text)
        except Exception:  # noqa: BLE001
            logger.exception("harness.status_note_failed")

    async def _finish_status(self, status: Any, result: RunResult) -> None:
        try:
            await status.finish(result)
        except Exception:  # noqa: BLE001 - delivery is best-effort
            logger.exception("harness.status_finish_failed")

    async def _react(self, inbound: Inbound, emoji: str) -> None:
        try:
            await self._bot.set_message_reaction(
                chat_id=inbound.chat_id, message_id=inbound.message_id, reaction=emoji
            )
        except Exception as exc:  # noqa: BLE001 - reactions are cosmetic
            logger.warning("harness.reaction_failed", emoji=emoji, error=str(exc))

    async def _end_run(self, run_id: str, status: str) -> None:
        try:
            await self._store.end_run(run_id, status)
        except Exception:  # noqa: BLE001
            logger.exception("harness.end_run_failed", run_id=run_id)

    # -------------------------------------------------------------- defaults

    def _default_run(
        self,
        options: ClaudeAgentOptions,
        on_progress: Callable[[ProgressEvent], Awaitable[None]],
    ) -> AgentRun:
        return AgentRun(options, on_progress, self._settings)

    def _default_status(self, inbound: Inbound) -> StatusMessage:
        return StatusMessage(
            self._bot,
            inbound.chat_id,
            inbound.thread_id,
            inbound.message_id,
            self._settings,
            self._clock,
        )


def _with_carryover(prompt: Prompt, summary: str) -> Prompt:
    """Prefix the carry-over summary; unchanged when there is none."""
    if not summary:
        return prompt
    header = f"{CARRYOVER_HEADER}\n{summary}"
    if isinstance(prompt, str):
        return f"{header}\n\n{prompt}"
    return [{"type": "text", "text": header}, *prompt]


def _prompt_head(prompt: Prompt) -> str:
    if isinstance(prompt, str):
        return prompt
    return " ".join(b.get("text", "") for b in prompt if b.get("type") == "text")
