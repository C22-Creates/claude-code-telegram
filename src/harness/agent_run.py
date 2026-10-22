"""AgentRun: one ClaudeSDKClient per run, in streaming-input mode.

A run sends one prompt, may receive injected follow-ups while it is live, and
always ends in a ``RunResult`` -- it never hangs and never raises (except to
propagate the caller's own cancellation). Three tasks cooperate:

* the reader drains ``receive_messages()`` and decides completion;
* the supervisor ticks once a second for the stall watchdog, heartbeats, the
  run cap and the end-turn grace period;
* ``run()`` itself waits for either, performs any interrupt, and tears down
  with hard bounds (a wedged CLI child is reaped, never awaited forever).
"""

import asyncio
import time
from typing import Any, Awaitable, Callable

import structlog
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
)
from claude_agent_sdk.types import StreamEvent

from src.config.settings import Settings
from src.harness.run_support import Prompt, as_query, bounded, first_line, join_parts
from src.harness.sdk_lifecycle import (
    CANCELLED_TASK_CLEANUP_TIMEOUT,
    await_cancelled,
    is_retryable_error,
    reap_wedged_child,
    turn_signal,
)
from src.harness.types import Outcome, ProgressEvent, RunResult

logger = structlog.get_logger()

INTERRUPT_TIMEOUT = 10.0  # the spike measured ~0.05s; 10s means wedged
PROGRESS_TIMEOUT = 10.0  # a hung Telegram call must not freeze the run
UNAVAILABLE_MCP_STATUSES = ("failed", "needs-auth")


class AgentRun:
    """Runs one agent request to a ``RunResult`` and supports live steering.

    ``on_progress`` receives ``ProgressEvent``s; each call is bounded and its
    exceptions are logged and swallowed, so a Telegram hiccup never kills or
    freezes a run. ``clock``/``sleep`` and the timeouts are injectable so tests
    run in milliseconds.
    """

    def __init__(
        self,
        options: ClaudeAgentOptions,
        on_progress: Callable[[ProgressEvent], Awaitable[None]],
        settings: Settings,
        client_factory: Callable[[ClaudeAgentOptions], Any] = ClaudeSDKClient,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        tick_seconds: float = 1.0,
        interrupt_timeout: float = INTERRUPT_TIMEOUT,
        cleanup_timeout: float = CANCELLED_TASK_CLEANUP_TIMEOUT,
        progress_timeout: float = PROGRESS_TIMEOUT,
    ) -> None:
        self._options = options
        self._on_progress = on_progress
        self._settings = settings
        self._client_factory = client_factory
        self._clock = clock
        self._sleep = sleep
        self._tick = tick_seconds
        self._interrupt_timeout = interrupt_timeout
        self._cleanup_timeout = cleanup_timeout
        self._progress_timeout = progress_timeout
        self.session_id: str | None = None
        self._active = False
        self._reset_run()

    # ------------------------------------------------------------ public API

    @property
    def is_active(self) -> bool:
        return self._active

    async def run(self, prompt: Prompt) -> RunResult:
        """Run *prompt* to completion; always returns a ``RunResult``."""
        if self._active:
            raise RuntimeError("AgentRun is already running")
        self._active = True
        self._reset_run()
        self._started_at = self._clock()
        self._queries_sent = 1  # the prompt; a retry's re-send is not counted
        try:
            for attempt in (1, 2):
                self._reset_attempt()
                await self._attempt(prompt)
                if attempt == 1 and self._should_retry():
                    logger.warning("harness.retry", error=str(self._error))
                    # Nothing was read: re-send what the dead client swallowed.
                    self._buffered[:0] = self._attempt_injections
                    continue
                break
        except asyncio.CancelledError:
            self._log_end("cancelled", "caller cancelled")
            raise
        except Exception as exc:  # noqa: BLE001 - run() must not raise
            logger.exception("harness.internal_error")
            self._finish(Outcome.ERROR, first_line(str(exc)) or type(exc).__name__)
        finally:
            self._active = False
            self._client = None
        result = RunResult(
            outcome=self._outcome or Outcome.ERROR,
            text=self._result_text(),
            elapsed_s=self._elapsed(),
            reason=self._reason,
            session_id=self.session_id,
            tool_calls=self._tool_calls,
        )
        self._log_end(result.outcome.value, result.reason)
        return result

    async def inject(self, content: Prompt) -> None:
        """Send a follow-up into the live run (consumed at the next model call).

        Before the initial prompt is written the follow-up is buffered and sent
        right after it. Raises RuntimeError when the run cannot take it (idle,
        stopping, or finished before the write landed) so the caller can
        re-queue the message instead of losing it.
        """
        if not self._active or self._stop_reason is not None:
            raise RuntimeError("no active run to inject into")
        if self._finished.is_set():
            raise RuntimeError("run already finished")
        # Counted before the write: a ResultMessage read while the write is in
        # flight must not complete the run under this injection.
        self._pending_injections += 1
        self._queries_sent += 1
        client = self._client
        if client is None:  # still connecting; _open() flushes the buffer
            self._buffered.append(content)
            return
        try:
            await client.query(as_query(content))
        except BaseException:
            self._pending_injections = max(0, self._pending_injections - 1)
            self._queries_sent -= 1
            raise
        self._attempt_injections.append(content)
        if self._finished.is_set():
            raise RuntimeError("run finished before the injection was read")

    async def interrupt(self, reason: str) -> None:
        """Ask the run to stop with *reason* (outcome STOPPED). Non-blocking.

        ``run()`` performs the interrupt and teardown in its own task, so this is
        safe to call from callbacks the SDK is awaiting.
        """
        if self._active and self._stop_reason is None:
            self._stop_reason = reason
            self._stop_requested.set()

    def pause_watchdog(self) -> None:
        """Freeze the stall clock (e.g. while a question awaits Carl)."""
        self._paused = True

    def resume_watchdog(self) -> None:
        """Restart the stall clock from now. The run cap was never paused."""
        self._paused = False
        self._last_event_at = self._clock()

    # ------------------------------------------------------------- lifecycle

    def _reset_run(self) -> None:
        self._client: Any = None
        self._started_at = self._last_event_at = 0.0
        self._answers: list[str] = []  # one per completed turn
        self._turn_text: list[str] = []  # main-agent text of the current call
        self._tool_calls = self._turn = self._queries_sent = 0
        self._pending_injections = 0
        self._buffered: list[Prompt] = []
        self._last_tool: str | None = None
        self._paused = False
        self._stop_reason: str | None = None
        self._stop_requested = asyncio.Event()
        self._finished = asyncio.Event()
        self._outcome: Outcome | None = None
        self._reason = ""

    def _reset_attempt(self) -> None:
        self._finished = asyncio.Event()
        self._outcome, self._reason = None, ""
        self._error: BaseException | None = None
        self._attempt_injections: list[Prompt] = []
        self._events_seen = 0
        self._turn_ended = False
        self._interrupt_sent = False
        self._wedged = False

    def _should_retry(self) -> bool:
        return (
            self._error is not None
            and self._events_seen == 0
            and self._stop_reason is None
            and is_retryable_error(self._error)
        )

    def _finish(self, outcome: Outcome, reason: str = "") -> None:
        if not self._finished.is_set():
            self._outcome, self._reason = outcome, reason
            self._finished.set()

    def _fail(self, exc: BaseException) -> None:
        """An SDK/CLI failure -> ERROR, unless the harness was already stopping."""
        logger.error("harness.sdk_error", error=str(exc), error_type=type(exc).__name__)
        if self._stop_reason is not None:
            self._finish(Outcome.STOPPED, self._stop_reason)
            return
        self._error = exc
        self._finish(Outcome.ERROR, first_line(str(exc)) or type(exc).__name__)

    async def _attempt(self, prompt: Prompt) -> None:
        client = reader = supervisor = None
        try:
            client = self._client_factory(self._options)
            if not await self._start(client, prompt):
                return
            self._last_event_at = self._clock()
            reader = asyncio.create_task(self._read(client))
            supervisor = asyncio.create_task(self._supervise())
            # Also wake if the reader or supervisor dies without finishing:
            # a silently dead supervisor would mean no stall/cap protection.
            waiters = [
                asyncio.ensure_future(self._stop_requested.wait()),
                asyncio.ensure_future(self._finished.wait()),
            ]
            try:
                await asyncio.wait(
                    [*waiters, reader, supervisor], return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                for waiter in waiters:
                    waiter.cancel()
            if not self._finished.is_set() and self._stop_reason is not None:
                await self._interrupt_client(client)
            self._finish(Outcome.ERROR, "agent run ended unexpectedly")
        except Exception as exc:  # noqa: BLE001 - e.g. client_factory failed
            self._fail(exc)
        finally:
            self._client = None
            await self._teardown(client, reader, supervisor)

    async def _start(self, client: Any, prompt: Prompt) -> bool:
        """Connect and send the prompt, racing a stop request and a timeout."""
        opener = asyncio.ensure_future(self._open(client, prompt))
        stop = asyncio.ensure_future(self._stop_requested.wait())
        try:
            await asyncio.wait(
                [opener, stop],
                timeout=self._settings.harness_stall_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
        except BaseException:  # caller cancelled: do not leave the opener running
            opener.cancel()
            raise
        finally:
            stop.cancel()
        if opener.done():
            if opener.exception() is None:
                return True
            self._fail(opener.exception())
            return False
        opener.cancel()
        if not await await_cancelled(opener, "connect", self._cleanup_timeout):
            self._wedged = True
        if self._stop_reason is not None:
            self._finish(Outcome.STOPPED, self._stop_reason)
        else:
            self._fail(TimeoutError("agent did not start in time"))
        return False

    async def _open(self, client: Any, prompt: Prompt) -> None:
        await client.connect()
        await client.query(as_query(prompt))
        # Injections that arrived while connecting go out right after the
        # prompt, in order; only then is the client exposed to inject().
        while self._buffered:
            content = self._buffered[0]  # popped only once written (retry-safe)
            await client.query(as_query(content))
            self._attempt_injections.append(self._buffered.pop(0))
        self._client = client

    async def _interrupt_client(self, client: Any) -> None:
        reason = self._stop_reason or "stopped"
        self._interrupt_sent = True
        if await bounded(client.interrupt(), self._interrupt_timeout, "interrupt"):
            # Let the reader see the post-interrupt ResultMessage (session id).
            await bounded(self._finished.wait(), self._interrupt_timeout, "settle")
        else:
            self._wedged = True
        self._finish(Outcome.STOPPED, reason)

    async def _teardown(self, client: Any, reader: Any, supervisor: Any) -> None:
        reason = self._stop_reason or "run_end"
        settled = True
        for task in (supervisor, reader):
            if task is not None and not task.done():
                task.cancel()
                ok = await await_cancelled(task, reason, self._cleanup_timeout)
                settled = settled and (ok or task is supervisor)
        if client is not None:
            if self._wedged or not settled:
                reap_wedged_child(client, reason)
            await bounded(client.disconnect(), self._cleanup_timeout, "disconnect")

    # ---------------------------------------------------------------- reader

    async def _read(self, client: Any) -> None:
        try:
            async for message in client.receive_messages():
                await self._on_message(message)
                if self._finished.is_set():
                    return
        except Exception as exc:  # noqa: BLE001 - becomes ERROR (or STOPPED)
            self._fail(exc)
            return
        self._fail(RuntimeError("agent stream closed without a result"))

    async def _on_message(self, message: Any) -> None:
        now = self._clock()
        self._last_event_at = now
        self._events_seen += 1
        self._log_event(message, now)
        signal = turn_signal(message)
        if signal is not None:
            self._turn_ended = signal
        if isinstance(message, StreamEvent):
            await self._on_stream_event(message)
        elif isinstance(message, AssistantMessage):
            await self._on_assistant(message)
        elif isinstance(message, SystemMessage) and message.subtype == "init":
            await self._on_init(message)
        elif isinstance(message, ResultMessage):
            self._on_result(message)

    async def _on_stream_event(self, event: StreamEvent) -> None:
        if event.parent_tool_use_id is None and (
            (event.event or {}).get("type") == "message_start"
        ):
            self._turn += 1
            self._turn_text = []  # earlier calls' text was narration
            self._pending_injections = 0  # the model is reading everything sent
            await self._emit("turn")

    async def _on_assistant(self, message: AssistantMessage) -> None:
        main = message.parent_tool_use_id is None
        for block in message.content or []:
            if isinstance(block, ToolUseBlock):
                self._tool_calls += 1
                self._last_tool = block.name
                await self._emit("tool", tool=block.name)
            elif isinstance(block, TextBlock) and main and block.text.strip():
                self._turn_text.append(block.text)
                await self._emit("text", detail=block.text)

    async def _on_init(self, message: SystemMessage) -> None:
        data = message.data or {}
        self.session_id = data.get("session_id") or self.session_id
        for server in data.get("mcp_servers") or []:
            if server.get("status") in UNAVAILABLE_MCP_STATUSES:
                await self._emit("mcp_unavailable", detail=server.get("name", "?"))

    def _on_result(self, message: ResultMessage) -> None:
        self.session_id = message.session_id or self.session_id
        if not message.is_error:
            self._close_answer(message.result)
        if self._interrupt_sent:
            self._finish(Outcome.STOPPED, self._stop_reason or "stopped")
        elif message.is_error and message.subtype == "error_max_turns":
            self._finish(Outcome.STOPPED, "hit max turns")
        elif message.is_error:
            self._finish(Outcome.ERROR, first_line(message.result) or message.subtype)
        elif self._pending_injections == 0:
            self._finish(Outcome.DONE)
        # else: an injection is unread; its turn's ResultMessage ends the run.

    def _close_answer(self, result_text: str | None) -> None:
        """Record a finished turn's answer: the CLI's result, else the last call."""
        answer = (result_text or "").strip() or join_parts(self._turn_text)
        if answer:
            self._answers.append(answer)
        self._turn_text = []

    def _result_text(self) -> str:
        """Answers of completed turns, plus the unfinished call's partial text."""
        if self._outcome is Outcome.DONE:
            return join_parts(self._answers)
        return join_parts([*self._answers, join_parts(self._turn_text)])

    # ------------------------------------------------------------ supervisor

    async def _supervise(self) -> None:
        try:
            await self._supervise_loop()
        except Exception as exc:  # noqa: BLE001 - never lose the watchdog quietly
            logger.exception("harness.supervisor_failed")
            self._fail(exc)

    async def _supervise_loop(self) -> None:
        s = self._settings
        marks = sorted(m * 60 for m in s.harness_heartbeat_minutes)
        cap = s.harness_run_cap_seconds
        while not self._finished.is_set() and self._stop_reason is None:
            await self._sleep(self._tick)
            now = self._clock()
            while marks and now - self._started_at >= marks[0]:
                await self._emit("heartbeat", detail=f"{marks.pop(0) // 60}m")
            idle = now - self._last_event_at
            if now - self._started_at >= cap:
                await self.interrupt(f"hit {round(cap / 60)}m cap")
            elif not self._paused and idle >= s.harness_stall_seconds:
                await self.interrupt(f"stalled on {self._last_tool or 'unknown'}")
            elif (
                self._turn_ended
                and self._pending_injections == 0
                and idle >= s.harness_end_turn_grace_seconds
            ):
                logger.warning("harness.end_turn_without_result", idle_s=idle)
                self._close_answer(None)
                self._finish(Outcome.DONE)

    # --------------------------------------------------------------- helpers

    def _elapsed(self) -> float:
        return self._clock() - self._started_at

    async def _emit(self, kind: Any, tool: str | None = None, detail: str = "") -> None:
        event = ProgressEvent(
            kind=kind,
            tool=tool,
            turn=self._turn,
            elapsed_s=self._elapsed(),
            detail=detail,
        )
        await bounded(
            self._on_progress(event), self._progress_timeout, f"on_progress:{kind}"
        )

    def _log_event(self, message: Any, now: float) -> None:
        tool = None
        if isinstance(message, AssistantMessage):
            tool = next(
                (b.name for b in message.content or [] if isinstance(b, ToolUseBlock)),
                None,
            )
        stream_type = (
            (message.event or {}).get("type")
            if isinstance(message, StreamEvent)
            else None
        )
        # Token deltas would drown the log; keep them at debug.
        log = logger.debug if stream_type == "content_block_delta" else logger.info
        log(
            "harness.sdk_event",
            type=type(message).__name__,
            stream_type=stream_type,
            subtype=getattr(message, "subtype", None),
            tool=tool,
            parent_tool_use_id=getattr(message, "parent_tool_use_id", None),
            t=round(now - self._started_at, 3),
        )

    def _log_end(self, outcome: str, reason: str) -> None:
        logger.info(
            "harness.run_end",
            outcome=outcome,
            reason=reason,
            elapsed_s=round(self._elapsed(), 3),
            tool_calls=self._tool_calls,
            queries_sent=self._queries_sent,
            session_id=self.session_id,
        )
