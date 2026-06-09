"""The dispatcher loop — the runtime half of the Hermes adoption.

Each tick:
  1. reclaim_stale  — circuit breaker: requeue tasks from crashed workers
  2. claim + run    — drain ready tasks through the injected runner
  3. reconcile      — complete / block / fail based on the run outcome AND any
                      lifecycle transition the worker made itself (via the CLI)
  4. notify         — surface blocks/results/failures to the originating chat

The board is injected (duck-typed) so this module stays free of the Hermes
import path and heavy Claude imports — keeping it cheap to unit-test. The
status constants below MIRROR hermes/board.py; keep them in sync.
"""

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

import structlog

from ..events.bus import EventBus
from ..events.types import AgentResponseEvent

logger = structlog.get_logger()

# Mirror of hermes/board.py lifecycle states.
PENDING = "pending"
CLAIMED = "claimed"
BLOCKED = "blocked"
DONE = "done"
FAILED = "failed"

# How much of a Claude response to retain on the board as the task result.
MAX_RESULT_CHARS = 4000


@dataclass
class RunOutcome:
    """What a TaskRunner reports back to the dispatcher.

    A worker can also transition its own task via the Hermes CLI mid-run (e.g.
    self-block when it needs a human). The dispatcher reconciles against live
    board state, so an outcome here only applies when the task is still claimed.
    """

    content: str = ""
    blocked: bool = False
    block_reason: Optional[str] = None


class TaskRunner(Protocol):
    """Executes a single board task. Production impl: ClaudeTaskRunner."""

    async def run(self, task: Dict[str, Any]) -> RunOutcome: ...


class DispatcherService:
    """Polls the task board and drives tasks to a terminal (or blocked) state."""

    def __init__(
        self,
        *,
        board: Any,
        runner: TaskRunner,
        event_bus: EventBus,
        worker_id: str = "dispatcher",
        tick_seconds: int = 60,
        stale_seconds: int = 600,
        max_per_tick: int = 10,
        heartbeat_seconds: int = 120,
    ) -> None:
        self._board = board
        self._runner = runner
        self._event_bus = event_bus
        self._worker_id = worker_id
        self._tick_seconds = tick_seconds
        self._stale_seconds = stale_seconds
        self._max_per_tick = max_per_tick
        self._heartbeat_seconds = heartbeat_seconds
        self._running = False
        self._loop_task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._loop_task = asyncio.create_task(self._loop())
        logger.info(
            "Dispatcher started",
            tick_seconds=self._tick_seconds,
            stale_seconds=self._stale_seconds,
        )

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        if self._loop_task:
            self._loop_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._loop_task
        logger.info("Dispatcher stopped")

    async def _loop(self) -> None:
        while self._running:
            try:
                await self.tick()
            except Exception:
                logger.exception("Dispatcher tick failed")
            # Sleep until the next tick, but wake immediately on stop().
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self._tick_seconds
                )

    # ---------------------------------------------------------------------- tick
    async def tick(self) -> int:
        """Run one dispatch cycle. Returns the number of tasks run this tick."""
        reclaimed = self._board.reclaim_stale(stale_seconds=self._stale_seconds)
        if reclaimed:
            logger.warning(
                "Reclaimed stale tasks from dead workers",
                count=len(reclaimed),
                task_ids=[t["id"] for t in reclaimed],
            )

        run_count = 0
        while run_count < self._max_per_tick:
            task = self._board.claim(worker=self._worker_id)
            if task is None:
                break
            run_count += 1
            await self._run_task(task)

        # Fan-in (piece 4): settle parent tasks whose children have all
        # finished, and send one consolidated completion per parent.
        settle = getattr(self._board, "settle_parents", None)
        if callable(settle):
            for parent in settle():
                await self._notify_parent_settled(parent)
        return run_count

    # ------------------------------------------------------------------ run task
    async def _run_task(self, task: Dict[str, Any]) -> None:
        task_id = task["id"]
        logger.info("Running task", task_id=task_id, title=task.get("title"))

        heartbeat = asyncio.create_task(self._heartbeat(task_id))
        error: Optional[Exception] = None
        outcome: Optional[RunOutcome] = None
        try:
            outcome = await self._runner.run(task)
        except Exception as exc:  # noqa: BLE001 — we record, never crash the loop
            error = exc
            logger.exception("Task runner raised", task_id=task_id)
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat

        final = self._reconcile(task_id, outcome, error)
        await self._notify(task, final, outcome)

    def _reconcile(
        self,
        task_id: str,
        outcome: Optional[RunOutcome],
        error: Optional[Exception],
    ) -> Optional[Dict[str, Any]]:
        """Apply the terminal transition — but only if the worker didn't already.

        If the worker self-transitioned (e.g. ran `cli.py block` mid-task), the
        status is no longer 'claimed' and we respect its choice verbatim.
        """
        current = self._board.get(task_id)
        if current is None or current["status"] != CLAIMED:
            return current

        if error is not None:
            return self._board.fail(task_id, f"{type(error).__name__}: {error}")
        if outcome is not None and outcome.blocked:
            reason = outcome.block_reason or "needs human input"
            return self._board.block(task_id, reason)
        result = None
        if outcome is not None and outcome.content:
            result = {"content": outcome.content[:MAX_RESULT_CHARS]}
        return self._board.complete(task_id, result=result)

    async def _heartbeat(self, task_id: str) -> None:
        """Prove the in-flight task is alive so the circuit breaker leaves it be."""
        while True:
            await asyncio.sleep(self._heartbeat_seconds)
            try:
                self._board.heartbeat(task_id)
            except Exception:
                # Task is no longer claimed (worker self-transitioned) — stop.
                return

    # ------------------------------------------------------------------ notify
    async def _notify(
        self,
        task: Dict[str, Any],
        final: Optional[Dict[str, Any]],
        outcome: Optional[RunOutcome],
    ) -> None:
        if final is None:
            return
        status = final["status"]
        chat_id, thread_id = self._target(task)
        title = task.get("title", "task")
        # Children of a fanned-out parent roll up into the parent's consolidated
        # completion — their individual done/failed notifications are noise. A
        # block always notifies, though, or the parent could never settle.
        is_child = bool(task.get("parent_id"))

        if status == BLOCKED:
            reason = final.get("block_reason") or "I need your input to continue."
            text = (
                f"🔔 Paused — I need your input to continue:\n\n"
                f"{title}\n\n{reason}\n\n(task {task['id'][:8]})"
            )
            if chat_id is None:
                logger.warning(
                    "Blocked task has no chat target; human cannot be asked",
                    task_id=task["id"],
                )
                return
        elif status == DONE:
            if is_child:
                return
            text = (outcome.content if outcome and outcome.content else "") or (
                self._result_content(final)
            )
            if not text:
                return  # nothing worth sending (e.g. silent internal task)
        elif status == FAILED:
            if is_child:
                return
            text = (
                f"⚠️ Task failed: {title}\n\n"
                f"{final.get('error') or 'unknown error'}\n\n(task {task['id'][:8]})"
            )
        else:
            # PENDING after a retry-with-backoff, or any non-terminal state.
            return

        if chat_id is None:
            return

        await self._event_bus.publish(
            AgentResponseEvent(
                chat_id=chat_id,
                text=text,
                parse_mode=None,  # reasons/errors may contain <, &, etc.
                message_thread_id=thread_id,
            )
        )

    async def _notify_parent_settled(self, parent: Dict[str, Any]) -> None:
        """Send one consolidated completion for a fanned-out parent task."""
        chat_id, thread_id = self._target(parent)
        if chat_id is None:
            return
        result = parent.get("result") if isinstance(parent.get("result"), dict) else {}
        counts = result.get("subtasks", {}) if isinstance(result, dict) else {}
        total = counts.get("total", 0)
        done = counts.get("done", 0)
        failed = counts.get("failed", 0)
        cancelled = counts.get("cancelled", 0)

        summary = f"{done}/{total} subtasks complete"
        if failed:
            summary += f" · {failed} failed"
        if cancelled:
            summary += f" · {cancelled} cancelled"
        lines = [f"✅ Done: {parent.get('title', 'task')}", summary]
        for child in (result.get("children", []) if isinstance(result, dict) else []):
            mark = {DONE: "✓", FAILED: "✗"}.get(child.get("status"), "–")
            lines.append(f"{mark} {child.get('title', '')}")

        await self._event_bus.publish(
            AgentResponseEvent(
                chat_id=chat_id,
                text="\n".join(lines),
                parse_mode=None,
                message_thread_id=thread_id,
            )
        )

    @staticmethod
    def _result_content(task: Dict[str, Any]) -> str:
        result = task.get("result")
        if isinstance(result, dict):
            return str(result.get("content", ""))
        return ""

    @staticmethod
    def _target(task: Dict[str, Any]) -> tuple[Optional[int], Optional[int]]:
        """Resolve (chat_id, message_thread_id) for delivery.

        chat_id is a board column; the optional Telegram topic thread id rides
        in the payload under the reserved key `_thread_id`.
        """
        raw = task.get("chat_id")
        if raw in (None, "", 0, "0"):
            return (None, None)
        try:
            chat_id: Optional[int] = int(raw)
        except (TypeError, ValueError):
            return (None, None)

        thread_id: Optional[int] = None
        payload = task.get("payload")
        if isinstance(payload, dict) and payload.get("_thread_id") is not None:
            try:
                thread_id = int(payload["_thread_id"])
            except (TypeError, ValueError):
                thread_id = None
        return (chat_id, thread_id)
