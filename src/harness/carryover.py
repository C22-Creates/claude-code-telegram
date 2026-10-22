"""Carry-over summary used when a topic's session is rotated."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Callable, Optional

import structlog
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

from src.harness.run_support import bounded
from src.harness.sdk_lifecycle import await_cancelled, reap_wedged_child

logger = structlog.get_logger()

CARRYOVER_PROMPT = (
    "Summarize this conversation for a fresh session in ≤300 words: "
    "open tasks, decisions, names, IDs."
)
CARRYOVER_TIMEOUT_SECONDS = 120.0
CARRYOVER_CLOSE_TIMEOUT_SECONDS = 2.0

QueryFn = Callable[..., AsyncIterator[Any]]
TransportFactory = Callable[..., Any]


async def _collect(stream: AsyncIterator[Any]) -> str:
    last_text = ""
    result: Optional[str] = None
    async for message in stream:
        if isinstance(message, AssistantMessage):
            text = "".join(b.text for b in message.content if isinstance(b, TextBlock))
            if text:
                last_text = text
        elif isinstance(message, ResultMessage) and message.result:
            result = message.result
    return (result or last_text).strip()


async def summarize_for_carryover(
    session_id: str,
    cwd: Path,
    settings: Any,
    query_fn: QueryFn = query,
    timeout: float = CARRYOVER_TIMEOUT_SECONDS,
    *,
    close_timeout: float = CARRYOVER_CLOSE_TIMEOUT_SECONDS,
    transport_factory: TransportFactory = SubprocessCLITransport,
) -> str:
    """Summarize ``session_id`` via a forked one-shot query; "" on any failure.

    ``fork_session=True`` keeps the old session untouched. Rotation proceeds
    without a summary rather than failing the user's message.

    This runs inside the topic's run slot before any watchdog, so it must
    return within ``timeout + 2 * close_timeout`` whatever the CLI does: a
    timeout cancels the query without waiting on the cancellation beyond a
    bound (a wedged child can swallow it, as on 2026-07-29), closes the stream
    (bounded) and reaps the CLI child, which is reachable because the
    transport is built here rather than inside ``query()``.
    """
    options = ClaudeAgentOptions(
        resume=session_id,
        fork_session=True,
        model=settings.harness_carryover_model,
        max_turns=1,
        allowed_tools=[],
        # No built-in tools at all: permission-free ones (Read/Glob/Grep) would
        # otherwise run with no can_use_tool guard on a resumed full session.
        tools=[],
        cwd=str(cwd),
        setting_sources=[],
    )
    transport: Any = None
    try:
        transport = transport_factory(prompt=CARRYOVER_PROMPT, options=options)
        stream = query_fn(prompt=CARRYOVER_PROMPT, options=options, transport=transport)
        task = asyncio.ensure_future(_collect(stream))
    except Exception as exc:  # noqa: BLE001 - any failure degrades to ""
        logger.warning(
            "Carry-over summary failed", session_id=session_id, error=repr(exc)
        )
        return ""
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
    except BaseException:  # caller cancelled: do not leave the query running
        task.cancel()
        raise
    if not done:
        logger.warning("Carry-over summary timed out", session_id=session_id)
        await _abandon(task, stream, transport, close_timeout)
        return ""
    if task.exception() is not None:
        logger.warning(
            "Carry-over summary failed",
            session_id=session_id,
            error=repr(task.exception()),
        )
        return ""
    words = task.result().split()
    return " ".join(words[: settings.harness_carryover_max_words])


async def _abandon(
    task: "asyncio.Future[str]",
    stream: AsyncIterator[Any],
    transport: Any,
    close_timeout: float,
) -> None:
    """Cancel, close and reap a timed-out summary; every wait is bounded."""
    task.cancel()
    if await await_cancelled(task, "carryover_timeout", close_timeout):
        # ``async for`` does not close its iterator on cancellation (PEP 533):
        # aclose() runs the SDK's own cleanup, which terminates the CLI.
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            await bounded(aclose(), close_timeout, "carryover_aclose")
    # A no-op when the child already exited; else SIGKILL it and its MCPs.
    reap_wedged_child(SimpleNamespace(_transport=transport), "carryover_timeout")
