"""Stateless helpers for AgentRun: prompt shaping, bounded awaits, text."""

import asyncio
from typing import Any, AsyncIterator, Awaitable

import structlog

logger = structlog.get_logger()

Prompt = str | list[dict[str, Any]]


def first_line(text: str | None) -> str:
    """First non-blank line of *text*, stripped ("" if none)."""
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


def as_query(prompt: Prompt) -> Any:
    """A plain string as-is; content blocks as a one-message async iterable.

    Mirrors the old ``_multimodal_prompt``: ``client.query()`` streams each
    yielded dict to the CLI as one user message.
    """
    if isinstance(prompt, str):
        return prompt
    message = {"type": "user", "message": {"role": "user", "content": prompt}}

    async def _one() -> AsyncIterator[dict[str, Any]]:
        yield message

    return _one()


async def bounded(aw: Awaitable[Any], timeout: float, what: str) -> bool:
    """Await *aw* for at most *timeout*s; True if it finished (even by raising).

    Unlike ``wait_for`` this never waits for a cancellation to land: a call
    blocked on a wedged child (or a hung Telegram request) may ignore it, and
    this runs on paths that must keep moving. Failures are logged, not raised.
    """
    task = asyncio.ensure_future(aw)
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
    except BaseException:
        task.cancel()
        raise
    if not done:
        task.cancel()
        logger.error("harness.bounded_call_timed_out", call=what, timeout_s=timeout)
        return False
    if not task.cancelled() and task.exception() is not None:
        logger.warning(
            "harness.bounded_call_failed", call=what, error=str(task.exception())
        )
    return True


def join_parts(parts: list[str]) -> str:
    """Join non-empty text parts with a blank line between them."""
    return "\n\n".join(p for p in (s.strip() for s in parts) if p)
