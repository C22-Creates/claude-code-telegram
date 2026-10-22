"""Carry-over summary used when a topic's session is rotated."""

import asyncio
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional

import structlog
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)

logger = structlog.get_logger()

CARRYOVER_PROMPT = (
    "Summarize this conversation for a fresh session in ≤300 words: "
    "open tasks, decisions, names, IDs."
)
CARRYOVER_TIMEOUT_SECONDS = 120.0

QueryFn = Callable[..., AsyncIterator[Any]]


async def _collect(query_fn: QueryFn, options: ClaudeAgentOptions) -> str:
    last_text = ""
    result: Optional[str] = None
    async for message in query_fn(prompt=CARRYOVER_PROMPT, options=options):
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
) -> str:
    """Summarize ``session_id`` via a forked one-shot query; "" on any failure.

    ``fork_session=True`` keeps the old session untouched. Rotation proceeds
    without a summary rather than failing the user's message.
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
    try:
        text = await asyncio.wait_for(_collect(query_fn, options), timeout)
    except Exception as exc:  # noqa: BLE001 - any failure degrades to ""
        logger.warning(
            "Carry-over summary failed", session_id=session_id, error=repr(exc)
        )
        return ""
    words = text.split()
    return " ".join(words[: settings.harness_carryover_max_words])
