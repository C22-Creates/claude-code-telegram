"""Core value types shared across the harness.

All frozen: a run's inputs and results are facts, never edited in place.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class Attachment:
    """A file the user sent along with a message."""

    kind: Literal["photo", "document"]
    path: Path
    mime: str


@dataclass(frozen=True)
class Inbound:
    """One user message, normalised from a Telegram update."""

    topic_key: str
    chat_id: int
    thread_id: int | None
    message_id: int
    user_id: int
    text: str
    attachments: list[Attachment] = field(default_factory=list)
    reply_to: int | None = None
    is_voice: bool = False


@dataclass(frozen=True)
class ProgressEvent:
    """A unit of visible progress emitted by a run."""

    kind: Literal["tool", "text", "turn", "heartbeat", "mcp_unavailable"]
    tool: str | None
    turn: int
    elapsed_s: float
    detail: str = ""


class Outcome(Enum):
    """How a run ended."""

    DONE = "done"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass(frozen=True)
class RunResult:
    """Final result of a run.

    ``reason`` explains STOPPED/ERROR and is empty for DONE. Callers should use
    keyword arguments.
    """

    outcome: Outcome
    text: str
    elapsed_s: float
    reason: str = ""
    session_id: str | None = None
    tool_calls: int = 0
