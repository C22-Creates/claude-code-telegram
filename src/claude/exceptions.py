"""Claude-specific exceptions."""

from typing import Optional


class ClaudeError(Exception):
    """Base Claude error."""


class ClaudeTimeoutError(ClaudeError):
    """Operation timed out."""


class ClaudeStreamStalledError(ClaudeTimeoutError):
    """The SDK message stream produced nothing for the idle-timeout window.

    A subclass of ClaudeTimeoutError on purpose: callers that already handle
    a timed-out turn (facade.py preserves the session for retry instead of
    discarding it; the Telegram handler shows the "Request Timeout" message)
    get correct behavior for this case for free via isinstance checks.

    Distinct in meaning, though: ClaudeTimeoutError says the whole turn took
    too long. This fires much sooner and specifically means a single step —
    almost always a hung outbound tool call, e.g. an MCP write that never
    returns — blocked the stream with no further progress. See failure mode
    #39: a session spent ~28 of its 30-minute budget wedged on two
    unanswered `mcp__notion__notion-update-page` calls with zero visible
    progress in between, discarding an already-fully-composed draft.
    """

    def __init__(self, last_tool_name: Optional[str], idle_timeout_seconds: float):
        self.last_tool_name = last_tool_name
        self.idle_timeout_seconds = idle_timeout_seconds
        tool_hint = (
            f"last tool call was {last_tool_name!r}"
            if last_tool_name
            else "no tool call was in flight — likely a stalled model turn"
        )
        super().__init__(
            f"No response from Claude for {idle_timeout_seconds:.0f}s ({tool_hint}). "
            "If a tool call stalled, its effect (e.g. a write) may have partially "
            "applied — verify before retrying."
        )


class ClaudeProcessError(ClaudeError):
    """Process execution failed."""


class ClaudeParsingError(ClaudeError):
    """Failed to parse output."""


class ClaudeSessionError(ClaudeError):
    """Session management error."""


class ClaudeMCPError(ClaudeError):
    """MCP server connection or configuration error."""

    def __init__(self, message: str, server_name: str = None):
        super().__init__(message)
        self.server_name = server_name
