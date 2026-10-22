"""SDK lifecycle helpers: turn detection, bounded cancellation, child reaping.

Ported verbatim from the old ``claude/sdk_integration.py`` (module-level
functions instead of ``ClaudeSDKManager`` methods; the old package is gone).
The docstrings carry the incident history -- keep them.
"""

import asyncio
import os
import signal
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from claude_agent_sdk import (
    AssistantMessage,
    CLIConnectionError,
    Message,
    ToolUseBlock,
)
from claude_agent_sdk.types import StreamEvent

logger = structlog.get_logger()

# How long to wait for an already-cancelled SDK task to actually settle before
# abandoning it. Cancellation is not guaranteed to land — see
# await_cancelled for the failure mode this bounds. Kept
# short: by this point the request has already failed, and the only thing left
# to do is hand control back so the caller's cleanup can run.
CANCELLED_TASK_CLEANUP_TIMEOUT = 10.0

# Maximum number of descendant processes to reap alongside a wedged SDK child.
# A wedged `claude` child owns its own MCP subprocesses (node, bun, ...), which
# survive as orphans if only the direct child is killed. The cap is a guard
# against a pathological /proc walk, not an expected limit.
MAX_REAPED_DESCENDANTS = 64


def turn_signal(message: Any) -> Optional[bool]:
    """Whether *message* shows the main agent's turn ending (True) or continuing
    (False); None when it says nothing about the turn.

    The live stream never sets AssistantMessage.stop_reason, so the end of a turn
    is the raw ``message_delta`` stream event (partial messages enabled) with
    stop_reason ``end_turn``. Subagent events (parent_tool_use_id set) are
    ignored — a subagent finishing is not the main turn finishing.
    """
    if isinstance(message, StreamEvent):
        if message.parent_tool_use_id is not None:
            return None
        event = message.event or {}
        if event.get("type") == "message_start":
            return False
        if event.get("type") == "message_delta":
            return (event.get("delta") or {}).get("stop_reason") == "end_turn"
        return None
    if isinstance(message, AssistantMessage) and any(
        isinstance(b, ToolUseBlock) for b in (message.content or [])
    ):
        return False
    return None


def is_retryable_error(exc: BaseException) -> bool:
    """Return True for transient errors that warrant a retry.
    asyncio.TimeoutError is intentional (user-configured timeout) — not retried.
    Non-MCP CLIConnectionError and control-protocol handshake timeouts are
    considered transient.
    """
    if isinstance(exc, CLIConnectionError):
        msg = str(exc).lower()
        return "mcp" not in msg  # "server" alone is too broad
    # claude-agent-sdk raises a bare Exception when the control-protocol
    # handshake times out (e.g. "Control request timeout: initialize").
    # This is a transient subprocess startup failure — not a user-configured
    # limit — so it is safe to retry. Match on the exact bare Exception type
    # plus message to avoid retrying unrelated programming errors.
    if type(exc) is Exception:
        return "control request timeout" in str(exc).lower()
    return False


async def await_cancelled(
    task: "asyncio.Task[None]",
    reason: str,
    timeout: float = CANCELLED_TASK_CLEANUP_TIMEOUT,
) -> bool:
    """Wait — with a hard bound — for an already-cancelled task to finish.

    A bare ``await task`` after ``task.cancel()`` is NOT guaranteed to
    return. If the SDK child process is wedged (blocked reading a stdout
    pipe that never yields and never closes), the CancelledError is never
    delivered at an await point and the await hangs forever.

    That hang propagates all the way up. ``execute_command`` never
    returns, so the orchestrator's ``finally: heartbeat.cancel()`` never
    runs — leaving the Telegram typing indicator running forever while
    the wedged child leaks. Observed 2026-07-29: the bot appeared
    permanently "typing" and a 15.5h-old orphaned child held 220MB plus
    its own MCP subprocesses. The 5-minute claude_timeout_seconds fired
    correctly and did not help, because the hang was in the cleanup path
    that runs *after* the timeout.

    Returns True if the task settled, False if it is still wedged. A
    False return is a leak we cannot fix from here, so it is logged at
    error level — but we return control to the caller either way, which
    is what stops the typing indicator and frees the request slot.
    """
    _, pending = await asyncio.wait({task}, timeout=timeout)
    if pending:
        logger.error(
            "Cancelled SDK task did not settle; abandoning it to avoid "
            "hanging the request. The child process may be orphaned.",
            reason=reason,
            timeout_seconds=timeout,
        )
        return False
    return True


def last_tool_name(messages: List[Message]) -> Optional[str]:
    """Return the tool name from the most recent ToolUseBlock, if any.

    Diagnostic-only: used to attribute a stream stall to the tool call
    that was in flight when it happened, since the stream going quiet
    right after a ToolUseBlock is, in practice, a hung outbound call
    (e.g. an MCP write that never returns) rather than a slow model
    turn. See failure mode #39.
    """
    for message in reversed(messages):
        if isinstance(message, AssistantMessage):
            content = getattr(message, "content", [])
            if isinstance(content, list):
                for block in reversed(content):
                    if isinstance(block, ToolUseBlock):
                        return block.name
    return None


def descendant_pids(root_pid: int) -> List[int]:
    """Return descendants of *root_pid*, deepest-last, via /proc.

    Linux-only and best-effort: a pid that exits mid-walk is skipped. Used
    only to reap a wedged SDK child, so a partial answer is still useful.
    Returns [] on any platform without /proc.
    """
    if not os.path.isdir("/proc"):
        return []

    children: Dict[int, List[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/status", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("PPid:"):
                        ppid = int(line.split()[1])
                        children.setdefault(ppid, []).append(int(entry))
                        break
        except (OSError, ValueError):
            continue  # process exited or /proc entry unreadable

    found: List[int] = []
    queue = [root_pid]
    while queue and len(found) < MAX_REAPED_DESCENDANTS:
        for child in children.get(queue.pop(0), []):
            if child not in found:
                found.append(child)
                queue.append(child)
    return found


def reap_wedged_child(client: Optional[Any], reason: str) -> None:
    """SIGKILL a wedged SDK child process and its descendants.

    Called only after `await_cancelled` reports that a cancelled
    task never settled. At that point the transport's own cooperative
    shutdown (terminate -> wait -> kill) is itself blocked, so the child
    would otherwise leak until the service restarts. Observed 2026-08-21:
    a wedged child survived 2h28m holding 660MB plus its MCP
    subprocesses.

    Reaches into SDK internals (``_transport._process``) deliberately —
    there is no public accessor. Every step is best-effort and failure is
    logged, never raised: this runs on a path that is already failing, and
    must not mask the original error.
    """
    try:
        transport = getattr(client, "_transport", None)
        process = getattr(transport, "_process", None)
        pid = getattr(process, "pid", None)
        if pid is None:
            return
        if getattr(process, "returncode", None) is not None:
            return  # already exited cleanly
        # Never signal ourselves, init, or a whole process group.
        if pid <= 1 or pid == os.getpid():
            logger.error("Refusing to reap unsafe pid", pid=pid, reason=reason)
            return

        # Snapshot descendants before killing the parent — once it dies
        # they reparent to init and the tree is no longer walkable.
        descendants = descendant_pids(pid)

        killed: List[int] = []
        for target in [pid, *descendants]:
            try:
                os.kill(target, signal.SIGKILL)
                killed.append(target)
            except ProcessLookupError:
                continue  # already gone
            except PermissionError:
                logger.warning("Not permitted to reap pid", pid=target)

        logger.error(
            "Reaped wedged SDK child process tree",
            reason=reason,
            child_pid=pid,
            killed_pids=killed,
            descendant_count=len(descendants),
        )
    except Exception as exc:  # noqa: BLE001 - cleanup must never raise
        logger.warning(
            "Failed to reap wedged SDK child",
            reason=reason,
            error=str(exc),
            error_type=type(exc).__name__,
        )


def load_mcp_config(config_path: Path) -> Dict[str, Any]:
    """Load MCP server configuration from a JSON file.

    The new claude-agent-sdk expects mcp_servers as a dict, not a file path.
    """
    import json  # noqa: PLC0415

    try:
        with open(config_path) as f:
            config_data = json.load(f)
        return config_data.get("mcpServers", {})
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to load MCP config", path=str(config_path), error=str(e))
        return {}
