"""Claude Code Python SDK integration."""

import asyncio
import os
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
)

import structlog
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ClaudeSDKError,
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    Message,
    PermissionResultAllow,
    PermissionResultDeny,
    ProcessError,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolPermissionContext,
    ToolUseBlock,
    UserMessage,
)
from claude_agent_sdk._errors import MessageParseError
from claude_agent_sdk._internal.message_parser import parse_message
from claude_agent_sdk.types import StreamEvent

from ..config.settings import Settings
from ..security.validators import SecurityValidator
from .exceptions import (
    ClaudeMCPError,
    ClaudeParsingError,
    ClaudeProcessError,
    ClaudeStreamStalledError,
    ClaudeTimeoutError,
)
from .monitor import _is_claude_internal_path, check_bash_directory_boundary

logger = structlog.get_logger()

# Fallback message when Claude produces no text but did use tools.
TASK_COMPLETED_MSG = "✅ Task completed. Tools used: {tools_summary}"

# How long to wait for an already-cancelled SDK task to actually settle before
# abandoning it. Cancellation is not guaranteed to land — see
# ClaudeSDKManager._await_cancelled for the failure mode this bounds. Kept
# short: by this point the request has already failed, and the only thing left
# to do is hand control back so the caller's cleanup can run.
CANCELLED_TASK_CLEANUP_TIMEOUT = 10.0

# Maximum number of descendant processes to reap alongside a wedged SDK child.
# A wedged `claude` child owns its own MCP subprocesses (node, bun, ...), which
# survive as orphans if only the direct child is killed. The cap is a guard
# against a pathological /proc walk, not an expected limit.
MAX_REAPED_DESCENDANTS = 64
# Fallback message when a run stopped early without producing any text. The
# stop-reason footer carries the warning and the reason, so this only reports
# what the run got done -- otherwise the two stack up as "stopped" twice.
TASK_STOPPED_MSG = "No final response. Tools used: {tools_summary}"

# ResultMessage.subtype reported by the CLI for a run that ran to completion.
RESULT_SUBTYPE_SUCCESS = "success"


def _as_error_list(value: Any) -> List[str]:
    """Normalise ResultMessage.errors into a list of strings."""
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)]


def _as_denial_list(value: Any) -> List[Dict[str, Any]]:
    """Normalise ResultMessage.permission_denials into a list of dicts.

    The SDK types this ``list[Any]`` and passes the CLI payload through
    untouched, so accept both snake_case and camelCase keys and tolerate
    entries that are not dicts at all.
    """
    if not value or not isinstance(value, list):
        return []

    denials: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            tool_name = item.get("tool_name") or item.get("toolName")
            tool_input = item.get("tool_input")
            if tool_input is None:
                tool_input = item.get("toolInput")
            denials.append(
                {
                    "tool_name": str(tool_name) if tool_name else "unknown",
                    "tool_input": tool_input if isinstance(tool_input, dict) else {},
                }
            )
        else:
            name = getattr(item, "tool_name", None)
            tool_input = getattr(item, "tool_input", None)
            denials.append(
                {
                    "tool_name": str(name) if name else "unknown",
                    "tool_input": tool_input if isinstance(tool_input, dict) else {},
                }
            )
    return denials


@dataclass
class ClaudeResponse:
    """Response from Claude Code SDK."""

    content: str
    session_id: str
    cost: float
    duration_ms: int
    num_turns: int
    is_error: bool = False
    error_type: Optional[str] = None
    tools_used: List[Dict[str, Any]] = field(default_factory=list)
    interrupted: bool = False
    # Why the run ended.  All optional so existing construction sites keep
    # working; populated from ResultMessage when the SDK reports them.
    result_subtype: Optional[str] = None
    stop_reason: Optional[str] = None
    terminal_reason: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    permission_denials: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def completed_normally(self) -> bool:
        """Whether the run reached its own end rather than being cut short.

        ``None`` means the CLI reported no subtype at all (older versions, or a
        result that bypassed the query loop), which we treat as normal so that
        nothing regresses into a spurious warning.
        """
        return self.result_subtype in (None, RESULT_SUBTYPE_SUCCESS)


@dataclass
class StreamUpdate:
    """Streaming update from Claude SDK."""

    type: str  # 'assistant', 'user', 'system', 'result', 'stream_delta'
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    progress: Optional[Dict[str, Any]] = None

    def get_tool_names(self) -> List[str]:
        """Return tool names from the stream payload."""
        names: List[str] = []

        if self.tool_calls:
            for tool_call in self.tool_calls:
                name = tool_call.get("name") if isinstance(tool_call, dict) else None
                if isinstance(name, str) and name:
                    names.append(name)

        if self.metadata:
            tool_name = self.metadata.get("tool_name")
            if isinstance(tool_name, str) and tool_name:
                names.append(tool_name)

            metadata_tools = self.metadata.get("tools")
            if isinstance(metadata_tools, list):
                for tool in metadata_tools:
                    if isinstance(tool, dict):
                        name = tool.get("name")
                    elif isinstance(tool, str):
                        name = tool
                    else:
                        name = None

                    if isinstance(name, str) and name:
                        names.append(name)

        # Preserve insertion order while de-duplicating.
        return list(dict.fromkeys(names))

    def is_error(self) -> bool:
        """Check whether this stream update represents an error."""
        if self.type == "error":
            return True

        if self.metadata:
            if self.metadata.get("is_error") is True:
                return True
            status = self.metadata.get("status")
            if isinstance(status, str) and status.lower() == "error":
                return True
            error_val = self.metadata.get("error")
            if isinstance(error_val, str) and error_val:
                return True
            error_msg_val = self.metadata.get("error_message")
            if isinstance(error_msg_val, str) and error_msg_val:
                return True

        if self.progress:
            status = self.progress.get("status")
            if isinstance(status, str) and status.lower() == "error":
                return True

        return False

    def get_error_message(self) -> str:
        """Get the best available error message from the stream payload."""
        if self.metadata:
            for key in ("error_message", "error", "message"):
                value = self.metadata.get(key)
                if isinstance(value, str) and value.strip():
                    return value

        if isinstance(self.content, str) and self.content.strip():
            return self.content

        if self.progress:
            value = self.progress.get("error")
            if isinstance(value, str) and value.strip():
                return value

        return "Unknown error"

    def get_progress_percentage(self) -> Optional[int]:
        """Extract progress percentage if present."""

        def _to_int(value: Any) -> Optional[int]:
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str) and value.strip():
                try:
                    return int(float(value))
                except ValueError:
                    return None
            return None

        if self.progress:
            for key in ("percentage", "percent", "progress"):
                percentage = _to_int(self.progress.get(key))
                if percentage is not None:
                    return max(0, min(100, percentage))

            step = _to_int(self.progress.get("step"))
            total_steps = _to_int(self.progress.get("total_steps"))
            if step is not None and total_steps and total_steps > 0:
                return max(0, min(100, int((step / total_steps) * 100)))

        if self.metadata:
            percentage = _to_int(self.metadata.get("progress_percentage"))
            if percentage is not None:
                return max(0, min(100, percentage))

        return None


# Tools whose file path the can_use_tool callback validates against the
# approved directory.
FILE_TOOLS = frozenset(
    {
        "Write",
        "Edit",
        "MultiEdit",
        "Read",
        "NotebookEdit",
        "NotebookRead",
        "create_file",
        "edit_file",
        "read_file",
    }
)

# Tools whose command the can_use_tool callback checks for directory escapes.
BASH_TOOLS = frozenset({"Bash", "bash", "shell"})

# Every tool the callback actually guards. These must be kept out of the
# ``allowed_tools`` list handed to the SDK: the CLI's permission engine resolves
# allow rules before consulting the permission prompt tool, so a tool named in
# ``allowed_tools`` is pre-approved and never produces a ``can_use_tool``
# control request -- leaving the checks below inert. See issue #219.
GUARDED_TOOLS = FILE_TOOLS | BASH_TOOLS

# Keys under which the guarded file tools pass their target path.
_FILE_PATH_KEYS = ("file_path", "path", "notebook_path")


def _make_can_use_tool_callback(
    security_validator: SecurityValidator,
    working_directory: Path,
    approved_directory: Path,
    approval_callback: Optional[
        Callable[[str, Dict[str, Any]], Awaitable[bool]]
    ] = None,
    approval_tool_names: FrozenSet[str] = frozenset(),
) -> Any:
    """Create a can_use_tool callback for SDK-level tool permission validation.

    The callback validates file path boundaries and bash directory boundaries
    *before* the SDK executes the tool, providing preventive security enforcement.
    If `approval_callback` is set, tools in `approval_tool_names` additionally
    require interactive human approval (e.g. via a Telegram Allow/Deny prompt)
    after the static checks pass.
    """

    async def can_use_tool(
        tool_name: str,
        tool_input: Dict[str, Any],
        context: ToolPermissionContext,
    ) -> Any:
        logger.debug("can_use_tool consulted", tool_name=tool_name)

        # File path validation
        if tool_name in FILE_TOOLS:
            file_path = next(
                (tool_input.get(key) for key in _FILE_PATH_KEYS if tool_input.get(key)),
                None,
            )
            if file_path:
                # Allow Claude Code internal paths (~/.claude/plans/, etc.)
                if _is_claude_internal_path(file_path):
                    return PermissionResultAllow()

                valid, _resolved, error = security_validator.validate_path(
                    file_path, working_directory
                )
                if not valid:
                    logger.warning(
                        "can_use_tool denied file operation",
                        tool_name=tool_name,
                        file_path=file_path,
                        error=error,
                    )
                    return PermissionResultDeny(message=error or "Invalid file path")

        # Bash directory boundary validation
        if tool_name in BASH_TOOLS:
            command = tool_input.get("command", "")
            if command:
                valid, error = check_bash_directory_boundary(
                    command, working_directory, approved_directory
                )
                if not valid:
                    logger.warning(
                        "can_use_tool denied bash command",
                        tool_name=tool_name,
                        command=command,
                        error=error,
                    )
                    return PermissionResultDeny(
                        message=error or "Bash directory boundary violation"
                    )

        # Interactive human-in-the-loop approval for configured tools
        if approval_callback is not None and tool_name in approval_tool_names:
            approved = await approval_callback(tool_name, tool_input)
            if not approved:
                logger.info(
                    "can_use_tool denied by interactive approval",
                    tool_name=tool_name,
                )
                return PermissionResultDeny(message="Denied by user via Telegram")

        return PermissionResultAllow()

    return can_use_tool


class ClaudeSDKManager:
    """Manage Claude Code SDK integration."""

    def __init__(
        self,
        config: Settings,
        security_validator: Optional[SecurityValidator] = None,
    ):
        """Initialize SDK manager with configuration."""
        self.config = config
        self.security_validator = security_validator

        # Set up environment for Claude Code SDK if API key is provided
        # If no API key is provided, the SDK will use existing CLI authentication
        if config.anthropic_api_key_str:
            os.environ["ANTHROPIC_API_KEY"] = config.anthropic_api_key_str
            logger.info("Using provided API key for Claude SDK authentication")
        else:
            logger.info("No API key provided, using existing Claude CLI authentication")

    def _is_retryable_error(self, exc: BaseException) -> bool:
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

    @staticmethod
    async def _await_cancelled(
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

    @staticmethod
    def _last_tool_name(messages: List[Message]) -> Optional[str]:
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

    @staticmethod
    def _descendant_pids(root_pid: int) -> List[int]:
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

    @classmethod
    def _reap_wedged_child(cls, client: Optional[Any], reason: str) -> None:
        """SIGKILL a wedged SDK child process and its descendants.

        Called only after :meth:`_await_cancelled` reports that a cancelled
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
            descendants = cls._descendant_pids(pid)

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

    async def execute_command(
        self,
        prompt: str,
        working_directory: Path,
        session_id: Optional[str] = None,
        continue_session: bool = False,
        stream_callback: Optional[Callable[[StreamUpdate], None]] = None,
        interrupt_event: Optional[asyncio.Event] = None,
        images: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        extra_disallowed_tools: Optional[List[str]] = None,
        topic_agent: Optional[str] = None,
        approval_callback: Optional[
            Callable[[str, Dict[str, Any]], Awaitable[bool]]
        ] = None,
    ) -> ClaudeResponse:
        """Execute Claude Code command via SDK."""
        start_time = asyncio.get_event_loop().time()

        logger.info(
            "Starting Claude SDK command",
            working_directory=str(working_directory),
            session_id=session_id,
            continue_session=continue_session,
        )

        try:
            # Capture stderr from Claude CLI for better error diagnostics
            stderr_lines: List[str] = []

            def _stderr_callback(line: str) -> None:
                stderr_lines.append(line)
                logger.debug("Claude CLI stderr", line=line)

            # Build system prompt, loading CLAUDE.md from working directory if present
            base_prompt = (
                f"All file operations must stay within {working_directory}. "
                "Use relative paths."
            )
            claude_md_path = Path(working_directory) / "CLAUDE.md"
            if claude_md_path.exists():
                base_prompt += "\n\n" + claude_md_path.read_text(encoding="utf-8")
                logger.info(
                    "Loaded CLAUDE.md into system prompt",
                    path=str(claude_md_path),
                )

            # Per-topic agent identity (agents/README.md § Loader contract):
            # callers that know their topic's declared agent (the Telegram
            # message path via _thread_context) pass it explicitly. Never
            # inferred from working_directory — the topic dirs are symlinks
            # that resolve to one checkout, and session state can rewrite the
            # working dir, so path-keying misidentifies the topic (live
            # incident 2026-09-04: Relationships session booted as coach and
            # had every Missive tool denied).
            if topic_agent and bool(getattr(self.config, "agent_loader_enabled", True)):
                from ..dispatcher.agent_loader import load_agent

                identity, topic_denied = load_agent(
                    topic_agent, Path(working_directory)
                )
                if identity:
                    base_prompt += "\n\n" + identity
                if topic_denied:
                    enforcement = str(
                        getattr(self.config, "agent_tool_enforcement", "enforce")
                    )
                    if enforcement == "enforce":
                        extra_disallowed_tools = [
                            *(extra_disallowed_tools or []),
                            *topic_denied,
                        ]
                    else:
                        logger.warning(
                            "Topic agent tool denials in warn mode; NOT enforced",
                            agent=topic_agent,
                            would_deny=topic_denied,
                        )

            # Always pass a list (never None) for allowed/disallowed tools.
            # ClaudeAgentOptions declares both as list[str] with
            # default_factory=list. 0.1.x guarded with a truthiness check and
            # tolerated None; 0.2 does not -- the transport calls
            # list(options.allowed_tools), and the connect-time shadowing
            # check iterates it, so None raises TypeError before the CLI even
            # starts. Both settings are Optional, and a true
            # DISABLE_TOOL_VALIDATION deliberately sends nothing (#206), so
            # normalise every path to a list. [] and None are both falsy, so
            # the CLI omits the flags either way.
            sdk_allowed_tools: List[str]
            sdk_disallowed_tools: List[str]
            if self.config.disable_tool_validation:
                sdk_allowed_tools = []
                sdk_disallowed_tools = []
            else:
                sdk_allowed_tools = list(self.config.claude_allowed_tools or [])
                sdk_disallowed_tools = list(self.config.claude_disallowed_tools or [])

            # The can_use_tool callback below is purely reactive: the SDK only
            # invokes it when the CLI sends a can_use_tool control request, and
            # the CLI resolves allow rules first. Any guarded tool left in
            # allowed_tools is therefore pre-approved and its boundary check
            # never runs. Strip them so each call is routed to the callback,
            # which allows everything that passes validation. Issue #219.
            boundary_checks_active = (
                self.security_validator is not None
                and not self.config.disable_tool_validation
            )

            # Interactive approval has the same requirement for its own gated
            # tools: a tool left in allowed_tools is pre-approved by the CLI
            # and never reaches can_use_tool, so the approval prompt (and the
            # static checks above) would never fire for it. This can gate
            # tools beyond GUARDED_TOOLS (INTERACTIVE_TOOL_APPROVAL_TOOLS is
            # user-configured), so it is a separate, additive strip.
            approval_tool_names_set: FrozenSet[str] = frozenset()
            if self.config.interactive_tool_approval:
                approval_tool_names_set = frozenset(
                    self.config.interactive_tool_approval_tools or ()
                )

            tools_to_strip: FrozenSet[str] = approval_tool_names_set
            if boundary_checks_active:
                tools_to_strip = tools_to_strip | GUARDED_TOOLS

            if tools_to_strip:
                gated = [t for t in sdk_allowed_tools if t in tools_to_strip]
                if gated:
                    sdk_allowed_tools = [
                        tool for tool in sdk_allowed_tools if tool not in tools_to_strip
                    ]
                    logger.debug(
                        "Routing guarded tools through can_use_tool",
                        gated_tools=gated,
                    )

            # The SDK auto-approves sandboxed Bash calls without ever invoking
            # can_use_tool (see ClaudeAgentOptions sandbox settings). That is
            # a second bypass of both the bash-boundary check and the
            # interactive-approval prompt, so it must be disabled whenever
            # either of those depends on Bash going through can_use_tool.
            bash_needs_approval = "Bash" in approval_tool_names_set

            # Agent-level hard denials (dispatcher loader, agents/scoping.md)
            # apply even when config-level tool validation is disabled — they
            # are enforcement policy, not schema validation.
            if extra_disallowed_tools:
                sdk_disallowed_tools = [
                    *(sdk_disallowed_tools or []),
                    *extra_disallowed_tools,
                ]
                logger.info(
                    "Agent tool denials active",
                    disallowed=extra_disallowed_tools,
                )

            # Build Claude Agent options
            options = ClaudeAgentOptions(
                max_turns=self.config.claude_max_turns,
                model=model or self.config.claude_model or None,
                max_budget_usd=self.config.claude_max_cost_per_request,
                cwd=str(working_directory),
                allowed_tools=sdk_allowed_tools,
                disallowed_tools=sdk_disallowed_tools,
                cli_path=self.config.claude_cli_path or None,
                include_partial_messages=stream_callback is not None,
                sandbox={
                    "enabled": self.config.sandbox_enabled,
                    # Auto-approving sandboxed bash is a second bypass of the
                    # control request the bash boundary check (and, if Bash is
                    # gated, the approval prompt) depends on, so it stays off
                    # whenever either is meant to run (#219).
                    "autoAllowBashIfSandboxed": not (
                        boundary_checks_active or bash_needs_approval
                    ),
                    "excludedCommands": self.config.sandbox_excluded_commands or [],
                },
                system_prompt=base_prompt,
                setting_sources=["project"],
                stderr=_stderr_callback,
            )

            # Pass MCP server configuration if enabled
            if self.config.enable_mcp and self.config.mcp_config_path:
                options.mcp_servers = self._load_mcp_config(self.config.mcp_config_path)
                logger.info(
                    "MCP servers configured",
                    mcp_config_path=str(self.config.mcp_config_path),
                )

            # Wire can_use_tool callback for preventive tool validation
            if self.security_validator:
                options.can_use_tool = _make_can_use_tool_callback(
                    security_validator=self.security_validator,
                    working_directory=working_directory,
                    approved_directory=self.config.approved_directory,
                    approval_callback=(
                        approval_callback
                        if self.config.interactive_tool_approval
                        else None
                    ),
                    approval_tool_names=approval_tool_names_set,
                )

            # Resume previous session if we have a session_id
            if session_id and continue_session:
                options.resume = session_id
                logger.info(
                    "Resuming previous session",
                    session_id=session_id,
                )

            # Collect messages via ClaudeSDKClient
            messages: List[Message] = []
            interrupted = False
            # Holds the live client so the cleanup path can reach its child
            # process if cancellation fails to settle. Rebound per attempt.
            client_holder: Dict[str, Any] = {}

            async def _run_client() -> None:
                client = ClaudeSDKClient(options)
                client_holder["client"] = client
                try:
                    await client.connect()

                    if images:
                        content_blocks: List[Dict[str, Any]] = []
                        for img in images:
                            media_type = img.get("media_type", "image/png")
                            content_blocks.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": img["data"],
                                    },
                                }
                            )
                        content_blocks.append({"type": "text", "text": prompt})

                        multimodal_msg = {
                            "type": "user",
                            "message": {
                                "role": "user",
                                "content": content_blocks,
                            },
                        }

                        async def _multimodal_prompt() -> AsyncIterator[Dict[str, Any]]:
                            yield multimodal_msg

                        await client.query(_multimodal_prompt())
                    else:
                        await client.query(prompt)

                    async for raw_data in client._query.receive_messages():
                        try:
                            message = parse_message(raw_data)
                        except MessageParseError as e:
                            logger.debug(
                                "Skipping unparseable message",
                                error=str(e),
                            )
                            continue

                        messages.append(message)

                        if isinstance(message, ResultMessage):
                            break

                        # Handle streaming callback
                        if stream_callback:
                            try:
                                await self._handle_stream_message(
                                    message, stream_callback
                                )
                            except Exception as callback_error:
                                logger.warning(
                                    "Stream callback failed",
                                    error=str(callback_error),
                                    error_type=type(callback_error).__name__,
                                )
                finally:
                    await client.disconnect()

            # Execute with timeout and retry, racing against optional interrupt
            max_attempts = max(1, self.config.claude_retry_max_attempts)
            last_exc: Optional[BaseException] = None
            idle_timeout = self.config.claude_tool_idle_timeout_seconds

            for attempt in range(max_attempts):
                # Reset message accumulator each attempt so that a failed attempt
                # does not pollute the next one with partial/duplicate messages.
                # _run_client() closes over `messages` by reference (late-binding
                # closure), so clearing it here is seen by every new call.
                messages.clear()
                stalled_tool_name: Optional[str] = None

                if attempt > 0:
                    delay = min(
                        self.config.claude_retry_base_delay
                        * (self.config.claude_retry_backoff_factor ** (attempt - 1)),
                        self.config.claude_retry_max_delay,
                    )
                    logger.warning(
                        "Retrying Claude SDK command",
                        attempt=attempt + 1,
                        max_attempts=max_attempts,
                        delay_seconds=delay,
                    )
                    await asyncio.sleep(delay)

                run_task = asyncio.create_task(_run_client())

                interrupt_watcher: Optional["asyncio.Task[None]"] = None
                if interrupt_event is not None:

                    async def _cancel_on_interrupt() -> None:
                        nonlocal interrupted
                        await interrupt_event.wait()
                        interrupted = True
                        run_task.cancel()

                    interrupt_watcher = asyncio.create_task(_cancel_on_interrupt())

                async def _cancel_on_stall() -> None:
                    # Watches `messages` (appended to by _run_client as it
                    # streams) for growth. No growth for idle_timeout seconds
                    # means the stream is stalled — in practice almost always
                    # a hung outbound tool call, e.g. an MCP write that never
                    # returns (failure mode #39) — and would otherwise
                    # silently burn the full claude_timeout_seconds budget
                    # with zero visible progress. Sets stalled_tool_name
                    # *before* cancelling, so the CancelledError handler below
                    # can never observe the cancellation without the reason
                    # already recorded.
                    nonlocal stalled_tool_name
                    loop = asyncio.get_event_loop()
                    last_count = 0
                    last_seen_at = loop.time()
                    while not run_task.done():
                        await asyncio.sleep(1.0)
                        if run_task.done():
                            return
                        current_count = len(messages)
                        now = loop.time()
                        if current_count != last_count:
                            last_count = current_count
                            last_seen_at = now
                            continue
                        if now - last_seen_at >= idle_timeout:
                            stalled_tool_name = (
                                self._last_tool_name(messages) or "unknown"
                            )
                            logger.error(
                                "Claude SDK message stream stalled",
                                idle_timeout_seconds=idle_timeout,
                                last_tool_name=stalled_tool_name,
                            )
                            run_task.cancel()
                            return

                stall_watcher = asyncio.create_task(_cancel_on_stall())

                # Note: asyncio.TimeoutError is intentionally NOT retried —
                # it reflects a user-configured hard limit. A stream stall
                # (below) isn't retried either — the underlying tool call may
                # still be in flight server-side, so retrying immediately
                # would just risk a second concurrent write.
                try:
                    await asyncio.wait_for(
                        asyncio.shield(run_task),
                        timeout=self.config.claude_timeout_seconds,
                    )
                    break  # success — exit retry loop
                except asyncio.CancelledError:
                    if stalled_tool_name is not None:
                        if not await self._await_cancelled(
                            run_task, reason="claude_stream_stalled"
                        ):
                            self._reap_wedged_child(
                                client_holder.get("client"),
                                reason="claude_stream_stalled",
                            )
                        raise ClaudeStreamStalledError(stalled_tool_name, idle_timeout)
                    if not interrupted:
                        raise
                    # Interrupt cancelled the task — wait for cleanup, but do
                    # not let a wedged child block the interrupt forever.
                    if not await self._await_cancelled(
                        run_task, reason="user_interrupt"
                    ):
                        self._reap_wedged_child(
                            client_holder.get("client"), reason="user_interrupt"
                        )
                    break  # user interrupted — don't retry
                except asyncio.TimeoutError:
                    run_task.cancel()
                    if not await self._await_cancelled(
                        run_task, reason="claude_timeout"
                    ):
                        self._reap_wedged_child(
                            client_holder.get("client"), reason="claude_timeout"
                        )
                    raise  # timeout — don't retry
                except CLIConnectionError as exc:
                    if self._is_retryable_error(exc) and attempt < max_attempts - 1:
                        last_exc = exc
                        logger.warning(
                            "Transient connection error, will retry",
                            attempt=attempt + 1,
                            error=str(exc),
                        )
                        continue
                    raise  # non-retryable or attempts exhausted
                except Exception as exc:  # noqa: BLE001
                    # Catches transient control-protocol handshake timeouts
                    # ("Control request timeout: initialize") that the SDK raises
                    # as a bare Exception. Only retried when _is_retryable_error
                    # matches; every other exception re-raises immediately, which
                    # preserves the prior behaviour.
                    if self._is_retryable_error(exc) and attempt < max_attempts - 1:
                        last_exc = exc
                        logger.warning(
                            "Transient SDK error, will retry",
                            attempt=attempt + 1,
                            error=str(exc),
                        )
                        continue
                    raise  # non-retryable or attempts exhausted
                finally:
                    stall_watcher.cancel()
                    if interrupt_watcher is not None:
                        interrupt_watcher.cancel()
            else:
                if last_exc is not None:
                    raise last_exc

            # Extract cost, tools, session_id and stop reason from result message
            cost = 0.0
            tools_used: List[Dict[str, Any]] = []
            claude_session_id = None
            result_content = None
            result_subtype: Optional[str] = None
            result_num_turns: Optional[int] = None
            stop_reason: Optional[str] = None
            terminal_reason: Optional[str] = None
            result_errors: List[str] = []
            permission_denials: List[Dict[str, Any]] = []
            for message in messages:
                if isinstance(message, ResultMessage):
                    cost = getattr(message, "total_cost_usd", 0.0) or 0.0
                    claude_session_id = getattr(message, "session_id", None)
                    result_content = getattr(message, "result", None)
                    # getattr (not attribute access) throughout: older CLI
                    # versions and the test doubles omit these fields.
                    result_subtype = getattr(message, "subtype", None)
                    result_num_turns = getattr(message, "num_turns", None)
                    stop_reason = getattr(message, "stop_reason", None)
                    terminal_reason = getattr(message, "terminal_reason", None)
                    result_errors = _as_error_list(getattr(message, "errors", None))
                    permission_denials = _as_denial_list(
                        getattr(message, "permission_denials", None)
                    )
                    current_time = asyncio.get_event_loop().time()
                    for msg in messages:
                        if isinstance(msg, AssistantMessage):
                            msg_content = getattr(msg, "content", [])
                            if msg_content and isinstance(msg_content, list):
                                for block in msg_content:
                                    if isinstance(block, ToolUseBlock):
                                        tools_used.append(
                                            {
                                                "name": getattr(
                                                    block, "name", "unknown"
                                                ),
                                                "timestamp": current_time,
                                                "input": getattr(block, "input", {}),
                                            }
                                        )
                    break

            # Fallback: extract session_id from StreamEvent messages if
            # ResultMessage didn't provide one (can happen with some CLI versions)
            if not claude_session_id:
                for message in messages:
                    msg_session_id = getattr(message, "session_id", None)
                    if msg_session_id and not isinstance(message, ResultMessage):
                        claude_session_id = msg_session_id
                        logger.info(
                            "Got session ID from stream event (fallback)",
                            session_id=claude_session_id,
                        )
                        break

            # Calculate duration
            duration_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

            # Use Claude's session_id if available, otherwise fall back
            final_session_id = claude_session_id or session_id or ""

            if claude_session_id and claude_session_id != session_id:
                logger.info(
                    "Got session ID from Claude",
                    claude_session_id=claude_session_id,
                    previous_session_id=session_id,
                )

            # Use ResultMessage.result if available, fall back to message extraction
            if result_content is not None:
                content = str(result_content).strip()
            else:
                content_parts = []
                for msg in messages:
                    if isinstance(msg, AssistantMessage):
                        msg_content = getattr(msg, "content", [])
                        if msg_content and isinstance(msg_content, list):
                            for block in msg_content:
                                if hasattr(block, "text"):
                                    content_parts.append(block.text)
                        elif msg_content:
                            content_parts.append(str(msg_content))
                content = "\n".join(content_parts).strip()

            ran_to_completion = result_subtype in (None, RESULT_SUBTYPE_SUCCESS)

            if not content and tools_used:
                tool_names = [
                    tool.get("name", "")
                    for tool in tools_used
                    if isinstance(tool.get("name"), str) and tool.get("name")
                ]
                unique_tool_names = list(dict.fromkeys(tool_names))
                tools_summary = ", ".join(unique_tool_names) or "unknown"
                # Only claim completion when the CLI says the run completed.
                # A run killed at the turn limit takes this same path (tools
                # ran, no final text) and must not report success (#172).
                template = TASK_COMPLETED_MSG if ran_to_completion else TASK_STOPPED_MSG
                content = template.format(tools_summary=tools_summary)

            # The CLI reports the authoritative turn count. Counting messages
            # over-reports it -- every tool result arrives as another
            # UserMessage -- and that number is now shown to the user in the
            # stop-reason footer, so the approximation is only a fallback for
            # a result that did not carry one.
            if isinstance(result_num_turns, int) and result_num_turns >= 0:
                num_turns = result_num_turns
            else:
                num_turns = len(
                    [
                        m
                        for m in messages
                        if isinstance(m, (UserMessage, AssistantMessage))
                    ]
                )

            if not ran_to_completion or permission_denials or result_errors:
                logger.info(
                    "Claude run did not end cleanly",
                    result_subtype=result_subtype,
                    stop_reason=stop_reason,
                    terminal_reason=terminal_reason,
                    permission_denials=len(permission_denials),
                    denied_tools=[d["tool_name"] for d in permission_denials],
                    errors=result_errors,
                    num_turns=num_turns,
                    session_id=final_session_id,
                )

            return ClaudeResponse(
                content=content,
                session_id=final_session_id,
                cost=cost,
                duration_ms=duration_ms,
                num_turns=num_turns,
                tools_used=tools_used,
                interrupted=interrupted,
                result_subtype=result_subtype,
                stop_reason=stop_reason,
                terminal_reason=terminal_reason,
                errors=result_errors,
                permission_denials=permission_denials,
            )

        except asyncio.TimeoutError:
            logger.error(
                "Claude SDK command timed out",
                timeout_seconds=self.config.claude_timeout_seconds,
            )
            raise ClaudeTimeoutError(
                f"Claude SDK timed out after {self.config.claude_timeout_seconds}s"
            )

        except ClaudeTimeoutError:
            # Already a well-typed, well-messaged error raised by the retry
            # loop above (including ClaudeStreamStalledError) — pass through
            # unchanged instead of falling into the generic Exception handler
            # below, which would re-wrap it as an opaque ClaudeProcessError
            # and lose both the specific type callers dispatch on (facade.py
            # preserves the session for retry) and the actionable message.
            raise

        except CLINotFoundError as e:
            logger.error("Claude CLI not found", error=str(e))
            error_msg = (
                "Claude Code not found. Please ensure Claude is installed:\n"
                "  npm install -g @anthropic-ai/claude-code\n\n"
                "If already installed, try one of these:\n"
                "  1. Add Claude to your PATH\n"
                "  2. Create a symlink: ln -s $(which claude) /usr/local/bin/claude\n"
                "  3. Set CLAUDE_CLI_PATH environment variable"
            )
            raise ClaudeProcessError(error_msg)

        except ProcessError as e:
            error_str = str(e)
            # Include captured stderr for better diagnostics
            captured_stderr = "\n".join(stderr_lines[-20:]) if stderr_lines else ""
            if captured_stderr:
                error_str = f"{error_str}\nStderr: {captured_stderr}"
            logger.error(
                "Claude process failed",
                error=error_str,
                exit_code=getattr(e, "exit_code", None),
                stderr=captured_stderr or None,
            )
            # Check if the process error is MCP-related
            if "mcp" in error_str.lower():
                raise ClaudeMCPError(f"MCP server error: {error_str}")
            raise ClaudeProcessError(f"Claude process error: {error_str}")

        except CLIConnectionError as e:
            error_str = str(e)
            logger.error("Claude connection error", error=error_str)
            # Check if the connection error is MCP-related
            if "mcp" in error_str.lower() or "server" in error_str.lower():
                raise ClaudeMCPError(f"MCP server connection failed: {error_str}")
            raise ClaudeProcessError(f"Failed to connect to Claude: {error_str}")

        except CLIJSONDecodeError as e:
            logger.error("Claude SDK JSON decode error", error=str(e))
            raise ClaudeParsingError(f"Failed to decode Claude response: {str(e)}")

        except ClaudeSDKError as e:
            logger.error("Claude SDK error", error=str(e))
            raise ClaudeProcessError(f"Claude SDK error: {str(e)}")

        except Exception as e:
            exceptions = getattr(e, "exceptions", None)
            if exceptions is not None:
                # ExceptionGroup from TaskGroup operations (Python 3.11+)
                logger.error(
                    "Task group error in Claude SDK",
                    error=str(e),
                    error_type=type(e).__name__,
                    exception_count=len(exceptions),
                    exceptions=[str(ex) for ex in exceptions[:3]],
                )
                raise ClaudeProcessError(
                    f"Claude SDK task error: {exceptions[0] if exceptions else e}"
                )

            logger.error(
                "Unexpected error in Claude SDK",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise ClaudeProcessError(f"Unexpected error: {str(e)}")

    async def _handle_stream_message(
        self, message: Message, stream_callback: Callable[[StreamUpdate], None]
    ) -> None:
        """Handle streaming message from claude-agent-sdk."""
        try:
            if isinstance(message, AssistantMessage):
                # Extract content from assistant message
                content = getattr(message, "content", [])
                text_parts = []
                tool_calls = []

                if content and isinstance(content, list):
                    for block in content:
                        if isinstance(block, ToolUseBlock):
                            tool_calls.append(
                                {
                                    "name": block.name,
                                    "input": block.input,
                                    "id": block.id,
                                }
                            )
                        elif isinstance(block, TextBlock):
                            text_parts.append(block.text)
                        elif isinstance(block, ThinkingBlock):
                            text_parts.append(block.thinking)

                if text_parts or tool_calls:
                    update = StreamUpdate(
                        type="assistant",
                        content=("\n".join(text_parts) if text_parts else None),
                        tool_calls=tool_calls if tool_calls else None,
                    )
                    await stream_callback(update)
                elif content:
                    # Fallback for non-list content
                    update = StreamUpdate(
                        type="assistant",
                        content=str(content),
                    )
                    await stream_callback(update)

            elif isinstance(message, StreamEvent):
                event = message.event or {}
                if event.get("type") == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            update = StreamUpdate(
                                type="stream_delta",
                                content=text,
                            )
                            await stream_callback(update)

            elif isinstance(message, UserMessage):
                content = getattr(message, "content", "")
                if content:
                    update = StreamUpdate(
                        type="user",
                        content=content,
                    )
                    await stream_callback(update)

        except Exception as e:
            logger.warning("Stream callback failed", error=str(e))

    def _load_mcp_config(self, config_path: Path) -> Dict[str, Any]:
        """Load MCP server configuration from a JSON file.

        The new claude-agent-sdk expects mcp_servers as a dict, not a file path.
        """
        import json

        try:
            with open(config_path) as f:
                config_data = json.load(f)
            return config_data.get("mcpServers", {})
        except (json.JSONDecodeError, OSError) as e:
            logger.error(
                "Failed to load MCP config", path=str(config_path), error=str(e)
            )
            return {}
