"""Path and bash boundary guard for every harness tool call.

Ported from ``claude/monitor.py`` and ``_make_can_use_tool_callback`` in
``claude/sdk_integration.py`` (both since deleted). Production routed every
tool call through these checks; ``guard_can_use_tool`` restores that for
interactive and unattended runs alike, delegating to the run's own callback
(AskBridge, allow-all, unattended) only after the checks pass.
"""

import shlex
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import structlog
from claude_agent_sdk import CanUseTool, PermissionResultDeny

from src.security.validators import SecurityValidator

logger = structlog.get_logger()

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

_FILE_PATH_KEYS = ("file_path", "path", "notebook_path")

# Subdirectories under ~/.claude/ that Claude Code uses internally.
_CLAUDE_INTERNAL_SUBDIRS: Set[str] = {"plans", "todos", "settings.json"}

# Commands that modify the filesystem or change context and should have paths checked
_FS_MODIFYING_COMMANDS: Set[str] = {
    "mkdir",
    "touch",
    "cp",
    "mv",
    "rm",
    "rmdir",
    "ln",
    "install",
    "tee",
    "cd",
}

# Commands that are read-only or don't take filesystem paths
_READ_ONLY_COMMANDS: Set[str] = {
    "cat",
    "ls",
    "head",
    "tail",
    "less",
    "more",
    "which",
    "whoami",
    "pwd",
    "echo",
    "printf",
    "env",
    "printenv",
    "date",
    "wc",
    "sort",
    "uniq",
    "diff",
    "file",
    "stat",
    "du",
    "df",
    "tree",
    "realpath",
    "dirname",
    "basename",
}

# Actions / expressions that make ``find`` a filesystem-modifying command
_FIND_MUTATING_ACTIONS: Set[str] = {"-delete", "-exec", "-execdir", "-ok", "-okdir"}

# Bash command separators
_COMMAND_SEPARATORS: Set[str] = {"&&", "||", ";", "|", "&"}


def check_bash_directory_boundary(
    command: str,
    working_directory: Path,
    approved_directory: Path,
) -> Tuple[bool, Optional[str]]:
    """Check if a bash command's paths stay within the approved directory."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        # If we can't parse the command, let it through —
        # the sandbox will catch it at the OS level
        return True, None

    if not tokens:
        return True, None

    # Split tokens into individual commands based on separators
    command_chains: list[list[str]] = []
    current_chain: list[str] = []

    for token in tokens:
        if token in _COMMAND_SEPARATORS:
            if current_chain:
                command_chains.append(current_chain)
            current_chain = []
        else:
            current_chain.append(token)

    if current_chain:
        command_chains.append(current_chain)

    resolved_approved = approved_directory.resolve()

    # Check each command in the chain
    for cmd_tokens in command_chains:
        if not cmd_tokens:
            continue

        base_command = Path(cmd_tokens[0]).name

        # Read-only commands are always allowed
        if base_command in _READ_ONLY_COMMANDS:
            continue

        # Determine if this specific command in the chain needs path validation
        needs_check = False
        if base_command == "find":
            needs_check = any(t in _FIND_MUTATING_ACTIONS for t in cmd_tokens[1:])
        elif base_command in _FS_MODIFYING_COMMANDS:
            needs_check = True

        if not needs_check:
            continue

        # Check each argument for paths outside the boundary
        for token in cmd_tokens[1:]:
            # Skip flags
            if token.startswith("-"):
                continue

            # Resolve both absolute and relative paths against the working
            # directory so that traversal sequences like ``../../evil`` are
            # caught instead of being silently allowed.
            try:
                if token.startswith("/"):
                    resolved = Path(token).resolve()
                else:
                    resolved = (working_directory / token).resolve()

                if not _is_within_directory(resolved, resolved_approved):
                    return False, (
                        f"Directory boundary violation: '{base_command}' targets "
                        f"'{token}' which is outside approved directory "
                        f"'{resolved_approved}'"
                    )
            except (ValueError, OSError):
                # If path resolution fails, the command might be malformed or
                # using bash features we can't statically analyze.
                # We skip checking this token and rely on the OS-level sandbox.
                continue

    return True, None


def _is_claude_internal_path(file_path: str) -> bool:
    """Check whether *file_path* points inside ``~/.claude/`` (allowed subdirs only)."""
    try:
        resolved = Path(file_path).resolve()
        home = Path.home().resolve()
        claude_dir = home / ".claude"

        # Path must be inside ~/.claude/
        try:
            rel = resolved.relative_to(claude_dir)
        except ValueError:
            return False

        # Must be in one of the known subdirectories (or a known file)
        top_part = rel.parts[0] if rel.parts else ""
        return top_part in _CLAUDE_INTERNAL_SUBDIRS

    except Exception:
        return False


def _is_within_directory(path: Path, directory: Path) -> bool:
    """Check if path is within directory."""
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def guard_can_use_tool(
    inner: CanUseTool,
    validator: SecurityValidator,
    working_directory: Path,
    approved_directory: Path,
) -> CanUseTool:
    """Wrap *inner* so file and bash boundary checks run first (deny on violation)."""

    async def guarded(tool_name: str, tool_input: Dict[str, Any], context: Any) -> Any:
        if tool_name in FILE_TOOLS:
            file_path = next(
                (tool_input.get(key) for key in _FILE_PATH_KEYS if tool_input.get(key)),
                None,
            )
            # Claude Code internal paths (~/.claude/plans/, etc.) skip validation.
            if file_path and not _is_claude_internal_path(file_path):
                valid, _resolved, error = validator.validate_path(
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

        return await inner(tool_name, tool_input, context)

    return guarded
