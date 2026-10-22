"""System prompt and SDK options for a harness run.

Mirrors the construction of the old ``claude/sdk_integration.py``
(``execute_command``, removed with the old stack) so a topic behaves the same
under the harness, except
for the two intentional changes: ``permission_mode="default"`` with the
caller's ``can_use_tool`` (the only combination that routes AskUserQuestion to
the callback). Path and bash boundary checks are not applied here: callers
wrap ``can_use_tool`` with ``tool_guard.guard_can_use_tool``.

Rule: the CLI auto-approves any tool listed in ``allowed_tools`` without
calling ``can_use_tool`` (production issue #219). So whenever a callback is
supplied, every tool it must see (AskUserQuestion, the guarded file and bash
tools, plus the configured interactive-approval tools) is stripped from ``allowed_tools`` with a warning.
A prompt file that is missing or not valid UTF-8 is skipped, never fatal.
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple

import structlog
from claude_agent_sdk import ClaudeAgentOptions

from src.config.settings import Settings
from src.dispatcher.agent_loader import load_agent
from src.harness.sdk_lifecycle import load_mcp_config
from src.harness.tool_guard import BASH_TOOLS, FILE_TOOLS

logger = structlog.get_logger()

# Always in front of the agent: on-demand files went unread and lookups
# thrashed (c22os failure-modes #41). Missing files are skipped, never fatal.
PROMPT_FILES = (
    "CLAUDE.md",
    "context/intent-map.md",
    "context/tool-recipes.md",
)
ASK_TOOL = "AskUserQuestion"


def build_system_prompt(
    cwd: Path, agent: Optional[str], enforcement: str = "enforce"
) -> Tuple[str, List[str]]:
    """Return ``(prompt, denied_tools)`` for a topic directory.

    ``denied_tools`` is empty unless the agent declares denials and
    ``enforcement == "enforce"``; in warn mode they are only logged.
    """
    cwd = Path(cwd)
    prompt = f"All file operations must stay within {cwd}. Use relative paths."
    for rel in PROMPT_FILES:
        path = cwd / rel
        try:
            prompt += "\n\n" + path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            logger.warning("Prompt file missing or unreadable", path=str(path))

    denied: List[str] = []
    if agent:
        identity, topic_denied = load_agent(agent, cwd)
        if identity:
            prompt += "\n\n" + identity
        if topic_denied:
            if enforcement == "enforce":
                denied = list(topic_denied)
            else:
                logger.warning(
                    "Topic agent tool denials in warn mode; NOT enforced",
                    agent=agent,
                    would_deny=topic_denied,
                )
    return prompt, denied


def _stderr_callback(line: str) -> None:
    logger.debug("Claude CLI stderr", line=line)


def build_options(
    cwd: Path,
    agent: Optional[str],
    settings: Settings,
    resume: Optional[str],
    can_use_tool: Any,
) -> ClaudeAgentOptions:
    """Build the ClaudeAgentOptions for one run of a topic."""
    agent_enabled = bool(getattr(settings, "agent_loader_enabled", True))
    prompt, agent_denied = build_system_prompt(
        cwd,
        agent if agent_enabled else None,
        str(getattr(settings, "agent_tool_enforcement", "enforce")),
    )

    # Always lists (never None): the 0.2 transport iterates both.
    if settings.disable_tool_validation:
        allowed: List[str] = []
        disallowed: List[str] = []
    else:
        allowed = list(settings.claude_allowed_tools or [])
        disallowed = list(settings.claude_disallowed_tools or [])
    # Agent-level denials are enforcement policy and apply regardless.
    disallowed = [*disallowed, *agent_denied]
    if can_use_tool is not None:
        must_see = {ASK_TOOL} | FILE_TOOLS | BASH_TOOLS
        if getattr(settings, "interactive_tool_approval", False):
            must_see |= set(settings.interactive_tool_approval_tools or ())
        stripped = [t for t in allowed if t in must_see]
        if stripped:
            logger.warning(
                "Stripping pre-approved tools so can_use_tool sees them",
                tools=stripped,
            )
            allowed = [t for t in allowed if t not in must_see]

    options = ClaudeAgentOptions(
        max_turns=settings.claude_max_turns,
        model=settings.claude_model or None,
        max_budget_usd=settings.claude_max_cost_per_request,
        cwd=str(cwd),
        allowed_tools=allowed,
        disallowed_tools=disallowed,
        cli_path=settings.claude_cli_path or None,
        include_partial_messages=True,
        permission_mode="default",
        can_use_tool=can_use_tool,
        sandbox={
            "enabled": settings.sandbox_enabled,
            # Auto-approving sandboxed Bash would bypass can_use_tool.
            "autoAllowBashIfSandboxed": can_use_tool is None,
            "excludedCommands": settings.sandbox_excluded_commands or [],
        },
        system_prompt=prompt,
        setting_sources=["project"],
        stderr=_stderr_callback,
    )
    if settings.enable_mcp and settings.mcp_config_path:
        options.mcp_servers = load_mcp_config(settings.mcp_config_path)
    if resume:
        options.resume = resume
    return options
