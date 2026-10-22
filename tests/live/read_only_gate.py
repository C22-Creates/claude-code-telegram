"""Deny-by-default read-only gate for the live replays.

Importable without HARNESS_LIVE so its unit test runs in the default suite.
No heuristics: tools and shell commands must match an explicit allowlist.
Caveat: a tool pre-approved by a skill's ``allowed-tools`` frontmatter would
bypass ``can_use_tool`` entirely (no c22 command declares it today). Reads of
local files (cat/grep) can expose local secrets; that is read-only by design.
"""

import re
from typing import Any

READ_ONLY_DENIAL = "live replay is read-only"
SAFE_TOOLS = frozenset(
    {"Skill", "Read", "Glob", "Grep", "ToolSearch", "TodoWrite", "WebSearch"}
)
READ_MCP_TOOLS = frozenset(
    {
        "mcp__notion__notion-query-data-sources",
        "mcp__notion__notion-search",
        "mcp__notion__notion-fetch",
        "mcp__missive__list_conversations",
        "mcp__missive__get_conversation",
        "mcp__missive__list_contacts",
        "mcp__missive__get_contact",
    }
)

_GOG_WRITE_FLAGS = re.compile(
    r"(?:^|\s)(?:--(?:create|delete|update|add|remove|patch|insert|send|"
    r"move|set|write)\S*|-d)(?=\s|=|$)"
)
_SEGMENT = re.compile(
    r"^(?:"
    r"gog\s+(?:-{1,2}\S+\s+)*calendar\s+events\b[^\n]*"
    r"|python3\s+scripts/contacts-lookup\.py\b[^\n]*"
    r"|(?:cat|ls|jq|head|tail|grep|wc|sort|pwd|echo)(?:\s[^\n]*)?"
    r")$"
)
_STDERR_TO_STDOUT = "2>&1"


def _split_segments(cmd: str) -> list[str] | None:
    """Split on ``&&``, ``||``, ``;``, ``|`` outside quotes; None if unsafe.

    Single quotes are fully literal (jq programs live there). Double quotes
    still allow ``$(`` and backticks, so those are rejected everywhere. Outside
    quotes, newlines, ``<``, ``>`` and a bare ``&`` are rejected; the only
    redirect allowed is the whole token ``2>&1``.
    """
    segments: list[str] = []
    cur: list[str] = []
    quote = ""
    i = 0
    while i < len(cmd):
        c = cmd[i]
        if c in "\n\r`" or cmd.startswith("$(", i):
            return None
        if quote:
            if c == quote:
                quote = ""
            elif c == "\\" and quote == '"':
                cur.append(c)
                i += 1
                c = cmd[i] if i < len(cmd) else ""
            cur.append(c)
            i += 1
            continue
        if c in "'\"":
            quote = c
        elif (
            cmd.startswith(_STDERR_TO_STDOUT, i)
            and (i == 0 or cmd[i - 1].isspace())
            and (i + 4 == len(cmd) or cmd[i + 4].isspace())
        ):
            cur.append(" ")
            i += 4
            continue
        elif c in "<>":
            return None
        elif cmd.startswith(("&&", "||"), i):
            segments.append("".join(cur))
            cur = []
            i += 2
            continue
        elif c in ";|":
            segments.append("".join(cur))
            cur = []
            i += 1
            continue
        elif c == "&":
            return None
        cur.append(c)
        i += 1
    if quote:
        return None
    segments.append("".join(cur))
    return [seg.strip() for seg in segments]


def bash_read_only(cmd: str) -> bool:
    """True only if every segment of ``cmd`` is an allowlisted read."""
    segments = _split_segments(cmd)
    if not segments or not all(segments):
        return False
    for seg in segments:
        if not _SEGMENT.match(seg):
            return False
        if seg.startswith("gog") and _GOG_WRITE_FLAGS.search(seg):
            return False
        if seg.startswith("sort") and re.search(r"\s-\w*o|--output", seg):
            return False
    return True


def is_read_only(tool: str, tool_input: dict[str, Any]) -> bool:
    """Allowlist for the live replays; everything else is denied."""
    if tool in SAFE_TOOLS or tool in READ_MCP_TOOLS:
        return True
    if tool == "Bash":
        return bash_read_only(str(tool_input.get("command", "")))
    return False
