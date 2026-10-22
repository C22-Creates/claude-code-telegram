"""Deny-by-default read-only gate for the live replays.

Importable without HARNESS_LIVE so its unit test runs in the default suite.
No heuristics: tools and shell commands must match an explicit allowlist.
Caveat: a tool pre-approved by a skill's ``allowed-tools`` frontmatter would
bypass ``can_use_tool`` entirely (no c22 command declares it today). Reads of
local files (cat/grep) can expose local secrets; that is read-only by design.
"""

import re
import shlex
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

_GOG_PREFIX = ("gog", "calendar", "events")
_GOG_BOOL_FLAGS = frozenset({"--json", "--results-only"})
_GOG_VALUE_FLAGS = frozenset({"--account", "--from", "--to", "--max"})
_GOG_VALUE = re.compile(r"^[\w.:+@/-]+$")


def gog_read_only(seg: str) -> bool:
    """``gog calendar events [<calendar-id>] [flags]`` with only the flags the
    replays use, spelled exactly (no abbreviations, no other subcommands)."""
    try:
        tokens = shlex.split(seg)
    except ValueError:
        return False
    if tuple(tokens[:3]) != _GOG_PREFIX:
        return False
    rest = tokens[3:]
    i = 0
    if rest and not rest[0].startswith("-"):
        if not _GOG_VALUE.match(rest[0]):
            return False
        i = 1
    while i < len(rest):
        flag = rest[i]
        if flag in _GOG_BOOL_FLAGS:
            i += 1
        elif flag in _GOG_VALUE_FLAGS and i + 1 < len(rest):
            if not _GOG_VALUE.match(rest[i + 1]) or rest[i + 1].startswith("-"):
                return False
            i += 2
        else:
            return False
    return True


_SEGMENT = re.compile(
    r"^(?:"
    r"gog\s[^\n]*"
    r"|python3\s+scripts/contacts-lookup\.py\b[^\n]*"
    r"|(?:cat|ls|jq|head|tail|grep|wc|pwd|echo)(?:\s[^\n]*)?"
    r")$"
)
_STDERR_TO_STDOUT = "2>&1"
# Deny by default: outside single quotes only these characters are allowed.
# Everything that bash expands or interprets ($ ` \ ! # ~ * ? [ ] { } ( ) < >,
# a bare &, newlines) is absent, so the scanner never has to model it.
_SAFE_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_./:=,+@%-"
)


def _split_segments(cmd: str) -> list[str] | None:
    """Split on ``&&``, ``||``, ``;``, ``|`` outside quotes; None if unsafe.

    Why a character allowlist rather than a list of dangerous constructs: the
    previous blocklist splitter was bypassed by ANSI-C ``$'...'`` quoting, where
    bash treats ``\\'`` as an escaped quote and the splitter did not. Here:

    - Outside quotes: only ``_SAFE_CHARS``, spaces, the separators, a whole
      ``2>&1`` token, and the two quote characters.
    - Single quotes: fully literal in bash (jq programs live there). Because
      ``$`` is never allowed outside quotes, ``'`` always opens a plain
      single-quoted region, never ANSI-C or locale quoting.
    - Double quotes: only ``_SAFE_CHARS`` and spaces, so nothing inside them
      can expand or escape.
    - Newline/CR are rejected everywhere.
    """
    if "\n" in cmd or "\r" in cmd:
        return None
    segments: list[str] = []
    cur: list[str] = []
    quote = ""
    i = 0
    while i < len(cmd):
        c = cmd[i]
        if quote:
            if c == quote:
                quote = ""
            elif quote == '"' and c not in _SAFE_CHARS and c != " ":
                return None
            cur.append(c)
            i += 1
            continue
        if c in "'\"":
            quote = c
        elif (
            cmd.startswith(_STDERR_TO_STDOUT, i)
            and (i == 0 or cmd[i - 1] == " ")
            and (i + 4 == len(cmd) or cmd[i + 4] == " ")
        ):
            cur.append(" ")
            i += 4
            continue
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
        elif c not in _SAFE_CHARS and c != " ":
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
        if seg.startswith("gog") and not gog_read_only(seg):
            return False
    return True


def is_read_only(tool: str, tool_input: dict[str, Any]) -> bool:
    """Allowlist for the live replays; everything else is denied."""
    if tool in SAFE_TOOLS or tool in READ_MCP_TOOLS:
        return True
    if tool == "Bash":
        return bash_read_only(str(tool_input.get("command", "")))
    return False
