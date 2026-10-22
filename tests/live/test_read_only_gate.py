"""Fast unit test of the live replays' read-only gate (no HARNESS_LIVE needed)."""

import pytest

from tests.live.read_only_gate import is_read_only

# Every Bash command recorded in the live runs (spike doc section (e)).
RECORDED = [
    'python3 scripts/contacts-lookup.py "McGovern"; echo "---CRYSTAL---"; '
    'python3 scripts/contacts-lookup.py "Crystal"',
    "gog calendar events carl@nevado.ai --account nevado --from 2026-10-22 "
    "--to 2026-10-23 --max 50 --json --results-only",
    "gog calendar events carl@c22creates.com --account c22 --from 2026-10-22 "
    "--to 2026-10-23 --max 50 --json --results-only",
    "gog calendar events carl@wondercare.com --account wondercare --from "
    "2026-10-22 --to 2026-10-23 --max 50 --json --results-only",
    "gog calendar events carl.niedbala@gmail.com --account personal --from "
    "2026-10-22 --to 2026-10-23 --max 50 --json --results-only",
    "gog calendar events carl@nevado.ai --account nevado --from 2026-10-22 "
    "--to 2026-10-23 --max 50 --json --results-only | jq -r '[.[] | {summary}]'",
    "jq -r '[.[] | select((.start.dateTime // .start.date) | "
    'startswith("2026-10-22"))] | sort_by(.start.dateTime)\' f.json',
    'python3 scripts/contacts-lookup.py "ocean-ledger.com" 2>&1 | head -60',
    # backslashes are literal inside single quotes (jq string interpolation)
    "jq -r '.[] | \"\\(.summary) \\(.start.date)\"' f.json",
]
BYPASSES = [
    "ls\nrm -rf /tmp/x",
    "ls\r\nrm x",
    "ls & rm /tmp/x",
    "ls &rm x",
    "gog gmail send --to a calendar events",
    "gog calendar events x --create",
    "gog calendar events x --delete=1",
    "gog calendar events x -d",
    "date -s 2020-01-01",
    "jq . f > out",
    "cat x | tee y",
    "cat x 2>&1> y",
    "cat x2>&1",
    "echo $(rm x)",
    "echo `rm x`",
    "ls ;",
    "rm -rf /tmp/x",
    "sort -o out f",
    "python3 -c 'import os'",
    "echo \\' ; rm x ; echo \\'",
    'echo "a\\" ; rm x ; echo \\""',
    "sort --ou=/tmp/f x",
    "sort -S 1 --compress-program=rm f",
    "gog calendar events create x",
    "gog calendar events --crea=1",
    "gog calendar events --out=/tmp/f",
    "gog calendar events x --client-secret-file f",
    "gog calendar events x --account",
    "gog --account c22 calendar events x",
    "gog calendar create x",
]


@pytest.mark.parametrize("cmd", RECORDED)
def test_recorded_commands_allowed(cmd):
    assert is_read_only("Bash", {"command": cmd})


@pytest.mark.parametrize("cmd", BYPASSES)
def test_bash_bypasses_denied(cmd):
    assert not is_read_only("Bash", {"command": cmd})


@pytest.mark.parametrize(
    "tool",
    [
        "mcp__read-ai__share_folder",
        "mcp__read-ai__share_meeting_report",
        "mcp__missive__create_draft",
        "mcp__missive__create_post",
        "mcp__notion__notion-update-page",
        "mcp__claude_ai_Spotify__generate_playlist",
        "Write",
        "Edit",
    ],
)
def test_write_tools_denied(tool):
    assert not is_read_only(tool, {})


@pytest.mark.parametrize(
    "tool",
    [
        "Skill",
        "mcp__notion__notion-query-data-sources",
        "mcp__missive__list_conversations",
    ],
)
def test_read_tools_allowed(tool):
    assert is_read_only(tool, {})
