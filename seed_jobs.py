"""Seed c22os scheduled jobs into the bot's SQLite database.

Run once after first install, or whenever you want to reset the schedule.
Usage: python seed_jobs.py [--chat-id YOUR_TELEGRAM_USER_ID] [--projects-root /path/to/Projects]

Jobs are defined inline in this file (JOBS list below). Each job specifies
its own project_slug and working_directory relative to --projects-root, so
different jobs can deliver into different Telegram forum topics. This matches
the multi-topic routing added on 2026-04-12.
"""

import argparse
import sqlite3
import uuid
from pathlib import Path


# Long prompt for the daily self-builder review (5 PM EST, Ops topic).
EVOLVE_SUGGEST_PROMPT = """\
You are the self-builder agent for c22os. Read agents/self-builder/SOUL.md, RULES.md, DUTIES.md and honor those boundaries strictly. This is the daily evening review.

Analyze what happened in Carl's day and in c22os today. Suggest 1-3 improvements worth building, or say nothing if there's nothing worth proposing. Quiet days are fine.

Sources to examine (use what's available; skip what isn't):
1. Git activity (24h): `git log --since="24 hours ago" --name-only --oneline`
2. Recently modified inbox/, workflows/, .claude/commands/
3. Today's calendar events across Carl's accounts (Google Workspace MCP; c22, wondercare, personal)
4. Today's email traffic patterns (Gmail MCP; volume and themes, not content)
5. Recent bot session history (what commands got used, what tripped up)
6. claude-mem observations if accessible

Look for:
- Manual workflow repeating that should be automated -> propose a new slash command
- Capability being reinvented across files -> propose a new skill
- Emerging process without documentation -> propose a new workflow doc
- Existing thing drifting from its original purpose -> propose a refactor
- Calendar + email patterns suggesting missing automation (e.g., "3 meetings today all needed the same pre-meeting research Carl did by hand")

Output (max 400 words, goes to Telegram):

## Today I saw
[1 sentence per observation, max 3]

## I'd build
1. [name] -- [one-line description]
   Effort: [S/M/L] | Impact: [who benefits, how often]
2. ...

## Skip for now
[anything considered but rejected, 1 line + reason. Omit this section if nothing was skipped.]

Rules:
- Be opinionated. "Build this" or "skip this." No hedging.
- Reference calendar/email at pattern level, not content. "4 meetings today" not "Carl met with Jane Doe about X."
- If genuinely nothing is worth building today, reply "Quiet day. Nothing worth building." One line. Don't pad.
- DO NOT write files or create branches. Telegram observation only. Carl will manually ask you to expand items into proposals."""


# Long prompt for the weekly self-builder synthesis (Fri 3 PM EST, Ops topic).
EVOLVE_SYNTHESIZE_PROMPT = """\
You are the self-builder agent for c22os. Read agents/self-builder/SOUL.md, RULES.md, DUTIES.md and honor those boundaries strictly. This is the weekly synthesis.

Look across the past 7 days for patterns that only emerge at weekly scale. The daily review catches same-day patterns; your job is to spot compounding signals.

Sources (7-day window):
1. `git log --since="7 days ago" --name-only`
2. inbox/ items added or modified this week
3. Calendar patterns this week across Carl's accounts (meeting types, recurring conflicts, prep time ratios)
4. Email volume + thread patterns this week (themes, not content)
5. Scheduled job output (what c22os delivered via Telegram this week)
6. claude-mem if accessible

Look for:
- Repeating weekly patterns that could become automated workflows
- Commands never run this week (candidates for retirement)
- Ideas captured in inbox/ that are now ripe to build (research is done, scope is clear)
- Drift between about-carl.md stated priorities and actual time allocation (calendar reveals this)
- Gaps in coverage: which priority areas got zero automation support this week?

Output (max 600 words, weekly synthesis is longer than daily):

## The week in patterns
[3-5 observations, grouped by theme if natural]

## Build this week
1. [name] -- [1-2 sentence description]
   Effort: [S/M/L] | Impact: [quantify: time saved, accuracy gained, revenue unlocked] | Dependencies: [what needs to be in place first]
2. ...

## Retire or refactor
[commands/workflows that are unused or drifted, with disposition recommendation]

## Time-alignment check
[one paragraph: is actual calendar time matching Carl's stated 50% WonderCare / 20% consulting / 15% learning / 15% admin target? where is the drift?]

Rules:
- Quantify. "Spent 6 hours on learning vs. 2 hours on consulting pipeline this week" is gold. Hand-waving is not.
- Be ruthlessly opinionated on priorities.
- DO NOT write files or branches. Telegram only. Manual build-initiation after.
- If the week genuinely has no patterns worth synthesizing, reply "Thin week. Nothing worth synthesizing." That should be rare at weekly scale."""


# ─── c22os Cron Schedule ───
# All times in EST (UTC-5). APScheduler uses the service's local TZ
# (America/New_York via systemd Environment). Cron format: m h dom mon dow.
#
# Each job's working_directory is a subdirectory of --projects-root, and
# project_slug maps to a project_threads row that resolves to a supergroup
# forum topic for delivery. Use symlinks to share the same checkout across
# multiple slugs for per-topic session isolation (e.g., c22os-journal -> c22os).
JOBS = [
    {
        "name": "journal-prompt",
        "cron": "55 6 * * *",  # 6:55 AM daily
        "skill": None,
        "project_slug": "journal",
        "working_directory_name": "c22os-journal",
        "prompt": (
            "Send Carl a warm, brief morning journal prompt. "
            "Ask him to share what's on his mind: goals for the day, "
            "joyous moments, zone of genius reflections, meaningful connections, "
            "or anything he wants to capture. Keep it to 2-3 sentences, "
            "inviting and not a wall of text. Vary the prompt each day. "
            "When he responds with his thoughts, process his reply using "
            "the /c22:journal workflow to structure it, infer Notion properties, "
            "and save to the Journal database."
        ),
    },
    {
        "name": "daily-digest",
        "cron": "0 7 * * *",  # 7:00 AM daily
        "skill": "daily-digest",
        "project_slug": "c22os",
        "working_directory_name": "c22os",
        "prompt": "",
    },
    {
        "name": "meeting-check",
        "cron": "35 7 * * *",  # 7:35 AM daily
        "skill": "c22:meeting-check",
        "project_slug": "relationships",
        "working_directory_name": "c22os-relationships",
        "prompt": "",
    },
    {
        "name": "stay-in-touch",
        "cron": "0 8 * * 3",  # Wed 8:00 AM
        "skill": "c22:stay-in-touch",
        "project_slug": "relationships",
        "working_directory_name": "c22os-relationships",
        "prompt": "",
    },
    {
        "name": "coach",
        "cron": "0 9 * * 5",  # Fri 9:00 AM
        "skill": "c22:coach",
        "project_slug": "journal",
        "working_directory_name": "c22os-journal",
        "prompt": "",
    },
    {
        "name": "evolve-suggest",
        "cron": "0 17 * * *",  # 5:00 PM daily
        "skill": None,
        "project_slug": "ops",
        "working_directory_name": "c22os-ops",
        "prompt": EVOLVE_SUGGEST_PROMPT,
    },
    {
        "name": "evolve-synthesize",
        "cron": "0 15 * * 5",  # Fri 3:00 PM
        "skill": None,
        "project_slug": "ops",
        "working_directory_name": "c22os-ops",
        "prompt": EVOLVE_SYNTHESIZE_PROMPT,
    },
]

DB_PATH = Path(__file__).parent / "data" / "bot.db"


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scheduled_jobs (
            job_id TEXT PRIMARY KEY,
            job_name TEXT NOT NULL,
            cron_expression TEXT NOT NULL,
            prompt TEXT NOT NULL DEFAULT '',
            target_chat_ids TEXT NOT NULL DEFAULT '',
            working_directory TEXT NOT NULL,
            skill_name TEXT,
            created_by INTEGER NOT NULL DEFAULT 0,
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            project_slug TEXT
        )
    """)
    # Idempotent column add for databases migrated from an earlier schema.
    try:
        conn.execute("ALTER TABLE scheduled_jobs ADD COLUMN project_slug TEXT")
    except sqlite3.OperationalError:
        pass  # column already exists
    conn.commit()


def resolve_project_thread(
    conn: sqlite3.Connection, slug: str
) -> tuple[int, int] | None:
    """Look up (chat_id, message_thread_id) for a project slug, if present."""
    row = conn.execute(
        """
        SELECT chat_id, message_thread_id
        FROM project_threads
        WHERE project_slug = ? AND is_active = 1
        LIMIT 1
        """,
        (slug,),
    ).fetchone()
    if row is None:
        return None
    return (int(row[0]), int(row[1]))


def seed(fallback_chat_id: str, projects_root: Path) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    ensure_table(conn)

    # Clear existing jobs that we're about to re-seed (matched by name)
    conn.execute(
        "DELETE FROM scheduled_jobs WHERE job_name IN ({})".format(
            ",".join(f"'{j['name']}'" for j in JOBS)
        )
    )

    # Sanity-check each unique project_slug resolves to a thread
    slugs = sorted({j["project_slug"] for j in JOBS})
    print("Project thread resolution:")
    for slug in slugs:
        thread_info = resolve_project_thread(conn, slug)
        if thread_info is None:
            print(
                f"  WARNING: no active project_threads row for slug='{slug}'. "
                f"Jobs with this slug will fall back to DM={fallback_chat_id}."
            )
        else:
            chat_id, thread_id = thread_info
            print(
                f"  {slug:15s} -> chat_id={chat_id}, message_thread_id={thread_id}"
            )
    print()

    for job in JOBS:
        working_dir = str(projects_root / job["working_directory_name"])
        conn.execute(
            """INSERT INTO scheduled_jobs
               (job_id, job_name, cron_expression, prompt, target_chat_ids,
                working_directory, skill_name, created_by, is_active, project_slug)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0, 1, ?)""",
            (
                str(uuid.uuid4()),
                job["name"],
                job["cron"],
                job["prompt"],
                fallback_chat_id,
                working_dir,
                job["skill"],
                job["project_slug"],
            ),
        )

    conn.commit()

    # Verify
    cursor = conn.execute(
        """
        SELECT sj.job_name, sj.cron_expression, sj.skill_name,
               sj.project_slug, sj.working_directory,
               pt.chat_id, pt.message_thread_id
        FROM scheduled_jobs sj
        LEFT JOIN project_threads pt
          ON pt.project_slug = sj.project_slug AND pt.is_active = 1
        WHERE sj.is_active = 1
        ORDER BY sj.cron_expression
        """
    )
    print("Seeded jobs:")
    for row in cursor.fetchall():
        name, cron, skill, slug, wd, chat, thread = row
        skill_str = f"/{skill}" if skill else "(free-form)"
        target = (
            f"chat={chat} thread={thread}"
            if chat is not None
            else f"DM fallback={fallback_chat_id}"
        )
        print(
            f"  {name:20s} | {cron:15s} | {skill_str:22s} "
            f"| slug={slug:14s} | wd={wd} | {target}"
        )

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed c22os cron jobs")
    parser.add_argument(
        "--chat-id",
        required=True,
        help=(
            "DM fallback Telegram user/chat ID. Jobs route to their "
            "project_threads topic when active; this is only used if that "
            "mapping is ever removed."
        ),
    )
    parser.add_argument(
        "--projects-root",
        default=str(Path.home() / "Projects"),
        help=(
            "Parent directory containing the c22os checkout and its "
            "per-topic symlinks (c22os, c22os-journal, c22os-relationships, "
            "c22os-ops). Defaults to ~/Projects."
        ),
    )
    args = parser.parse_args()
    seed(args.chat_id, Path(args.projects_root))
    print(f"\nDone. DB: {DB_PATH}")
