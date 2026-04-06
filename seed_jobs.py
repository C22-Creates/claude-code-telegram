"""Seed c22os scheduled jobs into the bot's SQLite database.

Run once after first install, or whenever you want to reset the schedule.
Usage: python seed_jobs.py [--chat-id YOUR_TELEGRAM_USER_ID]
"""

import argparse
import sqlite3
import uuid
from pathlib import Path

# ─── c22os Cron Schedule ───
# All times in EST (UTC-5). APScheduler uses local system TZ.
# Cron format: minute hour day_of_month month day_of_week
JOBS = [
    {
        "name": "journal-prompt",
        "cron": "55 6 * * *",  # 6:55 AM EST daily (before daily-digest)
        "skill": None,
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
        "cron": "0 7 * * *",  # 7:00 AM EST daily
        "skill": "daily-digest",
        "prompt": "",
    },
    {
        "name": "loom-check",
        "cron": "30 7 * * *",  # 7:30 AM EST daily
        "skill": "c22:loom-check",
        "prompt": "",
    },
    {
        "name": "meeting-check",
        "cron": "35 7 * * *",  # 7:35 AM EST daily (stagger 5 min after loom-check)
        "skill": "c22:meeting-check",
        "prompt": "",
    },
    {
        "name": "stay-in-touch",
        "cron": "0 8 * * 3",  # Wednesday 8:00 AM EST
        "skill": "c22:stay-in-touch",
        "prompt": "",
    },
    {
        "name": "coach",
        "cron": "0 9 * * 5",  # Friday 9:00 AM EST
        "skill": "c22:coach",
        "prompt": "",
    },
]

DB_PATH = Path(__file__).parent / "data" / "bot.db"
WORKING_DIR = "/home/carln/Projects/c22os"


PROJECT_SLUG = "c22os"


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


def resolve_project_thread(conn: sqlite3.Connection, slug: str) -> tuple[int, int] | None:
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


def seed(fallback_chat_id: str) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    ensure_table(conn)

    # Clear existing c22os jobs (matched by name)
    conn.execute("DELETE FROM scheduled_jobs WHERE job_name IN ({})".format(
        ",".join(f"'{j['name']}'" for j in JOBS)
    ))

    # Resolve the project_threads row for context / sanity check
    thread_info = resolve_project_thread(conn, PROJECT_SLUG)
    if thread_info is None:
        print(
            f"WARNING: no active project_threads row for slug='{PROJECT_SLUG}'. "
            f"Jobs will fall back to target_chat_ids='{fallback_chat_id}' (DM)."
        )
    else:
        chat_id, thread_id = thread_info
        print(
            f"Resolved project_slug='{PROJECT_SLUG}' → chat_id={chat_id}, "
            f"message_thread_id={thread_id}"
        )

    for job in JOBS:
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
                fallback_chat_id,  # retained as DM fallback if project_threads row goes away
                WORKING_DIR,
                job["skill"],
                PROJECT_SLUG,
            ),
        )

    conn.commit()

    # Verify
    cursor = conn.execute(
        """
        SELECT sj.job_name, sj.cron_expression, sj.skill_name,
               sj.project_slug, pt.chat_id, pt.message_thread_id
        FROM scheduled_jobs sj
        LEFT JOIN project_threads pt
          ON pt.project_slug = sj.project_slug AND pt.is_active = 1
        WHERE sj.is_active = 1
        ORDER BY sj.created_at
        """
    )
    print("\nSeeded jobs:")
    for row in cursor.fetchall():
        name, cron, skill, slug, chat, thread = row
        target = (
            f"chat={chat} thread={thread}" if chat is not None
            else f"DM fallback={fallback_chat_id}"
        )
        print(f"  {name:20s} | {cron:15s} | /{skill or '-'} | slug={slug} | {target}")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed c22os cron jobs")
    parser.add_argument(
        "--chat-id",
        required=True,
        help=(
            "DM fallback Telegram user/chat ID. Jobs route to the c22os "
            "project_threads topic when active; this is only used if that "
            "mapping is ever removed."
        ),
    )
    args = parser.parse_args()
    seed(args.chat_id)
    print(f"\nDone. DB: {DB_PATH}")
