"""Per-topic session persistence and run-state rows for restart recovery.

Timestamps are epoch seconds (float, UTC). ``now`` is injectable everywhere a
time is read so tests never sleep.
"""

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import structlog

logger = structlog.get_logger()

PROMPT_HEAD_CHARS = 200
FINAL_STATUSES = frozenset({"done", "stopped", "error", "abandoned"})


@dataclass(frozen=True)
class SessionRow:
    topic_key: str
    session_id: str
    cwd: str
    created_at: float
    last_active: float


@dataclass(frozen=True)
class RunRow:
    run_id: str
    topic_key: str
    chat_id: Optional[int]
    thread_id: Optional[int]
    status: str
    started_at: float
    ended_at: Optional[float]
    prompt_head: Optional[str]


def project_slug(cwd: Path) -> str:
    """Claude Code's transcript dir name: every "/" and "." becomes "-"."""
    return str(cwd).replace("/", "-").replace(".", "-")


class SessionStore:
    """Persistence for harness sessions and runs (tables from migration 7)."""

    def __init__(
        self,
        db_manager,
        projects_root: Optional[Path] = None,
        rotate_idle_seconds: int = 7200,
        rotate_transcript_bytes: int = 400_000,
    ):
        self._db = db_manager
        self._projects_root = (
            Path(projects_root)
            if projects_root is not None
            else Path.home() / ".claude" / "projects"
        )
        self.rotate_idle_seconds = rotate_idle_seconds
        self.rotate_transcript_bytes = rotate_transcript_bytes

    @classmethod
    def from_settings(cls, db_manager, settings, **kw) -> "SessionStore":
        return cls(
            db_manager,
            rotate_idle_seconds=settings.harness_rotate_idle_seconds,
            rotate_transcript_bytes=settings.harness_rotate_transcript_bytes,
            **kw,
        )

    async def get(self, topic_key: str) -> Optional[SessionRow]:
        async with self._db.get_connection() as conn:
            cur = await conn.execute(
                "SELECT topic_key, session_id, cwd, created_at, last_active "
                "FROM harness_sessions WHERE topic_key = ?",
                (topic_key,),
            )
            row = await cur.fetchone()
        return SessionRow(*tuple(row)) if row else None

    async def save(
        self,
        topic_key: str,
        session_id: str,
        cwd: Path,
        now: Optional[float] = None,
    ) -> None:
        """Upsert; ``created_at`` is kept while the session id is unchanged."""
        ts = time.time() if now is None else now
        async with self._db.get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO harness_sessions
                    (topic_key, session_id, cwd, created_at, last_active)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(topic_key) DO UPDATE SET
                    created_at = CASE
                        WHEN session_id = excluded.session_id THEN created_at
                        ELSE excluded.created_at END,
                    session_id = excluded.session_id,
                    cwd = excluded.cwd,
                    last_active = excluded.last_active
                """,
                (topic_key, session_id, str(cwd), ts, ts),
            )
            await conn.commit()

    async def clear(self, topic_key: str) -> None:
        async with self._db.get_connection() as conn:
            await conn.execute(
                "DELETE FROM harness_sessions WHERE topic_key = ?", (topic_key,)
            )
            await conn.commit()

    def transcript_bytes(self, session_id: str, cwd: Path) -> int:
        path = self._projects_root / project_slug(Path(cwd)) / f"{session_id}.jsonl"
        try:
            return path.stat().st_size
        except OSError:
            return 0

    def needs_rotation(self, row: SessionRow, now: float) -> bool:
        """True when idle over the limit or the transcript is over the cap."""
        if now - row.last_active > self.rotate_idle_seconds:
            return True
        size = self.transcript_bytes(row.session_id, Path(row.cwd))
        return size > self.rotate_transcript_bytes

    async def start_run(
        self,
        topic_key: str,
        chat_id: Optional[int],
        thread_id: Optional[int],
        prompt: str,
        now: Optional[float] = None,
    ) -> str:
        run_id = uuid.uuid4().hex
        ts = time.time() if now is None else now
        async with self._db.get_connection() as conn:
            await conn.execute(
                "INSERT INTO harness_runs (run_id, topic_key, chat_id, thread_id, "
                "status, started_at, prompt_head) VALUES (?, ?, ?, ?, 'running', ?, ?)",
                (run_id, topic_key, chat_id, thread_id, ts, prompt[:PROMPT_HEAD_CHARS]),
            )
            await conn.commit()
        return run_id

    async def end_run(
        self, run_id: str, status: str, now: Optional[float] = None
    ) -> None:
        if status not in FINAL_STATUSES:
            raise ValueError(f"invalid final run status: {status!r}")
        ts = time.time() if now is None else now
        async with self._db.get_connection() as conn:
            await conn.execute(
                "UPDATE harness_runs SET status = ?, ended_at = ? WHERE run_id = ?",
                (status, ts, run_id),
            )
            await conn.commit()

    async def running_runs(self) -> List[RunRow]:
        async with self._db.get_connection() as conn:
            cur = await conn.execute(
                "SELECT run_id, topic_key, chat_id, thread_id, status, started_at, "
                "ended_at, prompt_head FROM harness_runs WHERE status = 'running' "
                "ORDER BY started_at"
            )
            rows = await cur.fetchall()
        return [RunRow(*tuple(r)) for r in rows]

    async def abandon_all_running(self, now: Optional[float] = None) -> None:
        """Flip rows left 'running' by a crash/restart; other statuses stay."""
        ts = time.time() if now is None else now
        async with self._db.get_connection() as conn:
            await conn.execute(
                "UPDATE harness_runs SET status = 'abandoned', ended_at = ? "
                "WHERE status = 'running'",
                (ts,),
            )
            await conn.commit()
