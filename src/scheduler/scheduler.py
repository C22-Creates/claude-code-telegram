"""Job scheduler for recurring agent tasks.

Wraps APScheduler's AsyncIOScheduler and publishes ScheduledEvents
to the event bus when jobs fire.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from apscheduler.schedulers.asyncio import (
    AsyncIOScheduler,  # type: ignore[import-untyped]
)
from apscheduler.triggers.cron import CronTrigger  # type: ignore[import-untyped]

from ..events.bus import EventBus
from ..events.types import ScheduledEvent
from ..storage.database import DatabaseManager

logger = structlog.get_logger()


class JobScheduler:
    """Cron scheduler that publishes ScheduledEvents to the event bus."""

    def __init__(
        self,
        event_bus: EventBus,
        db_manager: DatabaseManager,
        default_working_directory: Path,
    ) -> None:
        self.event_bus = event_bus
        self.db_manager = db_manager
        self.default_working_directory = default_working_directory
        self._scheduler = AsyncIOScheduler()

    async def start(self) -> None:
        """Load persisted jobs and start the scheduler."""
        await self._load_jobs_from_db()
        self._scheduler.start()
        logger.info("Job scheduler started")

    async def stop(self) -> None:
        """Shutdown the scheduler gracefully."""
        self._scheduler.shutdown(wait=False)
        logger.info("Job scheduler stopped")

    async def add_job(
        self,
        job_name: str,
        cron_expression: str,
        prompt: str,
        target_chat_ids: Optional[List[int]] = None,
        working_directory: Optional[Path] = None,
        skill_name: Optional[str] = None,
        created_by: int = 0,
        project_slug: Optional[str] = None,
    ) -> str:
        """Add a new scheduled job.

        Args:
            job_name: Human-readable job name.
            cron_expression: Cron-style schedule (e.g. "0 9 * * 1-5").
            prompt: The prompt to send to Claude when the job fires.
            target_chat_ids: Telegram chat IDs to send the response to
                (used only when project_slug is not set, or as a DM fallback).
            working_directory: Working directory for Claude execution.
            skill_name: Optional skill to invoke.
            created_by: Telegram user ID of the creator.
            project_slug: If set, resolves at fire time to the matching
                project_threads row to route delivery into a specific
                forum topic in a supergroup. Overrides target_chat_ids.

        Returns:
            The job ID.
        """
        trigger = CronTrigger.from_crontab(cron_expression)
        work_dir = working_directory or self.default_working_directory

        # Resolve project_slug to (chat_id, thread_id) now so APScheduler
        # has stable kwargs. If the project_threads row changes later, the
        # bot restart (via _load_jobs_from_db) will pick up the new values.
        resolved_chat_id, resolved_thread_id = await self._resolve_project_slug(
            project_slug
        )
        effective_chat_ids = (
            [resolved_chat_id] if resolved_chat_id is not None else (target_chat_ids or [])
        )

        job = self._scheduler.add_job(
            self._fire_event,
            trigger=trigger,
            kwargs={
                "job_name": job_name,
                "prompt": prompt,
                "working_directory": str(work_dir),
                "target_chat_ids": effective_chat_ids,
                "skill_name": skill_name,
                "project_slug": project_slug,
                "message_thread_id": resolved_thread_id,
            },
            name=job_name,
        )

        # Persist to database — store original target_chat_ids (DM fallback)
        # and the project_slug; resolution happens on load/fire.
        await self._save_job(
            job_id=job.id,
            job_name=job_name,
            cron_expression=cron_expression,
            prompt=prompt,
            target_chat_ids=target_chat_ids or [],
            working_directory=str(work_dir),
            skill_name=skill_name,
            created_by=created_by,
            project_slug=project_slug,
        )

        logger.info(
            "Scheduled job added",
            job_id=job.id,
            job_name=job_name,
            cron=cron_expression,
        )
        return str(job.id)

    async def remove_job(self, job_id: str) -> bool:
        """Remove a scheduled job."""
        try:
            self._scheduler.remove_job(job_id)
        except Exception:
            logger.warning("Job not found in scheduler", job_id=job_id)

        await self._delete_job(job_id)
        logger.info("Scheduled job removed", job_id=job_id)
        return True

    async def list_jobs(self) -> List[Dict[str, Any]]:
        """List all scheduled jobs from the database."""
        async with self.db_manager.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM scheduled_jobs WHERE is_active = 1 ORDER BY created_at"
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def _fire_event(
        self,
        job_name: str,
        prompt: str,
        working_directory: str,
        target_chat_ids: List[int],
        skill_name: Optional[str],
        project_slug: Optional[str] = None,
        message_thread_id: Optional[int] = None,
    ) -> None:
        """Called by APScheduler when a job triggers. Publishes a ScheduledEvent."""
        event = ScheduledEvent(
            job_name=job_name,
            prompt=prompt,
            working_directory=Path(working_directory),
            target_chat_ids=target_chat_ids,
            skill_name=skill_name,
            project_slug=project_slug,
            message_thread_id=message_thread_id,
        )

        logger.info(
            "Scheduled job fired",
            job_name=job_name,
            event_id=event.id,
        )

        await self.event_bus.publish(event)

    async def _resolve_project_slug(
        self, project_slug: Optional[str]
    ) -> tuple[Optional[int], Optional[int]]:
        """Look up (chat_id, message_thread_id) for an active project_threads row.

        Returns (None, None) if slug is falsy or no active row exists.
        """
        if not project_slug:
            return (None, None)
        try:
            async with self.db_manager.get_connection() as conn:
                cursor = await conn.execute(
                    """
                    SELECT chat_id, message_thread_id
                    FROM project_threads
                    WHERE project_slug = ? AND is_active = 1
                    LIMIT 1
                    """,
                    (project_slug,),
                )
                row = await cursor.fetchone()
                if row is None:
                    logger.warning(
                        "project_slug has no active project_threads row; "
                        "falling back to target_chat_ids",
                        project_slug=project_slug,
                    )
                    return (None, None)
                return (int(row["chat_id"]), int(row["message_thread_id"]))
        except Exception:
            logger.exception(
                "Failed to resolve project_slug",
                project_slug=project_slug,
            )
            return (None, None)

    async def _load_jobs_from_db(self) -> None:
        """Load persisted jobs and re-register them with APScheduler."""
        try:
            async with self.db_manager.get_connection() as conn:
                cursor = await conn.execute(
                    """
                    SELECT sj.*, pt.chat_id AS pt_chat_id,
                           pt.message_thread_id AS pt_thread_id
                    FROM scheduled_jobs sj
                    LEFT JOIN project_threads pt
                      ON pt.project_slug = sj.project_slug
                     AND pt.is_active = 1
                    WHERE sj.is_active = 1
                    """
                )
                rows = list(await cursor.fetchall())

            for row in rows:
                row_dict = dict(row)
                try:
                    trigger = CronTrigger.from_crontab(row_dict["cron_expression"])

                    # Parse target_chat_ids from stored string (DM fallback)
                    chat_ids_str = row_dict.get("target_chat_ids", "")
                    chat_ids = (
                        [int(x) for x in chat_ids_str.split(",") if x.strip()]
                        if chat_ids_str
                        else []
                    )

                    # If this job is bound to a project_slug and the
                    # project_threads JOIN found a match, override delivery
                    # target with the group chat_id + topic thread id.
                    project_slug = row_dict.get("project_slug")
                    pt_chat_id = row_dict.get("pt_chat_id")
                    pt_thread_id = row_dict.get("pt_thread_id")
                    message_thread_id: Optional[int] = None
                    if project_slug and pt_chat_id is not None:
                        chat_ids = [int(pt_chat_id)]
                        message_thread_id = (
                            int(pt_thread_id) if pt_thread_id is not None else None
                        )
                    elif project_slug and pt_chat_id is None:
                        logger.warning(
                            "scheduled_job has project_slug with no "
                            "project_threads match; using target_chat_ids",
                            job_id=row_dict["job_id"],
                            project_slug=project_slug,
                        )

                    self._scheduler.add_job(
                        self._fire_event,
                        trigger=trigger,
                        kwargs={
                            "job_name": row_dict["job_name"],
                            "prompt": row_dict["prompt"],
                            "working_directory": row_dict["working_directory"],
                            "target_chat_ids": chat_ids,
                            "skill_name": row_dict.get("skill_name"),
                            "project_slug": project_slug,
                            "message_thread_id": message_thread_id,
                        },
                        id=row_dict["job_id"],
                        name=row_dict["job_name"],
                        replace_existing=True,
                    )
                    logger.debug(
                        "Loaded scheduled job from DB",
                        job_id=row_dict["job_id"],
                        job_name=row_dict["job_name"],
                    )
                except Exception:
                    logger.exception(
                        "Failed to load scheduled job",
                        job_id=row_dict.get("job_id"),
                    )

            logger.info("Loaded scheduled jobs from database", count=len(rows))
        except Exception:
            # Table might not exist yet on first run
            logger.debug("No scheduled_jobs table found, starting fresh")

    async def _save_job(
        self,
        job_id: str,
        job_name: str,
        cron_expression: str,
        prompt: str,
        target_chat_ids: List[int],
        working_directory: str,
        skill_name: Optional[str],
        created_by: int,
        project_slug: Optional[str] = None,
    ) -> None:
        """Persist a job definition to the database."""
        chat_ids_str = ",".join(str(cid) for cid in target_chat_ids)
        async with self.db_manager.get_connection() as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO scheduled_jobs
                (job_id, job_name, cron_expression, prompt, target_chat_ids,
                 working_directory, skill_name, created_by, is_active, project_slug)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    job_id,
                    job_name,
                    cron_expression,
                    prompt,
                    chat_ids_str,
                    working_directory,
                    skill_name,
                    created_by,
                    project_slug,
                ),
            )
            await conn.commit()

    async def _delete_job(self, job_id: str) -> None:
        """Soft-delete a job from the database."""
        async with self.db_manager.get_connection() as conn:
            await conn.execute(
                "UPDATE scheduled_jobs SET is_active = 0 WHERE job_id = ?",
                (job_id,),
            )
            await conn.commit()
