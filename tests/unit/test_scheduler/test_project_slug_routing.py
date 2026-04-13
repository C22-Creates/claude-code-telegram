"""Tests for scheduler → project_threads resolution.

Verifies that scheduled jobs bound to a project_slug deliver into the
matching supergroup topic via a JOIN against project_threads at load time.
Jobs without a project_slug still use target_chat_ids as a DM fallback.
"""

from pathlib import Path
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock

import pytest

from src.events.bus import EventBus
from src.events.types import ScheduledEvent
from src.scheduler.scheduler import JobScheduler


class _FakeRow(dict):
    """Row wrapper allowing both dict-style and aiosqlite.Row-style access."""

    def __getitem__(self, key: Any) -> Any:  # type: ignore[override]
        return super().__getitem__(key)


class _FakeCursor:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    async def fetchall(self) -> list[_FakeRow]:
        return [_FakeRow(r) for r in self._rows]

    async def fetchone(self) -> _FakeRow | None:
        return _FakeRow(self._rows[0]) if self._rows else None


class _FakeConnection:
    def __init__(self, rows_for_query: dict[str, list[dict[str, Any]]]) -> None:
        self._rows_for_query = rows_for_query
        self.commits = 0

    async def execute(self, sql: str, params: tuple = ()) -> _FakeCursor:
        # Route based on a substring in the SQL to pick the right row set
        for key, rows in self._rows_for_query.items():
            if key in sql:
                return _FakeCursor(rows)
        return _FakeCursor([])

    async def commit(self) -> None:
        self.commits += 1


class _FakeDBManager:
    def __init__(self, rows_for_query: dict[str, list[dict[str, Any]]]) -> None:
        self._conn = _FakeConnection(rows_for_query)

    def get_connection(self) -> "AsyncIterator[_FakeConnection]":
        db = self

        class _CM:
            async def __aenter__(self_inner) -> _FakeConnection:  # noqa: N805
                return db._conn

            async def __aexit__(self_inner, *_exc: Any) -> None:  # noqa: N805
                return None

        return _CM()  # type: ignore[return-value]


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


class TestProjectSlugResolution:
    """Scheduler resolves project_slug → (chat_id, message_thread_id)."""

    async def test_resolve_returns_thread_ids_when_slug_matches(
        self, event_bus: EventBus
    ) -> None:
        db = _FakeDBManager(
            rows_for_query={
                "FROM project_threads": [
                    {"chat_id": -1003895073557, "message_thread_id": 20}
                ]
            }
        )
        scheduler = JobScheduler(
            event_bus=event_bus,
            db_manager=db,  # type: ignore[arg-type]
            default_working_directory=Path("/tmp"),
        )

        chat_id, thread_id = await scheduler._resolve_project_slug("c22os")

        assert chat_id == -1003895073557
        assert thread_id == 20

    async def test_resolve_returns_none_when_slug_absent(
        self, event_bus: EventBus
    ) -> None:
        db = _FakeDBManager(rows_for_query={"FROM project_threads": []})
        scheduler = JobScheduler(
            event_bus=event_bus,
            db_manager=db,  # type: ignore[arg-type]
            default_working_directory=Path("/tmp"),
        )

        chat_id, thread_id = await scheduler._resolve_project_slug("nonexistent")

        assert chat_id is None
        assert thread_id is None

    async def test_resolve_returns_none_for_falsy_slug(
        self, event_bus: EventBus
    ) -> None:
        db = _FakeDBManager(rows_for_query={})
        scheduler = JobScheduler(
            event_bus=event_bus,
            db_manager=db,  # type: ignore[arg-type]
            default_working_directory=Path("/tmp"),
        )

        assert await scheduler._resolve_project_slug(None) == (None, None)
        assert await scheduler._resolve_project_slug("") == (None, None)


class TestFireEventPropagation:
    """_fire_event builds a ScheduledEvent carrying thread context."""

    async def test_fire_event_publishes_thread_id(
        self, event_bus: EventBus
    ) -> None:
        db = _FakeDBManager(rows_for_query={})
        scheduler = JobScheduler(
            event_bus=event_bus,
            db_manager=db,  # type: ignore[arg-type]
            default_working_directory=Path("/tmp"),
        )

        captured: list[ScheduledEvent] = []

        async def _capture(event: ScheduledEvent) -> None:
            if isinstance(event, ScheduledEvent):
                captured.append(event)

        event_bus.subscribe(ScheduledEvent, _capture)
        # Intercept publish so we don't depend on the background processor
        original_publish = event_bus.publish

        async def _sync_publish(event: Any) -> None:
            await original_publish(event)
            await event_bus._dispatch(event)

        event_bus.publish = _sync_publish  # type: ignore[method-assign]

        await scheduler._fire_event(
            job_name="daily-digest",
            prompt="",
            working_directory="/tmp",
            target_chat_ids=[-1003895073557],
            skill_name="daily-digest",
            project_slug="c22os",
            message_thread_id=20,
        )

        assert len(captured) == 1
        assert captured[0].project_slug == "c22os"
        assert captured[0].message_thread_id == 20
        assert captured[0].target_chat_ids == [-1003895073557]

    async def test_fire_event_dm_fallback_has_no_thread(
        self, event_bus: EventBus
    ) -> None:
        """Jobs without a project_slug still work; thread id stays None."""
        db = _FakeDBManager(rows_for_query={})
        scheduler = JobScheduler(
            event_bus=event_bus,
            db_manager=db,  # type: ignore[arg-type]
            default_working_directory=Path("/tmp"),
        )

        captured: list[ScheduledEvent] = []

        async def _capture(event: ScheduledEvent) -> None:
            if isinstance(event, ScheduledEvent):
                captured.append(event)

        event_bus.subscribe(ScheduledEvent, _capture)
        # Intercept publish so we don't depend on the background processor
        original_publish = event_bus.publish

        async def _sync_publish(event: Any) -> None:
            await original_publish(event)
            await event_bus._dispatch(event)

        event_bus.publish = _sync_publish  # type: ignore[method-assign]

        await scheduler._fire_event(
            job_name="dm-job",
            prompt="ping",
            working_directory="/tmp",
            target_chat_ids=[7842005042],
            skill_name=None,
        )

        assert len(captured) == 1
        assert captured[0].project_slug is None
        assert captured[0].message_thread_id is None
        assert captured[0].target_chat_ids == [7842005042]
