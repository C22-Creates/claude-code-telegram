"""Session scope isolation and conversation locking.

Regression coverage for the 2026-07-29 incident: background Hermes tasks were
evicting live Telegram conversations, a resume timeout discarded the whole
session, and concurrent messages in one topic raced two SDK subprocesses
against the same --resume id.
"""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.claude.exceptions import ClaudeTimeoutError
from src.claude.facade import ClaudeIntegration
from src.claude.sdk_integration import ClaudeResponse
from src.claude.session import ClaudeSession, SessionManager
from src.config.settings import Settings

from .conftest import InMemorySessionStorage

PROJECT = Path("/home/c22bot/Projects/c22os")
CHAT_SCOPE = "chat:-1003895073557:267"
OTHER_SCOPE = "chat:-1003895073557:53"


def _settings(**overrides) -> Settings:
    base = dict(
        telegram_bot_token="test-token",
        telegram_bot_username="test_bot",
        approved_directory=PROJECT,
        max_sessions_per_user=2,
        session_timeout_hours=24,
    )
    base.update(overrides)
    return Settings(**base)


def _session(session_id: str, scope_key, user_id: int = 7842005042) -> ClaudeSession:
    now = datetime.now(UTC)
    return ClaudeSession(
        session_id=session_id,
        user_id=user_id,
        project_path=PROJECT,
        created_at=now,
        last_used=now,
        scope_key=scope_key,
    )


def _response(session_id: str = "new-session") -> ClaudeResponse:
    return ClaudeResponse(
        content="ok",
        session_id=session_id,
        cost=0.01,
        duration_ms=10,
        num_turns=1,
    )


@pytest.fixture
def manager() -> SessionManager:
    return SessionManager(_settings(), InMemorySessionStorage())


class TestScopedEviction:
    """A scope may only evict its own sessions."""

    @pytest.mark.asyncio
    async def test_background_work_does_not_evict_a_conversation(self, manager):
        # Two live chat sessions — the pool limit for this user is 2.
        for i, sid in enumerate(["chat-a", "chat-b"]):
            await manager.storage.save_session(_session(sid, CHAT_SCOPE))

        # A Hermes task claims a slot. Before the fix this evicted "chat-a".
        await manager.get_or_create_session(7842005042, PROJECT, scope_key="hermes")

        surviving = {
            s.session_id for s in await manager.storage.get_user_sessions(7842005042)
        }
        assert surviving == {"chat-a", "chat-b"}

    @pytest.mark.asyncio
    async def test_scope_still_evicts_its_own_oldest(self, manager):
        old = _session("chat-old", CHAT_SCOPE)
        old.last_used = datetime(2020, 1, 1, tzinfo=UTC)
        await manager.storage.save_session(old)
        await manager.storage.save_session(_session("chat-new", CHAT_SCOPE))

        await manager.get_or_create_session(7842005042, PROJECT, scope_key=CHAT_SCOPE)

        surviving = {
            s.session_id for s in await manager.storage.get_user_sessions(7842005042)
        }
        assert "chat-old" not in surviving
        assert "chat-new" in surviving


class TestScopedResume:
    """Auto-resume never crosses a scope boundary."""

    @pytest.mark.asyncio
    async def test_resume_matches_own_scope(self, manager):
        await manager.storage.save_session(_session("journal", CHAT_SCOPE))
        integration = ClaudeIntegration(
            config=_settings(), sdk_manager=AsyncMock(), session_manager=manager
        )

        found = await integration._find_resumable_session(
            7842005042, PROJECT, CHAT_SCOPE
        )
        assert found is not None
        assert found.session_id == "journal"

    @pytest.mark.asyncio
    async def test_other_topic_is_not_resumed(self, manager):
        await manager.storage.save_session(_session("journal", CHAT_SCOPE))
        integration = ClaudeIntegration(
            config=_settings(), sdk_manager=AsyncMock(), session_manager=manager
        )

        assert (
            await integration._find_resumable_session(7842005042, PROJECT, OTHER_SCOPE)
            is None
        )

    @pytest.mark.asyncio
    async def test_background_session_is_not_resumed_by_chat(self, manager):
        await manager.storage.save_session(_session("hermes-task", "hermes"))
        integration = ClaudeIntegration(
            config=_settings(), sdk_manager=AsyncMock(), session_manager=manager
        )

        assert (
            await integration._find_resumable_session(7842005042, PROJECT, CHAT_SCOPE)
            is None
        )


class TestTimeoutPreservesSession:
    """A slow turn must not cost the user their conversation."""

    @pytest.mark.asyncio
    async def test_timeout_does_not_remove_session(self, manager):
        await manager.storage.save_session(_session("live-chat", CHAT_SCOPE))

        sdk = AsyncMock()
        sdk.execute_command = AsyncMock(side_effect=ClaudeTimeoutError("timed out"))
        integration = ClaudeIntegration(
            config=_settings(), sdk_manager=sdk, session_manager=manager
        )

        with pytest.raises(ClaudeTimeoutError):
            await integration.run_command(
                prompt="hello",
                working_directory=PROJECT,
                user_id=7842005042,
                scope_key=CHAT_SCOPE,
            )

        # Session survives, and only one attempt was made — no silent restart.
        remaining = await manager.storage.get_user_sessions(7842005042)
        assert [s.session_id for s in remaining] == ["live-chat"]
        assert sdk.execute_command.await_count == 1

    @pytest.mark.asyncio
    async def test_non_timeout_resume_failure_still_starts_fresh(self, manager):
        await manager.storage.save_session(_session("stale-chat", CHAT_SCOPE))

        sdk = AsyncMock()
        sdk.execute_command = AsyncMock(
            side_effect=[RuntimeError("session gone"), _response("fresh")]
        )
        integration = ClaudeIntegration(
            config=_settings(), sdk_manager=sdk, session_manager=manager
        )

        response = await integration.run_command(
            prompt="hello",
            working_directory=PROJECT,
            user_id=7842005042,
            scope_key=CHAT_SCOPE,
        )

        assert response.session_id == "fresh"
        assert sdk.execute_command.await_count == 2


class TestConversationLock:
    """Two turns in one conversation serialize instead of racing."""

    @pytest.mark.asyncio
    async def test_same_scope_serializes(self, manager):
        concurrent = 0
        peak = 0

        async def slow_execute(**_kwargs):
            nonlocal concurrent, peak
            concurrent += 1
            peak = max(peak, concurrent)
            await asyncio.sleep(0.05)
            concurrent -= 1
            return _response()

        sdk = AsyncMock()
        sdk.execute_command = AsyncMock(side_effect=slow_execute)
        integration = ClaudeIntegration(
            config=_settings(), sdk_manager=sdk, session_manager=manager
        )

        await asyncio.gather(
            *[
                integration.run_command(
                    prompt=f"message {i}",
                    working_directory=PROJECT,
                    user_id=7842005042,
                    scope_key=CHAT_SCOPE,
                )
                for i in range(3)
            ]
        )

        assert peak == 1, "messages in one topic must not run concurrently"

    @pytest.mark.asyncio
    async def test_different_scopes_run_concurrently(self, manager):
        started = asyncio.Event()
        release = asyncio.Event()

        async def execute(**kwargs):
            if kwargs.get("prompt") == "slow":
                started.set()
                await release.wait()
            return _response()

        sdk = AsyncMock()
        sdk.execute_command = AsyncMock(side_effect=execute)
        integration = ClaudeIntegration(
            config=_settings(), sdk_manager=sdk, session_manager=manager
        )

        first = asyncio.create_task(
            integration.run_command(
                prompt="slow",
                working_directory=PROJECT,
                user_id=7842005042,
                scope_key=CHAT_SCOPE,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)

        # A different topic must not be blocked behind the slow one.
        await asyncio.wait_for(
            integration.run_command(
                prompt="fast",
                working_directory=PROJECT,
                user_id=7842005042,
                scope_key=OTHER_SCOPE,
            ),
            timeout=1,
        )

        release.set()
        await first
