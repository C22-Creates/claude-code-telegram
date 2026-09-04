"""High-level Claude Code integration facade.

Provides simple interface for bot handlers.
"""

import asyncio
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import structlog

from ..config.settings import Settings
from .exceptions import ClaudeTimeoutError
from .sdk_integration import ClaudeResponse, ClaudeSDKManager, StreamUpdate
from .session import SessionManager

logger = structlog.get_logger()


class ClaudeIntegration:
    """Main integration point for Claude Code."""

    def __init__(
        self,
        config: Settings,
        sdk_manager: Optional[ClaudeSDKManager] = None,
        session_manager: Optional[SessionManager] = None,
    ):
        """Initialize Claude integration facade."""
        self.config = config
        self.sdk_manager = sdk_manager or ClaudeSDKManager(config)
        self.session_manager = session_manager
        # One lock per conversation. Two messages arriving in the same topic
        # while the first is still running used to spawn two SDK subprocesses
        # against the same `--resume <id>`; one of them would wedge with no
        # watchdog. Queueing the second message keeps the transcript linear.
        self._conversation_locks: Dict[tuple, asyncio.Lock] = {}

    def _conversation_lock(
        self, user_id: int, working_directory: Path, scope_key: Optional[str]
    ) -> asyncio.Lock:
        """Get (or create) the lock serializing one conversation's turns."""
        key = (user_id, str(working_directory), scope_key)
        lock = self._conversation_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._conversation_locks[key] = lock
        return lock

    async def run_command(
        self,
        prompt: str,
        working_directory: Path,
        user_id: int,
        session_id: Optional[str] = None,
        on_stream: Optional[Callable[[StreamUpdate], None]] = None,
        force_new: bool = False,
        interrupt_event: Optional["asyncio.Event"] = None,
        images: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        scope_key: Optional[str] = None,
        extra_disallowed_tools: Optional[List[str]] = None,
        topic_agent: Optional[str] = None,
    ) -> ClaudeResponse:
        """Run Claude Code command with full integration."""
        async with self._conversation_lock(user_id, working_directory, scope_key):
            return await self._run_command_locked(
                prompt=prompt,
                working_directory=working_directory,
                user_id=user_id,
                session_id=session_id,
                on_stream=on_stream,
                force_new=force_new,
                interrupt_event=interrupt_event,
                images=images,
                model=model,
                scope_key=scope_key,
                extra_disallowed_tools=extra_disallowed_tools,
                topic_agent=topic_agent,
            )

    async def _run_command_locked(
        self,
        prompt: str,
        working_directory: Path,
        user_id: int,
        session_id: Optional[str] = None,
        on_stream: Optional[Callable[[StreamUpdate], None]] = None,
        force_new: bool = False,
        interrupt_event: Optional["asyncio.Event"] = None,
        images: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        scope_key: Optional[str] = None,
        extra_disallowed_tools: Optional[List[str]] = None,
        topic_agent: Optional[str] = None,
    ) -> ClaudeResponse:
        """Run one command. Caller holds the conversation lock."""
        logger.info(
            "Running Claude command",
            user_id=user_id,
            working_directory=str(working_directory),
            session_id=session_id,
            prompt_length=len(prompt),
            force_new=force_new,
            scope_key=scope_key,
        )

        # If no session_id provided, try to find an existing session for this
        # user+directory+scope combination (auto-resume).
        # Skip auto-resume when force_new is set (e.g. after /new command).
        if not session_id and not force_new:
            existing_session = await self._find_resumable_session(
                user_id, working_directory, scope_key
            )
            if existing_session:
                session_id = existing_session.session_id
                logger.info(
                    "Auto-resuming existing session for project",
                    session_id=session_id,
                    project_path=str(working_directory),
                    user_id=user_id,
                    scope_key=scope_key,
                )

        # Get or create session
        session = await self.session_manager.get_or_create_session(
            user_id, working_directory, session_id, scope_key=scope_key
        )

        # Execute command
        try:
            # Continue session if we have an existing session with a real ID
            is_new = getattr(session, "is_new_session", False)
            should_continue = not is_new and bool(session.session_id)

            # For new sessions, don't pass session_id to Claude Code
            claude_session_id = session.session_id if should_continue else None

            try:
                response = await self._execute(
                    prompt=prompt,
                    working_directory=working_directory,
                    session_id=claude_session_id,
                    continue_session=should_continue,
                    stream_callback=on_stream,
                    interrupt_event=interrupt_event,
                    images=images,
                    model=model,
                    extra_disallowed_tools=extra_disallowed_tools,
                    topic_agent=topic_agent,
                )
            except ClaudeTimeoutError:
                # A timeout says the turn was too slow — NOT that the session is
                # gone. Discarding it here cost the user their whole thread and
                # started them over cold on the next message. Keep the session
                # intact so the conversation survives a slow turn.
                logger.error(
                    "Claude command timed out; session preserved for retry",
                    session_id=session.session_id,
                    user_id=user_id,
                    scope_key=scope_key,
                )
                raise
            except Exception as resume_error:
                # If resume failed (e.g., session expired/missing on Claude's side),
                # retry as a fresh session.  The CLI returns a generic exit-code-1
                # when the session is gone, so we catch *any* error during resume.
                if should_continue:
                    logger.warning(
                        "Session resume failed, starting fresh session",
                        failed_session_id=claude_session_id,
                        error=str(resume_error),
                    )
                    # Clean up the stale session
                    await self.session_manager.remove_session(session.session_id)

                    # Create a fresh session and retry
                    session = await self.session_manager.get_or_create_session(
                        user_id, working_directory, scope_key=scope_key
                    )
                    response = await self._execute(
                        prompt=prompt,
                        working_directory=working_directory,
                        extra_disallowed_tools=extra_disallowed_tools,
                        topic_agent=topic_agent,
                        session_id=None,
                        continue_session=False,
                        stream_callback=on_stream,
                        interrupt_event=interrupt_event,
                        images=images,
                        model=model,
                    )
                else:
                    raise

            # Update session (assigns real session_id for new sessions)
            await self.session_manager.update_session(session, response)

            # Ensure response has the session's final ID
            response.session_id = session.session_id

            if not response.session_id:
                logger.warning(
                    "No session_id after execution; session cannot be resumed",
                    user_id=user_id,
                )

            logger.info(
                "Claude command completed",
                session_id=response.session_id,
                cost=response.cost,
                duration_ms=response.duration_ms,
                num_turns=response.num_turns,
                is_error=response.is_error,
            )

            return response

        except Exception as e:
            logger.error(
                "Claude command failed",
                error=str(e),
                user_id=user_id,
                session_id=session.session_id,
            )
            raise

    async def _execute(
        self,
        prompt: str,
        working_directory: Path,
        session_id: Optional[str] = None,
        continue_session: bool = False,
        stream_callback: Optional[Callable] = None,
        interrupt_event: Optional[asyncio.Event] = None,
        images: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        extra_disallowed_tools: Optional[List[str]] = None,
        topic_agent: Optional[str] = None,
    ) -> ClaudeResponse:
        """Execute command via SDK."""
        return await self.sdk_manager.execute_command(
            prompt=prompt,
            working_directory=working_directory,
            session_id=session_id,
            continue_session=continue_session,
            stream_callback=stream_callback,
            interrupt_event=interrupt_event,
            images=images,
            model=model,
            extra_disallowed_tools=extra_disallowed_tools,
            topic_agent=topic_agent,
        )

    async def _find_resumable_session(
        self,
        user_id: int,
        working_directory: Path,
        scope_key: Optional[str] = None,
    ) -> Optional["ClaudeSession"]:  # noqa: F821
        """Find the most recent resumable session for a user in a directory.

        Returns the session if one exists that is non-expired, belongs to the
        same scope, and has a real (non-temporary) session ID from Claude.
        Returns None otherwise.

        The scope match is what stops one Telegram topic from resuming another
        topic's conversation — or a background job's.
        """

        sessions = await self.session_manager._get_user_sessions(user_id)

        matching_sessions = [
            s
            for s in sessions
            if s.project_path == working_directory
            and s.scope_key == scope_key
            and bool(s.session_id)
            and not s.is_expired(self.config.session_timeout_hours)
        ]

        if not matching_sessions:
            return None

        return max(matching_sessions, key=lambda s: s.last_used)

    async def continue_session(
        self,
        user_id: int,
        working_directory: Path,
        prompt: Optional[str] = None,
        on_stream: Optional[Callable[[StreamUpdate], None]] = None,
        scope_key: Optional[str] = None,
    ) -> Optional[ClaudeResponse]:
        """Continue the most recent session."""
        logger.info(
            "Continuing session",
            user_id=user_id,
            working_directory=str(working_directory),
            has_prompt=bool(prompt),
            scope_key=scope_key,
        )

        # Get user's sessions
        sessions = await self.session_manager._get_user_sessions(user_id)

        # Find most recent session in this directory and scope (exclude
        # sessions without IDs)
        matching_sessions = [
            s
            for s in sessions
            if s.project_path == working_directory
            and s.scope_key == scope_key
            and bool(s.session_id)
        ]

        if not matching_sessions:
            logger.info("No matching sessions found", user_id=user_id)
            return None

        # Get most recent
        latest_session = max(matching_sessions, key=lambda s: s.last_used)

        # Continue session with default prompt if none provided
        # Claude CLI requires a prompt, so we use a placeholder
        return await self.run_command(
            prompt=prompt or "Please continue where we left off",
            working_directory=working_directory,
            user_id=user_id,
            session_id=latest_session.session_id,
            on_stream=on_stream,
            scope_key=scope_key,
        )

    async def get_session_info(
        self, session_id: str, user_id: int
    ) -> Optional[Dict[str, Any]]:
        """Get session information (scoped to requesting user)."""
        return await self.session_manager.get_session_info(session_id, user_id)

    async def get_user_sessions(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all sessions for a user."""
        sessions = await self.session_manager._get_user_sessions(user_id)
        return [
            {
                "session_id": s.session_id,
                "project_path": str(s.project_path),
                "created_at": s.created_at.isoformat(),
                "last_used": s.last_used.isoformat(),
                "total_cost": s.total_cost,
                "message_count": s.message_count,
                "tools_used": s.tools_used,
                "expired": s.is_expired(self.config.session_timeout_hours),
            }
            for s in sessions
        ]

    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        return await self.session_manager.cleanup_expired_sessions()

    async def get_user_summary(self, user_id: int) -> Dict[str, Any]:
        """Get comprehensive user summary."""
        session_summary = await self.session_manager.get_user_session_summary(user_id)

        return {
            "user_id": user_id,
            **session_summary,
        }

    async def shutdown(self) -> None:
        """Shutdown integration and cleanup resources."""
        logger.info("Shutting down Claude integration")

        await self.cleanup_expired_sessions()

        logger.info("Claude integration shutdown complete")
