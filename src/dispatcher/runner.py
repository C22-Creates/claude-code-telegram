"""Production task runner: executes a board task as a Claude session.

Builds a prompt from the task, runs it via the existing ClaudeIntegration
(same path scheduled jobs and Telegram messages already use), and reports the
result. The worker is told how to self-block via the Hermes CLI when it needs
a human — the dispatcher reconciles against live board state afterward.
"""

import json
from pathlib import Path
from typing import Any, Dict

import structlog

from ..claude.facade import ClaudeIntegration
from .service import RunOutcome

logger = structlog.get_logger()


class ClaudeTaskRunner:
    """Runs a board task through Claude Code."""

    def __init__(
        self,
        claude_integration: ClaudeIntegration,
        default_working_directory: Path,
        board_db_path: Path,
        hermes_cli_path: Path,
        default_user_id: int = 0,
    ) -> None:
        self._claude = claude_integration
        self._default_working_directory = default_working_directory
        self._board_db_path = board_db_path
        self._hermes_cli_path = hermes_cli_path
        self._default_user_id = default_user_id

    async def run(self, task: Dict[str, Any]) -> RunOutcome:
        prompt = self._build_prompt(task)
        working_dir = self._working_directory(task)

        response = await self._claude.run_command(
            prompt=prompt,
            working_directory=working_dir,
            user_id=self._default_user_id,
            force_new=True,  # each task is an independent unit of work
        )

        if getattr(response, "is_error", False):
            raise RuntimeError(response.error_type or "Claude returned an error")

        return RunOutcome(content=response.content or "")

    def _working_directory(self, task: Dict[str, Any]) -> Path:
        payload = task.get("payload")
        if isinstance(payload, dict) and payload.get("_working_directory"):
            return Path(payload["_working_directory"])
        return self._default_working_directory

    def _build_prompt(self, task: Dict[str, Any]) -> str:
        task_id = task["id"]
        payload = task.get("payload")
        payload_str = (
            json.dumps(payload, indent=2) if payload is not None else "(none)"
        )
        assignee = task.get("assignee") or "the appropriate specialist"
        block_cmd = (
            f'python3 {self._hermes_cli_path} --db {self._board_db_path} '
            f'block {task_id} --reason "<your specific question>"'
        )
        return (
            f"You are working Hermes task `{task_id}`, acting as {assignee}.\n\n"
            f"## Task\n{task['title']}\n\n"
            f"## Payload\n{payload_str}\n\n"
            f"## How to work it\n"
            f"Complete the task. The dispatcher records completion automatically "
            f"once you finish — your final message is delivered to the originating "
            f"chat.\n\n"
            f"If you genuinely need Carl or Sisi to decide something before you can "
            f"continue, do NOT guess. Run this exact command, then stop:\n\n"
            f"    {block_cmd}\n\n"
            f"Any human answers will appear in the payload under `_resume_notes` "
            f"when the task is resumed."
        )
