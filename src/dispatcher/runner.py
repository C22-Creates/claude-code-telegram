"""Production task runner: executes a board task as a Claude session.

Builds a prompt from the task, runs it via the harness (``run_job``: fresh
session, ``scheduled`` slot priority), and reports the result. The worker is
told how to self-block via the Hermes CLI when it needs a human -- the
dispatcher reconciles against live board state afterward.
"""

import json
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

import structlog

from ..harness.jobs import run_job
from ..harness.run_slots import RunSlots
from ..harness.types import Outcome, RunResult
from ..harness.util import footer
from .service import RunOutcome

logger = structlog.get_logger()

# Model routing for Hermes tasks, keyed by (runner doc, role). Role names are
# NOT unique across runners -- "sweep" is routine ledger-filtering in the recap
# runner but includes drafting in the pipeline runner -- so a bare-role table
# would misroute. Every dispatched task payload carries both `runner_doc` and
# `role` (verified against live board data 2026-07-02); tasks missing either
# get no override. payload.model, if set, always wins over this table.
FAST_MODEL = "claude-haiku-4-5-20251001"
JUDGMENT_MODEL = "claude-fable-5"

RUNNER_ROLE_MODELS: Dict[tuple, str] = {
    # recap loop: discovery/filtering is routine; per-meeting synthesis,
    # area classification, and drafting are judgment.
    ("hermes-recap-runner.md", "sweep"): FAST_MODEL,
    ("hermes-recap-runner.md", "recap"): JUDGMENT_MODEL,
    # lead loop: ICP classification (Strong/Possible/Referral/Missing) IS the
    # loop's core judgment, and a misfiled lead is expensive -- both roles get
    # the judgment model. Revisit "write" (Phase 2, gated CRM writes) for a
    # downshift only after it has run trusted for a while.
    ("hermes-lead-runner.md", "digest"): JUDGMENT_MODEL,
    ("hermes-lead-runner.md", "write"): JUDGMENT_MODEL,
    # hermes-inbox-runner.md and hermes-pipeline-runner.md are intentionally
    # unmapped: their "sweep" roles draft content inline, so they keep the
    # configured global model until each is profiled separately.
}


def resolve_model(task: Dict[str, Any]) -> Optional[str]:
    """Pick a model override for a task, or None to use the SDK's configured default.

    Priority: explicit payload.model > (basename(payload.runner_doc),
    payload.role) lookup in RUNNER_ROLE_MODELS > None (no override; caller's
    default model config applies unchanged -- this keeps every task type this
    routing table doesn't know about behaving exactly as it did before this
    feature).
    """
    payload = task.get("payload")
    if not isinstance(payload, dict):
        return None
    explicit = payload.get("model")
    if explicit:
        return str(explicit)
    runner_doc = payload.get("runner_doc")
    role = payload.get("role")
    if not runner_doc or not role:
        return None
    key = (Path(str(runner_doc)).name, str(role))
    return RUNNER_ROLE_MODELS.get(key)


class ClaudeTaskRunner:
    """Runs a board task through Claude Code."""

    def __init__(
        self,
        default_working_directory: Path,
        board_db_path: Path,
        hermes_cli_path: Path,
        *,
        slots: RunSlots,
        settings: Any,
        security_validator: Any,
        run_job_fn: Callable[..., Awaitable[RunResult]] = run_job,
    ) -> None:
        self._default_working_directory = default_working_directory
        self._board_db_path = board_db_path
        self._hermes_cli_path = hermes_cli_path
        self._slots = slots
        self._validator = security_validator
        self._settings = settings
        self._run_job = run_job_fn

    async def run(self, task: Dict[str, Any]) -> RunOutcome:
        working_dir = self._working_directory(task)
        # The assignee's identity and tool denials are applied by the harness
        # (system prompt + disallowed tools), so the prompt carries no identity.
        prompt = self._build_prompt(task)
        model = resolve_model(task)
        if model:
            logger.info("Model override for task", task_id=task["id"], model=model)

        # Each task is an independent unit of work: run_job is always a fresh
        # session. Task state is durable in board.db, not in the session.
        result = await self._run_job(
            prompt,
            working_dir,
            task.get("assignee"),
            self._slots,
            self._settings,
            model,
            security_validator=self._validator,
        )

        if result.outcome is Outcome.DONE:
            return RunOutcome(content=result.text)
        # ERROR was a raise before; a watchdog/cap STOPPED means the task did
        # not finish, so it fails the same way rather than recording "done".
        raise RuntimeError(footer(result))

    def _working_directory(self, task: Dict[str, Any]) -> Path:
        payload = task.get("payload")
        if isinstance(payload, dict) and payload.get("_working_directory"):
            return Path(payload["_working_directory"])
        return self._default_working_directory

    def _build_prompt(self, task: Dict[str, Any]) -> str:
        task_id = task["id"]
        payload = task.get("payload")
        payload_str = json.dumps(payload, indent=2) if payload is not None else "(none)"
        assignee = task.get("assignee") or "the appropriate specialist"
        block_cmd = (
            f"python3 {self._hermes_cli_path} --db {self._board_db_path} "
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
