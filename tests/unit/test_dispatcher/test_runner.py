"""Tests for the Hermes task runner's model-routing logic.

Run:  .venv/bin/python -m unittest tests.unit.test_dispatcher.test_runner -v
"""

import unittest
from pathlib import Path
from typing import Any, Dict, Optional

from src.dispatcher.runner import (
    FAST_MODEL,
    JUDGMENT_MODEL,
    ClaudeTaskRunner,
    resolve_model,
)


RECAP_DOC = "workflows/hermes-recap-runner.md"
LEAD_DOC = "workflows/hermes-lead-runner.md"


class ResolveModelTest(unittest.TestCase):
    def test_no_payload_returns_none(self):
        self.assertIsNone(resolve_model({"id": "t1"}))

    def test_payload_without_role_or_model_returns_none(self):
        self.assertIsNone(resolve_model({"id": "t1", "payload": {"sessionId": "x"}}))

    def test_recap_sweep_gets_fast_model(self):
        task = {"id": "t1", "payload": {"role": "sweep", "runner_doc": RECAP_DOC}}
        self.assertEqual(resolve_model(task), FAST_MODEL)

    def test_recap_role_gets_judgment_model(self):
        task = {"id": "t1", "payload": {"role": "recap", "runner_doc": RECAP_DOC}}
        self.assertEqual(resolve_model(task), JUDGMENT_MODEL)

    def test_lead_digest_gets_judgment_model(self):
        task = {"id": "t1", "payload": {"role": "digest", "runner_doc": LEAD_DOC}}
        self.assertEqual(resolve_model(task), JUDGMENT_MODEL)

    def test_lead_write_gets_judgment_model(self):
        task = {"id": "t1", "payload": {"role": "write", "runner_doc": LEAD_DOC}}
        self.assertEqual(resolve_model(task), JUDGMENT_MODEL)

    def test_same_role_different_runner_is_not_shared(self):
        # "sweep" means routine filtering in the recap runner but includes
        # drafting in the pipeline runner -- the latter must NOT inherit the
        # recap sweep's fast model.
        task = {
            "id": "t1",
            "payload": {
                "role": "sweep",
                "runner_doc": "workflows/hermes-pipeline-runner.md",
            },
        }
        self.assertIsNone(resolve_model(task))

    def test_role_without_runner_doc_gets_no_override(self):
        task = {"id": "t1", "payload": {"role": "sweep"}}
        self.assertIsNone(resolve_model(task))

    def test_unknown_role_falls_back_to_none(self):
        task = {
            "id": "t1",
            "payload": {"role": "some-future-role", "runner_doc": RECAP_DOC},
        }
        self.assertIsNone(resolve_model(task))

    def test_explicit_payload_model_wins_over_role(self):
        task = {
            "id": "t1",
            "payload": {
                "role": "sweep",
                "runner_doc": RECAP_DOC,
                "model": "claude-opus-4-8",
            },
        }
        self.assertEqual(resolve_model(task), "claude-opus-4-8")

    def test_non_dict_payload_returns_none(self):
        self.assertIsNone(resolve_model({"id": "t1", "payload": "not a dict"}))
        self.assertIsNone(resolve_model({"id": "t1", "payload": None}))


class FakeClaudeIntegration:
    """Records the kwargs run_command was called with."""

    def __init__(self):
        self.calls = []

    async def run_command(self, **kwargs):
        self.calls.append(kwargs)

        class _Response:
            is_error = False
            error_type = None
            content = "ok"

        return _Response()


class ClaudeTaskRunnerModelPassthroughTest(unittest.IsolatedAsyncioTestCase):
    async def test_resolved_model_is_passed_to_run_command(self):
        claude = FakeClaudeIntegration()
        runner = ClaudeTaskRunner(
            claude_integration=claude,
            default_working_directory=Path("/tmp"),
            board_db_path=Path("/tmp/board.db"),
            hermes_cli_path=Path("/tmp/cli.py"),
        )
        task = {
            "id": "t1",
            "title": "Recap: Some Meeting",
            "assignee": "meeting-processor",
            "payload": {"role": "recap", "runner_doc": RECAP_DOC, "sessionId": "abc"},
        }

        await runner.run(task)

        self.assertEqual(len(claude.calls), 1)
        self.assertEqual(claude.calls[0]["model"], JUDGMENT_MODEL)

    async def test_no_role_leaves_model_none_unchanged_behavior(self):
        claude = FakeClaudeIntegration()
        runner = ClaudeTaskRunner(
            claude_integration=claude,
            default_working_directory=Path("/tmp"),
            board_db_path=Path("/tmp/board.db"),
            hermes_cli_path=Path("/tmp/cli.py"),
        )
        task = {"id": "t1", "title": "Ad-hoc task", "payload": None}

        await runner.run(task)

        self.assertIsNone(claude.calls[0]["model"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
