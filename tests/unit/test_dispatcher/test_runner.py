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
from src.harness.run_slots import RunSlots
from src.harness.types import Outcome, RunResult

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


class FakeRunJob:
    """Records the args run_job was called with."""

    def __init__(self, outcome=Outcome.DONE, text="ok", reason=""):
        self.calls = []
        self._result = RunResult(
            outcome=outcome, text=text, elapsed_s=3.0, reason=reason
        )

    async def __call__(self, *args, **kw):
        self.calls.append(args)
        return self._result


def make_runner(job):
    return ClaudeTaskRunner(
        default_working_directory=Path("/tmp"),
        board_db_path=Path("/tmp/board.db"),
        hermes_cli_path=Path("/tmp/cli.py"),
        slots=RunSlots(3),
        settings=object(),
        security_validator=object(),
        run_job_fn=job,
    )


class ClaudeTaskRunnerModelPassthroughTest(unittest.IsolatedAsyncioTestCase):
    async def test_resolved_model_is_passed_to_run_job(self):
        job = FakeRunJob()
        task = {
            "id": "t1",
            "title": "Recap: Some Meeting",
            "assignee": "meeting-processor",
            "payload": {"role": "recap", "runner_doc": RECAP_DOC, "sessionId": "abc"},
        }

        outcome = await make_runner(job).run(task)

        self.assertEqual(len(job.calls), 1)
        prompt, cwd, agent, _slots, _settings, model = job.calls[0]
        self.assertEqual(model, JUDGMENT_MODEL)
        self.assertEqual(agent, "meeting-processor")
        self.assertIn("Hermes task `t1`", prompt)
        self.assertEqual(outcome.content, "ok")

    async def test_no_role_leaves_model_none_unchanged_behavior(self):
        job = FakeRunJob()
        await make_runner(job).run({"id": "t1", "title": "Ad-hoc", "payload": None})
        self.assertIsNone(job.calls[0][5])

    async def test_error_raises_like_an_error_response_did(self):
        job = FakeRunJob(Outcome.ERROR, "", "boom")
        with self.assertRaises(RuntimeError):
            await make_runner(job).run({"id": "t1", "title": "x", "payload": None})

    async def test_stopped_task_fails_rather_than_recording_done(self):
        job = FakeRunJob(Outcome.STOPPED, "partial", "hit 60m cap")
        with self.assertRaisesRegex(RuntimeError, "hit 60m cap"):
            await make_runner(job).run({"id": "t1", "title": "x", "payload": None})


if __name__ == "__main__":
    unittest.main(verbosity=2)
