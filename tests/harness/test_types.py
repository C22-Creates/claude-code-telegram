"""Tests for harness settings, types and formatting helpers."""

from src.config.settings import Settings
from src.harness.types import Outcome, RunResult
from src.harness.util import fmt_elapsed, footer


def test_fmt_elapsed():
    assert fmt_elapsed(42) == "42s"
    assert fmt_elapsed(252) == "4m12s"
    assert fmt_elapsed(3600) == "60m0s"
    assert fmt_elapsed(0) == "0s"


def _r(outcome, reason="", elapsed=252):
    return RunResult(outcome=outcome, text="", elapsed_s=elapsed, reason=reason)


def test_footer_done():
    assert footer(_r(Outcome.DONE)) == "✅ done · 4m12s"


def test_footer_stopped():
    assert footer(_r(Outcome.STOPPED, "stalled", 42)) == "⚠️ stopped (stalled) · 42s"


def test_footer_error_uses_first_line_only():
    assert footer(_r(Outcome.ERROR, "boom\ntrace")) == "❌ error: boom"


def test_footer_error_empty_reason():
    assert footer(_r(Outcome.ERROR, "")) == "❌ error: "


def test_harness_settings_defaults(tmp_path):
    s = Settings(
        telegram_bot_token="t",
        telegram_bot_username="b",
        approved_directory=str(tmp_path),
    )
    assert s.harness_stall_seconds == 300
    assert s.harness_heartbeat_minutes == [20, 40]
    assert s.harness_run_cap_seconds == 3600
    assert s.harness_rotate_idle_seconds == 7200
    assert s.harness_rotate_transcript_bytes == 400_000
    assert s.harness_carryover_max_words == 300
    assert s.harness_concurrency_cap == 3
    assert s.harness_ask_expiry_seconds == 1800
    assert s.harness_status_edit_interval_seconds == 5
    assert s.harness_end_turn_grace_seconds == 15
    assert s.harness_carryover_model == "claude-haiku-5-5"


def test_harness_settings_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("HARNESS_STALL_SECONDS", "12")
    monkeypatch.setenv("HARNESS_HEARTBEAT_MINUTES", "[5, 10]")
    s = Settings(
        telegram_bot_token="t",
        telegram_bot_username="b",
        approved_directory=str(tmp_path),
    )
    assert s.harness_stall_seconds == 12
    assert s.harness_heartbeat_minutes == [5, 10]
