"""Tests for the harness prompt/options builder."""

import json

import pytest

from src.config.settings import Settings
from src.harness.prompt_builder import build_options, build_system_prompt


def _make_topic(tmp_path, recipes=True):
    (tmp_path / "context").mkdir()
    (tmp_path / "CLAUDE.md").write_text("CLAUDE-BODY")
    (tmp_path / "context" / "intent-map.md").write_text("INTENT-BODY")
    if recipes:
        (tmp_path / "context" / "tool-recipes.md").write_text("RECIPES-BODY")
    return tmp_path


def _agent(tmp_path, name="coach"):
    d = tmp_path / "agents" / name
    d.mkdir(parents=True)
    (d / "SOUL.md").write_text("I AM COACH")
    (d / "agent.yaml").write_text(
        "enforcement:\n  disallowed_tools:\n    - mcp__missive__send_message\n"
    )


@pytest.fixture
def make_settings(tmp_path):
    def make(**kw):
        return Settings(
            telegram_bot_token="t",
            telegram_bot_username="b",
            approved_directory=str(tmp_path),
            **kw,
        )

    return make


def test_prompt_order(tmp_path):
    cwd = _make_topic(tmp_path)
    prompt, denied = build_system_prompt(cwd, None)
    assert (
        prompt.index("CLAUDE-BODY")
        < prompt.index("INTENT-BODY")
        < prompt.index("RECIPES-BODY")
    )
    assert f"All file operations must stay within {cwd}" in prompt
    assert denied == []


def test_missing_file_skipped_with_warning(tmp_path):
    import structlog.testing

    cwd = _make_topic(tmp_path, recipes=False)
    with structlog.testing.capture_logs() as logs:
        prompt, _ = build_system_prompt(cwd, None)
    assert "INTENT-BODY" in prompt and "RECIPES-BODY" not in prompt
    assert any(
        entry["log_level"] == "warning" and "tool-recipes" in entry["path"]
        for entry in logs
    )


def test_agent_identity_and_denied(tmp_path):
    cwd = _make_topic(tmp_path)
    _agent(cwd)
    prompt, denied = build_system_prompt(cwd, "coach")
    assert "I AM COACH" in prompt
    assert prompt.index("RECIPES-BODY") < prompt.index("I AM COACH")
    assert denied == ["mcp__missive__send_message"]


def test_warn_mode_does_not_deny(tmp_path):
    cwd = _make_topic(tmp_path)
    _agent(cwd)
    _, denied = build_system_prompt(cwd, "coach", enforcement="warn")
    assert denied == []


def test_options_core(tmp_path, make_settings):
    cwd = _make_topic(tmp_path)
    _agent(cwd)

    async def cb(*a):  # pragma: no cover
        ...

    s = make_settings(claude_max_turns=7, claude_model="m-1")
    o = build_options(cwd, "coach", s, "sess-9", cb)
    assert o.include_partial_messages is True
    assert o.setting_sources == ["project"]
    assert o.permission_mode == "default"
    assert o.can_use_tool is cb
    assert o.resume == "sess-9"
    assert o.cwd == str(cwd)
    assert o.max_turns == 7 and o.model == "m-1"
    assert isinstance(o.allowed_tools, list)
    assert "AskUserQuestion" not in o.allowed_tools
    assert "mcp__missive__send_message" in o.disallowed_tools
    assert "I AM COACH" in o.system_prompt
    assert o.stderr is not None
    assert o.max_budget_usd == s.claude_max_cost_per_request


def test_options_no_resume_and_disable_validation(tmp_path, make_settings):
    cwd = _make_topic(tmp_path)
    s = make_settings(disable_tool_validation=True)
    o = build_options(cwd, None, s, None, None)
    assert o.resume is None
    assert o.allowed_tools == [] and o.disallowed_tools == []


def test_agent_denials_apply_even_when_validation_disabled(tmp_path, make_settings):
    cwd = _make_topic(tmp_path)
    _agent(cwd)
    s = make_settings(disable_tool_validation=True)
    o = build_options(cwd, "coach", s, None, None)
    assert o.disallowed_tools == ["mcp__missive__send_message"]


def test_askuserquestion_in_allowed_asserts(tmp_path, make_settings):
    cwd = _make_topic(tmp_path)
    s = make_settings(claude_allowed_tools=["Read", "AskUserQuestion"])
    with pytest.raises(AssertionError):
        build_options(cwd, None, s, None, None)


def test_mcp_servers_loaded(tmp_path, make_settings):
    cwd = _make_topic(tmp_path)
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {"x": {"command": "y"}}}))
    s = make_settings(enable_mcp=True, mcp_config_path=str(cfg))
    o = build_options(cwd, None, s, None, None)
    assert o.mcp_servers == {"x": {"command": "y"}}
