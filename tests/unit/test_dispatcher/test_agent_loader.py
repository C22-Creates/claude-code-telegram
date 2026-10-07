"""Tests for the agent identity/enforcement loader (agents/README.md contract)."""

from pathlib import Path

from src.dispatcher.agent_loader import load_agent


def _make_agent(tmp_path: Path, name: str = "relationship-engine") -> Path:
    agent_dir = tmp_path / "agents" / name
    (agent_dir / "memory").mkdir(parents=True)
    (agent_dir / "SOUL.md").write_text("# Soul\nBe the relationship engine.")
    (agent_dir / "RULES.md").write_text("# Rules\nNever send email.")
    (agent_dir / "DUTIES.md").write_text("# Duties\nPeople DB only.")
    (agent_dir / "memory" / "notes.md").write_text("- 2026-09-04: lesson one")
    (agent_dir / "agent.yaml").write_text(
        "name: relationship-engine\n"
        "enforcement:\n"
        "  disallowed_tools:\n"
        '    - "mcp__missive__send_message"\n'
        '    - "mcp__read-ai__*"\n'
    )
    return agent_dir


def test_loads_identity_and_denials(tmp_path):
    _make_agent(tmp_path)
    identity, denied = load_agent("relationship-engine", tmp_path)
    assert identity is not None
    assert "You are the `relationship-engine` agent" in identity
    assert "Be the relationship engine." in identity
    assert "Never send email." in identity
    assert "People DB only." in identity
    assert "lesson one" in identity
    assert denied == ["mcp__missive__send_message", "mcp__read-ai__*"]


def test_unknown_assignee_degrades_to_generic(tmp_path):
    identity, denied = load_agent("nonexistent", tmp_path)
    assert identity is None
    assert denied == []


def test_empty_and_path_shaped_assignees_are_refused(tmp_path):
    _make_agent(tmp_path)
    for bad in (None, "", "  ", "../coach", "a/b", ".hidden"):
        identity, denied = load_agent(bad, tmp_path)
        assert identity is None
        assert denied == []


def test_missing_optional_files_are_tolerated(tmp_path):
    agent_dir = tmp_path / "agents" / "coach"
    agent_dir.mkdir(parents=True)
    (agent_dir / "SOUL.md").write_text("# Soul only")
    identity, denied = load_agent("coach", tmp_path)
    assert identity is not None
    assert "Soul only" in identity
    assert denied == []


def test_malformed_manifest_never_raises(tmp_path):
    agent_dir = _make_agent(tmp_path, "coach")
    (agent_dir / "agent.yaml").write_text(": not [ valid yaml {{{")
    identity, denied = load_agent("coach", tmp_path)
    assert identity is None or isinstance(identity, str)
    assert denied == []


def test_oversized_identity_is_truncated(tmp_path):
    agent_dir = _make_agent(tmp_path, "coach")
    (agent_dir / "memory" / "notes.md").write_text("x" * 100_000)
    identity, _ = load_agent("coach", tmp_path)
    assert identity is not None
    assert len(identity) < 70_000
    assert "[identity block truncated]" in identity
