"""Tests for SessionStore, the harness migration, and carry-over summary."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

from src.harness.carryover import CARRYOVER_PROMPT, summarize_for_carryover
from src.harness.session_store import SessionRow, SessionStore, project_slug
from src.storage.database import DatabaseManager

HOUR = 3600.0


@pytest.fixture
async def db(tmp_path):
    manager = DatabaseManager(f"sqlite:///{tmp_path / 'bot.db'}")
    await manager.initialize()
    yield manager
    await manager.close()


@pytest.fixture
def store(db, tmp_path):
    return SessionStore(db, projects_root=tmp_path / "projects")


def _row(last_active, session_id="s1", cwd="/w"):
    return SessionRow("t", session_id, cwd, last_active - 10, last_active)


async def test_migration_idempotent(tmp_path):
    url = f"sqlite:///{tmp_path / 'twice.db'}"
    first = DatabaseManager(url)
    await first.initialize()
    await first.close()
    second = DatabaseManager(url)
    await second.initialize()
    async with second.get_connection() as conn:
        cur = await conn.execute(
            "SELECT name FROM sqlite_master WHERE name LIKE 'harness_%'"
        )
        names = {r[0] for r in await cur.fetchall()}
    await second.close()
    assert {"harness_sessions", "harness_runs"} <= names


async def test_save_get_clear(store):
    assert await store.get("a") is None
    await store.save("a", "sess-1", Path("/w"), now=100.0)
    row = await store.get("a")
    assert (row.session_id, row.cwd, row.created_at) == ("sess-1", "/w", 100.0)
    await store.save("a", "sess-1", Path("/w"), now=200.0)
    row = await store.get("a")
    assert (row.created_at, row.last_active) == (100.0, 200.0)
    await store.save("a", "sess-2", Path("/w"), now=300.0)
    assert (await store.get("a")).created_at == 300.0  # new session resets
    await store.clear("a")
    assert await store.get("a") is None


def test_project_slug():
    assert project_slug(Path("/home/c22bot/Projects/c22os")) == (
        "-home-c22bot-Projects-c22os"
    )
    assert project_slug(Path("/a/.b/c")) == "-a--b-c"


def test_transcript_bytes(store, tmp_path):
    cwd = Path("/w/x")
    assert store.transcript_bytes("s1", cwd) == 0
    d = tmp_path / "projects" / project_slug(cwd)
    d.mkdir(parents=True)
    (d / "s1.jsonl").write_bytes(b"x" * 123)
    assert store.transcript_bytes("s1", cwd) == 123


def test_needs_rotation_idle(store):
    now = 1_000_000.0
    assert store.needs_rotation(_row(now - 7201), now)
    assert not store.needs_rotation(_row(now - (HOUR + 59 * 60)), now)


def test_needs_rotation_bytes(store, tmp_path):
    now = 1_000_000.0
    cwd = Path("/w")
    d = tmp_path / "projects" / project_slug(cwd)
    d.mkdir(parents=True)
    path = d / "s1.jsonl"
    row = _row(now - 10, cwd=str(cwd))
    path.write_bytes(b"x" * 400_001)
    assert store.needs_rotation(row, now)
    path.write_bytes(b"x" * 399_999)
    assert not store.needs_rotation(row, now)


async def test_runs_lifecycle(store):
    r1 = await store.start_run("t", 1, 2, "hello", now=10.0)
    r2 = await store.start_run("t", 1, 3, "x" * 500, now=11.0)
    r3 = await store.start_run("u", 1, None, "z", now=12.0)
    await store.end_run(r3, "done", now=13.0)
    running = await store.running_runs()
    assert {r.run_id for r in running} == {r1, r2}
    assert max(len(r.prompt_head) for r in running) <= 200
    await store.abandon_all_running(now=20.0)
    assert await store.running_runs() == []
    async with store._db.get_connection() as conn:
        cur = await conn.execute("SELECT run_id, status FROM harness_runs")
        status = {r[0]: r[1] for r in await cur.fetchall()}
    assert status == {r1: "abandoned", r2: "abandoned", r3: "done"}


async def test_end_run_rejects_bad_status(store):
    rid = await store.start_run("t", 1, 2, "p")
    with pytest.raises(ValueError):
        await store.end_run(rid, "running")


# ---- carry-over -------------------------------------------------------


def _settings(max_words=300):
    return SimpleNamespace(
        harness_carryover_model="claude-haiku-5-5",
        harness_carryover_max_words=max_words,
    )


def _fake_query(messages=None, exc=None, calls=None):
    async def fake(*, prompt, options):
        if calls is not None:
            calls.append((prompt, options))
        if exc:
            raise exc
        for m in messages or []:
            yield m

    return fake


def _result(text):
    return ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="fork",
        result=text,
    )


async def test_carryover_success_and_options():
    calls = []
    out = await summarize_for_carryover(
        "s1",
        Path("/w"),
        _settings(),
        query_fn=_fake_query([_result("open: A")], calls=calls),
    )
    assert out == "open: A"
    prompt, opts = calls[0]
    assert prompt == CARRYOVER_PROMPT
    assert opts.resume == "s1" and opts.fork_session is True
    assert opts.model == "claude-haiku-5-5" and opts.max_turns == 1
    assert opts.allowed_tools == [] and opts.setting_sources == []
    assert opts.cwd == "/w"


async def test_carryover_truncates_words():
    text = " ".join(f"w{i}" for i in range(50))
    out = await summarize_for_carryover(
        "s1", Path("/w"), _settings(10), query_fn=_fake_query([_result(text)])
    )
    assert out.split() == [f"w{i}" for i in range(10)]


async def test_carryover_falls_back_to_assistant_text():
    msg = AssistantMessage(content=[TextBlock(text="last words")], model="m")
    out = await summarize_for_carryover(
        "s1", Path("/w"), _settings(), query_fn=_fake_query([msg, _result(None)])
    )
    assert out == "last words"


async def test_carryover_error_returns_empty():
    out = await summarize_for_carryover(
        "s1", Path("/w"), _settings(), query_fn=_fake_query(exc=RuntimeError("x"))
    )
    assert out == ""


async def test_carryover_timeout_returns_empty():
    async def hang(*, prompt, options):
        await asyncio.sleep(10)
        yield _result("never")

    out = await summarize_for_carryover(
        "s1", Path("/w"), _settings(), query_fn=hang, timeout=0.01
    )
    assert out == ""
