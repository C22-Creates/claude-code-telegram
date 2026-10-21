# Harness v2 — Track 2: Harness Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `src/bot/` + `src/claude/` with a ~1.5–2k-line `src/harness/` package so every Telegram run shows live status, can take follow-up messages mid-run, and always ends with ✅ / ⚠️ / ❌. No more silent 18–42-minute runs.

**Architecture:** One `TopicWorker` per forum topic owns at most one live `AgentRun`. An `AgentRun` wraps a single `ClaudeSDKClient` in streaming-input mode. It emits `ProgressEvent`s to a throttled `StatusMessage`, and it is bounded by a 5-min no-event watchdog and a 60-min cap. Scheduled jobs and Hermes dispatcher tasks run through the same `AgentRun`, with a fresh session each time. The scheduler, the notifications service, the dispatcher, the project registry and the SQLite file stay. Everything else in the upstream fork is deleted.

**Tech Stack:** Python 3.11+, `claude-agent-sdk` 0.2.157 (installed in `.venv`), `python-telegram-bot` 22.x, APScheduler 3.x, `aiosqlite`, `structlog`, pytest + pytest-asyncio + pytest-cov.

**Spec:** `docs/superpowers/specs/2026-10-17-telegram-harness-v2-design.md` §5 (Track 2), with §2 success criteria and §6 risks. (A copy is at c22os `inbox/2026-10-17-telegram-harness-v2-design.md`.)

## Global Constraints

- **Base branch is the deployed branch, not `main`.** `origin/main` is stale (it stops at PR #6). Production runs `fix/bounded-sdk-cleanup-and-reap` @ `84de931` (PRs #16–18). Rebase `rewrite/harness-v2` onto it in Task 0. The PR goes against it, not against `main`.
- Work only in the worktree `/home/c22bot/c22os-telegram-v2`. Never change the branch of the live checkout `/home/c22bot/c22os-telegram`.
- Defaults (add to `src/config/settings.py`, env-overridable, prefix `HARNESS_`): `stall_seconds=300`, `heartbeat_minutes=[20, 40]`, `run_cap_seconds=3600`, `rotate_idle_seconds=7200`, `rotate_transcript_bytes=400_000`, `carryover_max_words=300`, `concurrency_cap=3`, `ask_expiry_seconds=1800`, `status_edit_interval_seconds=5`, `end_turn_grace_seconds=15`, `carryover_model="claude-haiku-5-5"`.
- System prompt = the topic's `CLAUDE.md` + `context/intent-map.md` + `context/tool-recipes.md` (+ topic agent identity, if one is set). A missing file is skipped with a warning. The run never fails because of one.
- SDK options that must carry over from `src/claude/sdk_integration.py`: `setting_sources=["project"]`, `mcp_servers` from `config/mcp.json` (`_load_mcp_config` semantics), `include_partial_messages=True`, `cwd` = topic working dir, `model`/`max_turns` from existing settings, and list-typed (never `None`) `allowed_tools`/`disallowed_tools`.
- Port the lessons from these incidents unchanged, and keep their tests: the `message_delta` end-turn signal + 15s grace (PR #18), `_await_cancelled` with a hard bound (2026-07-29), and `_reap_wedged_child` (2026-08-21). These are bugs that have already happened. Do not "simplify" them away.
- Auth: only user IDs in `ALLOWED_USERS` (Carl, Sisi). Group messages only from `PROJECT_THREADS_CHAT_ID`; DMs from allowed users route to the `c22os` project. Everything else is ignored silently.
- Outcome footer copy, verbatim: `✅ done · {elapsed}`, `⚠️ stopped ({reason}) · {elapsed}`, `❌ error: {one line}`. Status line: `🔧 {tool} · turn {n} · {elapsed}`. Voice: `🎙️ heard: '{first 80 chars}' · starting`. Stall: `⚠️ stalled on {tool} for 5m, stopped. Reply to retry.` Cap: `⚠️ hit 60m cap`. Heartbeat: `still working · {n}m`. Restart: `♻️ restarted during your last request — reply to resume`. Queue: `⏳ queued`. Ask expiry: `no answer, stopped`. Cancel: `⚠️ cancelled`.
- `{elapsed}` format: `4m12s` (`42s` under a minute).
- Each component is tested with `tests/harness/fakes.py` (`FakeSDKClient`, `FakeBot`). No unit test touches the network or a real CLI. Coverage gate: `pytest tests/harness --cov=src/harness --cov-fail-under=80`.
- Commit format: `<type>(harness): <description>`, ending with the session's `Co-Authored-By` line.

## Review Focus

1. **Watchdog firing while the agent waits on a question.** An `AskUserQuestion` pending for 12 min must not trip the 5-min stall. Only the 30-min ask expiry applies. → Task 7 `test_watchdog_paused_while_question_pending`.
2. **A typed reply while buttons are showing.** Carl usually types instead of tapping. The text must answer the pending question (as "Other"), not be injected as a new prompt behind it. → Task 7 `test_text_reply_answers_pending_question`.
3. **Injection landing after the model's last turn but before `ResultMessage`.** The run must not post the final answer and close while an injected message is unanswered. → Task 3 `test_inject_after_end_turn_waits_for_second_result`.
4. **A final answer over 4096 chars, or with unbalanced Markdown/HTML.** It must arrive complete (split) and fall back to plain text if Telegram rejects the parse. It must not fail with a ❌ after the work finished. → Task 5 `test_long_answer_split_and_plaintext_fallback`.
5. **A scheduled job firing into a topic with a live interactive run.** It must queue behind (cap 3, interactive first), never inject into or interrupt Carl's run. → Task 9 `test_scheduled_job_never_injects_into_interactive_run`.

---

### Task 0: Branch setup and injection/ask spike

Validates spec §6 "Injection semantics" against the real SDK before anything depends on it.

**Files:**
- Create: `scripts/spike_streaming_input.py` (kept; it documents the verified behavior)
- Create: `docs/superpowers/plans/2026-10-20-harness-v2-track2-spike.md` (results)

- [ ] **Step 1:** `cd /home/c22bot/c22os-telegram-v2 && git rebase --onto 84de931 f0d5bae rewrite/harness-v2` → `git log --oneline -3` shows the spec commit on top of `84de931`.
- [ ] **Step 2:** Write the spike. Open `ClaudeSDKClient(options)` with `cwd=/home/c22bot/Projects/c22os`, `include_partial_messages=True`, no MCP, and a `can_use_tool` that logs its calls. `connect()` with no prompt, then `query("Count slowly from 1 to 30, one number per line, using a Bash `sleep 1` between each.")`. After 5s, call `query("Also: what is 2+2?")` while the first turn is still streaming. Then run a second scenario: `query("Use AskUserQuestion to ask me to pick red or blue, then say my choice.")`, and answer from `can_use_tool` by returning `PermissionResultAllow(updated_input={**input, "answers": {<question text>: "blue"}})`.
- [ ] **Step 3:** Run `.venv/bin/python scripts/spike_streaming_input.py`. Record in the results doc: (a) whether the mid-turn query is answered in the same session, and how many `ResultMessage`s arrive; (b) whether `AskUserQuestion` reaches `can_use_tool` and whether the answer comes back; (c) whether `client.interrupt()` stops a running Bash tool within 10s.
- [ ] **Step 4:** Decide. If (a) fails, Task 3 uses **queue-behind**: `inject()` buffers the message and sends it after the current `ResultMessage`, and Task 6 reacts 👀→⏳. If (b) fails, Task 7 falls back to a text question: the agent's question is forwarded as a message, and the reply is injected. Write the decision at the top of the results doc.
- [ ] **Step 5:** Commit: `chore(harness): rebase onto deployed branch; streaming-input spike`.

### Task 1: Settings, types, and test fakes

**Files:**
- Modify: `src/config/settings.py` (add the `harness_*` fields from Global Constraints)
- Create: `src/harness/__init__.py`, `src/harness/types.py`, `src/harness/util.py`
- Create: `tests/harness/__init__.py`, `tests/harness/fakes.py`, `tests/harness/test_types.py`

**Interfaces — Produces:**
- `Attachment(kind: Literal["photo","document"], path: Path, mime: str)`
- `Inbound(topic_key: str, chat_id: int, thread_id: int | None, message_id: int, user_id: int, text: str, attachments: list[Attachment], reply_to: int | None, is_voice: bool = False)`
- `ProgressEvent(kind: Literal["tool","text","turn","heartbeat","mcp_unavailable"], tool: str | None, turn: int, elapsed_s: float, detail: str = "")`
- `Outcome(Enum)`: `DONE`, `STOPPED`, `ERROR`
- `RunResult(outcome: Outcome, text: str, reason: str = "", session_id: str | None, elapsed_s: float, tool_calls: int)`
- `util.fmt_elapsed(seconds: float) -> str`, `util.footer(result: RunResult) -> str`
- `fakes.FakeSDKClient(script: list[ScriptStep])`: `connect()`, `query(msg)`, `receive_messages()` (async iterator yielding scripted SDK message objects; `ScriptStep` = `(delay_s, message)` or `("await_query",)`), `interrupt()`, `disconnect()`. Records `queries`, `interrupted`. Use real `claude_agent_sdk.types` classes for the messages (`StreamEvent`, `AssistantMessage`, `ToolUseBlock`, `TextBlock`, `ResultMessage`).
- `fakes.FakeBot`: `send_message`, `edit_message_text`, `set_message_reaction`, `answer_callback_query`. Each records `(method, kwargs)` in `calls` and returns objects with `message_id`. `fail_next(method, exc)` makes the next call raise.

- [ ] **Step 1: Failing tests** in `test_types.py`: `fmt_elapsed(42) == "42s"`, `fmt_elapsed(252) == "4m12s"`, `fmt_elapsed(3600) == "60m0s"`. `footer` produces the three outcome strings verbatim from Global Constraints. A settings test asserts every default listed there.
- [ ] **Step 2:** `.venv/bin/pytest tests/harness/test_types.py -v` → FAIL (module missing).
- [ ] **Step 3:** Implement the dataclasses (frozen), `util.py`, the fakes, and the settings fields.
- [ ] **Step 4:** Re-run → PASS. Also run `.venv/bin/pytest tests/unit/test_config.py` → still PASS.
- [ ] **Step 5:** Commit `feat(harness): settings, core types, test fakes`.

### Task 2: SDK lifecycle helpers (port, don't rewrite)

**Files:**
- Create: `src/harness/sdk_lifecycle.py`
- Test: `tests/harness/test_sdk_lifecycle.py`

**Interfaces — Produces:** `turn_signal(message) -> bool | None`, `async await_cancelled(task, reason, timeout=CANCELLED_TASK_CLEANUP_TIMEOUT) -> bool`, `descendant_pids(root_pid) -> list[int]`, `reap_wedged_child(client, reason) -> None`, `load_mcp_config(path: Path) -> dict`, `last_tool_name(messages) -> str | None`.

- [ ] **Step 1:** Move the tests for these behaviors from `tests/unit/test_claude/test_sdk_integration.py` (the turn-signal, end-turn grace, await-cancelled, reap and MCP-config tests) into `test_sdk_lifecycle.py`, retargeted at the new module-level functions. Run → FAIL (import).
- [ ] **Step 2:** Copy the bodies from `src/claude/sdk_integration.py` (`_turn_signal` L309, `_await_cancelled` L454, `_last_tool_name` L492, `_descendant_pids` L511, `_reap_wedged_child` L545, `_load_mcp_config` L1385) as module functions. Keep their docstrings, because the incident history lives there. Keep `CANCELLED_TASK_CLEANUP_TIMEOUT` and `MAX_REAPED_DESCENDANTS`.
- [ ] **Step 3:** Run → PASS.
- [ ] **Step 4:** Commit `refactor(harness): port SDK lifecycle helpers with their incident tests`.

### Task 3: AgentRun

**Files:**
- Create: `src/harness/agent_run.py`
- Test: `tests/harness/test_agent_run.py`

**Interfaces:**
- Consumes: Task 1 types, Task 2 helpers.
- Produces: `AgentRun(options: ClaudeAgentOptions, on_progress: Callable[[ProgressEvent], Awaitable[None]], settings: Settings, client_factory: Callable[[ClaudeAgentOptions], Any] = ClaudeSDKClient)`. Methods: `async run(prompt: str | list[dict]) -> RunResult`, `async inject(text: str | list[dict]) -> None`, `async interrupt(reason: str) -> None`, `pause_watchdog() / resume_watchdog()`, `session_id: str | None`, `is_active: bool`.

Algorithm (not fully determined by the tests):
- `run` connects, sends the prompt, and counts `queries_sent`. A reader task iterates `receive_messages()` and logs **every** message as `harness.sdk_event` (type, tool, parent_tool_use_id, monotonic t). It updates `last_event_at` and turns messages into `ProgressEvent`s (tool on `ToolUseBlock`, turn counter on main-agent `message_start`). It counts a `ResultMessage` toward `results_seen`. The run ends when `results_seen == queries_sent`. A main-turn end signal (`turn_signal`) followed by `end_turn_grace_seconds` of silence counts as a result (PR #18 behavior).
- Supervisor loop (1s tick): if not paused and `now - last_event_at ≥ stall_seconds`, stop with reason `stalled on {last_tool_name}`. At `heartbeat_minutes` marks, emit a `heartbeat` event. At `run_cap_seconds`, stop with reason `hit 60m cap`.
- Stop = `client.interrupt()` bounded at 10s. If that hangs, cancel the reader → `await_cancelled` → `reap_wedged_child`. The session ID is kept in every case.
- Final text = the concatenated `TextBlock`s of main-agent `AssistantMessage`s after the last user query. Partial text is included in a STOPPED result.
- An SDK/CLI exception → `Outcome.ERROR` with the first line of `str(exc)`. Retryable connect errors (`_is_retryable_error` semantics) are retried once before the first event, and never after.
- A `SystemMessage` init listing an MCP server with status `failed` → `ProgressEvent(kind="mcp_unavailable", detail=name)`, and the run continues.

- [ ] **Step 1: Failing tests:**
  - `test_happy_path_done`: script tool_use → text → ResultMessage ⇒ `outcome DONE`, `text` is the reply, `tool_calls == 1`, progress shows the tool.
  - `test_stall_interrupts_after_5m_keeps_session` (fake clock, `stall_seconds=300`): no events after a ToolUseBlock `mcp__missive__list_conversations` ⇒ `STOPPED`, `reason == "stalled on mcp__missive__list_conversations"`, `client.interrupted`, `session_id` set.
  - `test_heartbeats_at_20_and_40m` and `test_cap_at_60m_returns_partial_text`.
  - `test_end_turn_grace_completes_without_result`: `message_delta end_turn`, then silence ⇒ DONE after 15s.
  - `test_subagent_end_turn_ignored`: `message_delta` with `parent_tool_use_id` set does not start the grace period.
  - `test_inject_mid_turn_two_results` and `test_inject_after_end_turn_waits_for_second_result` (Review Focus 3).
  - `test_interrupt_hang_reaps_child`: fake `interrupt()` never returns ⇒ `reap_wedged_child` is called and `run` returns within 10s + cleanup timeout.
  - `test_cli_crash_is_error_one_line`, `test_mcp_failed_emits_unavailable_and_continues`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement. Inject a clock (`clock: Callable[[], float] = time.monotonic`, `sleep` likewise) so tests run in milliseconds. If Task 0 chose queue-behind, `inject` buffers and sends after the next result, and the two inject tests assert that ordering instead.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(harness): AgentRun with watchdog, cap, injection, bounded teardown`.

### Task 4: PromptBuilder and SessionStore

**Files:**
- Create: `src/harness/prompt_builder.py`, `src/harness/session_store.py`, `src/harness/carryover.py`
- Modify: `src/storage/database.py` (new migration: tables `harness_sessions`, `harness_runs`)
- Test: `tests/harness/test_prompt_builder.py`, `tests/harness/test_session_store.py`

**Interfaces — Produces:**
- `build_system_prompt(cwd: Path, agent: str | None) -> tuple[str, list[str]]`. Returns the prompt and the topic's denied tools. Reuses `src/dispatcher/agent_loader.load_agent` and honors `agent_tool_enforcement`.
- `build_options(cwd: Path, agent: str | None, settings: Settings, resume: str | None, can_use_tool) -> ClaudeAgentOptions`
- Schema: `harness_sessions(topic_key TEXT PK, session_id TEXT, cwd TEXT, created_at, last_active)` and `harness_runs(run_id TEXT PK, topic_key, chat_id, thread_id, status TEXT CHECK(status IN ('running','done','stopped','error','abandoned')), started_at, ended_at, prompt_head TEXT)`.
- `SessionStore(db_manager)`: `get(topic_key) -> SessionRow | None`, `save(topic_key, session_id, cwd)`, `clear(topic_key)`, `transcript_bytes(session_id, cwd) -> int`, `needs_rotation(row, now) -> bool`, `start_run(...) -> str`, `end_run(run_id, status)`, `running_runs() -> list[RunRow]`, `abandon_all_running()`.
- `transcript_bytes` = size of `~/.claude/projects/<cwd with / and . replaced by ->/<session_id>.jsonl`, or 0 if the file is missing.
- `async summarize_for_carryover(session_id: str, cwd: Path, settings) -> str`: a one-shot `claude_agent_sdk.query` with `resume=session_id`, `fork_session=True`, `model=carryover_model`, `max_turns=1`, `allowed_tools=[]`, prompt `"Summarize this conversation for a fresh session in ≤300 words: open tasks, decisions, names, IDs."`. Output is truncated to `carryover_max_words`. On any error it returns `""` (rotation then proceeds without a summary).

- [ ] **Step 1: Failing tests:** prompt contains all three files in order. A missing `tool-recipes.md` is skipped with a warning. Agent `coach` adds its identity, and its denied tools are returned. `needs_rotation` is true at `last_active` 2h+1s ago and at 400_001 bytes, false at 1h59m and 399_999. `abandon_all_running` flips only `running` rows. The migration is idempotent (run twice on a temp DB).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit `feat(harness): prompt builder, session store, carry-over summary`.

### Task 5: StatusMessage and formatting

**Files:**
- Create: `src/harness/status_message.py`, `src/harness/format.py` (move `src/bot/utils/html_format.py` here, plus `split_message(text, limit=4096)` from `NotificationService._split_message`)
- Test: `tests/harness/test_status_message.py`

**Interfaces — Produces:** `StatusMessage(bot, chat_id, thread_id, reply_to, settings, clock)`: `async open(text) -> int`, `async update(ev: ProgressEvent)` (coalesced, at most one edit per `status_edit_interval_seconds`, best-effort), `async finish(result: RunResult)` (edits the status to the footer, then posts the answer as new message(s)), `async note(text)` (immediate edit, used for ⏳/♻️/stall/MCP notes).

- [ ] **Step 1: Failing tests:** 20 updates in 3s produce ≤ 1 edit, and the last state wins on the next tick. A `RetryAfter`/`BadRequest` on edit is swallowed and logged. `finish` sends the answer as a **new** message replying to the user's message, then edits the status to `✅ done · 4m12s`. `test_long_answer_split_and_plaintext_fallback` (Review Focus 4): a 9000-char answer → 3 sends, and a `BadRequest("can't parse entities")` triggers a resend without parse_mode. A STOPPED result with empty text still posts the footer. `test_answer_send_retries_with_backoff`: on `NetworkError`/`RetryAfter`, the answer send is retried 3× (1s, 2s, 4s, or `retry_after`). Status edits are not retried.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat(harness): throttled status message and answer delivery`.

### Task 6: TopicRouter, TopicWorker, RunSlots

**Files:**
- Create: `src/harness/topic_router.py`, `src/harness/run_slots.py`
- Test: `tests/harness/test_topic_router.py`, `tests/harness/test_run_slots.py`

**Interfaces:**
- Consumes: `AgentRun`, `StatusMessage`, `SessionStore`, `build_options`, `summarize_for_carryover`.
- Produces: `RunSlots(cap: int)`: `async acquire(priority: Literal["interactive","scheduled"]) -> AsyncContextManager`, `would_wait() -> bool`. Waiting interactive callers are served before waiting scheduled ones.
- `TopicRouter(bot, store, slots, settings, resolve_topic: Callable[[Inbound], TopicContext])`, where `TopicContext(topic_key, cwd: Path, agent: str | None)`. Methods: `async handle(inbound: Inbound)`, `async new_session(topic_key)` (for `/new`), `async cancel(topic_key)` (for `/cancel`), `active_run(topic_key) -> AgentRun | None`.
- Flow: idle topic → open status (`⏳ queued` if `slots.would_wait()`) → acquire `interactive` → rotate if `needs_rotation` (summary prepended as `Context from previous session:\n…`) → `start_run` row → `AgentRun.run` → `finish` → `save` session → `end_run`. Busy topic → `inject` + 👀 reaction (⏳ in queue-behind mode). One `asyncio.Lock` per topic guards the idle/busy decision.

- [ ] **Step 1: Failing tests:** two topics run in parallel. A fourth concurrent run shows `⏳ queued` and starts when a slot frees. A second message in a busy topic is injected and gets 👀, and no second status message is created. `/cancel` → `⚠️ cancelled` and the session is kept. `/new` clears the session, so the next run has `resume=None` and no summary. A stale session (2h) runs the carry-over and starts fresh with the summary in the prompt. A summary failure still starts a fresh session. An `AgentRun` crash ends the run with ❌ and drops that topic's worker, so the next message gets a fresh client.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat(harness): topic router, per-topic workers, priority run slots`.

### Task 7: AskBridge

**Files:**
- Create: `src/harness/ask_bridge.py`
- Test: `tests/harness/test_ask_bridge.py`

**Interfaces — Produces:** `AskBridge(bot, settings, clock)`: `make_can_use_tool(topic_key, chat_id, thread_id, run: AgentRun) -> CanUseTool`. The callback allows every tool except `AskUserQuestion`. For `AskUserQuestion` it sends each question with an inline keyboard (one button per option + `Other…`, callback data `ask:{ask_id}:{idx}`), calls `run.pause_watchdog()`, and awaits the answer or `ask_expiry_seconds`. It then resumes the watchdog and returns `PermissionResultAllow(updated_input={**input, "answers": {...}})`. On expiry it returns `PermissionResultDeny(message="no answer", interrupt=True)` and edits the question to `no answer, stopped`. Also: `async on_callback(callback_query)` and `try_answer_with_text(topic_key, text) -> bool`.
- Wiring contract for Task 8: `TopicRouter.handle` calls `ask_bridge.try_answer_with_text` before injecting.

- [ ] **Step 1: Failing tests:** a button tap resolves the answer and removes the keyboard. `test_text_reply_answers_pending_question` (Review Focus 2). Expiry at 30 min → deny + interrupt + `no answer, stopped`. `test_watchdog_paused_while_question_pending` (Review Focus 1): an AgentRun with a fake clock and a pending ask for 12 min is not stopped, and the stall clock restarts from the answer. A stale button (unknown `ask_id`) answers the callback with "expired" and changes nothing. If Task 0 chose the text-question fallback, test that path instead.
- [ ] **Step 2–4:** FAIL → implement → PASS.
- [ ] **Step 5:** Commit `feat(harness): inline-button bridge for agent questions`.

### Task 8: TelegramAdapter, voice, attachments, commands

**Files:**
- Create: `src/harness/telegram_adapter.py`; move `src/bot/features/voice_handler.py` → `src/harness/voice.py` (`git mv`, and fix imports)
- Move: `tests/unit/test_bot/test_voice_handler.py` → `tests/harness/test_voice.py`
- Test: `tests/harness/test_telegram_adapter.py`

**Interfaces — Produces:** `TelegramAdapter(settings, router: TopicRouter, ask_bridge: AskBridge, voice: VoiceHandler, thread_manager: ProjectThreadManager, resume_bridge: ResumeBridge | None)`: `build_application() -> telegram.ext.Application` (with `concurrent_updates(True)`) and `resolve_topic(inbound) -> TopicContext` (via `project_threads` + `config/projects.yaml`, including `agent`; a DM → `c22os`). The handlers are text, voice, photo, document, callback query, `/new`, `/cancel`, and `/status` (lists active runs with elapsed time).
- Voice: open the status first (`🎙️ transcribing…`), transcribe, then update to `🎙️ heard: '{first 80}' · starting` and route. If the topic is busy, inject the transcript instead.
- Photos/documents: save to `data/inbox/<topic_key>/<YYYYmmdd-HHMMSS>-<name>`. The prompt gets `Attached file: <abs path>`. Photos are also sent as base64 image content blocks (the same shape as the current `_multimodal_prompt`).
- A reply to a dispatcher-blocked task message goes to `resume_bridge` first (existing behavior).

- [ ] **Step 1: Failing tests:** an unauthorized user or a foreign chat → no router call and no reply. A text update → `Inbound` with the right `topic_key` (`f"{chat_id}:{thread_id or 0}"`). A photo → a file saved under `data/inbox/…`, and an image block in the prompt. A voice note in a busy topic → injected transcript. A transcription failure → `❌ error: couldn't transcribe voice note`. The handler returns before the run finishes (the polling loop is never blocked). The callback query is routed to `ask_bridge.on_callback`.
- [ ] **Step 2–4:** FAIL → implement → PASS (this includes the moved voice tests).
- [ ] **Step 5:** Commit `feat(harness): Telegram adapter with voice, attachments, commands`.

### Task 9: Scheduled jobs and Hermes dispatcher through AgentRun

**Files:**
- Create: `src/harness/jobs.py`
- Modify: `src/dispatcher/runner.py` (swap `ClaudeIntegration.run_command` for `run_job`)
- Test: `tests/harness/test_jobs.py`; update `tests/unit/test_dispatcher/test_service.py` fakes if needed

**Interfaces — Produces:** `async run_job(prompt: str, cwd: Path, agent: str | None, slots: RunSlots, settings, model: str | None = None) -> RunResult`, which acquires a `scheduled` slot and always runs with a fresh session (`resume=None`), with no status message. `ScheduledJobHandler(event_bus, slots, settings)` subscribes to `ScheduledEvent` (replacing `AgentHandler.handle_scheduled`), calls `run_job`, and publishes `AgentResponseEvent` with the same fields the current handler sets, so `NotificationService` is unchanged. A ⚠️/❌ outcome is appended to the posted text with the footer.

- [ ] **Step 1: Failing tests:** a `ScheduledEvent` for `daily-digest` → one `AgentResponseEvent` with the original chat/thread IDs. `test_scheduled_job_never_injects_into_interactive_run` (Review Focus 5): with an interactive run active in the same topic, the job runs in its own `AgentRun`, waits for a slot when the cap is full, and never calls `inject`. Dispatcher `resolve_model` overrides still reach `AgentRun` options. The 60-min cap applies to jobs too.
- [ ] **Step 2–4:** FAIL → implement → PASS. Also run `.venv/bin/pytest tests/unit/test_dispatcher tests/unit/test_scheduler tests/unit/test_notifications` → PASS.
- [ ] **Step 5:** Commit `feat(harness): scheduled jobs and dispatcher run through AgentRun`.

### Task 10: App wiring, restart recovery, and deleting the old stack

**Files:**
- Create: `src/harness/app.py`
- Modify: `src/main.py` (build via `harness.app`. Keep logging setup, signal handling and the ordered shutdown: dispatcher → scheduler → notifications → adapter → storage)
- Delete: `src/bot/`, `src/claude/`, `src/api/`, `src/security/`, `src/storage/{facade,repositories,session_storage,models}.py` if nothing kept imports them, `src/config/features.py` flags that no longer apply, and their tests
- Test: `tests/harness/test_app.py`

**Interfaces — Produces:** `async build(settings) -> HarnessApp` and `HarnessApp.start() / stop()`. On start: `store.running_runs()` → post `♻️ restarted during your last request — reply to resume` to each topic, then `abandon_all_running()`.

- [ ] **Step 1: Failing tests:** two `running` rows at boot → two ♻️ messages, and both rows become `abandoned`. `stop()` interrupts active runs (bounded) and marks them `abandoned`, so the next boot posts ♻️.
- [ ] **Step 2–3:** FAIL → implement wiring.
- [ ] **Step 4:** Delete the old packages. `grep -rn "src.bot\|src.claude\|from \.\.claude\|from \.\.bot" src tests` → no hits. `.venv/bin/python -c "import src.main"` succeeds.
- [ ] **Step 5:** Full suite and gates: `.venv/bin/pytest -q` → all PASS. `.venv/bin/pytest tests/harness --cov=src/harness --cov-fail-under=80` → PASS. `wc -l src/harness/*.py | tail -1` should be ≤ ~2000. Report the number either way.
- [ ] **Step 6:** Commit `refactor(harness): wire app, restart recovery; remove upstream bot/claude stack`.

### Task 11: Live replay tests (real SDK, test topic)

**Files:**
- Create: `tests/live/test_replays.py` (marked `@pytest.mark.live`, skipped unless `HARNESS_LIVE=1`)

- [ ] **Step 1:** Write two replays. They drive `TopicRouter` with a real `ClaudeSDKClient` and a `FakeBot` that records edits, in the `c22os` cwd:
  1. The McGovern lookup (`"Do I have Steve McGovern at Crystal Ventures' email?"`). Assert: DONE in < 60s, ≤ 8 tool calls, and `contact-lookup` was invoked.
  2. The 2026-10-14 shape: a resumed long session plus a multi-step ask. Assert: no gap > 300s between FakeBot edits, and the run ends with one of the three footers.
- [ ] **Step 2:** `HARNESS_LIVE=1 .venv/bin/pytest tests/live -v -s` → PASS. Paste the timings into the spike results doc.
- [ ] **Step 3:** Commit `test(harness): live replay tests for the 10/14 and 10/17 incidents`.

### Task 12: Cutover (needs Carl's go-ahead at Step 3)

- [ ] **Step 1:** In the live checkout: `git tag pre-harness-v2 84de931 && git push origin pre-harness-v2`.
- [ ] **Step 2:** Push `rewrite/harness-v2` and open a PR against `fix/bounded-sdk-cleanup-and-reap`. The body includes the test plan, the line count, the spike results, and the rollback command.
- [ ] **Step 3: Stop and ask Carl** to merge and approve the restart. Check that no run is in flight first: `sudo journalctl -u c22os-telegram --since "-10 min"`.
- [ ] **Step 4:** After the merge: `cd /home/c22bot/c22os-telegram && git pull --ff-only && sudo systemctl restart c22os-telegram && sudo journalctl -u c22os-telegram -f`. Check the boot log and send a test message in the Ops topic → status edits, then ✅.
- [ ] **Step 5:** Run a 48h soak against spec §2. Criteria: zero 1800s-style timeouts, no silent period > 5 min, and every run with a footer. `journalctl -u c22os-telegram --since "48 hours ago" | grep -c harness.run_end` should match the count of outcome footers. A `harness.sdk_event` gap > 300s with no stop is a failure → `failure-modes.md`.
- [ ] **Step 6:** Rollback if needed: `git checkout pre-harness-v2 && sudo systemctl restart c22os-telegram`.
- [ ] **Step 7:** After a clean soak, update c22os: `workflows/telegram-bot.md`, `docs/runbooks/vps-runtime.md`, a `context/decisions.md` entry ("Harness v2 live; upstream syncs stopped"), and, if the §6 35-min gap root cause was observed, `context/examples/failure-modes.md`.
