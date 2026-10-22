# Track 2 spike results: streaming input, AskUserQuestion, interrupt

**Decisions:** injection = **live** (mid-turn `query()` accepted by the live client and answered in the same turn) | ask = **buttons via can_use_tool** (AskUserQuestion reaches the callback; `updated_input.answers` is honored) | interrupt = **works** (turn ended 0.05s after `client.interrupt()`; running Bash tool rejected/killed).

Script: `scripts/spike_streaming_input.py` (`injection|ask|interrupt|all [model]`). SDK claude-agent-sdk 0.2.157, model `claude-haiku-4-5-20251001` (worked), `permission_mode="default"`, no `allowed_tools`, no MCP, `include_partial_messages=True`, cwd `/home/c22bot/Projects/c22os`, `can_use_tool` logging and allowing.
Timestamps below are seconds on the script's monotonic clock.

## (a) Mid-turn injection: live
Run 1 and run 2 behaved identically.
- Run 2: first `query` sent; assistant emits Bash tool_use at 11.02s (`for i in 1..30; sleep 1`); second `query("Also: what is 2+2?")` sent at 10.45s while the turn was in progress (tool pending/running; can_use_tool fired at 11.77s; tool ran until 42.09s).
- The CLI accepted it without error and did not start a separate turn. After the tool_result (42.09s) the next assistant message (45.06s) answered both: "Done! I've counted ... **2 + 2 = 4**".
- Exactly **one** `ResultMessage` (turns=2) arrived, at 45.10s; none for the injected message. Run 1: same, ResultMessage at 43.13s, count=1. Waited 20s after the result for a second ResultMessage in both runs; none came.
- Semantics: an injected message is consumed at the next model-call boundary (after the current tool call finishes), merged into the current turn. It is not processed while a long tool runs (the 2+2 answer waited ~35s for the Bash tool). Turn boundary = ResultMessage, one per turn regardless of injections.
- Caveat: the injected text was picked up in a turn that had a tool call pending. Not tested: injection during a pure text-generation stretch with no tool call (expect the same, boundary-based) or after the ResultMessage (normal new turn).

## (b) AskUserQuestion: reaches can_use_tool
- `can_use_tool` fired for `AskUserQuestion` at 9.79s with input `{'questions': [{'question': 'Which color do you prefer?', 'header': 'Color Choice', 'options': [{'label': 'Red', ...}, {'label': 'Blue', ...}], 'multiSelect': False}]}`.
- Returning `PermissionResultAllow(updated_input={**input, "answers": {<question text>: "blue"}})` produced tool_result `The user answered: "Which color do you prefer?"="blue". ...`, and the final text was "You chose **blue**." One ResultMessage.
- The callback can therefore `await` a Telegram button press before returning (not tested with a long delay; the callback is just an awaitable, but a very long wait may hit a control-request timeout in the SDK; verify in Task 7 with a real delay).
- Works with `permission_mode="default"` and AskUserQuestion not in `allowed_tools`.

## (c) interrupt
- Initial attempt with `sleep 120` was refused by the CLI itself ("Blocked: standalone sleep"), so the scenario uses `python3 -c "import time; time.sleep(120)"`.
- Tool started (TaskStartedMessage 11.99s); `client.interrupt()` called at 15.37s. By 15.42s: tool_result "The user doesn't want to proceed with this tool use...", "[Request interrupted by user for tool use]", and `ResultMessage(is_error=True, result=None)`. Turn ended **0.05s** after interrupt (limit 10s).
- A TaskNotificationMessage arrived at 15.38s, and no `time.sleep` process remained 5s later (the only pgrep match was the observing shell). Immediately after the call pgrep showed one transient pid, which I take to be the shell wrapper being torn down. Not independently proven.
- Note: ResultMessage after interrupt has `is_error=True` and `result=None`; Task code must not treat that as a crash.

## Other observations
- The stream contains many `HookEventMessage`/`SystemMessage` items at startup (user hooks); the spike filters them.
- `StreamEvent` `message_delta` with `stop_reason` `tool_use`/`end_turn` appears per model call, so it is not a turn boundary on its own. Only `ResultMessage` is.
- `receive_messages()` stays open across turns, which is what the live-injection design needs. `receive_response()` would stop at the first ResultMessage.

## (d) long can_use_tool delay (Task 7)
- SDK source (`claude_agent_sdk/_internal/query.py`): inbound `can_use_tool` control requests are run as a spawned task (`_handle_control_request`) with no timeout; the only control-request timeouts are on requests the SDK sends to the CLI (`initialize` 60s, others 60s via `anyio.fail_after`). Nothing bounds how long our callback may take.
- Live check (real CLI, `claude-haiku-4-5-20251001`, no MCP, scratch cwd): AskUserQuestion callback slept 120s before returning `PermissionResultAllow(updated_input={..., "answers": {q: "blue"}})`. Callback fired at 7.24s, returned at 127.31s; the tool_result arrived 0.2s later ("The user answered: ... = \"blue\""), the run continued and ended with one ResultMessage ("You chose Blue"). The answer round-trips after a 2-minute human delay.
- Not tested beyond 120s; no hard CLI-side timeout was observed or found in the SDK. The bridge's own 30-minute expiry (`harness_ask_expiry_seconds=1800`) is the effective limit.

## (e) live replays (Task 11)

`HARNESS_LIVE=1 .venv/bin/pytest tests/live -s` (TopicRouter + real AgentRun +
real ClaudeSDKClient + live MCP config, FakeBot, cwd c22os, production model).
Max status gap = longest silence between consecutive bot sends/edits (status
edits coalesce at 5s, so ~5s is the floor). `contact-lookup` detected as a
`Skill` tool use whose input contains "contact-lookup".

| Replay | Run | Elapsed | Tool calls | contact-lookup | Max status gap | Footer |
|---|---|---|---|---|---|---|
| 10/17 McGovern lookup | 1 | 21.1s | 3 (Skill, Bash, notion-query-data-sources) | yes | 5.0s | ✅ done · 21s |
| 10/17 McGovern lookup | 2 | 16.6s | 3 (same) | yes | 5.0s | ✅ done · 16s |
| 10/14 resumed multi-step | 1 | 33.7s | 6 (Bash x6) | n/a | 8.3s | ✅ done · 33s |
| 10/14 resumed multi-step | 2 | 34.5s | 6 (Bash x6) | n/a | 5.0s | ✅ done · 34s |

Targets (< 60s, <= 8 tools, contact-lookup used; gap <= 300s; explicit footer):
all met in all 4 runs.

Read-only enforcement re-run (inner `can_use_tool` allowlist: Skill/Read/Glob/Grep,
read-named MCP tools, Bash limited to `gog` calendar/drive/docs/sheets reads, the
contacts-index script and plain inspection commands; anything else denied and the
test fails). One run each, all targets met, nothing denied:

| Replay | Elapsed | Tool calls | contact-lookup | Max status gap | Footer |
|---|---|---|---|---|---|
| 10/17 McGovern lookup | 18.4s | 3 (Skill, Bash, notion-query-data-sources) | yes (Skill) | 5.0s | ✅ done · 18s |
| 10/14 resumed multi-step | 34.4s | 6 (Bash) | n/a | 8.9s | ✅ done · 34s |

Bash commands recorded: 10/17 `python3 scripts/contacts-lookup.py "McGovern"; echo ...;
python3 scripts/contacts-lookup.py "Crystal"`. 10/14: four `gog calendar events <addr>
--account <alias> --from 2026-10-22 --to 2026-10-23 --max 50 --json --results-only`
(one per calendar), a `jq` filter, and `python3 scripts/contacts-lookup.py
"ocean-ledger.com" 2>&1 | head -60`.

Not reproduced: the 10/14 incident's long-resumed-context shape (a ~1.1 MB
transcript); the replay resumes a one-line warm-up session.
