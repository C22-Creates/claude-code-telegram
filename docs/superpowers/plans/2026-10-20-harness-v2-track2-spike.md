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
