"""Spike: verify streaming-input injection, AskUserQuestion via can_use_tool, interrupt.

Run: .venv/bin/python scripts/spike_streaming_input.py [injection|ask|interrupt|all] [model]
Kept as documentation of verified SDK behavior (see docs/superpowers/plans/
2026-10-20-harness-v2-track2-spike.md).
"""
import asyncio
import sys
import time

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    PermissionResultAllow,
)

CWD = "/home/c22bot/Projects/c22os"
MODEL = "claude-haiku-4-5-20251001"
SCENARIO_TIMEOUT = 180
T0 = time.monotonic()


def log(kind: str, detail: str = "") -> None:
    print(f"[{time.monotonic() - T0:7.2f}s] {kind:<22} {detail}", flush=True)


def make_options(can_use_tool, model: str) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        cwd=CWD,
        model=model,
        include_partial_messages=True,
        can_use_tool=can_use_tool,
        max_turns=40,
        permission_mode="default",
    )


def describe(msg) -> str:
    name = type(msg).__name__
    if name == "HookEventMessage":
        return ""  # noisy startup hooks
    if name == "StreamEvent":
        ev = msg.event
        et = ev.get("type")
        if et in ("message_start", "message_delta", "message_stop"):
            extra = ev.get("delta", {}).get("stop_reason", "") if et == "message_delta" else ""
            return f"StreamEvent:{et} {extra}"
        return ""  # skip noisy deltas
    if name == "ResultMessage":
        return f"ResultMessage turns={msg.num_turns} err={msg.is_error} result={str(msg.result)[:200]!r}"
    content = getattr(msg, "content", None)
    if isinstance(content, list):
        parts = []
        for b in content:
            bn = type(b).__name__
            if bn == "TextBlock":
                parts.append(f"text={b.text[:120]!r}")
            elif bn == "ToolUseBlock":
                parts.append(f"tool_use={b.name} {str(b.input)[:100]}")
            elif bn == "ToolResultBlock":
                parts.append(f"tool_result={str(b.content)[:80]!r}")
            else:
                parts.append(bn)
        return f"{name} " + " | ".join(parts)
    return name


async def pump(client, results: list, stop_after_results: int) -> None:
    async for msg in client.receive_messages():
        d = describe(msg)
        if d:
            log("MSG", d)
        if type(msg).__name__ == "ResultMessage":
            results.append(msg)
            if len(results) >= stop_after_results:
                return


async def injection(model: str) -> None:
    async def cut(tool, inp, ctx):
        log("can_use_tool", f"{tool} {str(inp)[:100]}")
        return PermissionResultAllow()

    results: list = []
    client = ClaudeSDKClient(make_options(cut, model))
    await client.connect()
    try:
        await client.query(
            "Count slowly from 1 to 30, one number per line, using a Bash `sleep 1` between each."
        )
        log("SENT", "first query")
        pump_task = asyncio.create_task(pump(client, results, 2))
        await asyncio.sleep(5)
        await client.query("Also: what is 2+2?")
        log("SENT", "mid-turn query 'Also: what is 2+2?'")
        # wait for 2 results, or 45s after the first result then give up
        deadline = time.monotonic() + 150
        while not pump_task.done() and time.monotonic() < deadline:
            await asyncio.sleep(1)
            if len(results) == 1 and not pump_task.done():
                await asyncio.sleep(20)
                break
        if not pump_task.done():
            pump_task.cancel()
        log("RESULT", f"ResultMessage count={len(results)}")
    finally:
        await client.disconnect()


async def ask(model: str) -> None:
    seen = []

    async def cut(tool, inp, ctx):
        log("can_use_tool", f"{tool} {str(inp)[:300]}")
        seen.append(tool)
        if tool == "AskUserQuestion":
            answers = {q["question"]: "blue" for q in inp.get("questions", [])}
            return PermissionResultAllow(updated_input={**inp, "answers": answers})
        return PermissionResultAllow()

    results: list = []
    client = ClaudeSDKClient(make_options(cut, model))
    await client.connect()
    try:
        await client.query(
            "Use AskUserQuestion to ask me to pick red or blue, then say my choice."
        )
        await pump(client, results, 1)
        log("RESULT", f"tools via can_use_tool={seen}")
    finally:
        await client.disconnect()


async def interrupt(model: str) -> None:
    async def cut(tool, inp, ctx):
        log("can_use_tool", f"{tool} {str(inp)[:100]}")
        return PermissionResultAllow()

    results: list = []
    client = ClaudeSDKClient(make_options(cut, model))
    await client.connect()
    try:
        await client.query("Run exactly this Bash command and then say done: python3 -c \"import time; time.sleep(120)  # SPIKEMARK\"")
        pump_task = asyncio.create_task(pump(client, results, 1))
        await asyncio.sleep(12)  # let Bash start
        t = time.monotonic()
        log("INTERRUPT", "calling client.interrupt()")
        await client.interrupt()
        try:
            await asyncio.wait_for(pump_task, 10)
            log("RESULT", f"turn ended {time.monotonic() - t:.2f}s after interrupt")
        except asyncio.TimeoutError:
            log("RESULT", "turn did NOT end within 10s of interrupt")
            pump_task.cancel()
        import subprocess
        out = subprocess.run(["pgrep", "-f", "SPIKEMARK"], capture_output=True, text=True).stdout
        log("PGREP", ("pids " + out.strip().replace(chr(10), ",")) if out.strip() else "no SPIKEMARK process remains")
    finally:
        await client.disconnect()


async def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    model = sys.argv[2] if len(sys.argv) > 2 else MODEL
    scenarios = {"injection": injection, "ask": ask, "interrupt": interrupt}
    for name, fn in scenarios.items():
        if which in ("all", name):
            log("SCENARIO", f"{name} model={model}")
            try:
                await asyncio.wait_for(fn(model), SCENARIO_TIMEOUT)
            except Exception as e:  # noqa: BLE001 - spike logs everything
                log("SCENARIO_ERROR", f"{name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
