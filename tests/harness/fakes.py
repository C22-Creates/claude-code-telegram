"""Test doubles for the harness: a scripted SDK client and a recording bot.

No network, no CLI. Time is injected (``sleep=``) so delayed scripts run in
milliseconds.
"""

import asyncio
import inspect
from itertools import count
from types import SimpleNamespace
from typing import Any, AsyncIterator, Awaitable, Callable

import telegram
from claude_agent_sdk.types import (
    AssistantMessage,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
)

# ---------------------------------------------------------------- SDK builders


_TOOL_IDS = count(1)


def message_start(parent_tool_use_id: str | None = None) -> StreamEvent:
    """Raw ``message_start`` stream event (a model call begins)."""
    return StreamEvent(
        uuid="u-start",
        session_id="sess-1",
        event={"type": "message_start"},
        parent_tool_use_id=parent_tool_use_id,
    )


def message_delta_end_turn(parent_tool_use_id: str | None = None) -> StreamEvent:
    """Raw ``message_delta`` with ``stop_reason=end_turn`` (model finished)."""
    return StreamEvent(
        uuid="u-delta",
        session_id="sess-1",
        event={"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        parent_tool_use_id=parent_tool_use_id,
    )


def tool_use(
    name: str,
    input: dict[str, Any] | None = None,
    parent_tool_use_id: str | None = None,
) -> AssistantMessage:
    """Assistant message carrying a single tool call."""
    return AssistantMessage(
        content=[
            ToolUseBlock(
                id=f"tu-{name}-{next(_TOOL_IDS)}", name=name, input=input or {}
            )
        ],
        model="fake-model",
        parent_tool_use_id=parent_tool_use_id,
    )


def text(s: str, parent_tool_use_id: str | None = None) -> AssistantMessage:
    """Assistant message carrying a single text block."""
    return AssistantMessage(
        content=[TextBlock(text=s)],
        model="fake-model",
        parent_tool_use_id=parent_tool_use_id,
    )


def result(
    session_id: str = "sess-1",
    is_error: bool = False,
    text: str | None = "ok",
    subtype: str | None = None,
) -> ResultMessage:
    """Turn-ending ``ResultMessage``."""
    return ResultMessage(
        subtype=subtype or ("error_during_execution" if is_error else "success"),
        duration_ms=1,
        duration_api_ms=1,
        is_error=is_error,
        num_turns=1,
        session_id=session_id,
        result=text,
    )


def system_init(mcp_servers: list[dict[str, str] | str] | None = None) -> SystemMessage:
    """``init`` system message; bare names get status ``connected``."""
    servers = [
        s if isinstance(s, dict) else {"name": s, "status": "connected"}
        for s in (mcp_servers or [])
    ]
    return SystemMessage(subtype="init", data={"mcp_servers": servers})


# ------------------------------------------------------------------ SDK client

# (delay_s, message) | ("await_query",) | ("raise", exc) | ("call", fn)
# ``("call", fn)`` runs ``fn()`` (awaited if it returns an awaitable) when the
# step is reached -- after the consumer has processed every earlier message.
# It lets a test act at an exact point in the stream (inject, pause, ...).
ScriptStep = tuple


class FakeSDKClient:
    """Scripted stand-in for ``ClaudeSDKClient``.

    ``receive_messages()`` replays ``script`` and then stays open (as the real
    one does across turns) until ``disconnect()``. ``interrupt()`` ends the
    current scripted turn: remaining steps up to and including the next
    ``ResultMessage`` are dropped and an interrupt result
    (``is_error=True, result=None``) is emitted instead.

    Args:
        script: steps; see ``ScriptStep``.
        options: the ``ClaudeAgentOptions`` the client was built with.
        sleep: awaited for each step delay (inject a no-op for speed).
        hang_interrupt: make ``interrupt()`` never return and never end the turn.
        raise_on_receive: raised as soon as ``receive_messages()`` is iterated.
    """

    def __init__(
        self,
        script: list[ScriptStep] | None = None,
        options: Any = None,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        hang_interrupt: bool = False,
        raise_on_receive: BaseException | None = None,
    ) -> None:
        self.script = list(script or [])
        self.options = options
        self.hang_interrupt = hang_interrupt
        self.raise_on_receive = raise_on_receive
        self.queries: list[Any] = []
        self.interrupted = 0
        self.connected = False
        self.disconnected = False
        self._sleep = sleep
        self._query_evt = asyncio.Event()
        self._interrupt_evt = asyncio.Event()
        self._closed_evt = asyncio.Event()

    @classmethod
    def factory(cls, script: list[ScriptStep] | None = None, **kwargs: Any):
        """Build a ``client_factory``: ``options -> client``.

        The callable always returns the same client (records the options on it)
        and exposes it as ``.client`` for assertions.
        """
        client = cls(script, **kwargs)

        def build(options: Any = None) -> "FakeSDKClient":
            client.options = options
            return client

        build.client = client  # type: ignore[attr-defined]
        return build

    async def connect(self, prompt: Any = None) -> None:
        self.connected = True

    async def query(self, message: Any, session_id: str = "default") -> None:
        self.queries.append(message)
        self._query_evt.set()

    async def interrupt(self) -> None:
        self.interrupted += 1
        if self.hang_interrupt:
            await self._closed_evt.wait()
            return
        self._interrupt_evt.set()

    async def disconnect(self) -> None:
        self.disconnected = True
        self._closed_evt.set()

    async def _wait(self, extra: Awaitable[Any] | None = None) -> str:
        """Wait for close, interrupt, or ``extra``; return which one won.

        Ties resolve deterministically: closed > interrupt > extra, judged from
        the events' state after waking (not from asyncio's unordered ``done`` set).
        """
        tasks = [
            asyncio.ensure_future(self._closed_evt.wait()),
            asyncio.ensure_future(self._interrupt_evt.wait()),
        ]
        if extra is not None:
            tasks.append(asyncio.ensure_future(extra))
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks:
                t.cancel()
        if self._closed_evt.is_set():
            return "closed"
        if self._interrupt_evt.is_set():
            return "interrupt"
        return "extra"

    def _interrupt_result(self) -> ResultMessage:
        return result(is_error=True, text=None, subtype="error_during_execution")

    def _consume_interrupt(self, steps: list[ScriptStep]) -> ResultMessage:
        """Drop steps through the next ResultMessage; return the interrupt result."""
        self._interrupt_evt.clear()
        while steps:
            step = steps.pop(0)
            if len(step) == 2 and isinstance(step[1], ResultMessage):
                break
        return self._interrupt_result()

    async def receive_messages(self) -> AsyncIterator[Any]:
        if self.raise_on_receive is not None:
            raise self.raise_on_receive
        steps = list(self.script)
        # ``await_query`` waits for a query() made after the step is reached,
        # except a leading one, which is satisfied by the initial query() that
        # the harness sends before it starts reading the stream.
        started = False
        await_target: int | None = None
        while True:
            if self._closed_evt.is_set():
                return
            if self._interrupt_evt.is_set():
                await_target = None
                yield self._consume_interrupt(steps)
                started = True
                continue
            if not steps:
                if await self._wait() == "closed":
                    return
                continue
            step = steps[0]
            if step[0] == "await_query":
                if await_target is None:
                    await_target = 0 if not started else len(self.queries)
                if len(self.queries) > await_target:
                    steps.pop(0)
                    await_target = None
                    started = True
                    continue
                self._query_evt.clear()
                if len(self.queries) > await_target:
                    continue
                if await self._wait(self._query_evt.wait()) == "closed":
                    return
                continue
            if step[0] == "raise":
                raise step[1]
            if step[0] == "call":
                steps.pop(0)
                started = True
                outcome = step[1]()
                if inspect.isawaitable(outcome):
                    await outcome
                continue
            delay, message = step
            if delay and await self._wait(self._sleep(delay)) != "extra":
                continue
            steps.pop(0)
            started = True
            yield message


# ------------------------------------------------------------------------- Bot


class FakeBot:
    """Recording stand-in for ``telegram.Bot``'s async API.

    Every call (including ones armed to fail) is appended to ``calls`` as
    ``(method, kwargs)`` with exactly the kwargs the caller passed. Message-
    producing methods return an object with an incrementing ``message_id``
    (``edit_message_text`` echoes the edited ``message_id``).
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._ids = count(1)
        self._failures: dict[str, list[BaseException]] = {}

    def fail_next(self, method: str, exc: BaseException) -> None:
        """Make the next ``method`` call raise ``exc`` (queued FIFO)."""
        self._failures.setdefault(method, []).append(exc)

    def calls_for(self, method: str) -> list[dict[str, Any]]:
        return [kw for m, kw in self.calls if m == method]

    def _record(self, method: str, kwargs: dict[str, Any]) -> None:
        params = inspect.signature(getattr(telegram.Bot, method)).parameters
        unknown = [k for k in kwargs if k not in params]
        if unknown:
            raise TypeError(f"{method}() got unexpected keyword argument(s): {unknown}")
        missing = [
            n
            for n, p in params.items()
            if n != "self" and p.default is inspect.Parameter.empty and n not in kwargs
        ]
        if method == "edit_message_text":
            # Real API: (chat_id and message_id) or inline_message_id.
            missing = [m for m in missing if m != "text"]
            has_chat = "chat_id" in kwargs and "message_id" in kwargs
            if "text" not in kwargs:
                missing.append("text")
            if not has_chat and "inline_message_id" not in kwargs:
                missing.append("chat_id+message_id or inline_message_id")
        if missing:
            raise TypeError(f"{method}() missing required argument(s): {missing}")
        self.calls.append((method, kwargs))
        queue = self._failures.get(method)
        if queue:
            raise queue.pop(0)

    async def send_message(self, **kwargs: Any) -> SimpleNamespace:
        self._record("send_message", kwargs)
        return SimpleNamespace(
            message_id=next(self._ids), chat_id=kwargs["chat_id"], text=kwargs["text"]
        )

    async def edit_message_text(self, **kwargs: Any) -> SimpleNamespace:
        self._record("edit_message_text", kwargs)
        return SimpleNamespace(
            message_id=kwargs.get("message_id", next(self._ids)),
            text=kwargs["text"],
        )

    async def set_message_reaction(self, **kwargs: Any) -> bool:
        self._record("set_message_reaction", kwargs)
        return True

    async def answer_callback_query(self, **kwargs: Any) -> bool:
        self._record("answer_callback_query", kwargs)
        return True
