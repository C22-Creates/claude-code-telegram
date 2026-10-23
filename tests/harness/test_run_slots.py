"""RunSlots: global run cap with interactive-before-scheduled priority."""

import asyncio

import pytest

from src.harness.run_slots import RunSlots


async def settle(n: int = 5) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


async def hold(slots, priority, log, name, release: asyncio.Event):
    async with slots.acquire(priority):
        log.append(name)
        await release.wait()


def test_cap_must_be_positive():
    with pytest.raises(ValueError):
        RunSlots(0)


def test_unknown_priority_rejected():
    with pytest.raises(ValueError):
        RunSlots(1).acquire("urgent")


async def test_up_to_cap_run_without_waiting():
    slots = RunSlots(2)
    assert not slots.would_wait()
    async with slots.acquire("interactive"):
        assert not slots.would_wait()
        async with slots.acquire("scheduled"):
            assert slots.would_wait()
        assert not slots.would_wait()
    assert slots.in_use == 0


async def test_overflow_waits_until_release():
    slots = RunSlots(1)
    log: list[str] = []
    gate_a, gate_b = asyncio.Event(), asyncio.Event()
    a = asyncio.create_task(hold(slots, "interactive", log, "a", gate_a))
    await settle()
    b = asyncio.create_task(hold(slots, "interactive", log, "b", gate_b))
    await settle()
    assert log == ["a"]
    gate_a.set()
    await settle()
    assert log == ["a", "b"]
    gate_b.set()
    await asyncio.gather(a, b)
    assert slots.in_use == 0


async def test_interactive_served_before_scheduled_fifo_within_class():
    slots = RunSlots(1)
    log: list[str] = []
    gates = {n: asyncio.Event() for n in ("first", "s1", "i1", "s2", "i2")}
    for g in gates.values():
        g.set()
    first_gate = asyncio.Event()
    first = asyncio.create_task(hold(slots, "interactive", log, "first", first_gate))
    await settle()
    tasks = []
    for name, prio in (
        ("s1", "scheduled"),
        ("i1", "interactive"),
        ("s2", "scheduled"),
        ("i2", "interactive"),
    ):
        tasks.append(asyncio.create_task(hold(slots, prio, log, name, gates[name])))
        await settle()
    first_gate.set()
    await asyncio.gather(first, *tasks)
    assert log == ["first", "i1", "i2", "s1", "s2"]
    assert slots.in_use == 0


async def test_new_acquirer_does_not_jump_queued_waiters():
    slots = RunSlots(1)
    log: list[str] = []
    gate = asyncio.Event()
    first = asyncio.create_task(hold(slots, "interactive", log, "first", gate))
    await settle()
    waiter = asyncio.create_task(hold(slots, "scheduled", log, "s", asyncio.Event()))
    await settle()
    gate.set()
    await settle()
    # The slot went straight to the waiter; nobody can sneak in.
    assert log == ["first", "s"]
    assert slots.would_wait()
    waiter.cancel()
    await asyncio.gather(first, waiter, return_exceptions=True)
    assert slots.in_use == 0


async def test_cancelled_waiter_does_not_leak_a_slot():
    slots = RunSlots(1)
    log: list[str] = []
    gate = asyncio.Event()
    first = asyncio.create_task(hold(slots, "interactive", log, "first", gate))
    await settle()
    doomed = asyncio.create_task(hold(slots, "interactive", log, "doomed", gate))
    await settle()
    doomed.cancel()
    await asyncio.gather(doomed, return_exceptions=True)
    gate.set()
    await first
    assert slots.in_use == 0
    assert not slots.would_wait()
    assert log == ["first"]


async def test_waiter_cancelled_after_grant_returns_the_slot():
    """Granted (future resolved) but cancelled before it resumed: no leak."""
    slots = RunSlots(1)
    log: list[str] = []
    holder = slots.acquire("interactive")
    await holder.__aenter__()
    late = asyncio.create_task(hold(slots, "interactive", log, "late", asyncio.Event()))
    await settle()
    # Release (grants late's future) and cancel late without yielding between.
    await holder.__aexit__(None, None, None)
    late.cancel()
    await asyncio.gather(late, return_exceptions=True)
    assert slots.in_use == 0
    assert log == []
