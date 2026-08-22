"""Circuits against a real database: shared, durable, and race-safe.

The in-memory store is a complete implementation and every state-machine
test runs on it. These cover the half it cannot: that the compare-and-set
is genuinely atomic, that a restart does not forget an outage, and that
two workers converge rather than each keeping a private opinion.

The concurrency tests use a file-backed SQLite database on purpose. A
shared in-memory connection serialises everything and would make the
race look correct whatever the code did; separate connections are what a
multi-process deployment actually looks like.
"""

from __future__ import annotations

import threading
from datetime import timedelta

import pytest
from resilience_fixtures import ALL_CAPABILITIES, Clock, profile
from sqlalchemy.ext.asyncio import create_async_engine

from luber_database import Base, ResilienceRepository
from luber_provider_resilience import (
    CircuitIdentity,
    CircuitPolicy,
    CircuitState,
    DurableCircuitStore,
    RequestNeeds,
    ResilienceManager,
    RoutingPolicy,
)

NEEDS = RequestNeeds(task_type="TEXT_TO_MUSIC")
IDENTITY = CircuitIdentity("provider_a", "TEXT_TO_MUSIC")


@pytest.fixture
async def engine(tmp_path):
    """A file-backed database: each session gets its own connection."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/resilience.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


def manager_for(engine, clock, policy: CircuitPolicy | None = None) -> ResilienceManager:
    """One worker's view of the shared circuits."""
    return ResilienceManager(
        [profile("provider_a", ALL_CAPABILITIES)],
        store=DurableCircuitStore(ResilienceRepository(engine)),
        circuit_policy=policy or CircuitPolicy(consecutive_failure_threshold=3),
        routing_policy=RoutingPolicy(),
        clock=clock,
    )


async def drive(manager, clock, count: int):
    for _ in range(count):
        decision = await manager.route(NEEDS)
        if decision.permitted:
            await manager.record(decision, succeeded=False, error_code="GENERATION_TIMEOUT")
        clock.advance(seconds=1)


async def test_a_circuit_round_trips_through_the_database(engine):
    clock = Clock()
    manager = manager_for(engine, clock)
    await drive(manager, clock, 3)

    stored = await ResilienceRepository(engine).load(IDENTITY.key())

    assert stored is not None
    assert stored["state"] == CircuitState.OPEN.value
    assert stored["consecutive_failures"] == 3
    assert stored["open_until"] is not None
    assert stored["open_until"].tzinfo is not None, "timestamps come back aware"
    assert stored["window"], "the evidence survives the round trip"


async def test_two_workers_converge_on_one_open_circuit(engine):
    """Worker A giving up and worker B carrying on for four more minutes
    is the failure durable state exists to prevent."""
    clock = Clock()
    first, second = manager_for(engine, clock), manager_for(engine, clock)

    for index in range(3):
        worker = first if index % 2 == 0 else second
        decision = await worker.route(NEEDS)
        if decision.permitted:
            await worker.record(decision, succeeded=False, error_code="GENERATION_TIMEOUT")
        clock.advance(seconds=1)

    assert (await first.store.load(IDENTITY)).state == CircuitState.OPEN.value
    assert (await second.store.load(IDENTITY)).state == CircuitState.OPEN.value
    assert not (await second.route(NEEDS)).permitted


async def test_an_open_circuit_survives_a_restart(engine):
    clock = Clock()
    await drive(manager_for(engine, clock), clock, 3)

    # A new process: fresh manager, fresh store, same database.
    restarted = manager_for(engine, Clock(clock.now))

    assert (await restarted.store.load(IDENTITY)).state == CircuitState.OPEN.value
    assert not (await restarted.route(NEEDS)).permitted


async def test_a_restart_after_the_cooldown_probes_rather_than_stays_open(engine):
    clock = Clock()
    await drive(manager_for(engine, clock), clock, 3)

    later = Clock(clock.now + timedelta(seconds=31))
    restarted = manager_for(engine, later)

    assert (await restarted.route(NEEDS)).outcome == "SELECTED_AS_PROBE"


async def test_the_transition_history_survives(engine):
    clock = Clock()
    manager = manager_for(engine, clock)
    await drive(manager, clock, 3)
    clock.advance(seconds=31)
    await manager.route(NEEDS)

    history = await ResilienceRepository(engine).transitions(circuit_key=IDENTITY.key())

    assert [(item["previous_state"], item["current_state"]) for item in history] == [
        ("OPEN", "HALF_OPEN"),
        ("CLOSED", "OPEN"),
    ]


async def test_a_manual_override_is_auditable_after_the_fact(engine):
    clock = Clock()
    manager = manager_for(engine, clock)
    await manager.open(IDENTITY, operator="alex", reason="draining for maintenance")

    history = await ResilienceRepository(engine).transitions(circuit_key=IDENTITY.key())

    assert history[0]["automatic"] is False
    assert history[0]["operator"] == "alex"
    assert "draining" in history[0]["reason"]


async def test_a_stale_write_is_refused(engine):
    """The compare-and-set itself: a caller holding an old revision
    cannot overwrite a newer one."""
    from luber_database.resilience_repository import CircuitConflict

    repository = ResilienceRepository(engine)
    clock = Clock()
    manager = manager_for(engine, clock)
    await drive(manager, clock, 1)

    stale = await repository.load(IDENTITY.key())
    await drive(manager, clock, 1)

    with pytest.raises(CircuitConflict):
        await repository.save(stale, expected_revision=stale["revision"])


def test_concurrent_failures_produce_exactly_one_transition(tmp_path):
    """Sixteen workers cross the threshold at the same instant.

    Run through threads with a real barrier, against separate database
    connections. A read-modify-write would let several of them each
    record "the circuit opened"; the conditional UPDATE means one does.
    """
    import asyncio

    async def scenario() -> tuple[str, int, list[str]]:
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/race.db")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        repository = ResilienceRepository(engine)
        clock = Clock()
        workers = [manager_for(engine, clock) for _ in range(16)]

        # Prime with one short of the threshold so the burst crosses it.
        primer = manager_for(engine, clock, CircuitPolicy(consecutive_failure_threshold=5))
        for _ in range(4):
            decision = await primer.route(NEEDS)
            await primer.record(decision, succeeded=False, error_code="GENERATION_TIMEOUT")

        errors: list[str] = []
        barrier = threading.Barrier(len(workers))

        def hammer(worker: ResilienceManager) -> None:
            async def run() -> None:
                barrier.wait()
                decision = await worker.route(NEEDS)
                if decision.permitted:
                    await worker.record(decision, succeeded=False, error_code="GENERATION_TIMEOUT")

            try:
                asyncio.run(run())
            except Exception as exc:
                errors.append(repr(exc))

        threads = [threading.Thread(target=hammer, args=(worker,)) for worker in workers]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        stored = await repository.load(IDENTITY.key())
        history = await repository.transitions(circuit_key=IDENTITY.key(), limit=100)
        opens = sum(1 for item in history if item["current_state"] == "OPEN")
        await engine.dispose()
        return stored["state"], opens, errors

    state, opens, errors = asyncio.run(scenario())

    assert state == CircuitState.OPEN.value
    assert opens == 1, "one opening, not one per worker that noticed"
    assert errors == [], "contention is the mechanism working, not an error"
