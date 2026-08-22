"""The breaker: when it opens, and — more often — when it must not.

Most of these assert restraint. A breaker that opens too readily turns a
transient blip into a self-inflicted outage, and an operator who has
seen that once stops trusting the thing that was supposed to help. So
the tests for "does not open" outnumber the tests for "opens", and they
are the ones to read first.
"""

from __future__ import annotations

import pytest
from resilience_fixtures import ALL_CAPABILITIES, Clock, profile

from luber_provider_resilience import (
    CircuitIdentity,
    CircuitPolicy,
    CircuitState,
    ControlMode,
    FailureCategory,
    InMemoryCircuitStore,
    RequestNeeds,
    ResilienceManager,
    RoutingPolicy,
)

NEEDS = RequestNeeds(task_type="TEXT_TO_MUSIC", duration_seconds=180.0)
IDENTITY = CircuitIdentity("provider_a", "TEXT_TO_MUSIC")


def build(policy: CircuitPolicy | None = None, routing: RoutingPolicy | None = None):
    clock = Clock()
    manager = ResilienceManager(
        [profile("provider_a")],
        store=InMemoryCircuitStore(),
        circuit_policy=policy or CircuitPolicy(),
        routing_policy=routing or RoutingPolicy(),
        clock=clock,
    )
    return manager, clock


async def drive(manager, clock, count: int, **failure):
    """Push *count* failures through, one second apart."""
    for _ in range(count):
        decision = await manager.route(NEEDS)
        if decision.permitted:
            await manager.record(decision, succeeded=False, **failure)
        clock.advance(seconds=1)


async def state(manager) -> str:
    return (await manager.store.load(IDENTITY)).state


# ── it must not open ─────────────────────────────────────────────────


async def test_one_failure_does_not_open_the_circuit():
    """One timeout is one timeout. Opening on it would convert every
    transient blip into a self-inflicted outage."""
    manager, clock = build()
    await drive(manager, clock, 1, error_code="GENERATION_TIMEOUT")
    assert await state(manager) == CircuitState.CLOSED.value


async def test_failures_below_the_threshold_do_not_open_it():
    manager, clock = build(CircuitPolicy(consecutive_failure_threshold=5))
    await drive(manager, clock, 4, error_code="GENERATION_TIMEOUT")
    assert await state(manager) == CircuitState.CLOSED.value


async def test_a_success_resets_the_consecutive_count():
    """Four failures, a success, four more. Never five in a row."""
    manager, clock = build(CircuitPolicy(consecutive_failure_threshold=5))
    await drive(manager, clock, 4, error_code="GENERATION_TIMEOUT")
    decision = await manager.route(NEEDS)
    await manager.record(decision, succeeded=True)
    await drive(manager, clock, 4, error_code="GENERATION_TIMEOUT")
    assert await state(manager) == CircuitState.CLOSED.value


@pytest.mark.parametrize(
    ("label", "failure"),
    [
        ("a reference the provider could not fetch", {"error_code": "REFERENCE_AUDIO_UNAVAILABLE"}),
        ("a request that was refused as malformed", {"status_code": 400}),
    ],
)
async def test_a_user_error_never_poisons_provider_health(label, failure):
    """The rule that stops one bad client taking the model offline for
    everybody. The next request from somebody else would have worked."""
    manager, clock = build()
    await drive(manager, clock, 20, **failure)

    record = await manager.store.load(IDENTITY)
    assert record.state == CircuitState.CLOSED.value, label
    assert record.sample_count() == 0, "user errors must not even be recorded as evidence"


async def test_cancellation_never_poisons_provider_health():
    """A UI change that made cancelling easier must not look like the
    model breaking."""
    manager, clock = build()
    await drive(manager, clock, 20, cancelled=True)

    record = await manager.store.load(IDENTITY)
    assert record.state == CircuitState.CLOSED.value
    assert record.sample_count() == 0


async def test_a_quality_rejection_never_opens_a_circuit():
    """Phase 29 refusing audio means the provider answered. A circuit is
    an availability device, and how a song sounds is a different axis."""
    manager, clock = build()
    await drive(manager, clock, 20, error_code="QUALITY_CHECK_FAILED")
    assert await state(manager) == CircuitState.CLOSED.value


async def test_a_local_failure_after_the_provider_answered_does_not_count():
    manager, clock = build()
    await drive(manager, clock, 20, error_code="UPLOAD_FAILED")
    assert await state(manager) == CircuitState.CLOSED.value


# ── it must open ─────────────────────────────────────────────────────


async def test_repeated_provider_failures_open_the_circuit():
    manager, clock = build(CircuitPolicy(consecutive_failure_threshold=5))
    await drive(manager, clock, 5, error_code="GENERATION_TIMEOUT")

    record = await manager.store.load(IDENTITY)
    assert record.state == CircuitState.OPEN.value
    assert "5 consecutive" in (record.open_reason or "")
    assert record.open_until is not None, "an open circuit must expire"


async def test_an_open_circuit_refuses_traffic_without_calling_the_provider():
    manager, clock = build(CircuitPolicy(consecutive_failure_threshold=3))
    await drive(manager, clock, 3, error_code="GENERATION_TIMEOUT")

    decision = await manager.route(NEEDS)

    assert not decision.permitted
    assert decision.outcome == "PROVIDER_UNAVAILABLE_CIRCUIT_OPEN"
    assert decision.selected is None


async def test_a_failure_rate_opens_a_busy_circuit_without_a_streak():
    """A provider failing half its requests is not serving, even if the
    failures never line up five in a row."""
    policy = CircuitPolicy(
        consecutive_failure_threshold=99, minimum_samples=10, failure_rate_threshold=0.5
    )
    manager, clock = build(policy)
    for index in range(20):
        decision = await manager.route(NEEDS)
        if not decision.permitted:
            break
        await manager.record(
            decision,
            succeeded=index % 2 == 0,
            error_code=None if index % 2 == 0 else "GENERATION_TIMEOUT",
        )
        clock.advance(seconds=1)

    record = await manager.store.load(IDENTITY)
    assert record.state == CircuitState.OPEN.value
    assert record.open_evidence["rule"] == "failure_rate"


async def test_a_rate_is_not_computed_from_too_few_samples():
    """Two failures out of three is not a 67% failure rate; it is three
    requests."""
    policy = CircuitPolicy(
        consecutive_failure_threshold=99, minimum_samples=10, failure_rate_threshold=0.5
    )
    manager, clock = build(policy)
    await drive(manager, clock, 3, error_code="GENERATION_TIMEOUT")
    assert await state(manager) == CircuitState.CLOSED.value


async def test_a_rate_limit_counts_less_than_a_crash():
    """A provider declining politely is less broken than one that cannot
    answer at all."""
    policy = CircuitPolicy(rate_limit_weight=0.5, minimum_samples=4)
    manager, clock = build(policy)
    for _ in range(4):
        decision = await manager.route(NEEDS)
        await manager.record(decision, succeeded=False, status_code=429)
        clock.advance(seconds=1)

    record = await manager.store.load(IDENTITY)
    assert record.last_failure_category == FailureCategory.PROVIDER_RATE_LIMIT.value
    assert record.failure_rate(policy) == pytest.approx(0.5)


# ── recovery ─────────────────────────────────────────────────────────


async def test_the_cooldown_expires_into_half_open():
    manager, clock = build(CircuitPolicy(consecutive_failure_threshold=3))
    await drive(manager, clock, 3, error_code="GENERATION_TIMEOUT")
    assert await state(manager) == CircuitState.OPEN.value

    clock.advance(seconds=31)
    decision = await manager.route(NEEDS)

    assert decision.outcome == "SELECTED_AS_PROBE"
    assert decision.probe_token is not None
    assert await state(manager) == CircuitState.HALF_OPEN.value


async def test_only_the_configured_number_of_probes_is_admitted():
    """Twenty requests arrive at a circuit that has just become
    probeable. One finds out whether the provider is back; the rest are
    not joining the stampede."""
    manager, clock = build(CircuitPolicy(consecutive_failure_threshold=3, probe_concurrency=1))
    await drive(manager, clock, 3, error_code="GENERATION_TIMEOUT")
    clock.advance(seconds=31)

    outcomes = [(await manager.route(NEEDS)).outcome for _ in range(20)]

    assert outcomes.count("SELECTED_AS_PROBE") == 1
    assert outcomes.count("PROBE_SLOTS_TAKEN") == 19


async def test_two_probe_slots_admit_two():
    manager, clock = build(CircuitPolicy(consecutive_failure_threshold=3, probe_concurrency=2))
    await drive(manager, clock, 3, error_code="GENERATION_TIMEOUT")
    clock.advance(seconds=31)

    outcomes = [(await manager.route(NEEDS)).outcome for _ in range(20)]

    assert outcomes.count("SELECTED_AS_PROBE") == 2


async def test_one_lucky_probe_does_not_close_the_circuit():
    manager, clock = build(
        CircuitPolicy(consecutive_failure_threshold=3, probe_successes_to_close=2)
    )
    await drive(manager, clock, 3, error_code="GENERATION_TIMEOUT")
    clock.advance(seconds=31)

    decision = await manager.route(NEEDS)
    await manager.record(decision, succeeded=True)

    assert await state(manager) == CircuitState.HALF_OPEN.value


async def test_enough_probe_successes_close_the_circuit():
    manager, clock = build(
        CircuitPolicy(consecutive_failure_threshold=3, probe_successes_to_close=2)
    )
    await drive(manager, clock, 3, error_code="GENERATION_TIMEOUT")
    clock.advance(seconds=31)

    for _ in range(2):
        decision = await manager.route(NEEDS)
        assert decision.permitted
        await manager.record(decision, succeeded=True)

    assert await state(manager) == CircuitState.CLOSED.value
    assert (await manager.route(NEEDS)).outcome == "SELECTED"


async def test_a_failed_probe_reopens_the_circuit_for_longer():
    """Each re-open doubles the cooldown, so a provider that keeps
    failing its probes is asked less and less often."""
    manager, clock = build(CircuitPolicy(consecutive_failure_threshold=3))
    await drive(manager, clock, 3, error_code="GENERATION_TIMEOUT")
    first_open = await manager.store.load(IDENTITY)
    clock.advance(seconds=31)

    decision = await manager.route(NEEDS)
    await manager.record(decision, succeeded=False, error_code="GENERATION_TIMEOUT")

    reopened = await manager.store.load(IDENTITY)
    assert reopened.state == CircuitState.OPEN.value
    assert reopened.consecutive_opens == 2
    assert reopened.open_until is not None and first_open.open_until is not None
    assert (reopened.open_until - reopened.opened_at) > (
        first_open.open_until - first_open.opened_at
    )


async def test_an_abandoned_probe_returns_its_slot_without_evidence():
    """A cancelled probe learned nothing about the provider, and must
    not leave the only slot held until its lease expires."""
    manager, clock = build(CircuitPolicy(consecutive_failure_threshold=3))
    await drive(manager, clock, 3, error_code="GENERATION_TIMEOUT")
    clock.advance(seconds=31)

    first = await manager.route(NEEDS)
    assert first.outcome == "SELECTED_AS_PROBE"
    await manager.abandon(first)

    second = await manager.route(NEEDS)
    assert second.outcome == "SELECTED_AS_PROBE", "the slot must be free again"


# ── operator override ────────────────────────────────────────────────


async def test_a_manual_open_stops_traffic_and_records_who_and_why():
    manager, _ = build()
    record = await manager.open(IDENTITY, operator="alex", reason="draining for maintenance")

    assert record.state == CircuitState.OPEN.value
    assert record.control == ControlMode.MANUAL.value
    assert record.manual_operator == "alex"
    assert record.manual_reason == "draining for maintenance"
    assert not (await manager.route(NEEDS)).permitted


async def test_a_cooldown_does_not_release_a_manual_pin():
    """A human closed this. A clock does not reopen it."""
    manager, clock = build()
    await manager.open(IDENTITY, operator="alex", reason="draining")
    clock.advance(hours=6)

    assert not (await manager.route(NEEDS)).permitted
    assert await state(manager) == CircuitState.OPEN.value


async def test_a_manual_open_needs_a_reason():
    manager, _ = build()
    with pytest.raises(ValueError, match="needs a reason"):
        await manager.open(IDENTITY, operator="alex", reason="   ")


async def test_a_manual_close_resumes_traffic_and_clears_stale_evidence():
    """Otherwise the very next failure re-opens it on evidence the
    operator has just said no longer applies."""
    manager, clock = build(CircuitPolicy(consecutive_failure_threshold=3))
    await drive(manager, clock, 3, error_code="GENERATION_TIMEOUT")

    record = await manager.close(IDENTITY, operator="alex", reason="provider confirmed healthy")

    assert record.state == CircuitState.CLOSED.value
    assert record.consecutive_failures == 0
    assert record.sample_count() == 0
    assert (await manager.route(NEEDS)).permitted


async def test_evidence_keeps_accumulating_under_a_manual_pin():
    """An operator who pinned a circuit still wants to see whether the
    provider recovered."""
    manager, clock = build()
    await manager.open(IDENTITY, operator="alex", reason="draining")

    decision = await manager.router.select(NEEDS, probe_token="t")
    assert not decision.permitted
    # Force-record against the pinned circuit the way a probe would.
    from luber_provider_resilience.circuit import Outcome, record_outcome

    record = await manager.store.load(IDENTITY)
    updated, transition = record_outcome(
        record,
        Outcome(at=clock.now, succeeded=False, category=FailureCategory.PROVIDER_TIMEOUT.value),
        policy=manager.circuit_policy,
    )
    assert updated.sample_count() == 1, "evidence is still recorded"
    assert transition is None, "but the policy does not move a pinned circuit"


async def test_reset_hands_the_circuit_back_to_the_policy():
    manager, _ = build()
    await manager.open(IDENTITY, operator="alex", reason="draining")

    record = await manager.reset(IDENTITY, operator="alex")

    assert record.control == ControlMode.AUTOMATIC.value
    assert record.state == CircuitState.CLOSED.value
    assert (await manager.route(NEEDS)).permitted


async def test_transitions_are_recorded_for_every_change():
    manager, clock = build(CircuitPolicy(consecutive_failure_threshold=3))
    await drive(manager, clock, 3, error_code="GENERATION_TIMEOUT")
    clock.advance(seconds=31)
    await manager.route(NEEDS)

    history = await manager.store.transitions(limit=10)
    kinds = [(item.previous, item.current) for item in history]

    assert ("CLOSED", "OPEN") in kinds
    assert ("OPEN", "HALF_OPEN") in kinds


# ── identity ─────────────────────────────────────────────────────────


async def test_one_broken_task_does_not_take_the_others_offline():
    """A provider whose cover endpoint is broken can still serve
    text-to-music. A provider-wide circuit would turn a partial outage
    into a total one."""
    clock = Clock()
    manager = ResilienceManager(
        [profile("provider_a", ALL_CAPABILITIES)],
        store=InMemoryCircuitStore(),
        circuit_policy=CircuitPolicy(consecutive_failure_threshold=3),
        routing_policy=RoutingPolicy(),
        clock=clock,
    )
    covers = RequestNeeds(task_type="COVER")
    for _ in range(3):
        decision = await manager.route(covers)
        await manager.record(decision, succeeded=False, error_code="GENERATION_TIMEOUT")
        clock.advance(seconds=1)

    assert not (await manager.route(covers)).permitted
    assert (await manager.route(NEEDS)).permitted, "text-to-music is unaffected"
