"""The awkward cases: late answers, partial capability, and overhead.

Three things a circuit breaker gets wrong quietly if nobody tests it.

A **late response** is the one that produces phantom state — an answer
arriving after everybody stopped waiting, folded into a circuit that has
moved on since. A **degraded capability** is where "keep serving" and
"serve what was asked for" pull in opposite directions. And **overhead**
is what turns a safety device into the thing that made everything slow.
"""

from __future__ import annotations

import statistics
import time

from resilience_fixtures import (
    ALL_CAPABILITIES,
    TEXT_ONLY_CAPABILITIES,
    Clock,
    profile,
)

from luber_provider_resilience import (
    Capability,
    CapabilityStatus,
    CircuitIdentity,
    CircuitPolicy,
    CircuitState,
    FailoverMode,
    InMemoryCircuitStore,
    RequestNeeds,
    ResilienceManager,
    RoutingOutcome,
    RoutingPolicy,
)

TEXT = RequestNeeds(task_type="TEXT_TO_MUSIC", duration_seconds=180.0)
REFERENCE = RequestNeeds(task_type="REFERENCE_CONDITIONED", has_reference=True)
TEXT_CIRCUIT = CircuitIdentity("provider_a", "TEXT_TO_MUSIC")
REFERENCE_CIRCUIT = CircuitIdentity("provider_a", "REFERENCE_CONDITIONED")


def build(
    failover: str = FailoverMode.SAFE_EQUIVALENT_ONLY.value,
    *,
    both: bool = True,
    **policy,
):
    clock = Clock()
    profiles = [profile("provider_a", ALL_CAPABILITIES)]
    if both:
        profiles.append(profile("provider_b", TEXT_ONLY_CAPABILITIES))
    manager = ResilienceManager(
        profiles,
        store=InMemoryCircuitStore(),
        circuit_policy=CircuitPolicy(
            consecutive_failure_threshold=3,
            probe_successes_to_close=2,
            probe_concurrency=1,
        ),
        routing_policy=RoutingPolicy(failover=failover, **policy),
        clock=clock,
    )
    return manager, clock


async def kill(manager, clock, needs, provider="provider_a"):
    pinned = RequestNeeds(**{**needs.__dict__, "requested_provider": provider})
    for _ in range(3):
        decision = await manager.router.select(pinned, probe_token="t")
        if decision.permitted:
            await manager.record(decision, succeeded=False, error_code="GENERATION_TIMEOUT")
        clock.advance(seconds=1)


# ── late responses ───────────────────────────────────────────────────


async def test_an_abandoned_probe_frees_its_slot_when_the_lease_expires():
    """A worker that dies mid-probe must not wedge the circuit.

    The probe slot is held by a lease, not by a promise to give it back.
    Without expiry, one killed worker leaves HALF_OPEN with its single
    slot taken and no probe will ever run again — a circuit that can
    never close, from a provider that recovered minutes ago.
    """
    manager, clock = build(both=False)
    await kill(manager, clock, TEXT)

    clock.advance(seconds=31)
    first = await manager.route(TEXT)
    assert first.outcome == RoutingOutcome.SELECTED_AS_PROBE.value

    # The worker holding it disappears. Nothing is ever recorded.
    blocked = await manager.route(TEXT)
    assert blocked.outcome == RoutingOutcome.PROBE_SLOTS_TAKEN.value

    clock.advance(minutes=6)
    after = await manager.route(TEXT)
    assert after.outcome == RoutingOutcome.SELECTED_AS_PROBE.value, "the lease expired"


async def test_a_late_failure_cannot_reopen_a_circuit_a_later_probe_closed():
    """The answer nobody was waiting for any more.

    Probe A's provider never answers, the lease expires, probe B runs and
    closes the circuit — and then A's failure finally arrives. It is real
    evidence and is recorded as such, but it is one sample against a
    closed circuit, not a probe failure against a HALF_OPEN one. Treating
    it as the latter would reopen a circuit on the strength of a request
    that was abandoned before the recovery even started.
    """
    manager, clock = build(both=False)
    await kill(manager, clock, TEXT)
    clock.advance(seconds=31)

    abandoned = await manager.route(TEXT)
    assert abandoned.outcome == RoutingOutcome.SELECTED_AS_PROBE.value

    clock.advance(minutes=6)
    for _ in range(2):
        probe = await manager.route(TEXT)
        assert probe.outcome == RoutingOutcome.SELECTED_AS_PROBE.value
        await manager.record(probe, succeeded=True, latency_seconds=1.0)
        clock.advance(seconds=1)

    assert (await manager.store.load(TEXT_CIRCUIT)).state == CircuitState.CLOSED.value

    # ... and now the abandoned attempt answers.
    await manager.record(abandoned, succeeded=False, error_code="GENERATION_TIMEOUT")

    record = await manager.store.load(TEXT_CIRCUIT)
    assert record.state == CircuitState.CLOSED.value
    assert record.consecutive_failures == 1, "counted once, as one ordinary failure"


async def test_releasing_a_probe_token_twice_is_harmless():
    """A retry of the release path must not free somebody else's slot."""
    manager, clock = build(both=False)
    await kill(manager, clock, TEXT)
    clock.advance(seconds=31)

    probe = await manager.route(TEXT)
    await manager.abandon(probe)
    await manager.abandon(probe)

    record = await manager.store.load(TEXT_CIRCUIT)
    assert record.active_probes(clock.now) == 0


# ── degraded capability ──────────────────────────────────────────────


async def test_one_broken_capability_does_not_take_down_the_others():
    """Circuits are per task, so a broken cover path still serves text."""
    manager, clock = build(both=False)
    await kill(manager, clock, REFERENCE)

    refused = await manager.route(REFERENCE)
    assert not refused.permitted

    still_serving = await manager.route(TEXT)
    assert still_serving.permitted, "text-to-music never failed and is unaffected"


async def test_a_reference_request_is_refused_rather_than_stripped():
    """The whole no-silent-degradation rule, in one assertion.

    Provider B is up and would happily generate something. It cannot
    take a reference track, so sending this request there would deliver
    a song built from the prompt alone — plausible, wrong, and silent.

    The refusal names the open circuit rather than the missing
    capability, and that is the right way round: B was never a candidate
    for this request, so the only thing standing between the user and
    their song is A being down. It also makes the refusal retryable,
    which "no equivalent provider" is not — A may be half-open in thirty
    seconds, whereas B will never grow a reference input.
    """
    manager, clock = build()
    await kill(manager, clock, REFERENCE)

    decision = await manager.route(REFERENCE)

    assert decision.selected is None, "not provider_b, and not anybody"
    assert decision.outcome == RoutingOutcome.PROVIDER_UNAVAILABLE_CIRCUIT_OPEN.value
    assert "provider_a" in decision.reason

    # And the same deployment routes a request B *can* serve.
    plain = await manager.route(REFERENCE.__class__(task_type="TEXT_TO_MUSIC"))
    assert plain.permitted


async def test_a_degraded_deployment_reports_which_parts_are_down():
    manager, clock = build(both=False)
    await kill(manager, clock, REFERENCE)

    report = await manager.readiness()

    assert report.generation_available, "text-to-music still works"
    assert report.degraded
    assert report.status_of(Capability.TEXT_TO_MUSIC.value) == CapabilityStatus.AVAILABLE.value
    assert (
        report.status_of(Capability.REFERENCE_CONDITIONED.value)
        == CapabilityStatus.UNAVAILABLE.value
    )
    assert Capability.REFERENCE_CONDITIONED.value in report.summary
    assert report.metrics["circuits_open"] == 1


async def test_a_recovering_capability_reads_as_degraded_not_available():
    """HALF_OPEN is not 'up'. Most requests are still refused."""
    manager, clock = build(both=False)
    await kill(manager, clock, TEXT)
    clock.advance(seconds=31)
    await manager.route(TEXT)  # promotes to HALF_OPEN

    report = await manager.readiness()

    assert report.status_of(Capability.TEXT_TO_MUSIC.value) == CapabilityStatus.DEGRADED.value
    assert "refused" in report.render()


async def test_a_capability_nobody_offers_is_not_reported_as_broken():
    """NOT_CONFIGURED and UNAVAILABLE mean different things.

    A deployment without a cover-capable provider is not in an incident.
    Collapsing the two would page somebody for a feature that was never
    installed.
    """
    manager, _ = build(both=False)
    manager.profiles = [profile("provider_a", TEXT_ONLY_CAPABILITIES)]
    manager.router.profiles = manager.profiles

    report = await manager.readiness()

    assert report.status_of(Capability.COVER.value) == CapabilityStatus.NOT_CONFIGURED.value
    assert not report.degraded, "nothing is broken"
    assert report.generation_available


async def test_everything_down_is_unavailable_rather_than_quietly_empty():
    manager, clock = build(both=False)
    for needs in (TEXT, REFERENCE):
        await kill(manager, clock, needs)
    for task in ("EXTEND", "REPLACE_RANGE", "COVER"):
        await kill(manager, clock, RequestNeeds(task_type=task))

    report = await manager.readiness()

    assert not report.generation_available
    assert not report.degraded, "degraded means partly working"
    assert "no capability can be served" in report.summary


# ── overhead ─────────────────────────────────────────────────────────


async def test_the_healthy_path_costs_almost_nothing():
    """Routing a request whose provider is fine is a dictionary lookup
    and a comparison, and it stays that way.

    The bound is deliberately loose — a hundred times the observed cost
    on this machine — because a benchmark that fails on a loaded CI box
    gets deleted, and a benchmark that catches a routing rewrite doing
    per-request I/O is worth keeping. The number that matters is the one
    printed, not the threshold.
    """
    manager, clock = build(both=False)

    samples: list[float] = []
    for _ in range(200):
        started = time.perf_counter()
        decision = await manager.route(TEXT)
        await manager.record(decision, succeeded=True, latency_seconds=1.0)
        samples.append((time.perf_counter() - started) * 1000)
        clock.advance(seconds=1)

    median = statistics.median(samples)
    worst = max(samples)
    print(f"\nfast-path route+record: median {median:.3f}ms, worst {worst:.3f}ms")

    assert median < 5.0, f"the fast path costs {median:.3f}ms per generation"


async def test_the_window_does_not_grow_without_bound():
    """The rolling window is pruned, so a long-lived circuit under
    steady traffic keeps a bounded record rather than every outcome
    since the process started."""
    manager, clock = build(both=False)

    for _ in range(500):
        decision = await manager.route(TEXT)
        await manager.record(decision, succeeded=True, latency_seconds=1.0)
        clock.advance(seconds=1)

    record = await manager.store.load(TEXT_CIRCUIT)
    assert len(record.window) <= 301, f"window held {len(record.window)} outcomes"
