"""Where a request goes, and the three ways it may not be moved.

Failover is most tempting exactly when things are going badly, which is
when substituting quietly does the most damage: the user gets a song
they did not ask for and nothing anywhere says so. So most of this file
is about refusals — a request that cannot be served the way it was made
fails, and names what was missing.
"""

from __future__ import annotations

from resilience_fixtures import ALL_CAPABILITIES, TEXT_ONLY_CAPABILITIES, Clock, profile

from luber_provider_resilience import (
    Capability,
    CircuitPolicy,
    FailoverMode,
    InMemoryCircuitStore,
    RequestNeeds,
    ResilienceManager,
    RoutingOutcome,
    RoutingPolicy,
    check_equivalence,
)

TEXT = RequestNeeds(task_type="TEXT_TO_MUSIC", duration_seconds=180.0)
COVER = RequestNeeds(task_type="COVER")
REFERENCE = RequestNeeds(task_type="REFERENCE_CONDITIONED", has_reference=True)


def build(failover: str = FailoverMode.SAFE_EQUIVALENT_ONLY.value, **policy):
    """Provider A does everything; provider B cannot take a reference or a cover."""
    clock = Clock()
    manager = ResilienceManager(
        [profile("provider_a", ALL_CAPABILITIES), profile("provider_b", TEXT_ONLY_CAPABILITIES)],
        store=InMemoryCircuitStore(),
        circuit_policy=CircuitPolicy(consecutive_failure_threshold=3),
        routing_policy=RoutingPolicy(failover=failover, **policy),
        clock=clock,
    )
    return manager, clock


async def kill(manager, clock, needs, provider="provider_a"):
    """Drive one provider's circuit open for one task."""
    pinned = RequestNeeds(**{**needs.__dict__, "requested_provider": provider})
    for _ in range(3):
        decision = await manager.router.select(pinned, probe_token="t")
        if decision.permitted:
            await manager.record(decision, succeeded=False, error_code="GENERATION_TIMEOUT")
        clock.advance(seconds=1)


# ── the healthy path ─────────────────────────────────────────────────


async def test_a_healthy_request_goes_to_the_preferred_provider():
    manager, _ = build()
    decision = await manager.route(TEXT)

    assert decision.selected == "provider_a"
    assert decision.fallback_used is False
    assert decision.circuit_state == "CLOSED"


async def test_preference_order_is_honoured():
    manager, _ = build(preference=("provider_b", "provider_a"))
    assert (await manager.route(TEXT)).selected == "provider_b"


# ── safe failover ────────────────────────────────────────────────────


async def test_an_equivalent_provider_takes_over_when_the_first_is_open():
    manager, clock = build()
    await kill(manager, clock, TEXT)

    decision = await manager.route(TEXT)

    assert decision.selected == "provider_b"
    assert decision.fallback_used is True
    assert any(
        item.provider == "provider_a" and not item.eligible for item in decision.considered
    ), "the trace must show why the first choice was passed over"


async def test_failover_is_off_by_default():
    """Another provider existing is not a reason to use it."""
    manager, clock = build(failover=FailoverMode.DISABLED.value)
    await kill(manager, clock, TEXT)

    decision = await manager.route(TEXT)

    assert not decision.permitted
    assert decision.selected is None
    assert "failover is disabled" in "; ".join(item.reason for item in decision.considered)


# ── unsafe failover ──────────────────────────────────────────────────


async def test_a_cover_is_not_moved_to_a_provider_that_cannot_do_covers():
    manager, clock = build()
    await kill(manager, clock, COVER)

    decision = await manager.route(COVER)

    assert not decision.permitted
    assert decision.selected != "provider_b"


async def test_a_reference_conditioned_request_is_never_stripped_of_its_reference():
    """The worst available outcome: a song that ignores the reference,
    delivered as a success."""
    manager, clock = build()
    await kill(manager, clock, REFERENCE)

    decision = await manager.route(REFERENCE)

    assert not decision.permitted
    assert decision.selected is None
    rejected = [item for item in decision.considered if item.provider == "provider_b"]
    assert rejected and rejected[0].equivalence is not None
    assert Capability.REFERENCE_CONDITIONED.value in rejected[0].equivalence["missing"]


def test_equivalence_names_every_missing_capability_not_just_the_first():
    verdict = check_equivalence(
        profile("provider_b", TEXT_ONLY_CAPABILITIES),
        RequestNeeds(task_type="COVER", has_reference=True),
    )
    assert not verdict.equivalent
    assert len(verdict.missing) >= 2
    assert verdict.explain()


def test_a_duration_outside_a_providers_range_is_not_equivalent():
    limited = profile("small", ALL_CAPABILITIES)
    limited = type(limited)(
        name="small",
        capabilities=ALL_CAPABILITIES,
        revision="test",
        maximum_duration_seconds=60.0,
    )
    verdict = check_equivalence(
        limited, RequestNeeds(task_type="TEXT_TO_MUSIC", duration_seconds=240.0)
    )
    assert not verdict.equivalent
    assert "maximum" in verdict.explain()


def test_an_unknown_task_type_is_refused_rather_than_guessed():
    verdict = check_equivalence(
        profile("provider_a", ALL_CAPABILITIES), RequestNeeds(task_type="SOMETHING_NEW")
    )
    assert not verdict.equivalent


# ── explicit provider ────────────────────────────────────────────────


async def test_an_explicitly_named_provider_is_never_substituted():
    """Silently sending the request elsewhere would answer a different
    question and report success."""
    manager, clock = build()
    await kill(manager, clock, TEXT)

    decision = await manager.route(
        RequestNeeds(task_type="TEXT_TO_MUSIC", requested_provider="provider_a")
    )

    assert decision.outcome == RoutingOutcome.EXPLICIT_PROVIDER_UNAVAILABLE.value
    assert decision.selected is None


async def test_an_explicit_provider_may_be_substituted_when_that_is_permitted():
    manager, clock = build(allow_failover_from_explicit=True)
    await kill(manager, clock, TEXT)

    decision = await manager.route(
        RequestNeeds(task_type="TEXT_TO_MUSIC", requested_provider="provider_a")
    )

    assert decision.selected == "provider_b"
    assert decision.fallback_used is True


async def test_naming_a_provider_that_is_not_configured_is_a_typed_refusal():
    manager, _ = build()
    decision = await manager.route(
        RequestNeeds(task_type="TEXT_TO_MUSIC", requested_provider="nonexistent")
    )
    assert decision.outcome == RoutingOutcome.PROVIDER_NOT_CONFIGURED.value


# ── budget ───────────────────────────────────────────────────────────


async def test_the_failover_budget_bounds_how_many_providers_are_touched():
    manager, clock = build(maximum_providers_per_generation=2)
    await kill(manager, clock, TEXT)

    decision = await manager.route(TEXT, attempted=("provider_a", "provider_b"))

    assert decision.outcome == RoutingOutcome.FAILOVER_BUDGET_EXHAUSTED.value
    assert "already tried 2 providers" in decision.reason


async def test_retrying_the_same_provider_is_not_a_failover():
    """Re-selecting a provider this generation already used is a quality
    retry — Phase 29's business — and must not consume failover budget."""
    manager, _ = build()
    decision = await manager.route(TEXT, attempted=("provider_a",))

    assert decision.selected == "provider_a"
    assert decision.fallback_used is False


# ── the trace ────────────────────────────────────────────────────────


async def test_every_decision_records_what_it_considered_and_why():
    manager, clock = build()
    await kill(manager, clock, TEXT)
    decision = await manager.route(TEXT)

    payload = decision.to_dict()

    assert payload["selected_provider"] == "provider_b"
    assert payload["fallback_used"] is True
    assert payload["circuit_policy_version"]
    assert payload["routing_policy_version"]
    assert payload["failover_policy_version"]
    assert len(payload["considered"]) == 2
    assert decision.explain()


async def test_a_refusal_explains_itself_in_a_sentence():
    manager, clock = build(failover=FailoverMode.DISABLED.value)
    await kill(manager, clock, TEXT)
    decision = await manager.route(TEXT)

    assert "no provider selected" in decision.explain()
    assert decision.reason


# ── grouping guard ───────────────────────────────────────────────────


async def test_a_provider_with_no_circuit_history_is_simply_closed():
    """Adding a provider needs no registration step."""
    manager, _ = build()
    assert (await manager.route(TEXT)).permitted
    assert await manager.circuits() == []
