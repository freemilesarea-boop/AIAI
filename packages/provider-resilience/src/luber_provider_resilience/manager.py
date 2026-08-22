"""The one object a caller holds: route, record, and ask what is available.

The pieces below it are deliberately pure — the circuit is a state
machine, the router is a selector, the store is persistence. This ties
them together so a caller does not have to know the order, and so the
order is written down once.

The order matters in one place especially. Recording an outcome may
produce a transition; that transition may deserve an alert and must
bump a counter; and the record must be written before either, because a
transition announced but not persisted is one another worker will
announce again. `apply_with_retry` handles the write and the losing
side, and everything else hangs off what it returns.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from luber_provider_resilience.capabilities import ProviderProfile, RequestNeeds
from luber_provider_resilience.circuit import (
    CircuitIdentity,
    CircuitPolicy,
    CircuitRecord,
    CircuitState,
    Outcome,
    Transition,
    manual_close,
    manual_open,
    record_outcome,
    release_probe,
    reset_to_policy,
)
from luber_provider_resilience.classification import FailureCategory, classify
from luber_provider_resilience.readiness import ReadinessReport, readiness
from luber_provider_resilience.router import (
    ProviderRouter,
    RoutingDecision,
    RoutingOutcome,
    RoutingPolicy,
)
from luber_provider_resilience.store import CircuitStore, apply_with_retry, utcnow
from luber_provider_resilience.telemetry import (
    Counters,
    Metric,
    ResilienceAlert,
    alert_for_transition,
)
from luber_provider_resilience.versions import version_block


@dataclass
class AttemptRecord:
    """One provider attempt, from the resilience layer's point of view.

    Kept beside Phase 29's candidate record rather than inside it: Phase
    29 describes what the *audio* was, this describes what the *call*
    was. Merging them would make a routing question look like a quality
    question on whichever screen showed it.
    """

    attempt: int
    provider: str
    provider_revision: str | None
    outcome: str
    category: str | None
    latency_seconds: float | None
    circuit_before: str
    circuit_after: str
    was_probe: bool
    was_fallback: bool
    at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "provider": self.provider,
            "provider_revision": self.provider_revision,
            "outcome": self.outcome,
            "failure_category": self.category,
            "latency_seconds": self.latency_seconds,
            "circuit_state_before": self.circuit_before,
            "circuit_state_after": self.circuit_after,
            "was_probe": self.was_probe,
            "was_fallback": self.was_fallback,
            "at": self.at.isoformat(),
        }


@dataclass
class ResilienceTrace:
    """Every routing decision and attempt for one generation.

    The answer to "why did this request go where it went", written as it
    happens. A trace assembled at the end would be missing exactly the
    case that needs it: the run that died halfway.
    """

    generation_id: str
    decisions: list[dict[str, Any]]
    attempts: list[AttemptRecord]
    providers_attempted: list[str]
    failovers: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            **version_block(),
            "generation_id": self.generation_id,
            "decisions": self.decisions,
            "attempts": [item.to_dict() for item in self.attempts],
            "providers_attempted": list(self.providers_attempted),
            "provider_failovers": self.failovers,
            "narrative": self.narrative(),
        }

    def narrative(self) -> list[str]:
        """The story, in order, in sentences.

        What an operator reads first. "Provider A timed out, circuit
        evidence updated, failover permitted, provider B succeeded" is
        the whole answer; the structured fields above are for when they
        want to check it.
        """
        lines: list[str] = []
        for decision in self.decisions:
            lines.append(decision.get("explanation", decision.get("outcome", "")))
        for attempt in self.attempts:
            if attempt.outcome == "SUCCEEDED":
                lines.append(f"{attempt.provider} succeeded on attempt {attempt.attempt + 1}")
            else:
                lines.append(
                    f"{attempt.provider} failed on attempt {attempt.attempt + 1}: "
                    f"{attempt.category} "
                    f"(circuit {attempt.circuit_before} → {attempt.circuit_after})"
                )
        return lines


class ResilienceManager:
    """Routing, circuit bookkeeping and readiness, in one place."""

    def __init__(
        self,
        profiles: Sequence[ProviderProfile],
        *,
        store: CircuitStore,
        routing_policy: RoutingPolicy | None = None,
        circuit_policy: CircuitPolicy | None = None,
        counters: Counters | None = None,
        clock: Any = utcnow,
    ) -> None:
        self.profiles = list(profiles)
        self.store = store
        self.circuit_policy = circuit_policy or CircuitPolicy()
        self.counters = counters or Counters()
        self._clock = clock
        self.router = ProviderRouter(
            self.profiles,
            store=store,
            policy=routing_policy or RoutingPolicy(),
            circuit_policy=self.circuit_policy,
            clock=clock,
        )
        self.alerts: list[ResilienceAlert] = []

    # ── routing ──────────────────────────────────────────────────────

    async def route(self, needs: RequestNeeds, *, attempted: Sequence[str] = ()) -> RoutingDecision:
        """Choose a provider, minting a probe token in case one is needed.

        The token is generated up front rather than inside the router so
        the caller always has the value it must return, whether or not
        the slot was used.
        """
        decision = await self.router.select(
            needs, attempted=attempted, probe_token=uuid.uuid4().hex
        )
        self._count_decision(decision, needs)
        return decision

    def _count_decision(self, decision: RoutingDecision, needs: RequestNeeds) -> None:
        if decision.outcome == RoutingOutcome.PROVIDER_UNAVAILABLE_CIRCUIT_OPEN.value:
            self.counters.increment(Metric.REQUESTS_REJECTED_CIRCUIT_OPEN.value)
        elif decision.outcome == RoutingOutcome.PROBE_SLOTS_TAKEN.value:
            self.counters.increment(Metric.PROBE_REFUSED_TOTAL.value)
        elif decision.outcome == RoutingOutcome.SELECTED_AS_PROBE.value:
            self.counters.increment(Metric.PROBE_ADMITTED_TOTAL.value)
        if decision.fallback_used and decision.permitted:
            self.counters.increment(Metric.PROVIDER_FAILOVER_TOTAL.value)

    # ── recording ────────────────────────────────────────────────────

    async def record(
        self,
        decision: RoutingDecision,
        *,
        succeeded: bool,
        error_code: str | None = None,
        status_code: int | None = None,
        cancelled: bool = False,
        timed_out: bool = False,
        transport_error: bool = False,
        latency_seconds: float | None = None,
        attempt: int = 0,
    ) -> AttemptRecord:
        """Fold an attempt's result into the circuit.

        Everything that could poison provider health is filtered by
        `classify` plus `Outcome.counts` rather than by this method
        deciding — the category carries the answer, so a new call site
        cannot get it wrong by omission.
        """
        now = self._clock()
        assert decision.selected is not None, "recording an attempt that never happened"
        identity = CircuitIdentity(decision.selected, decision.needs.task_type)
        before = await self.store.load(identity)

        category = (
            None
            if succeeded
            else classify(
                error_code=error_code,
                status_code=status_code,
                cancelled=cancelled,
                timed_out=timed_out,
                transport_error=transport_error,
            )
        )

        outcome = Outcome(
            at=now,
            succeeded=succeeded,
            category=category,
            latency_seconds=latency_seconds,
            provider_revision=decision.selected_revision,
        )

        def mutate(current: CircuitRecord) -> tuple[CircuitRecord, Transition | None]:
            updated, transition = record_outcome(current, outcome, policy=self.circuit_policy)
            if decision.probe_token is not None:
                # The slot is returned whatever happened, so a failed
                # probe does not leave HALF_OPEN wedged until the lease
                # expires.
                updated = release_probe(updated, token=decision.probe_token, now=now)
            return updated, transition

        # `required=False`: losing the race means another worker
        # recorded the same failure against the same circuit. Raising
        # here would fail a generation over a write nobody needed.
        after, transition = await apply_with_retry(self.store, identity, mutate, required=False)
        await self._after_transition(transition)

        if decision.fallback_used:
            self.counters.increment(
                Metric.PROVIDER_FAILOVER_SUCCESS.value
                if succeeded
                else Metric.PROVIDER_FAILOVER_FAILURE.value,
                identity=identity,
            )

        return AttemptRecord(
            attempt=attempt,
            provider=decision.selected,
            provider_revision=decision.selected_revision,
            outcome="SUCCEEDED" if succeeded else "FAILED",
            category=category,
            latency_seconds=latency_seconds,
            circuit_before=before.state,
            circuit_after=after.state,
            was_probe=decision.probe_token is not None
            and decision.outcome == RoutingOutcome.SELECTED_AS_PROBE.value,
            was_fallback=decision.fallback_used,
            at=now,
        )

    async def abandon(self, decision: RoutingDecision) -> None:
        """Give back a probe slot without recording anything.

        For cancellation. A request the user abandoned learned nothing
        about the provider, so it must not count as a success or a
        failure — but it must not hold the only probe slot either.
        """
        if decision.probe_token is None or decision.selected is None:
            return
        now = self._clock()
        identity = CircuitIdentity(decision.selected, decision.needs.task_type)
        await apply_with_retry(
            self.store,
            identity,
            lambda current: (
                release_probe(current, token=decision.probe_token or "", now=now),
                None,
            ),
            # Best effort: the slot's lease expires on its own if this
            # loses, and a cancelled request must not raise on its way out.
            required=False,
        )

    async def _after_transition(self, transition: Transition | None) -> None:
        if transition is None:
            return
        identity = transition.identity
        if transition.current == CircuitState.OPEN.value:
            self.counters.increment(Metric.CIRCUIT_OPEN_TOTAL.value, identity=identity)
        elif transition.current == CircuitState.HALF_OPEN.value:
            self.counters.increment(Metric.CIRCUIT_HALF_OPEN_TOTAL.value, identity=identity)
        elif transition.current == CircuitState.CLOSED.value:
            self.counters.increment(Metric.CIRCUIT_CLOSE_TOTAL.value, identity=identity)
        alert = alert_for_transition(transition)
        if alert is not None:
            self.alerts.append(alert)

    # ── operator actions ─────────────────────────────────────────────

    async def open(self, identity: CircuitIdentity, *, operator: str, reason: str) -> CircuitRecord:
        now = self._clock()
        record, transition = await apply_with_retry(
            self.store,
            identity,
            lambda current: manual_open(current, operator=operator, reason=reason, now=now),
        )
        await self._after_transition(transition)
        return record

    async def close(
        self, identity: CircuitIdentity, *, operator: str, reason: str
    ) -> CircuitRecord:
        now = self._clock()
        record, transition = await apply_with_retry(
            self.store,
            identity,
            lambda current: manual_close(current, operator=operator, reason=reason, now=now),
        )
        await self._after_transition(transition)
        return record

    async def reset(self, identity: CircuitIdentity, *, operator: str) -> CircuitRecord:
        now = self._clock()
        record, transition = await apply_with_retry(
            self.store,
            identity,
            lambda current: reset_to_policy(current, operator=operator, now=now),
        )
        await self._after_transition(transition)
        return record

    # ── views ────────────────────────────────────────────────────────

    async def readiness(self) -> ReadinessReport:
        return await readiness(self.profiles, store=self.store, now=self._clock())

    async def circuits(self) -> list[CircuitRecord]:
        return list(await self.store.all_circuits())

    async def status(self) -> dict[str, Any]:
        """Everything an operator asks for in one call."""
        report = await self.readiness()
        return {
            **version_block(),
            "at": self._clock().isoformat(),
            "readiness": report.to_dict(),
            "circuits": [record.to_dict() for record in await self.circuits()],
            "policy": {
                "routing": self.router.policy.to_dict(),
                "circuit": self.circuit_policy.to_dict(),
            },
            "metrics": self.counters.snapshot(),
            "providers": [profile.to_dict() for profile in self.profiles],
            "automatic_remediation": (
                "circuits open and close automatically; no provider is added, removed or "
                "reconfigured by this layer"
            ),
        }

    def drain_alerts(self) -> list[ResilienceAlert]:
        """Take the alerts raised since the last call."""
        pending, self.alerts = self.alerts, []
        return pending


#: Re-exported so callers importing the manager get the vocabulary too.
__all__ = [
    "AttemptRecord",
    "FailureCategory",
    "ResilienceManager",
    "ResilienceTrace",
]
