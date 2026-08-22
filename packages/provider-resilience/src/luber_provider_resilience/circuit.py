"""The circuit: when to stop calling a provider, and when to try again.

A breaker exists to convert a slow failure into a fast one. A provider
that is down does not answer quickly — it times out, four minutes at a
time, three attempts per request, for every request in the queue. The
circuit's value is refusing in microseconds instead, and saying so.

Everything here is a pure function of state plus evidence plus a clock.
No I/O, no locks, no globals: the store decides durability and the
caller supplies the time. That is what makes a state machine with
timeouts testable without sleeping, and what lets two workers agree by
both applying the same rules to the same row.

Four decisions worth defending.

**Identity is per provider *and* per task type.** One circuit per
provider would let a broken cover endpoint stop text-to-music, which is
the failure mode where a breaker does more damage than the outage.

**Evidence is bounded and dual.** A rolling window with a minimum sample
count, *and* a consecutive-failure rule. The window is right for a busy
provider and useless for one taking four requests an hour; the
consecutive rule is right for the quiet one and trigger-happy for the
busy one. Having both means neither has to be tuned for a traffic level
it will not see.

**Recovery is bounded too.** OPEN has an expiry, because a circuit
nobody resets is an outage that outlives its cause. HALF_OPEN admits a
counted number of probes, because releasing full traffic at a provider
that has been down for five minutes is how a recovering service is
knocked over again.

**A manual decision outranks the policy and is never silently undone.**
An operator who opens a circuit has a reason the evidence does not
contain.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from luber_provider_resilience.classification import (
    FailureCategory,
    counts_toward_circuit,
)
from luber_provider_resilience.versions import CIRCUIT_POLICY_VERSION, version_block


class CircuitState(StrEnum):
    """The three canonical states, and nothing else."""

    #: Normal traffic permitted.
    CLOSED = "CLOSED"
    #: Normal traffic refused. Bounded by `open_until`.
    OPEN = "OPEN"
    #: Only counted probe traffic permitted.
    HALF_OPEN = "HALF_OPEN"


class ControlMode(StrEnum):
    """Whether the policy or a human is deciding."""

    AUTOMATIC = "AUTOMATIC"
    #: An operator pinned it. The policy does not move it back.
    MANUAL = "MANUAL"


@dataclass(frozen=True, order=True)
class CircuitIdentity:
    """What a circuit is about.

    Provider *and* task type, deliberately. A provider whose cover
    endpoint is broken can still serve text-to-music, and a single
    provider-wide circuit would take both offline — turning a partial
    outage into a total one, which is the breaker doing more harm than
    the fault.

    Model revision is deliberately **not** part of identity. A revision
    changes under a running deployment, and keying on it would reset
    every circuit's evidence at exactly the moment a bad rollout needed
    it most. The revision is recorded on the evidence instead, so an
    operator can see it without the breaker forgetting.
    """

    provider: str
    task_type: str = "ANY"

    def key(self) -> str:
        return f"{self.provider}:{self.task_type}"

    def to_dict(self) -> dict[str, str]:
        return {"provider": self.provider, "task_type": self.task_type}

    def label(self) -> str:
        return f"{self.provider} ({self.task_type})"

    @classmethod
    def parse(cls, key: str) -> CircuitIdentity:
        provider, _, task = key.partition(":")
        return cls(provider=provider, task_type=task or "ANY")


@dataclass(frozen=True)
class CircuitPolicy:
    """When a circuit opens, how long it stays open, what closes it.

    Every default here is conservative in the direction of *keeping the
    provider in rotation*. A breaker that opens too readily is worse
    than none: it converts a transient blip into a self-inflicted
    outage, and an operator who has seen that once stops trusting it.
    """

    #: Rolling window the failure rate is measured over.
    window: timedelta = timedelta(minutes=5)

    #: Attempts needed in the window before a *rate* can open anything.
    #: Below this, only the consecutive rule applies — a rate computed
    #: from four attempts is not a rate.
    minimum_samples: int = 10

    #: Failure rate that opens the circuit, once there are enough
    #: samples. Half: a provider failing half its requests is not
    #: serving, and anything lower risks opening on a bad minute.
    failure_rate_threshold: float = 0.5

    #: Consecutive counted failures that open the circuit regardless of
    #: volume. Five, because a low-traffic provider may never reach the
    #: sample minimum, and five in a row is not a coincidence.
    consecutive_failure_threshold: int = 5

    #: How long OPEN lasts before a probe is allowed. Thirty seconds:
    #: long enough that a restarting provider is not hammered, short
    #: enough that a recovered one is back in under a minute.
    open_duration: timedelta = timedelta(seconds=30)

    #: Ceiling for repeated re-opens. Each consecutive open doubles the
    #: cooldown, so a provider that keeps failing its probes is asked
    #: less and less often — but never less than this.
    maximum_open_duration: timedelta = timedelta(minutes=10)

    #: How many probes may be in flight in HALF_OPEN. One: the point is
    #: to learn whether it works, and one request answers that.
    probe_concurrency: int = 1

    #: Consecutive probe successes required to close. Two, so a single
    #: lucky response does not restore full traffic to a provider that
    #: is still unwell.
    probe_successes_to_close: int = 2

    #: How long a probe slot may be held before it is assumed lost. A
    #: worker that dies mid-probe would otherwise hold the only slot
    #: forever and leave the circuit stuck in HALF_OPEN.
    probe_lease: timedelta = timedelta(minutes=5)

    #: Whether a rate limit counts as harshly as a failure. False: a 429
    #: is the provider working correctly and declining, so it makes the
    #: provider temporarily unusable without being evidence of breakage.
    #: It still counts toward the window — a provider refusing every
    #: request is unusable however politely it says so.
    rate_limit_weight: float = 0.5

    def open_for(self, consecutive_opens: int) -> timedelta:
        """How long to stay open, given how often this has happened.

        Exponential with a ceiling. A provider that fails its probe
        repeatedly is asked less often, which stops a recovery loop
        turning into a slow denial of service against a struggling
        service.
        """
        if consecutive_opens <= 1:
            return self.open_duration
        multiplier = 2 ** min(consecutive_opens - 1, 10)
        scaled: timedelta = self.open_duration * multiplier
        return min(scaled, self.maximum_open_duration)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_seconds": self.window.total_seconds(),
            "minimum_samples": self.minimum_samples,
            "failure_rate_threshold": self.failure_rate_threshold,
            "consecutive_failure_threshold": self.consecutive_failure_threshold,
            "open_duration_seconds": self.open_duration.total_seconds(),
            "maximum_open_duration_seconds": self.maximum_open_duration.total_seconds(),
            "probe_concurrency": self.probe_concurrency,
            "probe_successes_to_close": self.probe_successes_to_close,
            "probe_lease_seconds": self.probe_lease.total_seconds(),
            "rate_limit_weight": self.rate_limit_weight,
            "circuit_policy_version": CIRCUIT_POLICY_VERSION,
        }


@dataclass(frozen=True)
class Outcome:
    """One provider attempt, as far as the circuit cares.

    Deliberately thin. The circuit needs to know whether it worked, what
    kind of failure it was if not, and when — nothing about the audio,
    the request or the user.
    """

    at: datetime
    succeeded: bool
    category: str | None = None
    latency_seconds: float | None = None
    provider_revision: str | None = None

    @property
    def counts(self) -> bool:
        """Whether this outcome is evidence about the provider."""
        if self.succeeded:
            return True
        return self.category is not None and counts_toward_circuit(self.category)

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at.astimezone(UTC).isoformat(),
            "succeeded": self.succeeded,
            "category": self.category,
            "latency_seconds": self.latency_seconds,
            "provider_revision": self.provider_revision,
        }


@dataclass(frozen=True)
class Transition:
    """A state change, with the evidence that caused it."""

    identity: CircuitIdentity
    previous: str
    current: str
    at: datetime
    reason: str
    automatic: bool = True
    operator: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **version_block(),
            "identity": self.identity.to_dict(),
            "previous_state": self.previous,
            "current_state": self.current,
            "at": self.at.astimezone(UTC).isoformat(),
            "reason": self.reason,
            "automatic": self.automatic,
            "operator": self.operator,
            "evidence": self.evidence,
        }


@dataclass
class CircuitRecord:
    """One circuit's durable state.

    Mutable, because it is a row: the store loads it, a transition
    rewrites it, the store persists it under a compare-and-set. The
    functions below never mutate in place — they return a new record —
    so a caller cannot half-apply a transition and then fail.
    """

    identity: CircuitIdentity
    state: str = CircuitState.CLOSED.value
    control: str = ControlMode.AUTOMATIC.value

    #: Counted outcomes inside the rolling window, oldest first. Bounded
    #: by `_prune`: a busy provider must not grow this without limit.
    window: list[Outcome] = field(default_factory=list)
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    opened_at: datetime | None = None
    open_until: datetime | None = None
    #: How many times in a row this has opened without a clean close.
    #: Drives the exponential cooldown.
    consecutive_opens: int = 0
    open_reason: str | None = None
    open_evidence: dict[str, Any] = field(default_factory=dict)

    #: Probe slots currently held, as (token, expires_at).
    probes: dict[str, datetime] = field(default_factory=dict)
    probe_successes: int = 0

    last_failure_at: datetime | None = None
    last_failure_category: str | None = None
    last_success_at: datetime | None = None
    last_transition_at: datetime | None = None
    last_provider_revision: str | None = None

    manual_reason: str | None = None
    manual_operator: str | None = None
    manual_at: datetime | None = None

    #: Monotonic, bumped on every write. The store's compare-and-set
    #: value: two workers reading the same version and both writing
    #: means one of them loses and retries, which is what makes a
    #: threshold crossing produce exactly one transition.
    revision: int = 0

    circuit_policy_version: str = CIRCUIT_POLICY_VERSION

    # ── read-only questions ──────────────────────────────────────────

    def failure_count(self) -> int:
        return sum(1 for item in self.window if not item.succeeded)

    def success_count(self) -> int:
        return sum(1 for item in self.window if item.succeeded)

    def sample_count(self) -> int:
        return len(self.window)

    def failure_rate(self, policy: CircuitPolicy) -> float | None:
        """Weighted failure rate, or ``None`` when there is nothing to divide.

        Rate limits are weighted below a hard failure: a provider
        declining politely is less broken than one that cannot answer,
        and a burst of 429s should not read the same as a burst of
        connection resets.
        """
        if not self.window:
            return None
        weight = 0.0
        for item in self.window:
            if item.succeeded:
                continue
            if item.category == FailureCategory.PROVIDER_RATE_LIMIT.value:
                weight += policy.rate_limit_weight
            else:
                weight += 1.0
        return weight / len(self.window)

    def active_probes(self, now: datetime) -> int:
        """Probe slots still held. Expired leases do not count."""
        return sum(1 for expires in self.probes.values() if expires > now)

    def to_dict(self) -> dict[str, Any]:
        return {
            **version_block(),
            "identity": self.identity.to_dict(),
            "key": self.identity.key(),
            "state": self.state,
            "control": self.control,
            "sample_count": self.sample_count(),
            "failure_count": self.failure_count(),
            "success_count": self.success_count(),
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "opened_at": _iso(self.opened_at),
            "open_until": _iso(self.open_until),
            "consecutive_opens": self.consecutive_opens,
            "open_reason": self.open_reason,
            "open_evidence": self.open_evidence,
            "probes_held": len(self.probes),
            "probe_successes": self.probe_successes,
            "last_failure_at": _iso(self.last_failure_at),
            "last_failure_category": self.last_failure_category,
            "last_success_at": _iso(self.last_success_at),
            "last_transition_at": _iso(self.last_transition_at),
            "last_provider_revision": self.last_provider_revision,
            "manual_reason": self.manual_reason,
            "manual_operator": self.manual_operator,
            "manual_at": _iso(self.manual_at),
            "revision": self.revision,
        }


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(UTC).isoformat()


def _prune(record: CircuitRecord, policy: CircuitPolicy, now: datetime) -> list[Outcome]:
    """Outcomes still inside the rolling window.

    Also caps the list. A provider serving continuously would otherwise
    accumulate every outcome in the window in a column somebody has to
    load, and the rate is the same computed from the last few hundred.
    """
    horizon = now - policy.window
    inside = [item for item in record.window if item.at > horizon]
    return inside[-500:]


def _expire_probes(probes: dict[str, datetime], now: datetime) -> dict[str, datetime]:
    """Drop probe slots whose lease has run out.

    A worker that died mid-probe would otherwise hold the only slot
    forever, leaving the circuit stuck in HALF_OPEN and the provider
    permanently half-unavailable — a failure mode strictly worse than
    the outage it was reacting to.
    """
    return {token: expires for token, expires in probes.items() if expires > now}


# ── the state machine ────────────────────────────────────────────────


def allows(record: CircuitRecord, now: datetime) -> bool:
    """Whether normal (non-probe) traffic may go to this provider.

    Note that OPEN whose expiry has passed still refuses here. Promotion
    to HALF_OPEN is a transition somebody has to write down, not a fact
    that quietly becomes true at a moment nobody observed — otherwise
    two workers would disagree about the state of a circuit neither had
    touched.
    """
    return record.state == CircuitState.CLOSED.value


def ready_for_probe(record: CircuitRecord, now: datetime) -> bool:
    """Whether an OPEN circuit has waited long enough to be probed."""
    if record.state != CircuitState.OPEN.value:
        return False
    if record.control == ControlMode.MANUAL.value:
        # An operator pinned this open. The cooldown does not release a
        # decision the policy did not make.
        return False
    return record.open_until is not None and now >= record.open_until


def record_outcome(
    record: CircuitRecord,
    outcome: Outcome,
    *,
    policy: CircuitPolicy,
) -> tuple[CircuitRecord, Transition | None]:
    """Fold one attempt into a circuit, returning the new record.

    Pure: the input record is not mutated, so a caller that fails to
    persist leaves nothing half-applied.

    An outcome that is not evidence — a cancellation, a user error, a
    quality rejection — is dropped here rather than at the call site.
    That is the single point where the "user error must not poison
    provider health" rule is enforced, and it is enforced by the type of
    the failure rather than by whoever wrote the caller.
    """
    now = outcome.at
    if not outcome.counts:
        return record, None

    window = _prune(record, policy, now)
    window.append(outcome)

    updated = replace(
        record,
        window=window,
        consecutive_failures=0 if outcome.succeeded else record.consecutive_failures + 1,
        consecutive_successes=record.consecutive_successes + 1 if outcome.succeeded else 0,
        last_success_at=now if outcome.succeeded else record.last_success_at,
        last_failure_at=record.last_failure_at if outcome.succeeded else now,
        last_failure_category=(
            record.last_failure_category if outcome.succeeded else outcome.category
        ),
        last_provider_revision=outcome.provider_revision or record.last_provider_revision,
        revision=record.revision + 1,
    )

    if updated.control == ControlMode.MANUAL.value:
        # Evidence keeps accumulating under a manual pin — an operator
        # who opened a circuit still wants to see whether the provider
        # recovered — but the policy does not move the state.
        return updated, None

    if updated.state == CircuitState.HALF_OPEN.value:
        return _half_open_outcome(updated, outcome, policy=policy, now=now)

    if updated.state == CircuitState.CLOSED.value and not outcome.succeeded:
        return _maybe_open(updated, policy=policy, now=now)

    return updated, None


def _maybe_open(
    record: CircuitRecord, *, policy: CircuitPolicy, now: datetime
) -> tuple[CircuitRecord, Transition | None]:
    """Two independent rules, either of which opens a closed circuit."""
    consecutive = record.consecutive_failures
    rate = record.failure_rate(policy)
    samples = record.sample_count()

    if consecutive >= policy.consecutive_failure_threshold:
        reason = (
            f"{consecutive} consecutive provider failures "
            f"(threshold {policy.consecutive_failure_threshold})"
        )
        evidence = {
            "rule": "consecutive",
            "consecutive_failures": consecutive,
            "threshold": policy.consecutive_failure_threshold,
            "last_category": record.last_failure_category,
        }
        return _open(record, reason=reason, evidence=evidence, policy=policy, now=now)

    if samples >= policy.minimum_samples and rate is not None:
        if rate >= policy.failure_rate_threshold:
            reason = (
                f"{record.failure_count()}/{samples} attempts failed "
                f"({rate * 100:.0f}% weighted, threshold "
                f"{policy.failure_rate_threshold * 100:.0f}%) in the last "
                f"{policy.window.total_seconds():.0f}s"
            )
            evidence = {
                "rule": "failure_rate",
                "failures": record.failure_count(),
                "samples": samples,
                "weighted_rate": round(rate, 4),
                "threshold": policy.failure_rate_threshold,
                "window_seconds": policy.window.total_seconds(),
                "last_category": record.last_failure_category,
            }
            return _open(record, reason=reason, evidence=evidence, policy=policy, now=now)

    return record, None


def _open(
    record: CircuitRecord,
    *,
    reason: str,
    evidence: dict[str, Any],
    policy: CircuitPolicy,
    now: datetime,
    automatic: bool = True,
    operator: str | None = None,
) -> tuple[CircuitRecord, Transition]:
    opens = record.consecutive_opens + 1
    duration = policy.open_for(opens)
    updated = replace(
        record,
        state=CircuitState.OPEN.value,
        opened_at=now,
        open_until=now + duration,
        consecutive_opens=opens,
        open_reason=reason,
        open_evidence={**evidence, "cooldown_seconds": duration.total_seconds()},
        probes={},
        probe_successes=0,
        consecutive_successes=0,
        last_transition_at=now,
        revision=record.revision + 1,
    )
    return updated, Transition(
        identity=record.identity,
        previous=record.state,
        current=CircuitState.OPEN.value,
        at=now,
        reason=reason,
        automatic=automatic,
        operator=operator,
        evidence=updated.open_evidence,
    )


def _half_open_outcome(
    record: CircuitRecord, outcome: Outcome, *, policy: CircuitPolicy, now: datetime
) -> tuple[CircuitRecord, Transition | None]:
    """A probe came back. Close, re-open, or keep probing."""
    if not outcome.succeeded:
        reason = (
            f"probe failed ({outcome.category}); reopening for "
            f"{policy.open_for(record.consecutive_opens + 1).total_seconds():.0f}s"
        )
        return _open(
            record,
            reason=reason,
            evidence={
                "rule": "probe_failure",
                "category": outcome.category,
                "consecutive_opens": record.consecutive_opens + 1,
            },
            policy=policy,
            now=now,
        )

    successes = record.probe_successes + 1
    if successes < policy.probe_successes_to_close:
        # Still probing. One good answer from a provider that was down
        # is not yet evidence it is up.
        return replace(record, probe_successes=successes, revision=record.revision + 1), None

    reason = f"{successes} consecutive probe successes; provider is answering again"
    updated = replace(
        record,
        state=CircuitState.CLOSED.value,
        opened_at=None,
        open_until=None,
        open_reason=None,
        open_evidence={},
        consecutive_opens=0,
        probes={},
        probe_successes=0,
        last_transition_at=now,
        revision=record.revision + 1,
    )
    return updated, Transition(
        identity=record.identity,
        previous=CircuitState.HALF_OPEN.value,
        current=CircuitState.CLOSED.value,
        at=now,
        reason=reason,
        evidence={"probe_successes": successes},
    )


def promote_to_half_open(
    record: CircuitRecord, *, now: datetime
) -> tuple[CircuitRecord, Transition | None]:
    """OPEN → HALF_OPEN once the cooldown has expired.

    A written transition rather than a state that becomes true with the
    passage of time. Two workers reading an expired OPEN both attempt
    this; the store's compare-and-set means one wins and the other
    re-reads, so exactly one promotion is recorded.
    """
    if not ready_for_probe(record, now):
        return record, None
    updated = replace(
        record,
        state=CircuitState.HALF_OPEN.value,
        probes={},
        probe_successes=0,
        last_transition_at=now,
        revision=record.revision + 1,
    )
    return updated, Transition(
        identity=record.identity,
        previous=CircuitState.OPEN.value,
        current=CircuitState.HALF_OPEN.value,
        at=now,
        reason="cooldown expired; admitting bounded probe traffic",
        evidence={"open_until": _iso(record.open_until)},
    )


def claim_probe(
    record: CircuitRecord, *, token: str, policy: CircuitPolicy, now: datetime
) -> tuple[CircuitRecord, bool]:
    """Take a probe slot, if one is free.

    The slot, not the state, is what bounds recovery traffic. Twenty
    simultaneous requests all find the circuit HALF_OPEN; the number
    that reach the provider is the number that win a slot here, and the
    store's compare-and-set is what makes "win" mean one of them.
    """
    if record.state != CircuitState.HALF_OPEN.value:
        return record, False
    live = _expire_probes(record.probes, now)
    if len(live) >= policy.probe_concurrency:
        return replace(record, probes=live), False
    live[token] = now + policy.probe_lease
    return replace(record, probes=live, revision=record.revision + 1), True


def release_probe(record: CircuitRecord, *, token: str, now: datetime) -> CircuitRecord:
    """Give a probe slot back without recording an outcome.

    For the cancelled case: a probe abandoned mid-flight learned nothing
    about the provider, so it must free the slot without counting as
    either a success or a failure.
    """
    live = _expire_probes(record.probes, now)
    live.pop(token, None)
    return replace(record, probes=live, revision=record.revision + 1)


# ── operator overrides ───────────────────────────────────────────────


def manual_open(
    record: CircuitRecord, *, operator: str, reason: str, now: datetime
) -> tuple[CircuitRecord, Transition]:
    """Pin the circuit open. The policy will not move it back."""
    if not reason.strip():
        raise ValueError(
            "opening a circuit by hand needs a reason: the evidence does not contain "
            "whatever you know, and the next person will need it"
        )
    updated = replace(
        record,
        state=CircuitState.OPEN.value,
        control=ControlMode.MANUAL.value,
        opened_at=now,
        # No expiry: a manual open is released by a human, not by a
        # clock. `reset_to_policy` is how it goes back.
        open_until=None,
        open_reason=f"manual: {reason.strip()}",
        open_evidence={"rule": "manual", "operator": operator},
        probes={},
        probe_successes=0,
        manual_reason=reason.strip(),
        manual_operator=operator,
        manual_at=now,
        last_transition_at=now,
        revision=record.revision + 1,
    )
    return updated, Transition(
        identity=record.identity,
        previous=record.state,
        current=CircuitState.OPEN.value,
        at=now,
        reason=f"manual open by {operator}: {reason.strip()}",
        automatic=False,
        operator=operator,
        evidence={"rule": "manual"},
    )


def manual_close(
    record: CircuitRecord, *, operator: str, reason: str, now: datetime
) -> tuple[CircuitRecord, Transition]:
    """Pin the circuit closed, and clear the evidence that opened it.

    The counters are reset because leaving them would re-open the
    circuit on the next failure — an operator who closes a circuit is
    saying the past evidence no longer applies. The *history* is not
    deleted: transitions are append-only and the reason is recorded.
    """
    if not reason.strip():
        raise ValueError("closing a circuit by hand needs a reason")
    updated = replace(
        record,
        state=CircuitState.CLOSED.value,
        control=ControlMode.MANUAL.value,
        window=[],
        consecutive_failures=0,
        consecutive_successes=0,
        opened_at=None,
        open_until=None,
        consecutive_opens=0,
        open_reason=None,
        open_evidence={},
        probes={},
        probe_successes=0,
        manual_reason=reason.strip(),
        manual_operator=operator,
        manual_at=now,
        last_transition_at=now,
        revision=record.revision + 1,
    )
    return updated, Transition(
        identity=record.identity,
        previous=record.state,
        current=CircuitState.CLOSED.value,
        at=now,
        reason=f"manual close by {operator}: {reason.strip()}",
        automatic=False,
        operator=operator,
        evidence={"rule": "manual"},
    )


def reset_to_policy(
    record: CircuitRecord, *, operator: str, now: datetime
) -> tuple[CircuitRecord, Transition]:
    """Hand the circuit back to the policy, starting from clean evidence."""
    updated = replace(
        record,
        state=CircuitState.CLOSED.value,
        control=ControlMode.AUTOMATIC.value,
        window=[],
        consecutive_failures=0,
        consecutive_successes=0,
        opened_at=None,
        open_until=None,
        consecutive_opens=0,
        open_reason=None,
        open_evidence={},
        probes={},
        probe_successes=0,
        manual_reason=None,
        manual_operator=None,
        manual_at=None,
        last_transition_at=now,
        revision=record.revision + 1,
    )
    return updated, Transition(
        identity=record.identity,
        previous=record.state,
        current=CircuitState.CLOSED.value,
        at=now,
        reason=f"reset to automatic policy by {operator}",
        automatic=False,
        operator=operator,
        evidence={"rule": "reset"},
    )


__all__ = [
    "CircuitIdentity",
    "CircuitPolicy",
    "CircuitRecord",
    "CircuitState",
    "ControlMode",
    "Outcome",
    "Transition",
    "allows",
    "claim_probe",
    "manual_close",
    "manual_open",
    "promote_to_half_open",
    "ready_for_probe",
    "record_outcome",
    "release_probe",
    "reset_to_policy",
]
