"""Response models for the circuit view.

Two properties hold here for the same reason they hold in the inference
console's schemas, and both are structural rather than remembered.

**No field a prompt could occupy.** A circuit is about a provider name,
a task type and a count. There is no user in it, so there is nothing
below that could carry one — a response cannot leak lyrics if no model
can hold a string that came from a user.

**No field a credential could occupy.** Circuits open because a provider
timed out or answered 401; the *category* is recorded and the message is
not. `AUTH_FAILED` is what an operator needs, and it cannot contain a
key the way a provider's error body can.
"""

from __future__ import annotations

from pydantic import BaseModel


class VersionBlock(BaseModel):
    resilience_schema_version: str
    circuit_policy_version: str
    routing_policy_version: str
    failover_policy_version: str


class CircuitView(BaseModel):
    """One provider's circuit for one task type."""

    circuit_key: str
    provider: str
    task_type: str
    state: str
    control: str
    consecutive_failures: int
    consecutive_successes: int
    sample_count: int
    failure_count: int
    #: ``None`` when the window holds too few samples to divide by.
    #: Never a bare 0.0 standing in for "we have not seen anything".
    failure_rate: float | None = None
    opened_at: str | None = None
    open_until: str | None = None
    open_reason: str | None = None
    consecutive_opens: int = 0
    active_probes: int = 0
    probe_successes: int = 0
    last_failure_at: str | None = None
    last_failure_category: str | None = None
    last_success_at: str | None = None
    last_transition_at: str | None = None
    manual_reason: str | None = None
    manual_operator: str | None = None
    revision: int = 0


class CircuitListResponse(VersionBlock):
    at: str
    circuits: list[CircuitView]
    #: Present when a deployment has circuits for providers it no longer
    #: configures. Stated rather than filtered: a stale circuit is worth
    #: knowing about, and hiding it would make an open one invisible.
    unconfigured_providers: list[str] = []


class TransitionView(BaseModel):
    """One state change, as it was recorded."""

    id: str
    circuit_key: str
    provider: str
    task_type: str
    previous_state: str
    current_state: str
    occurred_at: str
    reason: str
    automatic: bool
    operator: str | None = None
    circuit_policy_version: str


class TransitionListResponse(VersionBlock):
    transitions: list[TransitionView]


class ProviderReadinessView(BaseModel):
    provider: str
    revision: str
    circuit_state: str
    control: str
    open_reason: str | None = None
    open_until: str | None = None


class CapabilityReadinessView(BaseModel):
    capability: str
    status: str
    detail: str
    providers: list[ProviderReadinessView]


class ReadinessResponse(VersionBlock):
    at: str
    generation_available: bool
    degraded: bool
    summary: str
    capabilities: list[CapabilityReadinessView]
    metrics: dict[str, int]


class PolicyResponse(VersionBlock):
    """The thresholds in force, so a state can be read against them.

    An operator looking at "3 consecutive failures" needs to know
    whether the threshold is 5 or 2. Serving the policy alongside the
    circuits is what makes the numbers mean something.
    """

    resilience_enabled: bool
    failover_mode: str
    #: True only when failover is on *and* somewhere to fail over to
    #: exists. A mode that reads SAFE_EQUIVALENT_ONLY on a single-provider
    #: deployment can never move a request, and the console says so
    #: rather than implying a redundancy that is not there.
    failover_possible: bool
    routable_providers: list[str]
    circuit_policy: dict[str, float | int | str]


__all__ = [
    "CapabilityReadinessView",
    "CircuitListResponse",
    "CircuitView",
    "PolicyResponse",
    "ProviderReadinessView",
    "ReadinessResponse",
    "TransitionListResponse",
    "TransitionView",
]
