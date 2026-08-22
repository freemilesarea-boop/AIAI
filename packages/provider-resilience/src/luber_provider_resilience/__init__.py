"""Containing provider failure without confusing it with anything else.

Phase 29 asks whether a candidate is good. Phase 30 asks whether the
system is trending worse. This asks a third question — *should the next
request go to this provider at all* — and it is the only one of the
three that changes what happens at runtime.

Four things are kept apart here that are constantly conflated:

**Transport retry** — the same request again, because a connection
dropped. Owned by Phase 29's controller, which already retries a
provider failure with the identical request.

**Quality retry** — a different seed, because the audio was wrong.
Owned by Phase 29.

**Provider failover** — a different provider, because this one cannot
serve the request. Owned here.

**User regeneration** — a new request. Owned by the user.

Nothing here multiplies attempts. Phase 29's budget remains the only
attempt budget; this layer decides *where* each attempt goes, not how
many there are.

And nothing here degrades a request silently. Failover requires that
every element of the request survives it — the reference track, the
lyrics, the duration, the task. Where it cannot, the request fails and
names what was missing, because delivering a song somebody did not ask
for and calling it success is worse than delivering nothing.
"""

from luber_provider_resilience.capabilities import (
    Capability,
    EquivalenceVerdict,
    ProviderProfile,
    RequestNeeds,
    check_equivalence,
    profile_provider,
)
from luber_provider_resilience.circuit import (
    CircuitIdentity,
    CircuitPolicy,
    CircuitRecord,
    CircuitState,
    ControlMode,
    Outcome,
    Transition,
    allows,
    claim_probe,
    manual_close,
    manual_open,
    promote_to_half_open,
    ready_for_probe,
    record_outcome,
    release_probe,
    reset_to_policy,
)
from luber_provider_resilience.classification import (
    COUNTS_TOWARD_CIRCUIT,
    NEVER_COUNTS,
    NON_RETRYABLE,
    FailureCategory,
    classify,
    counts_toward_circuit,
    error_code_for,
    is_retryable,
)
from luber_provider_resilience.durable import DurableCircuitStore
from luber_provider_resilience.manager import (
    AttemptRecord,
    ResilienceManager,
    ResilienceTrace,
)
from luber_provider_resilience.readiness import (
    CapabilityReadiness,
    CapabilityStatus,
    ReadinessReport,
    readiness,
)
from luber_provider_resilience.router import (
    ConsideredProvider,
    FailoverMode,
    ProviderRouter,
    RoutingDecision,
    RoutingOutcome,
    RoutingPolicy,
)
from luber_provider_resilience.store import (
    CircuitStore,
    ConcurrentModification,
    InMemoryCircuitStore,
    apply_with_retry,
)
from luber_provider_resilience.telemetry import (
    AlertKind,
    AlertSeverity,
    Counters,
    Metric,
    ResilienceAlert,
    alerts_for,
)
from luber_provider_resilience.versions import (
    CIRCUIT_POLICY_VERSION,
    FAILOVER_POLICY_VERSION,
    RESILIENCE_SCHEMA_VERSION,
    ROUTING_POLICY_VERSION,
    version_block,
)

__all__ = [
    "CIRCUIT_POLICY_VERSION",
    "COUNTS_TOWARD_CIRCUIT",
    "FAILOVER_POLICY_VERSION",
    "NEVER_COUNTS",
    "NON_RETRYABLE",
    "RESILIENCE_SCHEMA_VERSION",
    "ROUTING_POLICY_VERSION",
    "AlertKind",
    "AlertSeverity",
    "AttemptRecord",
    "Capability",
    "CapabilityReadiness",
    "CapabilityStatus",
    "CircuitIdentity",
    "CircuitPolicy",
    "CircuitRecord",
    "CircuitState",
    "CircuitStore",
    "ConcurrentModification",
    "ConsideredProvider",
    "ControlMode",
    "Counters",
    "DurableCircuitStore",
    "EquivalenceVerdict",
    "FailoverMode",
    "FailureCategory",
    "InMemoryCircuitStore",
    "Metric",
    "Outcome",
    "ProviderProfile",
    "ProviderRouter",
    "ReadinessReport",
    "RequestNeeds",
    "ResilienceAlert",
    "ResilienceManager",
    "ResilienceTrace",
    "RoutingDecision",
    "RoutingOutcome",
    "RoutingPolicy",
    "Transition",
    "alerts_for",
    "allows",
    "apply_with_retry",
    "check_equivalence",
    "claim_probe",
    "classify",
    "counts_toward_circuit",
    "error_code_for",
    "is_retryable",
    "manual_close",
    "manual_open",
    "profile_provider",
    "promote_to_half_open",
    "readiness",
    "ready_for_probe",
    "record_outcome",
    "release_probe",
    "reset_to_policy",
    "version_block",
]
