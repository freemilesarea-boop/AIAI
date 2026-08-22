"""Three policies, three versions, because they change for different reasons.

A circuit decision recorded last month and one recorded today are only
comparable if the rules that produced them agreed. When they do not, the
honest response is a version bump stamped on the decision, so a reader
can tell "the provider got worse" from "we started opening circuits
sooner".

They are separate because they move separately. Tightening the failure
threshold does not change which provider a healthy request goes to;
permitting failover for a new task type does not change when a circuit
opens. One combined version would make every stored decision look stale
whenever anything anywhere changed, and a staleness signal that fires
constantly is one nobody reads.
"""

from __future__ import annotations

from typing import Any

#: When a circuit opens, how long it stays open, what counts as evidence
#: and what recovery requires. Bump when the same failure stream would
#: produce a different circuit timeline.
CIRCUIT_POLICY_VERSION = "circuit-v1"

#: How a provider is chosen for a request: eligibility, capability
#: matching, ordering. Bump when the same request against the same
#: circuit states would select a different provider.
ROUTING_POLICY_VERSION = "routing-v1"

#: Whether and when a request may be moved to a different provider.
#: Bump when the same failure would produce a different failover
#: decision.
FAILOVER_POLICY_VERSION = "failover-v1"

#: The shape of a persisted circuit record.
RESILIENCE_SCHEMA_VERSION = "luber-provider-resilience/1"


def version_block() -> dict[str, Any]:
    """Stamped onto every routing decision, transition and report."""
    return {
        "resilience_schema_version": RESILIENCE_SCHEMA_VERSION,
        "circuit_policy_version": CIRCUIT_POLICY_VERSION,
        "routing_policy_version": ROUTING_POLICY_VERSION,
        "failover_policy_version": FAILOVER_POLICY_VERSION,
    }


__all__ = [
    "CIRCUIT_POLICY_VERSION",
    "FAILOVER_POLICY_VERSION",
    "RESILIENCE_SCHEMA_VERSION",
    "ROUTING_POLICY_VERSION",
    "version_block",
]
