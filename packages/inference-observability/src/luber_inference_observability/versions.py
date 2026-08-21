"""Four versions, because four things can change independently.

An aggregate computed last month and one computed today are only
comparable if the code that produced them agreed about what the numbers
mean. When it does not — a new dimension, a changed bucket boundary, a
different regression rule — the honest response is a version bump, so a
reader can tell "the rate went up" from "we started counting
differently".

They are separate because they move separately. Adding a dimension to
the observation schema does not change how a regression is decided;
changing a severity threshold does not invalidate an aggregate. One
combined version would force every stored result to look stale whenever
anything anywhere changed, and a staleness signal that fires constantly
is one nobody reads.
"""

from __future__ import annotations

from typing import Any

#: The shape of a stored observation. Bump when a field is added,
#: removed, or given a different meaning.
OBSERVABILITY_SCHEMA_VERSION = "luber-inference-observability/1"

#: How counters and rates are derived from observations, including the
#: duration bucket boundaries and what counts as a failure. Bump when a
#: metric would produce a different number from the same rows.
AGGREGATION_VERSION = "agg-v1"

#: How a baseline and a current window are compared into a finding.
#: Bump when a detector's semantics change — a new rule, a changed
#: comparison, a different minimum sample.
REGRESSION_ENGINE_VERSION = "regress-v1"

#: How findings become incidents: fingerprinting, severity mapping,
#: recovery and flapping control. Bump when the same finding stream
#: would produce a different incident timeline.
INCIDENT_POLICY_VERSION = "incident-v1"

#: The commit that introduced Phase 29. Generations created before it
#: have no candidate trace, so their retry counts are *unknown* rather
#: than zero, and any window overlapping it is reported as partial.
PHASE29_BOUNDARY_COMMIT = "460642e"


def version_block() -> dict[str, Any]:
    """Stamped onto every aggregate, finding, incident and report."""
    return {
        "observability_schema_version": OBSERVABILITY_SCHEMA_VERSION,
        "aggregation_version": AGGREGATION_VERSION,
        "regression_engine_version": REGRESSION_ENGINE_VERSION,
        "incident_policy_version": INCIDENT_POLICY_VERSION,
    }


__all__ = [
    "AGGREGATION_VERSION",
    "INCIDENT_POLICY_VERSION",
    "OBSERVABILITY_SCHEMA_VERSION",
    "PHASE29_BOUNDARY_COMMIT",
    "REGRESSION_ENGINE_VERSION",
    "version_block",
]
