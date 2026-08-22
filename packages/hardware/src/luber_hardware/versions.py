"""Three policies, three versions, because they move for different reasons.

A placement decision recorded last month and one recorded today are only
comparable if the rules that produced them agreed. When they do not, the
honest response is a version stamped on the decision, so a reader can
tell "the hardware changed" from "we started placing things differently".

They are separate because they change separately. Adding a device to the
precision matrix does not change which location a workload prefers;
allowing a workload onto MPS does not change what `auto` precision
resolves to. One combined version would make every stored decision look
stale whenever anything anywhere moved.
"""

from __future__ import annotations

from typing import Any

#: The shape of a normalized capability report.
CAPABILITY_SCHEMA_VERSION = "luber-hardware-capability/1"

#: How a requested precision resolves against a device. Bump when the
#: same request on the same hardware would produce a different dtype.
PRECISION_POLICY_VERSION = "precision-v1"

#: Which location and device a workload prefers, and what is refused.
#: Bump when the same workload against the same capability would be
#: placed somewhere else.
EXECUTION_PLACEMENT_POLICY_VERSION = "placement-v1"


def version_block() -> dict[str, Any]:
    """Stamped onto every capability report and placement decision."""
    return {
        "capability_schema_version": CAPABILITY_SCHEMA_VERSION,
        "precision_policy_version": PRECISION_POLICY_VERSION,
        "execution_placement_policy_version": EXECUTION_PLACEMENT_POLICY_VERSION,
    }


__all__ = [
    "CAPABILITY_SCHEMA_VERSION",
    "EXECUTION_PLACEMENT_POLICY_VERSION",
    "PRECISION_POLICY_VERSION",
    "version_block",
]
