"""Four versions, bumped independently, recorded on every decision.

They are separate because they change for different reasons and a
question asked later is usually about one of them. "Why did this song
get rejected in March but not in May" is a QC engine question. "Why did
we stop retrying so much" is a policy question. One combined version
would answer neither.

Every version changes when *behaviour* changes — a threshold, a
severity, an ordering rule, a budget default. None of them changes for a
refactor, a docstring, or a new measurement that drives no decision.
That distinction is the whole value: a version that moved on every
commit would tell nobody anything.
"""

from __future__ import annotations

#: The shape of the persisted trace. Bumped when a field is removed or
#: reinterpreted, not when one is added.
QC_SCHEMA_VERSION = "luber-inference-qc/1"

#: The checks and their thresholds. Bumped when a candidate that used to
#: pass would now fail, or the reverse.
QC_ENGINE_VERSION = "qc-v1"

#: Ranking and eligibility. Bumped when the same set of candidates would
#: now produce a different winner.
CANDIDATE_SELECTION_VERSION = "sel-v1"

#: Budgets, profiles, and what justifies a retry. Bumped when the same
#: findings would now produce a different number of provider calls.
RETRY_POLICY_VERSION = "retry-v1"


def version_block() -> dict[str, str]:
    """The four versions, for stamping onto a trace."""
    return {
        "qc_schema_version": QC_SCHEMA_VERSION,
        "qc_engine_version": QC_ENGINE_VERSION,
        "candidate_selection_version": CANDIDATE_SELECTION_VERSION,
        "retry_policy_version": RETRY_POLICY_VERSION,
    }


__all__ = [
    "CANDIDATE_SELECTION_VERSION",
    "QC_ENGINE_VERSION",
    "QC_SCHEMA_VERSION",
    "RETRY_POLICY_VERSION",
    "version_block",
]
