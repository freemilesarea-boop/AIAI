"""Eligibility, then ranking. In that order, and never merged.

Eligibility is a gate: a candidate with a critical finding cannot be
delivered, whatever else is true about it. Ranking is a comparison
between candidates that all passed the gate.

Keeping them apart is the point. A single scoring function that gave
invalid audio a low number would put it in the same ordering as a
working song and rely on arithmetic to keep it from winning — and
arithmetic is exactly the thing that can be wrong. A gate cannot rank
a rejected candidate first, because it never reaches the ranking.

The ordering below is deterministic and total. The same candidates
always produce the same winner, including the tiebreak: after every
measured axis has been compared, attempt order decides. Nothing is
randomised after generation, because a selection an operator cannot
reproduce is a selection they cannot audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from luber_inference_qc.candidate import CandidateGeneration, SelectionStatus
from luber_inference_qc.findings import Severity, by_severity


@dataclass(frozen=True)
class EligibilityVerdict:
    eligible: bool
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"eligible": self.eligible, "reasons": self.reasons}


def assess_eligibility(candidate: CandidateGeneration) -> EligibilityVerdict:
    """Whether this candidate may be delivered at all.

    Three conditions, and all of them are about facts rather than
    degrees: the provider returned audio, the file hashed, and nothing
    critical was found. A MAJOR finding does not block — it costs the
    candidate heavily in ranking, which is the right response to
    "measurably worse" as opposed to "cannot be shipped".
    """
    reasons: list[str] = []

    if candidate.raw_sha256 is None:
        reasons.append("no audio was produced for this attempt")
    if candidate.duration_seconds is None:
        reasons.append("the audio could not be measured")

    for finding in candidate.critical_findings:
        reasons.append(f"{finding.code}: {finding.detail}")

    return EligibilityVerdict(eligible=not reasons, reasons=reasons)


#: The comparison, in order. Each entry maps a candidate to a value where
#: **lower sorts first**, so the tuple can be handed straight to `sorted`.
#: Written as a list rather than one lambda so the order is legible and
#: so the trace can name which key separated two candidates.
def _sort_key(candidate: CandidateGeneration) -> tuple[float, ...]:
    major = len(by_severity(candidate.findings, Severity.MAJOR))
    minor = len(by_severity(candidate.findings, Severity.MINOR))
    control = candidate.score_components.get("control_adherence", 0.0)
    duration = candidate.score_components.get("duration_accuracy", 0.0)
    total = candidate.technical_selection_score or 0.0
    return (
        # 1. Fewest major findings. A candidate with a measurable control
        #    failure loses to one without, before any score is compared.
        float(major),
        # 2. Closest control adherence.
        -control,
        # 3. Closest duration.
        -duration,
        # 4. Fewest minor findings.
        float(minor),
        # 5. Highest technical score.
        -total,
        # 6. Attempt order. Deterministic, and it means the first healthy
        #    candidate wins a tie — which is also the cheapest outcome.
        float(candidate.attempt_index),
    )


@dataclass(frozen=True)
class Selection:
    """Who won, in what order, and why."""

    winner_candidate_id: str | None
    ranking: list[str]
    reasons: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "winner_candidate_id": self.winner_candidate_id,
            "ranking": self.ranking,
            "reasons": self.reasons,
        }


def select(candidates: list[CandidateGeneration]) -> Selection:
    """Rank the eligible candidates and pick the first.

    Mutates the candidates' `selection_status` so the trace records the
    outcome on each one rather than only in a separate ranking — a
    candidate that says NOT_SELECTED and carries the reason is readable
    on its own.
    """
    eligible = [item for item in candidates if item.eligible]
    for candidate in candidates:
        if not candidate.eligible:
            candidate.selection_status = SelectionStatus.NOT_SELECTED.value
            candidate.not_selected_reason = "not eligible: " + "; ".join(
                finding.code for finding in candidate.critical_findings
            )

    if not eligible:
        return Selection(winner_candidate_id=None, ranking=[], reasons={})

    ordered = sorted(eligible, key=_sort_key)
    winner = ordered[0]
    winner.selection_status = SelectionStatus.SELECTED.value
    winner.not_selected_reason = None

    reasons: dict[str, str] = {
        winner.candidate_id: _win_reason(winner, ordered[1] if len(ordered) > 1 else None)
    }
    for loser in ordered[1:]:
        loser.selection_status = SelectionStatus.NOT_SELECTED.value
        loser.not_selected_reason = _loss_reason(loser, winner)
        reasons[loser.candidate_id] = loser.not_selected_reason

    return Selection(
        winner_candidate_id=winner.candidate_id,
        ranking=[item.candidate_id for item in ordered],
        reasons=reasons,
    )


def _win_reason(winner: CandidateGeneration, runner_up: CandidateGeneration | None) -> str:
    if runner_up is None:
        return "the only eligible candidate"
    key = _separating_key(winner, runner_up)
    return f"ranked first on {key}"


def _loss_reason(loser: CandidateGeneration, winner: CandidateGeneration) -> str:
    key = _separating_key(winner, loser)
    if key == "attempt order":
        return "tied with the winner on every measured axis; the earlier attempt was kept"
    return f"lost to {winner.candidate_id} on {key}"


#: Names for the sort key positions, so a reason can say which axis
#: actually separated two candidates rather than gesturing at the score.
_KEY_NAMES = (
    "major finding count",
    "control adherence",
    "duration accuracy",
    "minor finding count",
    "technical selection score",
    "attempt order",
)


def _separating_key(first: CandidateGeneration, second: CandidateGeneration) -> str:
    left = _sort_key(first)
    right = _sort_key(second)
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        if a != b:
            return _KEY_NAMES[index]
    return "attempt order"


__all__ = ["EligibilityVerdict", "Selection", "assess_eligibility", "select"]
