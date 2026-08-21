"""The record that explains why this file was delivered.

Written so that six months later somebody can answer, from the trace
alone: why did this generation retry, why did candidate A lose, why was
B selected, how many provider calls did it cost, and what did the
finishing engine do afterwards.

Two things it deliberately does not contain.

**No musical score.** There is `technical_selection_score` with its
components, and nothing that claims the delivered song is good.

**No local paths.** Candidate audio lives in a worker's temporary
directory and is gone by the time anyone reads this. Recording the path
would describe a file that no longer exists on a machine the reader does
not have; the SHA-256 is recorded instead, which is what would let a
future phase recognise the bytes if it ever kept them.

The trace is written as the run proceeds rather than assembled at the
end, so a crash between the provider returning and QC finishing still
leaves a record that the call was made — and the resume path can reuse
that candidate instead of paying for another.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from luber_inference_qc.candidate import CandidateGeneration
from luber_inference_qc.policy import Budget, CandidatePolicy
from luber_inference_qc.selector import Selection
from luber_inference_qc.versions import version_block


class Outcome:
    """How the candidate phase ended.

    Deliberately not an enum shared with `GenerationStatus`: what a
    customer sees and why the controller stopped are different
    vocabularies, and merging them is how internal retry mechanics end
    up on a customer's screen.
    """

    SELECTED = "SELECTED"
    #: Every attempt was rejected and the budget allowed no more.
    RETRY_EXHAUSTED = "RETRY_EXHAUSTED"
    #: Every attempt was rejected and nothing further would have helped.
    ALL_CANDIDATES_REJECTED = "ALL_CANDIDATES_REJECTED"
    #: The provider could not be used at all.
    PROVIDER_FAILED = "PROVIDER_FAILED"


@dataclass
class QCTrace:
    """Everything one generation's candidate phase did."""

    generation_id: str
    request_sha256: str
    policy: CandidatePolicy
    candidates: list[CandidateGeneration] = field(default_factory=list)
    selection: Selection | None = None
    outcome: str = Outcome.SELECTED
    #: Why it ended, in the words the planner or the budget used.
    outcome_detail: str = ""
    #: What the finishing engine did with the winner, recorded here as
    #: well as in the finishing trace so one document answers the whole
    #: question.
    finishing_outcome: str | None = None
    base_seed: int | None = None
    timings: dict[str, float] = field(default_factory=dict)

    def add(self, candidate: CandidateGeneration) -> None:
        self.candidates.append(candidate)

    @property
    def selected_candidate_id(self) -> str | None:
        return self.selection.winner_candidate_id if self.selection else None

    @property
    def exhausted(self) -> bool:
        return self.outcome == Outcome.RETRY_EXHAUSTED

    def to_dict(self, budget: Budget | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            **version_block(),
            "generation_id": self.generation_id,
            "request_sha256": self.request_sha256,
            "base_seed": self.base_seed,
            "policy": self.policy.to_dict(),
            "attempts": [candidate.to_dict() for candidate in self.candidates],
            "selected_candidate_id": self.selected_candidate_id,
            "selection": self.selection.to_dict() if self.selection else None,
            "outcome": self.outcome,
            "outcome_detail": self.outcome_detail,
            "finishing_outcome": self.finishing_outcome,
            "exhausted": self.exhausted,
            "timings": {name: round(value, 3) for name, value in self.timings.items()},
        }
        if budget is not None:
            payload["budget"] = budget.to_dict()
        return payload

    def to_json(self, budget: Budget | None = None) -> str:
        return json.dumps(self.to_dict(budget), sort_keys=True, ensure_ascii=False)


def summarise(trace: dict[str, Any]) -> dict[str, Any]:
    """The short form, for an operator list or a metrics counter.

    Deliberately lossy and never the thing that gets stored: the whole
    trace is what answers a question, and a summary that replaced it
    would have to guess in advance which question that is.
    """
    attempts = trace.get("attempts", []) or []
    critical: list[str] = []
    for attempt in attempts:
        critical.extend(
            finding["code"]
            for finding in attempt.get("findings", []) or []
            if finding.get("severity") == "CRITICAL"
        )
    return {
        "generation_id": trace.get("generation_id"),
        "attempts": len(attempts),
        "provider_calls": (trace.get("budget") or {}).get("provider_calls_used"),
        "retries": max(0, len(attempts) - 1),
        "outcome": trace.get("outcome"),
        "selected_candidate_id": trace.get("selected_candidate_id"),
        "critical_findings": sorted(set(critical)),
        "policy": (trace.get("policy") or {}).get("name"),
    }


__all__ = ["Outcome", "QCTrace", "summarise"]
