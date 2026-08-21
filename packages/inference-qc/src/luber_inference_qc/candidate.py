"""One attempt at a generation, and the two things that can be said about it.

A candidate has two states because there are two independent questions:
*did this attempt produce something* and *did it win*. Collapsing them
would make "we generated it and it was fine but another one was better"
indistinguishable from "we generated it and it was broken", and those
call for opposite responses when a retry spike shows up in the metrics.

The entity deliberately does not overload `GenerationStatus`. That enum
is what a customer sees, and a customer has no business knowing there
were three attempts — see the phase brief's rule about not exposing
internal retry mechanics.

Candidate audio is not an asset. It lives in the worker's temporary
directory and only the winner is ever uploaded, so what persists for a
loser is this record: seed, digest, findings, why it lost. That is
enough to explain any delivery and cannot leak a broken file into a
library.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from luber_inference_qc.findings import QCFinding
from luber_inference_qc.versions import QC_SCHEMA_VERSION


def now() -> str:
    return datetime.now(UTC).isoformat()


class CandidateStatus(StrEnum):
    """What happened to this attempt."""

    GENERATING = "GENERATING"
    GENERATED = "GENERATED"
    ANALYZING = "ANALYZING"
    #: Measured, and nothing critical. May be selected.
    ELIGIBLE = "ELIGIBLE"
    #: Measured, and something critical. Cannot be selected, whatever
    #: its score — see `scoring`: it does not get one.
    REJECTED = "REJECTED"
    #: The provider call itself did not return audio.
    FAILED = "FAILED"


class SelectionStatus(StrEnum):
    UNDECIDED = "UNDECIDED"
    SELECTED = "SELECTED"
    NOT_SELECTED = "NOT_SELECTED"


class CallAttribution(StrEnum):
    """Why this provider call happened.

    Recorded per candidate so a future cost model can tell a user's
    request apart from the system's own retry. Phase 29 implements no
    billing; it makes the distinction available before it is needed,
    which is the only time it can be recorded accurately.
    """

    USER_REQUEST = "USER_REQUEST"
    QUALITY_RETRY = "QUALITY_RETRY"


@dataclass
class CandidateGeneration:
    """One attempt, from the call that made it to the reason it lost."""

    candidate_id: str
    generation_id: str
    attempt_index: int
    request_sha256: str
    attribution: str = CallAttribution.USER_REQUEST.value
    seed: int | None = None
    #: What the provider was actually sent, where the provider can
    #: describe itself. Distinct from `request_sha256`, which is what the
    #: user asked for.
    provider_request_sha256: str | None = None

    status: str = CandidateStatus.GENERATING.value
    selection_status: str = SelectionStatus.UNDECIDED.value

    #: Local path while the run is in flight. Never persisted to the
    #: trace: it is a path on one machine's temporary disk.
    audio_path: Path | None = None
    raw_sha256: str | None = None
    duration_seconds: float | None = None
    sample_rate: int | None = None
    channels: int | None = None

    findings: list[QCFinding] = field(default_factory=list)
    #: Named components of the technical score, so a ranking can be
    #: argued with rather than only read.
    score_components: dict[str, float] = field(default_factory=dict)
    technical_selection_score: float | None = None

    #: Why this attempt happened, when it was a retry.
    retry_reason: str | None = None
    #: The candidate whose failure caused this one to be generated.
    parent_candidate_id: str | None = None
    #: Why it did not win, when it was eligible and lost anyway.
    not_selected_reason: str | None = None

    provider_error_code: str | None = None
    provider_seconds: float | None = None
    qc_seconds: float | None = None

    created_at: str = field(default_factory=now)
    schema_version: str = QC_SCHEMA_VERSION

    @property
    def eligible(self) -> bool:
        return self.status == CandidateStatus.ELIGIBLE.value

    @property
    def critical_findings(self) -> list[QCFinding]:
        from luber_inference_qc.findings import critical

        return critical(self.findings)

    def finding_codes(self) -> set[str]:
        return {item.code for item in self.findings}

    def to_dict(self) -> dict[str, Any]:
        """The persisted form. Local paths are deliberately absent."""
        payload = asdict(self)
        payload.pop("audio_path", None)
        payload["findings"] = [item.to_dict() for item in self.findings]
        return payload


__all__ = [
    "CallAttribution",
    "CandidateGeneration",
    "CandidateStatus",
    "SelectionStatus",
    "now",
]
