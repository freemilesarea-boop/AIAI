"""Global human verdicts and two-stage triage scoring.

Phase 5 exposed a gap in the evaluation system: when a baseline is bad
enough to reject outright, forcing an evaluator through 15 numeric
fields per track wastes their time and produces noise rather than
signal. The rubric assumed the output was worth grading.

Two mechanisms fix that:

* :class:`GlobalVerdict` — one honest judgement over a whole baseline,
  with structured findings, when per-track scoring is not worth doing.
* Two-stage triage — a single overall score first; the detailed rubric
  only unlocks at >= 5, where fine-grained distinctions start to mean
  something.

Neither mechanism invents per-track numbers. An absent score stays
absent.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: Stage-A triage anchors. Below TRIAGE_DETAIL_THRESHOLD the detailed
#: rubric is optional, because differences between "unusable" and "very
#: poor" carry no product information.
TRIAGE_SCALE: dict[int, str] = {
    1: "unusable",
    2: "very poor",
    3: "poor",
    4: "demo only",
    5: "mediocre",
    6: "usable with major edits",
    7: "usable with edits",
    8: "commercially releasable",
    9: "professional",
    10: "top-tier",
}
TRIAGE_DETAIL_THRESHOLD = 5

#: Failure tags derived from the actual Phase 5 human evaluation. These
#: extend, and do not replace, the artifact taxonomy in scoring.py.
HUMAN_FAILURE_TAGS: tuple[str, ...] = (
    "FREQUENCY_BALANCE_BAD",
    "HIGH_END_OVERBOOST",
    "EXCESSIVE_SIBILANCE",
    "INSTRUMENT_FIDELITY_LOW",
    "TEXTURE_LOW_QUALITY",
    "KOREAN_LINE_OMISSION",
    "LYRIC_LINE_SKIP",
    "TROT_LIKE_VOCAL",
    "VOCAL_STYLE_OUTDATED",
)


class VerdictError(Exception):
    """Raised when a verdict or triage score is not usable."""


@dataclass
class GlobalVerdict:
    """One human judgement covering an entire baseline.

    Used when the evaluator reviewed the set and rejected it before
    per-track scoring became worthwhile.
    """

    baseline_id: str
    evaluator: str
    recorded_at: str
    tracks_reviewed: int
    tracks_accepted: int
    tracks_rejected: int
    overall_score: int
    commercially_usable: bool
    reason: str
    findings: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @property
    def acceptance_rate(self) -> float:
        return self.tracks_accepted / self.tracks_reviewed if self.tracks_reviewed else 0.0


def validate_verdict(verdict: GlobalVerdict) -> GlobalVerdict:
    """Check a global verdict for internal consistency."""
    if not 1 <= verdict.overall_score <= 10:
        raise VerdictError(f"overall_score {verdict.overall_score} outside 1-10")
    if verdict.tracks_reviewed <= 0:
        raise VerdictError("tracks_reviewed must be positive")
    if verdict.tracks_accepted + verdict.tracks_rejected != verdict.tracks_reviewed:
        raise VerdictError(
            "accepted + rejected must equal reviewed "
            f"({verdict.tracks_accepted} + {verdict.tracks_rejected} "
            f"!= {verdict.tracks_reviewed})"
        )
    if verdict.tracks_accepted < 0 or verdict.tracks_rejected < 0:
        raise VerdictError("track counts cannot be negative")
    if not verdict.reason.strip():
        raise VerdictError("a verdict must state its reason")
    # A fully-rejected set cannot also be commercially usable.
    if verdict.commercially_usable and verdict.tracks_accepted == 0:
        raise VerdictError("commercially_usable is true but no track was accepted")
    return verdict


def validate_triage(
    overall_score: int,
    *,
    detailed_scores: dict[str, int] | None = None,
    reject_tags: list[str] | None = None,
) -> None:
    """Validate a stage-A triage submission.

    Below the threshold the detailed rubric is optional but at least one
    reject tag is required, so a low score still carries a reason.
    """
    if not 1 <= overall_score <= 10:
        raise VerdictError(f"overall_score {overall_score} outside 1-10")

    if overall_score < TRIAGE_DETAIL_THRESHOLD:
        if not reject_tags:
            raise VerdictError(
                f"a score of {overall_score} requires at least one reject reason tag"
            )
    elif detailed_scores is None:
        raise VerdictError(
            f"a score of {overall_score} is at or above the detail threshold "
            f"({TRIAGE_DETAIL_THRESHOLD}); the full rubric is required"
        )


def validate_failure_tags(tags: list[str], *, allowed: tuple[str, ...] | None = None) -> list[str]:
    """Reject tags outside the combined taxonomy."""
    from bench.scoring import ARTIFACT_TAGS

    vocabulary = set(allowed or (HUMAN_FAILURE_TAGS + ARTIFACT_TAGS))
    unknown = sorted(set(tags) - vocabulary)
    if unknown:
        raise VerdictError(f"unknown failure tags: {', '.join(unknown)}")
    return tags


class VerdictStore:
    """Append-only JSONL store of global verdicts."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> list[dict[str, Any]]:
        if not self._path.is_file():
            return []
        out: list[dict[str, Any]] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def append(self, verdict: GlobalVerdict) -> None:
        validate_verdict(verdict)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(verdict.to_json() + "\n")

    def latest_for(self, baseline_id: str) -> dict[str, Any] | None:
        matches = [v for v in self.load() if v.get("baseline_id") == baseline_id]
        return matches[-1] if matches else None
