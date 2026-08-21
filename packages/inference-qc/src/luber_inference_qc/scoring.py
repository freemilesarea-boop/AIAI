"""A technical score, and everything it deliberately is not.

It is called `technical_selection_score` because that is all it is: a
number for ordering candidates that have already been judged
deliverable, on the axes this repository can actually measure. It is not
a quality score. Nothing in it knows whether a song is good, and a
candidate that scores 0.94 is not "better music" than one at 0.81 — it
is closer to the requested duration with fewer measurable defects.

Two rules keep it honest.

**Rejected candidates do not get a score.** Eligibility is decided
first, and only eligible candidates are scored. The alternative — a
score low enough that a broken candidate loses — puts invalid audio in
the same ranking as a working song and relies on arithmetic to keep it
out. Arithmetic that could be wrong, on a rule that must not be.

**Every component is recorded.** The score is a weighted sum of named
parts, each stored on the candidate. A ranking that could only be read
and not checked would be an authority rather than an argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from luber_inference_qc.checks import RequestExpectation
from luber_inference_qc.findings import QCFinding, Severity
from luber_inference_qc.measurement import CandidateMeasurement

#: What each component contributes. Duration dominates because it is the
#: one request parameter this engine can check precisely, and a song of
#: the wrong length is wrong in a way a listener notices immediately.
WEIGHTS: dict[str, float] = {
    "duration_accuracy": 0.35,
    "control_adherence": 0.20,
    "level_safety": 0.15,
    "collapse_safety": 0.15,
    "phase_safety": 0.10,
    "spectral_safety": 0.05,
}

#: Penalty per finding when computing the defect components. MINOR costs
#: little; MAJOR costs enough to lose a close ranking; CRITICAL never
#: appears here because a critical finding means the candidate was not
#: scored at all.
SEVERITY_PENALTY: dict[str, float] = {
    Severity.INFO.value: 0.0,
    Severity.MINOR.value: 0.10,
    Severity.MAJOR.value: 0.45,
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _penalty(findings: list[QCFinding], codes: set[str]) -> float:
    """1.0 with nothing wrong, falling as findings in *codes* accumulate."""
    total = sum(SEVERITY_PENALTY.get(item.severity, 0.0) for item in findings if item.code in codes)
    return _clamp(1.0 - total)


@dataclass(frozen=True)
class ScoreBreakdown:
    """The score and the parts it was made of."""

    total: float
    components: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "technical_selection_score": round(self.total, 4),
            "components": {name: round(value, 4) for name, value in self.components.items()},
            "weights": dict(WEIGHTS),
        }


def duration_accuracy(measurement: CandidateMeasurement, expectation: RequestExpectation) -> float:
    """1.0 for an exact match, falling to 0.0 at the hard tolerance.

    Returns 1.0 when no duration was requested: a candidate cannot be
    penalised for missing a target nobody set, and giving it 0.5 would
    make requests-without-durations rank below requests with them for no
    reason.
    """
    from luber_inference_qc import thresholds as t

    requested = expectation.duration_seconds
    if requested is None or requested <= 0:
        return 1.0
    relative = abs(measurement.duration_seconds - requested) / requested
    return _clamp(1.0 - (relative / t.DURATION_HARD_TOLERANCE_RATIO))


def score(
    measurement: CandidateMeasurement,
    findings: list[QCFinding],
    expectation: RequestExpectation,
) -> ScoreBreakdown:
    """Rank one eligible candidate. Never called on a rejected one."""
    from luber_inference_qc.findings import Finding

    components = {
        "duration_accuracy": duration_accuracy(measurement, expectation),
        "control_adherence": _penalty(
            findings,
            {
                Finding.CONTROL_BPM_MISMATCH.value,
                Finding.CONTROL_KEY_MISMATCH.value,
                Finding.CONTROL_VOCAL_MISMATCH.value,
            },
        ),
        "level_safety": _penalty(
            findings,
            {
                Finding.PEAK_OVERSHOOT.value,
                Finding.DC_OFFSET.value,
                Finding.NEAR_SILENT.value,
                Finding.EXCESSIVE_SILENCE.value,
            },
        ),
        "collapse_safety": _clamp(measurement.collapse.content_ratio),
        "phase_safety": _penalty(
            findings,
            {
                Finding.PHASE_UNSAFE.value,
                Finding.LOW_END_PHASE_RISK.value,
                Finding.CHANNEL_IMBALANCE.value,
            },
        ),
        # Narrow stereo and the harshness proxies are INFO, so they cost
        # nothing here. They are recorded because Phase 22 may act on
        # them, not because they are defects.
        "spectral_safety": _penalty(
            findings,
            {
                Finding.HIGH_HARSHNESS_PROXY.value,
                Finding.HIGH_SIBILANCE_PROXY.value,
                Finding.NARROW_STEREO.value,
            },
        ),
    }
    total = sum(WEIGHTS[name] * value for name, value in components.items())
    return ScoreBreakdown(total=_clamp(total), components=components)


__all__ = ["SEVERITY_PENALTY", "WEIGHTS", "ScoreBreakdown", "duration_accuracy", "score"]
