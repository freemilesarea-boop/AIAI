"""Judging one candidate: measure, check, decide eligibility, score.

The one entry point everything else uses. It exists so that the order of
those four steps is written down once — measure before checking, check
before deciding eligibility, and score *only* what survived — rather
than reassembled by each caller in whatever order seemed natural.

The order is the safety property. A caller that scored first and gated
second would have produced a number for a broken candidate, and a number
that exists is a number something will eventually compare.
"""

from __future__ import annotations

import time
from pathlib import Path

from luber_inference_qc.candidate import CandidateGeneration, CandidateStatus
from luber_inference_qc.checks import RequestExpectation, run_checks
from luber_inference_qc.detectors import VocalPresenceDetector
from luber_inference_qc.findings import Finding, QCFinding, Severity
from luber_inference_qc.measurement import (
    CandidateMeasurement,
    MeasurementCache,
    MeasurementError,
    measure,
)
from luber_inference_qc.scoring import score
from luber_inference_qc.selector import assess_eligibility


def judge(
    candidate: CandidateGeneration,
    audio_path: Path,
    expectation: RequestExpectation,
    *,
    detector: VocalPresenceDetector | None = None,
    cache: MeasurementCache | None = None,
    sha256: str | None = None,
) -> CandidateMeasurement | None:
    """Measure and judge one candidate in place.

    Returns the measurement, or ``None`` when the file could not be
    decoded at all — in which case the candidate carries
    ``INVALID_AUDIO`` and is rejected. A decode failure is the one case
    where there is nothing to measure, so it cannot be expressed as a
    finding derived from a measurement.
    """
    started = time.monotonic()
    candidate.status = CandidateStatus.ANALYZING.value

    try:
        measurement = measure(
            audio_path,
            detector=detector,
            cache=cache,
            sha256=sha256,
            # Tempo and key estimation cost a second pass and are only
            # consulted when the request asked for one of them.
            measure_musical=expectation.bpm is not None or bool(expectation.key_scale),
        )
    except MeasurementError as exc:
        candidate.findings = [
            QCFinding(
                code=Finding.INVALID_AUDIO.value,
                severity=Severity.CRITICAL.value,
                detail=str(exc),
                metric="decode",
            )
        ]
        candidate.status = CandidateStatus.REJECTED.value
        candidate.qc_seconds = time.monotonic() - started
        return None

    candidate.raw_sha256 = measurement.sha256
    candidate.duration_seconds = measurement.duration_seconds
    candidate.sample_rate = measurement.analysis.technical.sample_rate
    candidate.channels = measurement.analysis.technical.channels
    candidate.findings = run_checks(measurement, expectation)

    verdict = assess_eligibility(candidate)
    if verdict.eligible:
        candidate.status = CandidateStatus.ELIGIBLE.value
        breakdown = score(measurement, candidate.findings, expectation)
        candidate.technical_selection_score = breakdown.total
        candidate.score_components = breakdown.components
    else:
        candidate.status = CandidateStatus.REJECTED.value
        # No score. A rejected candidate does not get a number that
        # something could later compare against a working one.
        candidate.technical_selection_score = None
        candidate.score_components = {}

    candidate.qc_seconds = time.monotonic() - started
    return measurement


__all__ = ["judge"]
