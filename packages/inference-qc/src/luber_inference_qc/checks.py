"""Every check, and the finding it produces.

The organising rule is the one from the phase brief: judge only what can
be honestly measured. Concretely, three habits run through the file.

**A check that cannot establish an answer says so.** It does not skip
silently and it does not default to a pass. `CONTROL_NOT_MEASURABLE`
appears in the trace as often as any finding, and it is a real result.

**Critical is reserved for undeliverable.** A dark mix, a narrow image,
a harsh top end and a small peak overshoot are all recorded and none of
them rejects anything — the first is a production choice and the other
three are what Phase 22 exists to correct. Rejecting a candidate for a
defect the next stage repairs would spend an inference to avoid a
problem that was about to be solved.

**Positional evidence is required for positional claims.** A whole-file
silence ratio cannot distinguish a quiet outro from a generation that
stopped early, so `EARLY_COLLAPSE` is raised from `collapse`, which
measures where content actually ends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from luber_audio_finishing import RiskFlag
from luber_inference_qc import thresholds as t
from luber_inference_qc.collapse import MINIMUM_TRAILING_SILENCE_SECONDS
from luber_inference_qc.detectors import VocalPresence
from luber_inference_qc.findings import Finding, QCFinding, Severity
from luber_inference_qc.measurement import CandidateMeasurement


def _ratio_to_dbfs(ratio: float) -> float:
    """A full-scale ratio as dBFS.

    So a finding cites the units it measured in rather than two
    different ones: the thresholds are expressed as ratios because that
    is how the Phase 5 benchmark expressed them, and peaks are reported
    in dBFS because that is how anyone reads them.
    """
    return 20.0 * math.log10(ratio)


@dataclass(frozen=True)
class RequestExpectation:
    """What the request asked for, as far as QC can check it.

    Everything is optional because a request need not state any of it.
    An expectation that was never expressed is not a failure to meet.
    """

    duration_seconds: float | None = None
    bpm: int | None = None
    key_scale: str | None = None
    instrumental: bool | None = None
    #: Long-form material behaves differently — a ten-minute piece has
    #: more room for structured silence than a ninety-second one — and
    #: the collapse check widens its trailing-silence allowance for it
    #: rather than applying a short-song number.
    long_form_threshold_seconds: float = 300.0

    @property
    def is_long_form(self) -> bool:
        return (
            self.duration_seconds is not None
            and self.duration_seconds >= self.long_form_threshold_seconds
        )


def _finding(
    code: Finding,
    severity: Severity,
    detail: str,
    *,
    metric: str = "",
    measured: float | None = None,
    threshold: float | None = None,
    not_measurable: bool = False,
    evidence: dict[str, Any] | None = None,
) -> QCFinding:
    return QCFinding(
        code=code.value,
        severity=severity.value,
        detail=detail,
        metric=metric,
        measured=measured,
        threshold=threshold,
        not_measurable=not_measurable,
        evidence=evidence or {},
    )


# ── structural ───────────────────────────────────────────────────────


def check_structure(measurement: CandidateMeasurement) -> list[QCFinding]:
    """Samples that are numbers, in a file that decoded.

    A decode failure never reaches here — it raises in `measurement` and
    the controller records `INVALID_AUDIO` directly. What this catches is
    a file that parsed and holds NaN or infinity, which makes every other
    measurement in this module meaningless.
    """
    if measurement.non_finite_samples > 0:
        return [
            _finding(
                Finding.NON_FINITE_SAMPLES,
                Severity.CRITICAL,
                f"{measurement.non_finite_samples} sample(s) are not finite numbers; "
                "nothing measured about this file can be trusted",
                metric="non_finite_samples",
                measured=float(measurement.non_finite_samples),
                threshold=0.0,
            )
        ]
    return []


# ── duration ─────────────────────────────────────────────────────────


def check_duration(
    measurement: CandidateMeasurement, expectation: RequestExpectation
) -> list[QCFinding]:
    """Requested against actual, at two tolerances that already existed."""
    requested = expectation.duration_seconds
    if requested is None or requested <= 0:
        return []

    actual = measurement.duration_seconds
    error = actual - requested
    relative = abs(error) / requested
    short = error < 0
    code = Finding.DURATION_SHORT if short else Finding.DURATION_LONG
    evidence = {
        "requested_seconds": round(requested, 3),
        "actual_seconds": round(actual, 3),
        "absolute_error_seconds": round(abs(error), 3),
        "relative_error": round(relative, 4),
    }

    if relative > t.DURATION_HARD_TOLERANCE_RATIO:
        return [
            _finding(
                code,
                Severity.CRITICAL,
                f"{actual:.1f}s against a request for {requested:.1f}s — "
                f"{relative:.0%} out, past the {t.DURATION_HARD_TOLERANCE_RATIO:.0%} point "
                "where the provider has not answered the question that was asked",
                metric="duration_relative_error",
                measured=relative,
                threshold=t.DURATION_HARD_TOLERANCE_RATIO,
                evidence=evidence,
            )
        ]
    if relative > t.DURATION_SOFT_TOLERANCE_RATIO:
        return [
            _finding(
                code,
                Severity.MINOR,
                f"{actual:.1f}s against a request for {requested:.1f}s ({relative:.1%} out)",
                metric="duration_relative_error",
                measured=relative,
                threshold=t.DURATION_SOFT_TOLERANCE_RATIO,
                evidence=evidence,
            )
        ]
    return []


# ── level ────────────────────────────────────────────────────────────


def check_silence(measurement: CandidateMeasurement) -> list[QCFinding]:
    """Silent, near-silent, or unusually gappy."""
    level = measurement.analysis.level
    peak_ratio = 10.0 ** (level.peak_dbfs / 20.0)
    findings: list[QCFinding] = []

    if measurement.collapse.entirely_silent or peak_ratio < t.SILENCE_PEAK_RATIO:
        return [
            _finding(
                Finding.SILENT_OUTPUT,
                Severity.CRITICAL,
                f"peak is {level.peak_dbfs:.1f} dBFS — the file carries no audible content",
                metric="level.peak_dbfs",
                measured=level.peak_dbfs,
                threshold=_ratio_to_dbfs(t.SILENCE_PEAK_RATIO),
                evidence={"silence_ratio": round(level.silence_ratio, 4)},
            )
        ]

    if peak_ratio < t.NEAR_SILENCE_PEAK_RATIO:
        findings.append(
            _finding(
                Finding.NEAR_SILENT,
                Severity.CRITICAL,
                f"peak is {level.peak_dbfs:.1f} dBFS — audible only as noise, and raising it "
                "would raise the noise with it",
                metric="level.peak_dbfs",
                measured=level.peak_dbfs,
                threshold=_ratio_to_dbfs(t.NEAR_SILENCE_PEAK_RATIO),
            )
        )

    if level.silence_ratio > t.EXCESSIVE_SILENCE_RATIO:
        findings.append(
            _finding(
                Finding.EXCESSIVE_SILENCE,
                Severity.MAJOR,
                f"{level.silence_ratio:.0%} of the track is below -60 dBFS",
                metric="level.silence_ratio",
                measured=level.silence_ratio,
                threshold=t.EXCESSIVE_SILENCE_RATIO,
            )
        )
    return findings


def check_collapse(
    measurement: CandidateMeasurement, expectation: RequestExpectation
) -> list[QCFinding]:
    """Content that stops long before the file does.

    Two comparisons, deliberately separate. Against the *file* this
    catches a generation padded with digital silence; against the
    *request* it catches one that gave up early and returned a short
    file — though that case usually shows as a duration failure first.
    """
    collapse = measurement.collapse
    if collapse.entirely_silent:
        return []  # already reported as SILENT_OUTPUT; not two findings for one fact

    # Long-form pieces earn more room: a ten-minute track with a
    # twenty-second outro is not collapsing.
    allowance = MINIMUM_TRAILING_SILENCE_SECONDS
    if expectation.is_long_form:
        allowance = max(allowance, collapse.file_duration_seconds * 0.05)

    if collapse.trailing_silence_seconds > allowance and collapse.trailing_silence_ratio > 0.15:
        return [
            _finding(
                Finding.EARLY_COLLAPSE,
                Severity.CRITICAL,
                f"content ends at {collapse.content_end_seconds:.1f}s of "
                f"{collapse.file_duration_seconds:.1f}s, leaving "
                f"{collapse.trailing_silence_seconds:.1f}s of silence",
                metric="collapse.trailing_silence_ratio",
                measured=collapse.trailing_silence_ratio,
                threshold=0.15,
                evidence=collapse.to_dict(),
            )
        ]
    return []


def check_clipping(measurement: CandidateMeasurement) -> list[QCFinding]:
    """A peak the limiter handles, or distortion baked into the samples.

    The distinction is the share of samples at full scale. A few is a
    hot master and Phase 22 pulls the ceiling down; a thousandth of the
    file is a signal that was already destroyed when it was written, and
    limiting it only makes the distortion quieter.
    """
    level = measurement.analysis.level
    technical = measurement.analysis.technical
    total = max(1, technical.frames * technical.channels)
    ratio = level.clipped_samples / total

    if ratio >= t.SEVERE_CLIPPING_SAMPLE_RATIO:
        return [
            _finding(
                Finding.SEVERE_CLIPPING,
                Severity.CRITICAL,
                f"{level.clipped_samples} samples ({ratio:.3%}) are at or beyond full scale; "
                "the distortion is in the source and a limiter would only make it quieter",
                metric="clipped_sample_ratio",
                measured=ratio,
                threshold=t.SEVERE_CLIPPING_SAMPLE_RATIO,
                evidence={"peak_dbfs": round(level.peak_dbfs, 2)},
            )
        ]

    if level.peak_dbfs >= t.PEAK_OVERSHOOT_DBFS:
        return [
            _finding(
                Finding.PEAK_OVERSHOOT,
                Severity.MINOR,
                f"peak is {level.peak_dbfs:.2f} dBFS; the finishing limiter handles this",
                metric="level.peak_dbfs",
                measured=level.peak_dbfs,
                threshold=t.PEAK_OVERSHOOT_DBFS,
                evidence={"clipped_samples": level.clipped_samples},
            )
        ]
    return []


def check_dc(measurement: CandidateMeasurement) -> list[QCFinding]:
    """Offset that eats headroom, or offset the engine removes."""
    offset = measurement.analysis.level.dc_offset
    if offset > t.DC_OFFSET_SEVERE:
        return [
            _finding(
                Finding.DC_OFFSET,
                Severity.CRITICAL,
                f"DC offset of {offset:.4f} is large enough to be consuming headroom rather "
                "than sitting under the music",
                metric="level.dc_offset",
                measured=offset,
                threshold=t.DC_OFFSET_SEVERE,
            )
        ]
    if offset > t.DC_OFFSET_LIMIT:
        return [
            _finding(
                Finding.DC_OFFSET,
                Severity.MINOR,
                f"DC offset of {offset:.4f}; the finishing engine removes this",
                metric="level.dc_offset",
                measured=offset,
                threshold=t.DC_OFFSET_LIMIT,
            )
        ]
    return []


# ── stereo ───────────────────────────────────────────────────────────


def check_stereo(measurement: CandidateMeasurement) -> list[QCFinding]:
    """Phase, width and balance.

    Only genuine anti-phase rejects. Narrow is a production choice that
    Phase 22 may widen; a low-end phase risk is one it repairs; an
    imbalance of under a dB is a mix. What cannot be repaired is material
    inverted against itself, which does not narrow in mono — it cancels.
    """
    stereo = measurement.analysis.stereo
    findings: list[QCFinding] = []
    if not stereo.is_stereo:
        return findings

    correlation = stereo.correlation
    if correlation is not None:
        if correlation < t.PHASE_UNSAFE_CORRELATION:
            findings.append(
                _finding(
                    Finding.PHASE_UNSAFE,
                    Severity.CRITICAL,
                    f"channel correlation is {correlation:.2f}: the channels are inverted "
                    "against each other and the mono sum will cancel",
                    metric="stereo.correlation",
                    measured=correlation,
                    threshold=t.PHASE_UNSAFE_CORRELATION,
                )
            )
        elif correlation < t.PHASE_RISK_CORRELATION:
            findings.append(
                _finding(
                    Finding.PHASE_UNSAFE,
                    Severity.MINOR,
                    f"channel correlation is {correlation:.2f}; wide, and the engine will "
                    "check mono compatibility",
                    metric="stereo.correlation",
                    measured=correlation,
                    threshold=t.PHASE_RISK_CORRELATION,
                )
            )

    if any(item.flag == RiskFlag.LOW_END_PHASE_RISK for item in measurement.risks):
        findings.append(
            _finding(
                Finding.LOW_END_PHASE_RISK,
                Severity.MINOR,
                "low-frequency content is out of phase; the finishing engine mono-sums it",
                metric="stereo.low_band_correlation",
                measured=stereo.low_band_correlation,
            )
        )

    if stereo.width is not None and stereo.width < t.NARROW_STEREO_WIDTH:
        findings.append(
            _finding(
                Finding.NARROW_STEREO,
                Severity.INFO,
                f"stereo width is {stereo.width:.3f}; narrow, which is a production choice "
                "and never a reason to discard a generation",
                metric="stereo.width",
                measured=stereo.width,
                threshold=t.NARROW_STEREO_WIDTH,
            )
        )

    balance = stereo.lr_balance_db
    if balance is not None and abs(balance) > t.CHANNEL_IMBALANCE_SEVERE_DB:
        findings.append(
            _finding(
                Finding.CHANNEL_IMBALANCE,
                Severity.CRITICAL,
                f"channels differ by {balance:.1f} dB; one of them is effectively absent",
                metric="stereo.lr_balance_db",
                measured=abs(balance),
                threshold=t.CHANNEL_IMBALANCE_SEVERE_DB,
            )
        )
    elif balance is not None and abs(balance) > t.CHANNEL_IMBALANCE_DB:
        findings.append(
            _finding(
                Finding.CHANNEL_IMBALANCE,
                Severity.MINOR,
                f"channels differ by {balance:.1f} dB",
                metric="stereo.lr_balance_db",
                measured=abs(balance),
                threshold=t.CHANNEL_IMBALANCE_DB,
            )
        )
    return findings


# ── spectral ─────────────────────────────────────────────────────────


def check_spectrum(measurement: CandidateMeasurement) -> list[QCFinding]:
    """One rejection, and it measures concentration rather than darkness.

    Darkness does not separate a failed generation from a bass-heavy
    record — the corpus settles that, with real deliverable songs at a
    352 Hz rolloff. What separates them is whether the energy is spread
    across the spectrum at all: a mix, however dark, occupies several
    bands, and a degenerate tone occupies one.

    Nothing here rewards brightness. There is no finding for a bright
    mix, because brightness is not evidence of quality — and none for a
    dark one, because darkness is not evidence of failure.
    """
    frequency = measurement.analysis.frequency
    findings: list[QCFinding] = []

    shares = [(band.name, band.share) for band in frequency.bands if band.share is not None]
    if shares:
        name, share = max(shares, key=lambda item: item[1])
        if share >= t.SPECTRAL_CONCENTRATION_SHARE:
            findings.append(
                _finding(
                    Finding.SPECTRAL_COLLAPSE,
                    Severity.CRITICAL,
                    f"{share:.0%} of the energy is in the {name} band alone — this is one "
                    "tone, not a mix",
                    metric="frequency.max_band_share",
                    measured=share,
                    threshold=t.SPECTRAL_CONCENTRATION_SHARE,
                    evidence={
                        "band": name,
                        "spectral_rolloff85_hz": round(frequency.spectral_rolloff85_hz.p50, 1),
                        "spectral_slope_db_per_octave": round(
                            frequency.spectral_slope_db_per_octave, 2
                        ),
                    },
                )
            )

    sibilance = measurement.analysis.sibilance
    if sibilance.harshness_peak_excess_db > t.HARSHNESS_PEAK_EXCESS_DB:
        findings.append(
            _finding(
                Finding.HIGH_HARSHNESS_PROXY,
                Severity.INFO,
                f"harshness proxy peaks {sibilance.harshness_peak_excess_db:.1f} dB above the "
                "band average; the finishing engine may tame it",
                metric="sibilance.harshness_peak_excess_db",
                measured=sibilance.harshness_peak_excess_db,
                threshold=t.HARSHNESS_PEAK_EXCESS_DB,
            )
        )
    if sibilance.sibilance_peak_excess_db > t.SIBILANCE_PEAK_EXCESS_DB:
        findings.append(
            _finding(
                Finding.HIGH_SIBILANCE_PROXY,
                Severity.INFO,
                f"sibilance proxy peaks {sibilance.sibilance_peak_excess_db:.1f} dB above the "
                "band average; the finishing engine may tame it",
                metric="sibilance.sibilance_peak_excess_db",
                measured=sibilance.sibilance_peak_excess_db,
                threshold=t.SIBILANCE_PEAK_EXCESS_DB,
            )
        )
    return findings


# ── control adherence ────────────────────────────────────────────────


def check_bpm(
    measurement: CandidateMeasurement, expectation: RequestExpectation
) -> list[QCFinding]:
    """Only when a BPM was asked for and the estimator was confident."""
    requested = expectation.bpm
    if requested is None:
        return []

    confidence = measurement.bpm_confidence
    if measurement.bpm is None or confidence is None or confidence < t.BPM_CONFIDENCE_FLOOR:
        return [
            _finding(
                Finding.CONTROL_NOT_MEASURABLE,
                Severity.INFO,
                f"a tempo of {requested} was requested; the estimate was too weak to compare "
                f"against (confidence {confidence if confidence is not None else 0:.2f}, "
                f"floor {t.BPM_CONFIDENCE_FLOOR})",
                metric="bpm_confidence",
                measured=confidence,
                threshold=t.BPM_CONFIDENCE_FLOOR,
                not_measurable=True,
            )
        ]

    relative = abs(measurement.bpm - requested) / requested
    evidence = {
        "requested_bpm": requested,
        "measured_bpm": measurement.bpm,
        "confidence": confidence,
    }
    if relative > t.BPM_HARD_TOLERANCE_RATIO:
        return [
            _finding(
                Finding.CONTROL_BPM_MISMATCH,
                Severity.MAJOR,
                f"{measurement.bpm:.1f} BPM against a request for {requested}",
                metric="bpm_relative_error",
                measured=relative,
                threshold=t.BPM_HARD_TOLERANCE_RATIO,
                evidence=evidence,
            )
        ]
    if relative > t.BPM_SOFT_TOLERANCE_RATIO:
        return [
            _finding(
                Finding.CONTROL_BPM_MISMATCH,
                Severity.MINOR,
                f"{measurement.bpm:.1f} BPM against a request for {requested}",
                metric="bpm_relative_error",
                measured=relative,
                threshold=t.BPM_SOFT_TOLERANCE_RATIO,
                evidence=evidence,
            )
        ]
    return []


def check_key(
    measurement: CandidateMeasurement, expectation: RequestExpectation
) -> list[QCFinding]:
    """Advisory in every profile.

    Key estimation from a full mix is genuinely hard, and a song in the
    relative minor of the requested key is not a failed request. So this
    never rises above MINOR and never justifies a retry — it is here so
    that a pattern of mismatches is visible in the trace, not so that
    one is acted on.
    """
    requested = expectation.key_scale
    if not requested:
        return []

    confidence = measurement.key_confidence
    if measurement.key is None or confidence is None or confidence < t.KEY_CONFIDENCE_FLOOR:
        return [
            _finding(
                Finding.CONTROL_NOT_MEASURABLE,
                Severity.INFO,
                f"a key of {requested} was requested; the estimate was too weak to compare against",
                metric="key_confidence",
                measured=confidence,
                threshold=t.KEY_CONFIDENCE_FLOOR,
                not_measurable=True,
            )
        ]

    detected = f"{measurement.key} {measurement.key_mode}".strip().lower()
    if requested.strip().lower().replace("_", " ") not in detected:
        return [
            _finding(
                Finding.CONTROL_KEY_MISMATCH,
                Severity.MINOR,
                f"detected {detected} against a request for {requested}; advisory only, "
                "because a relative key is not a failed request",
                metric="key_confidence",
                measured=confidence,
                evidence={"requested": requested, "detected": detected},
            )
        ]
    return []


def check_vocal(
    measurement: CandidateMeasurement, expectation: RequestExpectation
) -> list[QCFinding]:
    """Instrumental requested, vocal delivered — or honestly unknown.

    In this build the answer is always unknown, because no validated
    detector exists and `NullVocalDetector` says so rather than guessing.
    The mismatch branch is written and tested against a stub so that a
    real detector plugs in and starts working, rather than needing to be
    built at the same time.
    """
    if expectation.instrumental is None:
        return []

    vocal = measurement.vocal
    if not vocal.usable:
        return [
            _finding(
                Finding.CONTROL_VOCAL_UNKNOWN,
                Severity.INFO,
                vocal.reason,
                metric="vocal.confidence",
                measured=vocal.confidence,
                threshold=None,
                not_measurable=True,
                evidence=vocal.to_dict(),
            )
        ]

    wanted = (
        VocalPresence.INSTRUMENTAL.value if expectation.instrumental else VocalPresence.VOCAL.value
    )
    if vocal.presence != wanted:
        return [
            _finding(
                Finding.CONTROL_VOCAL_MISMATCH,
                Severity.MAJOR,
                f"{vocal.detector} reports {vocal.presence} at confidence "
                f"{vocal.confidence:.2f}; {wanted} was requested",
                metric="vocal.confidence",
                measured=vocal.confidence,
                evidence=vocal.to_dict(),
            )
        ]
    return []


# ── the whole battery ────────────────────────────────────────────────


def run_checks(
    measurement: CandidateMeasurement, expectation: RequestExpectation
) -> list[QCFinding]:
    """Every check, in a fixed order.

    Structure first: if the samples are not numbers, nothing else means
    anything, and the remaining checks are skipped rather than allowed to
    report nonsense with confidence.
    """
    structural = check_structure(measurement)
    if structural:
        return structural

    findings: list[QCFinding] = []
    findings.extend(check_silence(measurement))
    findings.extend(check_collapse(measurement, expectation))
    findings.extend(check_duration(measurement, expectation))
    findings.extend(check_clipping(measurement))
    findings.extend(check_dc(measurement))
    findings.extend(check_stereo(measurement))
    findings.extend(check_spectrum(measurement))
    findings.extend(check_bpm(measurement, expectation))
    findings.extend(check_key(measurement, expectation))
    findings.extend(check_vocal(measurement, expectation))

    if not any(item.blocking for item in findings):
        findings.append(
            _finding(
                Finding.NO_CRITICAL_FINDINGS,
                Severity.INFO,
                "nothing measured about this candidate prevents delivery",
            )
        )
    return findings


__all__ = [
    "RequestExpectation",
    "check_bpm",
    "check_clipping",
    "check_collapse",
    "check_dc",
    "check_duration",
    "check_key",
    "check_silence",
    "check_spectrum",
    "check_stereo",
    "check_structure",
    "check_vocal",
    "run_checks",
]
