"""Quality control: flags, a score, and a tier — in that order.

A flag is a *finding*, not a verdict. Most of them describe something
worth knowing that does not disqualify anything: plenty of excellent
recordings are mono, plenty are loud, and a track that clips for four
samples is not ruined. Treating every flag as fatal is how a pipeline
quietly discards most of a real library, so only a small, named,
configurable set is disqualifying and everything else costs score.

The tier is what downstream code acts on, and the score is how the tier
was reached. Both are derived from the flags rather than measured
independently, so a track's tier can always be explained by pointing at
the flags that produced it — and every threshold that turned a
measurement into a flag lives in :mod:`config`, where it can be changed
per dataset without touching this logic.

Nothing here is silent. A rejected track keeps its flags, its score and
its reasons, and appears in the rejections file rather than vanishing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from luber_dataset.factory.audio_analysis import TechnicalAnalysis
from luber_dataset.factory.config import QualityThresholds
from luber_dataset.factory.decoder import DecodeResult, DecodeStatus

# ── flags ────────────────────────────────────────────────────────────
TOO_SHORT = "TOO_SHORT"
TOO_LONG = "TOO_LONG"
DECODE_ERROR = "DECODE_ERROR"
CLIPPING = "CLIPPING"
EXTREME_LOUDNESS = "EXTREME_LOUDNESS"
EXTREME_SILENCE = "EXTREME_SILENCE"
LOW_SAMPLE_RATE = "LOW_SAMPLE_RATE"
MONO = "MONO"
PHASE_RISK = "PHASE_RISK"
DC_OFFSET = "DC_OFFSET"
LOW_DYNAMIC_RANGE = "LOW_DYNAMIC_RANGE"
SUSPICIOUS_BANDWIDTH = "SUSPICIOUS_BANDWIDTH"
CORRUPT = "CORRUPT"
NEAR_DUPLICATE = "NEAR_DUPLICATE"
UNMEASURED = "UNMEASURED"

#: How much each flag costs. Severity is about how much the flag damages
#: the track's usefulness *as training material*, which is not the same
#: as how unusual it is.
SEVERE_FLAGS: frozenset[str] = frozenset({DECODE_ERROR, CORRUPT, CLIPPING, EXTREME_SILENCE})
MODERATE_FLAGS: frozenset[str] = frozenset(
    {TOO_SHORT, LOW_SAMPLE_RATE, PHASE_RISK, LOW_DYNAMIC_RANGE, SUSPICIOUS_BANDWIDTH, UNMEASURED}
)
#: Everything else is minor: real, worth recording, not damaging.
MINOR_FLAGS: frozenset[str] = frozenset(
    {TOO_LONG, MONO, DC_OFFSET, EXTREME_LOUDNESS, NEAR_DUPLICATE}
)


class QualityTier(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    REJECT = "REJECT"


#: Ordering for "at least tier B" comparisons.
TIER_ORDER: dict[str, int] = {
    QualityTier.A.value: 3,
    QualityTier.B.value: 2,
    QualityTier.C.value: 1,
    QualityTier.REJECT.value: 0,
}


@dataclass
class QualityAssessment:
    quality_flags: list[str] = field(default_factory=list)
    quality_score: float = 0.0
    quality_tier: str = QualityTier.REJECT.value
    #: Why the tier is what it is, in words.
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality_flags": sorted(self.quality_flags),
            "quality_score": round(self.quality_score, 4),
            "quality_tier": self.quality_tier,
            "reasons": list(self.reasons),
        }


def evaluate(
    decode: DecodeResult,
    analysis: TechnicalAnalysis,
    thresholds: QualityThresholds,
    *,
    near_duplicate: bool = False,
) -> QualityAssessment:
    """Flags, score and tier for one track."""
    flags: list[str] = []
    reasons: list[str] = []

    if decode.status is DecodeStatus.INVALID:
        flags.append(CORRUPT)
        reasons.append(f"decode failed: {decode.decode_error or 'unknown error'}")
    elif decode.status is DecodeStatus.PARTIAL:
        flags.append(DECODE_ERROR)
        reasons.append(f"decoded with errors: {decode.decode_error or 'unknown error'}")
    elif decode.status is DecodeStatus.UNSUPPORTED:
        flags.append(CORRUPT)
        reasons.append("no audio stream")

    if near_duplicate:
        flags.append(NEAR_DUPLICATE)
        reasons.append("fingerprint is close to another track; a human decides")

    if not analysis.measured:
        if CORRUPT not in flags:
            flags.append(UNMEASURED)
            reasons.append(analysis.analysis_error or "technical analysis did not run")
        return _score(flags, reasons, thresholds)

    duration = analysis.duration_seconds
    if duration is not None:
        if duration < thresholds.min_duration_seconds:
            flags.append(TOO_SHORT)
            reasons.append(
                f"{duration:.1f}s is below the {thresholds.min_duration_seconds:.0f}s minimum"
            )
        elif duration > thresholds.max_duration_seconds:
            flags.append(TOO_LONG)
            reasons.append(
                f"{duration:.0f}s exceeds the {thresholds.max_duration_seconds:.0f}s maximum"
            )

    if analysis.sample_rate is not None and analysis.sample_rate < thresholds.min_sample_rate:
        flags.append(LOW_SAMPLE_RATE)
        reasons.append(f"{analysis.sample_rate} Hz is below CD rate")

    if analysis.channels is not None and analysis.channels < 2:
        flags.append(MONO)
        reasons.append("mono source: no stereo image to learn from")

    ratio = analysis.clipping_sample_ratio
    if ratio is not None and ratio > thresholds.max_clipping_sample_ratio:
        flags.append(CLIPPING)
        reasons.append(f"{ratio:.4%} of samples sit at full scale")

    crest = analysis.crest_factor_db
    if crest is not None and crest < thresholds.min_crest_factor_db:
        flags.append(LOW_DYNAMIC_RANGE)
        reasons.append(f"crest factor {crest:.1f} dB suggests brickwalled mastering")

    dc = analysis.dc_offset
    if dc is not None and abs(dc) > thresholds.max_dc_offset:
        flags.append(DC_OFFSET)
        reasons.append(f"DC offset {dc:.4f}")

    lufs = analysis.integrated_lufs
    if lufs is not None and not (
        thresholds.min_integrated_lufs <= lufs <= thresholds.max_integrated_lufs
    ):
        flags.append(EXTREME_LOUDNESS)
        reasons.append(f"integrated loudness {lufs:.1f} LUFS is outside the expected range")

    silence = analysis.silence_ratio
    if silence is not None and silence > thresholds.max_silence_ratio:
        flags.append(EXTREME_SILENCE)
        reasons.append(f"{silence:.0%} of the file is effectively silent")

    correlation = analysis.phase_correlation
    if correlation is not None and correlation < thresholds.min_phase_correlation:
        flags.append(PHASE_RISK)
        reasons.append(f"channels correlate at {correlation:.2f}; much cancels in mono")

    # A high sample rate with nothing in the top octave means the file
    # was transcoded up from something lossy. The rate is claiming
    # information the audio does not contain.
    cutoff = analysis.high_frequency_cutoff_hz
    if (
        cutoff is not None
        and analysis.sample_rate is not None
        and analysis.sample_rate >= 44_100
        and cutoff < thresholds.suspicious_bandwidth_hz
    ):
        flags.append(SUSPICIOUS_BANDWIDTH)
        reasons.append(
            f"content stops at {cutoff:.0f} Hz despite a {analysis.sample_rate} Hz rate; "
            "likely upsampled from a lossy source"
        )

    return _score(flags, reasons, thresholds)


def _score(
    flags: list[str], reasons: list[str], thresholds: QualityThresholds
) -> QualityAssessment:
    """Turn flags into a score and a tier.

    Disqualification is checked before scoring, so a corrupt file lands
    in REJECT regardless of how well it does on everything else.
    """
    unique = sorted(set(flags))
    disqualifying = sorted(set(unique) & set(thresholds.disqualifying_flags))

    score = 1.0
    for flag in unique:
        if flag in SEVERE_FLAGS:
            score -= thresholds.severe_flag_penalty
        elif flag in MODERATE_FLAGS:
            score -= thresholds.moderate_flag_penalty
        else:
            score -= thresholds.minor_flag_penalty
    score = max(0.0, min(1.0, score))

    if disqualifying:
        tier = QualityTier.REJECT.value
        reasons = [*reasons, f"disqualifying flag(s): {', '.join(disqualifying)}"]
    elif score >= thresholds.tier_a_min_score:
        tier = QualityTier.A.value
    elif score >= thresholds.tier_b_min_score:
        tier = QualityTier.B.value
    elif score >= thresholds.tier_c_min_score:
        tier = QualityTier.C.value
    else:
        tier = QualityTier.REJECT.value
        reasons = [*reasons, f"score {score:.2f} is below the tier C floor"]

    return QualityAssessment(
        quality_flags=unique, quality_score=score, quality_tier=tier, reasons=reasons
    )


def meets_tier(tier: str, minimum: str) -> bool:
    return TIER_ORDER.get(tier, 0) >= TIER_ORDER.get(minimum, 0)
