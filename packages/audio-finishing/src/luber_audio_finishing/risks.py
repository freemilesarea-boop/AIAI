"""Turning measurements into named, evidenced risks.

A threshold is a claim, so each one here is stated with the reasoning
behind it and with what the Phase 14 baseline corpus of 40 LUBER masters
actually did against it. Two rules governed every choice:

*Thresholds are absolute, not corpus-relative.* Setting them at corpus
percentiles would define "correct" as "average for this model", which
guarantees a fixed fraction of tracks is always flagged and makes the
engine chase its own output. They are anchored instead on within-track
physical relationships that hold regardless of what produced the audio.

*Silence is the expected outcome.* 14 of the 40 baseline tracks trip no
flag at all. A flag that fires on most of a corpus is not detecting a
defect; it is describing the model, and correcting it would be an
opinion about how music should sound.

No commercial recording informed any number here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from luber_audio_finishing.analysis import AudioAnalysis
from luber_audio_finishing.bands import AIR, BASS, SUB


class RiskFlag(StrEnum):
    """Named conditions the decision engine may respond to."""

    LOW_END_EXCESS = "LOW_END_EXCESS"
    LOW_MID_MUD = "LOW_MID_MUD"
    PRESENCE_DEFICIT = "PRESENCE_DEFICIT"
    HIGH_FREQUENCY_DEFICIT = "HIGH_FREQUENCY_DEFICIT"
    AIR_DEFICIT = "AIR_DEFICIT"
    HARSHNESS_RISK = "HARSHNESS_RISK"
    SIBILANCE_RISK = "SIBILANCE_RISK"
    TRANSIENT_FLATNESS = "TRANSIENT_FLATNESS"
    STEREO_TOO_NARROW = "STEREO_TOO_NARROW"
    STEREO_TOO_WIDE = "STEREO_TOO_WIDE"
    STEREO_IMBALANCE = "STEREO_IMBALANCE"
    LOW_END_PHASE_RISK = "LOW_END_PHASE_RISK"
    CLIPPING_PRESENT = "CLIPPING_PRESENT"
    DC_OFFSET_PRESENT = "DC_OFFSET_PRESENT"


# ── Thresholds ───────────────────────────────────────────────────────
#
# Spectral slope: pink noise falls at exactly -3 dB/octave, and dense
# mixes sit somewhat steeper. -6.5 is well below that and below the
# corpus median of -6.2, so only genuinely dark masters qualify.
DARK_SLOPE_DB_PER_OCTAVE = -6.5
#: Every ratio below is measured against the 400 Hz-2 kHz midrange.
#:
#: 10-16 kHz relative to the midrange. Corpus median -26.1 dB, range
#: -39.0 to -13.4. -27 marks the darker half; -30 the darkest quarter.
HIGH_FREQUENCY_DEFICIT_AIR_DB = -27.0
AIR_DEFICIT_DB = -30.0
#: 2-5 kHz relative to the midrange. Corpus median -15.8 dB.
PRESENCE_DEFICIT_DB = -21.0
#: 150-400 Hz above the midrange by this much reads as thick, not warm.
#: Corpus median +2.8 dB, p90 +7.8; this catches the thickest quarter.
LOW_MID_MUD_DB = 5.5
#: Share of banded energy below 150 Hz. Music legitimately concentrates
#: energy there — the corpus median is 0.47 — so only the top of the
#: range qualifies as excess.
LOW_END_EXCESS_SHARE = 0.70

# Harshness and sibilance are judged by how far the band *spikes* above
# its own median, not by absolute level: 6-9 kHz holds cymbals, string
# noise and synth texture as well as sibilants, and a track where that
# band sits high but steady is bright, not harsh. The absolute floor
# stops a quiet-but-spiky band from qualifying.
HARSHNESS_PEAK_EXCESS_DB = 14.0
HARSHNESS_FLOOR_DB = -11.0
SIBILANCE_PEAK_EXCESS_DB = 17.0
SIBILANCE_FLOOR_DB = -17.0

#: Side/(mid+side) above 120 Hz. The corpus runs 0.073 to 0.374 with a
#: median of 0.184, and 0.11 cuts the narrow tail — four tracks, all of
#: which also sit low on channel correlation.
STEREO_NARROW_WIDTH = 0.11
#: A ceiling against over-widening, including any this engine might
#: itself cause. Deliberately above the corpus maximum of 0.374, which
#: belongs to a legitimately wide mix that correlates at 0.82 broadband
#: and 0.999 in the bass; narrowing it would be a matter of taste. Past
#: 0.45 the side channel carries more energy than the mid, which is a
#: phase problem rather than a wide image.
STEREO_WIDE_WIDTH = 0.45
STEREO_IMBALANCE_DB = 0.8
#: Correlation below 120 Hz. Below this, bass partially cancels in mono.
LOW_END_PHASE_CORRELATION = 0.90
#: Crest factor over 50 ms windows. The corpus sits at 7.0-9.3 dB, so
#: this fires on nothing here; it exists to detect a future regression.
TRANSIENT_FLAT_CREST_DB = 6.5
DC_OFFSET_LIMIT = 0.002


@dataclass(frozen=True)
class RiskFinding:
    """One flag, with the number that raised it.

    ``value`` and ``threshold`` are kept so a plan can be re-read later
    and checked, rather than believed.
    """

    flag: RiskFlag
    metric: str
    value: float
    threshold: float
    detail: str

    @property
    def margin(self) -> float:
        """How far past the threshold the measurement sits."""
        return self.value - self.threshold


def _band_share(analysis: AudioAnalysis, name: str) -> float | None:
    band = analysis.frequency.band(name)
    return None if band is None else band.share


def evaluate_risks(analysis: AudioAnalysis) -> tuple[RiskFinding, ...]:
    """Every risk the measurements support, and no others."""
    findings: list[RiskFinding] = []
    frequency = analysis.frequency
    sibilance = analysis.sibilance
    stereo = analysis.stereo

    air_band = frequency.band(AIR)
    # A sample rate that cannot represent 10-16 kHz has no air deficit to
    # find, and boosting a band that physically does not exist would add
    # nothing but noise.
    air_present = air_band is not None and air_band.energy_db is not None
    air = frequency.air_ratio_db.p50
    slope = frequency.spectral_slope_db_per_octave

    # Two conditions, because either alone is ambiguous: a dark slope can
    # come from a heavy low end, and a low air ratio can come from a mix
    # that is simply loud in the mids.
    if (
        air_present
        and not math.isnan(slope)
        and air < HIGH_FREQUENCY_DEFICIT_AIR_DB
        and slope < DARK_SLOPE_DB_PER_OCTAVE
    ):
        findings.append(
            RiskFinding(
                flag=RiskFlag.HIGH_FREQUENCY_DEFICIT,
                metric="frequency.spectral_slope_db_per_octave",
                value=slope,
                threshold=DARK_SLOPE_DB_PER_OCTAVE,
                detail=(
                    f"spectrum falls {slope:.2f} dB/octave with the air band "
                    f"{air:.1f} dB below the body of the mix"
                ),
            )
        )
    if air_present and air < AIR_DEFICIT_DB:
        findings.append(
            RiskFinding(
                flag=RiskFlag.AIR_DEFICIT,
                metric="frequency.air_ratio_db.p50",
                value=air,
                threshold=AIR_DEFICIT_DB,
                detail=f"10-16 kHz sits {air:.1f} dB below the 300 Hz-3 kHz body",
            )
        )

    presence = frequency.presence_ratio_db.p50
    if presence < PRESENCE_DEFICIT_DB:
        findings.append(
            RiskFinding(
                flag=RiskFlag.PRESENCE_DEFICIT,
                metric="frequency.presence_ratio_db.p50",
                value=presence,
                threshold=PRESENCE_DEFICIT_DB,
                detail=f"2-5 kHz sits {presence:.1f} dB below the body of the mix",
            )
        )

    low_mid = frequency.low_mid_ratio_db.p50
    if low_mid > LOW_MID_MUD_DB:
        findings.append(
            RiskFinding(
                flag=RiskFlag.LOW_MID_MUD,
                metric="frequency.low_mid_ratio_db.p50",
                value=low_mid,
                threshold=LOW_MID_MUD_DB,
                detail=f"150-400 Hz sits {low_mid:.1f} dB above the body of the mix",
            )
        )

    sub_share = _band_share(analysis, SUB)
    bass_share = _band_share(analysis, BASS)
    if sub_share is not None and bass_share is not None:
        low_share = sub_share + bass_share
        if low_share > LOW_END_EXCESS_SHARE:
            findings.append(
                RiskFinding(
                    flag=RiskFlag.LOW_END_EXCESS,
                    metric="frequency.bands.sub+bass.share",
                    value=low_share,
                    threshold=LOW_END_EXCESS_SHARE,
                    detail=f"{low_share:.0%} of banded energy sits below 150 Hz",
                )
            )

    if (
        sibilance.harshness_peak_excess_db > HARSHNESS_PEAK_EXCESS_DB
        and sibilance.harshness_ratio_db.p90 > HARSHNESS_FLOOR_DB
    ):
        findings.append(
            RiskFinding(
                flag=RiskFlag.HARSHNESS_RISK,
                metric="sibilance.harshness_peak_excess_db",
                value=sibilance.harshness_peak_excess_db,
                threshold=HARSHNESS_PEAK_EXCESS_DB,
                detail=(
                    f"2.5-5 kHz spikes {sibilance.harshness_peak_excess_db:.1f} dB above its "
                    f"own median, peaking at {sibilance.harshness_ratio_db.p90:.1f} dB"
                ),
            )
        )
    if (
        sibilance.sibilance_peak_excess_db > SIBILANCE_PEAK_EXCESS_DB
        and sibilance.sibilance_ratio_db.p90 > SIBILANCE_FLOOR_DB
    ):
        findings.append(
            RiskFinding(
                flag=RiskFlag.SIBILANCE_RISK,
                metric="sibilance.sibilance_peak_excess_db",
                value=sibilance.sibilance_peak_excess_db,
                threshold=SIBILANCE_PEAK_EXCESS_DB,
                detail=(
                    f"6-9 kHz spikes {sibilance.sibilance_peak_excess_db:.1f} dB above its "
                    f"own median, peaking at {sibilance.sibilance_ratio_db.p90:.1f} dB"
                ),
            )
        )

    crest = analysis.level.short_window_crest_db.p50
    if not math.isnan(crest) and crest < TRANSIENT_FLAT_CREST_DB:
        findings.append(
            RiskFinding(
                flag=RiskFlag.TRANSIENT_FLATNESS,
                metric="level.short_window_crest_db.p50",
                value=crest,
                threshold=TRANSIENT_FLAT_CREST_DB,
                detail=f"50 ms crest factor of {crest:.1f} dB indicates a flattened signal",
            )
        )

    if stereo.is_stereo:
        findings.extend(_stereo_risks(analysis))

    if analysis.level.clipped_samples > 0:
        findings.append(
            RiskFinding(
                flag=RiskFlag.CLIPPING_PRESENT,
                metric="level.clipped_samples",
                value=float(analysis.level.clipped_samples),
                threshold=0.0,
                detail=f"{analysis.level.clipped_samples} samples at or beyond full scale",
            )
        )
    if analysis.level.dc_offset > DC_OFFSET_LIMIT:
        findings.append(
            RiskFinding(
                flag=RiskFlag.DC_OFFSET_PRESENT,
                metric="level.dc_offset",
                value=analysis.level.dc_offset,
                threshold=DC_OFFSET_LIMIT,
                detail=f"DC offset of {analysis.level.dc_offset:.4f}",
            )
        )
    return tuple(findings)


def _stereo_risks(analysis: AudioAnalysis) -> list[RiskFinding]:
    stereo = analysis.stereo
    findings: list[RiskFinding] = []
    width = stereo.width
    balance = stereo.lr_balance_db
    low_correlation = stereo.low_band_correlation

    if width is not None and width < STEREO_NARROW_WIDTH:
        findings.append(
            RiskFinding(
                flag=RiskFlag.STEREO_TOO_NARROW,
                metric="stereo.width",
                value=width,
                threshold=STEREO_NARROW_WIDTH,
                detail=f"side energy is {width:.0%} of the image",
            )
        )
    if width is not None and width > STEREO_WIDE_WIDTH:
        findings.append(
            RiskFinding(
                flag=RiskFlag.STEREO_TOO_WIDE,
                metric="stereo.width",
                value=width,
                threshold=STEREO_WIDE_WIDTH,
                detail=f"side energy is {width:.0%} of the image",
            )
        )
    if balance is not None and abs(balance) > STEREO_IMBALANCE_DB:
        findings.append(
            RiskFinding(
                flag=RiskFlag.STEREO_IMBALANCE,
                metric="stereo.lr_balance_db",
                value=balance,
                threshold=STEREO_IMBALANCE_DB,
                detail=f"left is {balance:+.2f} dB relative to right",
            )
        )
    if low_correlation is not None and low_correlation < LOW_END_PHASE_CORRELATION:
        findings.append(
            RiskFinding(
                flag=RiskFlag.LOW_END_PHASE_RISK,
                metric="stereo.low_band_correlation",
                value=low_correlation,
                threshold=LOW_END_PHASE_CORRELATION,
                detail=(
                    f"channels correlate at {low_correlation:.2f} below 120 Hz, "
                    "so bass partially cancels in mono"
                ),
            )
        )
    return findings
