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
    EXCESSIVE_BRIGHTNESS = "EXCESSIVE_BRIGHTNESS"
    HARSHNESS_RISK = "HARSHNESS_RISK"
    SIBILANCE_RISK = "SIBILANCE_RISK"
    TRANSIENT_FLATNESS = "TRANSIENT_FLATNESS"
    STEREO_TOO_NARROW = "STEREO_TOO_NARROW"
    STEREO_TOO_WIDE = "STEREO_TOO_WIDE"
    STEREO_IMBALANCE = "STEREO_IMBALANCE"
    LOW_END_PHASE_RISK = "LOW_END_PHASE_RISK"
    BROADBAND_PHASE_RISK = "BROADBAND_PHASE_RISK"
    CLIPPING_PRESENT = "CLIPPING_PRESENT"
    DC_OFFSET_PRESENT = "DC_OFFSET_PRESENT"
    #: Not a defect. A heavy low end that is clean and mono-compatible is
    #: a production choice, and this records that the engine recognised
    #: it as one — see ``low_end_is_intentional``.
    LOW_END_INTENTIONAL = "LOW_END_INTENTIONAL"


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

# Bright is not the absence of dark, so it gets its own two conditions
# rather than an inverted threshold. Both must hold, because either alone
# is ambiguous: a shallow slope can come from a thin low end, and a high
# air ratio can come from a mix that is simply quiet in the mids.
#
# 10-16 kHz relative to the midrange. Corpus median -25.7 dB, p90 -18.3,
# maximum -12.9. -16 sits above the ninetieth percentile.
EXCESSIVE_BRIGHTNESS_AIR_DB = -16.0
#: Corpus slope runs -11.2 to -3.5 with a median of -6.2. Pink noise
#: falls at -3 dB/octave and dense mixes sit steeper, so a master barely
#: steeper than pink noise is tilted up relative to its own material.
#: Together these two fire on 3 of the 57-master corpus.
BRIGHT_SLOPE_DB_PER_OCTAVE = -5.0

#: Bass this well correlated between channels survives a mono fold-down
#: intact, which is most of what separates deliberate weight from a
#: low-end problem.
INTENTIONAL_LOW_END_CORRELATION = 0.90

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
#: Broadband correlation. The corpus runs 0.37 to 0.94, so this fires on
#: nothing here by design: it is not describing LUBER output, it is a
#: tripwire for a master that would partially cancel in mono across the
#: whole spectrum, which no amount of EQ can repair and which must never
#: be widened further.
BROADBAND_PHASE_CORRELATION = 0.20
#: Widening is refused below this. The side channel is what a mono
#: fold-down cancels, so boosting it on already-decorrelated material
#: trades a narrow image for a hollow one.
#:
#: Structurally this should never bind: width is side/(mid+side), so
#: decorrelated channels measure as *wide*, and a track narrow enough to
#: widen has high correlation by construction. It is a floor under the
#: widening rule rather than a rule of its own — the guarantee that no
#: future change to how width is measured can turn widening loose on
#: material that would collapse in mono.
SAFE_TO_WIDEN_CORRELATION = 0.50
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


def low_end_is_intentional(analysis: AudioAnalysis) -> bool:
    """Is a heavy low end a production choice rather than a problem?

    Bass-forward is an entire aesthetic, and a share threshold alone
    cannot tell a deliberately weighted mix from a boomy one — both put
    most of their energy below 150 Hz. What separates them is whether
    that energy is *controlled*, and two measurements say so:

    *It does not smear upward.* Deliberate weight stays in the sub and
    bass bands. A low-end problem bleeds into 150-400 Hz, which is what
    the mud measurement already detects and what actually obscures
    everything else.

    *It survives mono.* Deliberate weight is close to mono to begin
    with; decorrelated bass is an accident of how the audio was made.

    On the 57-master corpus, 7 tracks exceed the excess threshold and 5
    of them pass both tests. Cutting those 5 would remove the point of
    the track. The remaining 2 are thick and uncontrolled, and they are
    the ones the low-shelf rule exists for.
    """
    if analysis.frequency.low_mid_ratio_db.p50 > LOW_MID_MUD_DB:
        return False
    correlation = analysis.stereo.low_band_correlation
    if correlation is not None and correlation < INTENTIONAL_LOW_END_CORRELATION:
        return False
    return True


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

    # The opposite condition, detected on its own terms rather than as
    # the absence of the one above. A bright master is tilted up across
    # the top *steadily*; a harsh or sibilant one spikes in a band. The
    # two need opposite responses — a shelf trim against the first, no
    # shelf at all against the second — so they must not be conflated.
    if (
        air_present
        and not math.isnan(slope)
        and air > EXCESSIVE_BRIGHTNESS_AIR_DB
        and slope > BRIGHT_SLOPE_DB_PER_OCTAVE
    ):
        findings.append(
            RiskFinding(
                flag=RiskFlag.EXCESSIVE_BRIGHTNESS,
                metric="frequency.air_ratio_db.p50",
                value=air,
                threshold=EXCESSIVE_BRIGHTNESS_AIR_DB,
                detail=(
                    f"10-16 kHz sits {air:.1f} dB from the body with the spectrum "
                    f"falling only {slope:.2f} dB/octave"
                ),
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
            # Recorded alongside the excess, not instead of it. The
            # measurement is real either way; what this adds is that the
            # weight looks deliberate, which is the decision engine's
            # cue to leave it alone.
            if low_end_is_intentional(analysis):
                findings.append(
                    RiskFinding(
                        flag=RiskFlag.LOW_END_INTENTIONAL,
                        metric="frequency.bands.sub+bass.share",
                        value=low_share,
                        threshold=LOW_END_EXCESS_SHARE,
                        detail=(
                            "the low end is clean and mono-compatible, so its weight "
                            "reads as a production choice rather than a defect"
                        ),
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
    if stereo.correlation is not None and stereo.correlation < BROADBAND_PHASE_CORRELATION:
        findings.append(
            RiskFinding(
                flag=RiskFlag.BROADBAND_PHASE_RISK,
                metric="stereo.correlation",
                value=stereo.correlation,
                threshold=BROADBAND_PHASE_CORRELATION,
                detail=(
                    f"channels correlate at {stereo.correlation:.2f} broadband, so much "
                    "of the material cancels in mono"
                ),
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
