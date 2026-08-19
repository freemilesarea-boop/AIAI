"""Turning measured risks into a bounded, auditable finishing plan.

The plan is data, not sound. It is produced deterministically from an
analysis, it can be read and argued with before anything is rendered, and
every entry carries the measurement that caused it, the amount requested
by the rule, the ceiling that applied, and whether that ceiling bit. A
plan that cannot be checked is indistinguishable from a preset.

Three properties matter more than the individual numbers:

*Corrections are proportional and partial.* Every rule moves the signal a
fraction of the way toward its threshold, never all the way. A generation
that is 8 dB dark does not become a generation that is 0 dB dark; it
becomes one that is 5 dB dark. Full correction would impose one spectral
opinion on every song.

*Contradictions resolve toward doing less.* A track can be dark overall
and spiky at 7 kHz at once — eight of the forty baseline tracks are. The
brightness rule and the harshness rule then disagree, and the harshness
side wins by lowering the ceiling, because adding air to a sibilant
master trades a fixable dullness for an unfixable harshness.

*Nothing gets invented to fill a category.* Transient and spatial
processing are deferred with their reasons recorded, rather than shipped
small to look complete.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from luber_audio_finishing.analysis import AudioAnalysis
from luber_audio_finishing.risks import (
    AIR_DEFICIT_DB,
    EXCESSIVE_BRIGHTNESS_AIR_DB,
    HIGH_FREQUENCY_DEFICIT_AIR_DB,
    LOW_END_EXCESS_SHARE,
    LOW_MID_MUD_DB,
    PRESENCE_DEFICIT_DB,
    SAFE_TO_WIDEN_CORRELATION,
    STEREO_IMBALANCE_DB,
    STEREO_NARROW_WIDTH,
    STEREO_WIDE_WIDTH,
    RiskFinding,
    RiskFlag,
    evaluate_risks,
)
from luber_audio_finishing.version import FINISHING_VERSION

# ── Safety envelope ──────────────────────────────────────────────────
#
# Ceilings, not targets. A rule reaching its ceiling records that fact so
# a clamped decision is visible in the plan rather than looking like a
# considered choice.
MAX_HIGH_SHELF_DB = 3.0
#: Trimming brightness is held to a tighter ceiling than adding it. A
#: dark master is missing information and a lift cannot remove any; a
#: bright one has the information, and cutting too far throws away detail
#: that no later stage can put back.
MAX_HIGH_SHELF_CUT_DB = 2.0
MAX_PRESENCE_LIFT_DB = 2.0
MAX_LOW_MID_CUT_DB = 2.5
MAX_LOW_SHELF_CUT_DB = 1.5
MAX_WIDTH_ADJUST_DB = 1.5
MAX_BALANCE_CORRECTION_DB = 1.5

#: Harshness and sibilance override the brightness rules by lowering its
#: ceiling rather than by cancelling them, so a dark *and* sibilant track
#: still gets the small lift its darkness justifies.
SIBILANCE_SHELF_CEILING_DB = 1.0
HARSHNESS_SHELF_CEILING_DB = 1.5

#: Fraction of the measured deficit or excess any rule will correct.
CORRECTION_FRACTION = 0.4
MUD_CORRECTION_FRACTION = 0.5

#: Below this, a correction is inaudible and only costs a filter stage.
MIN_ACTION_DB = 0.3

#: Shelf corner frequencies. A track that is dark everywhere needs the
#: lift to start lower than one whose mids are fine and whose top octave
#: alone is missing.
BROAD_SHELF_HZ = 7_000.0
AIR_SHELF_HZ = 10_000.0
SHELF_Q = 0.7
PRESENCE_HZ = 3_200.0
PRESENCE_Q = 0.8
#: Wide enough (about 1.6 octaves) that the cut reads as less thickness
#: rather than as a hole in the mix.
LOW_MID_Q = 0.9
LOW_SHELF_HZ = 90.0
#: Bass below here is summed to mono. Standard practice, and low enough
#: to leave the stereo image of everything musical untouched.
LOW_MONO_CROSSOVER_HZ = 120.0

#: Width the narrow rule aims at — inside the healthy range and still
#: well below the corpus median of 0.184, so the correction is a nudge
#: toward normal rather than a push toward wide.
TARGET_NARROW_WIDTH = 0.14
TARGET_WIDE_WIDTH = 0.32

#: Output ceiling. Generated masters arrive at exactly -1.0 dBFS, so any
#: boost needs somewhere to go; this is where it goes.
OUTPUT_CEILING_DBFS = -1.0


class ActionKind(StrEnum):
    BALANCE_CORRECTION = "BALANCE_CORRECTION"
    LOW_SHELF_CUT = "LOW_SHELF_CUT"
    LOW_MID_CUT = "LOW_MID_CUT"
    PRESENCE_LIFT = "PRESENCE_LIFT"
    HIGH_SHELF_LIFT = "HIGH_SHELF_LIFT"
    HIGH_SHELF_CUT = "HIGH_SHELF_CUT"
    STEREO_WIDTH = "STEREO_WIDTH"
    LOW_FREQUENCY_MONO = "LOW_FREQUENCY_MONO"


@dataclass(frozen=True)
class FinishingAction:
    """One correction, with everything needed to justify or reverse it."""

    kind: ActionKind
    trigger: RiskFlag | None
    reason: str
    metric: str
    measured_value: float
    threshold: float
    gain_db: float | None
    #: What the rule asked for before the ceiling applied.
    requested_gain_db: float | None
    ceiling_db: float | None
    clamped: bool
    frequency_hz: float | None = None
    q: float | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SuppressedAction:
    """A correction the rules called for, and the engine declined.

    Distinct from ``DeferredDecision``, which is a standing choice about
    a whole area of processing. This is per-track and evidenced: the
    measurement said to act, something else said not to, and the second
    won. Recording it is what keeps the plan honest — a rule that
    silently returns nothing is indistinguishable from a rule that never
    fired, and the two mean opposite things about the audio.
    """

    kind: ActionKind
    trigger: RiskFlag | None
    reason: str
    metric: str
    measured_value: float


@dataclass(frozen=True)
class DeferredDecision:
    """Something deliberately not done, and why.

    Recorded in the plan because "we chose not to" and "we forgot" look
    identical in a system that only records what it did.
    """

    area: str
    reason: str


@dataclass(frozen=True)
class FinishingPlan:
    finishing_version: str
    actions: tuple[FinishingAction, ...]
    risks: tuple[RiskFinding, ...]
    deferred: tuple[DeferredDecision, ...]
    #: Corrections the rules called for and the engine declined, each
    #: with the evidence that overrode it.
    suppressed: tuple[SuppressedAction, ...] = ()
    #: Peak ceiling the processor must respect, in dBFS.
    output_ceiling_dbfs: float = OUTPUT_CEILING_DBFS
    #: Match the finished loudness to the source's, so an A/B comparison
    #: is about tone rather than about which file is louder.
    match_source_loudness: bool = True

    @property
    def is_no_action(self) -> bool:
        return not self.actions

    @property
    def flags(self) -> tuple[RiskFlag, ...]:
        return tuple(finding.flag for finding in self.risks)

    def action(self, kind: ActionKind) -> FinishingAction | None:
        for action in self.actions:
            if action.kind == kind:
                return action
        return None

    def has_flag(self, flag: RiskFlag) -> bool:
        return flag in self.flags


DEFERRED_TRANSIENT = DeferredDecision(
    area="transient shaping",
    reason=(
        "no measured need and no verified bounded implementation. The 40-track "
        "baseline sits at 7.0-9.3 dB of 50 ms crest factor against a flatness "
        "threshold of 6.5 dB, so nothing in the corpus asks for it, and a "
        "transient shaper applied without a need would attack sustained pads "
        "and reverb tails as readily as drums."
    ),
)

DEFERRED_SPATIAL = DeferredDecision(
    area="spatial and reverb",
    reason=(
        "the spatial measurements are proxies with known confounds — a held "
        "pad and a reverb tail decay alike, a wide synth and a room read alike "
        "— so no threshold among them can distinguish a dry mix from a "
        "deliberately dry one. Adding ambience to every master would be an "
        "aesthetic decision presented as a correction."
    ),
)

DEFERRED_DYNAMIC_EQ = DeferredDecision(
    area="dynamic harshness control",
    reason=(
        "measured, not assumed: ffmpeg's adynamicequalizer was run over a "
        "sibilant baseline track at detection thresholds from 0.0005 to 0.1 "
        "and in adaptive mode. Where it acted at all it moved the 6-9 kHz "
        "median and the 90th percentile by the same amount, leaving peak "
        "excess at 21.05-21.23 dB against an untreated 21.21 dB. That is a "
        "static cut wearing a dynamic label, and a static scoop is exactly "
        "what a de-esser must not be. Harshness and sibilance instead lower "
        "the high-shelf ceiling, which needs no new filter to be safe."
    ),
)


def _clamped_gain(requested: float, ceiling: float) -> tuple[float, bool]:
    limited = min(abs(requested), abs(ceiling))
    return (math.copysign(limited, requested), abs(requested) > abs(ceiling))


def _width_gain_db(current: float, target: float) -> float:
    """Side-channel gain that moves side/(mid+side) from *current* to *target*.

    Width is an energy ratio, so the algebra inverts exactly: with side
    energy scaled by g, w' = gS / (M + gS), which solves for g without
    approximation.
    """
    if not 0.0 < current < 1.0 or not 0.0 < target < 1.0:
        return 0.0
    ratio = (target / (1.0 - target)) * ((1.0 - current) / current)
    return 10.0 * math.log10(ratio)


class FinishingDecisionEngine:
    """Analysis in, plan out. No audio, no files, no side effects."""

    def __init__(self, *, finishing_version: str = FINISHING_VERSION) -> None:
        self._version = finishing_version

    def plan(self, analysis: AudioAnalysis) -> FinishingPlan:
        risks = evaluate_risks(analysis)
        present = {finding.flag: finding for finding in risks}
        actions: list[FinishingAction] = []
        suppressed: list[SuppressedAction] = []

        # Order matters and is the order of the rendered chain: balance,
        # then subtractive EQ, then additive EQ, then the stereo stage.
        # Cutting before boosting means the boost is applied to audio
        # that has already lost its excess rather than compounding it.
        actions.extend(self._balance(analysis, present))
        actions.extend(self._low_shelf_cut(analysis, present, suppressed))
        actions.extend(self._low_mid_cut(analysis, present))
        actions.extend(self._presence_lift(analysis, present, suppressed))
        actions.extend(self._high_shelf(analysis, present))
        actions.extend(self._stereo(analysis, present, suppressed))

        return FinishingPlan(
            finishing_version=self._version,
            actions=tuple(actions),
            risks=risks,
            deferred=(DEFERRED_DYNAMIC_EQ, DEFERRED_TRANSIENT, DEFERRED_SPATIAL),
            suppressed=tuple(suppressed),
        )

    # ── individual rules ─────────────────────────────────────────────

    def _balance(
        self, analysis: AudioAnalysis, present: dict[RiskFlag, RiskFinding]
    ) -> list[FinishingAction]:
        finding = present.get(RiskFlag.STEREO_IMBALANCE)
        balance = analysis.stereo.lr_balance_db
        if finding is None or balance is None:
            return []
        # Fully corrected, not partially: a centred image is not a matter
        # of taste, and the residual would serve no one.
        gain, clamped = _clamped_gain(-balance, MAX_BALANCE_CORRECTION_DB)
        if abs(gain) < MIN_ACTION_DB:
            return []
        return [
            FinishingAction(
                kind=ActionKind.BALANCE_CORRECTION,
                trigger=RiskFlag.STEREO_IMBALANCE,
                reason="channels are not level; centre the image",
                metric=finding.metric,
                measured_value=balance,
                threshold=STEREO_IMBALANCE_DB,
                gain_db=gain,
                requested_gain_db=-balance,
                ceiling_db=MAX_BALANCE_CORRECTION_DB,
                clamped=clamped,
                notes=("applied as half the correction to each channel",),
            )
        ]

    def _low_shelf_cut(
        self,
        analysis: AudioAnalysis,
        present: dict[RiskFlag, RiskFinding],
        suppressed: list[SuppressedAction],
    ) -> list[FinishingAction]:
        finding = present.get(RiskFlag.LOW_END_EXCESS)
        if finding is None:
            return []
        # A heavy low end that is clean and mono-compatible is the point
        # of the track, not a fault in it. Five of the seven corpus
        # masters over the excess threshold are in that position.
        intentional = present.get(RiskFlag.LOW_END_INTENTIONAL)
        if intentional is not None:
            suppressed.append(
                SuppressedAction(
                    kind=ActionKind.LOW_SHELF_CUT,
                    trigger=RiskFlag.LOW_END_EXCESS,
                    reason=(
                        "the low end is clean and mono-compatible, so its weight is "
                        "treated as deliberate rather than corrected away"
                    ),
                    metric=finding.metric,
                    measured_value=finding.value,
                )
            )
            return []
        # Share is a fraction, so it is converted to dB by how far past
        # the threshold it sits: 10 dB of correction per unit of share.
        requested = -(finding.value - LOW_END_EXCESS_SHARE) * 10.0
        gain, clamped = _clamped_gain(requested, MAX_LOW_SHELF_CUT_DB)
        if abs(gain) < MIN_ACTION_DB:
            return []
        return [
            FinishingAction(
                kind=ActionKind.LOW_SHELF_CUT,
                trigger=RiskFlag.LOW_END_EXCESS,
                reason="energy below 150 Hz dominates the spectrum",
                metric=finding.metric,
                measured_value=finding.value,
                threshold=LOW_END_EXCESS_SHARE,
                gain_db=gain,
                requested_gain_db=requested,
                ceiling_db=MAX_LOW_SHELF_CUT_DB,
                clamped=clamped,
                frequency_hz=LOW_SHELF_HZ,
                q=SHELF_Q,
                notes=("kept small: the low end carries the weight of the track",),
            )
        ]

    def _low_mid_cut(
        self, analysis: AudioAnalysis, present: dict[RiskFlag, RiskFinding]
    ) -> list[FinishingAction]:
        finding = present.get(RiskFlag.LOW_MID_MUD)
        if finding is None:
            return []
        peak = analysis.frequency.low_mid_peak_hz
        if math.isnan(peak):
            return []
        requested = -(finding.value - LOW_MID_MUD_DB) * MUD_CORRECTION_FRACTION
        gain, clamped = _clamped_gain(requested, MAX_LOW_MID_CUT_DB)
        if abs(gain) < MIN_ACTION_DB:
            return []
        return [
            FinishingAction(
                kind=ActionKind.LOW_MID_CUT,
                trigger=RiskFlag.LOW_MID_MUD,
                reason="150-400 Hz sits above the body of the mix",
                metric=finding.metric,
                measured_value=finding.value,
                threshold=LOW_MID_MUD_DB,
                gain_db=gain,
                requested_gain_db=requested,
                ceiling_db=MAX_LOW_MID_CUT_DB,
                clamped=clamped,
                frequency_hz=peak,
                q=LOW_MID_Q,
                notes=(f"centred on the track's own low-mid peak at {peak:.0f} Hz",),
            )
        ]

    def _presence_lift(
        self,
        analysis: AudioAnalysis,
        present: dict[RiskFlag, RiskFinding],
        suppressed: list[SuppressedAction],
    ) -> list[FinishingAction]:
        finding = present.get(RiskFlag.PRESENCE_DEFICIT)
        if finding is None:
            return []
        if RiskFlag.HARSHNESS_RISK in present:
            # The harshness band is 2.5-5 kHz and the presence band is
            # 2-5 kHz. Lifting one is lifting the other, so a track that
            # spikes here does not get a presence lift at all.
            suppressed.append(
                SuppressedAction(
                    kind=ActionKind.PRESENCE_LIFT,
                    trigger=RiskFlag.PRESENCE_DEFICIT,
                    reason=(
                        "2.5-5 kHz already spikes, and the presence band overlaps it; "
                        "lifting one would lift the other"
                    ),
                    metric=finding.metric,
                    measured_value=finding.value,
                )
            )
            return []
        requested = (PRESENCE_DEFICIT_DB - finding.value) * CORRECTION_FRACTION
        gain, clamped = _clamped_gain(requested, MAX_PRESENCE_LIFT_DB)
        if abs(gain) < MIN_ACTION_DB:
            return []
        return [
            FinishingAction(
                kind=ActionKind.PRESENCE_LIFT,
                trigger=RiskFlag.PRESENCE_DEFICIT,
                reason="2-5 kHz sits below the body of the mix",
                metric=finding.metric,
                measured_value=finding.value,
                threshold=PRESENCE_DEFICIT_DB,
                gain_db=gain,
                requested_gain_db=requested,
                ceiling_db=MAX_PRESENCE_LIFT_DB,
                clamped=clamped,
                frequency_hz=PRESENCE_HZ,
                q=PRESENCE_Q,
            )
        ]

    def _high_shelf(
        self, analysis: AudioAnalysis, present: dict[RiskFlag, RiskFinding]
    ) -> list[FinishingAction]:
        broad = present.get(RiskFlag.HIGH_FREQUENCY_DEFICIT)
        air = present.get(RiskFlag.AIR_DEFICIT)
        finding = broad or air
        if finding is None:
            # Dark and bright are mutually exclusive by construction —
            # their thresholds are 11 dB apart — so reaching the trim
            # rule means no deficit fired.
            return self._high_shelf_trim(analysis, present)

        measured_air = analysis.frequency.air_ratio_db.p50
        anchor = HIGH_FREQUENCY_DEFICIT_AIR_DB if broad is not None else AIR_DEFICIT_DB
        requested = (anchor - measured_air) * CORRECTION_FRACTION

        ceiling = MAX_HIGH_SHELF_DB
        notes: list[str] = []
        if RiskFlag.SIBILANCE_RISK in present:
            ceiling = min(ceiling, SIBILANCE_SHELF_CEILING_DB)
            notes.append("ceiling lowered because 6-9 kHz already spikes")
        if RiskFlag.HARSHNESS_RISK in present:
            ceiling = min(ceiling, HARSHNESS_SHELF_CEILING_DB)
            notes.append("ceiling lowered because 2.5-5 kHz already spikes")

        gain, clamped = _clamped_gain(requested, ceiling)
        if abs(gain) < MIN_ACTION_DB:
            return []

        # A track that is dark across the whole top needs the shelf to
        # start lower than one whose mids are healthy and whose top
        # octave alone is missing.
        corner = BROAD_SHELF_HZ if broad is not None else AIR_SHELF_HZ
        notes.append(f"shelf corner at {corner:.0f} Hz")
        return [
            FinishingAction(
                kind=ActionKind.HIGH_SHELF_LIFT,
                trigger=finding.flag,
                reason="high frequencies sit below the body of the mix",
                metric="frequency.air_ratio_db.p50",
                measured_value=measured_air,
                threshold=anchor,
                gain_db=gain,
                requested_gain_db=requested,
                ceiling_db=ceiling,
                clamped=clamped,
                frequency_hz=corner,
                q=SHELF_Q,
                notes=tuple(notes),
            )
        ]

    def _high_shelf_trim(
        self, analysis: AudioAnalysis, present: dict[RiskFlag, RiskFinding]
    ) -> list[FinishingAction]:
        """The mirror of the lift: a steadily over-bright master, trimmed.

        Only steady brightness qualifies. A track that spikes in 6-9 kHz
        is sibilant, not bright, and a shelf is the wrong tool for it —
        it would pull down the cymbals and string texture along with the
        sibilants and still leave the spikes proud of everything else.
        That is why the brightness flag requires a shallow slope as well
        as a high air ratio, and why nothing here reads the spike bands.
        """
        finding = present.get(RiskFlag.EXCESSIVE_BRIGHTNESS)
        if finding is None:
            return []

        measured_air = analysis.frequency.air_ratio_db.p50
        requested = (EXCESSIVE_BRIGHTNESS_AIR_DB - measured_air) * CORRECTION_FRACTION
        gain, clamped = _clamped_gain(requested, MAX_HIGH_SHELF_CUT_DB)
        if abs(gain) < MIN_ACTION_DB:
            return []
        return [
            FinishingAction(
                kind=ActionKind.HIGH_SHELF_CUT,
                trigger=RiskFlag.EXCESSIVE_BRIGHTNESS,
                reason="high frequencies sit above the body of the mix",
                metric="frequency.air_ratio_db.p50",
                measured_value=measured_air,
                threshold=EXCESSIVE_BRIGHTNESS_AIR_DB,
                gain_db=gain,
                requested_gain_db=requested,
                ceiling_db=MAX_HIGH_SHELF_CUT_DB,
                clamped=clamped,
                frequency_hz=AIR_SHELF_HZ,
                q=SHELF_Q,
                notes=(
                    "corner at 10 kHz, above the presence range, so the trim does "
                    "not touch vocal intelligibility",
                ),
            )
        ]

    def _stereo(
        self,
        analysis: AudioAnalysis,
        present: dict[RiskFlag, RiskFinding],
        suppressed: list[SuppressedAction],
    ) -> list[FinishingAction]:
        stereo = analysis.stereo
        if not stereo.is_stereo or stereo.width is None:
            return []

        actions: list[FinishingAction] = []
        narrow = present.get(RiskFlag.STEREO_TOO_NARROW)
        wide = present.get(RiskFlag.STEREO_TOO_WIDE)
        width_finding = narrow if narrow is not None else wide
        # Widening boosts the side channel, which is precisely what a
        # mono fold-down cancels. On material whose channels already
        # disagree that trades a narrow image for a hollow one, so the
        # phase measurement outranks the width measurement.
        if narrow is not None:
            correlation = stereo.correlation
            unsafe = RiskFlag.BROADBAND_PHASE_RISK in present or (
                correlation is not None and correlation < SAFE_TO_WIDEN_CORRELATION
            )
            if unsafe:
                suppressed.append(
                    SuppressedAction(
                        kind=ActionKind.STEREO_WIDTH,
                        trigger=RiskFlag.STEREO_TOO_NARROW,
                        reason=(
                            "the channels are already too decorrelated to widen safely; "
                            "boosting the side channel would cost mono compatibility"
                        ),
                        metric="stereo.correlation",
                        measured_value=(correlation if correlation is not None else float("nan")),
                    )
                )
                narrow = None
                width_finding = wide

        if width_finding is not None:
            widening = narrow is not None
            target = TARGET_NARROW_WIDTH if widening else TARGET_WIDE_WIDTH
            threshold = STEREO_NARROW_WIDTH if widening else STEREO_WIDE_WIDTH
            requested = _width_gain_db(stereo.width, target)
            gain, clamped = _clamped_gain(requested, MAX_WIDTH_ADJUST_DB)
            if abs(gain) >= MIN_ACTION_DB:
                actions.append(
                    FinishingAction(
                        kind=ActionKind.STEREO_WIDTH,
                        trigger=width_finding.flag,
                        reason=("stereo image is narrow" if widening else "stereo image is wide"),
                        metric="stereo.width",
                        measured_value=stereo.width,
                        threshold=threshold,
                        gain_db=gain,
                        requested_gain_db=requested,
                        ceiling_db=MAX_WIDTH_ADJUST_DB,
                        clamped=clamped,
                        notes=("applied to the side channel only",),
                    )
                )

        # Bass is summed to mono whenever the low band is out of phase,
        # and unconditionally whenever the width was touched. The second
        # case is the invariant that matters: a side-channel boost lifts
        # every frequency including the bass, and widened bass is the one
        # stereo failure that survives into mono as missing low end.
        phase = present.get(RiskFlag.LOW_END_PHASE_RISK)
        widened = any(
            action.kind == ActionKind.STEREO_WIDTH and (action.gain_db or 0.0) > 0.0
            for action in actions
        )
        if phase is not None or widened:
            correlation = stereo.low_band_correlation
            actions.append(
                FinishingAction(
                    kind=ActionKind.LOW_FREQUENCY_MONO,
                    trigger=RiskFlag.LOW_END_PHASE_RISK if phase is not None else None,
                    reason=(
                        "channels disagree below 120 Hz"
                        if phase is not None
                        else "bass must not be widened along with the rest of the image"
                    ),
                    metric="stereo.low_band_correlation",
                    measured_value=correlation if correlation is not None else float("nan"),
                    threshold=phase.threshold if phase is not None else float("nan"),
                    gain_db=None,
                    requested_gain_db=None,
                    ceiling_db=None,
                    clamped=False,
                    frequency_hz=LOW_MONO_CROSSOVER_HZ,
                    notes=("Linkwitz-Riley crossover; the two bands sum flat",),
                )
            )
        return actions
