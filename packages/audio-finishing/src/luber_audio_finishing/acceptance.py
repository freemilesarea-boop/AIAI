"""Deciding whether a finished render deserves to replace the raw master.

Phase 14 proved a render was *safe* — no clipping, no peak overrun, the
same duration and channel count. That is a lower bar than it sounds.
Every one of those checks passes on a render that did the opposite of
what it intended, because a mis-signed shelf is exactly as peak-safe as
a correct one. The engine could apply a lift, measurably darken the
track, and ship it.

So safety is necessary and not sufficient. A render is accepted here
only if it is also *effective* and *non-regressive*:

*Effective.* Every action the plan took names a metric and a direction.
The finished audio has to have moved that metric that way, by a
worthwhile share of what the rule asked for. A shelf that requested
3 dB of air and delivered 0.2 dB did not work, whatever the reason, and
shipping it spends a processing stage on nothing.

*Non-regressive.* Correction is not free. Lifting the top of a mix
raises sibilance along with everything else — measured, not assumed: on
one baseline master the Phase 14 shelf moved sibilance peak excess from
15.68 to 16.03 dB and harshness from 13.03 to 13.52 dB. Neither crossed
a threshold, so neither was caught, and nothing in the engine was
watching. The dimensions a plan did *not* target are therefore checked
too, and a render that trades a fixed dullness for a new harshness is
refused.

When a render fails either test the raw master is the deliverable. That
is the whole point of never overwriting it: there is always something
correct to fall back to, so the engine is free to refuse its own work
rather than ship the best of two bad options.

Rejection is not failure. A failure means the engine could not run; a
rejection means it ran, looked at what it produced, and judged the raw
master better. The two are recorded separately because they call for
completely different responses.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from luber_audio_finishing.analysis import AudioAnalysis
from luber_audio_finishing.decision import ActionKind, FinishingPlan
from luber_audio_finishing.risks import STEREO_IMBALANCE_DB

# ── Tolerances ───────────────────────────────────────────────────────
#
# True peak is allowed this much slack above the ceiling, covering
# inter-sample estimation differences between the measuring and
# rendering passes.
PEAK_TOLERANCE_DB = 0.1
#: Every filter in the chain is a zero-delay biquad or a channel-matrix
#: operation, so any duration change beyond a rounding error is a bug.
DURATION_TOLERANCE_SECONDS = 0.01

#: Share of the requested move a render must deliver, per filter shape.
#:
#: One floor for every action was the first attempt and it was wrong. How
#: much of a filter's gain reaches the metric depends on how well the
#: filter's shape overlaps the band the metric averages, and shelves and
#: bells differ completely there. Measured across the corpus:
#:
#: * ``HIGH_SHELF_LIFT`` delivered 0.90-1.34 of its gain (n=9)
#: * ``HIGH_SHELF_CUT`` delivered 0.74 (n=1)
#: * ``PRESENCE_LIFT`` delivered 0.86 (n=1)
#: * ``LOW_MID_CUT`` delivered 0.06-0.72, median 0.49 (n=8)
#:
#: A shelf moves an entire band, so nearly all of its gain reaches the
#: measurement. Half is a generous floor and any shelf failing it has
#: genuinely not worked.
MIN_EFFICACY_FRACTION = 0.5

#: The low-mid cut is the exception, and the reason is geometric. It is a
#: bell centred on the track's *own* low-mid peak, judged against a fixed
#: 150-400 Hz average. When the peak sits mid-band the two overlap well;
#: when it sits at the edge most of the filter acts outside the window
#: being measured. The two weakest results in the corpus — 0.06 and 0.26
#: — are bells centred at 387 Hz and 363 Hz, against a band ending at
#: 400. Those renders are not wrong; the metric is simply a poor witness
#: to what they did.
#:
#: So this floor is set to catch a cut that achieved essentially nothing
#: while accepting an edge-centred bell that genuinely acted. It is
#: knowingly the weakest check here: it can confirm that a bell moved the
#: band the right way, and it cannot hold one to a magnitude.
MIN_BELL_EFFICACY_FRACTION = 0.20

#: A metric that should not have moved may still drift this far. Filters
#: are not perfectly orthogonal and the analysis itself has run-to-run
#: variance well under this.
REGRESSION_TOLERANCE_DB = 0.75
#: Correlation is bounded and unitless, so it gets its own tolerance.
CORRELATION_REGRESSION = 0.05
#: Finished audio may not end up louder than the raw. A louder file wins
#: comparisons for reasons that have nothing to do with the processing.
MAX_LOUDNESS_INCREASE_LU = 0.1


class CheckKind(StrEnum):
    """Why a check exists, which determines what its failure means."""

    #: The render is technically unsound. Never shippable.
    SAFETY = "SAFETY"
    #: The render did not achieve what it set out to achieve.
    EFFICACY = "EFFICACY"
    #: The render damaged something it was not trying to change.
    REGRESSION = "REGRESSION"


class AcceptanceOutcome(StrEnum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class AcceptanceCheck:
    """One question asked of the finished audio, and its answer."""

    kind: CheckKind
    name: str
    passed: bool
    detail: str
    #: The numbers behind the verdict, so it can be re-read and argued
    #: with rather than believed.
    source_value: float | None = None
    finished_value: float | None = None
    tolerance: float | None = None


@dataclass(frozen=True)
class AcceptanceVerdict:
    outcome: AcceptanceOutcome
    checks: tuple[AcceptanceCheck, ...]

    @property
    def accepted(self) -> bool:
        return self.outcome is AcceptanceOutcome.ACCEPTED

    @property
    def failures(self) -> tuple[AcceptanceCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(f"{check.name}: {check.detail}" for check in self.failures)

    def summary(self) -> str:
        if self.accepted:
            return f"accepted on {len(self.checks)} checks"
        return "rejected — " + "; ".join(self.reasons)


@dataclass(frozen=True)
class _Target:
    """A metric an action claims to move, and which way."""

    name: str
    source: float | None
    finished: float | None
    requested_delta: float
    #: Share of the request this filter shape is expected to deliver.
    min_fraction: float = MIN_EFFICACY_FRACTION


def _metric(analysis: AudioAnalysis, name: str) -> float | None:
    """Read a dotted metric path, or ``None`` when it does not exist."""
    value: object = analysis
    for part in name.split("."):
        value = getattr(value, part, None)
        if value is None:
            return None
    if isinstance(value, (int, float)):
        number = float(value)
        return None if number != number else number  # NaN is not a measurement
    return None


def _targets(plan: FinishingPlan, source: AudioAnalysis, finished: AudioAnalysis) -> list[_Target]:
    """What each action promised to change, paired with what happened.

    Deliberately not every action. ``LOW_FREQUENCY_MONO`` has no gain and
    no promise — it is a constraint on the stereo stage, not a correction
    — and a track whose bass was already mono-compatible will show no
    movement at all in the metric. Judging it by movement would reject
    renders for doing the right thing.
    """
    metric_for: dict[ActionKind, str] = {
        ActionKind.LOW_MID_CUT: "frequency.low_mid_ratio_db.p50",
        ActionKind.PRESENCE_LIFT: "frequency.presence_ratio_db.p50",
        ActionKind.HIGH_SHELF_LIFT: "frequency.air_ratio_db.p50",
        ActionKind.HIGH_SHELF_CUT: "frequency.air_ratio_db.p50",
        ActionKind.LOW_SHELF_CUT: "frequency.bands.low_share",
        ActionKind.STEREO_WIDTH: "stereo.width",
        ActionKind.BALANCE_CORRECTION: "stereo.lr_balance_db",
    }
    targets: list[_Target] = []
    for action in plan.actions:
        metric = metric_for.get(action.kind)
        if metric is None or action.gain_db is None:
            continue
        # Width and balance are not in dB, so the requested move is
        # expressed in the metric's own units rather than the filter's.
        if action.kind == ActionKind.STEREO_WIDTH:
            before, after = source.stereo.width, finished.stereo.width
            requested = 1.0 if action.gain_db > 0 else -1.0
        elif action.kind == ActionKind.BALANCE_CORRECTION:
            before = source.stereo.lr_balance_db
            after = finished.stereo.lr_balance_db
            # Balance is judged by magnitude: the promise is "closer to
            # centre", whichever side it started on.
            before = None if before is None else abs(before)
            after = None if after is None else abs(after)
            requested = -abs(action.gain_db)
        elif action.kind == ActionKind.LOW_SHELF_CUT:
            before, after = _low_share(source), _low_share(finished)
            requested = -1.0
        else:
            before, after = _metric(source, metric), _metric(finished, metric)
            requested = action.gain_db
        targets.append(
            _Target(
                name=metric,
                source=before,
                finished=after,
                requested_delta=requested,
                min_fraction=(
                    MIN_BELL_EFFICACY_FRACTION
                    if action.kind is ActionKind.LOW_MID_CUT
                    else MIN_EFFICACY_FRACTION
                ),
            )
        )
    return targets


def _low_share(analysis: AudioAnalysis) -> float | None:
    """Share of banded energy below 150 Hz."""
    shares = [
        band.share
        for band in analysis.frequency.bands
        if band.name in ("sub", "bass") and band.share is not None
    ]
    return sum(shares) if shares else None


def _safety_checks(
    source: AudioAnalysis, finished: AudioAnalysis, ceiling_dbfs: float
) -> list[AcceptanceCheck]:
    checks: list[AcceptanceCheck] = []

    clipped = finished.level.clipped_samples
    checks.append(
        AcceptanceCheck(
            kind=CheckKind.SAFETY,
            name="no clipping",
            passed=clipped == 0,
            detail=(
                "no samples at or beyond full scale"
                if clipped == 0
                else f"{clipped} clipped samples"
            ),
            finished_value=float(clipped),
            tolerance=0.0,
        )
    )

    true_peak = finished.loudness.true_peak_dbfs
    if true_peak is not None:
        within = true_peak <= ceiling_dbfs + PEAK_TOLERANCE_DB
        checks.append(
            AcceptanceCheck(
                kind=CheckKind.SAFETY,
                name="true peak under ceiling",
                passed=within,
                detail=(
                    f"true peak {true_peak:.2f} dBFS against a {ceiling_dbfs:.2f} ceiling"
                    if within
                    else f"true peak {true_peak:.2f} dBFS exceeds the {ceiling_dbfs:.2f} ceiling"
                ),
                finished_value=true_peak,
                tolerance=ceiling_dbfs + PEAK_TOLERANCE_DB,
            )
        )

    sample_peak = finished.level.peak_dbfs
    within_sample = sample_peak <= ceiling_dbfs + PEAK_TOLERANCE_DB
    checks.append(
        AcceptanceCheck(
            kind=CheckKind.SAFETY,
            name="sample peak under ceiling",
            passed=within_sample,
            detail=(
                f"sample peak {sample_peak:.2f} dBFS"
                if within_sample
                else f"sample peak {sample_peak:.2f} dBFS exceeds the {ceiling_dbfs:.2f} ceiling"
            ),
            finished_value=sample_peak,
            tolerance=ceiling_dbfs + PEAK_TOLERANCE_DB,
        )
    )

    drift = abs(finished.technical.duration_seconds - source.technical.duration_seconds)
    checks.append(
        AcceptanceCheck(
            kind=CheckKind.SAFETY,
            name="duration preserved",
            passed=drift <= DURATION_TOLERANCE_SECONDS,
            detail=(
                f"duration moved by {drift:.4f} s"
                if drift > DURATION_TOLERANCE_SECONDS
                else "duration unchanged"
            ),
            source_value=source.technical.duration_seconds,
            finished_value=finished.technical.duration_seconds,
            tolerance=DURATION_TOLERANCE_SECONDS,
        )
    )

    same_rate = finished.technical.sample_rate == source.technical.sample_rate
    checks.append(
        AcceptanceCheck(
            kind=CheckKind.SAFETY,
            name="sample rate preserved",
            passed=same_rate,
            detail=(
                "sample rate unchanged"
                if same_rate
                else f"sample rate changed {source.technical.sample_rate} -> "
                f"{finished.technical.sample_rate}"
            ),
            source_value=float(source.technical.sample_rate),
            finished_value=float(finished.technical.sample_rate),
        )
    )

    same_channels = finished.technical.channels == source.technical.channels
    checks.append(
        AcceptanceCheck(
            kind=CheckKind.SAFETY,
            name="channel count preserved",
            passed=same_channels,
            detail=(
                "channel count unchanged"
                if same_channels
                else f"channel count changed {source.technical.channels} -> "
                f"{finished.technical.channels}"
            ),
            source_value=float(source.technical.channels),
            finished_value=float(finished.technical.channels),
        )
    )

    # The engine must never leave a file more off-centre than the amount
    # it treats as a defect in the first place. The mid/side stage can
    # shift balance as a side effect, so this is checked, not assumed.
    balance = finished.stereo.lr_balance_db
    if balance is not None:
        centred = abs(balance) <= STEREO_IMBALANCE_DB
        checks.append(
            AcceptanceCheck(
                kind=CheckKind.SAFETY,
                name="image is centred",
                passed=centred,
                detail=(
                    f"balance {balance:+.2f} dB"
                    if centred
                    else f"output balance {balance:+.2f} dB is off centre"
                ),
                source_value=source.stereo.lr_balance_db,
                finished_value=balance,
                tolerance=STEREO_IMBALANCE_DB,
            )
        )
    return checks


def _efficacy_checks(targets: list[_Target]) -> list[AcceptanceCheck]:
    checks: list[AcceptanceCheck] = []
    for target in targets:
        if target.source is None or target.finished is None:
            # Not measurable in both files. Silence is the honest answer;
            # inventing a pass or a fail would both be wrong.
            continue
        achieved = target.finished - target.source
        wanted = target.requested_delta
        required = abs(wanted) * target.min_fraction
        # Width and low-share requests carry direction only, so they are
        # judged on sign alone: any movement the right way counts.
        directional = abs(wanted) == 1.0 and target.name in (
            "stereo.width",
            "frequency.bands.low_share",
        )
        if directional:
            passed = (achieved * wanted) > 0.0
            detail = (
                f"{target.name} moved {achieved:+.4f}, the intended direction"
                if passed
                else f"{target.name} moved {achieved:+.4f}, the wrong direction"
            )
        else:
            passed = (achieved * wanted) > 0.0 and abs(achieved) >= required
            detail = (
                f"{target.name} moved {achieved:+.2f} against {wanted:+.2f} requested"
                if passed
                else (
                    f"{target.name} moved {achieved:+.2f} against {wanted:+.2f} requested; "
                    f"at least {required:.2f} in that direction was needed"
                )
            )
        checks.append(
            AcceptanceCheck(
                kind=CheckKind.EFFICACY,
                name=f"correction took effect: {target.name}",
                passed=passed,
                detail=detail,
                source_value=target.source,
                finished_value=target.finished,
                tolerance=required,
            )
        )
    return checks


def _regression_checks(
    source: AudioAnalysis, finished: AudioAnalysis, plan: FinishingPlan
) -> list[AcceptanceCheck]:
    """What the plan was not aiming at, and must not have broken."""
    checks: list[AcceptanceCheck] = []

    def rose(name: str, label: str, before: float | None, after: float | None, tol: float) -> None:
        if before is None or after is None:
            return
        delta = after - before
        checks.append(
            AcceptanceCheck(
                kind=CheckKind.REGRESSION,
                name=name,
                passed=delta <= tol,
                detail=(
                    f"{label} moved {delta:+.2f}"
                    if delta <= tol
                    else f"{label} worsened by {delta:+.2f}, past the {tol:.2f} tolerance"
                ),
                source_value=before,
                finished_value=after,
                tolerance=tol,
            )
        )

    def fell(name: str, label: str, before: float | None, after: float | None, tol: float) -> None:
        if before is None or after is None:
            return
        delta = after - before
        checks.append(
            AcceptanceCheck(
                kind=CheckKind.REGRESSION,
                name=name,
                passed=delta >= -tol,
                detail=(
                    f"{label} moved {delta:+.3f}"
                    if delta >= -tol
                    else f"{label} fell by {abs(delta):.3f}, past the {tol:.3f} tolerance"
                ),
                source_value=before,
                finished_value=after,
                tolerance=tol,
            )
        )

    # The failure mode the Phase 14 engine could not see: buying air with
    # sibilance. Both bands are watched whether or not the plan touched
    # them, because the shelf that lifts one lifts the other.
    rose(
        "sibilance not worsened",
        "6-9 kHz peak excess",
        source.sibilance.sibilance_peak_excess_db,
        finished.sibilance.sibilance_peak_excess_db,
        REGRESSION_TOLERANCE_DB,
    )
    rose(
        "harshness not worsened",
        "2.5-5 kHz peak excess",
        source.sibilance.harshness_peak_excess_db,
        finished.sibilance.harshness_peak_excess_db,
        REGRESSION_TOLERANCE_DB,
    )

    # Mono compatibility. Widening trades against it directly, so a
    # render that widened the image and lost bass correlation has made
    # the track worse on every phone in the world.
    fell(
        "mono compatibility preserved",
        "broadband correlation",
        source.stereo.correlation,
        finished.stereo.correlation,
        CORRELATION_REGRESSION,
    )
    fell(
        "bass mono compatibility preserved",
        "low-band correlation",
        source.stereo.low_band_correlation,
        finished.stereo.low_band_correlation,
        CORRELATION_REGRESSION,
    )

    # The engine owns no dynamics processing, so any material change in
    # crest factor is a side effect of something that was supposed to be
    # spectral or spatial.
    fell(
        "dynamics preserved",
        "crest factor",
        source.level.crest_factor_db,
        finished.level.crest_factor_db,
        REGRESSION_TOLERANCE_DB,
    )

    # Loudness may fall — peak safety sometimes requires it — but must
    # not rise, or the comparison stops being about tone.
    source_lufs = source.loudness.integrated_lufs
    finished_lufs = finished.loudness.integrated_lufs
    if source_lufs is not None and finished_lufs is not None:
        delta = finished_lufs - source_lufs
        checks.append(
            AcceptanceCheck(
                kind=CheckKind.REGRESSION,
                name="no loudness advantage",
                passed=delta <= MAX_LOUDNESS_INCREASE_LU,
                detail=(
                    f"loudness moved {delta:+.2f} LU"
                    if delta <= MAX_LOUDNESS_INCREASE_LU
                    else f"finished audio is {delta:+.2f} LU louder than the raw master"
                ),
                source_value=source_lufs,
                finished_value=finished_lufs,
                tolerance=MAX_LOUDNESS_INCREASE_LU,
            )
        )

    # Low-mid thickness, unless that is what the plan was correcting.
    if plan.action(ActionKind.LOW_MID_CUT) is None:
        rose(
            "low-mid not thickened",
            "150-400 Hz ratio",
            source.frequency.low_mid_ratio_db.p50,
            finished.frequency.low_mid_ratio_db.p50,
            REGRESSION_TOLERANCE_DB,
        )
    return checks


def adjudicate(
    plan: FinishingPlan,
    source: AudioAnalysis,
    finished: AudioAnalysis,
    *,
    ceiling_dbfs: float | None = None,
) -> AcceptanceVerdict:
    """Should this render replace the raw master?

    Every check runs even after one has failed, so the verdict carries
    the full picture rather than the first objection. Any failure, of
    any kind, rejects: an unsafe render, an ineffective one and a
    regressive one are all worse than the raw master, which is already
    a legitimate deliverable.
    """
    ceiling = plan.output_ceiling_dbfs if ceiling_dbfs is None else ceiling_dbfs
    checks = [
        *_safety_checks(source, finished, ceiling),
        *_efficacy_checks(_targets(plan, source, finished)),
        *_regression_checks(source, finished, plan),
    ]
    outcome = (
        AcceptanceOutcome.ACCEPTED
        if all(check.passed for check in checks)
        else AcceptanceOutcome.REJECTED
    )
    return AcceptanceVerdict(outcome=outcome, checks=tuple(checks))
