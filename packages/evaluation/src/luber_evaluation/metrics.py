"""Metrics that cannot lie about what they measured.

Three rules, and the whole module exists to enforce them.

**A missing measurement is never zero.** Every result carries a status,
and `NOT_MEASURABLE` is a first-class outcome. Substituting 0.0 for "no
ASR exists to check this" turns an absent capability into a terrible
score, and a comparison built on it would confidently report a
regression that never happened.

**Direction is declared, not assumed.** Higher is not always better. A
failure rate falling is an improvement; a spectral centroid rising is
neither — it is a fact about brightness, and rewarding it would push
every candidate toward a tonal preference nobody chose.

**Some dimensions have no honest automatic metric.** Melody, hook
strength, emotional impact, vocal naturalness, trot-like delivery: no
validated detector for any of them exists in this project. They are
declared `HUMAN_REQUIRED` and cannot be measured into a pass. Writing a
plausible heuristic and calling it a metric would be the single most
damaging thing this package could do, because the number would look
exactly like a real one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from statistics import median
from typing import Any

METRIC_SCHEMA_VERSION = "luber-evaluation-metric/1"


class MetricStatus(StrEnum):
    """Whether a number exists, and why not when it does not."""

    MEASURED = "MEASURED"
    #: No validated way to measure this exists. Not a bad score.
    NOT_MEASURABLE = "NOT_MEASURABLE"
    #: The metric does not apply to this case — vocal metrics on an
    #: instrumental, for instance.
    NOT_APPLICABLE = "NOT_APPLICABLE"
    #: Measurement was attempted and errored.
    FAILED = "FAILED"


class MetricDirection(StrEnum):
    HIGHER_BETTER = "HIGHER_BETTER"
    LOWER_BETTER = "LOWER_BETTER"
    #: Good inside a stated range; outside it in either direction is worse.
    TARGET_RANGE = "TARGET_RANGE"
    #: Worth recording, meaningless to optimise.
    INFORMATIONAL = "INFORMATIONAL"


class MetricSource(StrEnum):
    #: Computed from generated audio by the finishing engine's analyser.
    AUDIO_ANALYSIS = "AUDIO_ANALYSIS"
    #: Counted by the evaluation runner from generation outcomes.
    GENERATION_OUTCOME = "GENERATION_OUTCOME"
    #: Read from the training run's metrics.
    TRAINING_TELEMETRY = "TRAINING_TELEMETRY"
    #: Supplied by a human reviewer.
    HUMAN = "HUMAN"
    #: Produced by the synthetic backend. Never a measurement of audio.
    SIMULATED = "SIMULATED"


class MeasurementMode(StrEnum):
    """Whether a dimension can be measured at all, and by whom."""

    AUTOMATIC = "AUTOMATIC"
    HUMAN_REQUIRED = "HUMAN_REQUIRED"
    #: Automatic in principle; the component does not exist here yet.
    NOT_AVAILABLE = "NOT_AVAILABLE"


@dataclass(frozen=True)
class MetricDefinition:
    """What a metric means, and how it may be used.

    Versioned individually. Changing how a number is computed without
    bumping its version would make two evaluations silently
    incomparable — the worst kind of incomparable, because the columns
    still line up.
    """

    name: str
    unit: str
    direction: str
    mode: str
    description: str
    metric_version: str = "1"
    #: Only for TARGET_RANGE.
    target_low: float | None = None
    target_high: float | None = None
    #: Why no automatic measurement exists, when mode is not AUTOMATIC.
    unavailability_reason: str = ""
    #: Smallest movement this metric can resolve, on its own scale.
    #: Set only where the shared default is wrong for the quantity: a
    #: metric whose entire acceptable range sits below a generic noise
    #: floor could never register a change, and every improvement in it
    #: would be reported as inconclusive forever.
    noise_floor: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MetricResult:
    """One measurement, of one case, at one seed."""

    metric_name: str
    status: str
    case_id: str
    seed: int | None
    source: str
    metric_version: str = "1"
    value: float | None = None
    unit: str = ""
    confidence: float | None = None
    #: Why a non-MEASURED status was recorded.
    detail: str = ""

    def __post_init__(self) -> None:
        if self.status != MetricStatus.MEASURED.value and self.value is not None:
            raise ValueError(
                f"{self.metric_name} has status {self.status} but carries a value; "
                "a non-measured metric must not hold a number"
            )
        if self.status == MetricStatus.MEASURED.value and self.value is None:
            raise ValueError(f"{self.metric_name} is MEASURED but has no value")

    @property
    def measured(self) -> bool:
        return self.status == MetricStatus.MEASURED.value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def not_measurable(
    metric_name: str, case_id: str, reason: str, *, seed: int | None = None
) -> MetricResult:
    """A metric nothing here can measure. Never a zero."""
    return MetricResult(
        metric_name=metric_name,
        status=MetricStatus.NOT_MEASURABLE.value,
        case_id=case_id,
        seed=seed,
        source=MetricSource.AUDIO_ANALYSIS.value,
        detail=reason,
    )


# ── the catalogue ────────────────────────────────────────────────────
#
# Every metric this project can produce, with its direction and whether
# it is automatic at all. The `HUMAN_REQUIRED` entries are the honest
# part: they exist so a policy can *require* human evidence for them
# rather than a heuristic quietly standing in.

RELIABILITY_METRICS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        name="generation_success_rate",
        unit="fraction",
        direction=MetricDirection.HIGHER_BETTER.value,
        mode=MeasurementMode.AUTOMATIC.value,
        description="share of requested generations that returned decodable audio",
    ),
    MetricDefinition(
        name="generation_failure_rate",
        unit="fraction",
        direction=MetricDirection.LOWER_BETTER.value,
        mode=MeasurementMode.AUTOMATIC.value,
        description="share that errored",
    ),
    MetricDefinition(
        name="generation_timeout_rate",
        unit="fraction",
        direction=MetricDirection.LOWER_BETTER.value,
        mode=MeasurementMode.AUTOMATIC.value,
        description="share that exceeded the inference timeout",
    ),
    MetricDefinition(
        name="invalid_audio_rate",
        unit="fraction",
        direction=MetricDirection.LOWER_BETTER.value,
        mode=MeasurementMode.AUTOMATIC.value,
        description="share whose output would not decode or contained NaN/Inf",
    ),
    MetricDefinition(
        name="silent_output_rate",
        unit="fraction",
        direction=MetricDirection.LOWER_BETTER.value,
        mode=MeasurementMode.AUTOMATIC.value,
        description="share whose output was effectively silent",
    ),
    MetricDefinition(
        name="early_collapse_rate",
        unit="fraction",
        direction=MetricDirection.LOWER_BETTER.value,
        mode=MeasurementMode.AUTOMATIC.value,
        description="share that decayed into silence well before the requested duration",
    ),
    MetricDefinition(
        name="wrong_duration_rate",
        unit="fraction",
        direction=MetricDirection.LOWER_BETTER.value,
        mode=MeasurementMode.AUTOMATIC.value,
        description="share whose duration error exceeded the suite tolerance",
    ),
)

TECHNICAL_METRICS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        name="duration_absolute_error_seconds",
        unit="seconds",
        direction=MetricDirection.LOWER_BETTER.value,
        mode=MeasurementMode.AUTOMATIC.value,
        description="|actual - requested| duration",
    ),
    MetricDefinition(
        name="duration_relative_error",
        unit="fraction",
        direction=MetricDirection.LOWER_BETTER.value,
        mode=MeasurementMode.AUTOMATIC.value,
        description="absolute duration error over requested duration",
    ),
    MetricDefinition(
        name="clipping_sample_ratio",
        unit="fraction",
        direction=MetricDirection.LOWER_BETTER.value,
        mode=MeasurementMode.AUTOMATIC.value,
        description="share of samples at or beyond full scale",
        # A hundredth of the shared rate floor, because this is not a
        # count over cases but a share of samples within audio, and its
        # hard ceiling (0.01) is itself below that floor. The analyser
        # counts clipped samples exactly, so movement at this scale is
        # a change in behaviour rather than sampling luck.
        noise_floor=0.0002,
    ),
    MetricDefinition(
        name="silence_ratio",
        unit="fraction",
        direction=MetricDirection.LOWER_BETTER.value,
        mode=MeasurementMode.AUTOMATIC.value,
        description="share of the output that is effectively silent",
    ),
    MetricDefinition(
        name="peak_dbfs",
        unit="dBFS",
        direction=MetricDirection.INFORMATIONAL.value,
        mode=MeasurementMode.AUTOMATIC.value,
        description="sample peak",
    ),
    MetricDefinition(
        name="true_peak_dbtp",
        unit="dBTP",
        direction=MetricDirection.INFORMATIONAL.value,
        mode=MeasurementMode.AUTOMATIC.value,
        description="inter-sample peak",
    ),
    MetricDefinition(
        name="integrated_lufs",
        unit="LUFS",
        direction=MetricDirection.INFORMATIONAL.value,
        mode=MeasurementMode.AUTOMATIC.value,
        description="integrated loudness; a level, not a quality",
    ),
    MetricDefinition(
        name="crest_factor_db",
        unit="dB",
        direction=MetricDirection.INFORMATIONAL.value,
        mode=MeasurementMode.AUTOMATIC.value,
        description="peak over RMS; low values suggest brickwalling",
    ),
    MetricDefinition(
        name="spectral_centroid_hz",
        unit="Hz",
        direction=MetricDirection.INFORMATIONAL.value,
        mode=MeasurementMode.AUTOMATIC.value,
        description=(
            "brightness. Informational on purpose: rewarding a higher centroid would "
            "push every candidate toward a tonal preference nobody chose"
        ),
    ),
    MetricDefinition(
        name="high_frequency_energy_ratio",
        unit="fraction",
        direction=MetricDirection.INFORMATIONAL.value,
        mode=MeasurementMode.AUTOMATIC.value,
        description="share of banded energy above 2 kHz",
    ),
    MetricDefinition(
        name="stereo_width",
        unit="ratio",
        direction=MetricDirection.INFORMATIONAL.value,
        mode=MeasurementMode.AUTOMATIC.value,
        description="side over mid+side",
    ),
    MetricDefinition(
        name="phase_correlation",
        unit="correlation",
        direction=MetricDirection.HIGHER_BETTER.value,
        mode=MeasurementMode.AUTOMATIC.value,
        description="inter-channel correlation; low values collapse in mono",
    ),
    MetricDefinition(
        name="sample_rate",
        unit="Hz",
        direction=MetricDirection.INFORMATIONAL.value,
        mode=MeasurementMode.AUTOMATIC.value,
        description="output sample rate",
    ),
    MetricDefinition(
        name="channels",
        unit="count",
        direction=MetricDirection.INFORMATIONAL.value,
        mode=MeasurementMode.AUTOMATIC.value,
        description="output channel count",
    ),
)

ADHERENCE_METRICS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        name="bpm_absolute_error",
        unit="bpm",
        direction=MetricDirection.LOWER_BETTER.value,
        mode=MeasurementMode.AUTOMATIC.value,
        description=(
            "|detected - requested| tempo, recorded only when the detector's own "
            "confidence clears the suite gate"
        ),
    ),
    MetricDefinition(
        name="instrumental_adherence",
        unit="boolean",
        direction=MetricDirection.HIGHER_BETTER.value,
        mode=MeasurementMode.NOT_AVAILABLE.value,
        description="whether an instrumental request produced instrumental audio",
        unavailability_reason=(
            "no validated vocal/instrumental detector exists in this project. Phase 23 "
            "records UNCERTAIN for every track nobody labelled, and a spectral "
            "heuristic would produce errors indistinguishable from labels."
        ),
    ),
    MetricDefinition(
        name="key_adherence",
        unit="boolean",
        direction=MetricDirection.HIGHER_BETTER.value,
        mode=MeasurementMode.AUTOMATIC.value,
        description=(
            "whether the detected key matches the request, recorded only above the "
            "detector's confidence gate"
        ),
    ),
    MetricDefinition(
        name="lyric_line_coverage",
        unit="fraction",
        direction=MetricDirection.HIGHER_BETTER.value,
        mode=MeasurementMode.NOT_AVAILABLE.value,
        description="share of expected lyric lines present in the output",
        unavailability_reason=(
            "no validated speech recogniser is configured. An unvalidated ASR would "
            "report omissions the model never made, and a Korean word error rate from "
            "an unmeasured recogniser is not evidence about the model."
        ),
    ),
    MetricDefinition(
        name="lyric_word_coverage",
        unit="fraction",
        direction=MetricDirection.HIGHER_BETTER.value,
        mode=MeasurementMode.NOT_AVAILABLE.value,
        description="share of expected lyric words present in the output",
        unavailability_reason="no validated speech recogniser is configured",
    ),
)

#: Dimensions with no honest automatic metric. Declared so a policy can
#: demand human evidence for them rather than accept a heuristic.
HUMAN_REQUIRED_DIMENSIONS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        name="vocal_naturalness",
        unit="rating",
        direction=MetricDirection.HIGHER_BETTER.value,
        mode=MeasurementMode.HUMAN_REQUIRED.value,
        description="whether the voice sounds like a person singing",
        unavailability_reason="no validated automatic measure exists",
    ),
    MetricDefinition(
        name="korean_pronunciation",
        unit="rating",
        direction=MetricDirection.HIGHER_BETTER.value,
        mode=MeasurementMode.HUMAN_REQUIRED.value,
        description="whether Korean lyrics are pronounced naturally",
        unavailability_reason=(
            "needs both a validated recogniser and a native listener; neither is "
            "available automatically"
        ),
    ),
    MetricDefinition(
        name="trot_style_absence",
        unit="rating",
        direction=MetricDirection.HIGHER_BETTER.value,
        mode=MeasurementMode.HUMAN_REQUIRED.value,
        description="absence of trot-like vocal delivery where not requested",
        unavailability_reason=(
            "nothing in this project distinguishes trot from modern delivery. The P20 "
            "benchmark scores it by ear, and a fabricated detector would be worse than "
            "no measurement."
        ),
    ),
    MetricDefinition(
        name="melody_quality",
        unit="rating",
        direction=MetricDirection.HIGHER_BETTER.value,
        mode=MeasurementMode.HUMAN_REQUIRED.value,
        description="whether the melody is any good",
        unavailability_reason="no validated automatic measure exists",
    ),
    MetricDefinition(
        name="instrument_realism",
        unit="rating",
        direction=MetricDirection.HIGHER_BETTER.value,
        mode=MeasurementMode.HUMAN_REQUIRED.value,
        description="whether instruments sound like instruments",
        unavailability_reason="no validated automatic measure exists",
    ),
    MetricDefinition(
        name="musical_coherence",
        unit="rating",
        direction=MetricDirection.HIGHER_BETTER.value,
        mode=MeasurementMode.HUMAN_REQUIRED.value,
        description="whether the piece holds together as music",
        unavailability_reason="no validated automatic measure exists",
    ),
)

TRAINING_CONTEXT_METRICS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        name="final_train_loss",
        unit="loss",
        direction=MetricDirection.INFORMATIONAL.value,
        mode=MeasurementMode.AUTOMATIC.value,
        description=(
            "training loss at the end of the run. INFORMATIONAL on purpose: a lower "
            "loss is not a better model, and qualification never rests on it"
        ),
    ),
)

CATALOGUE: dict[str, MetricDefinition] = {
    definition.name: definition
    for group in (
        RELIABILITY_METRICS,
        TECHNICAL_METRICS,
        ADHERENCE_METRICS,
        HUMAN_REQUIRED_DIMENSIONS,
        TRAINING_CONTEXT_METRICS,
    )
    for definition in group
}


def definition(name: str) -> MetricDefinition:
    if name not in CATALOGUE:
        raise KeyError(
            f"unknown metric {name!r}. A metric must be declared in the catalogue with "
            "its direction and measurement mode before it can be used."
        )
    return CATALOGUE[name]


def is_human_required(name: str) -> bool:
    return definition(name).mode == MeasurementMode.HUMAN_REQUIRED.value


def automatic_metric_names() -> tuple[str, ...]:
    return tuple(
        sorted(
            name for name, spec in CATALOGUE.items() if spec.mode == MeasurementMode.AUTOMATIC.value
        )
    )


# ── aggregation ──────────────────────────────────────────────────────


@dataclass
class Aggregate:
    """A metric summarised across cases and seeds.

    Robust summaries, and the failure counts alongside them. A mean that
    silently omitted the runs that crashed would report an improvement
    on exactly the candidate that broke.
    """

    metric_name: str
    status: str
    unit: str = ""
    direction: str = MetricDirection.INFORMATIONAL.value
    count_measured: int = 0
    count_not_measurable: int = 0
    count_not_applicable: int = 0
    count_failed: int = 0
    median_value: float | None = None
    mean_value: float | None = None
    p10: float | None = None
    p90: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    #: Cases whose measurement failed, so a summary cannot hide them.
    failed_cases: list[str] = field(default_factory=list)
    detail: str = ""

    @property
    def total(self) -> int:
        return (
            self.count_measured
            + self.count_not_measurable
            + self.count_not_applicable
            + self.count_failed
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _quantile(ordered: list[float], fraction: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def aggregate(metric_name: str, results: list[MetricResult]) -> Aggregate:
    """Summarise one metric over many results.

    An aggregate with no measurements keeps the *reason* — if every
    result said NOT_MEASURABLE, the aggregate says so too rather than
    reporting an empty measurement that a comparison might treat as a
    number.
    """
    spec = CATALOGUE.get(metric_name)
    summary = Aggregate(
        metric_name=metric_name,
        status=MetricStatus.MEASURED.value,
        unit=spec.unit if spec else "",
        direction=spec.direction if spec else MetricDirection.INFORMATIONAL.value,
    )

    values: list[float] = []
    reasons: set[str] = set()
    for result in results:
        if result.status == MetricStatus.MEASURED.value and result.value is not None:
            values.append(float(result.value))
            summary.count_measured += 1
        elif result.status == MetricStatus.NOT_MEASURABLE.value:
            summary.count_not_measurable += 1
            if result.detail:
                reasons.add(result.detail)
        elif result.status == MetricStatus.NOT_APPLICABLE.value:
            summary.count_not_applicable += 1
        else:
            summary.count_failed += 1
            summary.failed_cases.append(result.case_id)
            if result.detail:
                reasons.add(result.detail)

    if not values:
        summary.status = (
            MetricStatus.NOT_MEASURABLE.value
            if summary.count_not_measurable or summary.count_not_applicable
            else MetricStatus.FAILED.value
            if summary.count_failed
            else MetricStatus.NOT_MEASURABLE.value
        )
        summary.detail = "; ".join(sorted(reasons)) or "no measurements were produced"
        return summary

    values.sort()
    summary.median_value = round(median(values), 6)
    summary.mean_value = round(sum(values) / len(values), 6)
    summary.p10 = round(_quantile(values, 0.10), 6)
    summary.p90 = round(_quantile(values, 0.90), 6)
    summary.minimum = round(values[0], 6)
    summary.maximum = round(values[-1], 6)
    if summary.count_failed:
        summary.detail = (
            f"{summary.count_failed} case(s) failed measurement and are excluded from "
            "the summary but counted here"
        )
    return summary
