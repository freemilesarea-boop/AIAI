"""Running an evaluation: generation backends, analysis, lifecycle.

Two backends, and the difference between them is enforced rather than
documented.

**A generation backend** produces audio for a case at a seed. The real
one will call the model; the synthetic one produces *metric inputs*
marked `SIMULATED` and no audio at all. A synthetic result can never be
mistaken for a measurement of a model, because it never claims a raw
audio digest and its samples carry `synthetic=True`.

**Analysis** reuses the finishing engine's analyser. Re-implementing
loudness or stereo measurement here would produce a second set of
numbers that disagree with Phase 22's in ways nobody could adjudicate.

The lifecycle enforces what the phase requires: identity is frozen when
a run starts, a cancelled run keeps everything it produced and is never
marked completed, and resume trusts a prior case only when its sample
digest still matches.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from luber_evaluation.metrics import (
    CATALOGUE,
    Aggregate,
    MeasurementMode,
    MetricResult,
    MetricSource,
    MetricStatus,
    aggregate,
)
from luber_evaluation.schemas import (
    ALLOWED_EVALUATION_TRANSITIONS,
    EVALUATION_ENGINE_VERSION,
    EVALUATION_SCHEMA_VERSION,
    CandidateLineage,
    EvaluationCase,
    EvaluationMode,
    EvaluationRunStatus,
    ModelRef,
    SampleProvenance,
    digest_of,
)
from luber_evaluation.suite import EvaluationSuite

#: A track quieter than this everywhere counts as silent output.
SILENCE_RMS_DBFS = -60.0
#: Share of the tail that must be silent to call it an early collapse.
EARLY_COLLAPSE_TAIL_SHARE = 0.25


class EvaluationError(RuntimeError):
    """Raised when an evaluation cannot proceed as requested."""


@dataclass
class GenerationOutcome:
    """What one generation attempt produced.

    ``audio_path`` is None for a synthetic backend, and that is what
    keeps synthetic results out of the audio-analysis path entirely.
    """

    case_id: str
    seed: int
    succeeded: bool
    audio_path: Path | None = None
    duration_seconds: float | None = None
    error: str | None = None
    timed_out: bool = False
    wall_seconds: float = 0.0
    #: Metric values a synthetic backend supplies directly, bypassing
    #: analysis. Always recorded with source SIMULATED.
    simulated_metrics: dict[str, float] = field(default_factory=dict)
    synthetic: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "seed": self.seed,
            "succeeded": self.succeeded,
            "audio_path": str(self.audio_path) if self.audio_path else None,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "timed_out": self.timed_out,
            "wall_seconds": self.wall_seconds,
            "synthetic": self.synthetic,
        }


class GenerationBackend(ABC):
    """Produces audio, or explicitly-marked synthetic stand-ins."""

    name: str = "abstract"
    #: Whether this backend produces real audio that can be analysed.
    produces_audio: bool = True

    @abstractmethod
    def generate(
        self, case: EvaluationCase, seed: int, model: ModelRef, output_dir: Path
    ) -> GenerationOutcome:
        """One generation for one case at one seed."""


@dataclass
class SyntheticProfile:
    """The behaviour a synthetic model exhibits.

    Explicit numbers a test dictates, not a simulation of music. This is
    how the comparison, aggregation, gate and qualification machinery
    gets exercised without a trained model existing — and every value it
    produces is stamped SIMULATED so it can never be read as evidence
    about a real one.
    """

    label: str
    metrics: dict[str, float] = field(default_factory=dict)
    failure_rate: float = 0.0
    #: Case ids the model fails outright.
    failing_cases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "metrics": dict(sorted(self.metrics.items())),
            "failure_rate": self.failure_rate,
            "failing_cases": list(self.failing_cases),
        }


class SyntheticGenerationBackend(GenerationBackend):
    """Produces metric inputs, never audio. Deterministic.

    The most important property is what it cannot do: it has no path to
    writing a WAV, so nothing downstream can mistake its output for a
    generated sample. `produces_audio` is False, and the runner routes
    around audio analysis entirely.
    """

    name = "synthetic"
    produces_audio = False

    def __init__(self, profile: SyntheticProfile) -> None:
        self.profile = profile

    def generate(
        self, case: EvaluationCase, seed: int, model: ModelRef, output_dir: Path
    ) -> GenerationOutcome:
        if case.case_id in self.profile.failing_cases:
            return GenerationOutcome(
                case_id=case.case_id,
                seed=seed,
                succeeded=False,
                error="synthetic profile marks this case as failing",
                synthetic=True,
            )
        return GenerationOutcome(
            case_id=case.case_id,
            seed=seed,
            succeeded=True,
            duration_seconds=case.spec.duration_seconds,
            simulated_metrics=dict(self.profile.metrics),
            synthetic=True,
        )


def analyse_audio_outcome(
    outcome: GenerationOutcome, case: EvaluationCase, suite: EvaluationSuite
) -> list[MetricResult]:
    """Technical metrics from real generated audio.

    Delegates measurement to `luber_audio_finishing.analyze_audio` — the
    same analyser the finishing engine decides from — and maps its
    results. Nothing is measured twice, and nothing is invented when a
    measurement is absent.
    """
    from luber_audio_finishing import AudioLoadError, analyze_audio

    results: list[MetricResult] = []

    def record(name: str, value: float | None, *, detail: str = "") -> None:
        spec = CATALOGUE.get(name)
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            results.append(
                MetricResult(
                    metric_name=name,
                    status=MetricStatus.NOT_MEASURABLE.value,
                    case_id=case.case_id,
                    seed=outcome.seed,
                    source=MetricSource.AUDIO_ANALYSIS.value,
                    detail=detail or "the analyser produced no finite value",
                )
            )
            return
        results.append(
            MetricResult(
                metric_name=name,
                status=MetricStatus.MEASURED.value,
                case_id=case.case_id,
                seed=outcome.seed,
                source=MetricSource.AUDIO_ANALYSIS.value,
                value=float(value),
                unit=spec.unit if spec else "",
            )
        )

    if outcome.audio_path is None or not outcome.audio_path.is_file():
        for name in case.applicable_metrics:
            spec = CATALOGUE.get(name)
            if spec and spec.mode == MeasurementMode.AUTOMATIC.value:
                results.append(
                    MetricResult(
                        metric_name=name,
                        status=MetricStatus.FAILED.value,
                        case_id=case.case_id,
                        seed=outcome.seed,
                        source=MetricSource.AUDIO_ANALYSIS.value,
                        detail="no audio was produced for this case",
                    )
                )
        return results

    try:
        analysis = analyze_audio(outcome.audio_path)
    except (AudioLoadError, OSError, ValueError) as exc:
        for name in case.applicable_metrics:
            spec = CATALOGUE.get(name)
            if spec and spec.mode == MeasurementMode.AUTOMATIC.value:
                results.append(
                    MetricResult(
                        metric_name=name,
                        status=MetricStatus.FAILED.value,
                        case_id=case.case_id,
                        seed=outcome.seed,
                        source=MetricSource.AUDIO_ANALYSIS.value,
                        detail=f"analysis failed: {exc}",
                    )
                )
        return results

    requested = case.spec.duration_seconds
    actual = analysis.technical.duration_seconds
    # The measured length replaces whatever the backend claimed. A
    # backend reporting the length it asked for would make truncated
    # output — the failure `wrong_duration_rate` exists to catch —
    # invisible, and a backend that measures nothing would have every
    # render counted as invalid.
    outcome.duration_seconds = actual
    record("duration_absolute_error_seconds", abs(actual - requested))
    record(
        "duration_relative_error",
        abs(actual - requested) / requested if requested > 0 else None,
        detail="the case requests no duration",
    )
    record(
        "clipping_sample_ratio",
        analysis.level.clipped_samples / analysis.technical.frames
        if analysis.technical.frames
        else None,
    )
    record("silence_ratio", analysis.level.silence_ratio)
    record("peak_dbfs", analysis.level.peak_dbfs)
    record("true_peak_dbtp", analysis.loudness.true_peak_dbfs, detail="R128 produced no reading")
    record("integrated_lufs", analysis.loudness.integrated_lufs, detail="R128 produced no reading")
    record("crest_factor_db", analysis.level.crest_factor_db)
    record("spectral_centroid_hz", analysis.frequency.spectral_centroid_hz.p50)
    record("stereo_width", analysis.stereo.width, detail="the output is mono")
    record("phase_correlation", analysis.stereo.correlation, detail="the output is mono")
    record("sample_rate", float(analysis.technical.sample_rate))
    record("channels", float(analysis.technical.channels))

    shares = {band.name: band.share for band in analysis.frequency.bands if band.share is not None}
    high = sum(shares.get(name, 0.0) for name in ("presence", "brilliance", "air", "ultra_high"))
    record("high_frequency_energy_ratio", high if shares else None)

    # Metrics nothing here can measure. Recorded with the reason rather
    # than omitted, so the gap is visible in every report.
    for name in case.applicable_metrics:
        spec = CATALOGUE.get(name)
        if spec is None:
            continue
        if spec.mode in (
            MeasurementMode.NOT_AVAILABLE.value,
            MeasurementMode.HUMAN_REQUIRED.value,
        ):
            results.append(
                MetricResult(
                    metric_name=name,
                    status=MetricStatus.NOT_MEASURABLE.value,
                    case_id=case.case_id,
                    seed=outcome.seed,
                    source=MetricSource.AUDIO_ANALYSIS.value,
                    detail=spec.unavailability_reason,
                )
            )
    return results


def synthetic_results(outcome: GenerationOutcome, case: EvaluationCase) -> list[MetricResult]:
    """Metric results from a synthetic outcome. All marked SIMULATED."""
    results: list[MetricResult] = []
    for name, value in sorted(outcome.simulated_metrics.items()):
        spec = CATALOGUE.get(name)
        if spec is None:
            continue
        if spec.mode != MeasurementMode.AUTOMATIC.value:
            # A synthetic backend must not supply a number for a
            # dimension no automatic metric can measure. Doing so would
            # be fabricating exactly the evidence this system refuses.
            results.append(
                MetricResult(
                    metric_name=name,
                    status=MetricStatus.NOT_MEASURABLE.value,
                    case_id=case.case_id,
                    seed=outcome.seed,
                    source=MetricSource.SIMULATED.value,
                    detail=(f"{name} is {spec.mode}; a synthetic backend may not supply it"),
                )
            )
            continue
        results.append(
            MetricResult(
                metric_name=name,
                status=MetricStatus.MEASURED.value,
                case_id=case.case_id,
                seed=outcome.seed,
                source=MetricSource.SIMULATED.value,
                value=float(value),
                unit=spec.unit,
            )
        )

    for name in case.applicable_metrics:
        spec = CATALOGUE.get(name)
        if spec is None or name in outcome.simulated_metrics:
            continue
        if spec.mode != MeasurementMode.AUTOMATIC.value:
            results.append(
                MetricResult(
                    metric_name=name,
                    status=MetricStatus.NOT_MEASURABLE.value,
                    case_id=case.case_id,
                    seed=outcome.seed,
                    source=MetricSource.SIMULATED.value,
                    detail=spec.unavailability_reason,
                )
            )
    return results


@dataclass
class SideResults:
    """Everything one model produced across the suite."""

    model: ModelRef
    outcomes: list[GenerationOutcome] = field(default_factory=list)
    metrics: list[MetricResult] = field(default_factory=list)
    samples: list[SampleProvenance] = field(default_factory=list)

    def reliability_metrics(self, suite: EvaluationSuite) -> list[MetricResult]:
        """Rates computed over the whole side, not per case.

        Reliability is a property of the run: one failure in three
        attempts is a 33% failure rate, and expressing it per case would
        make it disappear into a median.
        """
        total = len(self.outcomes)
        if total == 0:
            return []

        succeeded = sum(1 for o in self.outcomes if o.succeeded)
        failed = total - succeeded
        timed_out = sum(1 for o in self.outcomes if o.timed_out)

        invalid = 0
        silent = 0
        collapsed = 0
        wrong_duration = 0
        for outcome in self.outcomes:
            if not outcome.succeeded:
                continue
            case = suite.case(outcome.case_id)
            requested = case.spec.duration_seconds
            actual = outcome.duration_seconds
            if actual is None or not math.isfinite(actual) or actual <= 0:
                invalid += 1
                continue
            if requested > 0:
                error = abs(actual - requested) / requested
                if error > suite.duration_tolerance_relative:
                    wrong_duration += 1
            silence = self._metric_for(outcome, "silence_ratio")
            if silence is not None and silence >= 0.98:
                silent += 1
            elif silence is not None and silence >= EARLY_COLLAPSE_TAIL_SHARE:
                collapsed += 1

        def rate(name: str, count: int) -> MetricResult:
            return MetricResult(
                metric_name=name,
                status=MetricStatus.MEASURED.value,
                case_id="__suite__",
                seed=None,
                source=MetricSource.GENERATION_OUTCOME.value,
                value=count / total,
                unit="fraction",
            )

        return [
            rate("generation_success_rate", succeeded),
            rate("generation_failure_rate", failed),
            rate("generation_timeout_rate", timed_out),
            rate("invalid_audio_rate", invalid),
            rate("silent_output_rate", silent),
            rate("early_collapse_rate", collapsed),
            rate("wrong_duration_rate", wrong_duration),
        ]

    def _metric_for(self, outcome: GenerationOutcome, name: str) -> float | None:
        for result in self.metrics:
            if (
                result.metric_name == name
                and result.case_id == outcome.case_id
                and result.seed == outcome.seed
                and result.measured
            ):
                return result.value
        if name in outcome.simulated_metrics:
            return float(outcome.simulated_metrics[name])
        return None

    def aggregates(self, suite: EvaluationSuite) -> dict[str, Aggregate]:
        combined = [*self.metrics, *self.reliability_metrics(suite)]
        by_name: dict[str, list[MetricResult]] = {}
        for result in combined:
            by_name.setdefault(result.metric_name, []).append(result)
        return {name: aggregate(name, results) for name, results in sorted(by_name.items())}


@dataclass
class EvaluationRun:
    """One comparative evaluation, with its identity frozen at start."""

    evaluation_id: str
    suite: EvaluationSuite
    baseline: ModelRef
    candidate: ModelRef
    lineage: CandidateLineage
    #: What the experiment claimed. Carried on the run so it survives
    #: every save: a hypothesis that lived only in the record written at
    #: creation would vanish the moment the run started, and the gate
    #: that exists to stop a candidate qualifying without addressing its
    #: own claim would silently never fire.
    experiment_hypothesis: str = ""
    mode: str = EvaluationMode.RAW_MODEL.value
    status: str = EvaluationRunStatus.DRAFT.value
    suite_digest: str = ""
    policy_digest: str = ""
    seeds: tuple[int, ...] = ()
    started_at: str | None = None
    completed_at: str | None = None
    cancelled_at: str | None = None
    failed_at: str | None = None
    error: str | None = None
    #: Cost fields. Populated by a real backend; no pricing is fetched.
    wall_seconds: float = 0.0
    gpu_seconds: float | None = None
    generated_audio_seconds: float = 0.0
    engine_version: str = EVALUATION_ENGINE_VERSION
    schema_version: str = EVALUATION_SCHEMA_VERSION

    def can_transition_to(self, status: str) -> bool:
        return status in ALLOWED_EVALUATION_TRANSITIONS.get(self.status, frozenset())

    @property
    def frozen(self) -> bool:
        """Whether identity may still change."""
        return self.status != EvaluationRunStatus.DRAFT.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "engine_version": self.engine_version,
            "evaluation_id": self.evaluation_id,
            "status": self.status,
            "mode": self.mode,
            "suite_id": self.suite.suite_id,
            "suite_version": self.suite.suite_version,
            "suite_digest": self.suite_digest,
            "policy_digest": self.policy_digest,
            "seeds": list(self.seeds),
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "lineage": self.lineage.to_dict(),
            "experiment_hypothesis": self.experiment_hypothesis,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "cancelled_at": self.cancelled_at,
            "failed_at": self.failed_at,
            "error": self.error,
            "cost": {
                "wall_seconds": self.wall_seconds,
                "gpu_seconds": self.gpu_seconds,
                "generated_audio_seconds": self.generated_audio_seconds,
            },
        }


def execute_side(
    run: EvaluationRun,
    model: ModelRef,
    backend: GenerationBackend,
    output_dir: Path,
    *,
    completed: set[tuple[str, int]] | None = None,
    cancel_check: Any = None,
) -> SideResults:
    """Generate and measure one side of the comparison.

    ``completed`` names case/seed pairs a previous attempt finished, so
    a resumed run does not regenerate them. ``cancel_check`` is polled
    between cases so cancellation stops promptly and keeps what it has.
    """
    side = SideResults(model=model)
    for case in sorted(run.suite.cases, key=lambda c: c.case_id):
        for seed in run.seeds:
            if cancel_check is not None and cancel_check():
                return side
            if completed and (case.case_id, seed) in completed:
                continue

            outcome = backend.generate(case, seed, model, output_dir)
            side.outcomes.append(outcome)

            if backend.produces_audio:
                side.metrics.extend(analyse_audio_outcome(outcome, case, run.suite))
            else:
                side.metrics.extend(synthetic_results(outcome, case))

            # After analysis, so the cost counts audio that exists
            # rather than audio that was requested.
            if outcome.succeeded and outcome.duration_seconds:
                run.generated_audio_seconds += outcome.duration_seconds

            side.samples.append(
                SampleProvenance(
                    evaluation_id=run.evaluation_id,
                    case_id=case.case_id,
                    seed=seed,
                    model_id=model.model_id,
                    checkpoint_id=model.checkpoint_id,
                    mode=run.mode,
                    generation_spec_digest=digest_of(case.spec.to_dict()),
                    raw_sha256=_sha256_of(outcome.audio_path) if outcome.audio_path else None,
                    artifact_ref=str(outcome.audio_path) if outcome.audio_path else None,
                    duration_seconds=outcome.duration_seconds,
                    synthetic=outcome.synthetic,
                )
            )
    return side


def _sha256_of(path: Path) -> str | None:
    import hashlib

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def coverage_of(suite: EvaluationSuite, side: SideResults) -> tuple[int, int, int, int]:
    """(cases expected, cases with results, metrics expected, metrics measured).

    Coverage answers "did this evaluation gather evidence?", which is a
    different question from "did the candidate do well?". The
    distinction decides between BLOCKED and REJECTED, so it has to be
    drawn precisely.

    A case the model *tried* and failed is covered. The failure is the
    measurement — it is what `generation_failure_rate` counts, and the
    reliability gate is waiting for exactly that number. Counting only
    successes would mean a model that failed most of the suite came out
    BLOCKED for insufficient evidence, when the evidence was conclusive
    and damning. A case with no recorded outcome at all — a run
    cancelled partway, a crash — is genuinely uncovered, and that is
    what BLOCKED is for.

    The same reasoning applies to metrics: one whose measurement failed
    was attempted and produced a definite negative result. One recorded
    NOT_MEASURABLE produced no information, and is not counted.
    """
    expected_cases = len(suite.cases)
    cases_with_results = len({outcome.case_id for outcome in side.outcomes})
    expected_metrics = suite.required_metrics()
    aggregates = side.aggregates(suite)
    measured = sum(
        1
        for name in expected_metrics
        if name in aggregates
        and (
            aggregates[name].status == MetricStatus.MEASURED.value
            or aggregates[name].count_failed > 0
        )
    )
    return expected_cases, cases_with_results, len(expected_metrics), measured
