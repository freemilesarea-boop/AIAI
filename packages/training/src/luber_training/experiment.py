"""What a controlled experiment is, and what its numbers may be called.

Phase 35B's pilot asked one question: does the training path work at
all. This asks the next one, and only the next one — does a larger but
still bounded run on real music move the model in a way that shows up on
data it never trained on.

Two verdicts, kept apart on purpose, because conflating them is the
oldest mistake in the subject:

**Training signal.** Finite losses, finite non-zero gradients, adapter
tensors that moved, a base model that did not. This says the machinery
optimised something. Phase 35 already established the vocabulary and it
is reused unchanged.

**Generalization signal.** What the held-out validation loss did while
the training loss fell. This is the only thing in the phase that can
speak about learning rather than fitting, and it is deliberately hard to
make say anything: a validation curve over tens of steps on four tracks
is a small measurement, and the classifier refuses to call a small
measurement a result.

Nothing here can produce a quality verdict. There is no value in any
enum that means "better music", "converged" or "ready", because none of
those can be established by a loss number and the vocabulary should not
let anybody imply otherwise. What the phase can honestly conclude, when
the numbers are good, is that a listening evaluation is now worth doing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

#: Hard ceilings. No CLI flag, config field or request may raise these —
#: they are checked on construction and there is nothing to override.
EXPERIMENT_MAX_OPTIMIZER_STEPS = 240
EXPERIMENT_MAX_EPOCHS = 40
EXPERIMENT_MAX_WALL_CLOCK_SECONDS = 7_200.0
#: Whatever a caller asks for, one segment cannot run longer than this.
EXPERIMENT_ABSOLUTE_WALL_CLOCK_SECONDS = 14_400.0

#: Below this a validation curve is a statement about a couple of
#: recordings rather than about held-out behaviour.
MINIMUM_VALIDATION_TRACKS = 3
#: Fewer points than this and "the curve went down" is one number
#: compared with one other number.
MINIMUM_VALIDATION_POINTS = 4

#: A held-out loss must improve by at least this fraction of its own
#: starting value before the movement is called anything but noise.
#: Chosen, not measured — and the verdict says so wherever it is used.
GENERALIZATION_IMPROVEMENT_THRESHOLD = 0.02
#: Rising by more than this is worth naming, because a validation loss
#: that climbs while training loss falls is the one shape that means
#: something specific.
GENERALIZATION_DEGRADATION_THRESHOLD = 0.05


class ExperimentError(RuntimeError):
    """Raised when an experiment cannot be attempted as asked."""


class TrainingSignal(StrEnum):
    """Whether optimization happened. Never whether it was any good."""

    VALID_SIGNAL = "VALID_SIGNAL"
    NUMERICALLY_UNSTABLE = "NUMERICALLY_UNSTABLE"
    NO_UPDATE = "NO_UPDATE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class GeneralizationSignal(StrEnum):
    """What the held-out loss did, in the weakest words that fit.

    There is no CONVERGED and no IMPROVED_QUALITY. The strongest value
    says a listening evaluation is warranted, which is a statement about
    what to do next rather than about the model.
    """

    #: Held-out loss fell by more than the threshold while training ran.
    HELD_OUT_LOSS_IMPROVED = "HELD_OUT_LOSS_IMPROVED"
    #: Held-out loss rose while training loss fell — the shape worth
    #: naming, and the reason this enum is separate from the one above.
    HELD_OUT_LOSS_DEGRADED = "HELD_OUT_LOSS_DEGRADED"
    #: Movement smaller than the threshold. Not "no learning": no
    #: measurable movement, over a very short run.
    NO_MEASURABLE_CHANGE = "NO_MEASURABLE_CHANGE"
    #: Too few points, too few tracks, or nothing measured at all.
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ExperimentOutcome(StrEnum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    TIMED_OUT = "TIMED_OUT"
    STEP_CEILING_EXCEEDED = "STEP_CEILING_EXCEEDED"


class ExperimentFailure(StrEnum):
    RIGHTS_GATE_FAILED = "RIGHTS_GATE_FAILED"
    SPLIT_LEAKAGE = "SPLIT_LEAKAGE"
    DATASET_UNUSABLE = "DATASET_UNUSABLE"
    CAPACITY_NOT_QUALIFIED = "CAPACITY_NOT_QUALIFIED"
    PREFLIGHT_NOT_READY = "PREFLIGHT_NOT_READY"
    BUDGET_UNCOMPUTABLE = "BUDGET_UNCOMPUTABLE"
    TRAINER_FAILED = "TRAINER_FAILED"
    TIMED_OUT = "TIMED_OUT"
    STEP_CEILING_EXCEEDED = "STEP_CEILING_EXCEEDED"
    CHECKPOINT_MISSING = "CHECKPOINT_MISSING"
    PROVENANCE_INCOMPLETE = "PROVENANCE_INCOMPLETE"
    RESUME_FAILED = "RESUME_FAILED"
    BASE_MODEL_MODIFIED = "BASE_MODEL_MODIFIED"


#: What an experiment checkpoint is, and is not. Never widened by code.
ARTIFACT_CLASS: tuple[str, ...] = (
    "EXPERIMENTAL",
    "NON_PRODUCTION",
    "NEVER_AUTO_PROMOTE",
)


@dataclass(frozen=True)
class StepBudget:
    """How many optimizer steps a configuration will actually take.

    The trainer has no `--max-steps`: length is epochs, and the step
    count follows from the loader arithmetic. So the number is computed
    from that arithmetic and checked against the ceiling *before*
    anything launches, rather than hoped for and discovered afterwards.
    """

    samples: int
    micro_batch_size: int
    gradient_accumulation: int
    epochs: int
    world_size: int = 1

    def __post_init__(self) -> None:
        for name in ("samples", "micro_batch_size", "gradient_accumulation", "epochs"):
            if getattr(self, name) < 1:
                raise ExperimentError(f"{name} must be at least 1")
        if self.epochs > EXPERIMENT_MAX_EPOCHS:
            raise ExperimentError(
                f"{self.epochs} epochs exceeds the experiment ceiling of {EXPERIMENT_MAX_EPOCHS}"
            )
        if self.expected_steps > EXPERIMENT_MAX_OPTIMIZER_STEPS:
            raise ExperimentError(
                f"{self.expected_steps} optimizer steps exceeds the experiment ceiling of "
                f"{EXPERIMENT_MAX_OPTIMIZER_STEPS}"
            )

    @property
    def micro_batches_per_epoch(self) -> int:
        # `drop_last=False`, so a partial final batch still counts.
        return math.ceil(self.samples / self.micro_batch_size)

    @property
    def steps_per_epoch(self) -> int:
        return max(1, math.ceil(self.micro_batches_per_epoch / self.gradient_accumulation))

    @property
    def expected_steps(self) -> int:
        return self.steps_per_epoch * self.epochs

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation * self.world_size

    @classmethod
    def for_ceiling(
        cls,
        *,
        samples: int,
        micro_batch_size: int,
        gradient_accumulation: int,
        ceiling: int,
    ) -> StepBudget:
        """The longest run that still fits under *ceiling*."""
        if ceiling < 1:
            raise ExperimentError("a ceiling below one step leaves nothing to run")
        probe = cls(
            samples=samples,
            micro_batch_size=micro_batch_size,
            gradient_accumulation=gradient_accumulation,
            epochs=1,
        )
        epochs = min(
            EXPERIMENT_MAX_EPOCHS,
            max(1, min(ceiling, EXPERIMENT_MAX_OPTIMIZER_STEPS) // probe.steps_per_epoch),
        )
        return cls(
            samples=samples,
            micro_batch_size=micro_batch_size,
            gradient_accumulation=gradient_accumulation,
            epochs=epochs,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation": self.gradient_accumulation,
            "epochs": self.epochs,
            "world_size": self.world_size,
            "micro_batches_per_epoch": self.micro_batches_per_epoch,
            "steps_per_epoch": self.steps_per_epoch,
            "expected_steps": self.expected_steps,
            "effective_batch_size": self.effective_batch_size,
            "ceiling": EXPERIMENT_MAX_OPTIMIZER_STEPS,
            "derivation": (
                "max(1, ceil(ceil(samples / micro_batch) / accumulation)) * epochs — the "
                "trainer's own arithmetic at the pinned commit, with drop_last=False"
            ),
        }


@dataclass(frozen=True)
class LossPoint:
    """One recorded point on a curve."""

    step: int
    loss: float | None
    epoch: int | None = None
    learning_rate: float | None = None
    grad_norm: float | None = None
    elapsed_seconds: float | None = None
    segment: str = ""

    @property
    def finite(self) -> bool:
        return self.loss is not None and math.isfinite(self.loss)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "loss": self.loss,
            "epoch": self.epoch,
            "learning_rate": self.learning_rate,
            "grad_norm": self.grad_norm,
            "elapsed_seconds": self.elapsed_seconds,
            "segment": self.segment,
            "finite": self.finite,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LossPoint:
        return cls(
            step=int(payload.get("step") or 0),
            loss=payload.get("loss"),
            epoch=payload.get("epoch"),
            learning_rate=payload.get("learning_rate"),
            grad_norm=payload.get("grad_norm"),
            elapsed_seconds=payload.get("elapsed_seconds"),
            segment=str(payload.get("segment") or ""),
        )


@dataclass(frozen=True)
class LossSeries:
    """A curve, described only in ways the data supports."""

    name: str
    points: tuple[LossPoint, ...] = ()

    @property
    def finite_points(self) -> tuple[LossPoint, ...]:
        return tuple(point for point in self.points if point.finite)

    @property
    def finite_ratio(self) -> float | None:
        if not self.points:
            return None
        return len(self.finite_points) / len(self.points)

    def statistics(self) -> dict[str, Any]:
        values = [point.loss for point in self.finite_points if point.loss is not None]
        if not values:
            return {"count": len(self.points), "finite_count": 0}
        ordered = sorted(values)
        middle = len(ordered) // 2
        median = (
            ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
        )
        return {
            "count": len(self.points),
            "finite_count": len(values),
            "finite_ratio": self.finite_ratio,
            "first": values[0],
            "last": values[-1],
            "minimum": ordered[0],
            "maximum": ordered[-1],
            "mean": sum(values) / len(values),
            "median": median,
            "slope": self.slope(),
            "slope_source": "DERIVED",
            "slope_note": (
                "least squares over the finite losses. Not a convergence claim: over a "
                "run this short it is a line through noise"
            ),
        }

    def slope(self) -> float | None:
        """Least squares gradient, or ``None`` below three points."""
        points = [
            (float(point.step), point.loss)
            for point in self.finite_points
            if point.loss is not None
        ]
        if len(points) < 3:
            return None
        n = len(points)
        mean_x = sum(x for x, _ in points) / n
        mean_y = sum(y for _, y in points) / n
        denominator = sum((x - mean_x) ** 2 for x, _ in points)
        if denominator == 0:
            return None
        return sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "statistics": self.statistics(),
            "points": [point.to_dict() for point in self.points],
        }


@dataclass(frozen=True)
class ValidationPoint:
    """One held-out measurement, and how it was taken."""

    epoch: int
    step: int
    loss: float | None
    sample_count: int = 0
    finite_count: int = 0
    minimum: float | None = None
    maximum: float | None = None
    elapsed_seconds: float | None = None
    #: What the device reported holding at the end of the pass. Recorded
    #: because a validation forward at production sequence length costs
    #: memory the training-only capacity profile never measured.
    device_allocated_bytes: int | None = None
    #: The window this measurement covered, in latent frames. ``None``
    #: means the whole track.
    latent_length: int | None = None
    error: str = ""
    segment: str = ""

    @property
    def finite(self) -> bool:
        return self.loss is not None and math.isfinite(self.loss)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "loss": self.loss,
            "sample_count": self.sample_count,
            "finite_count": self.finite_count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "elapsed_seconds": self.elapsed_seconds,
            "device_allocated_bytes": self.device_allocated_bytes,
            "latent_length": self.latent_length,
            "error": self.error,
            "segment": self.segment,
            "finite": self.finite,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ValidationPoint:
        return cls(
            epoch=int(payload.get("epoch") or 0),
            step=int(payload.get("step") or 0),
            loss=payload.get("loss"),
            sample_count=int(payload.get("sample_count") or 0),
            finite_count=int(payload.get("finite_count") or 0),
            minimum=payload.get("minimum"),
            maximum=payload.get("maximum"),
            elapsed_seconds=payload.get("elapsed_seconds"),
            device_allocated_bytes=payload.get("device_allocated_bytes"),
            latent_length=payload.get("latent_length"),
            error=str(payload.get("error") or ""),
            segment=str(payload.get("segment") or ""),
        )


@dataclass(frozen=True)
class ParameterUpdateEvidence:
    """Whether the adapter moved, and whether the base model did not."""

    changed_tensor_count: int | None = None
    comparable_tensor_count: int | None = None
    trainable_parameter_count: int | None = None
    max_absolute_delta: float | None = None
    mean_absolute_delta: float | None = None
    trainable_before_digest: str | None = None
    trainable_after_digest: str | None = None
    base_model_digest_before: str | None = None
    base_model_digest_after: str | None = None
    detail: str = ""

    @property
    def parameters_changed(self) -> bool | None:
        """``None`` when the comparison could not be made.

        Never ``False`` for an unmakeable comparison: an unknown result
        reported as "nothing changed" is how a healthy run gets called
        NO_UPDATE, which Phase 35 learned the hard way.
        """
        if self.changed_tensor_count is None:
            return None
        return self.changed_tensor_count > 0

    @property
    def base_model_preserved(self) -> bool | None:
        if not self.base_model_digest_before or not self.base_model_digest_after:
            return None
        return self.base_model_digest_before == self.base_model_digest_after

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed_tensor_count": self.changed_tensor_count,
            "comparable_tensor_count": self.comparable_tensor_count,
            "trainable_parameter_count": self.trainable_parameter_count,
            "max_absolute_delta": self.max_absolute_delta,
            "mean_absolute_delta": self.mean_absolute_delta,
            "trainable_before_digest": self.trainable_before_digest,
            "trainable_after_digest": self.trainable_after_digest,
            "base_model_digest_before": self.base_model_digest_before,
            "base_model_digest_after": self.base_model_digest_after,
            "parameters_changed": self.parameters_changed,
            "base_model_preserved": self.base_model_preserved,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class GradientEvidence:
    """What the gradients did, as Fabric measured them."""

    observed_steps: int = 0
    finite_steps: int = 0
    nonzero_steps: int = 0
    min_grad_norm: float | None = None
    max_grad_norm: float | None = None
    mean_grad_norm: float | None = None
    detail: str = ""

    @property
    def all_finite(self) -> bool | None:
        if not self.observed_steps:
            return None
        return self.finite_steps == self.observed_steps

    @property
    def any_nonzero(self) -> bool | None:
        if not self.observed_steps:
            return None
        return self.nonzero_steps > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_steps": self.observed_steps,
            "finite_steps": self.finite_steps,
            "nonzero_steps": self.nonzero_steps,
            "min_grad_norm": self.min_grad_norm,
            "max_grad_norm": self.max_grad_norm,
            "mean_grad_norm": self.mean_grad_norm,
            "all_finite": self.all_finite,
            "any_nonzero": self.any_nonzero,
            "detail": self.detail,
        }


def classify_training_signal(
    *,
    loss: LossSeries,
    gradients: GradientEvidence,
    parameters: ParameterUpdateEvidence,
    expected_steps: int,
) -> tuple[str, str]:
    """Whether optimization happened. Never whether it was good."""
    finite = loss.finite_points
    if not finite or gradients.observed_steps == 0:
        return (
            TrainingSignal.INSUFFICIENT_EVIDENCE.value,
            "no finite loss or no gradient was observed, so nothing can be concluded",
        )
    if len(finite) < expected_steps:
        missing = expected_steps - len(finite)
        if loss.finite_ratio is not None and loss.finite_ratio < 1.0:
            return (
                TrainingSignal.NUMERICALLY_UNSTABLE.value,
                f"{missing} of {expected_steps} step(s) produced a non-finite loss",
            )
    if gradients.all_finite is False:
        return (
            TrainingSignal.NUMERICALLY_UNSTABLE.value,
            "at least one gradient norm was not finite",
        )
    if gradients.any_nonzero is False:
        return (
            TrainingSignal.NO_UPDATE.value,
            "every gradient norm was zero, so no step could have changed anything",
        )
    if parameters.parameters_changed is None:
        return (
            TrainingSignal.INSUFFICIENT_EVIDENCE.value,
            f"whether the adapter moved could not be established: {parameters.detail}",
        )
    if parameters.parameters_changed is False:
        return (
            TrainingSignal.NO_UPDATE.value,
            "no trainable tensor changed across the run",
        )
    if parameters.base_model_preserved is False:
        return (
            TrainingSignal.INSUFFICIENT_EVIDENCE.value,
            "the base model weights changed, which a LoRA run must never do",
        )
    return (
        TrainingSignal.VALID_SIGNAL.value,
        (
            f"{len(finite)} finite optimizer step(s), finite non-zero gradients, and "
            f"{parameters.changed_tensor_count} trainable tensor(s) changed. This says the "
            "training path works. It says nothing about convergence or quality"
        ),
    )


def classify_generalization(
    *,
    validation: tuple[ValidationPoint, ...],
    training: LossSeries,
    validation_track_count: int,
) -> tuple[str, str]:
    """What the held-out loss did, in the weakest words that fit.

    Deliberately hard to satisfy. Held-out loss over a few tracks and a
    few dozen steps is a small measurement, and the point of a separate
    verdict is to stop a small measurement being read as a result.
    """
    finite = [point for point in validation if point.finite and point.loss is not None]
    if validation_track_count < MINIMUM_VALIDATION_TRACKS:
        return (
            GeneralizationSignal.INSUFFICIENT_EVIDENCE.value,
            (
                f"{validation_track_count} validation track(s); below "
                f"{MINIMUM_VALIDATION_TRACKS} a held-out loss describes a couple of "
                "recordings rather than held-out behaviour"
            ),
        )
    if len(finite) < MINIMUM_VALIDATION_POINTS:
        return (
            GeneralizationSignal.INSUFFICIENT_EVIDENCE.value,
            (
                f"{len(finite)} finite validation point(s); below "
                f"{MINIMUM_VALIDATION_POINTS} a curve is one number next to another"
            ),
        )

    first = finite[0].loss
    last = finite[-1].loss
    assert first is not None and last is not None
    if first == 0:
        return (
            GeneralizationSignal.INSUFFICIENT_EVIDENCE.value,
            "the first held-out loss was zero, which no relative change can be taken from",
        )

    change = (last - first) / abs(first)
    training_slope = training.slope()
    shared = (
        f"held-out loss moved from {first:.4f} to {last:.4f} "
        f"({change * 100:+.2f}%) over {len(finite)} measurement(s) on "
        f"{validation_track_count} track(s)"
    )

    if change <= -GENERALIZATION_IMPROVEMENT_THRESHOLD:
        return (
            GeneralizationSignal.HELD_OUT_LOSS_IMPROVED.value,
            (
                f"{shared}. That is movement on data no optimizer step touched, which is "
                "the only thing here that speaks to learning rather than fitting. It is "
                "not a quality claim and not a convergence claim: a listening evaluation "
                "is what would decide either"
            ),
        )
    if change >= GENERALIZATION_DEGRADATION_THRESHOLD:
        detail = f"{shared}. Held-out loss rose"
        if training_slope is not None and training_slope < 0:
            detail += (
                " while the training loss fell, the shape that means the run fitted the "
                "training split rather than the material"
            )
        return (GeneralizationSignal.HELD_OUT_LOSS_DEGRADED.value, detail + ".")
    return (
        GeneralizationSignal.NO_MEASURABLE_CHANGE.value,
        (
            f"{shared}, inside the ±{GENERALIZATION_IMPROVEMENT_THRESHOLD:.0%} band this "
            "phase treats as noise. Not evidence of no learning — evidence of no "
            "measurable movement over a very short run"
        ),
    )


@dataclass
class ExperimentIdentity:
    """Everything that makes two experiments the same experiment."""

    plan_digest: str
    dataset_id: str
    train_split_digest: str
    validation_split_digest: str
    evaluation_split_digest: str
    base_model_id: str
    device: str
    precision: str
    optimizer: str
    lora_rank: int
    lora_alpha: int
    micro_batch_size: int
    gradient_accumulation: int
    epochs: int
    expected_steps: int
    learning_rate: float
    seed: int
    ace_step_commit: str = ""
    base_model_upstream_commit: str = ""

    def digest(self) -> str:
        import hashlib
        import json

        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def experiment_id(self) -> str:
        return (
            f"exp-{self.device.lower()}-{self.precision}-r{self.lora_rank}-"
            f"s{self.expected_steps}-{self.digest()[:12]}"
        )

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


@dataclass
class SegmentRecord:
    """One bounded stretch of training."""

    name: str
    completed_steps: int = 0
    first_step: int | None = None
    last_step: int | None = None
    exit_code: int | None = None
    wall_seconds: float = 0.0
    checkpoint_id: str | None = None
    resumed_from: str | None = None
    validation_count: int = 0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "completed_steps": self.completed_steps,
            "first_step": self.first_step,
            "last_step": self.last_step,
            "exit_code": self.exit_code,
            "wall_seconds": self.wall_seconds,
            "checkpoint_id": self.checkpoint_id,
            "resumed_from": self.resumed_from,
            "validation_count": self.validation_count,
            "detail": self.detail,
        }


@dataclass
class ExperimentResult:
    """What one controlled experiment produced, and what it may claim."""

    experiment_id: str
    identity: ExperimentIdentity
    outcome: str
    training_signal: str
    training_signal_detail: str
    generalization_signal: str
    generalization_signal_detail: str
    expected_steps: int
    completed_steps: int = 0
    step_ceiling: int = EXPERIMENT_MAX_OPTIMIZER_STEPS
    dataset_kind: str = ""
    failure: str | None = None
    failure_detail: str = ""
    train_loss: LossSeries = field(default_factory=lambda: LossSeries(name="train"))
    validation_loss: tuple[ValidationPoint, ...] = ()
    gradients: GradientEvidence = field(default_factory=GradientEvidence)
    parameters: ParameterUpdateEvidence = field(default_factory=ParameterUpdateEvidence)
    segments: tuple[SegmentRecord, ...] = ()
    checkpoint: dict[str, Any] = field(default_factory=dict)
    resume: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    splits: dict[str, Any] = field(default_factory=dict)
    capacity_qualification: str | None = None
    capacity_profile_id: str | None = None
    preflight_status: str | None = None
    wall_seconds: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    artifact_class: tuple[str, ...] = ARTIFACT_CLASS
    schema_version: str = "luber-experiment/1"

    @property
    def within_budget(self) -> bool:
        return self.completed_steps <= self.step_ceiling

    @property
    def listening_evaluation_required(self) -> bool:
        """Whether a human still has to listen before anything is claimed.

        Always true when the run produced a model. There is no number in
        this phase that can replace listening, and the field exists so
        the answer is written down rather than assumed.
        """
        return self.outcome == ExperimentOutcome.COMPLETED.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "identity": self.identity.to_dict(),
            "identity_digest": self.identity.digest(),
            "outcome": self.outcome,
            "training_signal": self.training_signal,
            "training_signal_detail": self.training_signal_detail,
            "generalization_signal": self.generalization_signal,
            "generalization_signal_detail": self.generalization_signal_detail,
            "listening_evaluation_required": self.listening_evaluation_required,
            "expected_steps": self.expected_steps,
            "completed_steps": self.completed_steps,
            "step_ceiling": self.step_ceiling,
            "within_budget": self.within_budget,
            "dataset_kind": self.dataset_kind,
            "failure": self.failure,
            "failure_detail": self.failure_detail,
            "train_loss": self.train_loss.to_dict(),
            "validation_loss": [point.to_dict() for point in self.validation_loss],
            "gradients": self.gradients.to_dict(),
            "parameters": self.parameters.to_dict(),
            "segments": [segment.to_dict() for segment in self.segments],
            "checkpoint": self.checkpoint,
            "resume": self.resume,
            "provenance": self.provenance,
            "splits": self.splits,
            "capacity_qualification": self.capacity_qualification,
            "capacity_profile_id": self.capacity_profile_id,
            "preflight_status": self.preflight_status,
            "wall_seconds": self.wall_seconds,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "artifact_class": list(self.artifact_class),
            "note": (
                "A controlled experiment establishes whether a training path optimises and "
                "whether held-out loss moved. It makes no claim about music quality, "
                "convergence or production readiness, and its checkpoint is EXPERIMENTAL, "
                "NON_PRODUCTION and NEVER_AUTO_PROMOTE."
            ),
        }


__all__ = [
    "ARTIFACT_CLASS",
    "EXPERIMENT_ABSOLUTE_WALL_CLOCK_SECONDS",
    "EXPERIMENT_MAX_EPOCHS",
    "EXPERIMENT_MAX_OPTIMIZER_STEPS",
    "EXPERIMENT_MAX_WALL_CLOCK_SECONDS",
    "GENERALIZATION_DEGRADATION_THRESHOLD",
    "GENERALIZATION_IMPROVEMENT_THRESHOLD",
    "MINIMUM_VALIDATION_POINTS",
    "MINIMUM_VALIDATION_TRACKS",
    "ExperimentError",
    "ExperimentFailure",
    "ExperimentIdentity",
    "ExperimentOutcome",
    "ExperimentResult",
    "GeneralizationSignal",
    "GradientEvidence",
    "LossPoint",
    "LossSeries",
    "ParameterUpdateEvidence",
    "SegmentRecord",
    "StepBudget",
    "TrainingSignal",
    "ValidationPoint",
    "classify_generalization",
    "classify_training_signal",
]
