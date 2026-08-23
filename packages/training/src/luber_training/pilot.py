"""A training run small enough to be safe, long enough to show a signal.

Phase 33 proved the trainer starts. Phase 34 measured what it costs.
Neither touched the question this module exists for: **does real data
through the real stack produce a coherent training signal at all?**

That question needs tens of optimizer steps, not one — and tens of steps
on a 2.4B model is the first thing in this repository that could
plausibly run away. So the shape of this module is mostly bounds.

**The step count is computed, not hoped for.** The installed trainer has
no `--max-steps`; length is epochs, and the number of optimizer steps
follows from the dataset size, the micro batch, the accumulation factor
and the epoch count. :class:`PilotStepBudget` reproduces the trainer's
own arithmetic — `max(1, ceil(len(loader) / accum)) * epochs`, read from
`trainer_fixed` at the pinned commit — and a pilot refuses to launch
before a process exists if that number exceeds the ceiling.

**The ceiling is a module constant.** No CLI flag raises it. A caller
asking for more gets :class:`PilotBudgetError`.

**A short run is not allowed to claim much.** The classification here
tops out at `VALID_SIGNAL`: loss stayed finite, gradients were finite
and non-zero, and the adapter's parameters actually moved. There is no
`CONVERGED` and no `GOOD_MODEL`, because forty steps cannot support
either, and a vocabulary that could express them would eventually be
used to.

Nothing here trains anything or imports torch. It is the contract the
runner fills in and the console reads.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

PILOT_SCHEMA_VERSION = "luber-training-pilot/1"

# ── the ceilings ─────────────────────────────────────────────────────

#: The most optimizer steps a pilot may take, across every segment.
#:
#: Forty-eight. The number is chosen from what the pilot has to show
#: rather than from what the hardware could stand: a loss series needs
#: enough points that "it did not diverge" is more than an anecdote, and
#: a bounded resume has to fit inside the same total with steps left on
#: both sides of it. Phase 34 measured a step at this configuration in
#: single-digit seconds, so forty-eight is minutes of compute — small
#: enough that an operator can watch it, far too small to train anything.
#:
#: Raising it is a code change with a diff and a review. There is no
#: flag.
PILOT_MAX_OPTIMIZER_STEPS = 48

#: The most a single segment may take.
#:
#: Half the total, exactly, so that a pilot split across a resume fits
#: inside one ceiling with nothing left over and nothing double-counted.
#: The runner derives its own segment ceiling from whether a resume was
#: asked for, so these two constants cannot drift into disagreeing.
PILOT_MAX_SEGMENT_STEPS = PILOT_MAX_OPTIMIZER_STEPS // 2

#: Wall clock for one segment. A pilot that has to be killed produces a
#: `TIMEOUT`, never a completion.
PILOT_MAX_WALL_CLOCK_SECONDS = 3600.0

#: The ceiling on the wall clock itself.
PILOT_ABSOLUTE_WALL_CLOCK_SECONDS = 5400.0

#: The smallest dataset a signal pilot is worth running on.
#:
#: Three. Below that the loss series is dominated by which single track
#: was sampled, and a "signal" would be a statement about one recording.
#: This is a floor on *evidence quality*, not a claim that three tracks
#: teach a model anything.
PILOT_MIN_TRACKS = 3


class PilotBudgetError(ValueError):
    """Raised when a pilot would exceed a bound. Never caught internally."""


# ── how many steps will actually happen ──────────────────────────────


@dataclass(frozen=True)
class PilotStepBudget:
    """The exact number of optimizer steps a configuration will take.

    Reproduces the installed trainer's own arithmetic rather than
    approximating it. From `trainer_fixed._train_fabric` at the pinned
    commit:

        steps_per_epoch = max(1, ceil(len(train_loader) / accum))
        total_steps     = steps_per_epoch * max_epochs

    and `PreprocessedDataModule` builds its loader with
    ``drop_last=False``, so ``len(train_loader) == ceil(samples / batch)``.

    The `max(1, …)` matters: a dataset smaller than one accumulation
    window still produces one step per epoch, because the loop flushes
    whatever it accumulated when the epoch ends. A budget that divided
    and floored would under-count exactly the small datasets a pilot
    uses.
    """

    samples: int
    micro_batch_size: int
    gradient_accumulation: int
    epochs: int
    #: Distributed world size. Anything above one changes what the
    #: sampler hands each rank, and this budget does not model it.
    world_size: int = 1

    def __post_init__(self) -> None:
        for name, value in (
            ("samples", self.samples),
            ("micro_batch_size", self.micro_batch_size),
            ("gradient_accumulation", self.gradient_accumulation),
            ("epochs", self.epochs),
        ):
            if value < 1:
                raise PilotBudgetError(f"{name} must be at least 1, got {value}")
        if self.world_size != 1:
            raise PilotBudgetError(
                f"a pilot budgets a single-process run; world_size={self.world_size} changes "
                "what the sampler hands each rank and this budget does not model it. Run the "
                "pilot on one device"
            )

    @property
    def micro_batches_per_epoch(self) -> int:
        """`len(train_loader)`, given ``drop_last=False``."""
        return math.ceil(self.samples / self.micro_batch_size)

    @property
    def steps_per_epoch(self) -> int:
        return max(1, math.ceil(self.micro_batches_per_epoch / self.gradient_accumulation))

    @property
    def expected_steps(self) -> int:
        return self.steps_per_epoch * self.epochs

    @property
    def effective_batch_size(self) -> int:
        """What the optimizer sees per step, where a window is full.

        The last window of an epoch may be short — the loop flushes a
        partial accumulation — so this is the nominal figure and not a
        promise about every step.
        """
        return self.micro_batch_size * self.gradient_accumulation

    def validate(self, *, ceiling: int = PILOT_MAX_OPTIMIZER_STEPS) -> None:
        """Refuse a plan that would exceed the ceiling. Before launch."""
        if self.expected_steps > ceiling:
            raise PilotBudgetError(
                f"{self.samples} sample(s) at micro batch {self.micro_batch_size}, "
                f"accumulation {self.gradient_accumulation}, for {self.epochs} epoch(s) is "
                f"{self.expected_steps} optimizer step(s); a pilot may take {ceiling}. "
                "Reduce the epochs or the dataset — the ceiling is not a parameter"
            )

    @classmethod
    def for_ceiling(
        cls,
        *,
        samples: int,
        micro_batch_size: int,
        gradient_accumulation: int,
        ceiling: int = PILOT_MAX_SEGMENT_STEPS,
    ) -> PilotStepBudget:
        """The most epochs that fit under *ceiling*, and the budget for them.

        Derived rather than asked for, so a caller cannot pick an epoch
        count that happens to overshoot. Where even one epoch overshoots
        — a large dataset — this raises, because the answer is a smaller
        dataset and not a quieter bound.
        """
        probe = cls(
            samples=samples,
            micro_batch_size=micro_batch_size,
            gradient_accumulation=gradient_accumulation,
            epochs=1,
        )
        if probe.steps_per_epoch > ceiling:
            raise PilotBudgetError(
                f"one epoch over {samples} sample(s) is already {probe.steps_per_epoch} "
                f"optimizer step(s), past the {ceiling}-step bound. A pilot needs a smaller "
                "dataset, not a larger bound"
            )
        return cls(
            samples=samples,
            micro_batch_size=micro_batch_size,
            gradient_accumulation=gradient_accumulation,
            epochs=max(1, ceiling // probe.steps_per_epoch),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation": self.gradient_accumulation,
            "effective_batch_size": self.effective_batch_size,
            "epochs": self.epochs,
            "world_size": self.world_size,
            "micro_batches_per_epoch": self.micro_batches_per_epoch,
            "steps_per_epoch": self.steps_per_epoch,
            "expected_steps": self.expected_steps,
            "ceiling": PILOT_MAX_OPTIMIZER_STEPS,
            "derivation": (
                "max(1, ceil(ceil(samples / micro_batch) / accumulation)) * epochs — the "
                "trainer's own arithmetic at the pinned commit, with drop_last=False"
            ),
        }


# ── what a pilot is ──────────────────────────────────────────────────


@dataclass(frozen=True)
class PilotIdentity:
    """The configuration and data one pilot's evidence belongs to.

    Everything that would make a different pilot a different experiment.
    Nothing volatile: no timestamp, no hostname, no pid, no free memory,
    so two runs of the same pilot produce the same identity.
    """

    plan_digest: str
    dataset_manifest_digest: str
    dataset_id: str
    base_model_id: str
    base_model_upstream_commit: str
    ace_step_commit: str
    device: str
    precision: str
    optimizer: str
    lora_rank: int
    lora_alpha: int
    micro_batch_size: int
    gradient_accumulation: int
    epochs: int
    expected_steps: int
    latent_length: int
    encoder_length: int
    seed: int

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_digest": self.plan_digest,
            "dataset_manifest_digest": self.dataset_manifest_digest,
            "dataset_id": self.dataset_id,
            "base_model_id": self.base_model_id,
            "base_model_upstream_commit": self.base_model_upstream_commit,
            "ace_step_commit": self.ace_step_commit,
            "device": self.device,
            "precision": self.precision,
            "optimizer": self.optimizer,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation": self.gradient_accumulation,
            "epochs": self.epochs,
            "expected_steps": self.expected_steps,
            "latent_length": self.latent_length,
            "encoder_length": self.encoder_length,
            "seed": self.seed,
        }

    def pilot_id(self) -> str:
        return (
            f"pilot-{self.device.lower()}-{self.precision}-r{self.lora_rank}-"
            f"s{self.expected_steps}-{self.digest()[:12]}"
        )


# ── outcomes ─────────────────────────────────────────────────────────


class PilotOutcome(StrEnum):
    """How a pilot ended.

    Deliberately not `PASS`. A pilot is a run, and a run either produced
    evidence of a training signal, produced evidence of something wrong,
    or did not get far enough to produce evidence at all.
    """

    COMPLETED_VALID_SIGNAL = "COMPLETED_VALID_SIGNAL"
    COMPLETED_INSUFFICIENT_SIGNAL = "COMPLETED_INSUFFICIENT_SIGNAL"
    BLOCKED = "BLOCKED"
    FAILED_NUMERIC = "FAILED_NUMERIC"
    FAILED_RUNTIME = "FAILED_RUNTIME"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    NOT_RUN = "NOT_RUN"


class TrainingSignal(StrEnum):
    """What the numbers support saying.

    The ceiling is `VALID_SIGNAL`: the arithmetic ran, stayed finite, and
    moved the adapter. `CONVERGED`, `IMPROVED` and `GOOD` are absent
    because tens of steps cannot support them, and a vocabulary that
    could express them would be used to.
    """

    VALID_SIGNAL = "VALID_SIGNAL"
    NUMERICALLY_UNSTABLE = "NUMERICALLY_UNSTABLE"
    NO_UPDATE = "NO_UPDATE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class PilotFailure(StrEnum):
    """Why a pilot could not produce evidence, in a closed vocabulary."""

    NO_RIGHTS_CLEARED_DATA = "NO_RIGHTS_CLEARED_DATA"
    DATASET_INVALID = "DATASET_INVALID"
    MANIFEST_DRIFT = "MANIFEST_DRIFT"
    STEP_BUDGET_EXCEEDED = "STEP_BUDGET_EXCEEDED"
    CAPACITY_NOT_QUALIFIED = "CAPACITY_NOT_QUALIFIED"
    PREFLIGHT_BLOCKED = "PREFLIGHT_BLOCKED"
    PREPROCESSING_FAILED = "PREPROCESSING_FAILED"
    TRAINER_FAILED = "TRAINER_FAILED"
    LOSS_NONFINITE = "LOSS_NONFINITE"
    GRADIENT_NONFINITE = "GRADIENT_NONFINITE"
    NO_PARAMETER_UPDATE = "NO_PARAMETER_UPDATE"
    CHECKPOINT_FAILED = "CHECKPOINT_FAILED"
    RESUME_FAILED = "RESUME_FAILED"
    STALE_CHECKPOINT = "STALE_CHECKPOINT"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"


#: What a pilot's artifacts are, stamped on every checkpoint it writes.
#: Three words rather than one flag, because each of them stops a
#: different mistake.
ARTIFACT_CLASS = ("EXPERIMENTAL", "NON_PRODUCTION", "NEVER_AUTO_PROMOTE")


# ── the loss series ──────────────────────────────────────────────────


@dataclass(frozen=True)
class LossPoint:
    """One optimizer step, as the trainer reported it."""

    step: int
    loss: float
    epoch: int | None = None
    learning_rate: float | None = None
    grad_norm: float | None = None
    elapsed_seconds: float | None = None
    #: Which segment produced it. A resumed pilot has two.
    segment: str = "A"

    @property
    def finite(self) -> bool:
        return math.isfinite(self.loss)

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
            step=int(payload.get("step", 0)),
            loss=float(payload.get("loss", float("nan"))),
            epoch=_optional_int(payload.get("epoch")),
            learning_rate=_optional_float(payload.get("learning_rate")),
            grad_norm=_optional_float(payload.get("grad_norm")),
            elapsed_seconds=_optional_float(payload.get("elapsed_seconds")),
            segment=str(payload.get("segment", "A")),
        )


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _optional_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


@dataclass
class LossSeries:
    """Every step's loss, and descriptive statistics over them.

    Descriptive on purpose. A pilot of tens of steps has a noisy series,
    and requiring a monotonic decrease would fail runs that are working
    and pass runs that are not. So this reports what happened — first,
    last, minimum, maximum, mean, median, how many were finite — and one
    derived slope, labelled as derived.

    The slope is **not** a convergence claim. Over forty steps it is a
    line through noise, and it is reported because its *sign and
    magnitude being absurd* is informative, not because a negative one
    means anything good.
    """

    points: list[LossPoint] = field(default_factory=list)

    @property
    def finite_points(self) -> list[LossPoint]:
        return [point for point in self.points if point.finite]

    @property
    def finite_ratio(self) -> float | None:
        if not self.points:
            return None
        return len(self.finite_points) / len(self.points)

    @property
    def all_finite(self) -> bool:
        return bool(self.points) and len(self.finite_points) == len(self.points)

    def statistics(self) -> dict[str, Any]:
        finite = [point.loss for point in self.finite_points]
        if not finite:
            return {
                "count": len(self.points),
                "finite_count": 0,
                "finite_ratio": self.finite_ratio,
                "first": None,
                "last": None,
                "minimum": None,
                "maximum": None,
                "mean": None,
                "median": None,
                "slope": None,
                "slope_source": "UNKNOWN",
            }
        return {
            "count": len(self.points),
            "finite_count": len(finite),
            "finite_ratio": self.finite_ratio,
            "first": finite[0],
            "last": finite[-1],
            "minimum": min(finite),
            "maximum": max(finite),
            "mean": statistics.fmean(finite),
            "median": statistics.median(finite),
            "slope": self.slope(),
            # Never MEASURED. A line fitted through observations is
            # arithmetic over them, and Phase 34's vocabulary already has
            # a word for that.
            "slope_source": "DERIVED",
            "slope_note": (
                "least squares over the finite losses. Not a convergence claim: over tens "
                "of steps this is a line through noise"
            ),
        }

    def slope(self) -> float | None:
        """Least-squares slope over the finite points, or None.

        Two points are enough to draw a line and not enough to mean
        anything, so fewer than three returns None rather than a number
        somebody might quote.
        """
        finite = self.finite_points
        if len(finite) < 3:
            return None
        xs = [float(point.step) for point in finite]
        ys = [point.loss for point in finite]
        mean_x = statistics.fmean(xs)
        mean_y = statistics.fmean(ys)
        denominator = sum((x - mean_x) ** 2 for x in xs)
        if denominator == 0:
            return None
        return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denominator

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": [point.to_dict() for point in self.points],
            "statistics": self.statistics(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LossSeries:
        return cls(
            points=[
                LossPoint.from_dict(item)
                for item in payload.get("points") or []
                if isinstance(item, dict)
            ]
        )


# ── did anything actually change ─────────────────────────────────────


@dataclass(frozen=True)
class ParameterUpdateEvidence:
    """Whether training moved the adapter, and left the base model alone.

    A fingerprint rather than the weights: a digest and a few summary
    statistics over the trainable tensors, taken before and after. The
    base model is checked by file digest, not by hashing 2.4 billion
    parameters — the question there is whether anything wrote to it, and
    a file digest answers that at a fraction of the cost.
    """

    trainable_before_digest: str | None = None
    trainable_after_digest: str | None = None
    trainable_tensor_count: int | None = None
    trainable_parameter_count: int | None = None
    #: How many trainable tensors changed at all.
    changed_tensor_count: int | None = None
    max_absolute_delta: float | None = None
    mean_absolute_delta: float | None = None
    base_model_digest_before: str | None = None
    base_model_digest_after: str | None = None
    detail: str = ""

    @property
    def parameters_changed(self) -> bool | None:
        """True when at least one trainable tensor moved."""
        if self.trainable_before_digest is None or self.trainable_after_digest is None:
            return None
        if self.changed_tensor_count is not None:
            return self.changed_tensor_count > 0
        return self.trainable_before_digest != self.trainable_after_digest

    @property
    def base_model_preserved(self) -> bool | None:
        if self.base_model_digest_before is None or self.base_model_digest_after is None:
            return None
        return self.base_model_digest_before == self.base_model_digest_after

    def to_dict(self) -> dict[str, Any]:
        return {
            "trainable_before_digest": self.trainable_before_digest,
            "trainable_after_digest": self.trainable_after_digest,
            "trainable_tensor_count": self.trainable_tensor_count,
            "trainable_parameter_count": self.trainable_parameter_count,
            "changed_tensor_count": self.changed_tensor_count,
            "max_absolute_delta": self.max_absolute_delta,
            "mean_absolute_delta": self.mean_absolute_delta,
            "parameters_changed": self.parameters_changed,
            "base_model_digest_before": self.base_model_digest_before,
            "base_model_digest_after": self.base_model_digest_after,
            "base_model_preserved": self.base_model_preserved,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ParameterUpdateEvidence:
        return cls(
            trainable_before_digest=_optional_str(payload.get("trainable_before_digest")),
            trainable_after_digest=_optional_str(payload.get("trainable_after_digest")),
            trainable_tensor_count=_optional_int(payload.get("trainable_tensor_count")),
            trainable_parameter_count=_optional_int(payload.get("trainable_parameter_count")),
            changed_tensor_count=_optional_int(payload.get("changed_tensor_count")),
            max_absolute_delta=_optional_float(payload.get("max_absolute_delta")),
            mean_absolute_delta=_optional_float(payload.get("mean_absolute_delta")),
            base_model_digest_before=_optional_str(payload.get("base_model_digest_before")),
            base_model_digest_after=_optional_str(payload.get("base_model_digest_after")),
            detail=str(payload.get("detail", "")),
        )


def _optional_str(value: Any) -> str | None:
    return str(value) if isinstance(value, str) and value.strip() else None


@dataclass(frozen=True)
class GradientEvidence:
    """Whether gradients were finite and non-zero, in summary only.

    Summary statistics rather than tensors. A pilot's job is to say that
    gradients existed and were sane; persisting them would put hundreds
    of megabytes into a result record for no additional evidence.
    """

    observed_steps: int = 0
    finite_steps: int = 0
    nonzero_steps: int = 0
    max_grad_norm: float | None = None
    min_grad_norm: float | None = None
    mean_grad_norm: float | None = None
    detail: str = ""

    @property
    def all_finite(self) -> bool | None:
        if self.observed_steps == 0:
            return None
        return self.finite_steps == self.observed_steps

    @property
    def any_nonzero(self) -> bool | None:
        if self.observed_steps == 0:
            return None
        return self.nonzero_steps > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_steps": self.observed_steps,
            "finite_steps": self.finite_steps,
            "nonzero_steps": self.nonzero_steps,
            "max_grad_norm": self.max_grad_norm,
            "min_grad_norm": self.min_grad_norm,
            "mean_grad_norm": self.mean_grad_norm,
            "all_finite": self.all_finite,
            "any_nonzero": self.any_nonzero,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GradientEvidence:
        return cls(
            observed_steps=int(payload.get("observed_steps") or 0),
            finite_steps=int(payload.get("finite_steps") or 0),
            nonzero_steps=int(payload.get("nonzero_steps") or 0),
            max_grad_norm=_optional_float(payload.get("max_grad_norm")),
            min_grad_norm=_optional_float(payload.get("min_grad_norm")),
            mean_grad_norm=_optional_float(payload.get("mean_grad_norm")),
            detail=str(payload.get("detail", "")),
        )


# ── classification ───────────────────────────────────────────────────


def classify_signal(
    *,
    loss: LossSeries,
    parameters: ParameterUpdateEvidence,
    gradients: GradientEvidence,
    expected_steps: int,
    completed_steps: int,
    minimum_steps: int = 3,
) -> tuple[str, str]:
    """What the numbers support saying, and why.

    Order matters. A run whose loss went non-finite is unstable whatever
    else happened; a run whose parameters never moved has not trained
    whatever its loss did; and a run too short to have a series has no
    evidence rather than a bad result.
    """
    if not loss.points:
        return (
            TrainingSignal.INSUFFICIENT_EVIDENCE.value,
            "no optimizer step reported a loss",
        )

    if not loss.all_finite:
        bad = len(loss.points) - len(loss.finite_points)
        return (
            TrainingSignal.NUMERICALLY_UNSTABLE.value,
            f"{bad} of {len(loss.points)} step(s) reported a non-finite loss",
        )

    if gradients.all_finite is False:
        return (
            TrainingSignal.NUMERICALLY_UNSTABLE.value,
            f"{gradients.observed_steps - gradients.finite_steps} step(s) had non-finite gradients",
        )

    if parameters.parameters_changed is False:
        return (
            TrainingSignal.NO_UPDATE.value,
            "the loss was finite throughout and no trainable parameter changed: the "
            "arithmetic ran and nothing was learned from it",
        )

    if parameters.base_model_preserved is False:
        return (
            TrainingSignal.NUMERICALLY_UNSTABLE.value,
            "the base model's weights changed, which a LoRA run must never do",
        )

    if completed_steps < minimum_steps:
        return (
            TrainingSignal.INSUFFICIENT_EVIDENCE.value,
            f"{completed_steps} step(s) completed of {expected_steps} expected; fewer than "
            f"{minimum_steps} is an anecdote rather than a series",
        )

    if parameters.parameters_changed is None or gradients.any_nonzero is None:
        return (
            TrainingSignal.INSUFFICIENT_EVIDENCE.value,
            "the loss stayed finite, but nobody established whether the adapter's parameters moved",
        )

    if gradients.any_nonzero is False:
        return (
            TrainingSignal.NO_UPDATE.value,
            "every observed gradient norm was zero, so nothing could have been learned",
        )

    return (
        TrainingSignal.VALID_SIGNAL.value,
        f"{completed_steps} finite optimizer step(s), finite non-zero gradients, and "
        f"{parameters.changed_tensor_count or 'at least one'} trainable tensor(s) changed. "
        "This says the training path works. It says nothing about convergence or quality",
    )


# ── the result ───────────────────────────────────────────────────────


@dataclass
class PilotSegment:
    """One bounded stretch of training within a pilot."""

    name: str
    step_budget: dict[str, Any] = field(default_factory=dict)
    first_step: int | None = None
    last_step: int | None = None
    completed_steps: int = 0
    checkpoint_id: str | None = None
    resumed_from: str | None = None
    exit_code: int | None = None
    wall_seconds: float | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "step_budget": self.step_budget,
            "first_step": self.first_step,
            "last_step": self.last_step,
            "completed_steps": self.completed_steps,
            "checkpoint_id": self.checkpoint_id,
            "resumed_from": self.resumed_from,
            "exit_code": self.exit_code,
            "wall_seconds": self.wall_seconds,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PilotSegment:
        return cls(
            name=str(payload.get("name", "")),
            step_budget=dict(payload.get("step_budget") or {}),
            first_step=_optional_int(payload.get("first_step")),
            last_step=_optional_int(payload.get("last_step")),
            completed_steps=int(payload.get("completed_steps") or 0),
            checkpoint_id=_optional_str(payload.get("checkpoint_id")),
            resumed_from=_optional_str(payload.get("resumed_from")),
            exit_code=_optional_int(payload.get("exit_code")),
            wall_seconds=_optional_float(payload.get("wall_seconds")),
            detail=str(payload.get("detail", "")),
        )


@dataclass
class PilotTrainingResult:
    """One pilot, everything it established, and what it may not claim."""

    pilot_id: str
    identity: PilotIdentity
    outcome: str = PilotOutcome.NOT_RUN.value
    signal: str = TrainingSignal.INSUFFICIENT_EVIDENCE.value
    signal_detail: str = ""
    failure: str | None = None
    failure_detail: str = ""
    expected_steps: int = 0
    completed_steps: int = 0
    loss: LossSeries = field(default_factory=LossSeries)
    parameters: ParameterUpdateEvidence = field(default_factory=ParameterUpdateEvidence)
    gradients: GradientEvidence = field(default_factory=GradientEvidence)
    segments: list[PilotSegment] = field(default_factory=list)
    checkpoint: dict[str, Any] | None = None
    resume: dict[str, Any] | None = None
    capacity_profile_id: str | None = None
    capacity_qualification: str | None = None
    preflight_status: str | None = None
    #: Whether the training material was real, authorised music or a
    #: synthetic fixture. A synthetic pilot validates mechanics and can
    #: never be real-data evidence, so the two are never conflated.
    dataset_kind: str = "UNKNOWN"
    started_at: str = ""
    finished_at: str = ""
    wall_seconds: float | None = None
    artifact_class: tuple[str, ...] = ARTIFACT_CLASS
    schema_version: str = PILOT_SCHEMA_VERSION

    @property
    def completed(self) -> bool:
        return self.outcome in (
            PilotOutcome.COMPLETED_VALID_SIGNAL.value,
            PilotOutcome.COMPLETED_INSUFFICIENT_SIGNAL.value,
        )

    @property
    def within_budget(self) -> bool:
        return self.completed_steps <= PILOT_MAX_OPTIMIZER_STEPS

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pilot_id": self.pilot_id,
            "identity_digest": self.identity.digest(),
            "identity": self.identity.to_dict(),
            "outcome": self.outcome,
            "signal": self.signal,
            "signal_detail": self.signal_detail,
            "failure": self.failure,
            "failure_detail": self.failure_detail,
            "expected_steps": self.expected_steps,
            "completed_steps": self.completed_steps,
            "within_budget": self.within_budget,
            "step_ceiling": PILOT_MAX_OPTIMIZER_STEPS,
            "loss": self.loss.to_dict(),
            "parameters": self.parameters.to_dict(),
            "gradients": self.gradients.to_dict(),
            "segments": [segment.to_dict() for segment in self.segments],
            "checkpoint": self.checkpoint,
            "resume": self.resume,
            "capacity_profile_id": self.capacity_profile_id,
            "capacity_qualification": self.capacity_qualification,
            "preflight_status": self.preflight_status,
            "dataset_kind": self.dataset_kind,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "wall_seconds": self.wall_seconds,
            "artifact_class": list(self.artifact_class),
            "note": (
                "A pilot establishes that the training path produces a coherent signal. It "
                "makes no claim about convergence, music quality or model improvement, and "
                "its checkpoint is EXPERIMENTAL, NON_PRODUCTION and NEVER_AUTO_PROMOTE."
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PilotTrainingResult:
        version = str(payload.get("schema_version", ""))
        if version != PILOT_SCHEMA_VERSION:
            raise PilotFormatError(
                f"pilot schema {version!r} is not {PILOT_SCHEMA_VERSION!r}; a record from a "
                "different build is refused rather than read with its unknown fields ignored"
            )
        identity = payload.get("identity")
        if not isinstance(identity, dict):
            raise PilotFormatError("a pilot result must carry its identity")
        return cls(
            pilot_id=str(payload.get("pilot_id", "")),
            identity=PilotIdentity(**identity),
            outcome=str(payload.get("outcome", PilotOutcome.NOT_RUN.value)),
            signal=str(payload.get("signal", TrainingSignal.INSUFFICIENT_EVIDENCE.value)),
            signal_detail=str(payload.get("signal_detail", "")),
            failure=_optional_str(payload.get("failure")),
            failure_detail=str(payload.get("failure_detail", "")),
            expected_steps=int(payload.get("expected_steps") or 0),
            completed_steps=int(payload.get("completed_steps") or 0),
            loss=LossSeries.from_dict(payload.get("loss") or {}),
            parameters=ParameterUpdateEvidence.from_dict(payload.get("parameters") or {}),
            gradients=GradientEvidence.from_dict(payload.get("gradients") or {}),
            segments=[
                PilotSegment.from_dict(item)
                for item in payload.get("segments") or []
                if isinstance(item, dict)
            ],
            checkpoint=payload.get("checkpoint"),
            resume=payload.get("resume"),
            capacity_profile_id=_optional_str(payload.get("capacity_profile_id")),
            capacity_qualification=_optional_str(payload.get("capacity_qualification")),
            preflight_status=_optional_str(payload.get("preflight_status")),
            dataset_kind=str(payload.get("dataset_kind", "UNKNOWN")),
            started_at=str(payload.get("started_at", "")),
            finished_at=str(payload.get("finished_at", "")),
            wall_seconds=_optional_float(payload.get("wall_seconds")),
        )


class PilotFormatError(ValueError):
    """Raised when a document is not a pilot result this build reads."""


def outcome_for(signal: str, *, completed: bool) -> str:
    """The outcome a signal implies for a run that finished.

    Kept as one function so the console, the CLI and the runner cannot
    map the same signal to different outcomes.
    """
    if not completed:
        return PilotOutcome.FAILED_RUNTIME.value
    if signal == TrainingSignal.VALID_SIGNAL.value:
        return PilotOutcome.COMPLETED_VALID_SIGNAL.value
    if signal in (TrainingSignal.NUMERICALLY_UNSTABLE.value, TrainingSignal.NO_UPDATE.value):
        return PilotOutcome.FAILED_NUMERIC.value
    return PilotOutcome.COMPLETED_INSUFFICIENT_SIGNAL.value


__all__ = [
    "ARTIFACT_CLASS",
    "PILOT_ABSOLUTE_WALL_CLOCK_SECONDS",
    "PILOT_MAX_OPTIMIZER_STEPS",
    "PILOT_MAX_SEGMENT_STEPS",
    "PILOT_MAX_WALL_CLOCK_SECONDS",
    "PILOT_MIN_TRACKS",
    "PILOT_SCHEMA_VERSION",
    "GradientEvidence",
    "LossPoint",
    "LossSeries",
    "ParameterUpdateEvidence",
    "PilotBudgetError",
    "PilotFailure",
    "PilotFormatError",
    "PilotIdentity",
    "PilotOutcome",
    "PilotSegment",
    "PilotStepBudget",
    "PilotTrainingResult",
    "TrainingSignal",
    "classify_signal",
    "outcome_for",
]
