"""Running one controlled experiment, and refusing to run a bad one.

The order of the gates is the point. Rights before splits, splits before
capacity, capacity before preflight, preflight before anything starts —
so the reason a run did not happen is always the first thing that was
wrong with it, and never "the trainer crashed" when the real answer was
that the evaluation set was in the training data.

Two bounded segments with a checkpoint between them, as Phase 35 did: a
run that never resumed has not shown that a longer one could survive an
interruption, and Phase 36 additionally needs the resumed checkpoint's
provenance to verify, which only a real save-stop-reload can establish.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from luber_training import _experiment_probe
from luber_training.canary import CheckpointIntegrity, inspect_checkpoint, latest_checkpoint
from luber_training.capacity_policy import CapacityDecision
from luber_training.checkpoint_provenance import (
    CHECKPOINT_PROVENANCE_NAME,
    CheckpointProvenance,
    ProvenanceError,
    verify_checkpoint_provenance,
)
from luber_training.experiment import (
    ARTIFACT_CLASS,
    EXPERIMENT_ABSOLUTE_WALL_CLOCK_SECONDS,
    EXPERIMENT_MAX_OPTIMIZER_STEPS,
    EXPERIMENT_MAX_WALL_CLOCK_SECONDS,
    ExperimentFailure,
    ExperimentIdentity,
    ExperimentOutcome,
    ExperimentResult,
    GeneralizationSignal,
    GradientEvidence,
    LossPoint,
    LossSeries,
    ParameterUpdateEvidence,
    SegmentRecord,
    StepBudget,
    TrainingSignal,
    ValidationPoint,
    classify_generalization,
    classify_training_signal,
)
from luber_training.gates import GateReport, split_leakage_gate
from luber_training.pilot_runner import DatasetKind, base_model_digest
from luber_training.plan import TrainingPlan
from luber_training.tensors import TensorReport
from luber_training.trainer_adapter import compile_command

#: Where an experiment keeps its workspace, beneath the trainer root.
#: ACE-Step validates `--dataset-dir` against its own working directory
#: and refuses anything outside it — after the model has loaded.
EXPERIMENT_SUBDIR = "experiment"

#: Default validation window: 2048 latent frames, about 82 seconds at
#: the measured 25 frames a second.
VALIDATION_LATENT_LENGTH = 2048

EXPERIMENT_JSON = "experiment.json"
EXPERIMENT_MARKDOWN = "experiment.md"


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class ExperimentRequest:
    """Everything one controlled experiment needs, and nothing that widens it."""

    plan: TrainingPlan
    train_dir: Path
    validation_dir: Path
    trainer_root: Path
    python_executable: Path
    model_dir: Path
    workspace: Path
    splits: dict[str, Any]
    code_commit: str
    dataset_id: str = "experiment"
    model_variant: str = "turbo"
    seed: int = 42
    gate_report: GateReport | None = None
    capacity: CapacityDecision | None = None
    preflight_status: str | None = None
    tensor_report: TensorReport | None = None
    validation_report: TensorReport | None = None
    #: How much of each held-out track the validation pass covers, in
    #: latent frames. A bounded window, because a full-length validation
    #: forward beside a training step exhausted this machine's unified
    #: memory; ``None`` measures the whole track where the device can
    #: afford it. Whatever it is, it is recorded on every point.
    validation_latent_length: int | None = VALIDATION_LATENT_LENGTH
    #: Total optimizer steps across both segments. Clamped by the module
    #: ceiling; there is no flag that raises it.
    step_budget: int = EXPERIMENT_MAX_OPTIMIZER_STEPS
    segment_timeout_seconds: float = EXPERIMENT_MAX_WALL_CLOCK_SECONDS
    #: Run the second, resumed segment. On by default.
    measure_resume: bool = True
    #: The visiting order weighted exposure produced, as
    #: :meth:`ExposurePlan.to_dict` writes it. ``None`` leaves the
    #: trainer's own shuffling alone, which is what Phase 37 and 38 did.
    #: When it *is* supplied the probe refuses to train unless it can
    #: install it — a weighting that silently fails to apply is the exact
    #: defect this exists to remove.
    exposure_plan: dict[str, Any] | None = None
    #: Sample names in the visiting order. Kept beside the plan because
    #: the plan's own ``repeats`` map is a summary and the order is what
    #: the sampler actually needs.
    exposure_order: tuple[str, ...] | None = None
    #: Path to the adapter the resume must reproduce. When set, the probe
    #: compares the weights actually loaded into the model against this
    #: file and refuses to train on a mismatch. Without it a resume that
    #: silently loaded nothing is indistinguishable from one that worked.
    verify_resume_against: Path | None = None
    #: How often the trainer writes a checkpoint, in epochs. ``None``
    #: keeps the historical behaviour of writing one at the end, which is
    #: fine for a short run and useless for inspecting progression across
    #: a long one.
    checkpoint_every_epochs: int | None = None
    #: Adapter directory the first segment starts from. Phase 38 and
    #: earlier always began at the base model; a continuation begins at a
    #: previous phase's weights. Pair it with ``verify_resume_against``:
    #: the trainer's resume is not strict, so "it started from Phase 38"
    #: is a claim that has to be checked rather than configured.
    resume_adapter: Path | None = None


def split_digests(splits: dict[str, Any]) -> tuple[str, str, str]:
    """The three digests, read from the split manifest as written."""
    return (
        str((splits.get("train") or {}).get("digest", "")),
        str((splits.get("validation") or {}).get("digest", "")),
        str((splits.get("evaluation") or {}).get("digest", "")),
    )


def dataset_digest(dataset_dir: Path) -> str:
    """A content digest over preprocessed tensors, by name and bytes."""
    digest = hashlib.sha256()
    for path in sorted(Path(dataset_dir).glob("*.pt")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def compose_provenance(
    request: ExperimentRequest, *, epoch: int = 0, step: int = 0
) -> CheckpointProvenance:
    """The record the probe will write beside every checkpoint.

    Composed and validated here, where the schema lives, and passed to
    the probe whole. The probe fills in the epoch, the step and the
    exact directory; it does not decide what a provenance record
    contains.

    ``checkpoint_path`` starts as the output directory the checkpoints
    will be written under rather than empty. It has to be *something*
    truthful, because the template is validated before launch and a
    required field left blank would fail that check — which is the
    check working, not a reason to relax it.
    """
    train_digest, validation_digest, evaluation_digest = split_digests(request.splits)
    plan = request.plan
    config = plan.config
    return CheckpointProvenance(
        experiment_id=plan.experiment_id,
        run_id=plan.run_id,
        checkpoint_path=str(request.workspace / "output"),
        epoch=epoch,
        step=step,
        base_model_id=plan.base_model_id,
        base_model_upstream_commit=plan.base_model_upstream_commit,
        base_model_digest=base_model_digest(request.model_dir, request.model_variant),
        dataset_id=plan.dataset_ref.dataset_id or request.dataset_id,
        dataset_lock_sha256=plan.dataset_ref.dataset_lock_sha256,
        curation_id=plan.dataset_ref.curation_id,
        curation_lock_sha256=plan.dataset_ref.curation_lock_sha256,
        train_split_digest=train_digest,
        validation_split_digest=validation_digest,
        evaluation_split_digest=evaluation_digest,
        config_digest=config.digest(),
        lora_rank=config.rank,
        precision=config.precision,
        device=plan.requirements.execution_device or "CPU",
        optimizer=config.optimizer_type,
        learning_rate=config.learning_rate,
        seed=request.seed,
        code_commit=request.code_commit,
        ace_step_commit=config.ace_step_commit,
        plan_id=plan.plan_id,
        plan_digest=plan.digest(),
        dataset_kind=DatasetKind.REAL_OPERATOR_AUTHORIZED,
        artifact_class=ARTIFACT_CLASS,
        notes=(
            "Produced by a bounded Phase 36 controlled experiment. Not a production "
            "model and never automatically promoted."
        ),
    )


def identity_for(request: ExperimentRequest, budget: StepBudget) -> ExperimentIdentity:
    train_digest, validation_digest, evaluation_digest = split_digests(request.splits)
    config = request.plan.config
    return ExperimentIdentity(
        plan_digest=request.plan.digest(),
        dataset_id=request.dataset_id,
        train_split_digest=train_digest,
        validation_split_digest=validation_digest,
        evaluation_split_digest=evaluation_digest,
        base_model_id=request.plan.base_model_id,
        device=request.plan.requirements.execution_device or "CPU",
        precision=config.precision,
        optimizer=config.optimizer_type,
        lora_rank=config.rank,
        lora_alpha=config.alpha,
        micro_batch_size=config.batch_size,
        gradient_accumulation=config.gradient_accumulation,
        epochs=budget.epochs,
        expected_steps=budget.expected_steps,
        learning_rate=config.learning_rate,
        seed=request.seed,
        ace_step_commit=config.ace_step_commit,
        base_model_upstream_commit=request.plan.base_model_upstream_commit,
    )


@dataclass
class SegmentOutcome:
    """What one bounded stretch produced, before it is interpreted."""

    name: str
    document: dict[str, Any] | None
    timed_out: bool
    exit_code: int | None
    wall_seconds: float
    log_tail: str = ""
    checkpoint: Path | None = None

    @property
    def points(self) -> list[LossPoint]:
        if not self.document:
            return []
        return [
            LossPoint.from_dict(item)
            for item in self.document.get("loss_points") or []
            if isinstance(item, dict)
        ]

    @property
    def validation(self) -> list[ValidationPoint]:
        if not self.document:
            return []
        return [
            ValidationPoint.from_dict(item)
            for item in self.document.get("validation_points") or []
            if isinstance(item, dict)
        ]


def _tail(path: Path, lines: int = 40) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(line for line in content[-lines:] if line.strip())


def _read_document(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict):
        return None
    version = str(document.get("protocol_version", ""))
    if version != _experiment_probe.EXPERIMENT_PROBE_PROTOCOL_VERSION:
        return {
            "outcome": "FAILED",
            "failure_reason": (
                f"the probe reported protocol {version!r}, and this build reads "
                f"{_experiment_probe.EXPERIMENT_PROBE_PROTOCOL_VERSION!r}"
            ),
            "loss_points": [],
            "validation_points": [],
        }
    return document


def _terminate(process: subprocess.Popen[bytes]) -> int | None:
    try:
        os.killpg(os.getpgid(process.pid), 15)
    except OSError:
        process.terminate()
    try:
        return process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), 9)
        except OSError:
            process.kill()
        return process.wait(timeout=30)


def _run_segment(
    request: ExperimentRequest,
    plan: TrainingPlan,
    *,
    name: str,
    step_ceiling: int,
    resume_from: Path | None,
    workspace: Path,
    provenance: CheckpointProvenance,
) -> SegmentOutcome:
    """Launch the trainer through the experiment probe, under a clock."""
    output_dir = workspace / "output"
    command = compile_command(
        plan,
        trainer_root=str(request.trainer_root),
        python_executable=str(request.python_executable),
        model_variant=request.model_variant,
        resume_from=None if resume_from is None else str(resume_from),
    )
    result_path = workspace / f"experiment_probe_{name}.json"
    payload = {
        "argv": command.argv[1:],
        "result_path": str(result_path),
        "step_ceiling": step_ceiling,
        "segment": name,
        "validation_dir": str(request.validation_dir),
        "validation_seed": request.seed,
        "validation_latent_length": request.validation_latent_length,
        "provenance": {**provenance.to_dict(), "_filename": CHECKPOINT_PROVENANCE_NAME},
        "exposure_order": list(request.exposure_order or ()) or None,
        "verify_resume_against": (
            str(request.verify_resume_against) if request.verify_resume_against else None
        ),
    }
    log_path = workspace / f"experiment-{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    script = Path(_experiment_probe.__file__).resolve()

    timeout = min(request.segment_timeout_seconds, EXPERIMENT_ABSOLUTE_WALL_CLOCK_SECONDS)
    started = time.perf_counter()
    timed_out = False
    exit_code: int | None = None
    with log_path.open("wb") as handle:
        process = subprocess.Popen(
            [str(request.python_executable), str(script)],
            cwd=str(request.trainer_root),
            stdin=subprocess.PIPE,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        try:
            assert process.stdin is not None
            process.stdin.write(json.dumps(payload).encode("utf-8"))
            process.stdin.close()
            exit_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = _terminate(process)
        except BrokenPipeError:
            exit_code = _terminate(process)

    return SegmentOutcome(
        name=name,
        document=_read_document(result_path),
        timed_out=timed_out,
        exit_code=exit_code,
        wall_seconds=round(time.perf_counter() - started, 3),
        log_tail=_tail(log_path),
        checkpoint=latest_checkpoint(output_dir),
    )


def _blocked(
    request: ExperimentRequest,
    identity: ExperimentIdentity,
    failure: str,
    detail: str,
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=identity.experiment_id(),
        identity=identity,
        outcome=ExperimentOutcome.BLOCKED.value,
        training_signal=TrainingSignal.INSUFFICIENT_EVIDENCE.value,
        training_signal_detail="the experiment did not run",
        generalization_signal=GeneralizationSignal.INSUFFICIENT_EVIDENCE.value,
        generalization_signal_detail="the experiment did not run",
        expected_steps=identity.expected_steps,
        failure=failure,
        failure_detail=detail,
        splits=_split_summary(request.splits),
        capacity_qualification=None if request.capacity is None else request.capacity.qualification,
        capacity_profile_id=None if request.capacity is None else request.capacity.profile_id,
        preflight_status=request.preflight_status,
        started_at=_now(),
        finished_at=_now(),
    )


def _split_summary(splits: dict[str, Any]) -> dict[str, Any]:
    """Counts and digests. Never track names or paths."""
    summary: dict[str, Any] = {"splits_digest": splits.get("splits_digest", "")}
    for key in ("train", "validation", "evaluation"):
        raw = splits.get(key) or {}
        summary[key] = {
            "track_count": raw.get("track_count"),
            "digest": raw.get("digest"),
            "group_distribution": raw.get("group_distribution"),
            "total_duration_seconds": raw.get("total_duration_seconds"),
        }
    return summary


def run_experiment(request: ExperimentRequest) -> ExperimentResult:
    """One controlled experiment, gated first and run second."""
    started_at = _now()
    started = time.perf_counter()

    # An epoch is however many samples the trainer visits, which is the
    # tensor count only when nothing reorders the loader. Weighted
    # exposure repeats samples, so its order *is* the epoch — budgeting
    # off the file count would under-count every epoch by the repeat
    # factor and the run would overshoot its ceiling and die mid-epoch
    # with nothing saved.
    train_samples = len(request.exposure_order or ()) or len(
        sorted(Path(request.train_dir).glob("*.pt"))
    )
    validation_tracks = len(sorted(Path(request.validation_dir).glob("*.pt")))

    segment_count = 2 if request.measure_resume else 1
    total_ceiling = min(request.step_budget, EXPERIMENT_MAX_OPTIMIZER_STEPS)
    segment_ceiling = max(1, total_ceiling // segment_count)

    try:
        budget = StepBudget.for_ceiling(
            samples=max(1, train_samples),
            micro_batch_size=request.plan.config.batch_size,
            gradient_accumulation=request.plan.config.gradient_accumulation,
            ceiling=segment_ceiling,
        )
    except Exception as exc:
        identity = identity_for(
            request,
            StepBudget(
                samples=max(1, train_samples),
                micro_batch_size=request.plan.config.batch_size,
                gradient_accumulation=request.plan.config.gradient_accumulation,
                epochs=1,
            ),
        )
        return _blocked(request, identity, ExperimentFailure.BUDGET_UNCOMPUTABLE.value, str(exc))

    identity = identity_for(request, budget)

    # ── the gates, in the order that makes the reason truthful ───────
    if request.gate_report is None or not request.gate_report.passed:
        first = None if request.gate_report is None else request.gate_report.first_failure
        return _blocked(
            request,
            identity,
            ExperimentFailure.RIGHTS_GATE_FAILED.value,
            (
                "no gate report was supplied, so the rights position of this material is "
                "unestablished"
                if request.gate_report is None
                else f"{first.name if first else 'a gate'} failed: {first.detail if first else ''}"
            ),
        )

    leakage = split_leakage_gate(request.splits)
    if not leakage.passed:
        return _blocked(request, identity, ExperimentFailure.SPLIT_LEAKAGE.value, leakage.detail)

    for label, report, directory in (
        ("training", request.tensor_report, request.train_dir),
        ("validation", request.validation_report, request.validation_dir),
    ):
        if report is None:
            continue
        if report.probe_failed:
            return _blocked(
                request,
                identity,
                ExperimentFailure.DATASET_UNUSABLE.value,
                f"the {label} tensors could not be read: {report.probe_failed}",
            )
        if report.rejected:
            reasons = "; ".join(
                f"{sample.name}: {sample.exclusion_reason}" for sample in report.rejected[:5]
            )
            return _blocked(
                request,
                identity,
                ExperimentFailure.DATASET_UNUSABLE.value,
                (
                    f"{len(report.rejected)} of {len(report.samples)} {label} tensor(s) in "
                    f"{directory.name} are unusable — {reasons}"
                ),
            )

    if request.capacity is not None and not request.capacity.permits_full_training:
        return _blocked(
            request,
            identity,
            ExperimentFailure.CAPACITY_NOT_QUALIFIED.value,
            (
                f"capacity is {request.capacity.qualification}; an experiment runs only on "
                "a machine measured evidence qualifies"
            ),
        )

    if request.preflight_status is not None and request.preflight_status != "READY":
        return _blocked(
            request,
            identity,
            ExperimentFailure.PREFLIGHT_NOT_READY.value,
            f"the preflight reported {request.preflight_status}",
        )

    provenance = compose_provenance(request)
    try:
        # Composed and validated before launch: a probe that writes an
        # incomplete record is worse than one that writes none, and the
        # place to catch that is here.
        from luber_training.checkpoint_provenance import write_checkpoint_provenance

        probe_dir = request.workspace / "_provenance_probe"
        probe_dir.mkdir(parents=True, exist_ok=True)
        write_checkpoint_provenance(probe_dir, provenance)
    except ProvenanceError as exc:
        return _blocked(request, identity, ExperimentFailure.PROVENANCE_INCOMPLETE.value, str(exc))

    return _execute(
        request,
        identity,
        budget,
        segment_ceiling=segment_ceiling,
        total_ceiling=total_ceiling,
        validation_tracks=validation_tracks,
        provenance=provenance,
        started=started,
        started_at=started_at,
    )


def _execute(
    request: ExperimentRequest,
    identity: ExperimentIdentity,
    budget: StepBudget,
    *,
    segment_ceiling: int,
    total_ceiling: int,
    validation_tracks: int,
    provenance: CheckpointProvenance,
    started: float,
    started_at: str,
) -> ExperimentResult:
    """Two bounded segments, a checkpoint, a resume, then a verdict."""
    from dataclasses import replace

    workspace = request.workspace
    workspace.mkdir(parents=True, exist_ok=True)
    plan = request.plan
    output_dir = workspace / "output"
    plan = replace(
        plan,
        config=plan.config.with_overrides(
            epochs=budget.epochs,
            seed=request.seed,
            # One update per step: the trainer yields a step update only
            # when `global_step % log_every == 0`, and a default of 10
            # would give a 120-step run twelve points.
            log_every_steps=1,
            # One checkpoint per segment, which is what the resume needs.
            checkpoint_every_epochs=(
                min(request.checkpoint_every_epochs, budget.epochs)
                if request.checkpoint_every_epochs
                else budget.epochs
            ),
            warmup_steps=0,
            num_workers=0,
            persistent_workers=False,
            prefetch_factor=0,
        ),
        # The placeholders a backend would substitute on a worker. This
        # runs locally, so they are substituted here — without this the
        # trainer is handed the literal `${LUBER_CHECKPOINT_DIR}` and
        # refuses before it loads anything.
        dataset_dir=str(request.train_dir),
        output_dir=str(output_dir),
        checkpoint_dir=str(request.model_dir),
    )

    # Recomposed from the *bounded* plan. The template validated before
    # launch was built from the request's plan, and bounding changes the
    # config digest — a provenance record citing a configuration that
    # never ran is exactly the kind of near-miss this file exists to
    # catch, and it caught it.
    provenance = replace(
        provenance,
        config_digest=plan.config.digest(),
        plan_digest=plan.digest(),
        checkpoint_path=str(output_dir),
    )

    digest_before = base_model_digest(request.model_dir, request.model_variant)

    segment_a = _run_segment(
        request,
        plan,
        name="A",
        step_ceiling=segment_ceiling,
        resume_from=request.resume_adapter,
        workspace=workspace,
        provenance=provenance,
    )

    segments = [
        SegmentRecord(
            name="A",
            completed_steps=len(segment_a.points),
            first_step=segment_a.points[0].step if segment_a.points else None,
            last_step=segment_a.points[-1].step if segment_a.points else None,
            exit_code=segment_a.exit_code,
            wall_seconds=segment_a.wall_seconds,
            checkpoint_id=None if segment_a.checkpoint is None else segment_a.checkpoint.name,
            validation_count=len(segment_a.validation),
            detail="" if segment_a.document else "the probe wrote no record",
        )
    ]

    points = list(segment_a.points)
    validation = list(segment_a.validation)
    checkpoint_integrity: CheckpointIntegrity | None = None
    resume: dict[str, Any] = {"performed": False}

    if segment_a.checkpoint is not None:
        checkpoint_integrity = inspect_checkpoint(
            segment_a.checkpoint, python_executable=request.python_executable
        )

    segment_b: SegmentOutcome | None = None
    if request.measure_resume and segment_a.checkpoint is not None:
        remaining = max(1, total_ceiling - len(points))
        # Epochs are cumulative in the trainer: a resume from epoch 10
        # with `--epochs 10` is already finished and exits in nine
        # seconds having trained nothing. The second segment therefore
        # asks for twice the epochs, and its extra half is what it runs.
        resumed_plan = replace(plan, config=plan.config.with_overrides(epochs=budget.epochs * 2))
        segment_b = _run_segment(
            request,
            resumed_plan,
            name="B",
            step_ceiling=min(total_ceiling, len(points) + remaining),
            resume_from=segment_a.checkpoint,
            workspace=workspace,
            provenance=provenance,
        )
        points += segment_b.points
        validation += segment_b.validation
        segments.append(
            SegmentRecord(
                name="B",
                completed_steps=len(segment_b.points),
                first_step=segment_b.points[0].step if segment_b.points else None,
                last_step=segment_b.points[-1].step if segment_b.points else None,
                exit_code=segment_b.exit_code,
                wall_seconds=segment_b.wall_seconds,
                checkpoint_id=(None if segment_b.checkpoint is None else segment_b.checkpoint.name),
                resumed_from=segment_a.checkpoint.name,
                validation_count=len(segment_b.validation),
            )
        )
        source_step = segments[0].last_step or 0
        final_step = segments[1].last_step or 0
        resume = {
            "performed": True,
            "resumed_from": segment_a.checkpoint.name,
            "source_step": source_step,
            "final_step": final_step,
            "advanced": final_step > source_step,
            "exit_code": segment_b.exit_code,
            "wall_seconds": segment_b.wall_seconds,
            "checkpoint_id": (None if segment_b.checkpoint is None else segment_b.checkpoint.name),
        }
        if segment_b.checkpoint is not None:
            checkpoint_integrity = inspect_checkpoint(
                segment_b.checkpoint, python_executable=request.python_executable
            )

    digest_after = base_model_digest(request.model_dir, request.model_variant)
    # The last segment that produced steps, not simply the last one. A
    # segment that exited without training has an empty fingerprint pair
    # and a zero gradient count, and reading those would report a
    # healthy run as NO_UPDATE.
    last = segment_b if (segment_b is not None and segment_b.points) else segment_a
    document = last.document or {}
    parameter_payload = dict(document.get("parameters") or {})
    parameters = ParameterUpdateEvidence(
        changed_tensor_count=parameter_payload.get("changed_tensor_count"),
        comparable_tensor_count=parameter_payload.get("comparable_tensor_count"),
        trainable_parameter_count=parameter_payload.get("trainable_parameter_count"),
        max_absolute_delta=parameter_payload.get("max_absolute_delta"),
        mean_absolute_delta=parameter_payload.get("mean_absolute_delta"),
        trainable_before_digest=parameter_payload.get("trainable_before_digest"),
        trainable_after_digest=parameter_payload.get("trainable_after_digest"),
        base_model_digest_before=digest_before,
        base_model_digest_after=digest_after,
        detail=str(parameter_payload.get("detail", "")),
    )
    # Summed across segments rather than read off the last one: the run
    # is both segments, and a gradient count covering half of it would
    # understate what was observed.
    payloads = [
        dict((outcome.document or {}).get("gradients") or {})
        for outcome in (segment_a, segment_b)
        if outcome is not None and outcome.points
    ]
    observed = sum(int(item.get("observed_steps") or 0) for item in payloads)
    minima: list[float] = [
        float(item["min_grad_norm"]) for item in payloads if item.get("min_grad_norm")
    ]
    maxima: list[float] = [
        float(item["max_grad_norm"]) for item in payloads if item.get("max_grad_norm")
    ]
    means: list[tuple[float, int]] = [
        (float(item["mean_grad_norm"]), int(item.get("observed_steps") or 0))
        for item in payloads
        if item.get("mean_grad_norm") is not None
    ]
    weighted = sum(value * count for value, count in means)
    counted = sum(count for _, count in means)
    gradients = GradientEvidence(
        observed_steps=observed,
        finite_steps=sum(int(item.get("finite_steps") or 0) for item in payloads),
        nonzero_steps=sum(int(item.get("nonzero_steps") or 0) for item in payloads),
        min_grad_norm=min(minima) if minima else None,
        max_grad_norm=max(maxima) if maxima else None,
        mean_grad_norm=(weighted / counted) if counted else None,
        detail=str(
            (payloads[0] if payloads else {}).get(
                "detail",
                "gradient norms as Lightning Fabric computed them while clipping",
            )
        ),
    )

    train_series = LossSeries(name="train", points=tuple(points))
    validation_points = tuple(validation)

    outcome = ExperimentOutcome.COMPLETED.value
    failure: str | None = None
    failure_detail = ""
    for segment_outcome in (segment_a, segment_b):
        if segment_outcome is None:
            continue
        if segment_outcome.timed_out:
            outcome = ExperimentOutcome.TIMED_OUT.value
            failure = ExperimentFailure.TIMED_OUT.value
            failure_detail = f"segment {segment_outcome.name} outran its wall clock"
            break
        recorded = (segment_outcome.document or {}).get("outcome")
        if recorded == "STEP_CEILING_EXCEEDED":
            outcome = ExperimentOutcome.STEP_CEILING_EXCEEDED.value
            failure = ExperimentFailure.STEP_CEILING_EXCEEDED.value
            failure_detail = str((segment_outcome.document or {}).get("failure_reason", ""))
            break
        if recorded != "COMPLETED" or segment_outcome.exit_code not in (0, None):
            outcome = ExperimentOutcome.FAILED.value
            failure = ExperimentFailure.TRAINER_FAILED.value
            failure_detail = (
                str((segment_outcome.document or {}).get("failure_reason", ""))
                or f"segment {segment_outcome.name} exited {segment_outcome.exit_code}"
            )
            break

    if failure is None and parameters.base_model_preserved is False:
        outcome = ExperimentOutcome.FAILED.value
        failure = ExperimentFailure.BASE_MODEL_MODIFIED.value
        failure_detail = "the base model weight files changed during the run"

    provenance_verdicts: dict[str, Any] = {}
    for segment_outcome in (segment_a, segment_b):
        if segment_outcome is None or segment_outcome.checkpoint is None:
            continue
        verdict = verify_checkpoint_provenance(
            segment_outcome.checkpoint,
            expected={
                "train_split_digest": identity.train_split_digest,
                "config_digest": plan.config.digest(),
                "run_id": plan.run_id,
            },
        )
        provenance_verdicts[segment_outcome.name] = verdict.to_dict()
        if failure is None and not verdict.ok:
            outcome = ExperimentOutcome.FAILED.value
            failure = ExperimentFailure.PROVENANCE_INCOMPLETE.value
            failure_detail = (
                f"the checkpoint from segment {segment_outcome.name} has no usable "
                f"provenance: {verdict.detail}"
            )

    if failure is None and request.measure_resume and not resume.get("advanced"):
        outcome = ExperimentOutcome.FAILED.value
        failure = ExperimentFailure.RESUME_FAILED.value
        failure_detail = "the resumed segment did not advance the optimizer step counter"

    training_signal, training_detail = classify_training_signal(
        loss=train_series,
        gradients=gradients,
        parameters=parameters,
        expected_steps=sum(segment.completed_steps for segment in segments),
    )
    generalization_signal, generalization_detail = classify_generalization(
        validation=validation_points,
        training=train_series,
        validation_track_count=validation_tracks,
    )

    return ExperimentResult(
        experiment_id=identity.experiment_id(),
        identity=identity,
        outcome=outcome,
        training_signal=training_signal,
        training_signal_detail=training_detail,
        generalization_signal=generalization_signal,
        generalization_signal_detail=generalization_detail,
        expected_steps=identity.expected_steps * (2 if request.measure_resume else 1),
        completed_steps=len(points),
        step_ceiling=total_ceiling,
        dataset_kind=DatasetKind.REAL_OPERATOR_AUTHORIZED,
        failure=failure,
        failure_detail=failure_detail,
        train_loss=train_series,
        validation_loss=validation_points,
        gradients=gradients,
        parameters=parameters,
        segments=tuple(segments),
        checkpoint={} if checkpoint_integrity is None else checkpoint_integrity.to_dict(),
        resume=resume,
        provenance=provenance_verdicts,
        splits=_split_summary(request.splits),
        capacity_qualification=(
            None if request.capacity is None else request.capacity.qualification
        ),
        capacity_profile_id=None if request.capacity is None else request.capacity.profile_id,
        preflight_status=request.preflight_status,
        wall_seconds=round(time.perf_counter() - started, 3),
        started_at=started_at,
        finished_at=_now(),
    )


def render_markdown(result: ExperimentResult) -> str:
    """A report an operator can read. Digests and counts, never paths."""
    train_stats = result.train_loss.statistics()
    validation = [point for point in result.validation_loss if point.finite]
    lines = [
        "# Controlled experiment",
        "",
        f"- experiment: `{result.experiment_id}`",
        f"- outcome: **{result.outcome}**",
        f"- training signal: **{result.training_signal}**",
        f"  - {result.training_signal_detail}",
        f"- generalization signal: **{result.generalization_signal}**",
        f"  - {result.generalization_signal_detail}",
        f"- listening evaluation required: **{result.listening_evaluation_required}**",
        f"- steps: {result.completed_steps} of a {result.step_ceiling} ceiling",
        f"- dataset kind: {result.dataset_kind}",
        "",
        "## Splits",
        "",
    ]
    for key in ("train", "validation", "evaluation"):
        split = result.splits.get(key) or {}
        lines.append(
            f"- {key}: {split.get('track_count')} track(s), digest "
            f"`{str(split.get('digest') or '')[:16]}`"
        )
    lines += ["", "## Loss", ""]
    if train_stats.get("finite_count"):
        lines += [
            f"- train: first {train_stats['first']:.4f}, last {train_stats['last']:.4f}, "
            f"min {train_stats['minimum']:.4f}, max {train_stats['maximum']:.4f}",
            f"- train finite ratio: {train_stats.get('finite_ratio')}",
            f"- train slope (DERIVED): {train_stats.get('slope')}",
        ]
    if validation:
        losses = [point.loss for point in validation if point.loss is not None]
        window = validation[0].latent_length
        lines += [
            f"- validation: first {losses[0]:.4f}, last {losses[-1]:.4f}, "
            f"min {min(losses):.4f}, max {max(losses):.4f}",
            f"- validation measurements: {len(validation)}",
            "- validation window: "
            + (
                f"{window} latent frames of each held-out track"
                if window
                else "the whole of each held-out track"
            ),
        ]
    else:
        lines.append("- validation: no finite measurement")
    lines += [
        "",
        "## Evidence",
        "",
        f"- gradients: {result.gradients.detail}",
        f"- parameters: {result.parameters.detail}",
        f"- base model preserved: {result.parameters.base_model_preserved}",
        f"- resume: {result.resume}",
        "",
        "No quality claim, no convergence claim. The checkpoint is "
        f"{', '.join(result.artifact_class)}.",
        "",
    ]
    return "\n".join(lines)


def write_experiment_artifacts(result: ExperimentResult, directory: Path) -> dict[str, str]:
    """Write the record and its report. Never into a source directory."""
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / EXPERIMENT_JSON
    json_path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path = directory / EXPERIMENT_MARKDOWN
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


__all__ = [
    "EXPERIMENT_JSON",
    "EXPERIMENT_MARKDOWN",
    "EXPERIMENT_SUBDIR",
    "VALIDATION_LATENT_LENGTH",
    "ExperimentRequest",
    "compose_provenance",
    "dataset_digest",
    "identity_for",
    "render_markdown",
    "run_experiment",
    "split_digests",
    "write_experiment_artifacts",
]
