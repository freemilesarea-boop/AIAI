"""Running a bounded pilot, and refusing to run one that is not ready.

The control-plane half of Phase 35. Most of it is refusals, in a fixed
order, because a pilot is the first thing in this repository that
touches real music and the first that could run for minutes rather than
seconds.

The order matters and is not negotiable:

1. **Rights.** Every Phase 25 gate, on the material itself. A pilot has
   no bypass and no `--force`.
2. **Dataset kind.** Real rights-cleared material, or a synthetic
   fixture — and a synthetic run is stamped as such for ever. A fixture
   validates mechanics; it can never be real-data evidence, and nothing
   downstream is allowed to read it as though it were.
3. **Step budget.** Computed from the trainer's own arithmetic and
   refused before a process exists if it exceeds the ceiling.
4. **Capacity.** An applicable Phase 34 profile that qualifies *this*
   configuration. A qualification for a different rank, batch, precision
   or sequence length is not evidence about this one.
5. **Preflight.** The whole Phase 33 gate.

Only then does anything start. The run itself is two bounded segments
with a checkpoint between them, so that resume is exercised rather than
assumed — and both segments together stay under one ceiling.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from luber_hardware import ComputeDevice
from luber_training import _pilot_probe
from luber_training.canary import (
    SYNTHETIC_FIXTURE_TYPE,
    inspect_checkpoint,
    latest_checkpoint,
    within,
)
from luber_training.capacity_policy import CapacityDecision, CapacityQualification
from luber_training.gates import GateReport
from luber_training.pilot import (
    ARTIFACT_CLASS,
    PILOT_ABSOLUTE_WALL_CLOCK_SECONDS,
    PILOT_MAX_OPTIMIZER_STEPS,
    PILOT_MAX_SEGMENT_STEPS,
    PILOT_MAX_WALL_CLOCK_SECONDS,
    PILOT_MIN_TRACKS,
    GradientEvidence,
    LossPoint,
    LossSeries,
    ParameterUpdateEvidence,
    PilotFailure,
    PilotIdentity,
    PilotOutcome,
    PilotSegment,
    PilotStepBudget,
    PilotTrainingResult,
    TrainingSignal,
    classify_signal,
    outcome_for,
)
from luber_training.plan import TrainingPlan
from luber_training.trainer_adapter import compile_command

#: Where a pilot keeps its workspace, beneath the trainer root. ACE-Step
#: validates `--dataset-dir` against its own working directory and
#: refuses anything outside it — after the model has loaded.
PILOT_SUBDIR = "pilot"

PILOT_LOSS_JSON = "pilot_loss.json"
PILOT_LOSS_MARKDOWN = "pilot_loss.md"
PILOT_DATASET_REPORT = "pilot_dataset.md"


class DatasetKind:
    """What a pilot trained on. Never blurred.

    The real-material value was renamed in Phase 36. It used to be
    ``REAL_RIGHTS_CLEARED``, and that name claimed more than the
    evidence supports: what actually clears this material is an
    operator's authorisation of a directory, with no ownership
    document, licence, publisher clearance or performer agreement
    behind it. A reader seeing "rights cleared" would reasonably infer
    all four. The old spelling still reads, because records written
    before the rename exist and rewriting history is worse than
    carrying a legacy name.
    """

    #: Real music an operator explicitly authorised for training. Says
    #: nothing about ownership or third-party clearance — see
    #: :class:`luber_dataset.RightsBasis.OPERATOR_AUTHORIZED_SCOPE`.
    REAL_OPERATOR_AUTHORIZED = "REAL_OPERATOR_AUTHORIZED"
    #: Deprecated spelling of the value above. Read, never written.
    REAL_RIGHTS_CLEARED = "REAL_RIGHTS_CLEARED"
    #: Generated tensors with no recording in them. Validates the
    #: mechanism and is never evidence about real data.
    SYNTHETIC_FIXTURE = "SYNTHETIC_FIXTURE"
    UNKNOWN = "UNKNOWN"

    #: Every spelling that means "real material a pilot may train on".
    REAL_VALUES = frozenset({REAL_OPERATOR_AUTHORIZED, REAL_RIGHTS_CLEARED})

    @classmethod
    def is_real(cls, value: str) -> bool:
        """Whether a recorded kind names real material, old name or new."""
        return value in cls.REAL_VALUES


def _now() -> str:
    return datetime.now(UTC).isoformat()


class PilotError(RuntimeError):
    """Raised when a pilot cannot be attempted as asked."""


@dataclass(frozen=True)
class DatasetVerdict:
    """Whether a pilot may train on a directory, and what it is."""

    permitted: bool
    kind: str
    sample_count: int = 0
    manifest_digest: str | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "permitted": self.permitted,
            "kind": self.kind,
            "sample_count": self.sample_count,
            "manifest_digest": self.manifest_digest,
            "detail": self.detail,
        }


def dataset_digest(dataset_dir: Path) -> str:
    """A content digest over the preprocessed tensors, by name and bytes.

    The pilot's immutability anchor. Once a plan cites this, the data
    may not change: a different digest is a different pilot, not an
    edited one.
    """
    digest = hashlib.sha256()
    for path in sorted(dataset_dir.glob("*.pt")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def verify_pilot_dataset(
    dataset_dir: Path,
    *,
    gate_report: GateReport | None = None,
    minimum_tracks: int = PILOT_MIN_TRACKS,
    allow_synthetic: bool = False,
) -> DatasetVerdict:
    """Whether a pilot may train here, and on what kind of material.

    Real material needs every Phase 25 gate to have passed and at least
    :data:`PILOT_MIN_TRACKS` samples — below that a loss series is a
    statement about one recording rather than about training.

    A synthetic fixture is permitted only when the caller asked for one
    explicitly, and the verdict says so, so that a mechanism check can
    never be mistaken for real-data evidence further down.
    """
    if not dataset_dir.is_dir():
        return DatasetVerdict(False, DatasetKind.UNKNOWN, detail=f"{dataset_dir} does not exist")

    samples = sorted(dataset_dir.glob("*.pt"))
    if not samples:
        return DatasetVerdict(
            False, DatasetKind.UNKNOWN, detail=f"{dataset_dir} holds no preprocessed tensors"
        )

    digest = dataset_digest(dataset_dir)
    manifest = dataset_dir / "manifest.json"
    synthetic = False
    if manifest.is_file():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            kind = ((payload.get("metadata") or {}).get("type")) or payload.get("type")
            synthetic = kind == SYNTHETIC_FIXTURE_TYPE
        except (OSError, json.JSONDecodeError):
            synthetic = False

    if synthetic:
        if not allow_synthetic:
            return DatasetVerdict(
                False,
                DatasetKind.SYNTHETIC_FIXTURE,
                sample_count=len(samples),
                manifest_digest=digest,
                detail=(
                    "this is a synthetic fixture. It can validate that the pilot mechanism "
                    "works and can never be evidence about real data, so it is refused "
                    "unless a caller asks for a mechanism check by name"
                ),
            )
        return DatasetVerdict(
            True,
            DatasetKind.SYNTHETIC_FIXTURE,
            sample_count=len(samples),
            manifest_digest=digest,
            detail=(
                f"{len(samples)} synthetic tensor sample(s); no audio and no rights-bearing "
                "material. A pass here is about the mechanism and about nothing else"
            ),
        )

    if gate_report is None or not gate_report.passed:
        failure = None if gate_report is None else gate_report.first_failure
        return DatasetVerdict(
            False,
            DatasetKind.UNKNOWN,
            sample_count=len(samples),
            manifest_digest=digest,
            detail=(
                "no gate report was supplied, so the rights position of this material is "
                "unestablished"
                if gate_report is None
                else f"a rights gate failed: {failure.name if failure else 'unknown'} — "
                f"{failure.detail if failure else ''}"
            ),
        )

    if len(samples) < minimum_tracks:
        return DatasetVerdict(
            False,
            DatasetKind.REAL_OPERATOR_AUTHORIZED,
            sample_count=len(samples),
            manifest_digest=digest,
            detail=(
                f"{len(samples)} sample(s); a signal pilot needs at least {minimum_tracks}, "
                "or its loss series describes one recording rather than training"
            ),
        )

    return DatasetVerdict(
        True,
        DatasetKind.REAL_OPERATOR_AUTHORIZED,
        sample_count=len(samples),
        manifest_digest=digest,
        detail=(
            f"{len(samples)} operator-authorised sample(s), every Phase 25 gate passed. "
            "The gates establish that an operator authorised this material, not that "
            "anyone verified ownership or a third-party licence"
        ),
    )


# ── the request ──────────────────────────────────────────────────────


@dataclass
class PilotRequest:
    """Everything one bounded pilot needs, and nothing that could widen it."""

    plan: TrainingPlan
    dataset_dir: Path
    trainer_root: Path
    python_executable: Path
    model_dir: Path
    workspace: Path
    dataset_id: str = "pilot"
    latent_length: int = 0
    encoder_length: int = 0
    seed: int = 42
    model_variant: str = "turbo"
    gate_report: GateReport | None = None
    capacity: CapacityDecision | None = None
    preflight_status: str | None = None
    allow_synthetic: bool = False
    #: Wall clock for one segment. Clamped by the module ceiling.
    segment_timeout_seconds: float = PILOT_MAX_WALL_CLOCK_SECONDS
    #: Whether to run the second, resumed segment. On by default: a
    #: pilot that never resumed has not shown that a longer run could
    #: survive an interruption.
    measure_resume: bool = True


def _blocked(
    request: PilotRequest,
    identity: PilotIdentity,
    failure: str,
    detail: str,
    *,
    dataset_kind: str = DatasetKind.UNKNOWN,
) -> PilotTrainingResult:
    return PilotTrainingResult(
        pilot_id=identity.pilot_id(),
        identity=identity,
        outcome=PilotOutcome.BLOCKED.value,
        signal=TrainingSignal.INSUFFICIENT_EVIDENCE.value,
        signal_detail="the pilot did not run",
        failure=failure,
        failure_detail=detail,
        expected_steps=identity.expected_steps,
        dataset_kind=dataset_kind,
        capacity_profile_id=None if request.capacity is None else request.capacity.profile_id,
        capacity_qualification=(
            None if request.capacity is None else request.capacity.qualification
        ),
        preflight_status=request.preflight_status,
        started_at=_now(),
        finished_at=_now(),
    )


def identity_for(request: PilotRequest, budget: PilotStepBudget, digest: str) -> PilotIdentity:
    config = request.plan.config
    return PilotIdentity(
        plan_digest=request.plan.digest(),
        dataset_manifest_digest=digest,
        dataset_id=request.dataset_id,
        base_model_id=request.plan.base_model_id,
        base_model_upstream_commit=request.plan.base_model_upstream_commit,
        ace_step_commit=config.ace_step_commit,
        device=request.plan.requirements.execution_device or ComputeDevice.CPU.value,
        precision=config.precision,
        optimizer=config.optimizer_type,
        lora_rank=config.rank,
        lora_alpha=config.alpha,
        micro_batch_size=config.batch_size,
        gradient_accumulation=config.gradient_accumulation,
        epochs=budget.epochs,
        expected_steps=budget.expected_steps,
        latent_length=request.latent_length,
        encoder_length=request.encoder_length,
        seed=request.seed,
    )


def base_model_digest(model_dir: Path, variant: str) -> str | None:
    """A digest over the base model's weight files, not its parameters.

    The question is whether anything wrote to the base model, and a file
    digest answers it. Hashing 2.4 billion parameters through torch
    would answer the same question at a hundred times the cost.
    """
    directory = Path(model_dir) / f"acestep-v15-{variant}"
    if not directory.is_dir():
        return None
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.safetensors")):
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(int(stat.st_mtime)).encode("utf-8"))
    return digest.hexdigest() or None


# ── running one segment ──────────────────────────────────────────────


@dataclass
class SegmentOutcome:
    """What one bounded stretch of training produced."""

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


def _run_segment(
    request: PilotRequest,
    plan: TrainingPlan,
    *,
    name: str,
    step_ceiling: int,
    resume_from: Path | None,
    workspace: Path,
) -> SegmentOutcome:
    """Launch the trainer through the pilot probe, under a wall clock."""
    output_dir = workspace / "output"
    command = compile_command(
        plan,
        trainer_root=str(request.trainer_root),
        python_executable=str(request.python_executable),
        model_variant=request.model_variant,
        resume_from=None if resume_from is None else str(resume_from),
    )
    result_path = workspace / f"pilot_probe_{name}.json"
    payload = {
        "argv": command.argv[1:],
        "result_path": str(result_path),
        "step_ceiling": step_ceiling,
        "segment": name,
    }
    log_path = workspace / f"pilot-{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    script = Path(_pilot_probe.__file__).resolve()

    timeout = min(request.segment_timeout_seconds, PILOT_ABSOLUTE_WALL_CLOCK_SECONDS)
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


def _terminate(process: subprocess.Popen[bytes]) -> int | None:
    """Stop a segment that outran its clock, process group and all."""
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


def _tail(path: Path, lines: int = 40) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(line for line in content[-lines:] if line.strip())


def _read_document(path: Path) -> dict[str, Any] | None:
    """Read the probe's record, refusing anything it cannot verify."""
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict):
        return None
    version = str(document.get("protocol_version", ""))
    if version != _pilot_probe.PILOT_PROBE_PROTOCOL_VERSION:
        return {
            "outcome": "FAILED",
            "failure_reason": (
                f"the probe reported protocol {version!r}, and this build reads "
                f"{_pilot_probe.PILOT_PROBE_PROTOCOL_VERSION!r}"
            ),
            "loss_points": [],
        }
    return document


# ── the pilot ────────────────────────────────────────────────────────


def run_pilot(request: PilotRequest) -> PilotTrainingResult:
    """Every gate, then two bounded segments, then a verdict."""
    workspace = Path(request.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(request.dataset_dir)

    verdict = verify_pilot_dataset(
        dataset_dir,
        gate_report=request.gate_report,
        allow_synthetic=request.allow_synthetic,
    )

    # Half the total when a resume is coming, all of it otherwise, so
    # the two segments together fill exactly one ceiling. Derived here
    # rather than configured, so the segment bound and the total bound
    # cannot drift into disagreeing.
    segment_ceiling = min(
        PILOT_MAX_SEGMENT_STEPS,
        PILOT_MAX_OPTIMIZER_STEPS // (2 if request.measure_resume else 1),
    )

    # The budget is computed even when something else blocks, so the
    # record says what would have run.
    try:
        budget = PilotStepBudget.for_ceiling(
            samples=max(1, verdict.sample_count),
            micro_batch_size=request.plan.config.batch_size,
            gradient_accumulation=request.plan.config.gradient_accumulation,
            ceiling=segment_ceiling,
        )
    except Exception as exc:
        budget = PilotStepBudget(
            samples=max(1, verdict.sample_count),
            micro_batch_size=request.plan.config.batch_size,
            gradient_accumulation=request.plan.config.gradient_accumulation,
            epochs=1,
        )
        identity = identity_for(request, budget, verdict.manifest_digest or "")
        return _blocked(
            request,
            identity,
            PilotFailure.STEP_BUDGET_EXCEEDED.value,
            str(exc),
            dataset_kind=verdict.kind,
        )

    identity = identity_for(request, budget, verdict.manifest_digest or "")

    if not verdict.permitted:
        failure = (
            PilotFailure.NO_RIGHTS_CLEARED_DATA.value
            if verdict.kind in (DatasetKind.UNKNOWN, DatasetKind.SYNTHETIC_FIXTURE)
            else PilotFailure.DATASET_INVALID.value
        )
        return _blocked(request, identity, failure, verdict.detail, dataset_kind=verdict.kind)

    # A resumed pilot runs both segments, so the total has to fit.
    total = budget.expected_steps * (2 if request.measure_resume else 1)
    if total > PILOT_MAX_OPTIMIZER_STEPS:
        return _blocked(
            request,
            identity,
            PilotFailure.STEP_BUDGET_EXCEEDED.value,
            f"two segments of {budget.expected_steps} step(s) is {total}, past the "
            f"{PILOT_MAX_OPTIMIZER_STEPS}-step pilot ceiling",
            dataset_kind=verdict.kind,
        )

    if request.capacity is None or request.capacity.qualification not in (
        CapacityQualification.QUALIFIED.value,
        CapacityQualification.MARGIN_LOW.value,
    ):
        return _blocked(
            request,
            identity,
            PilotFailure.CAPACITY_NOT_QUALIFIED.value,
            (
                "no applicable memory profile qualifies this configuration"
                if request.capacity is None
                else f"capacity is {request.capacity.qualification}: "
                + "; ".join(request.capacity.reasons[:2])
            ),
            dataset_kind=verdict.kind,
        )

    if request.preflight_status is not None and request.preflight_status != "READY":
        return _blocked(
            request,
            identity,
            PilotFailure.PREFLIGHT_BLOCKED.value,
            f"the training preflight is {request.preflight_status}",
            dataset_kind=verdict.kind,
        )

    for label, path in (
        ("trainer", request.trainer_root),
        ("base model root", request.model_dir),
    ):
        if not Path(path).is_dir():
            return _blocked(
                request,
                identity,
                PilotFailure.TRAINER_FAILED.value,
                f"the {label} is not present at {path}",
                dataset_kind=verdict.kind,
            )
    if not Path(request.python_executable).is_file():
        return _blocked(
            request,
            identity,
            PilotFailure.TRAINER_FAILED.value,
            f"no interpreter at {request.python_executable}",
            dataset_kind=verdict.kind,
        )
    if not within(dataset_dir, Path(request.trainer_root)):
        return _blocked(
            request,
            identity,
            PilotFailure.DATASET_INVALID.value,
            f"the pilot dataset at {dataset_dir} is outside the trainer's working directory "
            f"({request.trainer_root}); ACE-Step refuses it after the model has loaded",
            dataset_kind=verdict.kind,
        )

    return _execute(request, identity, budget, verdict, workspace)


def _execute(
    request: PilotRequest,
    identity: PilotIdentity,
    budget: PilotStepBudget,
    verdict: DatasetVerdict,
    workspace: Path,
) -> PilotTrainingResult:
    """Two bounded segments with a checkpoint between them."""
    output_dir = workspace / "output"
    started_at = _now()
    started = time.perf_counter()

    base_before = base_model_digest(request.model_dir, request.model_variant)

    bounded = replace(
        request.plan,
        config=request.plan.config.with_overrides(
            epochs=budget.epochs,
            seed=request.seed,
            # One checkpoint at the end of the segment: enough for the
            # next segment to resume from, and not one per epoch — a
            # 24-epoch segment writing 24 adapters fills a disk with 23
            # artifacts nobody will read.
            checkpoint_every_epochs=budget.epochs,
            # Every optimizer step, not every tenth. The trainer only
            # *yields* a step update when `global_step % log_every == 0`,
            # and the default of 10 would give a pilot four points out of
            # forty-eight — a loss series with nine tenths of itself
            # missing, silently.
            log_every_steps=1,
            warmup_steps=0,
            num_workers=0,
            persistent_workers=False,
            prefetch_factor=0,
        ),
        dataset_dir=str(request.dataset_dir),
        output_dir=str(output_dir),
        checkpoint_dir=str(request.model_dir),
    )

    result = PilotTrainingResult(
        pilot_id=identity.pilot_id(),
        identity=identity,
        expected_steps=budget.expected_steps * (2 if request.measure_resume else 1),
        dataset_kind=verdict.kind,
        capacity_profile_id=None if request.capacity is None else request.capacity.profile_id,
        capacity_qualification=(
            None if request.capacity is None else request.capacity.qualification
        ),
        preflight_status=request.preflight_status,
        started_at=started_at,
    )

    first = _run_segment(
        request,
        bounded,
        name="A",
        step_ceiling=PILOT_MAX_SEGMENT_STEPS,  # the probe's own last-resort guard
        resume_from=None,
        workspace=workspace,
    )
    points = first.points
    result.segments.append(
        PilotSegment(
            name="A",
            step_budget=budget.to_dict(),
            first_step=points[0].step if points else None,
            last_step=points[-1].step if points else None,
            completed_steps=len(points),
            checkpoint_id=None if first.checkpoint is None else first.checkpoint.name,
            exit_code=first.exit_code,
            wall_seconds=first.wall_seconds,
            detail=str((first.document or {}).get("failure_reason", "")),
        )
    )

    if first.timed_out:
        return _finish(
            request,
            result,
            points,
            first,
            None,
            base_before,
            started,
            outcome=PilotOutcome.TIMEOUT.value,
            failure=PilotFailure.TIMEOUT.value,
            failure_detail=(
                f"segment A exceeded its {request.segment_timeout_seconds:.0f}s wall clock "
                "and was stopped. A killed segment is not a completed one"
            ),
        )

    if first.document is None or first.document.get("outcome") != "COMPLETED":
        detail = (first.document or {}).get("failure_reason") or first.log_tail[-400:]
        return _finish(
            request,
            result,
            points,
            first,
            None,
            base_before,
            started,
            outcome=PilotOutcome.FAILED_RUNTIME.value,
            failure=PilotFailure.TRAINER_FAILED.value,
            failure_detail=str(detail),
        )

    checkpoint_integrity = None
    if first.checkpoint is not None:
        checkpoint_integrity = inspect_checkpoint(
            first.checkpoint, python_executable=request.python_executable
        )
        result.checkpoint = checkpoint_integrity.to_dict()
    if first.checkpoint is None or checkpoint_integrity is None or not checkpoint_integrity.exists:
        return _finish(
            request,
            result,
            points,
            first,
            None,
            base_before,
            started,
            outcome=PilotOutcome.FAILED_RUNTIME.value,
            failure=PilotFailure.CHECKPOINT_FAILED.value,
            failure_detail="segment A wrote no checkpoint, so there is nothing to resume from",
        )

    second: SegmentOutcome | None = None
    if request.measure_resume:
        resumed = replace(bounded, config=bounded.config.with_overrides(epochs=budget.epochs * 2))
        second = _run_segment(
            request,
            resumed,
            name="B",
            step_ceiling=PILOT_MAX_OPTIMIZER_STEPS,
            resume_from=first.checkpoint,
            workspace=workspace,
        )
        second_points = second.points
        result.segments.append(
            PilotSegment(
                name="B",
                step_budget=budget.to_dict(),
                first_step=second_points[0].step if second_points else None,
                last_step=second_points[-1].step if second_points else None,
                completed_steps=len(second_points),
                checkpoint_id=None if second.checkpoint is None else second.checkpoint.name,
                resumed_from=first.checkpoint.name,
                exit_code=second.exit_code,
                wall_seconds=second.wall_seconds,
                detail=str((second.document or {}).get("failure_reason", "")),
            )
        )
        points = points + second_points

        advanced = bool(
            second_points
            and points
            and second_points[-1].step > (first.points[-1].step if first.points else 0)
        )
        result.resume = {
            "performed": True,
            "resumed_from": first.checkpoint.name,
            "source_step": first.points[-1].step if first.points else None,
            "final_step": second_points[-1].step if second_points else None,
            "advanced": advanced,
            "exit_code": second.exit_code,
            "wall_seconds": second.wall_seconds,
            "checkpoint_id": None if second.checkpoint is None else second.checkpoint.name,
        }
        if (
            second.timed_out
            or second.document is None
            or second.document.get("outcome") != "COMPLETED"
        ):
            return _finish(
                request,
                result,
                points,
                first,
                second,
                base_before,
                started,
                outcome=(
                    PilotOutcome.TIMEOUT.value
                    if second.timed_out
                    else PilotOutcome.FAILED_RUNTIME.value
                ),
                failure=(
                    PilotFailure.TIMEOUT.value
                    if second.timed_out
                    else PilotFailure.RESUME_FAILED.value
                ),
                failure_detail=str(
                    (second.document or {}).get("failure_reason") or "the resumed segment failed"
                ),
            )
        if not advanced:
            return _finish(
                request,
                result,
                points,
                first,
                second,
                base_before,
                started,
                outcome=PilotOutcome.FAILED_RUNTIME.value,
                failure=PilotFailure.RESUME_FAILED.value,
                failure_detail=(
                    "the step counter did not advance across the resume; the run restarted "
                    "rather than continued"
                ),
            )

    return _finish(request, result, points, first, second, base_before, started)


def _finish(
    request: PilotRequest,
    result: PilotTrainingResult,
    points: list[LossPoint],
    first: SegmentOutcome,
    second: SegmentOutcome | None,
    base_before: str | None,
    started: float,
    *,
    outcome: str | None = None,
    failure: str | None = None,
    failure_detail: str = "",
) -> PilotTrainingResult:
    """Assemble the evidence and classify it, whatever happened."""
    result.loss = LossSeries(points=points)
    result.completed_steps = len(points)
    result.wall_seconds = round(time.perf_counter() - started, 3)
    result.finished_at = _now()

    last = second or first
    parameters = (last.document or {}).get("parameters") or {}
    gradients = (last.document or {}).get("gradients") or {}
    evidence = ParameterUpdateEvidence.from_dict(
        {
            **parameters,
            "base_model_digest_before": base_before,
            "base_model_digest_after": base_model_digest(request.model_dir, request.model_variant),
        }
    )
    result.parameters = evidence
    result.gradients = GradientEvidence.from_dict(gradients)

    signal, detail = classify_signal(
        loss=result.loss,
        parameters=result.parameters,
        gradients=result.gradients,
        expected_steps=result.expected_steps,
        completed_steps=result.completed_steps,
    )
    result.signal = signal
    result.signal_detail = detail

    if outcome is not None:
        result.outcome = outcome
        result.failure = failure
        result.failure_detail = failure_detail
        return result

    result.outcome = outcome_for(signal, completed=True)
    if result.outcome == PilotOutcome.FAILED_NUMERIC.value:
        result.failure = (
            PilotFailure.NO_PARAMETER_UPDATE.value
            if signal == TrainingSignal.NO_UPDATE.value
            else PilotFailure.LOSS_NONFINITE.value
        )
        result.failure_detail = detail
    return result


# ── artifacts ────────────────────────────────────────────────────────


def write_pilot_artifacts(result: PilotTrainingResult, directory: Path) -> dict[str, Path]:
    """Persist the loss series and its report. Operational, never committed."""
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / PILOT_LOSS_JSON
    json_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path = directory / PILOT_LOSS_MARKDOWN
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def render_markdown(result: PilotTrainingResult) -> str:
    """A pilot in the form an operator reads.

    Every number is descriptive. The slope is labelled derived, and the
    document says in two places that none of this is a quality or
    convergence claim — once at the top, where somebody skimming will
    see it, and once at the bottom, where somebody quoting will.
    """
    statistics = result.loss.statistics()
    identity = result.identity
    lines = [
        f"# Pilot — {result.pilot_id}",
        "",
        f"- outcome: **{result.outcome}**",
        f"- training signal: **{result.signal}**",
        f"  - {result.signal_detail}",
        f"- dataset kind: **{result.dataset_kind}**",
        f"- steps: **{result.completed_steps}** completed of {result.expected_steps} expected "
        f"(ceiling {PILOT_MAX_OPTIMIZER_STEPS})",
        f"- artifact class: {', '.join(result.artifact_class)}",
        "",
        "> A pilot shows that the training path produces a coherent signal.",
        "> It is not a convergence result and not a quality result.",
        "> Its checkpoint must never be promoted.",
        "",
        "## Configuration",
        "",
        f"- device **{identity.device}**, precision **{identity.precision}**, "
        f"optimizer **{identity.optimizer}**, seed {identity.seed}",
        f"- micro batch {identity.micro_batch_size}, accumulation "
        f"{identity.gradient_accumulation}, {identity.epochs} epoch(s) per segment",
        f"- LoRA rank {identity.lora_rank}, alpha {identity.lora_alpha}",
        f"- latent length {identity.latent_length}, encoder length {identity.encoder_length}",
        f"- plan `{identity.plan_digest[:16]}` · dataset "
        f"`{identity.dataset_manifest_digest[:16]}` · ACE-Step "
        f"`{identity.ace_step_commit[:12]}`",
        f"- capacity: {result.capacity_qualification} (profile {result.capacity_profile_id})",
        "",
        "## Loss",
        "",
        "| statistic | value |",
        "|---|---|",
    ]
    for key in (
        "count",
        "finite_count",
        "finite_ratio",
        "first",
        "last",
        "minimum",
        "maximum",
        "mean",
        "median",
    ):
        lines.append(f"| {key} | {statistics.get(key)} |")
    lines.append(f"| slope (DERIVED) | {statistics.get('slope')} |")
    lines += [
        "",
        f"_{statistics.get('slope_note', '')}_",
        "",
        "## Evidence",
        "",
        f"- trainable tensors changed: **{result.parameters.changed_tensor_count}** of "
        f"{result.parameters.trainable_tensor_count}",
        f"- max absolute delta: {result.parameters.max_absolute_delta}",
        f"- base model preserved: **{result.parameters.base_model_preserved}**",
        f"- gradients finite on {result.gradients.finite_steps} of "
        f"{result.gradients.observed_steps} observed step(s); "
        f"{result.gradients.nonzero_steps} non-zero",
        "",
        "## Segments",
        "",
    ]
    for segment in result.segments:
        lines.append(
            f"- **{segment.name}**: {segment.completed_steps} step(s) "
            f"({segment.first_step}→{segment.last_step}), checkpoint "
            f"{segment.checkpoint_id}, {segment.wall_seconds}s"
            + (f", resumed from {segment.resumed_from}" if segment.resumed_from else "")
        )
    lines += [
        "",
        "## What this does not say",
        "",
        "Nothing about convergence, music quality, generalisation, or whether the adapter",
        "improves anything. A pilot of tens of steps cannot support any of those, and this",
        "document deliberately has no vocabulary for them.",
        "",
    ]
    return "\n".join(lines)


def render_dataset_report(verdict: DatasetVerdict, identity: PilotIdentity) -> str:
    """The dataset an operator is about to train on, before it happens.

    Counts and digests, never track names or lyrics. A pilot report is
    read and shared; the material it describes is not.
    """
    return "\n".join(
        [
            "# Pilot dataset",
            "",
            f"- kind: **{verdict.kind}**",
            f"- permitted: **{verdict.permitted}**",
            f"- samples: **{verdict.sample_count}**",
            f"- manifest digest: `{(verdict.manifest_digest or '')[:16]}`",
            f"- dataset id: `{identity.dataset_id}`",
            f"- latent length: {identity.latent_length}, encoder length: {identity.encoder_length}",
            "",
            verdict.detail,
            "",
            "Sample identities are digests and counts. Track names, lyrics and paths are",
            "deliberately absent: this report is read and shared, and the material it",
            "describes is not.",
            "",
        ]
    )


__all__ = [
    "ARTIFACT_CLASS",
    "PILOT_DATASET_REPORT",
    "PILOT_LOSS_JSON",
    "PILOT_LOSS_MARKDOWN",
    "PILOT_SUBDIR",
    "DatasetKind",
    "DatasetVerdict",
    "PilotError",
    "PilotRequest",
    "SegmentOutcome",
    "base_model_digest",
    "dataset_digest",
    "identity_for",
    "render_dataset_report",
    "render_markdown",
    "run_pilot",
    "verify_pilot_dataset",
    "write_pilot_artifacts",
]
