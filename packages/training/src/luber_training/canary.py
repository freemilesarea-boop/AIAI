"""A training run small enough to be safe, real enough to be evidence.

The gap this closes is specific. Phase 25 can walk the whole lifecycle
without training anything; Phase 32 proved that a device can carry eight
steps of a two-layer toy network. Neither answers the question an
operator actually has before renting a GPU: *will this trainer, with
this config, on this machine, load a model, take a step and write a
checkpoint we can reopen?*

A canary answers it by doing it — on four synthetic tensors, for one
epoch, under a wall clock.

**Two canaries, and they are not interchangeable.**

``ORCHESTRATION`` proves LUBER: the plan compiles, the bounded config
is inside its envelope, the command the trainer would receive is the
command LUBER meant to send, and the directories exist. It trains
nothing and says so. It is what remains available when the trainer is
on a machine this process cannot reach.

``ACE_STEP`` proves the trainer: the real ACE-Step DiT is loaded, a real
LoRA is injected, real optimizer steps run, and a real checkpoint is
written and reopened. Only the *data* is synthetic — upstream's own
`make_test_fixtures`, which exists for exactly this and marks every
sample ``is_synthetic``. A pass here is evidence about the mechanism and
about nothing else: no music was involved, the adapter it produces has
learned noise, and the resulting checkpoint must never be promoted.

**The bounds are structural, not a habit.** Every limit is a module
constant, an envelope validates against it on construction, and the
bounded config is derived rather than supplied — so there is no
parameter anywhere in this module that can be raised to turn a canary
into a training run. A caller asking for nine optimizer steps gets
:class:`CanaryBoundsError`, not nine steps.

**Rights are not relaxed for being small.** A canary either trains on a
synthetic fixture that carries no rights at all, or on material whose
Phase 25 gates passed. There is no third option and no flag that
creates one.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from luber_training import _checkpoint_probe
from luber_training.config import TrainingConfig
from luber_training.gates import GateReport
from luber_training.plan import TrainingPlan
from luber_training.preflight import CanaryEvidence
from luber_training.trainer_adapter import CompiledCommand, compile_command

CANARY_SCHEMA_VERSION = "luber-training-canary/1"

# ── the ceilings ─────────────────────────────────────────────────────
#
# Module constants rather than defaults, so that raising one is a code
# change with a diff and a review rather than a parameter somebody
# passes at three in the morning.

#: Epochs a plain canary may run. The installed trainer measures length
#: in epochs only — there is no `--max-steps` — so this is the primary
#: bound and everything else follows from it.
CANARY_MAX_EPOCHS = 1

#: Epochs a *resume* canary may run in total. Two, and the second one is
#: the entire point: the trainer saves at the end of an epoch and
#: resumes at the start of the next, so a one-epoch resume would restart
#: into an empty range, take zero steps, and prove nothing.
CANARY_MAX_RESUME_EPOCHS = 2

#: Samples a canary dataset may contain.
CANARY_MAX_SAMPLES = 4

#: Optimizer steps a canary may take, over all its epochs.
CANARY_MAX_OPTIMIZER_STEPS = 8

#: Default wall clock. Generous for four short sequences and small
#: enough that a stuck model load is a failure rather than an evening.
CANARY_MAX_WALL_CLOCK_SECONDS = 1800.0

#: The ceiling on the wall clock itself. An envelope may not ask for
#: longer, whatever the caller believes about their hardware.
CANARY_ABSOLUTE_WALL_CLOCK_SECONDS = 3600.0

#: Sequence lengths for the synthetic fixture. Short on purpose: the
#: question is whether a step runs, not how long a real song takes.
CANARY_LATENT_LENGTH = 64
CANARY_ENCODER_LENGTH = 32

#: The marker upstream's fixture generator writes into its manifest.
#: A canary refuses any dataset that is neither this nor gate-cleared.
SYNTHETIC_FIXTURE_TYPE = "synthetic_test_fixtures"

#: Where the installed trainer writes what it saves, relative to
#: `--output-dir`. Read from `trainer_fixed._train_fabric` at the pinned
#: commit: per-epoch checkpoints under `checkpoints/`, the final adapter
#: under `final/`.
TRAINER_CHECKPOINTS_SUBDIR = "checkpoints"
TRAINER_FINAL_SUBDIR = "final"

#: `epoch_{n}_loss_{x}` — the directory name the trainer builds.
_CHECKPOINT_NAME = re.compile(r"^epoch_(\d+)_loss_")

#: The provenance sidecar LUBER writes beside a canary checkpoint.
PROVENANCE_NAME = "luber_canary_provenance.json"

#: Where a canary keeps its workspace, beneath the trainer root.
#:
#: Not a preference. ACE-Step's `path_safety._SAFE_ROOT` is the working
#: directory at import time, and `PreprocessedTensorDataset` puts
#: `--dataset-dir` through `safe_path` against it — so a dataset outside
#: the trainer's working directory is refused with "Path escapes safe
#: root" *after* the 2.4B model has been loaded. There is no environment
#: variable for it and `set_safe_root` is in-process only, so the only
#: way to satisfy the trainer is to put the data where it will look.
CANARY_TRAINER_SUBDIR = ".luber-canary"


def default_workspace(trainer_root: Path, name: str = "run") -> Path:
    """Where a canary may safely put its dataset on this trainer.

    Beneath the trainer root, because the trainer refuses a dataset
    anywhere else — see :data:`CANARY_TRAINER_SUBDIR`.
    """
    return Path(trainer_root) / CANARY_TRAINER_SUBDIR / name


def within(path: Path, root: Path) -> bool:
    """Whether *path* resolves inside *root*, symlinks included.

    The same question ACE-Step's `safe_path` asks, asked before the
    model is loaded rather than after.
    """
    resolved = Path(os.path.realpath(path))
    base = Path(os.path.realpath(root))
    return resolved == base or base in resolved.parents


class CanaryBoundsError(ValueError):
    """Raised when something asks for more than a canary may do."""


class CanaryMode(StrEnum):
    ORCHESTRATION = "ORCHESTRATION"
    ACE_STEP = "ACE_STEP"


class CanaryStatus(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    #: Something outside the canary prevents it — no trainer, no
    #: rights-cleared data, no interpreter. Distinct from FAILED, which
    #: means the canary ran and the trainer did not do its job.
    BLOCKED = "BLOCKED"
    NOT_RUN = "NOT_RUN"


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class CanaryEnvelope:
    """The hard bounds one canary runs inside.

    Validated on construction against the module ceilings, so an
    envelope that exists is an envelope that is safe. Nothing downstream
    re-checks, because nothing downstream can be handed an invalid one.
    """

    max_epochs: int = CANARY_MAX_EPOCHS
    max_samples: int = 2
    max_optimizer_steps: int = CANARY_MAX_OPTIMIZER_STEPS
    wall_clock_seconds: float = CANARY_MAX_WALL_CLOCK_SECONDS
    #: Whether this canary saves, stops, reloads and continues. It
    #: raises the epoch ceiling to two and nothing else.
    resume: bool = False

    def __post_init__(self) -> None:
        epoch_ceiling = CANARY_MAX_RESUME_EPOCHS if self.resume else CANARY_MAX_EPOCHS
        if not 1 <= self.max_epochs <= epoch_ceiling:
            raise CanaryBoundsError(
                f"a {'resume ' if self.resume else ''}canary may run 1..{epoch_ceiling} "
                f"epoch(s); {self.max_epochs} was asked for"
            )
        if not 1 <= self.max_samples <= CANARY_MAX_SAMPLES:
            raise CanaryBoundsError(
                f"a canary may use 1..{CANARY_MAX_SAMPLES} sample(s); "
                f"{self.max_samples} was asked for"
            )
        if not 1 <= self.max_optimizer_steps <= CANARY_MAX_OPTIMIZER_STEPS:
            raise CanaryBoundsError(
                f"a canary may take 1..{CANARY_MAX_OPTIMIZER_STEPS} optimizer step(s); "
                f"{self.max_optimizer_steps} was asked for"
            )
        if not 0 < self.wall_clock_seconds <= CANARY_ABSOLUTE_WALL_CLOCK_SECONDS:
            raise CanaryBoundsError(
                f"a canary may run for up to {CANARY_ABSOLUTE_WALL_CLOCK_SECONDS:.0f}s; "
                f"{self.wall_clock_seconds:.0f}s was asked for"
            )

    def upper_bound_steps(self, config: TrainingConfig) -> int:
        """The most optimizer steps this envelope could possibly produce.

        Computed from the *envelope*, not from what the trainer reports
        afterwards. A bound that could only be checked after the run
        would not be a bound.
        """
        per_epoch = math.ceil(self.max_samples / max(1, config.batch_size))
        return per_epoch * self.max_epochs

    def bound_config(self, config: TrainingConfig) -> TrainingConfig:
        """The config a canary actually runs, derived from *config*.

        Everything that decides length is overwritten rather than
        validated: a canary's epoch count is a property of being a
        canary, and accepting one from the caller would put the bound
        somewhere a caller could move it. What survives is what the
        canary is testing — the strategy, the adapter shape, the
        optimizer, the precision, the device.
        """
        bounded = config.with_overrides(
            epochs=self.max_epochs,
            checkpoint_every_epochs=1,
            warmup_steps=0,
            log_every_steps=1,
            log_heavy_every_steps=CANARY_MAX_OPTIMIZER_STEPS * 100,
            sample_every_n_epochs=0,
            # Worker processes buy nothing at four samples and cost a
            # fork per epoch; more usefully, num_workers=0 keeps the
            # whole run in one process where a timeout can reach it.
            num_workers=0,
            persistent_workers=False,
            prefetch_factor=0,
        )
        steps = self.upper_bound_steps(bounded)
        if steps > self.max_optimizer_steps:
            raise CanaryBoundsError(
                f"{self.max_samples} sample(s) at batch size {bounded.batch_size} for "
                f"{self.max_epochs} epoch(s) is up to {steps} optimizer step(s), and a "
                f"canary may take {self.max_optimizer_steps}"
            )
        return bounded

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_epochs": self.max_epochs,
            "max_samples": self.max_samples,
            "max_optimizer_steps": self.max_optimizer_steps,
            "wall_clock_seconds": self.wall_clock_seconds,
            "resume": self.resume,
            "ceilings": {
                "epochs": CANARY_MAX_EPOCHS,
                "resume_epochs": CANARY_MAX_RESUME_EPOCHS,
                "samples": CANARY_MAX_SAMPLES,
                "optimizer_steps": CANARY_MAX_OPTIMIZER_STEPS,
                "wall_clock_seconds": CANARY_ABSOLUTE_WALL_CLOCK_SECONDS,
            },
        }


def bound_plan(
    plan: TrainingPlan,
    envelope: CanaryEnvelope,
    *,
    dataset_dir: Path,
    output_dir: Path,
    model_dir: Path,
) -> TrainingPlan:
    """The plan a canary runs: the same intent, inside the envelope.

    A distinct plan with a distinct digest, deliberately. A canary is
    not the run — it trains for one epoch on four synthetic tensors —
    and giving it the run's identity would let its checkpoint be
    mistaken for the run's.
    """
    return replace(
        plan,
        config=envelope.bound_config(plan.config),
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        checkpoint_dir=str(model_dir),
    )


# ── dataset safety ───────────────────────────────────────────────────


@dataclass(frozen=True)
class DatasetVerdict:
    permitted: bool
    kind: str
    detail: str


def verify_canary_dataset(
    dataset_dir: Path,
    *,
    envelope: CanaryEnvelope,
    gate_report: GateReport | None = None,
) -> DatasetVerdict:
    """Whether a canary may train on this directory.

    Two ways to be permitted and no others.

    **Synthetic.** The directory carries upstream's fixture manifest
    marking it ``synthetic_test_fixtures``. Nothing in it came from a
    recording, so there is nothing to have rights in.

    **Gate-cleared.** Every Phase 25 gate passed for the material. The
    same gates as a full run, because a run being short is not a reason
    to train on something nobody is allowed to train on.
    """
    if not dataset_dir.is_dir():
        return DatasetVerdict(False, "ABSENT", f"{dataset_dir} does not exist")

    samples = sorted(dataset_dir.glob("*.pt"))
    if not samples:
        return DatasetVerdict(False, "EMPTY", f"{dataset_dir} contains no tensor samples")
    if len(samples) > envelope.max_samples:
        return DatasetVerdict(
            False,
            "OVER_BOUND",
            (
                f"{len(samples)} sample(s) present and the envelope allows "
                f"{envelope.max_samples}; a canary does not train on a directory bigger "
                "than its own bound"
            ),
        )

    manifest = dataset_dir / "manifest.json"
    if manifest.is_file():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            kind = ((payload.get("metadata") or {}).get("type")) or payload.get("type")
            if kind == SYNTHETIC_FIXTURE_TYPE:
                return DatasetVerdict(
                    True,
                    "SYNTHETIC",
                    (
                        f"{len(samples)} synthetic tensor sample(s); no audio, no "
                        "recording and nothing with rights in it"
                    ),
                )

    if gate_report is not None and gate_report.passed:
        return DatasetVerdict(
            True,
            "GATE_CLEARED",
            f"{len(samples)} sample(s) whose Phase 25 gates all passed",
        )

    return DatasetVerdict(
        False,
        "UNAUTHORISED",
        (
            "this directory is neither a synthetic fixture nor material whose rights "
            "gates passed. A canary obeys the same rights gates as a full run; being "
            "small is not an authorisation"
        ),
    )


def generate_synthetic_fixture(
    *,
    python_executable: str | Path,
    trainer_root: Path,
    destination: Path,
    envelope: CanaryEnvelope,
    timeout: float = 300.0,
) -> DatasetVerdict:
    """Build the canary's dataset with the trainer's own generator.

    Upstream ships `acestep.training_v2.make_test_fixtures` for exactly
    this: synthetic preprocessed tensors with the shapes the DiT expects
    and `is_synthetic` on every sample. Using it rather than writing our
    own means the shapes cannot drift from what the trainer reads, and
    it means LUBER never fabricates a training tensor.
    """
    destination.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            [
                str(python_executable),
                "-m",
                "acestep.training_v2.make_test_fixtures",
                "--output-dir",
                str(destination),
                "--num-samples",
                str(envelope.max_samples),
                "--latent-length",
                str(CANARY_LATENT_LENGTH),
                "--encoder-length",
                str(CANARY_ENCODER_LENGTH),
            ],
            cwd=str(trainer_root),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return DatasetVerdict(False, "GENERATOR_FAILED", f"{type(exc).__name__}: {exc}")
    if completed.returncode != 0:
        tail = (completed.stderr or "").strip().splitlines()
        return DatasetVerdict(
            False,
            "GENERATOR_FAILED",
            f"the fixture generator exited {completed.returncode}"
            + (f": {tail[-1]}" if tail else ""),
        )
    return verify_canary_dataset(destination, envelope=envelope)


# ── checkpoint integrity ─────────────────────────────────────────────


@dataclass
class CheckpointIntegrity:
    """What was found in a checkpoint, and whether it is usable."""

    path: str
    exists: bool = False
    file_count: int = 0
    size_bytes: int = 0
    reopened: bool | None = None
    tensor_count: int | None = None
    non_zero_parameters: int | None = None
    step: int | None = None
    epoch: int | None = None
    has_optimizer_state: bool | None = None
    provenance_present: bool = False
    provenance_plan_digest: str | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.exists and self.reopened is True and not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "exists": self.exists,
            "file_count": self.file_count,
            "size_bytes": self.size_bytes,
            "reopened": self.reopened,
            "tensor_count": self.tensor_count,
            "non_zero_parameters": self.non_zero_parameters,
            "step": self.step,
            "epoch": self.epoch,
            "has_optimizer_state": self.has_optimizer_state,
            "provenance_present": self.provenance_present,
            "provenance_plan_digest": self.provenance_plan_digest,
            "problems": list(self.problems),
            "ok": self.ok,
        }


def write_provenance(
    checkpoint_dir: Path,
    *,
    plan: TrainingPlan,
    envelope: CanaryEnvelope,
    mode: str,
    execution_location: str,
    execution_device: str | None,
    resolved_precision: str | None,
    dataset_kind: str,
    steps: int | None,
) -> Path:
    """Record what produced this checkpoint, beside the checkpoint.

    The trainer writes an adapter and a training state and knows nothing
    about plans, devices or rights. Without this file a checkpoint found
    on disk a month later is a directory of tensors whose origin is a
    guess — and the one thing that must never be guessed about a
    checkpoint is what it was trained on.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CANARY_SCHEMA_VERSION,
        "produced_by": "LUBER canary",
        "canary_mode": mode,
        "warning": (
            "This checkpoint was produced by a bounded canary. It exists to prove that "
            "the mechanism works and has learned nothing worth keeping. It must never be "
            "promoted, evaluated as a model, or shipped."
        ),
        "run_id": plan.run_id,
        "experiment_id": plan.experiment_id,
        "plan_id": plan.plan_id,
        "plan_digest": plan.digest(),
        "config_sha256": plan.config.digest(),
        "base_model_id": plan.base_model_id,
        "base_model_upstream_commit": plan.base_model_upstream_commit,
        "ace_step_commit": plan.config.ace_step_commit,
        "execution_location": execution_location,
        "execution_device": execution_device,
        "resolved_precision": resolved_precision,
        "optimizer": plan.config.optimizer_type,
        "strategy": plan.config.strategy,
        "dataset": {
            "kind": dataset_kind,
            "dataset_id": plan.dataset_ref.dataset_id,
            "curation_id": plan.dataset_ref.curation_id,
            "curated_manifest_sha256": plan.dataset_ref.curated_manifest_sha256,
        },
        "envelope": envelope.to_dict(),
        "optimizer_steps": steps,
        "written_at": _now(),
    }
    path = checkpoint_dir / PROVENANCE_NAME
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def inspect_checkpoint(
    checkpoint_dir: Path,
    *,
    python_executable: str | Path | None = None,
    expected_plan_digest: str | None = None,
    timeout: float = 300.0,
) -> CheckpointIntegrity:
    """Open a checkpoint and report what is really in it.

    Without an interpreter that has torch this reports the structural
    facts — the directory, its files, its provenance — and leaves
    ``reopened`` as ``None``. That is not a pass: a caller asking
    whether a checkpoint can be reopened gets "nobody could try",
    which is what UNVERIFIED means everywhere else in this phase.
    """
    integrity = CheckpointIntegrity(path=str(checkpoint_dir), exists=checkpoint_dir.is_dir())
    if not integrity.exists:
        integrity.problems.append("the checkpoint directory does not exist")
        return integrity

    files = [path for path in checkpoint_dir.rglob("*") if path.is_file()]
    integrity.file_count = len(files)
    integrity.size_bytes = sum(path.stat().st_size for path in files)
    if integrity.size_bytes == 0:
        integrity.problems.append("the checkpoint directory holds no bytes")

    provenance_path = checkpoint_dir / PROVENANCE_NAME
    integrity.provenance_present = provenance_path.is_file()
    if integrity.provenance_present:
        try:
            recorded = json.loads(provenance_path.read_text(encoding="utf-8"))
            integrity.provenance_plan_digest = recorded.get("plan_digest")
        except (OSError, json.JSONDecodeError):
            integrity.problems.append("the provenance record is unreadable")
    else:
        integrity.problems.append("no provenance record was written beside this checkpoint")

    if (
        expected_plan_digest is not None
        and integrity.provenance_plan_digest is not None
        and integrity.provenance_plan_digest != expected_plan_digest
    ):
        integrity.problems.append(
            f"the provenance cites plan {integrity.provenance_plan_digest[:12]} and this "
            f"checkpoint was expected from {expected_plan_digest[:12]}"
        )

    if python_executable is None:
        return integrity

    script = Path(_checkpoint_probe.__file__).resolve()
    try:
        completed = subprocess.run(
            [str(python_executable), str(script), str(checkpoint_dir)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        integrity.problems.append(f"the checkpoint could not be opened: {exc}")
        return integrity

    payload: dict[str, Any] | None = None
    for line in reversed((completed.stdout or "").strip().splitlines()):
        try:
            candidate = json.loads(line)
        except ValueError:
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
    if payload is None:
        integrity.problems.append("the checkpoint probe printed no JSON document")
        return integrity

    adapter = payload.get("adapter") or {}
    state = payload.get("training_state") or {}
    integrity.reopened = bool(adapter.get("reopened"))
    integrity.tensor_count = adapter.get("tensor_count")
    integrity.non_zero_parameters = adapter.get("non_zero_parameters")
    integrity.step = state.get("global_step")
    integrity.epoch = state.get("epoch")
    integrity.has_optimizer_state = state.get("has_optimizer_state")

    if not integrity.reopened:
        integrity.problems.append(
            f"the adapter could not be reopened: {adapter.get('error', 'no reason reported')}"
        )
    if adapter.get("all_zero"):
        integrity.problems.append(
            "every adapter tensor is zero, so nothing was learned and the checkpoint is "
            "not evidence that a step ran"
        )
    if state.get("present") and not state.get("reopened"):
        integrity.problems.append(
            f"the training state could not be reopened: {state.get('error', 'no reason')}"
        )
    return integrity


def latest_checkpoint(output_dir: Path) -> Path | None:
    """The newest per-epoch checkpoint the trainer wrote.

    Under ``output_dir/checkpoints``, which is where the installed
    trainer puts them — *not* the directory `--checkpoint-dir` names,
    which is the root it reads base weights from.
    """
    directory = output_dir / TRAINER_CHECKPOINTS_SUBDIR
    if not directory.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in directory.iterdir():
        if not path.is_dir():
            continue
        match = _CHECKPOINT_NAME.match(path.name)
        candidates.append((int(match.group(1)) if match else 0, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1].name))
    return candidates[-1][1]


# ── the canaries ─────────────────────────────────────────────────────


@dataclass
class CanaryResult:
    """One canary, its bounds, and what it established."""

    mode: str
    status: str
    detail: str = ""
    plan_digest: str | None = None
    execution_location: str | None = None
    execution_device: str | None = None
    resolved_precision: str | None = None
    optimizer: str | None = None
    dataset_kind: str | None = None
    envelope: dict[str, Any] = field(default_factory=dict)
    command: list[str] = field(default_factory=list)
    command_display: str = ""
    exit_code: int | None = None
    seconds: float | None = None
    steps: int | None = None
    checkpoint: dict[str, Any] | None = None
    resume: dict[str, Any] | None = None
    log_tail: list[str] = field(default_factory=list)
    started_at: str = field(default_factory=_now)
    schema_version: str = CANARY_SCHEMA_VERSION

    @property
    def passed(self) -> bool:
        return self.status == CanaryStatus.PASSED.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "status": self.status,
            "detail": self.detail,
            "plan_digest": self.plan_digest,
            "execution_location": self.execution_location,
            "execution_device": self.execution_device,
            "resolved_precision": self.resolved_precision,
            "optimizer": self.optimizer,
            "dataset_kind": self.dataset_kind,
            "envelope": self.envelope,
            "command": list(self.command),
            "command_display": self.command_display,
            "exit_code": self.exit_code,
            "seconds": self.seconds,
            "steps": self.steps,
            "checkpoint": self.checkpoint,
            "resume": self.resume,
            "log_tail": list(self.log_tail),
            "started_at": self.started_at,
            "note": (
                "A canary proves the mechanism, never the model. Any checkpoint it "
                "produced is worthless as a model and must not be promoted."
            ),
        }

    def as_evidence(self) -> CanaryEvidence:
        """This result in the shape the preflight reads."""
        checkpoint = self.checkpoint or {}
        resume = self.resume or {}
        return CanaryEvidence(
            status=self.status,
            mode=self.mode,
            detail=self.detail,
            steps=self.steps,
            checkpoint_ok=checkpoint.get("ok") if self.checkpoint is not None else None,
            checkpoint_detail=(
                "; ".join(checkpoint.get("problems") or [])
                or f"reopened with {checkpoint.get('tensor_count')} tensor(s), "
                f"step {checkpoint.get('step')}"
                if self.checkpoint is not None
                else ""
            ),
            resume_ok=resume.get("ok") if self.resume is not None else None,
            resume_detail=str(resume.get("detail", "")),
        )


def orchestration_canary(
    plan: TrainingPlan,
    envelope: CanaryEnvelope,
    *,
    trainer_root: str,
    python_executable: str = "python",
    execution_location: str,
    resolved_precision: str | None = None,
    model_dir: str | None = None,
) -> CanaryResult:
    """Prove LUBER's half, and be explicit that nothing was trained.

    The bounded plan compiles, the envelope holds, and the command the
    trainer would receive is produced and recorded. No process is
    started. This is what stays available when the trainer lives on a
    machine the control plane cannot reach, and it is deliberately not
    evidence about ACE-Step.
    """
    bounded = replace(plan, config=envelope.bound_config(plan.config))
    command = compile_command(
        bounded,
        trainer_root=trainer_root,
        python_executable=python_executable,
        model_dir=model_dir,
    )
    return CanaryResult(
        mode=CanaryMode.ORCHESTRATION.value,
        status=CanaryStatus.PASSED.value,
        detail=(
            "the bounded plan compiled inside its envelope and produced a trainer "
            "invocation. Nothing was executed and no model exists as a result"
        ),
        plan_digest=bounded.digest(),
        execution_location=execution_location,
        execution_device=bounded.requirements.execution_device,
        resolved_precision=resolved_precision,
        optimizer=bounded.config.optimizer_type,
        dataset_kind="NOT_READ",
        envelope=envelope.to_dict(),
        command=list(command.argv),
        command_display=command.display(),
        steps=0,
    )


def _launch(
    command: CompiledCommand,
    *,
    envelope: CanaryEnvelope,
    log_path: Path,
) -> tuple[int | None, float, list[str], bool]:
    """Run a bounded trainer invocation under a wall clock.

    stdin is closed, exactly as a detached launch has it, so the run
    proves the same thing a real dispatch would face. The timeout is the
    envelope's and is enforced by killing the process group: a canary
    that could outlive its bound would not be bounded.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    timed_out = False
    with log_path.open("wb") as handle:
        process = subprocess.Popen(
            command.argv,
            cwd=command.working_directory,
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        try:
            exit_code: int | None = process.wait(timeout=envelope.wall_clock_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(os.getpgid(process.pid), 15)
            except OSError:
                process.terminate()
            try:
                exit_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(process.pid), 9)
                except OSError:
                    process.kill()
                exit_code = process.wait(timeout=30)
    seconds = time.perf_counter() - started
    tail: list[str] = []
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = [line for line in lines[-40:] if line.strip()]
    except OSError:
        tail = []
    return exit_code, seconds, tail, timed_out


_STEP_LINE = re.compile(r"Step\s+(\d+)")


def _steps_from(tail: list[str]) -> int | None:
    steps = [int(match.group(1)) for line in tail if (match := _STEP_LINE.search(line))]
    return max(steps) if steps else None


def ace_step_canary(
    plan: TrainingPlan,
    envelope: CanaryEnvelope,
    *,
    trainer_root: Path,
    python_executable: str | Path,
    model_dir: Path,
    workspace: Path,
    execution_location: str,
    resolved_precision: str | None = None,
    gate_report: GateReport | None = None,
    dataset_dir: Path | None = None,
    model_variant: str = "turbo",
) -> CanaryResult:
    """The real trainer, on synthetic tensors, inside the envelope.

    ``workspace`` is where the canary lives — its fixture, its output,
    its logs. It is never the run's own directory: a canary's checkpoint
    is not the run's checkpoint and must not appear where one is looked
    for.

    ``dataset_dir`` lets a caller supply gate-cleared material instead of
    the synthetic fixture. It is checked by the same
    :func:`verify_canary_dataset`, which refuses anything that is
    neither synthetic nor gate-cleared.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    fixture_dir = dataset_dir or (workspace / "dataset")
    output_dir = workspace / "output"
    logs_dir = workspace / "logs"

    result = CanaryResult(
        mode=CanaryMode.ACE_STEP.value,
        status=CanaryStatus.BLOCKED.value,
        execution_location=execution_location,
        execution_device=plan.requirements.execution_device,
        resolved_precision=resolved_precision,
        optimizer=plan.config.optimizer_type,
        envelope=envelope.to_dict(),
    )

    if not Path(trainer_root).is_dir():
        result.detail = f"no trainer is installed at {trainer_root}"
        return result
    if not Path(python_executable).is_file():
        result.detail = f"no interpreter at {python_executable}"
        return result
    if not Path(model_dir).is_dir():
        result.detail = (
            f"the base model root {model_dir} does not exist. A canary never downloads "
            "weights: it runs against what is already installed or it does not run"
        )
        return result
    if not within(fixture_dir, Path(trainer_root)):
        result.detail = (
            f"the canary dataset at {fixture_dir} is outside the trainer's working "
            f"directory ({trainer_root}). ACE-Step validates --dataset-dir against the "
            "working directory at import time and refuses anything outside it, after "
            "loading the model. Put the dataset beneath the trainer root"
        )
        return result

    if dataset_dir is None:
        verdict = generate_synthetic_fixture(
            python_executable=python_executable,
            trainer_root=Path(trainer_root),
            destination=fixture_dir,
            envelope=envelope,
        )
    else:
        verdict = verify_canary_dataset(fixture_dir, envelope=envelope, gate_report=gate_report)
    result.dataset_kind = verdict.kind
    if not verdict.permitted:
        result.detail = verdict.detail
        return result

    bounded = bound_plan(
        plan, envelope, dataset_dir=fixture_dir, output_dir=output_dir, model_dir=Path(model_dir)
    )
    result.plan_digest = bounded.digest()
    command = compile_command(
        bounded,
        trainer_root=str(trainer_root),
        python_executable=str(python_executable),
        model_variant=model_variant,
    )
    result.command = list(command.argv)
    result.command_display = command.display()

    exit_code, seconds, tail, timed_out = _launch(
        command, envelope=envelope, log_path=logs_dir / "canary.log"
    )
    result.exit_code = exit_code
    result.seconds = round(seconds, 3)
    result.log_tail = tail
    result.steps = _steps_from(tail)

    if timed_out:
        result.status = CanaryStatus.FAILED.value
        result.detail = (
            f"the canary exceeded its {envelope.wall_clock_seconds:.0f}s wall clock and was "
            "stopped. Nothing about the trainer is established by a run that was killed"
        )
        return result
    if exit_code != 0:
        result.status = CanaryStatus.FAILED.value
        result.detail = f"the trainer exited {exit_code}"
        return result

    checkpoint_dir = latest_checkpoint(output_dir)
    if checkpoint_dir is None:
        result.status = CanaryStatus.FAILED.value
        result.detail = (
            "the trainer exited 0 and wrote no checkpoint. An exit code of zero is not "
            "evidence that training happened — the installed trainer returns 0 when its "
            "confirmation prompt is declined"
        )
        return result

    write_provenance(
        checkpoint_dir,
        plan=bounded,
        envelope=envelope,
        mode=CanaryMode.ACE_STEP.value,
        execution_location=execution_location,
        execution_device=bounded.requirements.execution_device,
        resolved_precision=resolved_precision,
        dataset_kind=verdict.kind,
        steps=result.steps,
    )
    integrity = inspect_checkpoint(
        checkpoint_dir,
        python_executable=python_executable,
        expected_plan_digest=bounded.digest(),
    )
    result.checkpoint = integrity.to_dict()
    if not integrity.ok:
        result.status = CanaryStatus.FAILED.value
        result.detail = "the checkpoint is not usable: " + "; ".join(integrity.problems)
        return result

    if envelope.resume:
        result.resume = _resume_leg(
            bounded,
            envelope,
            trainer_root=Path(trainer_root),
            python_executable=python_executable,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            logs_dir=logs_dir,
            model_variant=model_variant,
            first_step=integrity.step,
        )
        if not result.resume.get("ok"):
            result.status = CanaryStatus.FAILED.value
            result.detail = f"resume failed: {result.resume.get('detail')}"
            return result

    result.status = CanaryStatus.PASSED.value
    result.detail = (
        f"the installed trainer loaded the model, took up to "
        f"{envelope.upper_bound_steps(bounded.config)} optimizer step(s) on "
        f"{envelope.max_samples} synthetic sample(s) and wrote a checkpoint that reopens. "
        "This proves the mechanism and nothing about the model"
    )
    return result


def _resume_leg(
    bounded: TrainingPlan,
    envelope: CanaryEnvelope,
    *,
    trainer_root: Path,
    python_executable: str | Path,
    checkpoint_dir: Path,
    output_dir: Path,
    logs_dir: Path,
    model_variant: str,
    first_step: int | None,
) -> dict[str, Any]:
    """Save, stop, reload, continue — and check that it continued.

    Deserialising a file is not resumability. What makes a resume real
    is that the optimizer state came back, the step counter carried on
    from where it stopped, and another step ran. So the second leg runs
    one more epoch and the step count is compared.
    """
    second = replace(
        bounded,
        config=bounded.config.with_overrides(epochs=CANARY_MAX_RESUME_EPOCHS),
    )
    command = compile_command(
        second,
        trainer_root=str(trainer_root),
        python_executable=str(python_executable),
        model_variant=model_variant,
        resume_from=str(checkpoint_dir),
    )
    exit_code, seconds, tail, timed_out = _launch(
        command, envelope=envelope, log_path=logs_dir / "canary-resume.log"
    )
    outcome: dict[str, Any] = {
        "ok": False,
        "exit_code": exit_code,
        "seconds": round(seconds, 3),
        "resumed_from": checkpoint_dir.name,
        "first_step": first_step,
        "log_tail": tail[-20:],
        "command_display": command.display(),
    }
    if timed_out:
        outcome["detail"] = f"the resume leg exceeded {envelope.wall_clock_seconds:.0f}s"
        return outcome
    if exit_code != 0:
        outcome["detail"] = f"the trainer exited {exit_code} on resume"
        return outcome

    latest = latest_checkpoint(output_dir)
    if latest is None:
        outcome["detail"] = "the resumed run wrote no checkpoint"
        return outcome
    integrity = inspect_checkpoint(latest, python_executable=python_executable)
    outcome["second_checkpoint"] = latest.name
    outcome["second_step"] = integrity.step
    outcome["second_epoch"] = integrity.epoch
    if integrity.reopened is not True:
        outcome["detail"] = "the checkpoint written after resume could not be reopened"
        return outcome
    if first_step is None or integrity.step is None:
        outcome["detail"] = (
            "the step counter could not be read on one side of the resume, so whether "
            "training continued rather than restarted is unverified"
        )
        return outcome
    if integrity.step <= first_step:
        outcome["detail"] = (
            f"the step counter did not advance ({first_step} -> {integrity.step}); the "
            "run restarted rather than resumed"
        )
        return outcome
    outcome["ok"] = True
    outcome["detail"] = (
        f"training continued from step {first_step} to {integrity.step} after reloading "
        f"{checkpoint_dir.name}, with optimizer state restored"
    )
    return outcome


def cleanup_workspace(workspace: Path) -> None:
    """Remove a canary's workspace, weights and all.

    A canary's output is a checkpoint that must never be promoted and a
    dataset of noise. Leaving them on disk is how one of them ends up
    committed, evaluated, or mistaken for a run's artifact.
    """
    shutil.rmtree(workspace, ignore_errors=True)


__all__ = [
    "CANARY_ABSOLUTE_WALL_CLOCK_SECONDS",
    "CANARY_ENCODER_LENGTH",
    "CANARY_LATENT_LENGTH",
    "CANARY_MAX_EPOCHS",
    "CANARY_MAX_OPTIMIZER_STEPS",
    "CANARY_MAX_RESUME_EPOCHS",
    "CANARY_MAX_SAMPLES",
    "CANARY_MAX_WALL_CLOCK_SECONDS",
    "CANARY_SCHEMA_VERSION",
    "CANARY_TRAINER_SUBDIR",
    "PROVENANCE_NAME",
    "SYNTHETIC_FIXTURE_TYPE",
    "TRAINER_CHECKPOINTS_SUBDIR",
    "TRAINER_FINAL_SUBDIR",
    "CanaryBoundsError",
    "CanaryEnvelope",
    "CanaryMode",
    "CanaryResult",
    "CanaryStatus",
    "CheckpointIntegrity",
    "DatasetVerdict",
    "ace_step_canary",
    "bound_plan",
    "cleanup_workspace",
    "default_workspace",
    "generate_synthetic_fixture",
    "inspect_checkpoint",
    "latest_checkpoint",
    "orchestration_canary",
    "verify_canary_dataset",
    "within",
    "write_provenance",
]
