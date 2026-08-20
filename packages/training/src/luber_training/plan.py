"""The immutable training plan, and the environment it was built in.

A plan is everything a worker needs and nothing it must not have. It is
compiled once, hashed, and thereafter frozen: if any input changes, the
result is a *new run*, not an edited plan. That rule is what makes a
finished run explicable a year later — the plan hash names exactly one
combination of model, data, config and code.

Two things the plan deliberately does not contain.

**No secrets.** Not an SSH key, not a token, not a password. Where a
credential will eventually be needed the plan carries a *reference
name*, and the backend resolves it out of band. A plan is written to
disk, copied to a rented host and kept forever; it is the last place a
secret should be.

**No machine-specific absolute paths as identity.** The plan names
dataset and curation *identities and digests*. Where a path is
unavoidable it is a placeholder the backend substitutes, because a plan
built on a Mac has to execute on a Linux box.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from luber_training.config import TrainingConfig
from luber_training.entities import TrainingDatasetRef

TRAINING_PLAN_SCHEMA_VERSION = "luber-training-plan/1"

#: Placeholders a backend substitutes with real paths on the worker.
#: Written this way so a plan can be read, diffed and hashed without
#: containing anything about the machine that compiled it.
DATASET_DIR_PLACEHOLDER = "${LUBER_DATASET_DIR}"
OUTPUT_DIR_PLACEHOLDER = "${LUBER_OUTPUT_DIR}"
CHECKPOINT_DIR_PLACEHOLDER = "${LUBER_CHECKPOINT_DIR}"

#: Keys excluded from the plan hash: real information, none of it about
#: *what* is trained.
NON_CANONICAL_KEYS: frozenset[str] = frozenset({"compiled_at", "compiled_on_host"})


@dataclass
class HardwareRequirements:
    """What a plan needs from a worker.

    Every field may be ``UNKNOWN_REQUIREMENT``. Nothing in this project
    has measured VRAM for any configuration, and inventing "24 GB"
    because it sounds right is how a run gets scheduled onto hardware
    that cannot hold it. An unknown requirement is matched permissively
    and reported as unknown in the preflight.
    """

    requires_cuda: bool = True
    minimum_vram_mb: int | None = None
    minimum_gpu_count: int = 1
    supported_precision: tuple[str, ...] = ("bf16", "fp16", "fp32")
    #: Why a requirement is absent, when it is.
    unknown_requirements: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["supported_precision"] = list(self.supported_precision)
        payload["unknown_requirements"] = list(self.unknown_requirements)
        return payload


def default_requirements(config: TrainingConfig) -> HardwareRequirements:
    """Requirements derivable from a config, and no more.

    GPU count comes from ``num_devices`` because that is a fact about
    the config. VRAM does not, because nothing has measured it — so it
    is recorded as an explicit unknown rather than guessed.
    """
    return HardwareRequirements(
        requires_cuda=True,
        minimum_vram_mb=None,
        minimum_gpu_count=max(1, config.num_devices),
        supported_precision=("bf16", "fp16", "fp32"),
        unknown_requirements=(
            "minimum_vram_mb: no VRAM figure has been measured for any LUBER "
            "configuration on NVIDIA hardware",
        ),
    )


@dataclass
class EnvironmentLock:
    """The software the run executed against.

    Captured on the machine that runs the training, not on the one that
    compiled the plan. Everything is a version string or a commit; no
    environment variable values are read, so nothing here can leak a
    credential.
    """

    python_version: str = ""
    platform_name: str = ""
    torch_version: str | None = None
    cuda_version: str | None = None
    peft_version: str | None = None
    transformers_version: str | None = None
    ace_step_commit: str | None = None
    luber_commit: str | None = None
    luber_dirty: bool | None = None
    dependency_lock_sha256: str | None = None
    captured_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


@dataclass
class CodeVersion:
    """Which revision of LUBER is about to train.

    A dirty tree cannot be identified later: "commit abc123 plus
    whatever was in the editor" is not a revision anyone can reproduce.
    Production training therefore requires a clean tree, and the check
    is a gate rather than a warning.
    """

    commit: str | None
    dirty: bool
    branch: str | None = None

    @property
    def identifiable(self) -> bool:
        return self.commit is not None and not self.dirty

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def capture_code_version(repository_root: Path) -> CodeVersion:
    commit = _git(["rev-parse", "HEAD"], repository_root)
    status = _git(["status", "--porcelain"], repository_root)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repository_root)
    return CodeVersion(
        commit=commit,
        dirty=bool(status) if status is not None else True,
        branch=branch,
    )


def capture_environment(
    repository_root: Path, *, ace_step_commit: str | None = None
) -> EnvironmentLock:
    """Record the software environment. Reads versions, never values."""
    code = capture_code_version(repository_root)
    lock_file = repository_root / "uv.lock"
    dependency_digest = None
    if lock_file.is_file():
        dependency_digest = hashlib.sha256(lock_file.read_bytes()).hexdigest()

    torch_version = _package_version("torch")
    cuda_version: str | None = None
    if torch_version is not None:
        try:
            # Imported lazily and defensively: this package never
            # requires torch, and on an orchestration host it is
            # normally absent.
            import torch  # type: ignore[import-not-found]

            cuda_version = getattr(torch.version, "cuda", None)
        except Exception:
            cuda_version = None

    return EnvironmentLock(
        python_version=sys.version.split()[0],
        platform_name=f"{platform.system()} {platform.machine()}",
        torch_version=torch_version,
        cuda_version=cuda_version,
        peft_version=_package_version("peft"),
        transformers_version=_package_version("transformers"),
        ace_step_commit=ace_step_commit,
        luber_commit=code.commit,
        luber_dirty=code.dirty,
        dependency_lock_sha256=dependency_digest,
    )


@dataclass
class TrainingPlan:
    """Everything a worker needs to execute one run, frozen."""

    plan_id: str
    run_id: str
    experiment_id: str
    base_model_id: str
    base_model_upstream_commit: str
    dataset_ref: TrainingDatasetRef
    config: TrainingConfig
    execution_backend: str
    requirements: HardwareRequirements
    #: Placeholders the backend resolves on the worker.
    dataset_dir: str = DATASET_DIR_PLACEHOLDER
    output_dir: str = OUTPUT_DIR_PLACEHOLDER
    checkpoint_dir: str = CHECKPOINT_DIR_PLACEHOLDER
    resume_from_checkpoint_id: str | None = None
    #: Names of credentials the backend will need. Never values.
    secret_refs: tuple[str, ...] = ()
    schema_version: str = TRAINING_PLAN_SCHEMA_VERSION
    compiled_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "base_model_id": self.base_model_id,
            "base_model_upstream_commit": self.base_model_upstream_commit,
            "dataset_ref": self.dataset_ref.to_dict(),
            "config": self.config.to_dict(),
            "config_sha256": self.config.digest(),
            "execution_backend": self.execution_backend,
            "requirements": self.requirements.to_dict(),
            "dataset_dir": self.dataset_dir,
            "output_dir": self.output_dir,
            "checkpoint_dir": self.checkpoint_dir,
            "resume_from_checkpoint_id": self.resume_from_checkpoint_id,
            "secret_refs": list(self.secret_refs),
            "compiled_at": self.compiled_at,
        }

    def canonical_dict(self) -> dict[str, Any]:
        """The plan with compile-time noise removed.

        ``plan_id`` is excluded as well as the timestamp: two plans that
        would train identically must hash identically, and a fresh
        random id on each compile would defeat that entirely.
        """
        payload = {
            key: value
            for key, value in self.to_dict().items()
            if key not in NON_CANONICAL_KEYS and key != "plan_id"
        }
        return payload

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.canonical_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()


class PlanError(RuntimeError):
    """Raised when a plan cannot be compiled as requested."""
