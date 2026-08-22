"""Going and looking: the I/O half of the training preflight.

:mod:`luber_training.preflight` decides; this collects. The split is
what makes the decision deterministic and testable — a check that both
read the filesystem and judged what it found could only be tested by
building a filesystem, and the CUDA branches could only be tested by
owning a GPU.

Everything here returns evidence dataclasses with ``None`` for anything
it could not establish. Nothing here guesses, and nothing here treats
"the directory was not visible from this machine" as "the directory is
fine".
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from luber_hardware import ExecutionLocation
from luber_training import _trainer_probe
from luber_training.entities import TrainingWorker
from luber_training.gates import selected_records
from luber_training.plan import TrainingPlan
from luber_training.preflight import (
    OPTIMIZER_PACKAGES,
    DatasetEvidence,
    RemoteEvidence,
    StorageEvidence,
    TrainerEvidence,
)

#: How long the trainer's interpreter has to answer. Importing ACE-Step
#: pulls in torch and transformers; on a cold cache that is tens of
#: seconds, not milliseconds. Long enough to be fair, short enough that
#: an operator is not left staring at a prompt.
TRAINER_PROBE_TIMEOUT_SECONDS = 300.0


def _now() -> datetime:
    return datetime.now(UTC)


# ── dataset ──────────────────────────────────────────────────────────


def _curated_digest(records: list[dict[str, Any]]) -> str:
    """The digest Phase 24's lock records, recomputed the same way."""
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item.get("track_id"))):
        digest.update(
            json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def collect_dataset_evidence(
    plan: TrainingPlan,
    *,
    curated_manifest_path: Path | None,
    locks_verified: bool | None = None,
    check_source_files: bool = False,
) -> DatasetEvidence:
    """What is true of the data this plan cites, right now.

    ``check_source_files`` is off by default and that is deliberate: the
    manifest holds the paths of the machine that built it, and a control
    plane that will dispatch this run to a rented Linux box cannot see
    them. Reporting them missing from here would be a false refusal;
    the worker's own preflight is what establishes that the files
    arrived.
    """
    if curated_manifest_path is None or not curated_manifest_path.is_file():
        return DatasetEvidence(
            manifest_present=False if curated_manifest_path is not None else None,
            locks_verified=locks_verified,
            detail=(
                "no curated manifest path was supplied to this preflight"
                if curated_manifest_path is None
                else f"the curated manifest is not at {curated_manifest_path.name}"
            ),
        )

    records: list[dict[str, Any]] = []
    for number, line in enumerate(
        curated_manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            return DatasetEvidence(
                manifest_present=True,
                locks_verified=locks_verified,
                detail=f"the curated manifest is not valid JSON at line {number}",
            )

    selected = selected_records(records)
    evaluation_only = sum(1 for item in selected if str(item.get("split", "TRAIN")) != "TRAIN")

    missing: int | None = None
    present: bool | None = None
    if check_source_files:
        missing = 0
        for item in selected:
            source = item.get("source") or {}
            path = source.get("source_path")
            if not isinstance(path, str) or not Path(path).is_file():
                missing += 1
        present = missing == 0

    return DatasetEvidence(
        manifest_present=True,
        locks_verified=locks_verified,
        eligible_sample_count=len(selected),
        evaluation_only_count=evaluation_only,
        observed_manifest_sha256=_curated_digest(records),
        referenced_files_present=present,
        missing_file_count=missing,
        detail=(
            f"{len(selected)} selected of {len(records)} curated record(s) in "
            f"{curated_manifest_path.name}"
        ),
    )


def synthetic_dataset_evidence(
    plan: TrainingPlan, dataset_dir: Path, *, sample_glob: str = "*.pt"
) -> DatasetEvidence:
    """Evidence for a canary's own synthetic tensor fixture.

    A canary does not train on the run's dataset — it trains on a
    generated fixture with no audio in it — so the manifest questions do
    not apply and are answered as such rather than left blank. The drift
    digest is reported as the plan's own so a canary is never mistaken
    for evidence about the real selection.
    """
    samples = sorted(dataset_dir.glob(sample_glob)) if dataset_dir.is_dir() else []
    return DatasetEvidence(
        manifest_present=dataset_dir.is_dir(),
        locks_verified=True,
        eligible_sample_count=len(samples),
        evaluation_only_count=0,
        observed_manifest_sha256=plan.dataset_ref.curated_manifest_sha256,
        referenced_files_present=bool(samples),
        missing_file_count=0 if samples else None,
        detail=(
            f"{len(samples)} synthetic tensor sample(s) in {dataset_dir.name}; this is a "
            "canary fixture and contains no audio and no rights-bearing material"
        ),
    )


# ── trainer ──────────────────────────────────────────────────────────


def ace_step_commit(trainer_root: Path | None) -> str | None:
    """The revision installed at *trainer_root*, or None."""
    if trainer_root is None or not Path(trainer_root).is_dir():
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(trainer_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (result.stdout.strip() or None) if result.returncode == 0 else None


def probe_trainer(
    *,
    python_executable: str | Path,
    trainer_root: Path,
    argv: list[str],
    packages: list[str],
    timeout: float = TRAINER_PROBE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run the trainer probe under the trainer's interpreter.

    The probe file itself is executed rather than a copy of its source
    being sent over: one implementation, no string of Python to keep in
    step with it.
    """
    script = Path(_trainer_probe.__file__).resolve()
    request = json.dumps({"argv": argv, "packages": packages})
    try:
        completed = subprocess.run(
            [str(python_executable), str(script)],
            cwd=str(trainer_root),
            input=request,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        return {"error": f"no interpreter at {python_executable}: {exc}"}
    except subprocess.TimeoutExpired:
        return {"error": f"{python_executable} did not answer within {timeout:.0f}s"}
    except OSError as exc:
        return {"error": f"could not run {python_executable}: {exc}"}

    if completed.returncode != 0:
        tail = (completed.stderr or "").strip().splitlines()
        return {
            "error": f"the trainer probe exited {completed.returncode}"
            + (f": {tail[-1]}" if tail else "")
        }
    # The trainer's own imports log to stdout on some builds, so the
    # JSON document is the last line rather than the whole stream.
    for line in reversed((completed.stdout or "").strip().splitlines()):
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict):
            return payload
    return {"error": "the trainer probe printed no JSON document"}


def collect_trainer_evidence(
    plan: TrainingPlan,
    *,
    trainer_root: Path | None,
    python_executable: str | Path | None,
    argv: list[str] | None = None,
    timeout: float = TRAINER_PROBE_TIMEOUT_SECONDS,
) -> TrainerEvidence:
    """Everything establishable about the trainer, from this machine.

    Where the trainer is on another host this returns UNKNOWNs, which is
    the truth: the control plane cannot import a package on a rented
    box. Phase 27's worker-side preflight answers the same questions
    where it can actually see them.
    """
    root_present = None if trainer_root is None else Path(trainer_root).is_dir()
    entry_present = (
        None
        if trainer_root is None or not root_present
        else (Path(trainer_root) / "train.py").is_file()
    )
    interpreter_present = (
        None if python_executable is None else Path(str(python_executable)).is_file()
    )
    commit = ace_step_commit(Path(trainer_root) if trainer_root is not None else None)
    required = OPTIMIZER_PACKAGES.get(plan.config.optimizer_type)
    packages = [required] if required else []

    if not (root_present and interpreter_present):
        return TrainerEvidence(
            trainer_root_present=root_present,
            entrypoint_present=entry_present,
            interpreter_present=interpreter_present,
            observed_ace_step_commit=commit,
            unchecked_packages=tuple(packages),
            detail=(
                "the trainer or its interpreter is not on this machine, so nothing about "
                "its runtime could be established from here"
            ),
        )

    assert trainer_root is not None and python_executable is not None
    result = probe_trainer(
        python_executable=python_executable,
        trainer_root=Path(trainer_root),
        argv=argv or [],
        packages=packages,
        timeout=timeout,
    )
    if "error" in result:
        return TrainerEvidence(
            trainer_root_present=root_present,
            entrypoint_present=entry_present,
            interpreter_present=interpreter_present,
            observed_ace_step_commit=commit,
            unchecked_packages=tuple(packages),
            detail=str(result["error"]),
        )

    reported = result.get("packages") or {}
    missing = tuple(
        name for name in packages if not (reported.get(name) or {}).get("importable", False)
    )
    return TrainerEvidence(
        trainer_root_present=root_present,
        entrypoint_present=entry_present,
        interpreter_present=interpreter_present,
        torch_importable=result.get("torch_importable"),
        observed_ace_step_commit=commit,
        command_accepted=result.get("command_accepted"),
        command_detail=str(result.get("command_detail", "")),
        missing_packages=missing,
        detail=f"probed through {python_executable}",
    )


# ── storage ──────────────────────────────────────────────────────────


def _writable(path: Path | None) -> bool | None:
    if path is None:
        return None
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".luber-writable"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError:
        return False
    return True


def collect_storage_evidence(
    *,
    dataset_dir: Path | None = None,
    output_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    disk_probe_path: Path | None = None,
    trainer_root: Path | None = None,
) -> StorageEvidence:
    """Whether the run can read its input and write its output.

    Writability is established by writing, not by reading a permission
    bit: a full filesystem, a read-only mount and a missing parent all
    have correct-looking permissions.
    """
    readable: bool | None = None
    if dataset_dir is not None:
        readable = dataset_dir.is_dir() and any(dataset_dir.iterdir())

    # Writability first, because proving it creates the directories the
    # disk measurement then asks about. The other order measures the
    # filesystem holding whichever ancestor happened to exist, which is
    # usually the same answer and occasionally a different disk.
    output_writable = _writable(output_dir)
    checkpoint_writable = _writable(checkpoint_dir)

    free_mb: int | None = None
    probe = disk_probe_path or output_dir or checkpoint_dir
    while probe is not None and not probe.exists():
        probe = probe.parent if probe.parent != probe else None
    if probe is not None:
        try:
            import shutil

            free_mb = int(shutil.disk_usage(probe).free // (1024 * 1024))
        except OSError:
            free_mb = None

    inside: bool | None = None
    if dataset_dir is not None and trainer_root is not None:
        from luber_training.canary import within

        inside = within(dataset_dir, trainer_root)

    return StorageEvidence(
        dataset_readable=readable,
        output_writable=output_writable,
        checkpoint_writable=checkpoint_writable,
        free_disk_mb=free_mb,
        dataset_within_trainer_root=inside,
        detail="measured on the machine running this preflight",
    )


# ── remote ───────────────────────────────────────────────────────────


def collect_remote_evidence(
    worker: TrainingWorker | None,
    *,
    location: str,
    reachable: bool | None = None,
    transport: str | None = None,
    detail: str = "",
    now: datetime | None = None,
) -> RemoteEvidence:
    """How old the worker's report is, and whether it answered.

    Reachability is passed in rather than probed here. Reaching a rented
    machine needs credentials the control plane holds deliberately and a
    browser-reachable process does not, and this module must be callable
    from both.
    """
    if location != ExecutionLocation.REMOTE.value or worker is None:
        return RemoteEvidence(reachable=reachable, transport=transport, detail=detail)

    reported_at = worker.capabilities.reported_at
    age: float | None = None
    if reported_at:
        try:
            stamp = datetime.fromisoformat(reported_at)
        except ValueError:
            stamp = None
        if stamp is not None:
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=UTC)
            age = max(0.0, ((now or _now()) - stamp).total_seconds())

    return RemoteEvidence(
        capability_age_seconds=age,
        reachable=reachable,
        transport=transport,
        detail=detail,
    )


__all__ = [
    "TRAINER_PROBE_TIMEOUT_SECONDS",
    "ace_step_commit",
    "collect_dataset_evidence",
    "collect_remote_evidence",
    "collect_storage_evidence",
    "collect_trainer_evidence",
    "probe_trainer",
    "synthetic_dataset_evidence",
]
