"""Append-only metrics, and checkpoints that cannot lie about being ready.

Metrics are JSONL. Not Prometheus, not MLflow, not W&B — a training run
writes a few thousand numbers and an operator reads them afterwards, and
introducing a service for that would be infrastructure nobody asked for.
Appending line by line also means a run killed halfway leaves every
metric it had already emitted, which a batched writer would not.

Every event carries a ``source``. A number produced by a real trainer
and a number produced by a dry run must never be indistinguishable in
storage, because they will eventually be plotted on the same axes.

Checkpoint finalisation is the other half of this module and the more
dangerous one. A partially-written adapter that reads as ``READY`` is a
model that will be evaluated, maybe promoted, and is corrupt. So the
write goes to a temporary path, is validated and hashed there, and only
then moves atomically into place — an interrupted write leaves a temp
file and no registry entry, never a READY checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

METRICS_FILE_NAME = "metrics.jsonl"


class MetricSource(StrEnum):
    """Where a number came from. Never collapsed away."""

    TRAINER = "TRAINER"
    WORKER_TELEMETRY = "WORKER_TELEMETRY"
    ORCHESTRATOR = "ORCHESTRATOR"
    #: Produced by a dry run. Not a measurement of anything.
    SIMULATED = "SIMULATED"


#: Metric names the system expects. Not enforced — a trainer may emit
#: something new and losing it would be worse than not recognising it —
#: but documented so dashboards have a vocabulary.
KNOWN_METRICS: frozenset[str] = frozenset(
    {
        "train_loss",
        "learning_rate",
        "grad_norm",
        "samples_per_second",
        "step_time_seconds",
        "gpu_memory_mb",
        "gpu_utilization_percent",
        "gpu_power_watts",
        "cpu_percent",
        "ram_mb",
        "disk_free_mb",
        "epoch_time_seconds",
    }
)


@dataclass
class MetricEvent:
    run_id: str
    metric_name: str
    value: float
    source: str
    step: int | None = None
    epoch: int | None = None
    unit: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def is_simulated(self) -> bool:
        return self.source == MetricSource.SIMULATED.value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MetricWriter:
    """Append-only writer. Survives a run dying mid-stream."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: MetricEvent) -> None:
        self.append_many([event])

    def append_many(self, events: list[MetricEvent]) -> None:
        if not events:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def read(self) -> list[MetricEvent]:
        return list(iter_metrics(self.path))


def iter_metrics(path: Path) -> Iterator[MetricEvent]:
    """Read metrics, tolerating a torn final line.

    A run killed mid-write can leave a partial line. Refusing to read
    the whole file because of it would discard everything the run did
    emit, which is the opposite of what append-only storage is for.
    """
    if not Path(path).is_file():
        return
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or "metric_name" not in payload:
            continue
        yield MetricEvent(
            run_id=str(payload.get("run_id", "")),
            metric_name=str(payload["metric_name"]),
            value=float(payload.get("value", 0.0)),
            source=str(payload.get("source", MetricSource.ORCHESTRATOR.value)),
            step=payload.get("step"),
            epoch=payload.get("epoch"),
            unit=str(payload.get("unit", "")),
            timestamp=str(payload.get("timestamp", "")),
        )


def summarize(events: list[MetricEvent]) -> dict[str, Any]:
    """Last value and count per metric, keeping sources apart."""
    summary: dict[str, Any] = {}
    for event in events:
        entry = summary.setdefault(
            event.metric_name,
            {"count": 0, "last_value": None, "unit": event.unit, "sources": set()},
        )
        entry["count"] += 1
        entry["last_value"] = event.value
        entry["sources"].add(event.source)
    return {
        name: {**entry, "sources": sorted(entry["sources"])}
        for name, entry in sorted(summary.items())
    }


# ── checkpoint finalisation ──────────────────────────────────────────


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be finalised safely."""


def _digest_tree(root: Path) -> tuple[str, int]:
    """Content digest and total size of a checkpoint directory.

    Hashes relative paths as well as bytes, so two checkpoints with the
    same file contents under different names are different — the file
    layout is part of what a loader reads.
    """
    digest = hashlib.sha256()
    total = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        size = path.stat().st_size
        total += size
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest(), total


#: Files a PEFT adapter checkpoint must contain to be loadable. Taken
#: from what `trainer_helpers.save_adapter_flat` writes.
REQUIRED_ADAPTER_FILES: tuple[str, ...] = ("adapter_config.json", "adapter_model.safetensors")


@dataclass
class FinalizedCheckpoint:
    path: Path
    sha256: str
    size_bytes: int
    file_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "file_count": self.file_count,
        }


def validate_adapter_directory(path: Path) -> list[str]:
    """Whether a directory looks like a loadable adapter checkpoint."""
    problems: list[str] = []
    if not path.is_dir():
        return [f"{path} is not a directory"]
    for name in REQUIRED_ADAPTER_FILES:
        candidate = path / name
        if not candidate.is_file():
            problems.append(f"missing {name}")
        elif candidate.stat().st_size == 0:
            problems.append(f"{name} is empty")
    return problems


def finalize_checkpoint(
    staging_path: Path,
    destination: Path,
    *,
    validate: bool = True,
    require_adapter_files: bool = True,
) -> FinalizedCheckpoint:
    """Validate, hash, then atomically move a checkpoint into place.

    The order is the whole point. Validation and hashing happen at the
    staging path, so a checkpoint that fails either never reaches its
    destination — and because the final step is a directory rename,
    there is no window in which the destination exists but is
    incomplete.

    A crash at any point leaves a staging directory and no destination.
    That is recoverable and obvious. The alternative — a half-written
    directory sitting where a loader expects a model — is neither.
    """
    staging_path = Path(staging_path)
    destination = Path(destination)

    if not staging_path.exists():
        raise CheckpointError(f"nothing staged at {staging_path}")
    if destination.exists():
        raise CheckpointError(f"{destination} already exists; checkpoints are never overwritten")

    if validate and require_adapter_files:
        problems = validate_adapter_directory(staging_path)
        if problems:
            raise CheckpointError(f"staged checkpoint is not loadable: {'; '.join(problems)}")

    digest, size = _digest_tree(staging_path)
    file_count = sum(1 for path in staging_path.rglob("*") if path.is_file())
    if file_count == 0:
        raise CheckpointError("staged checkpoint contains no files")

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Same filesystem, so this is a rename rather than a copy: atomic,
    # and no window where the destination is partially populated.
    os.replace(staging_path, destination)
    return FinalizedCheckpoint(
        path=destination, sha256=digest, size_bytes=size, file_count=file_count
    )


@dataclass
class StagedCheckpoint:
    """A checkpoint being written. Never registered as READY."""

    staging_path: Path

    def __enter__(self) -> Path:
        return self.staging_path

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        # Left in place on failure: a staging directory is diagnostic,
        # and deleting the evidence of a failed write helps nobody.
        return None


def new_staging(root: Path, checkpoint_id: str) -> StagedCheckpoint:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".staging-{checkpoint_id}-", dir=str(root)))
    return StagedCheckpoint(staging_path=staging)


# ── retention ────────────────────────────────────────────────────────


@dataclass
class RetentionPolicy:
    """What to keep. Phase 25 plans deletions and performs none.

    Deleting model files is destructive and irreversible, and a
    retention pass that ran automatically would eventually delete
    something during an experiment nobody wanted to repeat. So this
    produces a *plan*, and an operator executes it deliberately.
    """

    keep_latest_n: int = 3
    keep_best_n: int = 1
    keep_every_n_steps: int | None = None
    keep_final: bool = True
    best_metric: str = "train_loss"
    #: Whether a lower value of `best_metric` is better.
    lower_is_better: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetentionPlan:
    keep: list[str] = field(default_factory=list)
    delete: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "keep": sorted(self.keep),
            "delete": sorted(self.delete),
            "reasons": dict(sorted(self.reasons.items())),
            "note": "a plan only; nothing is deleted without explicit operator action",
        }


def plan_retention(checkpoints: list[dict[str, Any]], policy: RetentionPolicy) -> RetentionPlan:
    """Which checkpoints a policy would keep, and why.

    Every kept checkpoint records *which* rule kept it, so an operator
    reviewing a deletion list can see whether the policy did what they
    meant rather than only what it did.
    """
    plan = RetentionPlan()
    ready = [checkpoint for checkpoint in checkpoints if checkpoint.get("status") == "READY"]
    ordered = sorted(ready, key=lambda c: (c.get("step") or 0, str(c.get("checkpoint_id"))))
    if not ordered:
        return plan

    keep: dict[str, str] = {}

    for checkpoint in ordered[-policy.keep_latest_n :] if policy.keep_latest_n else []:
        keep[str(checkpoint["checkpoint_id"])] = f"among the latest {policy.keep_latest_n}"

    if policy.keep_final:
        keep[str(ordered[-1]["checkpoint_id"])] = "final checkpoint"

    if policy.keep_best_n:
        scored = [
            checkpoint
            for checkpoint in ordered
            if isinstance(
                (checkpoint.get("metrics_snapshot") or {}).get(policy.best_metric), (int, float)
            )
        ]
        scored.sort(
            key=lambda c: float(c["metrics_snapshot"][policy.best_metric]),
            reverse=not policy.lower_is_better,
        )
        for checkpoint in scored[: policy.keep_best_n]:
            keep.setdefault(str(checkpoint["checkpoint_id"]), f"best {policy.best_metric}")

    if policy.keep_every_n_steps:
        for checkpoint in ordered:
            step = checkpoint.get("step")
            if isinstance(step, int) and step % policy.keep_every_n_steps == 0:
                keep.setdefault(
                    str(checkpoint["checkpoint_id"]),
                    f"step {step} is a multiple of {policy.keep_every_n_steps}",
                )

    plan.keep = list(keep)
    plan.reasons = keep
    plan.delete = [
        str(checkpoint["checkpoint_id"])
        for checkpoint in ordered
        if str(checkpoint["checkpoint_id"]) not in keep
    ]
    return plan


def execute_retention(plan: RetentionPlan, resolve: Any, *, confirm: bool = False) -> list[str]:
    """Delete what a retention plan lists. Requires explicit confirmation.

    ``confirm`` has no default that would let a caller delete by
    forgetting an argument.
    """
    if not confirm:
        raise CheckpointError(
            "retention execution requires confirm=True; Phase 25 plans deletions and "
            "does not perform them implicitly"
        )
    removed: list[str] = []
    for checkpoint_id in plan.delete:
        path = resolve(checkpoint_id)
        if path is None:
            continue
        target = Path(path)
        if target.is_dir():
            shutil.rmtree(target)
            removed.append(checkpoint_id)
        elif target.is_file():
            target.unlink()
            removed.append(checkpoint_id)
    return removed
