"""Checkpoints on the worker, and the record of what a run produced.

Two ideas, and the first one is the reason for the second.

**A remote checkpoint is not a ready checkpoint.** It has been written
on a machine the control plane cannot see, by a process nobody watched,
and the bytes have not crossed the network yet. So it gets its own
state — `READY_REMOTE` — and the Phase 25 registry does not learn about
it until the file has arrived locally and hashed to the value the worker
reported. Registering it earlier would mean a checkpoint marked READY
that might be a truncated file on a machine that no longer exists.

**Discovery is by contract, not by glob.** The checkpoint directory
belongs to the trainer, and trainers write all sorts of things into
theirs: optimiser state, temp files, sample audio. Registering every
directory found there would eventually register something that is not a
model. A candidate must look like what `save_adapter_flat` writes, and
one that does not is reported as rejected — with the reason — rather
than skipped silently.

The result manifest is what the control plane reads after the fact. It
holds identity, timing, exit status, digests and checkpoint entries. It
holds no secret values; the redaction pass runs over it before it is
written, because the trainer's own log lines end up quoted in it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from luber_training.metrics import REQUIRED_ADAPTER_FILES
from luber_training.remote.paths import RunLayout
from luber_training.remote.protocol import REMOTE_PROTOCOL_VERSION, now
from luber_training.remote.secrets import redact_mapping

RESULT_SCHEMA_VERSION = "luber-remote-result/1"

#: Files a directory must not be judged on. Trainers leave these around
#: and their presence says nothing about whether a checkpoint is whole.
IGNORED_NAMES: frozenset[str] = frozenset({".DS_Store", "__pycache__", ".ipynb_checkpoints"})


class RemoteCheckpointStatus(StrEnum):
    """Where a checkpoint is in its journey to being trusted."""

    #: The trainer is still writing it, or it looks incomplete.
    WRITING = "WRITING"
    #: Complete and hashed on the worker. Not yet anywhere else.
    READY_REMOTE = "READY_REMOTE"
    #: Present but not a loadable checkpoint. Kept, never registered.
    REJECTED = "REJECTED"


class ArtifactLocation(StrEnum):
    """Where a checkpoint's bytes physically are.

    Separate from identity on purpose. A checkpoint is the same
    checkpoint whether it sits on a rented box, in object storage, or on
    the operator's disk, and the registry has to be able to say which
    without that changing what the checkpoint *is*. Phase 27 implements
    LOCAL collection; the vocabulary is complete so a later phase adds a
    backend rather than a concept.
    """

    REMOTE_ONLY = "REMOTE_ONLY"
    LOCAL = "LOCAL"
    OBJECT_STORE = "OBJECT_STORE"


def _digest_tree(root: Path) -> tuple[str, int, int]:
    """Content digest, total size and file count of a directory.

    Relative paths are hashed alongside the bytes, so two checkpoints
    with identical file contents under different names are different —
    the layout is part of what a loader reads. The same algorithm as
    Phase 25's `_digest_tree`, so a digest computed remotely and one
    computed locally are comparable.
    """
    digest = hashlib.sha256()
    total = 0
    count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in IGNORED_NAMES:
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        size = path.stat().st_size
        total += size
        count += 1
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest(), total, count


@dataclass
class RemoteCheckpoint:
    """One checkpoint as it exists on the worker."""

    checkpoint_id: str
    relative_path: str
    status: str
    sha256: str | None = None
    size_bytes: int = 0
    file_count: int = 0
    step: int | None = None
    epoch: int | None = None
    format: str = "peft-adapter-safetensors"
    problems: list[str] = field(default_factory=list)
    discovered_at: str = field(default_factory=now)

    @property
    def collectable(self) -> bool:
        return self.status == RemoteCheckpointStatus.READY_REMOTE.value and bool(self.sha256)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RemoteCheckpoint:
        known = {key: value for key, value in payload.items() if key in cls.__annotations__}
        return cls(**known)


def _parse_step_epoch(name: str) -> tuple[int | None, int | None]:
    """Step and epoch from a directory name, where it states them.

    Read only from names that plainly contain them, and left as None
    otherwise. A checkpoint whose name says nothing is not thereby step
    zero, and inventing a number would put it in a wrong place in every
    ordering.
    """
    import re

    step = re.search(r"(?:^|[_-])step[_-]?(\d+)", name, re.IGNORECASE)
    epoch = re.search(r"(?:^|[_-])epoch[_-]?(\d+)", name, re.IGNORECASE)
    return (
        int(step.group(1)) if step else None,
        int(epoch.group(1)) if epoch else None,
    )


def discover_checkpoints(
    layout: RunLayout, *, require_adapter_files: bool = True
) -> list[RemoteCheckpoint]:
    """Find what the trainer wrote, judging each against the contract.

    Only immediate subdirectories of the checkpoint directory are
    considered: `save_adapter_flat` writes one directory per checkpoint,
    and recursing would treat a checkpoint's internals as checkpoints.

    A directory that fails validation is returned as REJECTED with the
    reasons, not omitted. An operator looking at a run that produced
    "nothing" needs to see that something was written and why it was not
    accepted.
    """
    checkpoints_dir = layout.checkpoints_dir
    if not checkpoints_dir.is_dir():
        return []

    found: list[RemoteCheckpoint] = []
    for path in sorted(checkpoints_dir.iterdir()):
        if not path.is_dir() or path.name in IGNORED_NAMES:
            continue
        if path.name.endswith((".tmp", ".partial", ".staging")):
            # Plainly still being written. Not an error, just not ready.
            found.append(
                RemoteCheckpoint(
                    checkpoint_id=path.name,
                    relative_path=path.relative_to(layout.root).as_posix(),
                    status=RemoteCheckpointStatus.WRITING.value,
                    problems=["the directory name marks it as incomplete"],
                )
            )
            continue

        problems: list[str] = []
        if require_adapter_files:
            for required in REQUIRED_ADAPTER_FILES:
                candidate = path / required
                if not candidate.is_file():
                    problems.append(f"missing {required}")
                elif candidate.stat().st_size == 0:
                    problems.append(f"{required} is empty")

        digest, size, count = _digest_tree(path)
        if count == 0:
            problems.append("the directory contains no files")

        step, epoch = _parse_step_epoch(path.name)
        found.append(
            RemoteCheckpoint(
                checkpoint_id=path.name,
                relative_path=path.relative_to(layout.root).as_posix(),
                status=(
                    RemoteCheckpointStatus.REJECTED.value
                    if problems
                    else RemoteCheckpointStatus.READY_REMOTE.value
                ),
                sha256=None if problems else digest,
                size_bytes=size,
                file_count=count,
                step=step,
                epoch=epoch,
                problems=problems,
            )
        )
    return found


@dataclass
class RemoteResult:
    """What a run produced, as the worker saw it.

    Written when the run reaches a terminal state, and again on demand,
    so a control plane that reconnects after losing contact has a single
    file to read rather than a lifecycle to reconstruct.
    """

    run_id: str
    worker_id: str
    training_plan_sha256: str
    manifest_sha256: str
    worker_state: str
    exit_code: int | None = None
    trainer_status: str = ""
    failure_code: str | None = None
    detail: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    wall_seconds: float | None = None
    gpu_seconds: float | None = None
    checkpoints: list[RemoteCheckpoint] = field(default_factory=list)
    metrics_digest: str | None = None
    metrics_count: int = 0
    logs_digest: str | None = None
    logs_bytes: int = 0
    environment_digest: str | None = None
    capability_signature: str | None = None
    protocol_version: str = REMOTE_PROTOCOL_VERSION
    schema_version: str = RESULT_SCHEMA_VERSION
    created_at: str = field(default_factory=now)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checkpoints"] = [checkpoint.to_dict() for checkpoint in self.checkpoints]
        # The detail field quotes trainer output, which is the one place
        # in this record where an operator's token could plausibly turn up.
        return redact_mapping(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RemoteResult:
        known = {
            key: value
            for key, value in payload.items()
            if key in cls.__annotations__ and key != "checkpoints"
        }
        result = cls(**known)
        result.checkpoints = [
            RemoteCheckpoint.from_dict(item) for item in payload.get("checkpoints", [])
        ]
        return result

    def write(self, layout: RunLayout) -> Path:
        layout.root.mkdir(parents=True, exist_ok=True)
        layout.result_json.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return layout.result_json

    @classmethod
    def read(cls, layout: RunLayout) -> RemoteResult | None:
        if not layout.result_json.is_file():
            return None
        try:
            return cls.from_dict(json.loads(layout.result_json.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError, ValueError):
            return None


def _file_digest(path: Path) -> tuple[str | None, int]:
    if not path.is_file():
        return None, 0
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def build_result(
    *,
    layout: RunLayout,
    worker_id: str,
    state: Any,
    capability_signature: str | None = None,
) -> RemoteResult:
    """Summarise a finished run from what is on the worker's disk.

    Everything is derived from files rather than from memory, so this
    produces the same answer whether it runs in the invocation that
    launched the trainer or in one that arrived long afterwards.
    """
    metrics_digest, _ = _file_digest(layout.metrics_jsonl)
    metrics_count = 0
    if layout.metrics_jsonl.is_file():
        metrics_count = sum(
            1
            for line in layout.metrics_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    logs_hash = hashlib.sha256()
    logs_bytes = 0
    for log in (layout.stdout_log, layout.stderr_log):
        digest, size = _file_digest(log)
        if digest:
            logs_hash.update(digest.encode("utf-8"))
            logs_bytes += size

    environment_digest = None
    if layout.environment_json.is_file():
        environment_digest, _ = _file_digest(layout.environment_json)

    process = getattr(state, "process", None)
    wall_seconds = None
    if getattr(state, "started_at", None) and process is not None and process.finished_at:
        from datetime import datetime

        try:
            begin = datetime.fromisoformat(state.started_at)
            end = datetime.fromisoformat(process.finished_at)
            wall_seconds = max(0.0, (end - begin).total_seconds())
        except ValueError:
            wall_seconds = None

    return RemoteResult(
        run_id=layout.root.name,
        worker_id=worker_id,
        training_plan_sha256=getattr(state, "training_plan_sha256", "") or "",
        manifest_sha256=getattr(state, "manifest_sha256", "") or "",
        worker_state=getattr(state, "state", ""),
        exit_code=getattr(state, "exit_code", None),
        trainer_status=(
            "exited cleanly"
            if getattr(state, "exit_code", None) == 0
            else "did not complete successfully"
        ),
        failure_code=getattr(state, "failure_code", None),
        detail=getattr(state, "detail", ""),
        started_at=getattr(state, "started_at", None),
        completed_at=getattr(state, "completed_at", None)
        or getattr(state, "failed_at", None)
        or getattr(state, "cancelled_at", None),
        wall_seconds=wall_seconds,
        # No GPU-second accounting is claimed. Nothing here samples the
        # device continuously, and a figure derived from wall time would
        # be wall time wearing a different name.
        gpu_seconds=None,
        checkpoints=discover_checkpoints(layout),
        metrics_digest=metrics_digest,
        metrics_count=metrics_count,
        logs_digest=logs_hash.hexdigest() if logs_bytes else None,
        logs_bytes=logs_bytes,
        environment_digest=environment_digest,
        capability_signature=capability_signature,
    )


__all__ = [
    "RESULT_SCHEMA_VERSION",
    "ArtifactLocation",
    "RemoteCheckpoint",
    "RemoteCheckpointStatus",
    "RemoteResult",
    "build_result",
    "discover_checkpoints",
]
