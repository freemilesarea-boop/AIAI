"""Bringing a checkpoint home, and only then calling it ready.

The rule this module exists to enforce: **a checkpoint becomes READY in
the Phase 25 registry after its bytes are local and hash to the value
the worker reported, and at no earlier point.**

Everything follows from that. Files land under temporary names and are
renamed only after verification. The directory digest is recomputed
locally with the same algorithm the worker used, so the comparison is
between two independent measurements rather than between a number and
itself. A mismatch fails the collection, keeps the diagnostics, and
leaves the remote copy alone — the remote copy is the known-good one,
and deleting it because the transfer went wrong would destroy the only
intact artifact.

Retry is safe by construction. A failed collection leaves the local
destination absent, so calling again re-transfers; a successful one
leaves it present and hashed, so calling again is a no-op.

Nothing here creates an evaluation candidate. Phase 25 refuses MOCK
artifacts at that boundary and this module does not go near it: a
checkpoint that was collected is a checkpoint that exists locally, which
is a different claim from one worth evaluating.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from luber_training.entities import Checkpoint, CheckpointKind, CheckpointStatus
from luber_training.remote.protocol import now
from luber_training.remote.result import (
    IGNORED_NAMES,
    ArtifactLocation,
    RemoteCheckpoint,
    RemoteCheckpointStatus,
    RemoteResult,
)
from luber_training.remote.transport import ArtifactTransport, IntegrityError, TransportError

#: Suffix for a checkpoint directory still being assembled locally.
STAGING_SUFFIX = ".collecting"


class CollectionError(RuntimeError):
    """Raised when a checkpoint could not be brought back intact."""


def digest_tree(root: Path) -> tuple[str, int, int]:
    """The same digest the worker computed, computed here.

    Byte-for-byte the same algorithm as `result._digest_tree`, and that
    is the point: two independent runs of one algorithm over two copies
    of one directory is a real comparison. Two different algorithms
    would only ever produce two numbers.
    """
    digest = hashlib.sha256()
    total = 0
    count = 0
    for path in sorted(Path(root).rglob("*")):
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
class CollectedCheckpoint:
    """One checkpoint that made it back, or did not."""

    checkpoint_id: str
    ok: bool
    local_path: str | None = None
    sha256: str | None = None
    size_bytes: int = 0
    file_count: int = 0
    files_transferred: int = 0
    files_skipped: int = 0
    problem: str | None = None
    location: str = ArtifactLocation.REMOTE_ONLY.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "ok": self.ok,
            "local_path": self.local_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "file_count": self.file_count,
            "files_transferred": self.files_transferred,
            "files_skipped": self.files_skipped,
            "problem": self.problem,
            "location": self.location,
        }


@dataclass
class CollectionReport:
    """Everything one collection attempt did."""

    run_id: str
    collected: list[CollectedCheckpoint] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = field(default_factory=now)

    @property
    def ok(self) -> bool:
        return all(item.ok for item in self.collected)

    @property
    def successful(self) -> list[CollectedCheckpoint]:
        return [item for item in self.collected if item.ok]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "ok": self.ok,
            "collected": [item.to_dict() for item in self.collected],
            "skipped": self.skipped,
            "started_at": self.started_at,
        }


def collect_checkpoint(
    transport: ArtifactTransport,
    remote: RemoteCheckpoint,
    *,
    run_id: str,
    destination_root: Path,
    remote_prefix: str | None = None,
) -> CollectedCheckpoint:
    """Fetch one checkpoint directory and verify it as a whole.

    Files are pulled into a staging directory beside the destination.
    The destination only comes into existence via a rename, after the
    tree digest matches, so a reader that finds the directory finds a
    complete checkpoint or nothing.

    Resume is per-file: a file already present locally with the right
    size is not re-fetched. The whole-tree digest at the end is what
    makes that safe — if a skipped file was wrong, the tree hash fails.

    ``remote_prefix`` is how the run is addressed through the transport.
    The worker records a checkpoint's path relative to its *run* root,
    while a transport is rooted at the directory holding every run — so
    the run id has to be put back on the front, and defaulting to it
    keeps the two conventions from drifting apart silently.
    """
    if not remote.collectable:
        return CollectedCheckpoint(
            checkpoint_id=remote.checkpoint_id,
            ok=False,
            problem=(
                f"the worker reports this checkpoint as {remote.status}"
                + (f": {'; '.join(remote.problems)}" if remote.problems else "")
            ),
        )

    destination = Path(destination_root) / remote.checkpoint_id
    staging = destination.with_name(destination.name + STAGING_SUFFIX)

    if destination.is_dir():
        digest, size, count = digest_tree(destination)
        if digest == remote.sha256:
            return CollectedCheckpoint(
                checkpoint_id=remote.checkpoint_id,
                ok=True,
                local_path=str(destination),
                sha256=digest,
                size_bytes=size,
                file_count=count,
                files_skipped=count,
                location=ArtifactLocation.LOCAL.value,
            )
        # Present but wrong. Left in place rather than deleted: it is
        # evidence about a failed transfer, and the caller decides.
        return CollectedCheckpoint(
            checkpoint_id=remote.checkpoint_id,
            ok=False,
            local_path=str(destination),
            sha256=digest,
            problem=(
                f"a local copy already exists and hashes to {digest[:12]}, but the worker "
                f"reports {(remote.sha256 or '')[:12]}. It has not been overwritten"
            ),
        )

    staging.mkdir(parents=True, exist_ok=True)
    transferred = 0
    skipped = 0

    scope = remote_prefix if remote_prefix is not None else run_id
    remote_dir = f"{scope}/{remote.relative_path}" if scope else remote.relative_path

    try:
        remote_files = transport.list_files(remote_dir)
        if not remote_files:
            raise CollectionError(f"the worker lists no files under {remote_dir}")

        prefix = remote_dir.rstrip("/") + "/"
        for entry in remote_files:
            relative = entry.path[len(prefix) :] if entry.path.startswith(prefix) else entry.path
            local = staging / relative
            if local.is_file() and local.stat().st_size == entry.size_bytes:
                # Resume: this file survived an earlier attempt. The
                # tree digest below is what proves it is the right one.
                skipped += 1
                continue
            transport.download(entry.path, local, expected_sha256=entry.sha256)
            transferred += 1

        digest, size, count = digest_tree(staging)
        if digest != remote.sha256:
            raise IntegrityError(
                f"the collected checkpoint hashes to {digest[:12]} but the worker reported "
                f"{(remote.sha256 or '')[:12]}. It has not been installed, and the remote "
                "copy has been left untouched so the transfer can be retried"
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(destination)
        return CollectedCheckpoint(
            checkpoint_id=remote.checkpoint_id,
            ok=True,
            local_path=str(destination),
            sha256=digest,
            size_bytes=size,
            file_count=count,
            files_transferred=transferred,
            files_skipped=skipped,
            location=ArtifactLocation.LOCAL.value,
        )
    except (IntegrityError, CollectionError, TransportError, OSError) as exc:
        # The staging directory stays. It holds whatever arrived, which
        # is what makes the next attempt a resume rather than a restart,
        # and it is evidence if the failure repeats.
        return CollectedCheckpoint(
            checkpoint_id=remote.checkpoint_id,
            ok=False,
            local_path=str(staging),
            files_transferred=transferred,
            files_skipped=skipped,
            problem=str(exc),
        )


def collect_run(
    transport: ArtifactTransport,
    result: RemoteResult,
    *,
    destination_root: Path,
    remote_prefix: str | None = None,
) -> CollectionReport:
    """Bring back every collectable checkpoint a run produced.

    Rejected and still-writing checkpoints are recorded as skipped with
    the worker's reason rather than dropped. A run that produced
    "nothing" usually produced something that failed validation, and an
    operator needs to see which.
    """
    report = CollectionReport(run_id=result.run_id)
    for remote in result.checkpoints:
        if remote.status != RemoteCheckpointStatus.READY_REMOTE.value:
            report.skipped.append(
                {
                    "checkpoint_id": remote.checkpoint_id,
                    "status": remote.status,
                    "problems": remote.problems,
                }
            )
            continue
        report.collected.append(
            collect_checkpoint(
                transport,
                remote,
                run_id=result.run_id,
                destination_root=Path(destination_root),
                remote_prefix=remote_prefix,
            )
        )
    return report


def register_collected(
    orchestrator: Any,
    *,
    run_id: str,
    collected: CollectedCheckpoint,
    remote: RemoteCheckpoint,
    kind: str = CheckpointKind.ADAPTER.value,
) -> Checkpoint:
    """Put a verified local checkpoint into the Phase 25 registry.

    Two writes, matching Phase 25's own contract: a WRITING record
    first, then `finalize_checkpoint_record` with the digest that was
    measured locally. Nothing else in this project sets READY, and this
    module does not either — it supplies the evidence and lets the
    orchestrator apply its own rule.

    Refuses an unverified collection outright. A checkpoint that failed
    its hash has no business in a registry that other phases read as
    authoritative.
    """
    if not collected.ok or not collected.sha256:
        raise CollectionError(
            f"checkpoint {collected.checkpoint_id} was not collected successfully "
            f"({collected.problem}); it will not be registered"
        )

    checkpoint = Checkpoint(
        checkpoint_id=f"ckpt_{collected.sha256[:16]}",
        run_id=run_id,
        kind=kind,
        step=remote.step,
        epoch=remote.epoch,
        status=CheckpointStatus.WRITING.value,
        checkpoint_format=remote.format,
    )
    orchestrator.register_checkpoint(checkpoint)
    finalized: Checkpoint = orchestrator.finalize_checkpoint_record(
        checkpoint.checkpoint_id,
        sha256=collected.sha256,
        size_bytes=collected.size_bytes,
        reference=str(collected.local_path),
    )
    return finalized


@dataclass
class RetentionDecision:
    """What to do with a remote checkpoint after it has been collected.

    Nothing is deleted automatically. A rented instance can be
    terminated at any moment and take its disk with it, so the remote
    copy is a second copy for exactly as long as the machine exists —
    and that window closes without warning. Deleting it to reclaim space
    the operator did not ask to reclaim is a trade nobody authorised.
    """

    checkpoint_id: str
    action: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "action": self.action,
            "reason": self.reason,
        }


def plan_remote_retention(report: CollectionReport) -> list[RetentionDecision]:
    """Recommend, never execute. Deletion is an operator decision."""
    decisions: list[RetentionDecision] = []
    for item in report.collected:
        if item.ok:
            decisions.append(
                RetentionDecision(
                    checkpoint_id=item.checkpoint_id,
                    action="RETAIN_UNTIL_OPERATOR_APPROVES",
                    reason=(
                        "collected and verified locally. The remote copy is a second copy "
                        "until the instance is terminated, which may happen without notice; "
                        "it is not deleted automatically"
                    ),
                )
            )
        else:
            decisions.append(
                RetentionDecision(
                    checkpoint_id=item.checkpoint_id,
                    action="RETAIN_REQUIRED",
                    reason=(
                        "collection failed, so the remote copy is the only known-good one. "
                        "Deleting it would destroy the artifact the retry needs"
                    ),
                )
            )
    return decisions


def clean_staging(destination_root: Path) -> list[str]:
    """Remove abandoned partial collections. Never a finished one."""
    removed: list[str] = []
    root = Path(destination_root)
    if not root.is_dir():
        return removed
    for path in sorted(root.iterdir()):
        if path.is_dir() and path.name.endswith(STAGING_SUFFIX):
            removed.append(path.name)
            shutil.rmtree(path, ignore_errors=True)
    return removed


__all__ = [
    "STAGING_SUFFIX",
    "CollectedCheckpoint",
    "CollectionError",
    "CollectionReport",
    "RetentionDecision",
    "clean_staging",
    "collect_checkpoint",
    "collect_run",
    "digest_tree",
    "plan_remote_retention",
    "register_collected",
]
