"""Where a worker may write, and the rules that keep it there.

A remote worker receives paths from the control plane. Those paths come
from a manifest which came from a dataset build which came, ultimately,
from filenames on someone's disk. Treating any of that as trusted input
to a filesystem write is how an artifact named `../../.ssh/authorized_keys`
ends up doing exactly what it says.

So every relative path crossing the boundary is validated the same way,
by the same function, before it becomes part of a real path. The check
is deliberately strict — absolute paths, parent traversal, drive
letters, null bytes and reserved names are all refused rather than
sanitised. Sanitising invites the question of whether the sanitiser is
complete; refusing does not.

The second idea here is that a plan never contains machine paths. It
names *logical roots* — code, data, run, checkpoint — and each worker's
registration says where those live on that machine. A plan compiled on a
Mac executes unchanged on a Linux box because it never claimed to know
where anything was.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

#: Logical roots a plan may refer to. A worker resolves each to a real
#: directory at registration time.
CODE_ROOT = "code_root"
DATA_ROOT = "data_root"
RUN_ROOT = "run_root"
CHECKPOINT_ROOT = "checkpoint_root"
CACHE_ROOT = "cache_root"

LOGICAL_ROOTS: tuple[str, ...] = (CODE_ROOT, DATA_ROOT, RUN_ROOT, CHECKPOINT_ROOT, CACHE_ROOT)

#: Names Windows refuses to create, refused here too. A worker will
#: almost certainly be Linux, but a manifest built once may be replayed
#: anywhere, and a path that works on one machine and silently fails on
#: another is worse than one that fails everywhere.
_RESERVED = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\.|$)", re.IGNORECASE)

#: Longest single component. Most filesystems stop at 255 bytes.
MAX_COMPONENT_LENGTH = 200
#: Longest whole relative path, well inside every common PATH_MAX.
MAX_RELATIVE_LENGTH = 1024


class UnsafePathError(ValueError):
    """Raised when a path would escape, or could not be written safely."""


def validate_relative(candidate: str) -> str:
    """Return *candidate* as a safe POSIX-relative path, or raise.

    Every rejection here corresponds to a way a path can leave the
    directory it was supposed to stay in, or a way two different strings
    can name the same file — which matters because the manifest keys on
    the string and the filesystem keys on the file.
    """
    if not isinstance(candidate, str) or not candidate.strip():
        raise UnsafePathError("an artifact path may not be empty")
    if "\x00" in candidate:
        raise UnsafePathError("an artifact path may not contain a null byte")
    if len(candidate) > MAX_RELATIVE_LENGTH:
        raise UnsafePathError(f"artifact path is longer than {MAX_RELATIVE_LENGTH} characters")

    # Backslashes are separators on Windows and legal filename
    # characters on POSIX. A path containing one means two different
    # things depending on the worker, so it is refused rather than
    # interpreted.
    if "\\" in candidate:
        raise UnsafePathError(f"artifact path {candidate!r} contains a backslash")
    if candidate.startswith("/") or PurePosixPath(candidate).is_absolute():
        raise UnsafePathError(f"artifact path {candidate!r} is absolute")
    if re.match(r"^[A-Za-z]:", candidate):
        raise UnsafePathError(f"artifact path {candidate!r} names a drive")
    if candidate.startswith("~"):
        raise UnsafePathError(f"artifact path {candidate!r} refers to a home directory")

    parts = [part for part in candidate.split("/") if part not in ("", ".")]
    if not parts:
        raise UnsafePathError(f"artifact path {candidate!r} names no file")
    for part in parts:
        if part == "..":
            raise UnsafePathError(f"artifact path {candidate!r} traverses upwards")
        if len(part) > MAX_COMPONENT_LENGTH:
            raise UnsafePathError(f"path component {part!r} is too long")
        if _RESERVED.match(part):
            raise UnsafePathError(f"path component {part!r} is a reserved device name")
        if part.endswith((" ", ".")):
            raise UnsafePathError(
                f"path component {part!r} ends in a space or dot, which some "
                "filesystems silently strip"
            )

    normalised = posixpath.normpath("/".join(parts))
    # normpath cannot reintroduce traversal here, but the assertion is
    # cheap and this is the function everything else trusts.
    if normalised.startswith("..") or normalised.startswith("/"):
        raise UnsafePathError(f"artifact path {candidate!r} does not normalise to a safe path")
    return normalised


def resolve_within(root: Path, relative: str) -> Path:
    """Join *relative* onto *root*, proving the result stays inside.

    The string check alone is not enough once symlinks exist: a
    perfectly well-formed relative path can land on a symlink that
    points elsewhere. So the joined path is resolved against a resolved
    root and the containment is checked on the real paths.
    """
    safe = validate_relative(relative)
    root = Path(root)
    base = root.resolve() if root.exists() else root.absolute()
    target = (base / safe).absolute()

    # Resolve as far as the path exists. A not-yet-created file cannot
    # be a symlink, but its parent directories can be.
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    real_probe = probe.resolve()
    if real_probe != base and base not in real_probe.parents:
        raise UnsafePathError(
            f"{relative!r} resolves outside the run root via a link: "
            f"{real_probe} is not under {base}"
        )

    resolved_target = real_probe / target.relative_to(probe) if probe != target else real_probe
    if resolved_target != base and base not in resolved_target.parents:
        raise UnsafePathError(f"{relative!r} would write outside {base}")
    return resolved_target


@dataclass
class RemoteRoots:
    """Where a particular worker keeps things.

    Supplied at registration, never assumed. There is no universal
    filesystem layout: a rented box might mount fast scratch at
    ``/workspace`` and persistent storage at ``/data``, and the next one
    will do it differently. The plan names roles; this names places.
    """

    code_root: str
    data_root: str
    run_root: str
    checkpoint_root: str
    cache_root: str

    @classmethod
    def under(cls, base: str | Path) -> RemoteRoots:
        """A conventional layout beneath one base directory.

        Offered as a starting point for a fresh host, not as a default
        anything falls back to. A worker that never declared its roots
        has no roots.
        """
        root = PurePosixPath(str(base))
        return cls(
            code_root=str(root / "code"),
            data_root=str(root / "data"),
            run_root=str(root / "runs"),
            checkpoint_root=str(root / "checkpoints"),
            cache_root=str(root / "cache"),
        )

    def get(self, logical: str) -> str:
        if logical not in LOGICAL_ROOTS:
            raise UnsafePathError(f"{logical!r} is not a logical root")
        value: str = getattr(self, logical)
        return value

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunLayout:
    """The directory contract for one run on a worker.

    Fixed, so that a reconnecting control plane and a restarted worker
    look in the same places without negotiating. Every path a run
    touches is beneath ``root``; there is no state anywhere else, which
    is what makes cleanup and reconciliation tractable.
    """

    root: Path

    @classmethod
    def for_run(cls, run_root: Path | str, run_id: str) -> RunLayout:
        # The run id goes through the same validation as any other
        # untrusted component. It is generated by this project, but the
        # place to enforce that is where the path is built.
        return cls(root=Path(run_root) / validate_relative(run_id))

    @property
    def plan_json(self) -> Path:
        return self.root / "plan.json"

    @property
    def manifest_json(self) -> Path:
        return self.root / "artifact_manifest.json"

    @property
    def environment_json(self) -> Path:
        return self.root / "environment_lock.json"

    @property
    def lease_json(self) -> Path:
        return self.root / "lease.json"

    @property
    def status_json(self) -> Path:
        return self.root / "status.json"

    @property
    def preflight_json(self) -> Path:
        return self.root / "remote_preflight.json"

    @property
    def result_json(self) -> Path:
        return self.root / "remote_result.json"

    @property
    def trainer_dir(self) -> Path:
        return self.root / "trainer"

    @property
    def dataset_dir(self) -> Path:
        return self.root / "dataset"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def metrics_dir(self) -> Path:
        return self.root / "metrics"

    @property
    def metrics_jsonl(self) -> Path:
        return self.metrics_dir / "metrics.jsonl"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def temp_dir(self) -> Path:
        return self.root / "temp"

    @property
    def stdout_log(self) -> Path:
        return self.logs_dir / "trainer.stdout.log"

    @property
    def stderr_log(self) -> Path:
        return self.logs_dir / "trainer.stderr.log"

    def ensure(self) -> None:
        for path in (
            self.root,
            self.trainer_dir,
            self.dataset_dir,
            self.logs_dir,
            self.metrics_dir,
            self.checkpoints_dir,
            self.output_dir,
            self.temp_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative: str) -> Path:
        """A manifest path, as a real path guaranteed inside this run."""
        return resolve_within(self.root, relative)

    def to_dict(self) -> dict[str, Any]:
        return {"root": str(self.root)}


__all__ = [
    "CACHE_ROOT",
    "CHECKPOINT_ROOT",
    "CODE_ROOT",
    "DATA_ROOT",
    "LOGICAL_ROOTS",
    "MAX_COMPONENT_LENGTH",
    "MAX_RELATIVE_LENGTH",
    "RUN_ROOT",
    "RemoteRoots",
    "RunLayout",
    "UnsafePathError",
    "resolve_within",
    "validate_relative",
]
