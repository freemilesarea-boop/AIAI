"""Moving bytes to a worker and back, without trusting that they arrived.

One interface, so orchestration never learns what a network is. A
transport can copy a file, say whether one is already there, verify a
digest, and remove its own temporaries — and whether it does that over
SSH, over a filesystem, or over something not yet invented is not a
question anything above it asks.

Three rules every implementation honours.

**Nothing is finished until it is verified.** A file is written to a
temporary name, hashed where it landed, and only then moved into place.
A partially transferred file therefore never occupies the path a reader
would look at, which matters because the reader is a trainer that will
happily start on a truncated dataset.

**The move is atomic.** `os.replace` within one filesystem, `mv` on the
remote side. Not copy-then-delete, which has a window where both or
neither exist.

**Resume is by content, not by optimism.** Before sending anything the
transport asks what the destination already has and compares digests. A
file whose digest matches is skipped; a file whose digest differs is
re-sent whole. This is file-granular, and the docstrings say so — byte
ranges are not resumed, and claiming otherwise would invite someone to
rely on it.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from luber_training.remote.manifest import (
    RemoteArtifactManifest,
    TransferPlan,
    plan_transfer,
    sha256_file,
)
from luber_training.remote.paths import resolve_within, validate_relative

#: Suffix marking a file that is still arriving. Distinctive so a
#: cleanup pass can recognise one, and so a trainer globbing for audio
#: never matches a partial file.
PARTIAL_SUFFIX = ".luber-partial"

HASH_BLOCK = 1 << 20


class TransportError(RuntimeError):
    """Raised when a transfer cannot be completed or cannot be trusted."""


class IntegrityError(TransportError):
    """Raised when bytes arrived but are not the bytes that were sent."""


@dataclass
class TransferResult:
    """What one transfer actually did."""

    uploaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    bytes_uploaded: int = 0
    bytes_skipped: int = 0

    @property
    def ok(self) -> bool:
        return not self.failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "uploaded": sorted(self.uploaded),
            "skipped": sorted(self.skipped),
            "failed": dict(sorted(self.failed.items())),
            "bytes_uploaded": self.bytes_uploaded,
            "bytes_skipped": self.bytes_skipped,
        }


@dataclass
class RemoteFile:
    """What the far side reports about one path."""

    path: str
    size_bytes: int
    sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "size_bytes": self.size_bytes, "sha256": self.sha256}


class ArtifactTransport(ABC):
    """Byte movement, with verification built into the contract."""

    name: str = "abstract"
    #: Whether an interrupted single file can continue from its offset.
    #: False everywhere in this build, and stated rather than implied.
    supports_byte_range_resume: bool = False

    @abstractmethod
    def probe(self) -> dict[str, Any]:
        """Whether the far side is reachable, and what it says it is."""

    @abstractmethod
    def exists(self, relative: str) -> bool:
        """Whether a path exists on the far side."""

    @abstractmethod
    def stat(self, relative: str) -> RemoteFile | None:
        """Size and digest of a remote file, or None if absent."""

    @abstractmethod
    def upload(self, local: Path, relative: str, *, expected_sha256: str) -> RemoteFile:
        """Send one file, verify it, and put it in place atomically."""

    @abstractmethod
    def download(self, relative: str, local: Path, *, expected_sha256: str | None = None) -> Path:
        """Fetch one file, verifying it before it reaches *local*."""

    @abstractmethod
    def list_files(self, relative_dir: str) -> list[RemoteFile]:
        """Files beneath a remote directory."""

    @abstractmethod
    def remove_temp(self) -> list[str]:
        """Delete this transport's own partial files. Nothing else."""

    # ── shared behaviour ─────────────────────────────────────────────
    def present_digests(self, manifest: RemoteArtifactManifest) -> frozenset[str]:
        """Which of a manifest's contents the far side already holds.

        Asks about each distinct digest once, not each entry: a dataset
        with the same audio under two names is one question.
        """
        present: set[str] = set()
        for digest, entries in manifest.unique_contents().items():
            for entry in entries:
                remote = self.stat(entry.target_path)
                if remote and remote.sha256 == digest:
                    present.add(digest)
                    break
        return frozenset(present)

    def plan(self, manifest: RemoteArtifactManifest) -> TransferPlan:
        return plan_transfer(manifest, present_digests=self.present_digests(manifest))

    def upload_manifest(
        self,
        manifest: RemoteArtifactManifest,
        *,
        resolve: Any = None,
        skip_present: bool = True,
    ) -> TransferResult:
        """Send everything in a manifest that is not already there.

        Resume lives here: a transfer interrupted halfway is restarted
        by calling this again, and the files that made it are skipped
        because their digests match. There is no bookkeeping to
        reconcile — the destination's contents *are* the record.
        """
        result = TransferResult()
        seen: dict[str, str] = {}

        for entry in sorted(manifest.entries, key=lambda item: item.target_path):
            source = Path(resolve(entry) if resolve else entry.source_reference)
            try:
                if skip_present:
                    remote = self.stat(entry.target_path)
                    if remote is not None and remote.sha256 == entry.sha256:
                        result.skipped.append(entry.target_path)
                        result.bytes_skipped += entry.size_bytes
                        seen[entry.sha256] = entry.target_path
                        continue

                if not source.is_file():
                    if entry.required:
                        result.failed[entry.target_path] = f"source {source} is missing"
                    else:
                        result.skipped.append(entry.target_path)
                    continue

                self.upload(source, entry.target_path, expected_sha256=entry.sha256)
                result.uploaded.append(entry.target_path)
                result.bytes_uploaded += entry.size_bytes
                seen[entry.sha256] = entry.target_path
            except (TransportError, OSError) as exc:
                result.failed[entry.target_path] = str(exc)
        return result

    def verify_manifest(self, manifest: RemoteArtifactManifest) -> dict[str, str]:
        """Recheck every required entry. Returns path -> problem.

        Run after a transfer and again at preflight. Checking twice is
        not redundancy for its own sake: the second check happens on the
        worker, after everything has settled, and it is the one that
        catches a file that arrived intact and was then damaged.
        """
        problems: dict[str, str] = {}
        for entry in manifest.required_entries:
            remote = self.stat(entry.target_path)
            if remote is None:
                problems[entry.target_path] = "missing on the worker"
            elif remote.size_bytes != entry.size_bytes:
                problems[entry.target_path] = (
                    f"size is {remote.size_bytes}, manifest says {entry.size_bytes}"
                )
            elif remote.sha256 != entry.sha256:
                problems[entry.target_path] = (
                    f"digest is {(remote.sha256 or 'unreadable')[:12]}, manifest says "
                    f"{entry.sha256[:12]}"
                )
        return problems


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(HASH_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


class LocalArtifactTransport(ArtifactTransport):
    """A second machine simulated by a second directory.

    Not a stub. It performs real filesystem writes, real hashing, real
    atomic renames and real partial-file handling, so the lifecycle
    tests exercise the code that runs against a real remote rather than
    a mock that agrees with whatever it is told. The only thing it does
    not do is cross a network.

    ``fail_after_bytes`` and ``corrupt`` exist so a test can produce the
    two failures that matter — a transfer cut off mid-file, and bytes
    that arrive wrong — without unplugging anything.
    """

    name = "local"

    def __init__(
        self,
        root: Path,
        *,
        fail_after_bytes: int | None = None,
        corrupt_paths: frozenset[str] = frozenset(),
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.fail_after_bytes = fail_after_bytes
        self.corrupt_paths = corrupt_paths
        self._digest_cache: dict[tuple[str, int, int], str] = {}

    @property
    def _block(self) -> int:
        """Read size. Smaller when a cut-off is being simulated.

        A one-megabyte block would write most test files whole before
        the first check, so the "disconnection" would only ever land on
        a file boundary — which is not the case worth testing.
        """
        if self.fail_after_bytes is None:
            return HASH_BLOCK
        return max(1, min(HASH_BLOCK, self.fail_after_bytes))

    # ── helpers ──
    def _path(self, relative: str) -> Path:
        return resolve_within(self.root, relative)

    def probe(self) -> dict[str, Any]:
        usage = shutil.disk_usage(self.root)
        return {
            "transport": self.name,
            "root": str(self.root),
            "reachable": True,
            "free_disk_mb": int(usage.free / (1024 * 1024)),
        }

    def exists(self, relative: str) -> bool:
        try:
            return self._path(relative).is_file()
        except ValueError:
            return False

    def stat(self, relative: str) -> RemoteFile | None:
        path = self._path(relative)
        if not path.is_file():
            return None
        info = path.stat()
        # Cached on (path, size, mtime): hashing every dataset file on
        # every poll would make a large manifest quadratic in wall time
        # for no new information.
        key = (str(path), info.st_size, info.st_mtime_ns)
        digest = self._digest_cache.get(key)
        if digest is None:
            digest = _digest(path)
            self._digest_cache[key] = digest
        return RemoteFile(path=relative, size_bytes=info.st_size, sha256=digest)

    def upload(self, local: Path, relative: str, *, expected_sha256: str) -> RemoteFile:
        local = Path(local)
        if not local.is_file():
            raise TransportError(f"{local} does not exist")

        destination = self._path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + PARTIAL_SUFFIX)

        written = 0
        with local.open("rb") as source, partial.open("wb") as target:
            for block in iter(lambda: source.read(self._block), b""):
                if self.fail_after_bytes is not None and written >= self.fail_after_bytes:
                    # The partial file is left behind deliberately: a
                    # resumed transfer must find the destination absent,
                    # and cleanup must find something to clean up.
                    raise TransportError(
                        f"simulated disconnection after {written} bytes of {relative}"
                    )
                if relative in self.corrupt_paths:
                    block = bytes(byte ^ 0xFF for byte in block)
                target.write(block)
                written += len(block)
            target.flush()
            os.fsync(target.fileno())

        actual = _digest(partial)
        if actual != expected_sha256:
            partial.unlink(missing_ok=True)
            raise IntegrityError(
                f"{relative} arrived with digest {actual[:12]}, expected "
                f"{expected_sha256[:12]}; the partial file has been removed"
            )

        os.replace(partial, destination)
        info = destination.stat()
        return RemoteFile(path=relative, size_bytes=info.st_size, sha256=actual)

    def download(self, relative: str, local: Path, *, expected_sha256: str | None = None) -> Path:
        source = self._path(relative)
        if not source.is_file():
            raise TransportError(f"{relative} does not exist on the worker")

        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        partial = local.with_name(local.name + PARTIAL_SUFFIX)

        written = 0
        with source.open("rb") as reader, partial.open("wb") as writer:
            for block in iter(lambda: reader.read(self._block), b""):
                if self.fail_after_bytes is not None and written >= self.fail_after_bytes:
                    raise TransportError(
                        f"simulated disconnection after {written} bytes of {relative}"
                    )
                if relative in self.corrupt_paths:
                    block = bytes(byte ^ 0xFF for byte in block)
                writer.write(block)
                written += len(block)
            writer.flush()
            os.fsync(writer.fileno())

        if expected_sha256 is not None:
            actual = _digest(partial)
            if actual != expected_sha256:
                partial.unlink(missing_ok=True)
                raise IntegrityError(
                    f"{relative} downloaded with digest {actual[:12]}, expected "
                    f"{expected_sha256[:12]}"
                )

        os.replace(partial, local)
        return local

    def list_files(self, relative_dir: str) -> list[RemoteFile]:
        base = self._path(relative_dir) if relative_dir else self.root
        if not base.is_dir():
            return []
        files: list[RemoteFile] = []
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.name.endswith(PARTIAL_SUFFIX):
                continue
            relative = path.relative_to(self.root).as_posix()
            files.append(
                RemoteFile(path=relative, size_bytes=path.stat().st_size, sha256=_digest(path))
            )
        return files

    def remove_temp(self) -> list[str]:
        removed: list[str] = []
        for path in sorted(self.root.rglob(f"*{PARTIAL_SUFFIX}")):
            if path.is_file():
                removed.append(path.relative_to(self.root).as_posix())
                path.unlink()
        return removed


class ContentCache:
    """Immutable artifacts on the worker, addressed by digest.

    Worth having because dataset audio does not change between runs: the
    second experiment on the same corpus transfers nothing. Files are
    stored under their own digest, so there is no filename to be stale,
    and — this is the part that matters — a cache hit is *verified*
    before it is used. A file present under the right name but with the
    wrong contents is treated as absent, because the name is a claim and
    the digest is the fact.

    The cache assumes a single trusted operator domain: one person's
    runs on one rented box. It is not a shared cache, and it must not
    become one without a rights review, because content addressing alone
    would happily serve one tenant's audio to another.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, digest: str) -> Path:
        if len(digest) != 64 or not all(character in "0123456789abcdef" for character in digest):
            raise TransportError(f"{digest!r} is not a sha256 digest")
        # Two-character fan-out: ten thousand entries in one directory
        # is slow to list on most filesystems.
        return self.root / digest[:2] / digest

    def has(self, digest: str, *, expected_size: int | None = None) -> bool:
        """Whether the cache holds this content, verified.

        Never trusts the filename. A cached file is hashed before it is
        reported present, and one that fails is removed — a corrupt
        cache entry that keeps being served would poison every future
        run silently.
        """
        path = self.path_for(digest)
        if not path.is_file():
            return False
        if expected_size is not None and path.stat().st_size != expected_size:
            path.unlink(missing_ok=True)
            return False
        if _digest(path) != digest:
            path.unlink(missing_ok=True)
            return False
        return True

    def store(self, source: Path, digest: str) -> Path:
        """Add content, atomically, after verifying it."""
        source = Path(source)
        actual = _digest(source)
        if actual != digest:
            raise IntegrityError(f"{source} hashes to {actual[:12]}, not {digest[:12]}")
        destination = self.path_for(digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            return destination
        partial = destination.with_name(destination.name + PARTIAL_SUFFIX)
        shutil.copyfile(source, partial)
        os.replace(partial, destination)
        return destination

    def materialise(self, digest: str, target: Path) -> Path:
        """Put cached content where a run expects it.

        Copied rather than linked. A hard link would let a trainer that
        writes in place corrupt the cache for every future run, and the
        space saved is not worth that.
        """
        if not self.has(digest):
            raise TransportError(f"{digest[:12]} is not in the cache")
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + PARTIAL_SUFFIX)
        shutil.copyfile(self.path_for(digest), partial)
        os.replace(partial, target)
        return target

    def digests(self) -> frozenset[str]:
        return frozenset(
            path.name
            for path in self.root.rglob("*")
            if path.is_file() and not path.name.endswith(PARTIAL_SUFFIX)
        )

    def plan(self, manifest: RemoteArtifactManifest) -> TransferPlan:
        """What this cache would save on a given manifest."""
        cacheable = frozenset(
            digest
            for digest, entries in manifest.unique_contents().items()
            if entries[0].cacheable and self.has(digest, expected_size=entries[0].size_bytes)
        )
        return plan_transfer(manifest, present_digests=cacheable)


def verified_copy(source: Path, destination: Path, *, expected_sha256: str | None = None) -> str:
    """Copy a file through a temporary name, verifying before it lands.

    Used wherever a file has to appear complete or not at all — staging,
    checkpoint collection — so the same guarantee holds whether or not a
    transport was involved.
    """
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + PARTIAL_SUFFIX)
    shutil.copyfile(source, partial)
    actual = _digest(partial)
    if expected_sha256 is not None and actual != expected_sha256:
        partial.unlink(missing_ok=True)
        raise IntegrityError(
            f"{source} copied to {destination} with digest {actual[:12]}, expected "
            f"{expected_sha256[:12]}"
        )
    os.replace(partial, destination)
    return actual


__all__ = [
    "PARTIAL_SUFFIX",
    "ArtifactTransport",
    "ContentCache",
    "IntegrityError",
    "LocalArtifactTransport",
    "RemoteFile",
    "TransferResult",
    "TransportError",
    "sha256_file",
    "validate_relative",
    "verified_copy",
]
