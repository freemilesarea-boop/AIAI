"""What has to reach the worker, named by content rather than by place.

A manifest is the complete list of files a run needs on the other
machine, and every entry carries a SHA-256. That single decision buys
four things at once: transfer can skip what is already there, an
interrupted transfer can resume by comparing digests, corruption is
detectable rather than merely unlikely, and the whole set has a
deterministic identity — the same inputs produce the same manifest hash
on any machine, which is what makes a staged run reproducible.

Absolute paths appear nowhere in the recorded manifest. Each entry has a
*relative target path* under the run root, and a source reference the
control plane resolves locally. A manifest built on a Mac is executable
on a Linux box because it never said where anything was.

Roles are explicit because they mean different things downstream: a
dataset audio file may be cached and shared between runs, a plan may
not; a missing optional artifact is a warning, a missing required one
stops the run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from luber_training.remote.paths import validate_relative
from luber_training.remote.protocol import REMOTE_PROTOCOL_VERSION, now

MANIFEST_SCHEMA_VERSION = "luber-remote-manifest/1"

#: Read size for hashing. One megabyte keeps a large file off the heap
#: without making a syscall per kilobyte.
HASH_BLOCK = 1 << 20


class ArtifactRole(StrEnum):
    """What an artifact is for, which decides how it is handled.

    ``DATASET_AUDIO`` is the only role that is both large and immutable
    across runs, so it is the one that makes the content cache worth
    having. ``PLAN`` and ``TRAINER_CONFIG`` are small and specific to a
    run; caching them would save nothing and risk a stale plan.
    """

    PLAN = "PLAN"
    ENVIRONMENT_LOCK = "ENVIRONMENT_LOCK"
    TRAINER_CONFIG = "TRAINER_CONFIG"
    TRAINER_DATASET = "TRAINER_DATASET"
    DATASET_MANIFEST = "DATASET_MANIFEST"
    CURATED_MANIFEST = "CURATED_MANIFEST"
    SAMPLING_WEIGHTS = "SAMPLING_WEIGHTS"
    DATASET_AUDIO = "DATASET_AUDIO"
    CODE_BUNDLE = "CODE_BUNDLE"
    METADATA = "METADATA"


#: Roles whose content may be reused across runs from the content cache.
#: Restricted to what is genuinely immutable and identified by digest.
CACHEABLE_ROLES: frozenset[str] = frozenset(
    {ArtifactRole.DATASET_AUDIO.value, ArtifactRole.CODE_BUNDLE.value}
)


class ManifestError(ValueError):
    """Raised when a manifest is inconsistent or cannot be built."""


def sha256_file(path: Path) -> tuple[str, int]:
    """Digest and size of one file, read in blocks."""
    digest = hashlib.sha256()
    total = 0
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(HASH_BLOCK), b""):
            digest.update(block)
            total += len(block)
    return digest.hexdigest(), total


@dataclass
class ArtifactEntry:
    """One file that has to exist on the worker.

    ``source_reference`` is how the control plane finds the bytes on its
    own disk. It is recorded for provenance and deliberately excluded
    from the manifest digest — where a file lives locally is not part of
    what the run is.
    """

    artifact_id: str
    role: str
    target_path: str
    sha256: str
    size_bytes: int
    required: bool = True
    source_reference: str = ""
    #: Set for dataset entries so a rights or leakage question can be
    #: traced back to the track it concerns without reopening the
    #: curated manifest.
    track_id: str | None = None

    def __post_init__(self) -> None:
        self.target_path = validate_relative(self.target_path)
        if len(self.sha256) != 64:
            raise ManifestError(f"{self.artifact_id}: sha256 must be a 64-character digest")
        if self.size_bytes < 0:
            raise ManifestError(f"{self.artifact_id}: size cannot be negative")

    @property
    def cacheable(self) -> bool:
        return self.role in CACHEABLE_ROLES

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical(self) -> dict[str, Any]:
        """The fields that define the artifact, for the digest.

        Local source path is excluded on purpose: the same run staged
        from two checkouts must produce the same manifest hash.
        """
        return {
            "artifact_id": self.artifact_id,
            "role": self.role,
            "target_path": self.target_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "required": self.required,
            "track_id": self.track_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArtifactEntry:
        known = {key: value for key, value in payload.items() if key in cls.__annotations__}
        return cls(**known)

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        role: str,
        target_path: str,
        artifact_id: str | None = None,
        required: bool = True,
        track_id: str | None = None,
    ) -> ArtifactEntry:
        path = Path(path)
        if not path.is_file():
            raise ManifestError(f"{path} does not exist and cannot be staged")
        digest, size = sha256_file(path)
        return cls(
            artifact_id=artifact_id or digest[:16],
            role=role,
            target_path=target_path,
            sha256=digest,
            size_bytes=size,
            required=required,
            source_reference=str(path),
            track_id=track_id,
        )


@dataclass
class RemoteArtifactManifest:
    """Everything one run needs remotely, with a stable identity."""

    run_id: str
    training_plan_sha256: str
    entries: list[ArtifactEntry] = field(default_factory=list)
    schema_version: str = MANIFEST_SCHEMA_VERSION
    protocol_version: str = REMOTE_PROTOCOL_VERSION
    built_at: str = field(default_factory=now)

    def add(self, entry: ArtifactEntry) -> ArtifactEntry:
        """Append an entry, refusing a second claim on one path.

        Two entries at one target path is not a merge to resolve — it
        means the caller believes two different files belong in the same
        place, and whichever arrived last would silently win.
        """
        for existing in self.entries:
            if existing.target_path == entry.target_path:
                if existing.sha256 == entry.sha256:
                    return existing
                raise ManifestError(
                    f"two different files both claim {entry.target_path}: "
                    f"{existing.sha256[:12]} and {entry.sha256[:12]}"
                )
        self.entries.append(entry)
        return entry

    @property
    def total_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self.entries)

    @property
    def required_entries(self) -> list[ArtifactEntry]:
        return [entry for entry in self.entries if entry.required]

    def by_role(self, role: str) -> list[ArtifactEntry]:
        return [entry for entry in self.entries if entry.role == role]

    def by_path(self, target_path: str) -> ArtifactEntry | None:
        for entry in self.entries:
            if entry.target_path == target_path:
                return entry
        return None

    def unique_contents(self) -> dict[str, list[ArtifactEntry]]:
        """Entries grouped by digest.

        Content addressing pays off here: a dataset that contains the
        same audio twice under two names transfers once. The grouping is
        also what the cache planner reads.
        """
        grouped: dict[str, list[ArtifactEntry]] = {}
        for entry in self.entries:
            grouped.setdefault(entry.sha256, []).append(entry)
        return grouped

    def transfer_bytes(self) -> int:
        """Bytes actually moved once identical content is deduplicated."""
        return sum(group[0].size_bytes for group in self.unique_contents().values())

    def canonical_dict(self) -> dict[str, Any]:
        """The manifest without build-time noise, for hashing."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "training_plan_sha256": self.training_plan_sha256,
            "entries": [
                entry.canonical()
                for entry in sorted(self.entries, key=lambda item: item.target_path)
            ],
        }

    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.canonical_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "run_id": self.run_id,
            "training_plan_sha256": self.training_plan_sha256,
            "manifest_sha256": self.digest(),
            "built_at": self.built_at,
            "entry_count": len(self.entries),
            "total_bytes": self.total_bytes,
            "transfer_bytes": self.transfer_bytes(),
            "entries": [
                entry.to_dict() for entry in sorted(self.entries, key=lambda item: item.target_path)
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RemoteArtifactManifest:
        manifest = cls(
            run_id=str(payload["run_id"]),
            training_plan_sha256=str(payload["training_plan_sha256"]),
            schema_version=str(payload.get("schema_version", MANIFEST_SCHEMA_VERSION)),
            protocol_version=str(payload.get("protocol_version", REMOTE_PROTOCOL_VERSION)),
            built_at=str(payload.get("built_at", "")),
        )
        manifest.entries = [ArtifactEntry.from_dict(item) for item in payload.get("entries", [])]

        recorded = payload.get("manifest_sha256")
        if recorded and recorded != manifest.digest():
            raise ManifestError(
                "the manifest's recorded digest does not match its contents; it has been "
                "edited or truncated since it was written"
            )
        return manifest

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def read(cls, path: Path) -> RemoteArtifactManifest:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class TransferPlan:
    """What a transfer will actually do, before it does it.

    Computed so an operator can see the cost of a dispatch — and so the
    disk check has a real number rather than an estimate. Every figure
    here is derived from measured file sizes; nothing is projected.
    """

    total_entries: int
    unique_contents: int
    total_bytes: int
    #: Bytes after deduplicating identical content within this manifest.
    deduplicated_bytes: int
    #: Bytes the worker already holds, by digest.
    cached_bytes: int
    #: What is left to move.
    upload_bytes: int
    upload_entries: int
    cached_entries: int

    @property
    def cache_hit_ratio(self) -> float:
        return self.cached_bytes / self.deduplicated_bytes if self.deduplicated_bytes else 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cache_hit_ratio"] = round(self.cache_hit_ratio, 4)
        return payload


def plan_transfer(
    manifest: RemoteArtifactManifest, *, present_digests: frozenset[str] = frozenset()
) -> TransferPlan:
    """Work out what has to move, given what the worker already has.

    Linear in the number of entries and linear in the number of distinct
    digests. A ten-thousand-entry manifest is grouped once and consulted
    with set membership, never compared pairwise.
    """
    grouped = manifest.unique_contents()
    cached_bytes = 0
    upload_bytes = 0
    cached_entries = 0
    upload_entries = 0

    for digest, entries in grouped.items():
        size = entries[0].size_bytes
        if digest in present_digests:
            cached_bytes += size
            cached_entries += len(entries)
        else:
            upload_bytes += size
            upload_entries += len(entries)

    return TransferPlan(
        total_entries=len(manifest.entries),
        unique_contents=len(grouped),
        total_bytes=manifest.total_bytes,
        deduplicated_bytes=sum(group[0].size_bytes for group in grouped.values()),
        cached_bytes=cached_bytes,
        upload_bytes=upload_bytes,
        upload_entries=upload_entries,
        cached_entries=cached_entries,
    )


@dataclass
class DiskRequirement:
    """How much space a run needs, and what is not known about it.

    Deliberately split. Artifact and dataset sizes are measured, so they
    are stated. Checkpoint size has never been measured for any LUBER
    configuration, so it is reported as unknown rather than folded into
    a total that would look authoritative.
    """

    artifact_bytes: int
    #: Multiplier over measured bytes, for trainer scratch and temp copies.
    safety_margin: float
    required_bytes: int
    checkpoint_bytes: int | None = None
    unknown: list[str] = field(default_factory=list)

    @property
    def required_mb(self) -> int:
        return int(self.required_bytes / (1024 * 1024))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["required_mb"] = self.required_mb
        return payload


def disk_requirement(
    plan: TransferPlan,
    *,
    safety_margin: float = 1.5,
    checkpoint_bytes: int | None = None,
) -> DiskRequirement:
    """Space needed on the worker, from measured sizes only.

    The margin covers the transfer's own temporary files and the
    trainer's scratch. It is not a guess about checkpoints: if nobody
    has measured what a checkpoint weighs, that is recorded as unknown
    and the caller decides what to do about it, rather than a number
    being invented to make the check pass.
    """
    unknown: list[str] = []
    required = int(plan.upload_bytes * safety_margin)
    if checkpoint_bytes is None:
        unknown.append(
            "checkpoint size has never been measured for any LUBER configuration, so the "
            "space training will need beyond its inputs is unknown"
        )
    else:
        required += checkpoint_bytes
    return DiskRequirement(
        artifact_bytes=plan.upload_bytes,
        safety_margin=safety_margin,
        required_bytes=required,
        checkpoint_bytes=checkpoint_bytes,
        unknown=unknown,
    )


__all__ = [
    "CACHEABLE_ROLES",
    "MANIFEST_SCHEMA_VERSION",
    "ArtifactEntry",
    "ArtifactRole",
    "DiskRequirement",
    "ManifestError",
    "RemoteArtifactManifest",
    "TransferPlan",
    "disk_requirement",
    "plan_transfer",
    "sha256_file",
]
