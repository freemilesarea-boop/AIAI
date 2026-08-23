"""Deterministic selection of a small pilot subset from a training library.

A bounded pilot trains on a handful of tracks, not the whole library.
Which handful matters more than it looks: pick them by iteration order
and the subset changes when the operator renames a folder; pick them at
random and two runs of the same "experiment" are not the same
experiment, so nothing measured on one transfers to the other.

So selection here is a pure function of the manifest. Tracks are keyed
by their audio SHA-256 — the one identifier that follows the bytes
rather than the filesystem — and drawn round-robin across source groups
so a subset of four does not come entirely from whichever group happens
to sort first. The same manifest and the same size always produce the
same subset, on any machine, in any directory.

Nothing here judges quality or musical fit. It selects; the caller
decides what the selection is good for.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

#: Below three tracks a pilot cannot show that the trainer handled more
#: than one piece of audio; above ten it stops being bounded.
PILOT_SUBSET_MIN = 3
PILOT_SUBSET_MAX = 10


class SubsetError(RuntimeError):
    """Raised when a subset cannot be selected honestly."""


@dataclass(frozen=True)
class SubsetMember:
    """One selected track, with the identity the selection was made on."""

    track_id: str
    audio_sha256: str
    source_group: str
    duration_seconds: float


@dataclass(frozen=True)
class PilotSubset:
    """A selection, plus enough context to audit how it was made."""

    dataset_id: str
    library_content_hash: str
    requested_size: int
    members: tuple[SubsetMember, ...]

    @property
    def track_ids(self) -> tuple[str, ...]:
        return tuple(member.track_id for member in self.members)

    @property
    def group_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for member in self.members:
            counts[member.source_group] = counts.get(member.source_group, 0) + 1
        return counts

    @property
    def total_duration_seconds(self) -> float:
        return sum(member.duration_seconds for member in self.members)

    def digest(self) -> str:
        """Identity of the selection itself.

        Over the audio digests only. Two selections of the same audio
        are the same selection even if the library was rebuilt, and a
        selection of different audio can never collide with it.
        """
        running = hashlib.sha256()
        for member in self.members:
            running.update(member.audio_sha256.encode("utf-8"))
        return running.hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "library_content_hash": self.library_content_hash,
            "requested_size": self.requested_size,
            "subset_digest": self.digest(),
            "track_count": len(self.members),
            "total_duration_seconds": round(self.total_duration_seconds, 3),
            "group_distribution": self.group_distribution,
            "tracks": [
                {
                    "track_id": member.track_id,
                    "audio_sha256": member.audio_sha256,
                    "source_group": member.source_group,
                    "duration_seconds": member.duration_seconds,
                }
                for member in self.members
            ],
        }


def _candidates(tracks: Iterable[dict[str, Any]]) -> list[SubsetMember]:
    """Trainable tracks only, as members.

    A track the rights record does not clear for training is not a
    candidate, whatever else recommends it. The manifest already
    derived that verdict; this reads it rather than re-deciding it.
    """
    members: list[SubsetMember] = []
    for raw in tracks:
        if not raw.get("training_allowed", False):
            continue
        digest = str(raw.get("audio_sha256", ""))
        if len(digest) != 64:
            raise SubsetError(
                f"track {raw.get('track_id', '?')!r} has no usable audio_sha256; "
                "selection needs a content identity, not a filename"
            )
        members.append(
            SubsetMember(
                track_id=str(raw["track_id"]),
                audio_sha256=digest,
                source_group=str(raw.get("source_group", "")),
                duration_seconds=float(raw.get("duration_seconds", 0.0)),
            )
        )
    return members


def select_pilot_subset(
    manifest: dict[str, Any],
    *,
    size: int,
    groups: Sequence[str] | None = None,
) -> PilotSubset:
    """Choose *size* tracks from *manifest*, the same way every time.

    ``groups`` restricts selection to the named source groups. Naming a
    group that the library does not contain is an error rather than an
    empty stratum: it is nearly always a typo, and silently returning a
    subset drawn from everything else would hide it.
    """
    if not PILOT_SUBSET_MIN <= size <= PILOT_SUBSET_MAX:
        raise SubsetError(
            f"pilot subset size {size} is outside the bounded range "
            f"{PILOT_SUBSET_MIN}-{PILOT_SUBSET_MAX}"
        )

    members = _candidates(manifest.get("tracks", []))
    if groups is not None:
        available = {member.source_group for member in members}
        missing = [name for name in groups if name not in available]
        if missing:
            raise SubsetError(
                f"source group(s) {missing} are not in the library; it has {sorted(available)}"
            )
        wanted = set(groups)
        members = [member for member in members if member.source_group in wanted]

    if len(members) < size:
        raise SubsetError(
            f"asked for {size} tracks but only {len(members)} are trainable candidates"
        )

    by_group: dict[str, list[SubsetMember]] = {}
    for member in members:
        by_group.setdefault(member.source_group, []).append(member)
    for bucket in by_group.values():
        bucket.sort(key=lambda member: member.audio_sha256)

    # Round-robin over groups in name order. Draining a group simply
    # drops it from the rotation, so an unbalanced library still fills
    # the subset instead of failing.
    order = sorted(by_group)
    chosen: list[SubsetMember] = []
    cursor = 0
    while len(chosen) < size:
        drew = False
        for name in order:
            bucket = by_group[name]
            if cursor < len(bucket):
                chosen.append(bucket[cursor])
                drew = True
                if len(chosen) == size:
                    break
        if not drew:  # pragma: no cover - guarded by the length check above
            break
        cursor += 1

    chosen.sort(key=lambda member: member.audio_sha256)
    return PilotSubset(
        dataset_id=str(manifest.get("dataset_id", "")),
        library_content_hash=str(manifest.get("content_hash", "")),
        requested_size=size,
        members=tuple(chosen),
    )
