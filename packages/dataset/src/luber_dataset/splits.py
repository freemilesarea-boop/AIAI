"""Deterministic train / validation / evaluation splits.

The thing this module exists to prevent is quiet contamination. A track
that appears in both the training set and the evaluation set turns every
later measurement into a statement about memorisation, and the mistake
leaves no trace: the numbers simply come out better than they should.

So splitting here is a pure function — of the library manifest and a
seed, and of nothing else. Tracks are keyed by their audio SHA-256, the
one identifier that follows the bytes rather than the filesystem, and
assigned round-robin across source groups so no split is drawn entirely
from whichever folder sorts first. The same library and seed give the
same three sets on any machine, in any directory.

Exclusivity is a property of the construction, not a hope about it:
:func:`build_experiment_splits` assigns each digest exactly once and
:func:`leakage_report` re-checks the result independently. Both, because
a builder that is correct today can stop being correct, and the check
that catches it must not share its logic.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

#: The three roles a track can hold. A track holds exactly one.
TRAIN = "TRAIN"
VALIDATION = "VALIDATION"
EVALUATION = "EVALUATION"
SPLIT_NAMES: tuple[str, ...] = (TRAIN, VALIDATION, EVALUATION)


class SplitError(RuntimeError):
    """Raised when a split cannot be built honestly."""


@dataclass(frozen=True)
class SplitMember:
    """One track, and the identity it was assigned on."""

    track_id: str
    audio_sha256: str
    source_group: str
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "audio_sha256": self.audio_sha256,
            "source_group": self.source_group,
            "duration_seconds": self.duration_seconds,
        }


@dataclass(frozen=True)
class Split:
    """One named set of tracks."""

    name: str
    members: tuple[SplitMember, ...]

    @property
    def track_ids(self) -> tuple[str, ...]:
        return tuple(member.track_id for member in self.members)

    @property
    def digests(self) -> frozenset[str]:
        return frozenset(member.audio_sha256 for member in self.members)

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
        """Identity of the set, over its audio and nothing else."""
        running = hashlib.sha256()
        running.update(self.name.encode("utf-8"))
        for member in sorted(self.members, key=lambda item: item.audio_sha256):
            running.update(b"\0")
            running.update(member.audio_sha256.encode("utf-8"))
        return running.hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "digest": self.digest(),
            "track_count": len(self.members),
            "total_duration_seconds": round(self.total_duration_seconds, 3),
            "group_distribution": self.group_distribution,
            "tracks": [member.to_dict() for member in self.members],
        }


@dataclass(frozen=True)
class ExperimentSplits:
    """Three exclusive sets, plus what they were built from."""

    dataset_id: str
    library_content_hash: str
    seed: int
    train: Split
    validation: Split
    evaluation: Split

    @property
    def splits(self) -> tuple[Split, ...]:
        return (self.train, self.validation, self.evaluation)

    @property
    def track_count(self) -> int:
        return sum(len(split.members) for split in self.splits)

    def digest(self) -> str:
        """Identity of the whole allocation, order-independent per split."""
        running = hashlib.sha256()
        running.update(str(self.seed).encode("utf-8"))
        for split in self.splits:
            running.update(b"\0")
            running.update(split.digest().encode("utf-8"))
        return running.hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "library_content_hash": self.library_content_hash,
            "seed": self.seed,
            "splits_digest": self.digest(),
            "track_count": self.track_count,
            "train": self.train.to_dict(),
            "validation": self.validation.to_dict(),
            "evaluation": self.evaluation.to_dict(),
        }


@dataclass(frozen=True)
class LeakageFinding:
    """One way two splits overlap."""

    kind: str
    left: str
    right: str
    identities: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "left": self.left,
            "right": self.right,
            "identities": list(self.identities),
        }


@dataclass(frozen=True)
class LeakageReport:
    """Whether an allocation is contaminated, and where."""

    passed: bool
    findings: tuple[LeakageFinding, ...] = ()
    checked_pairs: tuple[tuple[str, str], ...] = ()
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "detail": self.detail,
            "checked_pairs": [list(pair) for pair in self.checked_pairs],
            "findings": [finding.to_dict() for finding in self.findings],
        }


def _candidates(tracks: Iterable[dict[str, Any]]) -> list[SplitMember]:
    """Trainable tracks, as members, refusing anything without identity."""
    members: list[SplitMember] = []
    for raw in tracks:
        if not raw.get("training_allowed", False):
            continue
        digest = str(raw.get("audio_sha256", ""))
        if len(digest) != 64:
            raise SplitError(
                f"track {raw.get('track_id', '?')!r} has no usable audio_sha256; "
                "a split needs a content identity, not a filename"
            )
        members.append(
            SplitMember(
                track_id=str(raw["track_id"]),
                audio_sha256=digest,
                source_group=str(raw.get("source_group", "")),
                duration_seconds=float(raw.get("duration_seconds", 0.0)),
            )
        )
    return members


def _ordered(members: Sequence[SplitMember], seed: int) -> list[SplitMember]:
    """A stable order that the seed can change but the filesystem cannot.

    Sorting by a seeded hash of the audio digest rather than shuffling a
    list: the result depends on the seed and the content, and on nothing
    about how the manifest happened to be written.
    """

    def key(member: SplitMember) -> str:
        return hashlib.sha256(f"{seed}:{member.audio_sha256}".encode()).hexdigest()

    return sorted(members, key=key)


def build_experiment_splits(
    manifest: dict[str, Any],
    *,
    train_size: int,
    validation_size: int,
    evaluation_size: int,
    seed: int,
) -> ExperimentSplits:
    """Allocate three exclusive sets from *manifest*, reproducibly.

    Each group's candidates are ordered by a seeded hash and then dealt
    round-robin across groups, evaluation first. Evaluation is dealt
    first on purpose: it is the set whose composition matters most and
    the one that must not be whatever happened to be left over.
    """
    sizes = {
        EVALUATION: int(evaluation_size),
        VALIDATION: int(validation_size),
        TRAIN: int(train_size),
    }
    for name, size in sizes.items():
        if size < 0:
            raise SplitError(f"{name} size {size} is negative")
    wanted = sum(sizes.values())
    if wanted <= 0:
        raise SplitError("a split of nothing is not a split")

    members = _candidates(manifest.get("tracks", []))
    # One digest, one track. A library holding the same audio twice must
    # not be able to place it in two splits.
    unique: dict[str, SplitMember] = {}
    for member in members:
        unique.setdefault(member.audio_sha256, member)
    members = list(unique.values())

    if len(members) < wanted:
        raise SplitError(
            f"asked for {wanted} tracks across three splits and only {len(members)} "
            "distinct trainable candidates exist"
        )

    by_group: dict[str, list[SplitMember]] = {}
    for member in _ordered(members, seed):
        by_group.setdefault(member.source_group, []).append(member)
    order = sorted(by_group)
    cursors = dict.fromkeys(order, 0)

    assigned: dict[str, list[SplitMember]] = {name: [] for name in SPLIT_NAMES}
    taken: set[str] = set()
    for name in (EVALUATION, VALIDATION, TRAIN):
        target = sizes[name]
        while len(assigned[name]) < target:
            drew = False
            for group in order:
                if len(assigned[name]) >= target:
                    break
                bucket = by_group[group]
                index = cursors[group]
                while index < len(bucket) and bucket[index].audio_sha256 in taken:
                    index += 1
                cursors[group] = index
                if index >= len(bucket):
                    continue
                member = bucket[index]
                cursors[group] = index + 1
                taken.add(member.audio_sha256)
                assigned[name].append(member)
                drew = True
            if not drew:  # pragma: no cover - guarded by the length check
                raise SplitError(f"ran out of candidates while filling {name}")

    def freeze(name: str) -> Split:
        ordered = sorted(assigned[name], key=lambda member: member.audio_sha256)
        return Split(name=name, members=tuple(ordered))

    return ExperimentSplits(
        dataset_id=str(manifest.get("dataset_id", "")),
        library_content_hash=str(manifest.get("content_hash", "")),
        seed=int(seed),
        train=freeze(TRAIN),
        validation=freeze(VALIDATION),
        evaluation=freeze(EVALUATION),
    )


def leakage_report(splits: ExperimentSplits) -> LeakageReport:
    """Re-check exclusivity from the outside.

    Deliberately independent of the builder. It compares every pair of
    splits by audio digest *and* by track id, because the two can
    disagree — the same audio filed twice under different ids is exactly
    the contamination a track-id check alone would miss.
    """
    named = {split.name: split for split in splits.splits}
    pairs = ((TRAIN, VALIDATION), (TRAIN, EVALUATION), (VALIDATION, EVALUATION))
    findings: list[LeakageFinding] = []

    for left, right in pairs:
        shared_audio = sorted(named[left].digests & named[right].digests)
        if shared_audio:
            findings.append(
                LeakageFinding(
                    kind="AUDIO_DIGEST_COLLISION",
                    left=left,
                    right=right,
                    identities=tuple(shared_audio),
                )
            )
        shared_ids = sorted(set(named[left].track_ids) & set(named[right].track_ids))
        if shared_ids:
            findings.append(
                LeakageFinding(
                    kind="TRACK_ID_COLLISION",
                    left=left,
                    right=right,
                    identities=tuple(shared_ids),
                )
            )

    for split in splits.splits:
        seen: dict[str, int] = {}
        for member in split.members:
            seen[member.audio_sha256] = seen.get(member.audio_sha256, 0) + 1
        repeated = sorted(digest for digest, count in seen.items() if count > 1)
        if repeated:
            findings.append(
                LeakageFinding(
                    kind="DUPLICATE_WITHIN_SPLIT",
                    left=split.name,
                    right=split.name,
                    identities=tuple(repeated),
                )
            )

    if findings:
        summary = ", ".join(
            f"{finding.kind} between {finding.left} and {finding.right} ({len(finding.identities)})"
            for finding in findings
        )
        return LeakageReport(
            passed=False,
            findings=tuple(findings),
            checked_pairs=pairs,
            detail=f"contaminated: {summary}",
        )

    return LeakageReport(
        passed=True,
        checked_pairs=pairs,
        detail=(
            f"{splits.track_count} track(s) across three splits, no shared audio digest "
            "and no shared track id between any pair"
        ),
    )


__all__ = [
    "EVALUATION",
    "SPLIT_NAMES",
    "TRAIN",
    "VALIDATION",
    "ExperimentSplits",
    "LeakageFinding",
    "LeakageReport",
    "Split",
    "SplitError",
    "SplitMember",
    "build_experiment_splits",
    "leakage_report",
]
