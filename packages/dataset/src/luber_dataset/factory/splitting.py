"""Eligibility, and splits that do not leak.

Two jobs, joined because the second depends on the first.

**Eligibility** is four independent questions, not one. A track can be
worth analysing and unfit to train on; it can be fit to train on and
held out for test. Every "no" records why, because a dataset that shrank
by six hundred tracks and cannot say which or why is not auditable.

**Splitting** has one failure mode that matters. If the same song
appears in train and in test — as a duplicate, an alternate mix, another
track from the same album — the test score measures memorisation and
reports it as generalisation. The number comes out *better*, which is
why the bug survives: nobody investigates a good result.

So tracks are grouped before they are split, and the group is what gets
assigned. The grouping key is the strongest identity available: a
duplicate group if one exists, otherwise artist and album, otherwise the
containing folder, and only a lone track with none of those is split on
its own id.

Determinism comes from hashing the group key with the seed rather than
from shuffling a list. A hash does not care what order it is given
things in, so adding one track to a library cannot move an unrelated
track from train to test.
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from luber_dataset.factory.config import SplitConfig
from luber_dataset.factory.dedup import DedupDecision, DedupRecord
from luber_dataset.factory.provenance import Provenance
from luber_dataset.factory.quality import QualityAssessment, QualityTier, meets_tier


class Split(StrEnum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"
    EXCLUDED = "EXCLUDED"


# ── eligibility reasons ──────────────────────────────────────────────
RIGHTS_UNKNOWN = "RIGHTS_UNKNOWN"
RIGHTS_DENIED = "RIGHTS_DENIED"
RIGHTS_HARD_BLOCK = "RIGHTS_HARD_BLOCK"
QUALITY_REJECTED = "QUALITY_REJECTED"
QUALITY_BELOW_MINIMUM_TIER = "QUALITY_BELOW_MINIMUM_TIER"
DUPLICATE_OF_ANOTHER_TRACK = "DUPLICATE_OF_ANOTHER_TRACK"
NEAR_DUPLICATE_REVIEW_REQUIRED = "NEAR_DUPLICATE_REVIEW_REQUIRED"
DECODE_FAILED = "DECODE_FAILED"
TOO_SHORT_FOR_TRAINING = "TOO_SHORT_FOR_TRAINING"


@dataclass
class Eligibility:
    analysis_eligible: bool = True
    training_eligible: bool = False
    validation_eligible: bool = False
    test_eligible: bool = False
    eligibility_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_eligible": self.analysis_eligible,
            "training_eligible": self.training_eligible,
            "validation_eligible": self.validation_eligible,
            "test_eligible": self.test_eligible,
            "eligibility_reasons": sorted(set(self.eligibility_reasons)),
        }


def assess_eligibility(
    *,
    decoded: bool,
    quality: QualityAssessment,
    provenance: Provenance,
    dedup: DedupRecord,
    min_tier: str,
    include_rights_unknown: bool,
) -> Eligibility:
    """Decide what this track may be used for, and record every refusal.

    Analysis eligibility is deliberately generous: measuring a file is
    not using it, and the operator cannot decide about audio they have
    no measurements for. Training eligibility is where every gate binds.
    """
    result = Eligibility(analysis_eligible=decoded)
    reasons: list[str] = []

    if not decoded:
        reasons.append(DECODE_FAILED)

    if provenance.hard_blocks:
        # Self-model output and unlawful acquisition. No configuration
        # flag reaches these; there is no override and there should not
        # be one.
        reasons.append(RIGHTS_HARD_BLOCK)
    elif provenance.commercial_training_allowed == "FALSE":
        reasons.append(RIGHTS_DENIED)
    elif not provenance.training_permitted:
        reasons.append(RIGHTS_UNKNOWN)

    if quality.quality_tier == QualityTier.REJECT.value:
        reasons.append(QUALITY_REJECTED)
    elif not meets_tier(quality.quality_tier, min_tier):
        reasons.append(QUALITY_BELOW_MINIMUM_TIER)

    if dedup.dedup_decision == DedupDecision.MERGED.value:
        reasons.append(DUPLICATE_OF_ANOTHER_TRACK)
    elif dedup.dedup_decision == DedupDecision.REVIEW_REQUIRED.value:
        reasons.append(NEAR_DUPLICATE_REVIEW_REQUIRED)

    # `include_rights_unknown` waives exactly one reason and nothing
    # else. It cannot clear a hard block, a denial, a quality rejection
    # or a duplicate — an override that waived everything would not be
    # an override, it would be an off switch.
    blocking = set(reasons)
    if include_rights_unknown:
        blocking.discard(RIGHTS_UNKNOWN)

    eligible = not blocking
    result.training_eligible = eligible
    result.validation_eligible = eligible
    result.test_eligible = eligible
    result.eligibility_reasons = sorted(set(reasons))
    return result


def group_key(
    *,
    track_id: str,
    duplicate_group_id: str | None,
    artist: str | None,
    album: str | None,
    parent_directory: str | None,
) -> str:
    """The identity a split is assigned to.

    Strongest available signal wins. Two tracks sharing a duplicate
    group are the same recording; two sharing an artist and album came
    off the same session and share far more than a random pair; two in
    the same folder were filed together by someone who thought they
    belonged together.
    """
    if duplicate_group_id:
        return f"dup:{duplicate_group_id}"
    normalised_artist = _normalise(artist)
    normalised_album = _normalise(album)
    if normalised_artist and normalised_album:
        return f"aa:{normalised_artist}|{normalised_album}"
    if normalised_artist:
        return f"a:{normalised_artist}"
    if parent_directory:
        return f"d:{_normalise(parent_directory)}"
    return f"t:{track_id}"


def _normalise(value: str | None) -> str:
    if not value:
        return ""
    return unicodedata.normalize("NFC", value).strip().casefold()


def _fraction(key: str, seed: int) -> float:
    """A stable number in [0, 1) for this group.

    Hashing rather than shuffling is what makes the split insensitive to
    input order and to the size of the dataset: a group's number depends
    on the group and the seed, and on nothing else in the run.
    """
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def assign_splits(
    groups: dict[str, list[str]],
    config: SplitConfig,
) -> dict[str, str]:
    """Map every track id to a split, keeping whole groups together.

    Groups are assigned, not tracks, so a duplicate pair or an album
    lands entirely on one side. Exact proportions therefore cannot be
    guaranteed — a group is indivisible — and honouring the boundary
    matters more than hitting 90/5/5 precisely.
    """
    train_edge = config.train
    validation_edge = config.train + config.validation

    assignment: dict[str, str] = {}
    for key in sorted(groups):
        position = _fraction(key, config.seed)
        if position < train_edge:
            split = Split.TRAIN.value
        elif position < validation_edge:
            split = Split.VALIDATION.value
        else:
            split = Split.TEST.value
        for track_id in sorted(groups[key]):
            assignment[track_id] = split
    return assignment


def build_groups(records: list[tuple[str, str]]) -> dict[str, list[str]]:
    """``[(track_id, group_key)]`` -> ``{group_key: [track_id]}``."""
    grouped: dict[str, list[str]] = {}
    for track_id, key in records:
        grouped.setdefault(key, []).append(track_id)
    return grouped


def verify_no_leakage(groups: dict[str, list[str]], assignment: dict[str, str]) -> list[str]:
    """Group keys whose members landed in more than one split.

    Asserted rather than assumed. This is the invariant whose violation
    makes evaluation results *better* and therefore never gets noticed.
    """
    leaked: list[str] = []
    for key, members in groups.items():
        splits = {assignment.get(track_id) for track_id in members}
        splits.discard(None)
        splits.discard(Split.EXCLUDED.value)
        if len(splits) > 1:
            leaked.append(key)
    return sorted(leaked)
