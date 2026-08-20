"""Finding the same song twice, without merging two different ones.

Byte identity is not enough. The same master routinely exists as a WAV,
a FLAC and two MP3s at different bitrates, with different tags — four
files, four SHA-256s, one song. Training on all four teaches the model
that song is four times as important as its neighbours.

The asymmetry that governs every threshold here: **a false merge
silently deletes a distinct track and nothing downstream can tell, while
a false split leaves one extra track in a corpus of thousands.** Those
costs are nowhere near equal. So automatic merging happens only at
near-certainty, and everything between "certainly the same" and
"certainly different" is handed to a person.

The fingerprint is a Philips-style robust hash reduced to its most
codec-stable component: for each 100 ms frame, one bit per adjacent band
pair recording which of the two carried more energy. That is the
*spectral shape* of the frame, and it is what survives re-encoding.

Measured, not assumed. Three signals were re-encoded five ways each and
compared against their sources:

* lossless (FLAC): 1.000
* MP3 320k: 0.988 - 0.991
* MP3 192k: 0.949 - 0.981
* MP3 128k: 0.917 - 0.956
* AAC 128k: 0.879 - 0.946

and unrelated pairs scored 0.386 - 0.900. **Those ranges overlap.** An
AAC-128 copy of a track scored 0.879 while two genuinely different
tracks scored 0.900, so no single threshold separates same from
different once lossy encoding is involved.

That measurement decides the policy. Only a match at 0.995 or above —
in practice a lossless transcode — is merged automatically. Every lossy
re-encode lands in the review band and goes to a person, along with the
occasional unrelated pair that happens to score highly. Reviewing a few
false alarms is cheap; merging two different songs is not.

An earlier version differenced across time as well, which is the
textbook construction. It measured worse on exactly the case that
matters: 0.862 for a WAV/MP3 pair against 0.976 for this one, because a
codec's frame boundaries and encoder delay disturb time differences far
more than they disturb the shape of any single frame.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import pairwise
from typing import Any

import numpy as np

from luber_dataset.factory.config import DedupThresholds

#: Log-spaced band edges. Coarse on purpose: fine bands would track the
#: codec's decisions rather than the music.
FINGERPRINT_BANDS: tuple[float, ...] = (
    100.0,
    200.0,
    300.0,
    450.0,
    650.0,
    900.0,
    1_200.0,
    1_600.0,
    2_200.0,
    3_000.0,
    4_000.0,
    5_500.0,
    7_500.0,
)
#: One fingerprint frame per this many seconds.
FRAME_SECONDS = 0.1
#: Only the first stretch of a track is fingerprinted. Two files that
#: agree for three minutes are the same recording; carrying on costs
#: time and changes no answer.
MAX_FINGERPRINT_SECONDS = 180.0
#: A fingerprint shorter than this describes too little to compare.
MIN_FINGERPRINT_FRAMES = 40


class DuplicateType(StrEnum):
    NONE = "NONE"
    #: Identical bytes.
    EXACT_FILE = "EXACT_FILE"
    #: Different container or codec, same audio.
    EXACT_AUDIO = "EXACT_AUDIO"
    #: Similar enough to be suspicious, not enough to act on.
    NEAR_AUDIO = "NEAR_AUDIO"


class DedupDecision(StrEnum):
    KEEP = "KEEP"
    #: Folded into a canonical track automatically.
    MERGED = "MERGED"
    #: A person decides. The factory will not.
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


@dataclass
class DedupRecord:
    canonical_track_id: str
    duplicate_of: str | None = None
    duplicate_group_id: str | None = None
    duplicate_type: str = DuplicateType.NONE.value
    similarity_score: float | None = None
    dedup_decision: str = DedupDecision.KEEP.value
    all_source_paths: list[str] = field(default_factory=list)
    fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_track_id": self.canonical_track_id,
            "duplicate_of": self.duplicate_of,
            "duplicate_group_id": self.duplicate_group_id,
            "duplicate_type": self.duplicate_type,
            "similarity_score": self.similarity_score,
            "dedup_decision": self.dedup_decision,
            "all_source_paths": sorted(self.all_source_paths),
            "fingerprint": self.fingerprint,
        }


def compute_fingerprint(mono: np.ndarray, sample_rate: int) -> str | None:
    """A hex string that survives re-encoding.

    Returns ``None`` when the audio is too short to characterise —
    silence about a fingerprint is better than a fingerprint of noise,
    because a meaningless one would collide with other meaningless ones
    and merge unrelated files.
    """
    if sample_rate <= 0 or mono.size == 0:
        return None

    limit = int(MAX_FINGERPRINT_SECONDS * sample_rate)
    signal = mono[:limit]
    hop = max(1, int(FRAME_SECONDS * sample_rate))
    window_size = hop * 2
    if signal.size < window_size * MIN_FINGERPRINT_FRAMES:
        return None

    frames = 1 + (signal.size - window_size) // hop
    if frames < MIN_FINGERPRINT_FRAMES:
        return None

    window = np.hanning(window_size)
    freqs = np.fft.rfftfreq(window_size, 1.0 / sample_rate)
    slices = [(freqs >= low) & (freqs < high) for low, high in pairwise(FINGERPRINT_BANDS)]
    if not any(mask.any() for mask in slices):
        return None

    energies = np.empty((frames, len(slices)), dtype=np.float64)
    for index in range(frames):
        start = index * hop
        power = np.abs(np.fft.rfft(signal[start : start + window_size] * window)) ** 2
        for band, mask in enumerate(slices):
            energies[index, band] = float(power[mask].sum()) if mask.any() else 0.0

    # One bit per adjacent band pair per frame: was the upper band louder
    # than the lower one. Differencing across bands and not across time
    # removes the overall level — so a volume change cannot alter a
    # single bit — while leaving the frame's spectral shape, which is
    # the part a codec preserves.
    compressed = np.log1p(energies)
    bits = (np.diff(compressed, axis=1) > 0).astype(np.uint8).flatten()
    packed = np.packbits(bits)
    return str(packed.tobytes().hex())


def _unpack(fingerprint: str) -> np.ndarray | None:
    try:
        raw = bytes.fromhex(fingerprint)
    except ValueError:
        return None
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8))


def similarity(left: str | None, right: str | None) -> float | None:
    """Fraction of bits two fingerprints agree on, over their overlap.

    ``None`` when either side is missing or they share too little to
    compare — which must not be read as "different". An unknown is not a
    negative, and treating it as one is how a real duplicate gets in.
    """
    if not left or not right:
        return None
    a, b = _unpack(left), _unpack(right)
    if a is None or b is None:
        return None
    length = min(a.size, b.size)
    if length < MIN_FINGERPRINT_FRAMES * (len(FINGERPRINT_BANDS) - 2):
        return None
    agreement = float(np.count_nonzero(a[:length] == b[:length])) / length
    return float(round(agreement, 6))


@dataclass
class _Candidate:
    track_id: str
    sha256: str
    source_path: str
    fingerprint: str | None
    duration_seconds: float | None


def group_exact(candidates: list[_Candidate]) -> dict[str, list[_Candidate]]:
    """By content hash. Identical bytes are identical audio, always."""
    grouped: dict[str, list[_Candidate]] = defaultdict(list)
    for item in candidates:
        grouped[item.sha256].append(item)
    return grouped


def _comparable_duration(left: _Candidate, right: _Candidate, tolerance: float) -> bool:
    """Two songs can share a spectral shape; rarely at the same length.

    A missing duration is not evidence of a match, so a pair where
    either side is unknown does not clear this gate.
    """
    if left.duration_seconds is None or right.duration_seconds is None:
        return False
    return abs(left.duration_seconds - right.duration_seconds) <= tolerance


def analyse_duplicates(
    candidates: list[_Candidate], thresholds: DedupThresholds
) -> dict[str, DedupRecord]:
    """Assign every candidate a canonical track and a dedup verdict.

    Deterministic throughout: candidates are processed in sorted order
    and the canonical member of a group is the lowest track id, so the
    same inputs always elect the same representative.
    """
    ordered = sorted(candidates, key=lambda c: (c.track_id, c.source_path))
    records: dict[str, DedupRecord] = {}

    # ── exact file duplicates ────────────────────────────────────────
    canonical_for_hash: dict[str, _Candidate] = {}
    by_hash = group_exact(ordered)
    for digest, group in sorted(by_hash.items()):
        canonical = group[0]
        canonical_for_hash[digest] = canonical
        paths = sorted({item.source_path for item in group})
        group_id = f"dup_{digest[:12]}" if len(group) > 1 else None
        records[canonical.track_id] = DedupRecord(
            canonical_track_id=canonical.track_id,
            duplicate_group_id=group_id,
            duplicate_type=(
                DuplicateType.EXACT_FILE.value if len(group) > 1 else DuplicateType.NONE.value
            ),
            similarity_score=1.0 if len(group) > 1 else None,
            # Merging identical bytes is the one automatic merge that
            # cannot be wrong.
            dedup_decision=DedupDecision.KEEP.value,
            all_source_paths=paths,
            fingerprint=canonical.fingerprint,
        )

    # ── same audio, different container ──────────────────────────────
    unique = [canonical_for_hash[digest] for digest in sorted(canonical_for_hash)]
    merged_into: dict[str, str] = {}
    review_pairs: list[tuple[_Candidate, _Candidate, float]] = []

    for index, left in enumerate(unique):
        if left.track_id in merged_into:
            continue
        for right in unique[index + 1 :]:
            if right.track_id in merged_into:
                continue
            if not _comparable_duration(left, right, thresholds.max_duration_delta_seconds):
                continue
            score = similarity(left.fingerprint, right.fingerprint)
            if score is None:
                continue
            if score >= thresholds.exact_audio_similarity:
                merged_into[right.track_id] = left.track_id
                records[right.track_id].duplicate_of = left.track_id
                records[right.track_id].duplicate_type = DuplicateType.EXACT_AUDIO.value
                records[right.track_id].similarity_score = score
                records[right.track_id].dedup_decision = DedupDecision.MERGED.value
                records[right.track_id].canonical_track_id = left.track_id
                group_id = records[left.track_id].duplicate_group_id or f"dup_{left.track_id[4:16]}"
                records[left.track_id].duplicate_group_id = group_id
                records[right.track_id].duplicate_group_id = group_id
                records[left.track_id].all_source_paths = sorted(
                    set(records[left.track_id].all_source_paths)
                    | set(records[right.track_id].all_source_paths)
                )
            elif score >= thresholds.near_audio_similarity:
                review_pairs.append((left, right, score))

    # ── near matches: reported, never acted on ───────────────────────
    for left, right, score in review_pairs:
        for item, other in ((right, left), (left, right)):
            record = records[item.track_id]
            if record.dedup_decision == DedupDecision.MERGED.value:
                continue
            record.duplicate_type = DuplicateType.NEAR_AUDIO.value
            record.dedup_decision = DedupDecision.REVIEW_REQUIRED.value
            # Best evidence wins if a track is near-matched more than once.
            if record.similarity_score is None or score > record.similarity_score:
                record.similarity_score = score
                record.duplicate_of = other.track_id
            group_id = (
                records[left.track_id].duplicate_group_id
                or records[right.track_id].duplicate_group_id
                or f"near_{left.track_id[4:16]}"
            )
            records[left.track_id].duplicate_group_id = group_id
            records[right.track_id].duplicate_group_id = group_id

    return records
