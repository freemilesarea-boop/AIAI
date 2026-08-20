"""Everything the factory decides by, in one place with a hash.

Two properties matter more than the individual numbers.

*Every threshold is configurable.* A quality tier is a policy, not a
fact, and a policy baked into code cannot be argued with or varied per
dataset. The measurements are objective; where the line sits is not.

*The configuration hashes.* A manifest produced under one policy and a
manifest produced under another are different artifacts even when the
audio is identical, so the cache is keyed by this hash and the dataset
lock records it. Change a threshold and the affected analysis is
recomputed; change nothing and the second run is free.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any

#: Bumped when a manifest record's *shape* changes. Consumers pin it.
SCHEMA_VERSION = "luber-dataset-factory/1"

#: Bumped when an analysis *algorithm* changes in a way that would give a
#: different answer for the same bytes. This invalidates cached analysis
#: independently of the schema, so a scoring change does not force a
#: re-decode and a decoder change does not force a re-hash.
DECODE_VERSION = "decode/1"
TECHNICAL_ANALYSIS_VERSION = "technical/1"
MUSICAL_ANALYSIS_VERSION = "musical/1"
FINGERPRINT_VERSION = "fingerprint/1"

FACTORY_VERSION = "luber-dataset-factory/1.0.0"


@dataclass(frozen=True)
class QualityThresholds:
    """Where the lines sit. Every one of these is a policy choice.

    The defaults are deliberately permissive about *flagging* and strict
    about *rejecting*: a flag is information, and only a few of them are
    grounds for exclusion. A pipeline that rejects on every flag throws
    away most of a real library.
    """

    min_duration_seconds: float = 20.0
    max_duration_seconds: float = 900.0
    #: Below CD rate the top octave does not exist to learn from.
    min_sample_rate: int = 44_100
    #: Share of samples at full scale that indicates real clipping damage.
    max_clipping_sample_ratio: float = 0.0005
    #: Crest factor below this suggests brickwalled mastering.
    min_crest_factor_db: float = 6.0
    max_dc_offset: float = 0.01
    #: Integrated loudness outside this range is extreme in one direction
    #: or the other; neither is fatal, both are worth knowing.
    min_integrated_lufs: float = -30.0
    max_integrated_lufs: float = -5.0
    #: Share of the file that is effectively silent.
    max_silence_ratio: float = 0.5
    #: Broadband correlation below this means much of the track cancels
    #: in mono.
    min_phase_correlation: float = 0.0
    #: A cutoff below this suggests the file was transcoded up from
    #: something lossy: the sample rate claims a top octave the audio
    #: does not contain. Set from measurement of decoded-back-to-WAV
    #: encodes of the same source — native 44.1 kHz reads 22050 Hz,
    #: MP3 320k reads 20144, 192k reads 18745, 128k reads 16667, and a
    #: brick-walled 15 kHz source reads 17054. 17.5 kHz separates
    #: 128k-grade and brick-walled material from everything above it.
    suspicious_bandwidth_hz: float = 17_500.0

    # ── tiering ──────────────────────────────────────────────────────
    #: Flags that place a track in REJECT however good the rest is.
    disqualifying_flags: tuple[str, ...] = (
        "DECODE_ERROR",
        "CORRUPT",
        "TOO_SHORT",
    )
    #: Score floors for each tier, applied after disqualification.
    tier_a_min_score: float = 0.90
    tier_b_min_score: float = 0.70
    tier_c_min_score: float = 0.40
    #: Weight removed from a perfect score per flag, by severity.
    severe_flag_penalty: float = 0.30
    moderate_flag_penalty: float = 0.12
    minor_flag_penalty: float = 0.04


@dataclass(frozen=True)
class DedupThresholds:
    """Conservative by construction.

    A false-positive merge silently deletes a distinct track from the
    dataset and nothing downstream can tell. A false negative leaves one
    extra track in a corpus of thousands. The two errors are not
    remotely equal, so every uncertain case goes to a human instead of
    being decided here.
    """

    #: Above this, two fingerprints are treated as the same audio in a
    #: different container and merged. Set from measurement: lossless
    #: transcodes score 1.000, and the highest-scoring *unrelated* pair
    #: measured 0.900. In practice only a lossless re-encode clears it,
    #: which is the intent — it is the one automatic merge the factory
    #: performs beyond byte identity.
    exact_audio_similarity: float = 0.995
    #: Between this and the above, a pair is reported for review and
    #: never merged. Set low enough to catch an AAC-128 re-encode of the
    #: same master, which measured 0.879. That inevitably also catches
    #: some unrelated pairs — reviewing a false alarm costs a minute,
    #: and merging two different songs costs one of them permanently.
    near_audio_similarity: float = 0.85
    #: Duration difference beyond this rules a pair out entirely, however
    #: similar the fingerprints. Two different songs can share a
    #: spectral shape; they rarely share it at the same length.
    max_duration_delta_seconds: float = 2.0


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.90
    validation: float = 0.05
    test: float = 0.05
    seed: int = 20260819

    def __post_init__(self) -> None:
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split fractions must sum to 1.0, got {total}")
        for name in ("train", "validation", "test"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"split fraction {name} must not be negative")


@dataclass(frozen=True)
class FactoryConfig:
    """The whole policy, hashable."""

    quality: QualityThresholds = field(default_factory=QualityThresholds)
    dedup: DedupThresholds = field(default_factory=DedupThresholds)
    split: SplitConfig = field(default_factory=SplitConfig)

    #: Whether tracks whose rights are unknown may enter a training
    #: export. Default false, and see ``provenance`` for why this is the
    #: one setting that must be turned on deliberately.
    include_rights_unknown: bool = False
    #: Minimum tier admitted to training.
    min_training_tier: str = "B"
    workers: int = 0  # 0 = choose from cpu count

    def with_overrides(self, **kwargs: Any) -> FactoryConfig:
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def configuration_hash(self) -> str:
        """Stable digest of every policy value.

        Sorted keys and a compact separator, so the hash depends on the
        values and not on how the dataclass happened to be laid out.
        """
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def analysis_key(self, stage: str) -> str:
        """Cache key component for one analysis stage.

        Deliberately per stage. A change to the quality thresholds must
        not invalidate a decode result, and a decoder change must not
        force every fingerprint to be recomputed — so each stage mixes
        its own algorithm version with the configuration digest rather
        than sharing one global key.
        """
        return f"{stage}:{self.configuration_hash()[:16]}"
