"""Training track schema and vocal-style vocabulary.

Field choices are driven by two things: what upstream's LoRA trainer
actually consumes (`bpm`, `keyscale`, `timesignature`, `language`,
lyrics with section tags — see `docs/PHASE6_ACE_STEP_LORA_AUDIT.md`),
and the specific failures the Phase 5 human evaluation found.

The vocal-style vocabulary exists because the single loudest human
finding was an unwanted trot-like (뽕끼) delivery. If the dataset cannot
express "this is a modern K-pop vocal and that is not", the training
set cannot correct the bias it is meant to correct.

Descriptors are controlled terms, never artist names. Labelling data
with real performers would both invite rights problems and teach the
model to imitate identifiable people.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from luber_dataset.rights import RightsRecord


class VocalStyle(StrEnum):
    """Controlled vocal-identity descriptors for the target domains."""

    MODERN_KPOP_FEMALE = "modern_kpop_female"
    MODERN_KPOP_MALE = "modern_kpop_male"
    K_RNB_SOFT = "k_rnb_soft"
    INDIE_BREATHY = "indie_breathy"
    MODERN_BALLAD_CLEAN = "modern_ballad_clean"
    BAND_POP_CLEAN = "band_pop_clean"
    #: Present so unwanted styles can be labelled and excluded, not so
    #: they can be trained on.
    TRADITIONAL_TROT = "traditional_trot"
    OTHER = "other"
    INSTRUMENTAL = "instrumental"


#: Styles the Phase 6 pilot deliberately avoids over-representing. The
#: baseline already leans this way; more of it would entrench the bias.
DISCOURAGED_STYLES: frozenset[VocalStyle] = frozenset({VocalStyle.TRADITIONAL_TROT})


class Delivery(StrEnum):
    SMOOTH = "smooth"
    BREATHY = "breathy"
    POWERFUL = "powerful"
    CONVERSATIONAL = "conversational"
    RHYTHMIC = "rhythmic"
    EMOTIVE = "emotive"


class VibratoAmount(StrEnum):
    NONE = "none"
    SUBTLE = "subtle"
    MODERATE = "moderate"
    HEAVY = "heavy"


class VibratoCharacter(StrEnum):
    STRAIGHT = "straight"
    NATURAL = "natural"
    WIDE = "wide"
    #: The fast, wide oscillation characteristic of trot delivery.
    FAST_OSCILLATING = "fast_oscillating"


class PronunciationStyle(StrEnum):
    MODERN_STANDARD = "modern_standard"
    SOFT_CONTEMPORARY = "soft_contemporary"
    ARTICULATED = "articulated"
    TRADITIONAL = "traditional"


class QualityGrade(StrEnum):
    """Overall suitability as training material."""

    REFERENCE = "reference"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    REJECTED = "rejected"


ACCEPTABLE_GRADES: frozenset[QualityGrade] = frozenset(
    {QualityGrade.REFERENCE, QualityGrade.GOOD, QualityGrade.ACCEPTABLE}
)


@dataclass
class VocalAnnotation:
    """Vocal identity labels. Absent for instrumentals."""

    vocal_style: VocalStyle
    delivery: Delivery
    vibrato_amount: VibratoAmount
    vibrato_character: VibratoCharacter
    pronunciation_style: PronunciationStyle
    genre_vocal_identity: str

    def to_dict(self) -> dict[str, Any]:
        return {k: str(v) for k, v in asdict(self).items()}


@dataclass
class TrainingTrack:
    """One candidate track for a LUBER training set."""

    track_id: str
    source: str
    rights: RightsRecord

    audio_sha256: str
    duration_seconds: float
    sample_rate: int
    channels: int

    language: str
    genre: str
    subgenre: str
    vocal_gender: str
    lyrics_available: bool

    # Upstream training annotations (see the LoRA audit).
    bpm: int | None = None
    key_scale: str | None = None
    time_signature: str | None = None

    production_style: str = ""
    instrumentation: list[str] = field(default_factory=list)
    vocal: VocalAnnotation | None = None

    quality_grade: QualityGrade = QualityGrade.ACCEPTABLE
    audio_quality_flags: list[str] = field(default_factory=list)
    lyrics_qa_flags: list[str] = field(default_factory=list)

    caption: str = ""
    notes: str = ""

    @property
    def has_vocals(self) -> bool:
        return self.vocal_gender != "instrumental"

    @property
    def training_allowed(self) -> bool:
        """Derived, never asserted by hand."""
        from luber_dataset.rights import is_trainable

        return is_trainable(
            self.rights, has_lyrics=self.lyrics_available, has_vocals=self.has_vocals
        )

    @property
    def commercial_training_allowed(self) -> bool:
        return self.training_allowed and self.rights.commercial_training_allowed

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "track_id": self.track_id,
            "source": self.source,
            "rights": {
                k: str(v) if isinstance(v, StrEnum) else v for k, v in asdict(self.rights).items()
            },
            "audio_sha256": self.audio_sha256,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "language": self.language,
            "genre": self.genre,
            "subgenre": self.subgenre,
            "vocal_gender": self.vocal_gender,
            "lyrics_available": self.lyrics_available,
            "bpm": self.bpm,
            "key_scale": self.key_scale,
            "time_signature": self.time_signature,
            "production_style": self.production_style,
            "instrumentation": list(self.instrumentation),
            "vocal": self.vocal.to_dict() if self.vocal else None,
            "quality_grade": str(self.quality_grade),
            "audio_quality_flags": list(self.audio_quality_flags),
            "lyrics_qa_flags": list(self.lyrics_qa_flags),
            "caption": self.caption,
            "notes": self.notes,
            # Derived, so a manifest can never claim rights the record
            # does not support.
            "training_allowed": self.training_allowed,
            "commercial_training_allowed": self.commercial_training_allowed,
        }
        return data
