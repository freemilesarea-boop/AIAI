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
    """Controlled vocal-identity descriptors for the target domains.

    Gender is a separate field on the track, so it is deliberately not
    repeated here — a style is an idiom, not a voice type.
    """

    CONTEMPORARY_KPOP = "contemporary_kpop"
    CONTEMPORARY_KRNB = "contemporary_krnb"
    MODERN_BALLAD_CLEAN = "modern_ballad_clean"
    INDIE_BREATHY = "indie_breathy"
    BAND_POP_CLEAN = "band_pop_clean"
    #: Present so unwanted styles can be labelled and excluded, not so
    #: they can be trained on. These two are the Phase 5 failure modes.
    TRADITIONAL_TROT = "traditional_trot"
    BALLAD_LEGACY = "ballad_legacy"
    OTHER = "other"
    INSTRUMENTAL = "instrumental"


#: Styles the pilot deliberately avoids over-representing. The baseline
#: already leans this way — the human verdict named trot-like delivery
#: and dated vocal style explicitly — so more of it would entrench the
#: exact bias the training set exists to correct.
DISCOURAGED_STYLES: frozenset[VocalStyle] = frozenset(
    {VocalStyle.TRADITIONAL_TROT, VocalStyle.BALLAD_LEGACY}
)


class Delivery(StrEnum):
    SMOOTH = "smooth"
    BREATHY = "breathy"
    CLEAN = "clean"
    POWERFUL = "powerful"
    INTIMATE = "intimate"
    RESTRAINED = "restrained"
    CONVERSATIONAL = "conversational"
    RHYTHMIC = "rhythmic"
    EMOTIVE = "emotive"


class VocalTimbre(StrEnum):
    """Timbre, tracked separately from delivery.

    The Phase 5 evaluator described the baseline vocal as trot-like and
    dated; nasality and airiness are the two timbral axes that most
    distinguish that from a contemporary K-pop vocal, so they are
    labelled rather than folded into a single style tag.
    """

    AIRY = "airy"
    NASAL = "nasal"
    WARM = "warm"
    BRIGHT = "bright"
    NEUTRAL = "neutral"


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


class VocalPresence(StrEnum):
    """Whether a track has vocals, and whose.

    ``UNKNOWN`` is a first-class state, not a placeholder. A track
    nobody has listened to or annotated is unknown — inferring
    ``INSTRUMENTAL`` from the absence of a label would put a musical
    claim into the manifest that no evidence supports, and an
    instrumental label changes which rights are required and which
    rubric dimensions apply.
    """

    FEMALE = "female"
    MALE = "male"
    INSTRUMENTAL = "instrumental"
    UNKNOWN = "unknown"


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
    timbre: VocalTimbre = VocalTimbre.NEUTRAL

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
    def vocal_presence(self) -> VocalPresence:
        """Parsed vocal state; anything unrecognised reads as UNKNOWN."""
        try:
            return VocalPresence(self.vocal_gender)
        except ValueError:
            return VocalPresence.UNKNOWN

    @property
    def has_vocals(self) -> bool:
        """Conservative: unknown counts as *may have vocals*.

        Treating unknown as vocal-free would waive the performer-rights
        check on tracks that could well contain a voice.
        """
        return self.vocal_presence is not VocalPresence.INSTRUMENTAL

    @property
    def vocals_confirmed_absent(self) -> bool:
        """True only when a track was positively annotated instrumental."""
        return self.vocal_presence is VocalPresence.INSTRUMENTAL

    @property
    def vocal_annotation_status(self) -> str:
        if self.vocal is not None:
            return "ANNOTATED"
        if self.vocal_presence is VocalPresence.INSTRUMENTAL:
            return "INSTRUMENTAL_DECLARED"
        return "UNANNOTATED"

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
            "vocal_gender": str(self.vocal_presence),
            "vocal_annotation_status": self.vocal_annotation_status,
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
