"""Vocals and language: what can be stated, and what cannot.

This module says "I don't know" more than it says anything else, and
that is the correct output rather than a gap to be filled later.

*Vocal presence.* The repository has no vocal/instrumental classifier and
no labelled data to validate one against. A spectral heuristic could be
written in an afternoon and would be wrong often enough to matter — and
its errors would be invisible, because a wrongly-labelled instrumental
looks exactly like a correctly-labelled one in a manifest. So: an
operator statement is authoritative, and everything else is UNCERTAIN.

There is one measurement worth reporting alongside it. Lead vocals sit
almost entirely in the centre of a stereo image, so a track whose mid
channel carries far more 200-4000 Hz energy than its side channel is
*consistent with* a vocal. That is evidence, recorded as evidence, and
it never sets the class on its own.

*Language.* No language detector exists here, and there is no honest way
to infer language from audio without one. Guessing from a folder name is
worse than useless: `가요/` tells you what the operator filed it under,
not what is sung. Language comes from a sidecar or an embedded tag, or
it is ``unknown``.

*Gender.* Never inferred. Not from filenames, not from pitch. A filename
is not a person's voice, and average pitch does not determine gender.
Only an operator statement records it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from luber_dataset.factory.metadata import MetadataSource, Sidecar


class VocalClass(StrEnum):
    VOCAL = "VOCAL"
    INSTRUMENTAL = "INSTRUMENTAL"
    UNCERTAIN = "UNCERTAIN"


#: Language codes the schema recognises. `other` is a real answer;
#: `unknown` means nobody has established one.
SUPPORTED_LANGUAGES: frozenset[str] = frozenset({"ko", "en", "ja", "zh", "other", "unknown"})

#: Sidecar vocal_type strings mapped onto the class. Anything else is
#: recorded verbatim as the operator's own label and leaves the class
#: UNCERTAIN rather than being forced into a bucket.
_VOCAL_TYPE_MAP = {
    "vocal": VocalClass.VOCAL,
    "vocals": VocalClass.VOCAL,
    "male": VocalClass.VOCAL,
    "female": VocalClass.VOCAL,
    "mixed": VocalClass.VOCAL,
    "duet": VocalClass.VOCAL,
    "instrumental": VocalClass.INSTRUMENTAL,
    "inst": VocalClass.INSTRUMENTAL,
    "none": VocalClass.INSTRUMENTAL,
}

#: Vocal-type strings that name a performer's gender. Preserved as the
#: operator's own words; never derived.
_GENDER_LABELS = frozenset({"male", "female", "mixed", "duet"})


@dataclass
class VocalAssessment:
    vocal_class: str = VocalClass.UNCERTAIN.value
    vocal_confidence: float | None = None
    vocal_source: str = MetadataSource.NONE.value
    #: The operator's own gender label, when they supplied one.
    vocal_gender: str | None = None
    vocal_gender_source: str = MetadataSource.NONE.value
    #: Mid-vs-side energy in the vocal range. Evidence, not a verdict.
    centre_dominance_db: float | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocal_class": self.vocal_class,
            "vocal_confidence": self.vocal_confidence,
            "vocal_source": self.vocal_source,
            "vocal_gender": self.vocal_gender,
            "vocal_gender_source": self.vocal_gender_source,
            "centre_dominance_db": self.centre_dominance_db,
            "reason": self.reason,
        }


@dataclass
class LanguageAssessment:
    language: str = "unknown"
    language_confidence: float | None = None
    language_source: str = MetadataSource.NONE.value
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "language_confidence": self.language_confidence,
            "language_source": self.language_source,
            "reason": self.reason,
        }


@dataclass
class TextAssessment:
    """Lyrics and transcript. Neither is ever generated."""

    lyrics: str | None = None
    lyrics_source: str = MetadataSource.NONE.value
    lyrics_confidence: float | None = None
    transcript: str | None = None
    transcript_source: str = MetadataSource.NONE.value
    transcript_confidence: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lyrics": self.lyrics,
            "lyrics_source": self.lyrics_source,
            "lyrics_confidence": self.lyrics_confidence,
            "transcript": self.transcript,
            "transcript_source": self.transcript_source,
            "transcript_confidence": self.transcript_confidence,
            "notes": list(self.notes),
        }


def assess_vocals(
    sidecar: Sidecar | None,
    *,
    centre_dominance_db: float | None = None,
) -> VocalAssessment:
    """Operator statement if there is one; UNCERTAIN otherwise."""
    result = VocalAssessment(centre_dominance_db=centre_dominance_db)

    declared = (sidecar.get("vocal_type") if sidecar else None) or ""
    key = declared.strip().lower()
    if key:
        mapped = _VOCAL_TYPE_MAP.get(key)
        if mapped is not None:
            result.vocal_class = mapped.value
            result.vocal_confidence = 1.0
            result.vocal_source = MetadataSource.USER.value
            result.reason = f"operator declared vocal_type={declared!r}"
        else:
            result.reason = (
                f"operator supplied vocal_type={declared!r}, which is not one of the "
                "recognised values; class left UNCERTAIN rather than guessed"
            )
        if key in _GENDER_LABELS:
            result.vocal_gender = key
            result.vocal_gender_source = MetadataSource.USER.value
        return result

    result.reason = (
        "no operator statement and no validated vocal detector is available; "
        "a spectral heuristic would produce errors indistinguishable from labels"
    )
    return result


def assess_language(
    sidecar: Sidecar | None,
    embedded: dict[str, str] | None = None,
    *,
    lyrics: str | None = None,
) -> LanguageAssessment:
    """From an operator statement, an embedded tag, or the lyrics script.

    Script detection over supplied lyrics is the one inference made here,
    and only because it is a fact about the text rather than a guess
    about the audio: text containing Hangul is Korean text. It never runs
    on audio and never runs without lyrics.
    """
    result = LanguageAssessment()

    declared = (sidecar.get("language") if sidecar else None) or ""
    if declared.strip().lower() in SUPPORTED_LANGUAGES:
        result.language = declared.strip().lower()
        result.language_confidence = 1.0
        result.language_source = MetadataSource.USER.value
        result.reason = "operator declared the language"
        return result
    if declared.strip():
        result.reason = (
            f"operator supplied language={declared!r}, which is not a recognised code; "
            "left unknown rather than coerced"
        )
        return result

    tag = (embedded or {}).get("language", "").strip().lower()
    if tag in SUPPORTED_LANGUAGES and tag != "unknown":
        result.language = tag
        result.language_confidence = 0.6
        result.language_source = MetadataSource.EMBEDDED.value
        result.reason = "taken from an embedded container tag, which nobody verified"
        return result

    if lyrics:
        script = _detect_script(lyrics)
        if script is not None:
            result.language = script
            result.language_confidence = 0.8
            result.language_source = MetadataSource.SIDECAR.value
            result.reason = "script of the supplied lyrics, not analysis of the audio"
            return result

    result.reason = (
        "no operator statement, no usable tag and no lyrics to read; no language "
        "detector is available and a folder name is not evidence"
    )
    return result


def _detect_script(text: str) -> str | None:
    """Language from writing system. Only where the script is decisive."""
    hangul = sum(1 for ch in text if "가" <= ch <= "힣")
    kana = sum(1 for ch in text if "぀" <= ch <= "ヿ")
    han = sum(1 for ch in text if "一" <= ch <= "鿿")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    total = hangul + kana + han + latin
    if total < 8:
        return None

    if hangul / total > 0.3:
        return "ko"
    # Kana is decisive for Japanese; Han alone is not, because Japanese
    # and Chinese share the characters.
    if kana / total > 0.15:
        return "ja"
    if han / total > 0.3 and kana == 0:
        return "zh"
    if latin / total > 0.7:
        return "en"
    return None


def assess_text(
    sidecar: Sidecar | None,
    lyrics_file: tuple[str, str] | None,
    embedded: dict[str, str] | None = None,
) -> TextAssessment:
    """Collect lyrics from the places a human may have put them.

    No ASR runs and none is available, so ``transcript`` is always null.
    That is a deliberate absence: an ASR transcript recorded in the
    ``lyrics`` field would be a machine's guess wearing a human's label.
    """
    result = TextAssessment()

    if sidecar is not None and sidecar.get("lyrics"):
        result.lyrics = str(sidecar.get("lyrics"))
        result.lyrics_source = MetadataSource.USER.value
        result.lyrics_confidence = 1.0
    elif lyrics_file is not None:
        text, path = lyrics_file
        result.lyrics = text
        result.lyrics_source = MetadataSource.SIDECAR.value
        result.lyrics_confidence = 0.9
        result.notes.append(f"read from {path}")
    elif embedded and embedded.get("lyrics"):
        result.lyrics = embedded["lyrics"]
        result.lyrics_source = MetadataSource.EMBEDDED.value
        result.lyrics_confidence = 0.5
        result.notes.append("embedded container tag; unverified")

    result.notes.append(
        "no speech recogniser is configured; transcript is absent rather than generated"
    )
    return result


def centre_dominance(mid_energy_db: float | None, side_energy_db: float | None) -> float | None:
    """How far the centre sits above the sides, in dB.

    Reported as evidence for a human reading the review queue. A large
    positive value is consistent with a centred lead vocal and equally
    consistent with a mono-ish mix of anything else, which is precisely
    why it does not decide the class.
    """
    if mid_energy_db is None or side_energy_db is None:
        return None
    return round(mid_energy_db - side_energy_db, 3)
