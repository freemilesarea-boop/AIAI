"""Rights validation for training data.

Two facts about a track are independent and must not be conflated:

* **where the audio came from** (`origin_type`) — a human recording, a
  generative model, or a hybrid;
* **whether we may train on it** (`training_rights_status`).

An earlier revision rejected every AI-generated file outright. That was
too coarse: AI-generated audio the operator owns and has cleared is
legitimate training material, while a human recording with no licence
is not. Origin does not decide eligibility; rights do.

Two things remain absolute:

1. **Self-model output is never trainable.** Audio this project's own
   ACE-Step pipeline produced is refused regardless of rights, because
   training a model on its own output teaches it back its own
   artifacts — and the Phase 5 human verdict rated that output 2/10.
2. **A file path is not a licence.** Nothing derived from a folder
   name, a filename, or the mere fact that a file exists on disk may
   set `CONFIRMED`. Only an operator decision backed by a documented
   rights record can do that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class OriginType(StrEnum):
    """How the audio was produced."""

    HUMAN_RECORDED = "HUMAN_RECORDED"
    AI_GENERATED = "AI_GENERATED"
    HYBRID = "HYBRID"
    #: Produced by this project's own ACE-Step pipeline. Never trainable.
    SELF_MODEL_OUTPUT = "SELF_MODEL_OUTPUT"
    UNKNOWN = "UNKNOWN"


class TrainingRightsStatus(StrEnum):
    """Whether we may train on this track."""

    CONFIRMED = "CONFIRMED"
    UNVERIFIED = "UNVERIFIED"
    DENIED = "DENIED"


class RightsBasis(StrEnum):
    """How the right to train was established, when it was."""

    ORIGINAL_WORK = "ORIGINAL_WORK"
    LICENSED_FOR_TRAINING = "LICENSED_FOR_TRAINING"
    PUBLIC_DOMAIN = "PUBLIC_DOMAIN"
    CC_TRAINING_PERMITTED = "CC_TRAINING_PERMITTED"
    RIGHTS_HOLDER_PERMISSION = "RIGHTS_HOLDER_PERMISSION"
    #: Output of a generative service whose terms grant the operator
    #: commercial rights, where the operator has confirmed this.
    AI_SERVICE_OUTPUT_OWNED = "AI_SERVICE_OUTPUT_OWNED"
    NONE = "NONE"


class SourceClass(StrEnum):
    """Reporting classification of a discovered candidate."""

    AI_GENERATED_RIGHTS_CLEARED = "AI_GENERATED_RIGHTS_CLEARED"
    AI_GENERATED_RIGHTS_UNVERIFIED = "AI_GENERATED_RIGHTS_UNVERIFIED"
    SELF_MODEL_GENERATED = "SELF_MODEL_GENERATED"
    HUMAN_PRODUCED_RIGHTS_CLEARED = "HUMAN_PRODUCED_RIGHTS_CLEARED"
    COMMERCIAL_REFERENCE_UNVERIFIED = "COMMERCIAL_REFERENCE_UNVERIFIED"
    UNKNOWN = "UNKNOWN"


#: Classes usable as listening/reference targets but never placed in a
#: training manifest.
REFERENCE_ONLY_CLASSES: frozenset[SourceClass] = frozenset(
    {SourceClass.COMMERCIAL_REFERENCE_UNVERIFIED}
)

#: Classes that may enter a training manifest once rights are confirmed.
TRAINABLE_CLASSES: frozenset[SourceClass] = frozenset(
    {
        SourceClass.AI_GENERATED_RIGHTS_CLEARED,
        SourceClass.HUMAN_PRODUCED_RIGHTS_CLEARED,
    }
)

#: Provenance describing unlawful acquisition. Refused regardless of any
#: rights claim: this is about how audio was obtained, not about whether
#: a model made it.
UNLAWFUL_ACQUISITION_MARKERS: tuple[str, ...] = (
    "scrape",
    "scraped",
    "crawler",
    "crawled",
    "torrent",
    "youtube-dl",
    "yt-dlp",
    "ripped",
    "leaked",
    "pirated",
)

#: Markers indicating this project's own engine produced the audio.
SELF_MODEL_MARKERS: tuple[str, ...] = (
    "acestep",
    "ace-step",
    "luber-generated",
    "self-model",
)


class RightsError(Exception):
    """Raised when a track may not be used for training."""


@dataclass(frozen=True)
class RightsRecord:
    """Documented basis for training on one track."""

    origin_type: OriginType
    training_rights_status: TrainingRightsStatus
    basis: RightsBasis
    #: Where the audio came from, in plain words.
    source: str
    rights_holder: str
    #: Contract, licence id, service account, or file reference.
    document_reference: str
    #: ISO date the rights were confirmed.
    confirmed_on: str
    #: Confirmed separately — one licence does not imply the others.
    audio_use_confirmed: bool
    lyrics_rights_confirmed: bool
    performer_rights_confirmed: bool
    commercial_training_allowed: bool
    notes: str = ""


def _matches(haystack: str, markers: tuple[str, ...]) -> str | None:
    """Word-boundary match so "LUBER studio" never trips "udio"."""
    for marker in markers:
        if re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", haystack):
            return marker
    return None


def validate_rights(record: RightsRecord, *, has_lyrics: bool, has_vocals: bool) -> None:
    """Raise unless this track may be used for commercial ML training.

    ``has_lyrics`` and ``has_vocals`` decide which sub-rights apply, so
    an instrumental is not blocked for lacking lyric clearance it does
    not need.
    """
    # 1. Self-model output: refused before anything else is considered.
    if record.origin_type is OriginType.SELF_MODEL_OUTPUT:
        raise RightsError(
            "origin is this project's own model output; training ACE-Step on "
            "ACE-Step output is never permitted"
        )

    haystack = f"{record.source} {record.document_reference} {record.notes}".lower()
    if (marker := _matches(haystack, SELF_MODEL_MARKERS)) is not None:
        raise RightsError(
            f"source describes this project's own model output ({marker!r}); "
            "self-model audio is never trainable"
        )

    # 2. Unlawful acquisition disqualifies whatever the claim says.
    if (marker := _matches(haystack, UNLAWFUL_ACQUISITION_MARKERS)) is not None:
        raise RightsError(
            f"source describes unlawful acquisition ({marker!r}); "
            "no rights claim can cure how the audio was obtained"
        )

    # 3. Rights must be affirmatively confirmed. AI origin is not a
    #    reason to refuse; unconfirmed rights are.
    if record.training_rights_status is not TrainingRightsStatus.CONFIRMED:
        raise RightsError(
            f"training_rights_status is {record.training_rights_status}; "
            "training requires CONFIRMED"
        )
    if record.basis is RightsBasis.NONE:
        raise RightsError("rights are marked CONFIRMED but no basis is recorded")

    for field_name in ("source", "rights_holder", "document_reference", "confirmed_on"):
        if not str(getattr(record, field_name)).strip():
            raise RightsError(f"rights record is missing {field_name}")

    if not record.audio_use_confirmed:
        raise RightsError("audio use rights are not confirmed")
    if not record.commercial_training_allowed:
        raise RightsError("commercial ML training rights are not confirmed")
    if has_lyrics and not record.lyrics_rights_confirmed:
        raise RightsError("track has lyrics but lyrics rights are not confirmed")
    if has_vocals and not record.performer_rights_confirmed:
        raise RightsError("track has vocals but performer rights are not confirmed")


def is_trainable(record: RightsRecord, *, has_lyrics: bool, has_vocals: bool) -> bool:
    """Non-raising form of :func:`validate_rights`."""
    try:
        validate_rights(record, has_lyrics=has_lyrics, has_vocals=has_vocals)
    except RightsError:
        return False
    return True


def classify(
    *,
    origin_type: OriginType,
    training_rights_status: TrainingRightsStatus,
    commercial_reference: bool = False,
) -> SourceClass:
    """Map origin and rights onto a reporting class.

    Pure classification for inventory reporting. It confers nothing:
    :func:`validate_rights` remains the only thing that decides
    eligibility.
    """
    if origin_type is OriginType.SELF_MODEL_OUTPUT:
        return SourceClass.SELF_MODEL_GENERATED
    if commercial_reference and training_rights_status is not TrainingRightsStatus.CONFIRMED:
        return SourceClass.COMMERCIAL_REFERENCE_UNVERIFIED
    if origin_type is OriginType.AI_GENERATED:
        return (
            SourceClass.AI_GENERATED_RIGHTS_CLEARED
            if training_rights_status is TrainingRightsStatus.CONFIRMED
            else SourceClass.AI_GENERATED_RIGHTS_UNVERIFIED
        )
    if origin_type in (OriginType.HUMAN_RECORDED, OriginType.HYBRID):
        if training_rights_status is TrainingRightsStatus.CONFIRMED:
            return SourceClass.HUMAN_PRODUCED_RIGHTS_CLEARED
        return SourceClass.UNKNOWN
    return SourceClass.UNKNOWN
