"""Rights validation for training data.

The gate, not a warning. A track whose rights are not affirmatively
confirmed is **excluded from the dataset in code** — there is no
override flag, no "review later" state, and no way for an unconfirmed
track to reach preprocessing.

The rule this enforces: public accessibility is not a training licence.
Neither is "we found it online", "it has no visible copyright notice",
or "it is only a pilot". Upstream's own LoRA tutorial says the same
thing — it demonstrates on a commercial album while instructing readers
to "use your own original works".

Prohibited outright, regardless of any metadata a caller supplies:
scraped output from other generative music services, scraped commercial
catalogues, and any copyrighted material without an explicit training
grant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class RightsStatus(StrEnum):
    """How the right to train on a track was established."""

    #: Written by or for LUBER; we hold everything.
    ORIGINAL_WORK = "ORIGINAL_WORK"
    #: Explicit written licence covering ML training.
    LICENSED_FOR_TRAINING = "LICENSED_FOR_TRAINING"
    #: Public domain, verified — not merely assumed.
    PUBLIC_DOMAIN = "PUBLIC_DOMAIN"
    #: Creative Commons licence whose terms permit training and the
    #: intended commercial use.
    CC_TRAINING_PERMITTED = "CC_TRAINING_PERMITTED"
    #: Rights holder gave documented permission.
    RIGHTS_HOLDER_PERMISSION = "RIGHTS_HOLDER_PERMISSION"
    #: Not established. Always excluded.
    UNVERIFIED = "UNVERIFIED"
    #: Actively known to be unusable.
    PROHIBITED = "PROHIBITED"
    #: Output of a generative model — including our own. Always
    #: excluded: training on generated audio teaches the model its own
    #: artifacts back, and this repository contains 100+ such files from
    #: benchmarking that must never be mistaken for training material.
    AI_GENERATED = "AI_GENERATED"


#: The only statuses that may enter a training set.
ACCEPTABLE_STATUSES: frozenset[RightsStatus] = frozenset(
    {
        RightsStatus.ORIGINAL_WORK,
        RightsStatus.LICENSED_FOR_TRAINING,
        RightsStatus.PUBLIC_DOMAIN,
        RightsStatus.CC_TRAINING_PERMITTED,
        RightsStatus.RIGHTS_HOLDER_PERMISSION,
    }
)

#: Provenance descriptions that are refused no matter what status is
#: claimed alongside them.
PROHIBITED_SOURCE_MARKERS: tuple[str, ...] = (
    "suno",
    "udio",
    # Our own engine output. Phase 5 rated it 2/10; training on it would
    # entrench exactly the failures the training set exists to fix.
    "acestep",
    "ace-step",
    "generated",
    "ai-generated",
    "synthetic",
    "fixture",
    "mock",
    "scrape",
    "scraped",
    "crawler",
    "crawled",
    "torrent",
    "youtube-dl",
    "yt-dlp",
    "spotify-rip",
    "ripped",
    "leaked",
)


class RightsError(Exception):
    """Raised when a track may not be used for training."""


@dataclass(frozen=True)
class RightsRecord:
    """Documented basis for training on one track.

    Every field is required. A rights claim without a holder, a
    document reference, and a date is not a rights claim.
    """

    status: RightsStatus
    #: Where the audio came from, in plain words.
    source: str
    #: Who holds the rights.
    rights_holder: str
    #: Contract, licence id, release URL, or file reference.
    document_reference: str
    #: ISO date the rights were confirmed.
    confirmed_on: str
    #: Confirmed separately — a track licence does not imply these.
    audio_use_confirmed: bool
    lyrics_rights_confirmed: bool
    performer_rights_confirmed: bool
    commercial_training_allowed: bool
    notes: str = ""


def validate_rights(record: RightsRecord, *, has_lyrics: bool, has_vocals: bool) -> None:
    """Raise unless this track may be used for commercial ML training.

    ``has_lyrics`` and ``has_vocals`` decide which sub-rights are
    actually required, so an instrumental is not blocked for lacking
    lyric clearance it does not need.
    """
    if record.status not in ACCEPTABLE_STATUSES:
        raise RightsError(
            f"rights status {record.status} is not acceptable for training "
            "(unverified and prohibited material is always excluded)"
        )

    haystack = f"{record.source} {record.document_reference} {record.notes}".lower()
    for marker in PROHIBITED_SOURCE_MARKERS:
        # Word-boundary matched so "LUBER studio" does not trip the
        # "udio" marker. Hyphens are *not* boundaries-in-reverse:
        # "suno-export" must still be caught.
        if re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", haystack):
            raise RightsError(
                f"source describes prohibited provenance ({marker!r}); "
                "scraped or generated-service audio is never trainable here"
            )

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
