"""The reference-audio contract.

A reference track is an **input**, and that is the whole reason this is
its own type rather than another ``AssetType``. Every existing audio
asset is something a generation produced, keyed by the generation it
belongs to; a reference exists before any generation does, can be used
by several, and must never be reachable through the routes that serve
masters. Modelling it as an asset role would have put it one enum value
away from being downloadable as somebody's finished song.

It is also not any of the *source* audio the product already has.
Extend, Replace Section and Cover all read a previous LUBER master and
reach ACE-Step as ``src_audio``; a reference reaches it as ``ref_audio``
and drives a different mechanism entirely — the timbre encoder rather
than repaint or the semantic sketch. Phase 13E established that the two
are separate paths in the engine, and conflating them here would produce
a different operation than the user asked for.

Limits live here rather than in the API layer because the worker
validates against the same numbers the upload was accepted under.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

#: Container formats accepted on upload, mapped to the content types a
#: browser will actually send for them. Anything else is refused: the
#: list is what has been decoded successfully, not what might work.
SUPPORTED_REFERENCE_MIME_TYPES: dict[str, tuple[str, ...]] = {
    "wav": ("audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"),
    "mp3": ("audio/mpeg", "audio/mp3"),
    "flac": ("audio/flac", "audio/x-flac"),
    "m4a": ("audio/mp4", "audio/x-m4a", "audio/aac"),
    "ogg": ("audio/ogg", "application/ogg"),
}

SUPPORTED_REFERENCE_EXTENSIONS: tuple[str, ...] = tuple(SUPPORTED_REFERENCE_MIME_TYPES)

#: 40 MB. Comfortably holds several minutes of lossless stereo while
#: bounding what one request can make the server decode.
MAX_REFERENCE_FILE_BYTES = 40 * 1024 * 1024

#: The engine reads a fixed-length window from the reference, so a long
#: upload buys nothing and costs decode time on every generation.
MAX_REFERENCE_DURATION_SECONDS = 600.0
#: Below this there is not enough signal for the timbre encoder to
#: describe, and accepting it would promise an effect that cannot happen.
MIN_REFERENCE_DURATION_SECONDS = 1.0

#: Canonical stored form. Format normalisation only — no loudness work,
#: no EQ, and explicitly none of the Phase 14 finishing engine: a
#: reference is something to listen *to*, not something to improve.
CANONICAL_REFERENCE_SAMPLE_RATE = 48_000
CANONICAL_REFERENCE_CHANNELS = 2
CANONICAL_REFERENCE_FORMAT = "wav"
CANONICAL_REFERENCE_EXTENSION = "wav"


def reference_storage_key(reference_id: uuid.UUID) -> str:
    """Deterministic key for a stored reference.

    Deliberately outside the ``audio/<generation-id>/`` namespace that
    masters and previews live in. The download route resolves objects
    from ``audio_assets`` rows, so a reference has no row that could name
    it and no key shape that could collide with one — the separation is
    structural rather than a check somebody has to remember to write.
    """
    return f"reference/{reference_id}/source.{CANONICAL_REFERENCE_EXTENSION}"


@dataclass(frozen=True)
class ReferenceAudioCondition:
    """A reference attached to an otherwise ordinary generation request.

    Phase 13E sketched this as a path plus a duration. A path is wrong at
    this boundary: the worker may not share a filesystem with the API,
    and a path is the one field a client must never be able to influence.
    The stable identifier is carried instead, and audio is materialised
    from storage at the point of use.

    The reference *modulates* a text-to-music request rather than
    replacing it — the prompt still drives the song, which is what the
    engine actually does, and the measurements in Phase 13E showed the
    prompt is the stronger of the two.
    """

    reference_id: uuid.UUID
    storage_key: str
    duration_seconds: float
    sample_rate: int
    channels: int
    #: Digest of the canonical stored bytes, so provenance can prove
    #: which audio conditioned a generation even if it is later deleted.
    sha256: str

    def __post_init__(self) -> None:
        if self.duration_seconds <= 0:
            raise ValueError("reference audio has no duration")
        if not self.storage_key.startswith("reference/"):
            raise ValueError(f"not a reference storage key: {self.storage_key!r}")


def is_supported_reference_extension(extension: str) -> bool:
    return extension.lower().lstrip(".") in SUPPORTED_REFERENCE_MIME_TYPES


def extension_for_content_type(content_type: str | None) -> str | None:
    """Container implied by a request's content type, if it is one we take."""
    if not content_type:
        return None
    normalised = content_type.split(";")[0].strip().lower()
    for extension, accepted in SUPPORTED_REFERENCE_MIME_TYPES.items():
        if normalised in accepted:
            return extension
    return None
