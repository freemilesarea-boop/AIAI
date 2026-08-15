"""Turning an uploaded file into a reference the engine can be given.

Everything here treats the upload as hostile until proved otherwise. It
arrives over HTTP from a browser, so the container it claims to be, the
name it carries and the size it declares are all assertions rather than
facts, and each is checked against the bytes.

Three rules shape the implementation:

*The bytes decide the format, not the filename.* The declared content
type is used to reject early, but acceptance comes from ffprobe finding
a real audio stream with a real duration. A ``.wav`` full of HTML fails.

*The client filename never becomes a path.* It is kept as a display
label and nothing else; the storage key is built from a server-generated
UUID. There is no code path in which user text reaches the filesystem.

*Normalisation is format-only.* The canonical form is 48 kHz stereo WAV
because that is what the engine reads. No loudness work, no EQ, and
explicitly none of the Phase 14 finishing engine — a reference is
something the model listens to, not a deliverable to improve.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from luber_audio_utils.transcode import (
    AudioProcessingError,
    _require_binary,
    _run,
    probe_audio,
)
from luber_audio_utils.wav import sha256_file
from luber_schemas import (
    CANONICAL_REFERENCE_CHANNELS,
    CANONICAL_REFERENCE_SAMPLE_RATE,
    MAX_REFERENCE_DURATION_SECONDS,
    MAX_REFERENCE_FILE_BYTES,
    MIN_REFERENCE_DURATION_SECONDS,
    extension_for_content_type,
    is_supported_reference_extension,
)


class ReferenceAudioRejected(Exception):
    """The upload is not usable as reference audio.

    Carries a message safe to show a user: it describes what was wrong
    with their file and never quotes a path or an internal error.
    """


@dataclass(frozen=True)
class NormalizedReference:
    """A validated upload, converted to the canonical stored form."""

    path: Path
    duration_seconds: float
    sample_rate: int
    channels: int
    file_size: int
    sha256: str
    source_sha256: str
    source_format: str


def resolve_upload_format(filename: str | None, content_type: str | None) -> str:
    """Container to accept the upload as, from its declared metadata.

    Both signals are advisory, so either may establish the format and
    disagreement is tolerated — the decode check is what actually
    decides. What this refuses is an upload claiming nothing recognisable
    at all, which is cheaper to reject before writing bytes to disk.
    """
    from_type = extension_for_content_type(content_type)
    if from_type is not None:
        return from_type
    suffix = Path(filename or "").suffix.lower().lstrip(".")
    if suffix and is_supported_reference_extension(suffix):
        return suffix
    raise ReferenceAudioRejected(
        "Unsupported audio format. Upload a WAV, MP3, FLAC, M4A or OGG file."
    )


def check_upload_size(size_bytes: int) -> None:
    if size_bytes <= 0:
        raise ReferenceAudioRejected("That file is empty.")
    if size_bytes > MAX_REFERENCE_FILE_BYTES:
        limit_mb = MAX_REFERENCE_FILE_BYTES // (1024 * 1024)
        raise ReferenceAudioRejected(f"That file is larger than {limit_mb} MB.")


def inspect_upload(source: Path) -> tuple[float, int, int]:
    """Decode-check the bytes. Returns duration, sample rate, channels.

    This is the check that matters: a file can pass every declared-metadata
    test and still be a renamed image.
    """
    try:
        probe = probe_audio(source)
    except AudioProcessingError as exc:
        raise ReferenceAudioRejected(
            "That file could not be read as audio. It may be corrupt or not an audio file."
        ) from exc

    if probe.channels <= 0 or probe.sample_rate <= 0:
        raise ReferenceAudioRejected("That file has no usable audio stream.")
    if probe.duration_seconds < MIN_REFERENCE_DURATION_SECONDS:
        raise ReferenceAudioRejected(
            f"Reference audio must be at least {MIN_REFERENCE_DURATION_SECONDS:.0f} second long."
        )
    if probe.duration_seconds > MAX_REFERENCE_DURATION_SECONDS:
        limit = int(MAX_REFERENCE_DURATION_SECONDS // 60)
        raise ReferenceAudioRejected(f"Reference audio must be shorter than {limit} minutes.")
    return probe.duration_seconds, probe.sample_rate, probe.channels


def normalize_reference(
    source: Path, destination: Path, *, source_format: str
) -> NormalizedReference:
    """Validate an upload and write its canonical form.

    Raises :class:`ReferenceAudioRejected` for anything the user can fix
    and :class:`AudioProcessingError` for a genuine server-side failure —
    the two are different outcomes and the API answers them differently.
    """
    if not source.is_file():
        raise ReferenceAudioRejected("That file could not be read.")
    check_upload_size(source.stat().st_size)
    inspect_upload(source)
    source_digest = sha256_file(source)

    ffmpeg = _require_binary("ffmpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        _run(
            [
                ffmpeg,
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(source),
                "-map",
                "a:0",
                # Metadata is dropped: an uploaded file can carry tags,
                # cover art and comments that have no business being
                # stored, and none of it reaches the engine anyway.
                "-map_metadata",
                "-1",
                "-c:a",
                "pcm_s16le",
                "-ar",
                str(CANONICAL_REFERENCE_SAMPLE_RATE),
                "-ac",
                str(CANONICAL_REFERENCE_CHANNELS),
                str(destination),
            ]
        )
    except AudioProcessingError as exc:
        # Decode succeeded during inspection, so a failure here is ours.
        raise AudioProcessingError(f"reference normalisation failed: {exc}") from exc

    probe = probe_audio(destination)
    # Re-checked after conversion rather than trusted: a stream that
    # probes as ten minutes and converts to silence is still unusable.
    if probe.duration_seconds < MIN_REFERENCE_DURATION_SECONDS:
        raise ReferenceAudioRejected("That file has no usable audio in it.")

    return NormalizedReference(
        path=destination,
        duration_seconds=probe.duration_seconds,
        sample_rate=probe.sample_rate,
        channels=probe.channels,
        file_size=destination.stat().st_size,
        sha256=sha256_file(destination),
        source_sha256=source_digest,
        source_format=source_format,
    )


def safe_display_name(filename: str | None) -> str | None:
    """A label for the UI, stripped of anything path-like.

    Never used to build a storage key or a filesystem path — the key
    comes from a server-generated UUID. This exists so the user can
    recognise which file they picked, which means the only requirement is
    that it cannot be mistaken for a location.
    """
    if not filename:
        return None
    # Take the final component under either separator, so neither
    # "../../etc/passwd" nor a Windows path survives as one.
    tail = filename.replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = "".join(ch for ch in tail if ch.isprintable() and ch not in '\\/:*?"<>|').strip()
    cleaned = cleaned.lstrip(".").strip()
    return cleaned[:200] or None
