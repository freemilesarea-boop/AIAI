"""Operator-supplied metadata, kept separate from anything inferred.

Every value the factory records carries where it came from. That is not
bookkeeping — it is the difference between "the operator states this
track is theirs" and "a folder was called `originals`", and those two
must never be allowed to look alike, because only one of them can
support a rights decision.

Sidecars are the only channel through which a human asserts a fact. A
`track.json` beside `track.wav` is read, validated against a closed
schema, and recorded with source ``USER``. Unknown keys are rejected
rather than ignored: a typo in ``commercial_training_allowed`` that
silently does nothing is the worst possible failure mode for the one
field that governs whether audio may be trained on.

Embedded tags are read where the container has them, and recorded with
source ``EMBEDDED``. They are useful for grouping — an album field
prevents a leak across a train/test split — and they establish nothing
about rights, because anyone can write anything into an ID3 frame.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from luber_dataset.factory.decoder import probe as _probe_unused  # noqa: F401  (documents kinship)


class MetadataSource(StrEnum):
    USER = "USER"
    SIDECAR = "SIDECAR"
    EMBEDDED = "EMBEDDED"
    ASR = "ASR"
    INFERRED = "INFERRED"
    NONE = "NONE"


@dataclass(frozen=True)
class Attributed:
    """One value, and where it came from.

    Confidence is ``1.0`` for a human assertion and less for anything
    measured. A field with no value still records its source as ``NONE``
    so "nobody said" stays distinct from "somebody said nothing".
    """

    value: Any
    source: str = MetadataSource.NONE.value
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "source": self.source, "confidence": self.confidence}


ABSENT = Attributed(value=None, source=MetadataSource.NONE.value, confidence=None)


class SidecarError(ValueError):
    """Raised when a sidecar exists but cannot be trusted."""


#: The closed set a sidecar may declare. Anything else is a mistake the
#: operator needs told about.
SIDECAR_FIELDS: frozenset[str] = frozenset(
    {
        "title",
        "artist",
        "album",
        "genre",
        "subgenre",
        "mood",
        "language",
        "lyrics",
        "vocal_type",
        "source",
        "source_type",
        "rights_status",
        "license",
        "commercial_training_allowed",
        "notes",
    }
)

_STRING_FIELDS = SIDECAR_FIELDS - {"commercial_training_allowed"}

#: Tri-state, and the string forms accepted for each. Deliberately not
#: Python truthiness: ``bool("false")`` is ``True``, and a sidecar saying
#: "false" must never grant a training right.
_TRISTATE = {
    "true": "TRUE",
    "yes": "TRUE",
    "false": "FALSE",
    "no": "FALSE",
    "unknown": "UNKNOWN",
    "": "UNKNOWN",
}


@dataclass
class Sidecar:
    """Validated operator metadata for one track."""

    path: str
    fields: dict[str, Any] = field(default_factory=dict)

    def get(self, name: str) -> Any:
        return self.fields.get(name)


def sidecar_path(audio: Path) -> Path:
    """`track.wav` -> `track.json`."""
    return audio.with_suffix(".json")


def _normalise_tristate(value: Any, *, field_name: str) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if value is None:
        return "UNKNOWN"
    key = str(value).strip().lower()
    if key not in _TRISTATE:
        raise SidecarError(
            f"{field_name} must be true, false or unknown; got {value!r}. "
            "A value that cannot be read is treated as UNKNOWN, never as permission."
        )
    return _TRISTATE[key]


def load_sidecar(audio: Path) -> Sidecar | None:
    """Read and validate the sidecar beside *audio*, if there is one.

    Raises rather than degrading. A malformed sidecar means the operator
    tried to say something and it did not arrive; continuing as though
    they had said nothing would discard exactly the assertion they took
    the trouble to make.
    """
    path = sidecar_path(audio)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise SidecarError(f"{path.name} could not be read: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SidecarError(f"{path.name} is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise SidecarError(f"{path.name} must contain a JSON object")

    unknown = sorted(set(raw) - SIDECAR_FIELDS)
    if unknown:
        raise SidecarError(
            f"{path.name} has unrecognised field(s): {', '.join(unknown)}. "
            f"Permitted: {', '.join(sorted(SIDECAR_FIELDS))}"
        )

    cleaned: dict[str, Any] = {}
    for name, value in raw.items():
        if value is None:
            continue
        if name == "commercial_training_allowed":
            cleaned[name] = _normalise_tristate(value, field_name=name)
            continue
        if name in _STRING_FIELDS:
            if not isinstance(value, str):
                raise SidecarError(
                    f"{path.name}: {name} must be a string, got {type(value).__name__}"
                )
            text = unicodedata.normalize("NFC", value).strip()
            if text:
                cleaned[name] = text
    return Sidecar(path=str(path), fields=cleaned)


#: ffprobe tag names, lowercased, mapped onto ours.
_TAG_ALIASES = {
    "title": "title",
    "artist": "artist",
    "album_artist": "artist",
    "albumartist": "artist",
    "album": "album",
    "genre": "genre",
    "language": "language",
    "lyrics": "lyrics",
    "unsyncedlyrics": "lyrics",
}


def read_embedded_tags(path: Path) -> dict[str, str]:
    """Tags the container carries. Grouping evidence, never rights.

    Failure is silence: a file with unreadable tags is a file without
    tags, and that is not a reason to reject its audio.
    """
    from luber_dataset.factory.decoder import _binary, _run  # local: keeps ffprobe use in one place

    try:
        ffprobe = _binary("ffprobe")
    except Exception:
        return {}
    try:
        completed = _run(
            [
                ffprobe,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                str(path),
            ],
            30.0,
        )
    except Exception:
        return {}
    if completed.returncode != 0:
        return {}
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return {}

    fmt = payload.get("format")
    tags = fmt.get("tags") if isinstance(fmt, dict) else None
    if not isinstance(tags, dict):
        return {}

    result: dict[str, str] = {}
    for key, value in tags.items():
        name = _TAG_ALIASES.get(str(key).strip().lower())
        if name is None or not isinstance(value, str):
            continue
        text = unicodedata.normalize("NFC", value).strip()
        # First writer wins, so `artist` is not overwritten by
        # `album_artist` on a compilation.
        if text and name not in result:
            result[name] = text
    return result


def find_lyrics_sidecar(audio: Path) -> tuple[str, str] | None:
    """Lyrics from a `.txt` or `.lrc` beside the audio.

    Returns ``(text, path)``. Nothing is parsed into structure and
    nothing is generated: if there is no file, there are no lyrics, and
    the record says so.
    """
    for suffix in (".txt", ".lrc"):
        candidate = audio.with_suffix(suffix)
        if not candidate.is_file():
            continue
        try:
            text = candidate.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            continue
        if text:
            return unicodedata.normalize("NFC", text), str(candidate)
    return None
