"""Read-only discovery of candidate audio on the operator's machine.

Strictly non-destructive: this module opens files for reading and does
nothing else. It never writes, renames, moves, converts, tags, or
deletes anything under the scanned root, and it never copies audio into
the repository.

Path context is used to form a **hypothesis** about a file's origin —
never to grant rights. A folder called `발매음원` suggests released
material; it does not establish that anyone may train on it. Only an
operator decision recorded against a documented rights record can do
that (`rights.validate_rights`).

Absolute paths are personal data. Catalogs keep them for the operator's
own use, and :func:`sanitize` strips them before anything is reported
or committed.
"""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
import wave
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

AUDIO_EXTENSIONS: frozenset[str] = frozenset({".wav", ".flac", ".aiff", ".aif", ".mp3", ".m4a"})
LYRIC_EXTENSIONS: frozenset[str] = frozenset({".txt", ".lrc", ".json", ".md", ".csv"})

#: Directories never worth walking.
SKIP_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".next",
        "Library",
        "System",
        ".Trash",
        "site-packages",
        "checkpoints",
    }
)

#: Application bundles contain UI sounds and samples that are not music
#: and are not the operator's to train on.
SKIP_DIR_SUFFIXES: tuple[str, ...] = (".app", ".bundle", ".framework")

# ── path hypotheses (Korean and English) ──────────────────────────────
# These set a *hypothesis* only. Nothing here can confirm rights.

AI_HINTS: tuple[str, ...] = (
    "ai 음원",
    "ai음원",
    "ai-음원",
    "ai music",
    "ai_music",
    "aimusic",
    "suno",
    "udio",
    "ai 생성",
    "ai generated",
    "ai-generated",
)
SELF_MODEL_HINTS: tuple[str, ...] = (
    "luber-music-ai",
    "ace-step",
    "acestep",
    "raw-model-output",
    "ab-experiment",
    "bench",
    "phase-3-real",
    "phase-4-real",
)
ORIGINAL_HINTS: tuple[str, ...] = (
    "제작음원",
    "제작 음원",
    "발매음원",
    "발매 음원",
    "발매",
    "original",
    "master",
    "마스터",
    "믹스",
    "mixdown",
    "stem",
    "project",
)
COMMERCIAL_HINTS: tuple[str, ...] = (
    "기성음원",
    "기성 음원",
    "billboard",
    "빌보드",
    "레퍼런스",
    "reference",
    "차트",
    "chart",
    "top100",
    "가요",
    "commercial",
)


@dataclass
class DiscoveredFile:
    """One audio file found during the scan."""

    absolute_path: str
    filename: str
    extension: str
    byte_size: int
    parent_directory: str
    sha256: str
    #: Populated for WAV only; other formats need a decoder.
    duration_seconds: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    origin_hypothesis: str = "UNKNOWN"
    commercial_reference_hypothesis: bool = False
    duplicate_group: str = ""
    adjacent_lyrics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _wav_properties(path: Path) -> tuple[float, int, int] | None:
    """Cheap header read for WAV; None for anything else or on error."""
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as w:
            rate = w.getframerate()
            frames = w.getnframes()
            channels = w.getnchannels()
    except Exception:
        return None
    if rate <= 0:
        return None
    return frames / rate, rate, channels


def _contains_hint(haystack: str, hint: str) -> bool:
    """Token-aware containment.

    Latin hints must match on word boundaries — "audio" contains "udio",
    which would otherwise flag every UI sound as AI music. Korean has no
    word boundaries in the regex sense, so those match as substrings.
    """
    if hint.isascii():
        return re.search(rf"(?<![a-z0-9]){re.escape(hint)}(?![a-z0-9])", haystack) is not None
    return hint in haystack


def hypothesize_origin(path: Path) -> tuple[str, bool]:
    """Guess origin from path context. Returns (hypothesis, is_commercial).

    A hypothesis is a prompt for the operator to decide, nothing more.
    """
    # macOS stores filenames NFD-decomposed; Korean literals here are
    # NFC. Without normalizing, 기성 음원 never matches 기성 음원.
    haystack = unicodedata.normalize("NFC", str(path)).lower()

    # Self-model detection wins: our own output must never be proposed
    # as training material, whatever folder it happens to sit in.
    if any(_contains_hint(haystack, h) for h in SELF_MODEL_HINTS):
        return "SELF_MODEL_OUTPUT", False
    if any(_contains_hint(haystack, h) for h in COMMERCIAL_HINTS):
        return "UNKNOWN", True
    if any(_contains_hint(haystack, h) for h in AI_HINTS):
        return "AI_GENERATED", False
    if any(_contains_hint(haystack, h) for h in ORIGINAL_HINTS):
        # The operator's own project or release folder. Whether the audio
        # inside was performed or generated is still undetermined — the
        # folder name cannot distinguish them, and neither confers rights.
        return "ORIGINAL_PROJECT", False
    return "UNKNOWN", False


def find_adjacent_lyrics(audio: Path) -> list[str]:
    """Sibling files that might hold lyrics or metadata for this track.

    Matched by basename only. Nothing is parsed or assumed to be lyrics
    here — the ingester decides that, and missing lyrics stay missing
    rather than being invented.
    """
    stem = audio.stem.lower()
    matches: list[str] = []
    try:
        siblings = list(audio.parent.iterdir())
    except OSError:
        return matches
    for sibling in siblings:
        if sibling == audio or not sibling.is_file():
            continue
        if sibling.suffix.lower() not in LYRIC_EXTENSIONS:
            continue
        other = sibling.stem.lower()
        if other == stem or other.startswith(stem) or stem.startswith(other):
            matches.append(sibling.name)
    return sorted(matches)


def scan(
    root: Path,
    *,
    exclude_roots: tuple[Path, ...] = (),
    hash_files: bool = True,
    max_files: int | None = None,
) -> list[DiscoveredFile]:
    """Walk *root* read-only and catalog audio files."""
    excluded = tuple(p.resolve() for p in exclude_roots)
    found: list[DiscoveredFile] = []

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        dirnames[:] = [
            d
            for d in dirnames
            if d not in SKIP_DIR_NAMES
            and not d.startswith(".")
            and not d.lower().endswith(SKIP_DIR_SUFFIXES)
        ]
        resolved = current.resolve()
        if any(resolved == ex or ex in resolved.parents for ex in excluded):
            dirnames[:] = []
            continue

        for name in filenames:
            path = current / name
            if path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size == 0:
                continue

            origin, commercial = hypothesize_origin(path)
            properties = _wav_properties(path)
            found.append(
                DiscoveredFile(
                    absolute_path=str(path),
                    filename=name,
                    extension=path.suffix.lower(),
                    byte_size=size,
                    parent_directory=current.name,
                    sha256=sha256_file(path) if hash_files else "",
                    duration_seconds=round(properties[0], 2) if properties else None,
                    sample_rate=properties[1] if properties else None,
                    channels=properties[2] if properties else None,
                    origin_hypothesis=origin,
                    commercial_reference_hypothesis=commercial,
                    adjacent_lyrics=find_adjacent_lyrics(path),
                )
            )
            if max_files is not None and len(found) >= max_files:
                return _mark_duplicates(found)

    return _mark_duplicates(found)


def _mark_duplicates(files: list[DiscoveredFile]) -> list[DiscoveredFile]:
    by_hash: dict[str, list[DiscoveredFile]] = defaultdict(list)
    for item in files:
        if item.sha256:
            by_hash[item.sha256].append(item)
    for digest, group in by_hash.items():
        if len(group) > 1:
            for item in group:
                item.duplicate_group = digest[:12]
    return files


_PERSONAL_SEGMENT = re.compile(r"^/Users/[^/]+")


def sanitize(path: str, *, root: Path | None = None) -> str:
    """Strip personal path prefixes so a catalog can be reported.

    Output is NFC-normalized: macOS returns NFD-decomposed filenames,
    and Korean string comparisons downstream fail silently against NFC
    literals otherwise.
    """
    if root is not None:
        try:
            return unicodedata.normalize("NFC", str(Path(path).relative_to(root)))
        except ValueError:
            pass
    return unicodedata.normalize("NFC", _PERSONAL_SEGMENT.sub("~", path))


def summarize(files: list[DiscoveredFile]) -> dict[str, Any]:
    """Aggregate inventory counts. Contains no absolute paths."""
    unique_hashes = {f.sha256 for f in files if f.sha256}
    duplicates = sum(1 for f in files if f.duplicate_group)
    by_extension: dict[str, int] = defaultdict(int)
    by_origin: dict[str, int] = defaultdict(int)
    total_seconds = 0.0
    measured = 0

    for item in files:
        by_extension[item.extension] += 1
        key = (
            "COMMERCIAL_REFERENCE"
            if item.commercial_reference_hypothesis
            else item.origin_hypothesis
        )
        by_origin[key] += 1
        if item.duration_seconds:
            total_seconds += item.duration_seconds
            measured += 1

    return {
        "total_files": len(files),
        "unique_hashes": len(unique_hashes),
        "duplicate_files": duplicates,
        "by_extension": dict(sorted(by_extension.items())),
        "by_origin_hypothesis": dict(sorted(by_origin.items())),
        "measured_duration_files": measured,
        "measured_hours": round(total_seconds / 3600, 2),
        "total_bytes": sum(f.byte_size for f in files),
        "with_adjacent_lyrics": sum(1 for f in files if f.adjacent_lyrics),
    }
