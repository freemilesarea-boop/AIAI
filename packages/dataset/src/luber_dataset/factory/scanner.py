"""Recursive read-only discovery of candidate audio.

Read-only is not a comment here, it is the contract. Nothing in this
module opens a file for writing, and the pipeline verifies afterwards
that every source digest is unchanged.

Identity comes from content, not from a path. Two copies of the same
bytes under different names are one track with two source references,
and a file that gets renamed between runs keeps its identity and its
cached analysis. That is why the internal id is derived from the
SHA-256 rather than from the filename.
"""

from __future__ import annotations

import os
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from luber_dataset.discovery import SKIP_DIR_NAMES, SKIP_DIR_SUFFIXES, sha256_file

#: Formats the factory will attempt. The first four decode without
#: ffmpeg for WAV; everything else needs it, which the pipeline already
#: requires for delivery.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".aiff", ".aif", ".wma"}
)

#: Sidecars and reports that live beside audio and must never be scanned
#: as audio themselves.
NON_AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".json", ".txt", ".lrc", ".md", ".csv", ".pdf", ".jpg", ".jpeg", ".png", ".webp"}
)

#: Partial downloads and editor droppings. A `.part` file is a file that
#: is still being written, and hashing one produces an identity that
#: changes underneath the run.
IGNORED_SUFFIXES: tuple[str, ...] = (
    ".part",
    ".crdownload",
    ".download",
    ".tmp",
    ".temp",
    ".partial",
    ".!ut",
    "~",
)

#: macOS resource forks and Windows thumbnail caches share the property
#: of looking like real files to `os.walk`.
IGNORED_NAMES: frozenset[str] = frozenset({".DS_Store", "Thumbs.db", "desktop.ini", "Icon\r"})


@dataclass(frozen=True)
class ScannedFile:
    """One candidate, identified by its content."""

    source_path: str
    source_filename: str
    source_extension: str
    source_size_bytes: int
    source_mtime: float
    sha256: str

    @property
    def file_id(self) -> str:
        """Stable id from content, not from the name.

        Short enough to read in a report, long enough that a collision
        across a personal library is not a practical concern.
        """
        return f"trk_{self.sha256[:16]}"


@dataclass
class ScanResult:
    files: list[ScannedFile] = field(default_factory=list)
    #: Paths skipped, with the reason, so "the scan missed it" and "the
    #: scan deliberately ignored it" stay distinguishable.
    skipped: list[tuple[str, str]] = field(default_factory=list)

    @property
    def total_files(self) -> int:
        return len(self.files)


def _should_ignore(name: str) -> str | None:
    """Reason to skip this filename, or None to keep looking at it."""
    if name in IGNORED_NAMES:
        return "OS_METADATA"
    # AppleDouble sidecars begin with `._` and hold no audio, but do
    # carry the extension of the file they shadow.
    if name.startswith("._"):
        return "OS_METADATA"
    if name.startswith("."):
        return "HIDDEN"
    lowered = name.lower()
    if lowered.endswith(IGNORED_SUFFIXES):
        return "PARTIAL_OR_TEMPORARY"
    return None


def _classify_extension(suffix: str) -> str | None:
    if suffix in SUPPORTED_EXTENSIONS:
        return None
    if suffix in NON_AUDIO_EXTENSIONS:
        return "NOT_AUDIO"
    return "UNSUPPORTED_FORMAT"


def iter_candidates(root: Path) -> Iterator[tuple[Path, str | None]]:
    """Walk *root*, yielding every file with a skip reason or None.

    Symlinks are not followed. A library with a loop in it would
    otherwise scan forever, and a symlink into a system directory would
    pull in audio the operator never meant to offer.
    """
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d not in SKIP_DIR_NAMES
            and not d.startswith(".")
            and not d.lower().endswith(SKIP_DIR_SUFFIXES)
        )
        for name in sorted(filenames):
            path = current / name
            reason = _should_ignore(name)
            if reason is None:
                reason = _classify_extension(path.suffix.lower())
            yield path, reason


def scan(root: Path, *, max_files: int | None = None) -> ScanResult:
    """Catalog every supported audio file under *root*.

    Ordered by path so a run over an unchanged tree visits files in the
    same sequence every time — one of the several things that have to
    hold for two runs to produce an identical manifest.
    """
    result = ScanResult()
    if not root.exists():
        raise FileNotFoundError(f"scan root does not exist: {root}")

    for path, skip_reason in iter_candidates(root):
        if skip_reason is not None:
            if skip_reason != "NOT_AUDIO":
                result.skipped.append((str(path), skip_reason))
            continue
        try:
            stat = path.stat()
        except OSError as exc:
            result.skipped.append((str(path), f"UNREADABLE: {exc.strerror}"))
            continue
        if stat.st_size == 0:
            result.skipped.append((str(path), "EMPTY_FILE"))
            continue
        try:
            digest = sha256_file(path)
        except OSError as exc:
            result.skipped.append((str(path), f"UNREADABLE: {exc.strerror}"))
            continue

        result.files.append(
            ScannedFile(
                # NFC throughout: macOS hands back decomposed filenames,
                # so a Korean title compares unequal to the same title
                # read from a sidecar unless both are normalised.
                source_path=unicodedata.normalize("NFC", str(path)),
                source_filename=unicodedata.normalize("NFC", path.name),
                source_extension=path.suffix.lower(),
                source_size_bytes=stat.st_size,
                source_mtime=stat.st_mtime,
                sha256=digest,
            )
        )
        if max_files is not None and len(result.files) >= max_files:
            break

    result.files.sort(key=lambda f: (f.sha256, f.source_path))
    return result


def verify_unchanged(files: list[ScannedFile]) -> list[str]:
    """Re-hash every source and report any that moved.

    Run after the pipeline finishes. The immutability policy is worth
    nothing if it is only asserted in prose, and a bug that writes to a
    source would otherwise be discovered by its consequences.
    """
    changed: list[str] = []
    for item in files:
        path = Path(item.source_path)
        try:
            if not path.is_file() or sha256_file(path) != item.sha256:
                changed.append(item.source_path)
        except OSError:
            changed.append(item.source_path)
    return changed
