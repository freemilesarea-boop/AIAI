"""Import of externally-produced reference audio for comparison.

Reference tracks make blinded comparison against another system
possible. They may only ever be **supplied by the user** from audio they
already hold and have the right to compare.

This module deliberately provides no way to fetch anything. There is no
downloader, no scraper, no API client, and no account automation for any
external music service. Import is local-file only, and every import must
declare its provenance.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

REQUIRED_METADATA = ("source", "version", "prompt", "lyrics", "date", "sha256")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ReferenceImportError(Exception):
    """Raised when a reference track cannot be accepted."""


@dataclass(frozen=True)
class ReferenceTrack:
    reference_id: str
    reference_system: str
    version: str
    prompt: str
    lyrics: str
    date: str
    sha256: str
    file_size: int
    storage_path: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_reference(
    audio_path: Path,
    metadata: dict[str, str],
    *,
    reference_id: str,
) -> ReferenceTrack:
    """Register a user-supplied reference track.

    The declared SHA-256 must match the file, so a reference cannot be
    silently swapped after it was evaluated.
    """
    missing = [field for field in REQUIRED_METADATA if not metadata.get(field)]
    if missing:
        raise ReferenceImportError(
            f"reference import requires provenance metadata: {', '.join(missing)}"
        )

    if not audio_path.is_file():
        raise ReferenceImportError(f"reference audio not found: {audio_path}")

    if not _ISO_DATE.match(metadata["date"]):
        raise ReferenceImportError("reference date must be ISO format YYYY-MM-DD")
    try:
        date.fromisoformat(metadata["date"])
    except ValueError as exc:
        raise ReferenceImportError(f"invalid reference date: {exc}") from exc

    actual = sha256_file(audio_path)
    if actual != metadata["sha256"]:
        raise ReferenceImportError(
            "declared sha256 does not match the supplied file; refusing the import"
        )

    return ReferenceTrack(
        reference_id=reference_id,
        reference_system=metadata["source"],
        version=metadata["version"],
        prompt=metadata["prompt"],
        lyrics=metadata["lyrics"],
        date=metadata["date"],
        sha256=actual,
        file_size=audio_path.stat().st_size,
        storage_path=str(audio_path),
    )
