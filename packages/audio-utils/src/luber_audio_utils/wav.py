"""Minimal WAV inspection and validation (stdlib only).

Phase 1 scope: structural validation of PCM WAV files plus SHA256.
The full production audio pipeline (resampling, peak safety, MP3
encoding via ffmpeg) arrives in Phase 4.
"""

from __future__ import annotations

import hashlib
import wave
from dataclasses import dataclass
from pathlib import Path


class WavValidationError(Exception):
    """Raised when a file is missing, unreadable, or not a valid PCM WAV."""


@dataclass(frozen=True)
class WavInfo:
    path: Path
    file_size: int
    sample_rate: int
    channels: int
    bit_depth: int
    frames: int
    duration_seconds: float
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_wav(path: Path) -> WavInfo:
    """Validate a WAV file and return its metadata.

    Checks: file exists, size > 0, valid WAV header, sample rate > 0,
    channels > 0, duration > 0. Raises :class:`WavValidationError` on
    any failure.
    """
    if not path.is_file():
        raise WavValidationError(f"audio file does not exist: {path}")
    file_size = path.stat().st_size
    if file_size <= 0:
        raise WavValidationError(f"audio file is empty: {path}")

    try:
        with wave.open(str(path), "rb") as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frames = wav.getnframes()
    except (wave.Error, EOFError) as exc:
        raise WavValidationError(f"not a valid WAV file: {path}: {exc}") from exc

    if sample_rate <= 0:
        raise WavValidationError(f"invalid sample rate {sample_rate}: {path}")
    if channels <= 0:
        raise WavValidationError(f"invalid channel count {channels}: {path}")
    if frames <= 0:
        raise WavValidationError(f"zero-length audio: {path}")

    return WavInfo(
        path=path,
        file_size=file_size,
        sample_rate=sample_rate,
        channels=channels,
        bit_depth=sample_width * 8,
        frames=frames,
        duration_seconds=frames / sample_rate,
        sha256=sha256_file(path),
    )
