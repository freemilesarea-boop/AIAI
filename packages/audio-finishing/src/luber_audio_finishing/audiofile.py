"""Reading audio into numpy without adding a decoding dependency.

PCM WAV is read with the standard library, so analysis and its tests run
with nothing installed but numpy. Anything else — float WAV, MP3, a
sample rate the model happened to emit — falls back to ffmpeg, which the
delivery pipeline already requires.

Samples are returned as float64 in [-1, 1] shaped ``(frames, channels)``.
Float WAV can exceed that range; it is returned as written rather than
normalised, because clipping detection has to see the real values.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class AudioLoadError(Exception):
    """Raised when a file cannot be read as audio."""


@dataclass(frozen=True)
class LoadedAudio:
    """Decoded samples plus the properties needed to interpret them."""

    samples: np.ndarray
    sample_rate: int
    #: Source bit depth; ``None`` when the source was compressed or float.
    bit_depth: int | None

    @property
    def frames(self) -> int:
        return int(self.samples.shape[0])

    @property
    def channels(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate else 0.0

    @property
    def is_stereo(self) -> bool:
        return self.channels == 2

    def mono(self) -> np.ndarray:
        """Channel mean. The reference signal for spectral analysis."""
        return np.asarray(self.samples.mean(axis=1), dtype=np.float64)


def _decode_pcm(raw: bytes, width: int, channels: int) -> np.ndarray:
    if width == 1:
        # 8-bit WAV is unsigned with an offset of 128.
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 3:
        # 24-bit has no numpy dtype: widen each little-endian triple.
        as_bytes = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        as_int = (
            as_bytes[:, 0].astype(np.int32)
            | (as_bytes[:, 1].astype(np.int32) << 8)
            | (as_bytes[:, 2].astype(np.int32) << 16)
        )
        as_int = np.where(as_int & 0x800000, as_int - 0x1000000, as_int)
        data = as_int.astype(np.float64) / 8388608.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise AudioLoadError(f"unsupported PCM sample width: {width} bytes")
    if channels <= 0:
        raise AudioLoadError(f"invalid channel count: {channels}")
    if data.size % channels:
        raise AudioLoadError("audio data does not divide evenly into channels")
    return data.reshape(-1, channels)


def _load_via_ffmpeg(path: Path) -> LoadedAudio:
    """Decode to float32 on stdout, preserving rate and channel count."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise AudioLoadError(
            f"{path.name} is not PCM WAV and ffmpeg/ffprobe are unavailable to decode it"
        )

    probe = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    fields = probe.stdout.strip().split(",")
    if probe.returncode != 0 or len(fields) < 2:
        raise AudioLoadError(f"ffprobe could not read an audio stream from {path}")
    try:
        sample_rate, channels = int(fields[0]), int(fields[1])
    except ValueError as exc:
        raise AudioLoadError(f"ffprobe returned unreadable stream properties for {path}") from exc

    decoded = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "a:0",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    if decoded.returncode != 0 or not decoded.stdout:
        raise AudioLoadError(f"ffmpeg could not decode {path}")

    flat = np.frombuffer(decoded.stdout, dtype="<f4").astype(np.float64)
    if channels <= 0 or flat.size % channels:
        raise AudioLoadError(f"decoded sample count is not a multiple of {channels} channels")
    return LoadedAudio(samples=flat.reshape(-1, channels), sample_rate=sample_rate, bit_depth=None)


def load_audio(path: Path) -> LoadedAudio:
    """Read an audio file into memory.

    Raises :class:`AudioLoadError` rather than returning empty audio, so
    a caller cannot mistake an unreadable file for a silent one.
    """
    if not path.is_file():
        raise AudioLoadError(f"audio file does not exist: {path}")
    if path.stat().st_size == 0:
        raise AudioLoadError(f"audio file is empty: {path}")

    try:
        with wave.open(str(path), "rb") as handle:
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            frame_count = handle.getnframes()
            raw = handle.readframes(frame_count)
    except (wave.Error, EOFError, struct.error):
        # Not a PCM WAV the stdlib understands (float WAV, MP3, exotic
        # chunk layout). ffmpeg is the fallback, not the default, because
        # a subprocess per file is the expensive path.
        return _load_via_ffmpeg(path)

    if sample_rate <= 0:
        raise AudioLoadError(f"invalid sample rate {sample_rate}: {path}")
    if frame_count <= 0 or not raw:
        raise AudioLoadError(f"zero-length audio: {path}")

    return LoadedAudio(
        samples=_decode_pcm(raw, width, channels),
        sample_rate=sample_rate,
        bit_depth=width * 8,
    )
