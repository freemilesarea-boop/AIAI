"""Delivery-format normalization via ffmpeg.

This module converts whatever the model produced into the two shipping
formats, and does **nothing else**. It is explicitly not a mastering
stage: no loudness normalization, no limiter, no compressor, no EQ, no
stereo widening, no dithering choices beyond ffmpeg's default sample
format conversion. The audio content is preserved; only the container,
sample rate, channel count, and sample width are normalized.

Every command is deterministic for a given input and ffmpeg build:
metadata is stripped (``-map_metadata -1``), no timestamps are written,
and no filters are applied.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from luber_audio_utils.constants import (
    MASTER_BIT_DEPTH,
    MASTER_CHANNELS,
    MASTER_SAMPLE_RATE,
    PREVIEW_BITRATE_BPS,
    PREVIEW_CHANNELS,
    PREVIEW_SAMPLE_RATE,
)


class AudioProcessingError(Exception):
    """Raised when probing, transcoding, or encoding audio fails."""


@dataclass(frozen=True)
class AudioProbe:
    """Objective properties of an audio file, read back with ffprobe."""

    codec_name: str
    sample_rate: int
    channels: int
    duration_seconds: float
    file_size: int
    #: PCM bit depth; ``None`` for compressed formats such as MP3.
    bit_depth: int | None
    #: Stream bitrate in bits/s; ``None`` when ffprobe does not report one.
    bitrate_bps: int | None


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise AudioProcessingError(
            f"{name} is required for audio post-processing but was not found on PATH"
        )
    return path


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        # Fixed argv, never a shell string: no interpolation of user input.
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        # stderr can be long; keep the tail, which holds the actual error.
        detail = (exc.stderr or "").strip().splitlines()
        raise AudioProcessingError(
            f"{Path(command[0]).name} failed (exit {exc.returncode}): "
            f"{detail[-1] if detail else 'no output'}"
        ) from exc


def probe_audio(path: Path) -> AudioProbe:
    """Read objective audio properties back from a file."""
    if not path.is_file():
        raise AudioProcessingError(f"audio file does not exist: {path}")

    ffprobe = _require_binary("ffprobe")
    result = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,bits_per_raw_sample,bits_per_sample,bit_rate",
            "-show_entries",
            "format=duration,size,bit_rate",
            "-of",
            "json",
            str(path),
        ]
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AudioProcessingError(f"ffprobe returned unreadable output for {path}") from exc

    streams = payload.get("streams") or []
    if not streams:
        raise AudioProcessingError(f"no audio stream found in {path}")
    stream = streams[0]
    fmt = payload.get("format") or {}

    def _int_or_none(value: object) -> int | None:
        try:
            parsed = int(str(value))
        except (TypeError, ValueError):
            return None
        return parsed or None

    # ffmpeg reports 24-bit PCM as bits_per_raw_sample=24 inside an s32
    # container, so prefer the raw-sample field when present.
    bit_depth = _int_or_none(stream.get("bits_per_raw_sample")) or _int_or_none(
        stream.get("bits_per_sample")
    )
    bitrate = _int_or_none(stream.get("bit_rate")) or _int_or_none(fmt.get("bit_rate"))

    try:
        duration = float(fmt["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AudioProcessingError(f"ffprobe reported no duration for {path}") from exc
    if duration <= 0:
        raise AudioProcessingError(f"audio has non-positive duration: {path}")

    return AudioProbe(
        codec_name=str(stream.get("codec_name", "")),
        sample_rate=int(stream.get("sample_rate", 0)),
        channels=int(stream.get("channels", 0)),
        duration_seconds=duration,
        file_size=path.stat().st_size,
        bit_depth=bit_depth,
        bitrate_bps=bitrate,
    )


def transcode_master_wav(source: Path, destination: Path) -> AudioProbe:
    """Normalize any input audio to the production master WAV format.

    48 kHz, stereo, 24-bit little-endian PCM. Format conversion only —
    see the module docstring for what is deliberately not done here.
    """
    if not source.is_file():
        raise AudioProcessingError(f"source audio does not exist: {source}")

    ffmpeg = _require_binary("ffmpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
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
            "-map_metadata",
            "-1",
            "-c:a",
            "pcm_s24le",
            "-ar",
            str(MASTER_SAMPLE_RATE),
            "-ac",
            str(MASTER_CHANNELS),
            str(destination),
        ]
    )

    probe = probe_audio(destination)
    if (
        probe.sample_rate != MASTER_SAMPLE_RATE
        or probe.channels != MASTER_CHANNELS
        or probe.bit_depth != MASTER_BIT_DEPTH
    ):
        raise AudioProcessingError(
            "master transcode did not produce the required format "
            f"(got {probe.sample_rate}Hz/{probe.channels}ch/{probe.bit_depth}bit)"
        )
    return probe


def encode_preview_mp3(source: Path, destination: Path) -> AudioProbe:
    """Encode the preview MP3 from an already-normalized master.

    48 kHz, stereo, 320 kbps constant bitrate. The preview never replaces
    the master; it exists for fast in-browser playback.
    """
    if not source.is_file():
        raise AudioProcessingError(f"source audio does not exist: {source}")

    ffmpeg = _require_binary("ffmpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
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
            "-map_metadata",
            "-1",
            "-c:a",
            "libmp3lame",
            # -b:a with CBR mode: no VBR, no ABR.
            "-b:a",
            f"{PREVIEW_BITRATE_BPS // 1000}k",
            "-ar",
            str(PREVIEW_SAMPLE_RATE),
            "-ac",
            str(PREVIEW_CHANNELS),
            "-write_xing",
            "0",
            str(destination),
        ]
    )

    probe = probe_audio(destination)
    if probe.sample_rate != PREVIEW_SAMPLE_RATE or probe.channels != PREVIEW_CHANNELS:
        raise AudioProcessingError(
            "preview encode did not produce the required format "
            f"(got {probe.sample_rate}Hz/{probe.channels}ch)"
        )
    return probe


async def transcode_master_wav_async(source: Path, destination: Path) -> AudioProbe:
    """Off-thread :func:`transcode_master_wav` for async callers."""
    return await asyncio.to_thread(transcode_master_wav, source, destination)


async def encode_preview_mp3_async(source: Path, destination: Path) -> AudioProbe:
    """Off-thread :func:`encode_preview_mp3` for async callers."""
    return await asyncio.to_thread(encode_preview_mp3, source, destination)
