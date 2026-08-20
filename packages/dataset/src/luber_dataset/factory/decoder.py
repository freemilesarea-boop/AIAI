"""Proving a file is audio before anything expensive touches it.

A library assembled over years contains files that are not what their
extension claims: truncated downloads, zero-length placeholders, a JPEG
somebody renamed, an MP3 whose last frames never arrived. Every one of
them will crash or silently mislead an analysis stage further down.

So each candidate is decoded before it is measured, and the result is
recorded rather than raised. **A corrupt file must cost one record, not
the run.** Twelve hours into a scan of forty thousand files is the worst
possible moment to discover that one of them raises.

Decoding is done with ffprobe and ffmpeg, which the delivery pipeline
already requires, so the factory adds no new binary dependency.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class DecodeStatus(StrEnum):
    VALID = "VALID"
    #: Opens and reports properties, but decoding hit an error partway.
    #: Usually a truncated download; the audio up to that point is real.
    PARTIAL = "PARTIAL"
    #: Not decodable as audio at all.
    INVALID = "INVALID"
    #: Recognised as media, but not audio this toolchain can read.
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class DecodeResult:
    status: DecodeStatus
    decode_error: str | None = None
    duration_seconds: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    bit_depth: int | None = None
    codec: str | None = None
    container: str | None = None

    @property
    def usable(self) -> bool:
        """PARTIAL counts: truncated audio is still audio.

        Whether to *train* on it is a quality decision made later with
        the flag visible, not a decoding decision made here.
        """
        return self.status in (DecodeStatus.VALID, DecodeStatus.PARTIAL)


class DecoderUnavailableError(RuntimeError):
    """Raised when ffprobe/ffmpeg are absent.

    Distinct from a file being undecodable: the whole run is wrong, and
    marking forty thousand files INVALID would be a lie about the audio.
    """


def _binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise DecoderUnavailableError(
            f"{name} is required for dataset decode validation but is not on PATH"
        )
    return path


def _run(command: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    # Fixed argv, never a shell string: filenames in a music library
    # contain quotes, semicolons and newlines.
    return subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout)


def _first_audio_stream(payload: dict[str, object]) -> dict[str, object] | None:
    streams = payload.get("streams")
    if not isinstance(streams, list):
        return None
    for stream in streams:
        if isinstance(stream, dict) and stream.get("codec_type") == "audio":
            return stream
    return None


def _as_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _as_float(value: object) -> float | None:
    try:
        result = float(str(value))
    except (TypeError, ValueError):
        return None
    return None if result != result else result  # NaN is not a duration


def _bit_depth(stream: dict[str, object]) -> int | None:
    """Only for formats that genuinely have one.

    ffprobe reports `bits_per_raw_sample` for PCM and lossless codecs and
    leaves it empty for lossy ones, where the concept does not apply. An
    MP3 has no bit depth, and reporting the decoder's output width as
    though it were a property of the source would be inventing a fact.
    """
    for key in ("bits_per_raw_sample", "bits_per_sample"):
        depth = _as_int(stream.get(key))
        if depth is not None and depth > 0:
            return depth
    return None


def probe(path: Path, *, timeout: float = 60.0) -> DecodeResult:
    """Read declared properties. Cheap; does not prove decodability."""
    try:
        ffprobe = _binary("ffprobe")
    except DecoderUnavailableError:
        raise
    try:
        completed = _run(
            [
                ffprobe,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            timeout,
        )
    except subprocess.TimeoutExpired:
        return DecodeResult(status=DecodeStatus.INVALID, decode_error="ffprobe timed out")

    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().splitlines()
        return DecodeResult(
            status=DecodeStatus.INVALID,
            decode_error=detail[-1] if detail else "ffprobe failed",
        )
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return DecodeResult(status=DecodeStatus.INVALID, decode_error="ffprobe output unreadable")

    stream = _first_audio_stream(payload)
    if stream is None:
        return DecodeResult(
            status=DecodeStatus.UNSUPPORTED,
            decode_error="no audio stream",
            container=str((payload.get("format") or {}).get("format_name") or "") or None,
        )

    fmt = payload.get("format")
    fmt_dict = fmt if isinstance(fmt, dict) else {}
    duration = _as_float(stream.get("duration")) or _as_float(fmt_dict.get("duration"))
    return DecodeResult(
        status=DecodeStatus.VALID,
        duration_seconds=duration,
        sample_rate=_as_int(stream.get("sample_rate")),
        channels=_as_int(stream.get("channels")),
        bit_depth=_bit_depth(stream),
        codec=str(stream.get("codec_name") or "") or None,
        container=str(fmt_dict.get("format_name") or "") or None,
    )


def decode_check(path: Path, *, timeout: float = 600.0) -> DecodeResult:
    """Probe, then actually decode every frame to null.

    The full decode is the point. A truncated MP3 probes perfectly —
    the header is intact and declares a duration the file does not
    contain — and only fails when something reads to the end. Discovering
    that during training is discovering it too late.
    """
    result = probe(path, timeout=min(timeout, 60.0))
    if result.status is not DecodeStatus.VALID:
        return result

    if result.duration_seconds is None or result.duration_seconds <= 0:
        return DecodeResult(
            status=DecodeStatus.INVALID,
            decode_error="no usable duration",
            sample_rate=result.sample_rate,
            channels=result.channels,
            codec=result.codec,
            container=result.container,
        )

    try:
        ffmpeg = _binary("ffmpeg")
        completed = _run(
            [ffmpeg, "-nostdin", "-v", "error", "-i", str(path), "-f", "null", "-"],
            timeout,
        )
    except subprocess.TimeoutExpired:
        return DecodeResult(
            status=DecodeStatus.PARTIAL,
            decode_error="decode timed out",
            duration_seconds=result.duration_seconds,
            sample_rate=result.sample_rate,
            channels=result.channels,
            bit_depth=result.bit_depth,
            codec=result.codec,
            container=result.container,
        )

    stderr = (completed.stderr or "").strip()
    if completed.returncode != 0 or stderr:
        # ffmpeg exits 0 on a recoverable error but still complains, and
        # a complaint means some frames did not survive. The properties
        # are kept: what was read is still real audio.
        detail = stderr.splitlines()
        return DecodeResult(
            status=DecodeStatus.PARTIAL if completed.returncode == 0 else DecodeStatus.INVALID,
            decode_error=detail[-1] if detail else "decode failed",
            duration_seconds=result.duration_seconds,
            sample_rate=result.sample_rate,
            channels=result.channels,
            bit_depth=result.bit_depth,
            codec=result.codec,
            container=result.container,
        )
    return result
