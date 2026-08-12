"""Objective audio measurements and failure detection.

These are *technical* measurements only. A track that passes every check
here can still be musically worthless — technical metrics and musical
quality are deliberately kept in separate namespaces so a clean
waveform is never mistaken for a good song.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ── objective failure flags ───────────────────────────────────────────
SILENT_OUTPUT = "SILENT_OUTPUT"
NEAR_SILENT_OUTPUT = "NEAR_SILENT_OUTPUT"
CLIPPING = "CLIPPING"
INVALID_DURATION = "INVALID_DURATION"
CORRUPTED_AUDIO = "CORRUPTED_AUDIO"
GENERATION_FAILED = "GENERATION_FAILED"
EXCESSIVE_SILENCE = "EXCESSIVE_SILENCE"
TOO_SHORT = "TOO_SHORT"
TOO_LONG = "TOO_LONG"

#: Peak below this (relative to full scale) counts as silence.
SILENCE_PEAK_RATIO = 0.001
NEAR_SILENCE_PEAK_RATIO = 0.02
#: Share of samples at/near full scale that indicates real clipping.
CLIPPING_SAMPLE_RATIO = 0.001
#: Share of the track allowed to be silent before it is suspicious.
EXCESSIVE_SILENCE_RATIO = 0.35
#: Tolerance on requested vs actual duration.
DURATION_TOLERANCE_RATIO = 0.20


@dataclass
class AudioMetrics:
    """Objective properties of one generated file."""

    file_size: int = 0
    sample_rate: int = 0
    channels: int = 0
    bit_depth: int | None = None
    duration_seconds: float = 0.0
    peak: float = 0.0
    peak_dbfs: float = -math.inf
    rms: float = 0.0
    rms_dbfs: float = -math.inf
    silence_ratio: float = 0.0
    clipping_sample_ratio: float = 0.0
    dc_offset: float = 0.0
    decoded: bool = False
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        # -inf is not valid JSON.
        for key in ("peak_dbfs", "rms_dbfs"):
            value = data[key]
            if isinstance(value, float) and not math.isfinite(value):
                data[key] = None
        return data


def _dbfs(ratio: float) -> float:
    return 20 * math.log10(ratio) if ratio > 0 else -math.inf


def measure_wav(path: Path, *, requested_duration: float | None = None) -> AudioMetrics:
    """Measure a PCM WAV file with the standard library only.

    Never raises for bad audio: a file that cannot be decoded comes back
    flagged ``CORRUPTED_AUDIO`` so the benchmark records the failure
    instead of aborting the run.
    """
    metrics = AudioMetrics()
    if not path.is_file():
        metrics.flags.append(CORRUPTED_AUDIO)
        return metrics

    metrics.file_size = path.stat().st_size
    if metrics.file_size == 0:
        metrics.flags.append(CORRUPTED_AUDIO)
        return metrics

    try:
        with wave.open(str(path), "rb") as w:
            channels = w.getnchannels()
            width = w.getsampwidth()
            rate = w.getframerate()
            frames = w.getnframes()
            raw = w.readframes(frames)
    except Exception:
        metrics.flags.append(CORRUPTED_AUDIO)
        return metrics

    metrics.decoded = True
    metrics.channels = channels
    metrics.bit_depth = width * 8
    metrics.sample_rate = rate
    metrics.duration_seconds = frames / rate if rate else 0.0

    if rate <= 0 or channels <= 0 or frames <= 0:
        metrics.flags.append(INVALID_DURATION)
        return metrics

    full_scale = float(1 << (width * 8 - 1))
    samples = [
        int.from_bytes(raw[i : i + width], "little", signed=True)
        for i in range(0, len(raw) - width + 1, width)
    ]
    if not samples:
        metrics.flags.append(CORRUPTED_AUDIO)
        return metrics

    peak_abs = max(max(samples), -min(samples))
    metrics.peak = peak_abs / full_scale
    metrics.peak_dbfs = _dbfs(metrics.peak)
    metrics.rms = math.sqrt(sum(s * s for s in samples) / len(samples)) / full_scale
    metrics.rms_dbfs = _dbfs(metrics.rms)
    metrics.dc_offset = (sum(samples) / len(samples)) / full_scale

    near_full = full_scale * 0.999
    metrics.clipping_sample_ratio = sum(1 for s in samples if abs(s) >= near_full) / len(samples)

    # Silence measured per 100 ms window rather than per sample, so a
    # zero-crossing is not counted as silence.
    window = max(1, int(rate * 0.1) * channels)
    quiet_windows = 0
    total_windows = 0
    threshold = full_scale * SILENCE_PEAK_RATIO
    for start in range(0, len(samples), window):
        chunk = samples[start : start + window]
        if not chunk:
            continue
        total_windows += 1
        if max(max(chunk), -min(chunk)) < threshold:
            quiet_windows += 1
    metrics.silence_ratio = quiet_windows / total_windows if total_windows else 0.0

    metrics.flags = _flag(metrics, requested_duration)
    return metrics


def _flag(metrics: AudioMetrics, requested_duration: float | None) -> list[str]:
    flags: list[str] = []
    if metrics.peak < SILENCE_PEAK_RATIO:
        flags.append(SILENT_OUTPUT)
    elif metrics.peak < NEAR_SILENCE_PEAK_RATIO:
        flags.append(NEAR_SILENT_OUTPUT)

    if metrics.clipping_sample_ratio > CLIPPING_SAMPLE_RATIO:
        flags.append(CLIPPING)

    if metrics.silence_ratio > EXCESSIVE_SILENCE_RATIO:
        flags.append(EXCESSIVE_SILENCE)

    if metrics.duration_seconds <= 0:
        flags.append(INVALID_DURATION)
    elif requested_duration:
        low = requested_duration * (1 - DURATION_TOLERANCE_RATIO)
        high = requested_duration * (1 + DURATION_TOLERANCE_RATIO)
        if metrics.duration_seconds < low:
            flags.append(TOO_SHORT)
        elif metrics.duration_seconds > high:
            flags.append(TOO_LONG)
    return flags


def probe_with_ffprobe(path: Path) -> dict[str, object]:
    """Optional richer probe; returns {} when ffprobe is unavailable."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None or not path.is_file():
        return {}
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name,sample_rate,channels,bit_rate,bits_per_raw_sample",
                "-show_entries",
                "format=duration,size",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        parsed: dict[str, object] = json.loads(result.stdout)
        return parsed
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return {}


def real_time_factor(generation_seconds: float, audio_seconds: float) -> float | None:
    """Wall-clock seconds spent per second of audio produced."""
    if audio_seconds <= 0:
        return None
    return generation_seconds / audio_seconds
