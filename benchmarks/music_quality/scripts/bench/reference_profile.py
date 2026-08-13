"""Aggregate technical profile of commercially released music.

Phase 10. The purpose is to turn "the model sounds too bright" into a
measurable statement, by establishing what the technical descriptors of
commercially released music actually look like *in aggregate*.

**This is reference analysis, not training data, and not a musical
target.** The rules this module is built to enforce:

- Nothing is copied into the repository. Only aggregate statistics
  leave this function.
- No per-track values are ever emitted. The output is medians and
  percentile ranges over a cohort.
- No filenames, paths, tags, or anything identifying a specific
  recording appears in the output.
- No fingerprint or waveform data is persisted. File hashes are
  computed in memory purely to skip duplicates within a run.
- The result is an **audio target band**, not a spectral shape to copy.
  Matching the median of a distribution is not reproducing a recording.

Measurement strategy: loudness, peak and RMS come from ffmpeg's own
`ebur128` and `astats` filters over the whole file, because those are
well-tested C implementations of standard measurements. Spectral band
analysis runs in numpy over a bounded middle excerpt of each track,
which is both far cheaper and more representative of a song's body than
its intro or fade-out.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: Analysis sample rate. High enough for the 10 kHz+ questions Phase 10
#: is asking, and matches LUBER's own delivery rate.
ANALYSIS_RATE = 48000

#: Seconds of the middle of each track used for spectral analysis. The
#: body of a song is more representative of its tonal balance than its
#: intro or outro, and a bounded excerpt keeps 200-file runs tractable.
EXCERPT_SECONDS = 60.0

#: Broad bands, in Hz. Deliberately coarse: Phase 10 corrects toward a
#: distribution, not toward a specific recording's spectral shape.
BANDS: dict[str, tuple[float, float]] = {
    "low": (20.0, 250.0),
    "mid": (250.0, 4000.0),
    "high": (4000.0, 20000.0),
}

#: The two thresholds the Phase 9 brightness finding is stated against.
HF_THRESHOLDS_HZ = (8000.0, 10000.0)

#: Percentiles reported for every descriptor. p10-p90 is the "target
#: band"; the median is the centre of mass, not a goal to hit exactly.
PERCENTILES = (10, 25, 50, 75, 90)


@dataclass
class TrackDescriptors:
    """Technical descriptors of one track. Never persisted."""

    source_sample_rate: int = 0
    channels: int = 0
    duration_seconds: float = 0.0
    peak_dbfs: float | None = None
    rms_dbfs: float | None = None
    crest_factor_db: float | None = None
    integrated_lufs: float | None = None
    loudness_range_lu: float | None = None
    spectral_centroid_hz: float = 0.0
    spectral_rolloff85_hz: float = 0.0
    energy_above_8k: float = 0.0
    energy_above_10k: float = 0.0
    band_low: float = 0.0
    band_mid: float = 0.0
    band_high: float = 0.0
    stereo_correlation: float | None = None
    silence_ratio: float = 0.0
    dynamic_range_proxy_db: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def _probe(path: Path) -> tuple[int, int, float]:
    """Original sample rate, channel count and duration via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    stream = (data.get("streams") or [{}])[0]
    return (
        int(stream.get("sample_rate", 0) or 0),
        int(stream.get("channels", 0) or 0),
        float((data.get("format") or {}).get("duration", 0.0) or 0.0),
    )


_EBUR128_I = re.compile(r"\bI:\s*(-?[\d.]+)\s*LUFS")
_EBUR128_LRA = re.compile(r"\bLRA:\s*(-?[\d.]+)\s*LU")
_ASTATS = {
    "peak_dbfs": re.compile(r"Peak level dB:\s*(-?[\d.]+|-?inf)"),
    "rms_dbfs": re.compile(r"RMS level dB:\s*(-?[\d.]+|-?inf)"),
}


def _loudness_and_levels(path: Path) -> dict[str, float | None]:
    """Whole-file loudness and level stats from ffmpeg's own filters."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-v",
            "info",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-af",
            "ebur128=peak=true,astats=measure_perchannel=none",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    log = result.stderr
    out: dict[str, float | None] = {}
    # ebur128 prints a running measurement per frame and the final
    # integrated figure last, so the last match is the real answer —
    # the first is the -70 LUFS placeholder from t=0.
    found = _EBUR128_I.findall(log)
    out["integrated_lufs"] = float(found[-1]) if found else None
    found = _EBUR128_LRA.findall(log)
    out["loudness_range_lu"] = float(found[-1]) if found else None
    for key, pattern in _ASTATS.items():
        matches = pattern.findall(log)
        value = matches[-1] if matches else None
        try:
            out[key] = float(value) if value not in (None, "inf", "-inf") else None
        except ValueError:
            out[key] = None
    # astats reports peak and RMS but no crest factor in this mode;
    # crest is their difference by definition.
    peak, rms = out.get("peak_dbfs"), out.get("rms_dbfs")
    out["crest_factor_db"] = peak - rms if peak is not None and rms is not None else None
    return out


def _decode_excerpt(path: Path, duration: float) -> np.ndarray:
    """Decode the middle ``EXCERPT_SECONDS`` as float32 stereo."""
    start = max(0.0, (duration - EXCERPT_SECONDS) / 2.0) if duration > EXCERPT_SECONDS else 0.0
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-v",
            "error",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{EXCERPT_SECONDS:.3f}",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-ac",
            "2",
            "-ar",
            str(ANALYSIS_RATE),
            "-f",
            "f32le",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    samples = np.frombuffer(result.stdout, dtype="<f4")
    usable = (samples.size // 2) * 2
    return samples[:usable].reshape(-1, 2)


def _spectral(mono: np.ndarray) -> dict[str, float]:
    """Spectral descriptors from an averaged magnitude spectrum."""
    frame, hop = 4096, 2048
    if mono.size < frame:
        return {}
    window = np.hanning(frame).astype(np.float32)
    frames = [
        np.abs(np.fft.rfft(mono[start : start + frame] * window))
        for start in range(0, mono.size - frame + 1, hop)
    ]
    magnitude = np.mean(frames, axis=0)
    freqs = np.fft.rfftfreq(frame, 1.0 / ANALYSIS_RATE)
    total = float(np.sum(magnitude))
    if total <= 0:
        return {}

    cumulative = np.cumsum(magnitude) / total
    rolloff_index = int(np.searchsorted(cumulative, 0.85))
    out: dict[str, float] = {
        "spectral_centroid_hz": float(np.sum(freqs * magnitude) / total),
        "spectral_rolloff85_hz": float(freqs[min(rolloff_index, freqs.size - 1)]),
        "energy_above_8k": float(np.sum(magnitude[freqs >= 8000.0]) / total),
        "energy_above_10k": float(np.sum(magnitude[freqs >= 10000.0]) / total),
    }
    for name, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs < high)
        out[f"band_{name}"] = float(np.sum(magnitude[mask]) / total)
    return out


def describe_track(path: Path) -> TrackDescriptors | None:
    """Measure one file. Returns ``None`` if it cannot be decoded."""
    try:
        rate, channels, duration = _probe(path)
        stereo = _decode_excerpt(path, duration)
    except (subprocess.CalledProcessError, ValueError, json.JSONDecodeError):
        return None
    if stereo.size == 0:
        return None

    descriptors = TrackDescriptors(
        source_sample_rate=rate, channels=channels, duration_seconds=duration
    )
    for key, value in _loudness_and_levels(path).items():
        setattr(descriptors, key, value)

    mono = stereo.mean(axis=1)
    for key, value in _spectral(mono).items():
        setattr(descriptors, key, value)

    if stereo.shape[1] == 2 and np.any(stereo[:, 0]) and np.any(stereo[:, 1]):
        correlation = float(np.corrcoef(stereo[:, 0], stereo[:, 1])[0, 1])
        descriptors.stereo_correlation = correlation if np.isfinite(correlation) else None

    descriptors.silence_ratio = float(np.mean(np.abs(mono) < 0.001))

    # Dynamic range proxy: spread between the loudest and quietest
    # one-second window, which tracks how compressed a master is.
    window = ANALYSIS_RATE
    if mono.size >= window * 4:
        count = mono.size // window
        rms = np.sqrt(np.mean(mono[: count * window].reshape(count, window) ** 2, axis=1))
        rms = rms[rms > 0]
        if rms.size >= 4:
            loud = float(20 * np.log10(np.percentile(rms, 95)))
            quiet = float(20 * np.log10(np.percentile(rms, 10)))
            descriptors.dynamic_range_proxy_db = loud - quiet
    return descriptors


def _summarise(values: list[float]) -> dict[str, float | int] | None:
    clean = [v for v in values if v is not None and np.isfinite(v)]
    if not clean:
        return None
    summary: dict[str, float | int] = {"n": len(clean)}
    for p in PERCENTILES:
        summary[f"p{p}"] = round(float(np.percentile(clean, p)), 6)
    summary["mean"] = round(float(statistics.fmean(clean)), 6)
    summary["min"] = round(min(clean), 6)
    summary["max"] = round(max(clean), 6)
    return summary


@dataclass
class ReferenceProfile:
    """Aggregate-only description of a cohort of commercial recordings."""

    cohort: str
    track_count: int
    descriptors: dict[str, dict[str, float | int]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cohort": self.cohort,
            "track_count": self.track_count,
            "descriptors": self.descriptors,
            "notes": self.notes,
        }

    def band(self, name: str) -> tuple[float, float] | None:
        """The p10-p90 target band for one descriptor."""
        entry = self.descriptors.get(name)
        if not entry:
            return None
        return float(entry["p10"]), float(entry["p90"])

    def median(self, name: str) -> float | None:
        entry = self.descriptors.get(name)
        return float(entry["p50"]) if entry else None


#: Descriptors aggregated into the profile. Anything not listed here is
#: measured for the run and discarded.
AGGREGATED = (
    "source_sample_rate",
    "channels",
    "duration_seconds",
    "peak_dbfs",
    "rms_dbfs",
    "crest_factor_db",
    "integrated_lufs",
    "loudness_range_lu",
    "spectral_centroid_hz",
    "spectral_rolloff85_hz",
    "energy_above_8k",
    "energy_above_10k",
    "band_low",
    "band_mid",
    "band_high",
    "stereo_correlation",
    "silence_ratio",
    "dynamic_range_proxy_db",
)


def build_profile(
    paths: list[Path], *, cohort: str, progress: bool = False
) -> tuple[ReferenceProfile, int]:
    """Measure *paths* and reduce them to aggregate statistics only.

    Duplicates are skipped by content hash, computed in memory and
    discarded — the profile never carries a fingerprint of any input.
    """
    seen: set[str] = set()
    collected: dict[str, list[float]] = {name: [] for name in AGGREGATED}
    analysed = failed = duplicates = 0

    for index, path in enumerate(sorted(paths), start=1):
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            failed += 1
            continue
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)

        descriptors = describe_track(path)
        if descriptors is None:
            failed += 1
            continue
        analysed += 1
        for name in AGGREGATED:
            value = getattr(descriptors, name, None)
            if value is not None:
                collected[name].append(float(value))
        if progress and index % 20 == 0:
            print(f"  analysed {analysed}/{len(paths)}", flush=True)

    profile = ReferenceProfile(cohort=cohort, track_count=analysed)
    for name, values in collected.items():
        summary = _summarise(values)
        if summary:
            profile.descriptors[name] = summary
    profile.notes = [
        "Aggregate statistics only. No per-track values, filenames, paths, or "
        "fingerprints are recorded.",
        f"Spectral descriptors measured over the middle {EXCERPT_SECONDS:.0f}s of each "
        f"track at {ANALYSIS_RATE} Hz; loudness and level from ffmpeg ebur128/astats "
        f"over the whole file.",
        "This is an audio target band, not a musical or spectral target. It is not "
        "training data and was not used to train anything.",
        f"{duplicates} duplicate file(s) skipped by content hash; {failed} file(s) "
        f"could not be decoded.",
    ]
    return profile, analysed
