"""Objective musical-structure analysis.

**This is not a substitute for listening.** Nothing here can judge
whether a melody is good, whether a vocal sounds human, or whether
Korean is pronounced correctly. Those require ears and belong to the
human rubric.

What this *can* do is measure properties that correlate with specific
musical failure modes and that a human evaluator would otherwise have
to estimate by feel:

* **Sectional contrast** — does the track change over time, or is it one
  texture end to end? (structure collapse, generic composition)
* **Self-similarity / repetition** — do later sections restate earlier
  ones, as a chorus should? (chorus recurrence vs endless loop)
* **Long-form drift** — do spectrum and level wander between the first
  and last third? (the classic long-form failure)
* **Seed divergence** — how different are two takes of the same prompt?

Every number here is a *measurement*, reported alongside human scores,
never in place of them.
"""

from __future__ import annotations

import itertools
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

#: Analysis frame length in seconds.
FRAME_SECONDS = 1.0
#: Bands (Hz) used for a coarse spectral profile.
BAND_EDGES = (0, 120, 500, 2000, 6000, 20000)


@dataclass
class StructureAnalysis:
    frames: int
    frame_seconds: float
    #: Std-dev of per-frame loudness; low means one flat texture.
    energy_variation: float
    #: Mean absolute spectral change between adjacent frames.
    spectral_flux: float
    #: Highest off-diagonal self-similarity: how strongly some later
    #: section restates an earlier one.
    max_repetition: float
    #: Mean self-similarity across the whole track.
    mean_self_similarity: float
    #: Number of detected section boundaries (novelty peaks).
    section_changes: int
    #: Spectral centroid in Hz, averaged over the track.
    spectral_centroid_hz: float
    #: Absolute spectral difference between the first and last third.
    long_form_drift: float
    #: Loudness difference (dB) between the first and last third.
    level_drift_db: float
    band_energy: list[float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        channels = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())

    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        # 24-bit little-endian: widen to int32 with sign extension.
        buf = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
        packed = buf[:, 0] | (buf[:, 1] << 8) | (buf[:, 2] << 16)
        signed = np.where(packed & 0x800000, packed.astype(np.int64) - (1 << 24), packed)
        data = signed.astype(np.float32) / float(1 << 23)
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / float(1 << 31)
    else:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

    if channels > 1:
        usable = (len(data) // channels) * channels
        data = data[:usable].reshape(-1, channels).mean(axis=1)
    return data, rate


def analyze_structure(path: Path, *, frame_seconds: float = FRAME_SECONDS) -> StructureAnalysis:
    """Measure structural properties of one rendered track."""
    samples, rate = _load_mono(path)
    frame_len = max(1, int(rate * frame_seconds))
    frame_count = max(1, len(samples) // frame_len)
    frames = samples[: frame_count * frame_len].reshape(frame_count, frame_len)

    window = np.hanning(frame_len).astype(np.float32)
    spectra = np.abs(np.fft.rfft(frames * window, axis=1))
    freqs = np.fft.rfftfreq(frame_len, 1 / rate)

    # Log-magnitude spectra, L2-normalized per frame, so similarity
    # compares timbre/content rather than loudness.
    log_spec = np.log1p(spectra)
    norms = np.linalg.norm(log_spec, axis=1, keepdims=True)
    normed = log_spec / np.maximum(norms, 1e-9)

    energy = np.sqrt((frames**2).mean(axis=1))
    energy_db = 20 * np.log10(np.maximum(energy, 1e-9))
    energy_variation = float(np.std(energy_db))

    flux = float(np.mean(np.abs(np.diff(normed, axis=0)).sum(axis=1))) if frame_count > 1 else 0.0

    similarity = normed @ normed.T
    if frame_count > 2:
        # Ignore the diagonal band: adjacent frames are trivially similar.
        mask = np.abs(np.subtract.outer(np.arange(frame_count), np.arange(frame_count))) > max(
            2, frame_count // 10
        )
        off = similarity[mask]
        max_repetition = float(off.max()) if off.size else 0.0
        mean_similarity = float(off.mean()) if off.size else 0.0
    else:
        max_repetition = 0.0
        mean_similarity = 0.0

    # Novelty: how much the spectrum changes frame to frame; peaks above
    # mean + 1 sd are treated as section boundaries.
    if frame_count > 3:
        novelty = np.abs(np.diff(normed, axis=0)).sum(axis=1)
        threshold = novelty.mean() + novelty.std()
        section_changes = int(np.sum(novelty > threshold))
    else:
        section_changes = 0

    magnitude = spectra.mean(axis=0)
    centroid = float((freqs * magnitude).sum() / max(magnitude.sum(), 1e-9))

    bands: list[float] = []
    total = magnitude.sum()
    for low, high in itertools.pairwise(BAND_EDGES):
        sel = (freqs >= low) & (freqs < high)
        bands.append(float(magnitude[sel].sum() / max(total, 1e-9)))

    third = max(1, frame_count // 3)
    if frame_count >= 3:
        head = normed[:third].mean(axis=0)
        tail = normed[-third:].mean(axis=0)
        drift = float(np.abs(head - tail).sum())
        level_drift = float(energy_db[-third:].mean() - energy_db[:third].mean())
    else:
        drift = 0.0
        level_drift = 0.0

    return StructureAnalysis(
        frames=frame_count,
        frame_seconds=frame_seconds,
        energy_variation=round(energy_variation, 3),
        spectral_flux=round(flux, 4),
        max_repetition=round(max_repetition, 4),
        mean_self_similarity=round(mean_similarity, 4),
        section_changes=section_changes,
        spectral_centroid_hz=round(centroid, 1),
        long_form_drift=round(drift, 4),
        level_drift_db=round(level_drift, 2),
        band_energy=[round(b, 4) for b in bands],
    )


def seed_divergence(paths: list[Path]) -> float | None:
    """Mean pairwise spectral distance between takes of one prompt.

    0 would mean the seeds produced acoustically identical audio; larger
    values mean the model explores more per seed.
    """
    if len(paths) < 2:
        return None
    profiles: list[np.ndarray] = []
    for path in paths:
        samples, rate = _load_mono(path)
        # Compare a common prefix so different lengths do not dominate.
        clip = samples[: rate * 20]
        spectrum = np.abs(np.fft.rfft(clip * np.hanning(len(clip)).astype(np.float32)))
        log_spec = np.log1p(spectrum)
        profiles.append(log_spec / max(float(np.linalg.norm(log_spec)), 1e-9))

    size = min(len(p) for p in profiles)
    trimmed = [p[:size] for p in profiles]
    distances = [
        float(np.linalg.norm(trimmed[i] - trimmed[j]))
        for i in range(len(trimmed))
        for j in range(i + 1, len(trimmed))
    ]
    return round(sum(distances) / len(distances), 4) if distances else None
