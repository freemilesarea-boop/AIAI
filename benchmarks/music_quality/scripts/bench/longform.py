"""Long-form QA: window drift, sibilance risk, and control verification.

Phase 5 found frequency-balance and high-end problems in 30-second
output. The question this module exists to answer is whether those
problems get *worse* over the length of a full song — a defect you
cannot see by measuring a track as one lump.

Everything here is **observability**. Nothing in this module changes
audio, and nothing here fails a generation. Where a measurement cannot
be made honestly, the result says so rather than guessing.

Three deliberate limitations, stated up front because they bound every
number this module produces:

- **`SIBILANCE_RISK_PROXY` is not a sibilance detector.** It is the
  share of energy in the 5-10 kHz band. Cymbals, synths and noise live
  there too. It is useful for *comparing windows of the same track* and
  tracks against each other; it is not evidence that a vocal is sibilant.
- **BPM estimation is validated against synthetic click tracks only.**
  It reports a number and a confidence, and disagreement with the
  request is a flag for a human, never a failure.
- **Key estimation is the weakest measurement here.** See
  :func:`estimate_key`; mode (major/minor) in particular is not
  trustworthy on this implementation and is reported as such.

Implemented with numpy and the standard library only — no librosa,
scipy, or essentia is available in this environment, and adding one for
QA tooling was judged not worth the dependency weight.
"""

from __future__ import annotations

import math
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

#: Number of equal windows a long track is split into for drift analysis
#: (0-25%, 25-50%, 50-75%, 75-100%).
WINDOW_COUNT = 4

#: The band the sibilance proxy measures. Human sibilance ("s", "sh",
#: Korean ㅅ/ㅆ/ㅊ) concentrates here, but so does a lot of music.
SIBILANCE_BAND_HZ = (5000.0, 10000.0)
#: "High frequency" for the high-frequency energy ratio.
HIGH_FREQUENCY_CUTOFF_HZ = 8000.0

#: FFT frame for spectral measurements.
SPECTRAL_FRAME = 4096
SPECTRAL_HOP = 2048

#: Tempo search range, matching the engine's own accepted BPM bounds.
BPM_SEARCH_MIN = 30.0
BPM_SEARCH_MAX = 300.0

#: Relative level drift across windows beyond which the track is
#: probably not holding its level. Heuristic, for flagging only.
LEVEL_DRIFT_DB_FLAG = 6.0
#: Growth in the sibilance proxy from first window to last, beyond which
#: the high end is plausibly deteriorating. Heuristic.
SIBILANCE_GROWTH_FLAG = 1.5

#: Krumhansl-Schmuckler profiles, the standard published weights.
_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
#: LUBER stores flats for some keys; map to the sharp spelling we emit.
_ENHARMONIC = {
    "Db": "C#",
    "Eb": "D#",
    "Gb": "F#",
    "Ab": "G#",
    "Bb": "A#",
    "Cb": "B",
    "Fb": "E",
    "E#": "F",
    "B#": "C",
}


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    """Decode a PCM WAV to mono float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as w:
        channels, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())

    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
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


def _dbfs(ratio: float) -> float | None:
    return 20 * math.log10(ratio) if ratio > 0 else None


def _round_optional(value: float | None, digits: int = 2) -> float | None:
    return round(value, digits) if value is not None else None


def _spectrum(samples: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Average magnitude spectrum over Hann-windowed frames."""
    if samples.size < SPECTRAL_FRAME:
        padded = np.zeros(SPECTRAL_FRAME, dtype=np.float32)
        padded[: samples.size] = samples
        samples = padded
    window = np.hanning(SPECTRAL_FRAME).astype(np.float32)
    frames = []
    for start in range(0, samples.size - SPECTRAL_FRAME + 1, SPECTRAL_HOP):
        frames.append(np.abs(np.fft.rfft(samples[start : start + SPECTRAL_FRAME] * window)))
    magnitude = np.mean(frames, axis=0) if frames else np.zeros(SPECTRAL_FRAME // 2 + 1)
    freqs = np.fft.rfftfreq(SPECTRAL_FRAME, 1.0 / rate)
    return freqs, magnitude


@dataclass
class WindowMetrics:
    """Measurements for one slice of the track."""

    index: int
    start_seconds: float
    end_seconds: float
    peak: float
    rms: float
    rms_dbfs: float | None
    crest_factor_db: float | None
    spectral_centroid_hz: float
    high_frequency_ratio: float
    sibilance_risk_proxy: float
    silence_ratio: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class LongFormAnalysis:
    """Whole-track measurements plus per-window drift."""

    duration_seconds: float
    sample_rate: int
    channels: int
    bit_depth: int | None
    peak: float
    peak_dbfs: float | None
    rms: float
    rms_dbfs: float | None
    crest_factor_db: float | None
    clipping_sample_ratio: float
    silence_ratio: float
    spectral_centroid_hz: float
    high_frequency_ratio: float
    sibilance_risk_proxy: float
    windows: list[WindowMetrics] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def level_drift_db(self) -> float | None:
        """Spread between the loudest and quietest window, in dB."""
        levels = [w.rms_dbfs for w in self.windows if w.rms_dbfs is not None]
        return max(levels) - min(levels) if len(levels) >= 2 else None

    @property
    def sibilance_growth(self) -> float | None:
        """Last window's sibilance proxy relative to the first."""
        if len(self.windows) < 2 or self.windows[0].sibilance_risk_proxy <= 0:
            return None
        return self.windows[-1].sibilance_risk_proxy / self.windows[0].sibilance_risk_proxy

    @property
    def centroid_drift_hz(self) -> float | None:
        if len(self.windows) < 2:
            return None
        centroids = [w.spectral_centroid_hz for w in self.windows]
        return max(centroids) - min(centroids)

    def to_dict(self) -> dict[str, object]:
        return {
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bit_depth": self.bit_depth,
            "peak": self.peak,
            "peak_dbfs": self.peak_dbfs,
            "rms": self.rms,
            "rms_dbfs": self.rms_dbfs,
            "crest_factor_db": self.crest_factor_db,
            "clipping_sample_ratio": self.clipping_sample_ratio,
            "silence_ratio": self.silence_ratio,
            "spectral_centroid_hz": self.spectral_centroid_hz,
            "high_frequency_ratio": self.high_frequency_ratio,
            "sibilance_risk_proxy": self.sibilance_risk_proxy,
            "level_drift_db": self.level_drift_db,
            "sibilance_growth": self.sibilance_growth,
            "centroid_drift_hz": self.centroid_drift_hz,
            "windows": [w.to_dict() for w in self.windows],
            "flags": list(self.flags),
        }


def _measure_segment(
    samples: np.ndarray, rate: int, index: int, start: float, end: float
) -> WindowMetrics:
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    rms_db = _dbfs(rms)
    peak_db = _dbfs(peak)
    crest = (peak_db - rms_db) if (peak_db is not None and rms_db is not None) else None

    freqs, magnitude = _spectrum(samples, rate)
    total = float(np.sum(magnitude))
    if total > 0:
        centroid = float(np.sum(freqs * magnitude) / total)
        high = float(np.sum(magnitude[freqs >= HIGH_FREQUENCY_CUTOFF_HZ]) / total)
        band = (freqs >= SIBILANCE_BAND_HZ[0]) & (freqs < SIBILANCE_BAND_HZ[1])
        sibilance = float(np.sum(magnitude[band]) / total)
    else:
        centroid = high = sibilance = 0.0

    quiet = float(np.mean(np.abs(samples) < 0.001)) if samples.size else 1.0
    return WindowMetrics(
        index=index,
        start_seconds=round(start, 2),
        end_seconds=round(end, 2),
        peak=round(peak, 6),
        rms=round(rms, 6),
        rms_dbfs=round(rms_db, 2) if rms_db is not None else None,
        crest_factor_db=round(crest, 2) if crest is not None else None,
        spectral_centroid_hz=round(centroid, 1),
        high_frequency_ratio=round(high, 5),
        sibilance_risk_proxy=round(sibilance, 5),
        silence_ratio=round(quiet, 4),
    )


def analyze_long_form(path: Path, *, window_count: int = WINDOW_COUNT) -> LongFormAnalysis:
    """Measure a rendered track as a whole and across equal windows."""
    samples, rate = load_mono(path)
    with wave.open(str(path), "rb") as w:
        channels, width = w.getnchannels(), w.getsampwidth()

    duration = samples.size / rate if rate else 0.0
    whole = _measure_segment(samples, rate, -1, 0.0, duration)
    clipping = float(np.mean(np.abs(samples) >= 0.999)) if samples.size else 0.0

    windows: list[WindowMetrics] = []
    if samples.size and window_count > 0:
        edges = np.linspace(0, samples.size, window_count + 1).astype(int)
        for i in range(window_count):
            segment = samples[edges[i] : edges[i + 1]]
            windows.append(_measure_segment(segment, rate, i, edges[i] / rate, edges[i + 1] / rate))

    analysis = LongFormAnalysis(
        duration_seconds=round(duration, 3),
        sample_rate=rate,
        channels=channels,
        bit_depth=width * 8,
        peak=whole.peak,
        peak_dbfs=_round_optional(_dbfs(whole.peak)),
        rms=whole.rms,
        rms_dbfs=whole.rms_dbfs,
        crest_factor_db=whole.crest_factor_db,
        clipping_sample_ratio=round(clipping, 6),
        silence_ratio=whole.silence_ratio,
        spectral_centroid_hz=whole.spectral_centroid_hz,
        high_frequency_ratio=whole.high_frequency_ratio,
        sibilance_risk_proxy=whole.sibilance_risk_proxy,
        windows=windows,
    )

    drift = analysis.level_drift_db
    if drift is not None and drift > LEVEL_DRIFT_DB_FLAG:
        analysis.flags.append("LEVEL_DRIFT")
    growth = analysis.sibilance_growth
    if growth is not None and growth > SIBILANCE_GROWTH_FLAG:
        analysis.flags.append("SIBILANCE_GROWTH")
    if clipping > 0.001:
        analysis.flags.append("CLIPPING")
    if analysis.silence_ratio > 0.35:
        analysis.flags.append("EXCESSIVE_SILENCE")
    return analysis


# ── BPM estimation ────────────────────────────────────────────────────


@dataclass
class TempoEstimate:
    """An estimated tempo and how much to trust it."""

    estimated_bpm: float | None
    #: 0-1, from the autocorrelation peak's prominence. Not a
    #: probability — only comparable between runs of this estimator.
    confidence: float
    requested_bpm: int | None = None

    @property
    def difference(self) -> float | None:
        if self.estimated_bpm is None or self.requested_bpm is None:
            return None
        return abs(self.estimated_bpm - self.requested_bpm)

    @property
    def octave_equivalent(self) -> bool:
        """Whether estimate and request match at half or double time.

        A tempo estimator confusing 84 with 168 is the single most common
        failure mode; reporting it as a mismatch would be misleading.
        """
        if self.estimated_bpm is None or self.requested_bpm is None:
            return False
        for factor in (0.5, 2.0):
            if abs(self.estimated_bpm - self.requested_bpm * factor) <= 3.0:
                return True
        return False

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_bpm": self.requested_bpm,
            "estimated_bpm": round(self.estimated_bpm, 2) if self.estimated_bpm else None,
            "confidence": round(self.confidence, 3),
            "difference": round(self.difference, 2) if self.difference is not None else None,
            "octave_equivalent": self.octave_equivalent,
        }


def estimate_bpm(path: Path, *, requested_bpm: int | None = None) -> TempoEstimate:
    """Estimate tempo from an onset-strength envelope.

    Spectral flux gives an onset envelope; its autocorrelation peaks at
    the beat period. Validated against synthetic click tracks in
    ``test_longform.py`` — that is the only ground truth available here,
    so treat the number as a QA signal, not a measurement.
    """
    samples, rate = load_mono(path)
    if samples.size < rate:
        return TempoEstimate(None, 0.0, requested_bpm)

    hop = 512
    frame = 1024
    window = np.hanning(frame).astype(np.float32)
    starts = range(0, samples.size - frame + 1, hop)
    spectra = np.array(
        [np.abs(np.fft.rfft(samples[s : s + frame] * window)) for s in starts],
        dtype=np.float32,
    )
    if spectra.shape[0] < 4:
        return TempoEstimate(None, 0.0, requested_bpm)

    # Spectral flux: positive frame-to-frame energy increase.
    flux = np.sum(np.maximum(np.diff(spectra, axis=0), 0.0), axis=1)
    flux -= flux.mean()
    if not np.any(flux):
        return TempoEstimate(None, 0.0, requested_bpm)

    envelope_rate = rate / hop
    correlation = np.correlate(flux, flux, mode="full")[len(flux) - 1 :]
    if correlation[0] > 0:
        correlation = correlation / correlation[0]

    min_lag = max(1, int(envelope_rate * 60.0 / BPM_SEARCH_MAX))
    max_lag = min(len(correlation) - 1, int(envelope_rate * 60.0 / BPM_SEARCH_MIN))
    if max_lag <= min_lag:
        return TempoEstimate(None, 0.0, requested_bpm)

    search = correlation[min_lag : max_lag + 1]
    best = int(np.argmax(search))
    peak = float(search[best])
    lag = min_lag + best
    bpm = 60.0 * envelope_rate / lag

    # Confidence: how far the peak stands above the search-window mean.
    baseline = float(np.mean(np.abs(search)))
    confidence = (
        0.0 if baseline <= 0 else min(1.0, max(0.0, (peak - baseline) / (1.0 - baseline + 1e-9)))
    )
    return TempoEstimate(bpm, confidence, requested_bpm)


# ── Key estimation ────────────────────────────────────────────────────


@dataclass
class KeyEstimate:
    """An estimated key, with an explicit reliability verdict."""

    estimated_key: str | None
    confidence: float
    requested_key: str | None = None
    #: Set when the estimator declines to make a claim.
    verdict: str = "ESTIMATED"

    @property
    def tonic_matches(self) -> bool | None:
        """Whether the tonic agrees, ignoring mode.

        Mode is reported separately because this estimator is not
        trustworthy about major vs minor (see ``estimate_key``).
        """
        if not self.estimated_key or not self.requested_key:
            return None
        return _normalize_tonic(self.estimated_key) == _normalize_tonic(self.requested_key)

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_key": self.requested_key,
            "estimated_key": self.estimated_key,
            "confidence": round(self.confidence, 3),
            "tonic_matches": self.tonic_matches,
            "verdict": self.verdict,
        }


def _normalize_tonic(key: str) -> str:
    tonic = key.split()[0] if key else ""
    return _ENHARMONIC.get(tonic, tonic)


def estimate_key(path: Path, *, requested_key: str | None = None) -> KeyEstimate:
    """Estimate musical key by Krumhansl-Schmuckler correlation on chroma.

    **Reliability warning, deliberately encoded in the return value.**
    This is a chroma histogram from a plain FFT with no harmonic
    whitening, no tuning correction, and no temporal weighting. On
    synthetic material with unambiguous pitch content it recovers the
    tonic; on dense produced music its mode decision in particular is
    unreliable, because the major and minor KS profiles correlate
    strongly with each other.

    The honest output is therefore a *tonic* estimate with a confidence,
    and ``verdict="LOW_CONFIDENCE"`` when the top two candidates are too
    close to separate. Callers must not present a low-confidence result
    as verification. Full key verification remains
    ``HUMAN_OR_EXTERNAL_ANALYSIS_REQUIRED``.
    """
    samples, rate = load_mono(path)
    if samples.size < rate:
        return KeyEstimate(None, 0.0, requested_key, verdict="INSUFFICIENT_AUDIO")

    freqs, magnitude = _spectrum(samples, rate)
    usable = (freqs >= 55.0) & (freqs <= 2000.0)
    freqs, magnitude = freqs[usable], magnitude[usable]
    if not np.any(magnitude):
        return KeyEstimate(None, 0.0, requested_key, verdict="INSUFFICIENT_AUDIO")

    midi = 69 + 12 * np.log2(np.maximum(freqs, 1e-9) / 440.0)
    chroma = np.zeros(12, dtype=np.float64)
    np.add.at(chroma, np.rint(midi).astype(int) % 12, magnitude)
    if chroma.sum() <= 0:
        return KeyEstimate(None, 0.0, requested_key, verdict="INSUFFICIENT_AUDIO")
    chroma /= chroma.sum()

    scores: list[tuple[float, str]] = []
    for tonic in range(12):
        for profile, mode in ((_KS_MAJOR, "major"), (_KS_MINOR, "minor")):
            rotated = np.roll(profile, tonic)
            corr = float(np.corrcoef(chroma, rotated)[0, 1])
            if not math.isnan(corr):
                scores.append((corr, f"{_PITCH_CLASSES[tonic]} {mode}"))
    if not scores:
        return KeyEstimate(None, 0.0, requested_key, verdict="INSUFFICIENT_AUDIO")

    scores.sort(reverse=True)
    best_score, best_key = scores[0]
    margin = best_score - scores[1][0] if len(scores) > 1 else 0.0
    confidence = max(0.0, min(1.0, margin * 5.0))
    verdict = "ESTIMATED" if confidence >= 0.2 else "LOW_CONFIDENCE"
    return KeyEstimate(best_key, confidence, requested_key, verdict=verdict)


#: Time signature has no validated automatic method here. Reported as a
#: constant so callers cannot mistake silence for agreement.
TIME_SIGNATURE_VERDICT = "HUMAN_OR_EXTERNAL_ANALYSIS_REQUIRED"


def verify_controls(
    path: Path,
    *,
    requested_bpm: int | None = None,
    requested_key: str | None = None,
    requested_time_signature: str | None = None,
) -> dict[str, object]:
    """Everything measurable about whether the controls were honoured.

    Observability only: nothing here fails a generation, and a
    disagreement is a prompt for a human to listen, not a verdict.
    """
    return {
        "bpm": estimate_bpm(path, requested_bpm=requested_bpm).to_dict(),
        "key": estimate_key(path, requested_key=requested_key).to_dict(),
        "time_signature": {
            "requested": requested_time_signature,
            "estimated": None,
            "verdict": TIME_SIGNATURE_VERDICT,
        },
    }
