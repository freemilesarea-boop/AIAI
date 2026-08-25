"""What a recording actually sounds like, measured rather than assumed.

Phase 38 needs to know which authorised tracks have live high end, a
steady pulse and a busy arrangement, and *where inside them* those
qualities sit. Nothing in the library carries that information: the
operator supplied folders named `POP` and `Lofi` and nothing else.

Everything here is computed from the audio with numpy and the standard
library's `wave` module. No scipy, no librosa, no new dependency — the
source is 16-bit PCM at 48 kHz, which is the easy case, and a short STFT
gives every measure below.

The measures are deliberately plain and their limits are stated with
them. "Onset density" is spectral-flux peaks per second, not a
transcription. The "drum/bass alignment" figure is a correlation between
two band-limited onset envelopes, which is a *proxy* and named one.
Nothing here identifies an instrument, and no value should be read as if
it did.

All features are computed per analysis window as well as per track, so a
window chooser can prefer the busy half of a song over its intro.
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: STFT geometry. 2048 samples at 48 kHz is ~43 ms, long enough to
#: resolve bass and short enough to keep transients from smearing; the
#: 512 hop gives ~94 frames a second, which is ample for onset work.
FFT_SIZE = 2048
HOP_SIZE = 512

#: Where "high end" begins for this analysis. Chosen, not derived: 8 kHz
#: is above the fundamental range of everything in a mix and squarely in
#: the region a dull master loses first.
HIGH_BAND_HZ = 8_000.0

#: The band the Phase 38 listening evaluation actually complained about.
#: Energy there was never the problem — texture was, so it is measured
#: separately and over its own range rather than folded into the 8 kHz
#: figure above.
HIGH_TEXTURE_LOW_HZ = 6_000.0
HIGH_TEXTURE_HIGH_HZ = 16_000.0

#: How far above the local spectral floor a bin must stand to count as a
#: narrow resonance rather than as part of the broadband texture. 6 dB is
#: a chosen threshold; reference material sits well under it and the
#: generated audio that prompted this sits well over.
RESONANCE_EXCESS_DB = 6.0
#: Width of the running median that estimates that floor, in bins. Odd so
#: the window is centred; ~31 bins is ~730 Hz at this FFT size, wide
#: enough to ignore a single resonance and narrow enough to follow the
#: real spectral shape.
RESONANCE_FLOOR_BINS = 31
#: Bass band for the rhythmic-alignment proxy.
BASS_BAND_HZ = 200.0
#: Percussive band for the same proxy — cymbals and snare snap.
PERCUSSIVE_BAND_HZ = 4_000.0

#: Tempo search range, in BPM. Outside this an autocorrelation peak is
#: far more likely to be a bar or a subdivision than a pulse.
MIN_BPM = 60.0
MAX_BPM = 180.0

#: Segment length for tempo-consistency, in seconds. Long enough to hold
#: several bars at any tempo in range.
TEMPO_SEGMENT_SECONDS = 15.0


class AudioFeatureError(RuntimeError):
    """Raised when a file cannot be analysed as authored."""


@dataclass(frozen=True)
class AudioFeatures:
    """Measured properties of one stretch of audio.

    Every field is a measurement of *this* audio. None of them is a
    judgement, and the tiering that consumes them lives elsewhere so the
    two never get confused.
    """

    duration_seconds: float
    sample_rate: int
    channels: int

    #: Share of spectral energy above :data:`HIGH_BAND_HZ`.
    high_frequency_energy_ratio: float
    #: Energy-weighted mean frequency, in Hz.
    spectral_centroid_hz: float
    #: RMS of the high band alone, in dBFS.
    high_band_rms_db: float
    #: Broadband RMS, in dBFS. Context for the figure above.
    rms_db: float
    #: Sharp amplitude rises per second.
    transient_density_per_second: float
    #: Spectral-flux onset peaks per second.
    onset_density_per_second: float
    #: How concentrated inter-onset intervals are around one period.
    #: 0 is formless, 1 is a metronome.
    beat_stability: float
    #: Agreement between per-segment tempo estimates. 1 is unvarying.
    tempo_consistency: float
    #: Estimated pulse, in BPM. ``None`` when no periodicity was found.
    tempo_bpm: float | None
    #: Correlation of the bass and percussive onset envelopes. A proxy
    #: for a locked rhythm section, and named one.
    drum_bass_alignment: float
    #: Spectral entropy across bands, normalised. Higher means energy is
    #: spread across more of the spectrum at once — a proxy for how many
    #: layers are sounding, not a count of instruments.
    layer_density: float
    #: Share of frames whose spectrum is broadly occupied.
    active_band_fraction: float

    #: Spectral flatness of the 6-16 kHz band: geometric mean over
    #: arithmetic mean, averaged across frames. 1.0 is noise-like — which
    #: is what air *is*. Low values mean the band is carried by tones.
    #: Independent of level: a quiet band and a loud band with the same
    #: shape score the same.
    high_band_flatness: float = 0.0
    #: Share of 6-16 kHz bins standing more than :data:`RESONANCE_EXCESS_DB`
    #: above the local spectral floor. Broadband air gives ~0; a handful
    #: of ringing partials gives a measurable fraction.
    high_band_resonance_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_seconds": round(self.duration_seconds, 3),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "high_frequency_energy_ratio": round(self.high_frequency_energy_ratio, 6),
            "spectral_centroid_hz": round(self.spectral_centroid_hz, 2),
            "high_band_rms_db": round(self.high_band_rms_db, 2),
            "rms_db": round(self.rms_db, 2),
            "transient_density_per_second": round(self.transient_density_per_second, 4),
            "onset_density_per_second": round(self.onset_density_per_second, 4),
            "beat_stability": round(self.beat_stability, 4),
            "tempo_consistency": round(self.tempo_consistency, 4),
            "tempo_bpm": None if self.tempo_bpm is None else round(self.tempo_bpm, 2),
            "drum_bass_alignment": round(self.drum_bass_alignment, 4),
            "layer_density": round(self.layer_density, 4),
            "active_band_fraction": round(self.active_band_fraction, 4),
            "high_band_flatness": round(self.high_band_flatness, 4),
            "high_band_resonance_ratio": round(self.high_band_resonance_ratio, 4),
        }


@dataclass(frozen=True)
class TrackAnalysis:
    """One track's features, plus the onset grid a window chooser needs."""

    track_id: str
    audio_sha256: str
    source_group: str
    features: AudioFeatures
    #: Onset times in seconds, for beat-aware window placement.
    onset_times: tuple[float, ...] = ()
    #: Per-window features, keyed by window start in seconds.
    window_features: dict[float, AudioFeatures] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "audio_sha256": self.audio_sha256,
            "source_group": self.source_group,
            "features": self.features.to_dict(),
            "onset_count": len(self.onset_times),
            "window_features": {
                str(start): value.to_dict() for start, value in sorted(self.window_features.items())
            },
        }


# ── reading ──────────────────────────────────────────────────────────


def read_wav_mono(path: Path) -> tuple[np.ndarray, int, int]:
    """A WAV as mono float32 in [-1, 1], with its rate and channel count.

    Uses the standard library because the authorised source is 16-bit
    PCM and adding a decoder dependency to read it would be ceremony.
    Anything else is refused rather than guessed at.
    """
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.getnframes()
        if width != 2:
            raise AudioFeatureError(
                f"{path.name}: {width * 8}-bit samples; this reader handles 16-bit PCM"
            )
        raw = handle.readframes(frames)

    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        usable = (data.size // channels) * channels
        data = data[:usable].reshape(-1, channels).mean(axis=1)
    return data, rate, channels


def _stft_magnitude(signal: np.ndarray) -> np.ndarray:
    """Magnitude spectrogram, frames along axis 0."""
    if signal.size < FFT_SIZE:
        return np.zeros((0, FFT_SIZE // 2 + 1), dtype=np.float32)
    count = 1 + (signal.size - FFT_SIZE) // HOP_SIZE
    window = np.hanning(FFT_SIZE).astype(np.float32)
    # A strided view rather than a copy: a four-minute track is 11
    # million samples and materialising every frame would be gigabytes.
    frames = np.lib.stride_tricks.as_strided(
        signal,
        shape=(count, FFT_SIZE),
        strides=(signal.strides[0] * HOP_SIZE, signal.strides[0]),
        writeable=False,
    )
    return np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32)


def _db(value: float) -> float:
    return 20.0 * math.log10(value) if value > 1e-12 else -120.0


def _onset_envelope(magnitude: np.ndarray) -> np.ndarray:
    """Spectral flux: the positive change in each bin, summed."""
    if magnitude.shape[0] < 2:
        return np.zeros(0, dtype=np.float32)
    flux = np.diff(magnitude, axis=0)
    return np.maximum(flux, 0.0).sum(axis=1)


def _pick_peaks(envelope: np.ndarray, *, sensitivity: float = 1.5) -> np.ndarray:
    """Indices where the envelope rises well above its local level.

    A local mean rather than a global threshold, so a quiet intro and a
    loud chorus are judged on their own terms instead of the chorus
    swamping the whole track.
    """
    if envelope.size < 3:
        return np.zeros(0, dtype=np.int64)
    span = 21
    padded = np.pad(envelope, span // 2, mode="edge")
    kernel = np.ones(span, dtype=np.float32) / span
    local = np.convolve(padded, kernel, mode="valid")[: envelope.size]
    threshold = local * sensitivity + 1e-6
    above = envelope > threshold
    # Only the rising edge of each excursion counts, or one hit would be
    # reported as several.
    edges = above & ~np.concatenate(([False], above[:-1]))
    return np.flatnonzero(edges)


def _autocorrelation_tempo(envelope: np.ndarray, rate: float) -> tuple[float | None, float]:
    """Pulse period from the onset envelope, and how sharp the peak is."""
    if envelope.size < 16:
        return None, 0.0
    centred = envelope - envelope.mean()
    if not np.any(centred):
        return None, 0.0
    correlation = np.correlate(centred, centred, mode="full")[centred.size - 1 :]
    if correlation[0] <= 0:
        return None, 0.0
    correlation = correlation / correlation[0]

    low = max(1, round(rate * 60.0 / MAX_BPM))
    high = min(correlation.size - 1, round(rate * 60.0 / MIN_BPM))
    if high <= low:
        return None, 0.0
    band = correlation[low : high + 1]
    index = int(np.argmax(band)) + low
    period = index / rate
    if period <= 0:
        return None, 0.0
    # Sharpness: how far the winning lag stands above the band's mean.
    peak = float(band.max())
    sharpness = max(0.0, min(1.0, peak - float(band.mean())))
    return 60.0 / period, sharpness


def _beat_stability(onset_frames: np.ndarray, rate: float) -> float:
    """How regular the gaps between onsets are.

    The coefficient of variation of inter-onset intervals, inverted and
    clamped. A metronome gives 1; a free-time performance gives near 0.
    """
    if onset_frames.size < 4:
        return 0.0
    intervals = np.diff(onset_frames) / rate
    intervals = intervals[intervals > 0]
    if intervals.size < 3:
        return 0.0
    mean = float(intervals.mean())
    if mean <= 0:
        return 0.0
    variation = float(intervals.std()) / mean
    return max(0.0, min(1.0, 1.0 - variation))


def _tempo_consistency(envelope: np.ndarray, rate: float) -> float:
    """Agreement between tempo estimates taken across the track.

    One tempo for the whole song says nothing about whether it drifted.
    Several estimates that agree do.
    """
    per_segment = int(TEMPO_SEGMENT_SECONDS * rate)
    if per_segment < 16 or envelope.size < per_segment * 2:
        return 0.0
    tempos: list[float] = []
    for start in range(0, envelope.size - per_segment + 1, per_segment):
        tempo, _ = _autocorrelation_tempo(envelope[start : start + per_segment], rate)
        if tempo is not None:
            tempos.append(tempo)
    if len(tempos) < 2:
        return 0.0
    values = np.array(tempos, dtype=np.float64)
    mean = float(values.mean())
    if mean <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - float(values.std()) / mean))


def _band_envelope(magnitude: np.ndarray, freqs: np.ndarray, low: float, high: float) -> np.ndarray:
    mask = (freqs >= low) & (freqs < high)
    if not mask.any() or magnitude.shape[0] < 2:
        return np.zeros(0, dtype=np.float32)
    band = magnitude[:, mask]
    return np.maximum(np.diff(band, axis=0), 0.0).sum(axis=1)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    size = min(left.size, right.size)
    if size < 4:
        return 0.0
    a, b = left[:size], right[:size]
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0:
        return 0.0
    return max(-1.0, min(1.0, float(np.dot(a, b)) / denominator))


def _layer_density(magnitude: np.ndarray) -> tuple[float, float]:
    """Spectral entropy and how often the spectrum is broadly occupied.

    A proxy for arrangement density: a solo voice concentrates energy in
    a few bands, a full mix spreads it. It counts nothing and identifies
    nothing, which is why it is not called an instrument count.
    """
    if magnitude.shape[0] == 0:
        return 0.0, 0.0
    power = magnitude.astype(np.float64) ** 2
    totals = power.sum(axis=1, keepdims=True)
    usable = totals[:, 0] > 1e-9
    if not usable.any():
        return 0.0, 0.0
    distribution = power[usable] / totals[usable]
    entropy = -(distribution * np.log(distribution + 1e-12)).sum(axis=1)
    normalised = entropy / math.log(distribution.shape[1])
    # "Broadly occupied" means at least a quarter of bands carry a
    # non-trivial share of that frame's energy.
    share = distribution.shape[1] * distribution
    occupied = (share > 0.25).sum(axis=1) / distribution.shape[1]
    return float(normalised.mean()), float((occupied > 0.25).mean())


def _rolling_median(values: np.ndarray, size: int) -> np.ndarray:
    """Running median of a 1-D array, edges held rather than shrunk.

    scipy has this; adding scipy to read 16-bit PCM would not be worth
    one filter, and the arrays here are a few hundred bins wide.
    """
    if values.size == 0:
        return values
    half = max(1, size // 2)
    if values.size <= 2 * half:
        return np.full_like(values, float(np.median(values)))
    padded = np.pad(values, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    return np.median(windows, axis=1)[: values.size]


def _high_band_texture(magnitude: np.ndarray, freqs: np.ndarray) -> tuple[float, float]:
    """Flatness and narrow-resonance share of the 6-16 kHz band.

    Two numbers about *texture*, deliberately independent of level. The
    Phase 38 listening evaluation reported a high band that was both flat
    and metallic, and no energy measure can tell those apart: a band can
    be loud and tonal, or quiet and airy. Flatness says which.

    Neither value judges anything. Reference material happens to sit near
    0.92 flatness with almost no narrow peaks; that is an observation
    about this library, not a target defined here.
    """
    band = (freqs >= HIGH_TEXTURE_LOW_HZ) & (freqs < HIGH_TEXTURE_HIGH_HZ)
    if magnitude.shape[0] == 0 or band.sum() < 8:
        return 0.0, 0.0

    power = magnitude[:, band].astype(np.float64) ** 2
    # Flatness per frame, then averaged over the frames that carry any
    # energy at all — silence has no texture and averaging it in would
    # drag every quiet passage toward zero.
    arithmetic = power.mean(axis=1)
    usable = arithmetic > 1e-18
    if not usable.any():
        return 0.0, 0.0
    geometric = np.exp(np.log(power[usable] + 1e-18).mean(axis=1))
    flatness = float(np.clip(geometric / arithmetic[usable], 0.0, 1.0).mean())

    # Resonances are read off the time-averaged spectrum: a partial that
    # rings through the whole window is the thing being counted, and a
    # single frame's noise is not.
    average = power.mean(axis=0)
    decibels = 10.0 * np.log10(average + 1e-18)
    excess = decibels - _rolling_median(decibels, RESONANCE_FLOOR_BINS)
    resonance = float((excess > RESONANCE_EXCESS_DB).mean())
    return flatness, resonance


def analyse_signal(signal: np.ndarray, rate: int, *, channels: int = 1) -> AudioFeatures:
    """Every feature, from one stretch of mono audio."""
    magnitude = _stft_magnitude(np.ascontiguousarray(signal))
    freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / rate)
    rms = float(np.sqrt(np.mean(signal.astype(np.float64) ** 2))) if signal.size else 0.0
    duration = signal.size / rate if rate else 0.0
    return _features_from_magnitude(
        magnitude, freqs, rate, rms=rms, duration=duration, channels=channels
    )


def _features_from_magnitude(
    magnitude: np.ndarray,
    freqs: np.ndarray,
    rate: int,
    *,
    rms: float,
    duration: float,
    channels: int,
) -> AudioFeatures:
    """Everything derivable from one spectrogram.

    Split out from :func:`analyse_signal` so a caller analysing many
    windows of one track computes the transform once and slices it.
    A four-minute track has fifteen candidate windows; recomputing the
    STFT for each would do the same work fifteen times.
    """
    frame_rate = rate / HOP_SIZE

    if magnitude.shape[0] == 0:
        return AudioFeatures(
            duration_seconds=duration,
            sample_rate=rate,
            channels=channels,
            high_frequency_energy_ratio=0.0,
            spectral_centroid_hz=0.0,
            high_band_rms_db=-120.0,
            rms_db=-120.0,
            transient_density_per_second=0.0,
            onset_density_per_second=0.0,
            beat_stability=0.0,
            tempo_consistency=0.0,
            tempo_bpm=None,
            drum_bass_alignment=0.0,
            layer_density=0.0,
            active_band_fraction=0.0,
        )

    power = magnitude.astype(np.float64) ** 2
    total_energy = float(power.sum()) + 1e-12
    high_mask = freqs >= HIGH_BAND_HZ
    high_ratio = float(power[:, high_mask].sum()) / total_energy

    per_frame = power.sum(axis=1) + 1e-12
    centroid = float(((power * freqs).sum(axis=1) / per_frame).mean())

    # High-band level reconstructed from its spectral share, which keeps
    # the whole analysis in one domain.
    high_rms = rms * math.sqrt(max(high_ratio, 0.0))

    envelope = _onset_envelope(magnitude)
    onsets = _pick_peaks(envelope)
    onset_density = onsets.size / duration if duration > 0 else 0.0

    # Transients are the sharper subset: a rise well above the local
    # level rather than merely above it.
    transients = _pick_peaks(envelope, sensitivity=2.5)
    transient_density = transients.size / duration if duration > 0 else 0.0

    tempo, sharpness = _autocorrelation_tempo(envelope, frame_rate)
    stability = max(_beat_stability(onsets, frame_rate), 0.0) * 0.5 + sharpness * 0.5
    consistency = _tempo_consistency(envelope, frame_rate)

    bass = _band_envelope(magnitude, freqs, 0.0, BASS_BAND_HZ)
    percussive = _band_envelope(magnitude, freqs, PERCUSSIVE_BAND_HZ, float(freqs[-1]) + 1.0)
    alignment = max(0.0, _correlation(bass, percussive))

    density, active = _layer_density(magnitude)
    flatness, resonance = _high_band_texture(magnitude, freqs)

    return AudioFeatures(
        duration_seconds=duration,
        sample_rate=rate,
        channels=channels,
        high_frequency_energy_ratio=high_ratio,
        spectral_centroid_hz=centroid,
        high_band_rms_db=_db(high_rms),
        rms_db=_db(rms),
        transient_density_per_second=transient_density,
        onset_density_per_second=onset_density,
        beat_stability=max(0.0, min(1.0, stability)),
        tempo_consistency=consistency,
        tempo_bpm=tempo,
        drum_bass_alignment=alignment,
        layer_density=density,
        active_band_fraction=active,
        high_band_flatness=flatness,
        high_band_resonance_ratio=resonance,
    )


def onset_times(signal: np.ndarray, rate: int) -> tuple[float, ...]:
    """Onset positions in seconds, for beat-aware window placement."""
    magnitude = _stft_magnitude(np.ascontiguousarray(signal))
    if magnitude.shape[0] == 0:
        return ()
    frames = _pick_peaks(_onset_envelope(magnitude))
    return tuple(float(index) * HOP_SIZE / rate for index in frames)


def analyse_track(
    path: Path,
    *,
    track_id: str,
    audio_sha256: str,
    source_group: str = "",
    window_seconds: float | None = None,
    window_starts: tuple[float, ...] = (),
) -> TrackAnalysis:
    """Analyse a whole track, and optionally named windows inside it."""
    signal, rate, channels = read_wav_mono(path)
    signal = np.ascontiguousarray(signal)
    magnitude = _stft_magnitude(signal)
    freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / rate)
    whole_rms = float(np.sqrt(np.mean(signal.astype(np.float64) ** 2))) if signal.size else 0.0
    features = _features_from_magnitude(
        magnitude,
        freqs,
        rate,
        rms=whole_rms,
        duration=signal.size / rate if rate else 0.0,
        channels=channels,
    )

    windows: dict[float, AudioFeatures] = {}
    if window_seconds:
        length = int(window_seconds * rate)
        frames_per_window = max(1, (length - FFT_SIZE) // HOP_SIZE + 1)
        for start in window_starts:
            begin = int(start * rate)
            chunk = signal[begin : begin + length]
            if chunk.size < length:
                continue
            first = begin // HOP_SIZE
            slice_ = magnitude[first : first + frames_per_window]
            if slice_.shape[0] == 0:
                continue
            windows[float(start)] = _features_from_magnitude(
                slice_,
                freqs,
                rate,
                rms=float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2))),
                duration=window_seconds,
                channels=channels,
            )

    return TrackAnalysis(
        track_id=track_id,
        audio_sha256=audio_sha256,
        source_group=source_group,
        features=features,
        onset_times=onset_times(signal, rate),
        window_features=windows,
    )


__all__ = [
    "BASS_BAND_HZ",
    "FFT_SIZE",
    "HIGH_BAND_HZ",
    "HIGH_TEXTURE_HIGH_HZ",
    "HIGH_TEXTURE_LOW_HZ",
    "HOP_SIZE",
    "MAX_BPM",
    "MIN_BPM",
    "PERCUSSIVE_BAND_HZ",
    "RESONANCE_EXCESS_DB",
    "RESONANCE_FLOOR_BINS",
    "AudioFeatureError",
    "AudioFeatures",
    "TrackAnalysis",
    "analyse_signal",
    "analyse_track",
    "onset_times",
    "read_wav_mono",
]
