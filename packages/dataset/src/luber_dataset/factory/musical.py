"""Tempo and key, measured — and structure, which is not.

The project has no librosa, no essentia and no madmom, so this is
written against numpy directly. That constrains what can honestly be
claimed, and the split is deliberate:

*Tempo and key are computed.* Both are classical signal processing with
well-defined algorithms — onset autocorrelation and Krumhansl-Schmuckler
profile correlation — and both are verified in the tests against
synthetic signals built at a known tempo and in a known key. An
estimator that can recover a value it was never told is doing real work.

*Structure is not.* Segmenting a song into verse and chorus needs either
a trained model or a hand-built heuristic that would be wrong often
enough to poison the labels it produced. There is no honest way to
produce it here, so ``estimated_structure`` is null and
``structure_status`` says ``UNAVAILABLE``. A fabricated section label is
worse than a missing one: missing data is visibly missing, and invented
data trains a model on a lie.

Every estimate carries a confidence, and the pipeline is free to discard
low-confidence values rather than record a guess as a fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

#: Tempo search range. Below 50 and above 210 the autocorrelation peak
#: is almost always a harmonic of the real pulse rather than the pulse.
MIN_BPM = 50.0
MAX_BPM = 210.0

#: Centre and width of the log-normal tempo prior that resolves octave
#: ambiguity. 120 BPM is the classic centre of human tempo perception;
#: the width is wide enough that a genuine 70 or 170 BPM track still
#: wins its own lag, and narrow enough to break a tie between a pulse
#: and half that pulse.
TEMPO_PRIOR_CENTRE_BPM = 120.0
TEMPO_PRIOR_WIDTH = 1.0

#: Below this the estimate is reported but marked unreliable, and the
#: pipeline does not treat it as known.
MIN_TEMPO_CONFIDENCE = 0.15
MIN_KEY_CONFIDENCE = 0.10

PITCH_CLASSES: tuple[str, ...] = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)

#: Krumhansl-Kessler probe-tone profiles: how strongly each pitch class
#: is perceived as belonging to a key. Published values, used unchanged.
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


class StructureStatus(StrEnum):
    UNAVAILABLE = "UNAVAILABLE"
    #: Reserved for a future analyser. Never produced today.
    ESTIMATED = "ESTIMATED"
    USER_SUPPLIED = "USER_SUPPLIED"


@dataclass
class MusicalAnalysis:
    bpm: float | None = None
    bpm_confidence: float | None = None
    key: str | None = None
    key_confidence: float | None = None
    mode: str | None = None
    estimated_downbeat_seconds: float | None = None
    estimated_structure: list[dict[str, Any]] | None = None
    structure_status: str = StructureStatus.UNAVAILABLE.value
    unavailable: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bpm": self.bpm,
            "bpm_confidence": self.bpm_confidence,
            "key": self.key,
            "key_confidence": self.key_confidence,
            "mode": self.mode,
            "estimated_downbeat_seconds": self.estimated_downbeat_seconds,
            "estimated_structure": self.estimated_structure,
            "structure_status": self.structure_status,
            "unavailable": dict(sorted(self.unavailable.items())),
        }


def onset_envelope(mono: np.ndarray, sample_rate: int, hop: int = 512) -> np.ndarray:
    """Positive spectral flux: where energy arrives.

    Half-wave rectified, because only *increases* in energy are onsets.
    Including decreases would make every note ending look like a note
    beginning and halve the apparent tempo.
    """
    window_size = 2048
    if mono.size < window_size * 2:
        return np.zeros(0, dtype=np.float64)

    window = np.hanning(window_size)
    frames = 1 + (mono.size - window_size) // hop
    magnitudes = np.empty((frames, window_size // 2 + 1), dtype=np.float64)
    for index in range(frames):
        start = index * hop
        magnitudes[index] = np.abs(np.fft.rfft(mono[start : start + window_size] * window))

    # Log compression before differencing, so a loud section does not
    # dominate the envelope purely by being loud.
    compressed = np.log1p(magnitudes)
    flux = np.diff(compressed, axis=0)
    envelope = np.maximum(flux, 0.0).sum(axis=1)
    if envelope.size == 0:
        return envelope
    envelope -= envelope.mean()
    return envelope


def estimate_tempo(
    mono: np.ndarray, sample_rate: int, *, hop: int = 512
) -> tuple[float | None, float | None]:
    """Tempo by autocorrelating the onset envelope.

    Returns ``(bpm, confidence)``. Confidence is how far the winning lag
    stands above the mean of the plausible lags — a track with a steady
    pulse produces one clear peak, and a rubato or ambient one produces
    a flat curve that should not be reported as a tempo.
    """
    envelope = onset_envelope(mono, sample_rate, hop=hop)
    if envelope.size < 16:
        return None, None

    correlation = np.correlate(envelope, envelope, mode="full")[envelope.size - 1 :]
    if correlation.size == 0 or correlation[0] <= 0:
        return None, None
    correlation = correlation / correlation[0]

    frames_per_second = sample_rate / hop
    min_lag = max(1, round(frames_per_second * 60.0 / MAX_BPM))
    max_lag = round(frames_per_second * 60.0 / MIN_BPM)
    max_lag = min(max_lag, correlation.size - 1)
    if max_lag <= min_lag:
        return None, None

    window = correlation[min_lag : max_lag + 1]

    # Autocorrelation cannot distinguish a pulse from half that pulse:
    # a 120 BPM track correlates just as well at the 60 BPM lag, and
    # unweighted this estimator returned 60.09 for 120 and 69.84 for 140.
    # That is a *wrong* answer rather than a missing one, which is worse.
    #
    # So the curve is weighted by a log-normal prior over tempo before
    # the peak is chosen. It does not invent a pulse — the peak still has
    # to be there — it decides which octave of a real pulse to report,
    # using the fact that human tempo perception centres near 120 BPM and
    # falls off symmetrically in log space.
    lags = np.arange(min_lag, max_lag + 1, dtype=np.float64)
    candidate_bpm = 60.0 * frames_per_second / lags
    prior = np.exp(
        -0.5 * (np.log2(candidate_bpm / TEMPO_PRIOR_CENTRE_BPM) / TEMPO_PRIOR_WIDTH) ** 2
    )
    weighted = window * prior

    best = int(np.argmax(weighted))

    # Lags are whole frames, and a tempo rarely is. At this hop a 120 BPM
    # pulse falls at lag 21.5, so the nearest integer lags report 117.5 or
    # 123 — a quantisation error, not a measurement. Fitting a parabola
    # through the peak and its two neighbours recovers the fractional lag
    # and with it the tempo between them.
    lag = float(min_lag + best)
    if 0 < best < weighted.size - 1:
        left, centre, right = (
            float(weighted[best - 1]),
            float(weighted[best]),
            float(weighted[best + 1]),
        )
        denominator = left - 2.0 * centre + right
        if abs(denominator) > 1e-12:
            offset = 0.5 * (left - right) / denominator
            # A vertex further than one bin away means the parabola did
            # not fit; trust the integer lag rather than an extrapolation.
            if abs(offset) <= 1.0:
                lag += offset
    bpm = 60.0 * frames_per_second / lag

    baseline = float(np.mean(weighted))
    spread = float(np.std(weighted))
    if spread <= 1e-9:
        return round(bpm, 2), 0.0
    # Peak height above the local mean, in standard deviations, squashed
    # into [0, 1]. A tempo is "confident" when its lag stands out, not
    # when its raw correlation is high.
    confidence = float(np.tanh(max(0.0, float(weighted[best]) - baseline) / (2.0 * spread)))
    return round(bpm, 2), round(confidence, 4)


def chroma(mono: np.ndarray, sample_rate: int) -> np.ndarray | None:
    """Energy per pitch class, summed over the track.

    Bins are mapped to pitch classes by frequency, restricted to 55 Hz -
    2 kHz: below that the bin spacing is too coarse to separate
    semitones, and above it harmonics dominate the fundamental.
    """
    window_size = 8192
    if mono.size < window_size * 2 or sample_rate <= 0:
        return None

    hop = window_size // 2
    window = np.hanning(window_size)
    freqs = np.fft.rfftfreq(window_size, 1.0 / sample_rate)
    usable = (freqs >= 55.0) & (freqs <= 2_000.0)
    if not usable.any():
        return None

    # MIDI note number, then pitch class. 69 = A4 = 440 Hz.
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = 69.0 + 12.0 * np.log2(np.where(freqs > 0, freqs, 1.0) / 440.0)
    classes = np.mod(np.rint(midi).astype(int), 12)

    frames = 1 + (mono.size - window_size) // hop
    stride = max(1, frames // 256)
    result = np.zeros(12, dtype=np.float64)
    counted = 0
    for index in range(0, frames, stride):
        start = index * hop
        segment = mono[start : start + window_size]
        if segment.size < window_size:
            break
        power = np.abs(np.fft.rfft(segment * window)) ** 2
        np.add.at(result, classes[usable], power[usable])
        counted += 1
    if counted == 0 or result.sum() <= 0:
        return None
    return result / result.sum()


def estimate_key(mono: np.ndarray, sample_rate: int) -> tuple[str | None, str | None, float | None]:
    """Key and mode by correlating chroma against the K-K profiles.

    Returns ``(key, mode, confidence)``. Confidence is the margin
    between the best-fitting key and the runner-up: a track that fits
    two keys almost equally well has not established one, and the
    difference between C major and A minor is exactly that kind of tie.
    """
    profile = chroma(mono, sample_rate)
    if profile is None:
        return None, None, None

    centred = profile - profile.mean()
    if float(np.linalg.norm(centred)) <= 1e-12:
        return None, None, None

    scores: list[tuple[float, str, str]] = []
    for tonic in range(12):
        for name, reference in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            rotated = np.roll(reference, tonic)
            rotated = rotated - rotated.mean()
            denominator = float(np.linalg.norm(centred) * np.linalg.norm(rotated))
            if denominator <= 1e-12:
                continue
            scores.append(
                (float(np.dot(centred, rotated) / denominator), PITCH_CLASSES[tonic], name)
            )
    if not scores:
        return None, None, None

    scores.sort(key=lambda item: item[0], reverse=True)
    best_score, best_key, best_mode = scores[0]
    runner_up = scores[1][0] if len(scores) > 1 else 0.0
    if best_score <= 0.0:
        return None, None, 0.0
    confidence = max(0.0, min(1.0, best_score - runner_up))
    return best_key, best_mode, round(confidence, 4)


def analyse(mono: np.ndarray, sample_rate: int) -> MusicalAnalysis:
    """Tempo and key where they can be established; never structure."""
    result = MusicalAnalysis()

    bpm, bpm_confidence = estimate_tempo(mono, sample_rate)
    if bpm is not None and bpm_confidence is not None and bpm_confidence >= MIN_TEMPO_CONFIDENCE:
        result.bpm = bpm
        result.bpm_confidence = bpm_confidence
    else:
        result.unavailable["bpm"] = (
            "no pulse stood out from the onset autocorrelation"
            if bpm is None
            else f"tempo estimate too weak to record (confidence {bpm_confidence})"
        )

    key, mode, key_confidence = estimate_key(mono, sample_rate)
    if key is not None and key_confidence is not None and key_confidence >= MIN_KEY_CONFIDENCE:
        result.key = key
        result.mode = mode
        result.key_confidence = key_confidence
    else:
        result.unavailable["key"] = (
            "chroma could not be computed for this file"
            if key is None
            else f"two keys fit almost equally well (margin {key_confidence})"
        )

    # Downbeat requires beat tracking with phase, not just a period.
    # The tempo estimator recovers how often beats occur and says
    # nothing about where they land.
    result.unavailable["estimated_downbeat_seconds"] = (
        "tempo estimation recovers the beat period but not its phase; no beat tracker is available"
    )
    result.structure_status = StructureStatus.UNAVAILABLE.value
    result.unavailable["estimated_structure"] = (
        "no trained structural segmenter is available, and a heuristic would "
        "produce labels wrong often enough to poison what it labelled"
    )
    return result
