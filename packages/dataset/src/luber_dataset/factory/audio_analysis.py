"""Technical measurement, delegated to the engine that already does it.

`luber_audio_finishing.analyze_audio` is the project's trusted analysis:
it is what the finishing engine decides from, it is tested against
constructed spectra, and it handles every format through the same ffmpeg
fallback the rest of the pipeline uses. Re-implementing loudness or
stereo measurement here would produce a second set of numbers that
disagree with the first in ways nobody could adjudicate.

So this module maps rather than measures. The only things computed here
are the two the finishing engine has no reason to know about — spectral
rolloff and the high-frequency cutoff — and they come from the same
spectral pass the musical analysis already needs.

Anything that cannot be measured is ``None`` **with a reason**. A null
that means "this file has no bit depth because it is an MP3" and a null
that means "the loudness meter did not run" are different facts, and a
bare null makes them look identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from luber_audio_finishing import AudioLoadError, analyze_audio
from luber_audio_finishing.bands import AIR, BASS, BRILLIANCE, PRESENCE, SUB, ULTRA_HIGH
from luber_dataset.factory.decoder import DecodeResult


@dataclass
class TechnicalAnalysis:
    """Step 7's measurements, each nullable with an explanation."""

    duration_seconds: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    bit_depth: int | None = None
    peak_dbfs: float | None = None
    true_peak_dbtp: float | None = None
    integrated_lufs: float | None = None
    loudness_range_lu: float | None = None
    rms_dbfs: float | None = None
    crest_factor_db: float | None = None
    dc_offset: float | None = None
    silence_ratio: float | None = None
    clipping_sample_ratio: float | None = None
    spectral_centroid_hz: float | None = None
    spectral_rolloff_hz: float | None = None
    low_energy_ratio: float | None = None
    high_energy_ratio: float | None = None
    stereo_width: float | None = None
    phase_correlation: float | None = None
    dynamic_range_proxy_db: float | None = None
    #: Highest frequency carrying real energy. A 48 kHz file that stops
    #: at 16 kHz was transcoded up from something lossy.
    high_frequency_cutoff_hz: float | None = None

    #: metric name -> why it is null. Only populated for absences that
    #: are not simply "the analysis did not run".
    unavailable: dict[str, str] = field(default_factory=dict)
    analysis_error: str | None = None

    @property
    def measured(self) -> bool:
        return self.analysis_error is None and self.duration_seconds is not None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("unavailable", "analysis_error")
        }
        payload["unavailable"] = dict(sorted(self.unavailable.items()))
        payload["analysis_error"] = self.analysis_error
        return payload


def _finite(value: float | None) -> float | None:
    """NaN and infinity are not measurements."""
    if value is None:
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return round(number, 6)


def spectral_shape(mono: np.ndarray, sample_rate: int) -> tuple[float | None, float | None]:
    """Rolloff and high-frequency cutoff from one averaged spectrum.

    Rolloff is the frequency below which 85% of the energy sits — the
    conventional definition, and a compact description of how bright a
    track is. The cutoff is the highest frequency still carrying a
    thousandth of the peak bin's energy, which is what exposes a lossy
    source upsampled to a rate it has no content for.
    """
    if mono.size < 2048 or sample_rate <= 0:
        return None, None

    window_size = 4096
    hop = window_size // 2
    window = np.hanning(window_size)
    frames = 1 + max(0, (mono.size - window_size) // hop)
    if frames < 1:
        return None, None

    # Cap the number of frames analysed: a ten-minute file does not
    # need every window to characterise its spectral shape, and the
    # stride keeps the sampling spread across the whole track rather
    # than concentrated at the start.
    stride = max(1, frames // 512)
    accumulated = np.zeros(window_size // 2 + 1, dtype=np.float64)
    counted = 0
    for index in range(0, frames, stride):
        start = index * hop
        segment = mono[start : start + window_size]
        if segment.size < window_size:
            break
        spectrum = np.fft.rfft(segment * window)
        accumulated += np.abs(spectrum) ** 2
        counted += 1
    if counted == 0:
        return None, None
    accumulated /= counted

    freqs = np.fft.rfftfreq(window_size, 1.0 / sample_rate)
    total = float(accumulated.sum())
    if total <= 0:
        return None, None

    cumulative = np.cumsum(accumulated)
    rolloff_index = int(np.searchsorted(cumulative, 0.85 * total))
    rolloff = float(freqs[min(rolloff_index, freqs.size - 1)])

    # The cutoff is measured against nearby high-frequency content, not
    # against the global peak. A dense mix falls at roughly -4 dB per
    # octave, so by 16 kHz it sits ~60 dB below its low-frequency peak
    # purely from natural rolloff — against a peak-relative threshold
    # almost every real track reads as bandwidth-limited, which is a
    # detector that fires on everything and therefore says nothing.
    #
    # A codec lowpass is different in kind: it zeroes the spectrum, so
    # above its corner the level drops to the noise floor rather than
    # continuing a slope. Referencing 5 kHz — inside the range every
    # lossy encoder keeps — turns "is there a cliff" into the question
    # being asked.
    reference_band = (freqs >= 4_000.0) & (freqs <= 6_000.0)
    if not reference_band.any():
        return rolloff, None
    reference = float(accumulated[reference_band].mean())
    if reference <= 0:
        return rolloff, None
    significant = np.nonzero(accumulated >= reference * 1e-3)[0]
    cutoff = float(freqs[int(significant[-1])]) if significant.size else None
    return rolloff, cutoff


def analyse(
    path: Path,
    decode: DecodeResult,
    *,
    measure_loudness: bool = True,
) -> TechnicalAnalysis:
    """Measure one file, reusing the finishing engine's analysis."""
    result = TechnicalAnalysis()
    if not decode.usable:
        result.analysis_error = f"not analysed: decode status {decode.status.value}"
        return result

    try:
        analysis = analyze_audio(path, measure_r128=measure_loudness)
    except (AudioLoadError, OSError, ValueError) as exc:
        result.analysis_error = f"analysis failed: {exc}"
        return result

    technical, level, loudness = analysis.technical, analysis.level, analysis.loudness
    result.duration_seconds = _finite(technical.duration_seconds)
    result.sample_rate = technical.sample_rate
    result.channels = technical.channels
    result.bit_depth = technical.bit_depth
    if technical.bit_depth is None:
        result.unavailable["bit_depth"] = (
            f"{decode.codec or 'source'} is compressed or float; bit depth is not "
            "a property of the source"
        )

    result.peak_dbfs = _finite(level.peak_dbfs)
    result.rms_dbfs = _finite(level.rms_dbfs)
    result.crest_factor_db = _finite(level.crest_factor_db)
    result.dc_offset = _finite(level.dc_offset)
    result.silence_ratio = _finite(level.silence_ratio)
    result.clipping_sample_ratio = (
        _finite(level.clipped_samples / technical.frames) if technical.frames else None
    )

    result.integrated_lufs = _finite(loudness.integrated_lufs)
    result.loudness_range_lu = _finite(loudness.loudness_range_lu)
    result.true_peak_dbtp = _finite(loudness.true_peak_dbfs)
    if not measure_loudness:
        for name in ("integrated_lufs", "loudness_range_lu", "true_peak_dbtp"):
            result.unavailable[name] = "R128 measurement was disabled for this run"
    elif loudness.integrated_lufs is None:
        for name in ("integrated_lufs", "loudness_range_lu", "true_peak_dbtp"):
            result.unavailable[name] = "the R128 meter produced no reading for this file"

    # Dynamic range proxy: how far the loud passages sit above the quiet
    # ones. Loudness range is the real measurement; crest factor stands
    # in when R128 did not run, and the two are not interchangeable, so
    # which one was used is recorded.
    if loudness.loudness_range_lu is not None:
        result.dynamic_range_proxy_db = _finite(loudness.loudness_range_lu)
    else:
        result.dynamic_range_proxy_db = _finite(level.crest_factor_db)
        result.unavailable["dynamic_range_proxy_db"] = (
            "loudness range unavailable; crest factor used as a weaker proxy"
        )

    result.spectral_centroid_hz = _finite(analysis.frequency.spectral_centroid_hz.p50)

    shares = {band.name: band.share for band in analysis.frequency.bands if band.share is not None}
    low = sum(shares.get(name, 0.0) for name in (SUB, BASS))
    high = sum(shares.get(name, 0.0) for name in (PRESENCE, BRILLIANCE, AIR, ULTRA_HIGH))
    result.low_energy_ratio = _finite(low) if shares else None
    result.high_energy_ratio = _finite(high) if shares else None
    if analysis.absent_bands:
        result.unavailable["high_energy_ratio"] = (
            "bands above Nyquist for this sample rate are absent rather than empty: "
            + ", ".join(analysis.absent_bands)
        )

    stereo = analysis.stereo
    result.stereo_width = _finite(stereo.width)
    result.phase_correlation = _finite(stereo.correlation)
    if not stereo.is_stereo:
        result.unavailable["stereo_width"] = "the file is mono; width is not defined"
        result.unavailable["phase_correlation"] = (
            "the file is mono; inter-channel correlation is not defined"
        )
    return result
