"""Synthetic audio with known defects.

Testing a corrective engine on real music is circular: the corpus is
what the thresholds were derived from, so it would confirm whatever was
built. These fixtures instead construct signals whose spectrum, stereo
image and dynamics are set on purpose, so a test can say what the
analyser is supposed to report and be wrong when it does not.

Signals are shaped in the frequency domain. That makes "a spectrum
falling at 8 dB per octave" an input rather than something approximated
with filters and hoped for.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

RATE = 48_000
#: Long enough for stable percentiles and R128 gating, short enough that
#: a full test run stays a few seconds.
SECONDS = 6.0

#: Pink noise falls at -3 dB/octave; -4 is a plausible healthy mix.
HEALTHY_SLOPE = -4.0

#: Trim applied above 10 kHz to make the baseline spectrally *neutral*
#: rather than merely deficit-free.
#:
#: Broadband noise at -4 dB/octave carries far more top end than music
#: does: it measures an air ratio of -9.2 dB, brighter than any of the 57
#: real masters, whose maximum is -12.9. That was invisible while the
#: engine only detected darkness, and the original constant was chosen
#: "comfortably clear of every deficit threshold" — clear on one side
#: only. With brightness now detected too, an untrimmed baseline is an
#: over-bright one, and every fixture built on it would carry a defect it
#: was not meant to have.
#:
#: -8 dB above 10 kHz lands the baseline at a slope of -6.04 dB/octave
#: and an air ratio of -17.2 dB. That clears the dark rule's slope
#: condition (-6.5) and the bright rule's air condition (-16.0)
#: *separately*, so the fixture does not depend on one half of an AND to
#: stay healthy. Trimming further would widen the air margin but push the
#: slope past -6.5, which is the wrong trade: it would leave the baseline
#: one condition away from reading as dark.
#:
#: Applied to the whole stereo helper rather than to the healthy fixture
#: alone, so a fixture built to be muddy or sibilant is muddy or sibilant
#: and nothing else.
NEUTRAL_TOP_TRIM = (10_000.0, 24_000.0, -8.0)


def _spectrum_shape(freqs: np.ndarray, slope_db_per_octave: float) -> np.ndarray:
    """Amplitude per bin for a given log-log slope."""
    shape = np.zeros_like(freqs)
    usable = freqs > 0
    # slope dB/octave -> power exponent: 10*log10(2**a) = 3.0103*a.
    exponent = slope_db_per_octave / 3.0103
    shape[usable] = freqs[usable] ** (exponent / 2.0)
    return shape


def _apply_band_gains(
    freqs: np.ndarray, shape: np.ndarray, band_gains: tuple[tuple[float, float, float], ...]
) -> np.ndarray:
    adjusted = shape.copy()
    for low, high, gain_db in band_gains:
        window = (freqs >= low) & (freqs < high)
        adjusted[window] *= 10.0 ** (gain_db / 20.0)
    return adjusted


def shaped_noise(
    *,
    seconds: float = SECONDS,
    rate: int = RATE,
    slope_db_per_octave: float = HEALTHY_SLOPE,
    band_gains: tuple[tuple[float, float, float], ...] = (),
    highpass_hz: float = 0.0,
    lowpass_hz: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Mono noise with an exactly specified spectral shape."""
    length = int(seconds * rate)
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.standard_normal(length))
    freqs = np.fft.rfftfreq(length, 1.0 / rate)
    shape = _apply_band_gains(freqs, _spectrum_shape(freqs, slope_db_per_octave), band_gains)

    # The scale comes from the *unfiltered* signal, and the band limits
    # are applied afterwards. Two normalisations that look equivalent are
    # not: rescaling after the high-pass would give a band-limited
    # component the same energy as a full-range one, so on a steep
    # spectrum the stereo helper's decorrelated part would tower over the
    # correlated part above 200 Hz, and a fixture meant to be dull would
    # measure as extremely wide instead.
    full_band = np.fft.irfft(spectrum * shape, n=length)
    rms = float(np.sqrt(np.mean(full_band**2)))
    if highpass_hz > 0:
        shape[freqs < highpass_hz] = 0.0
    if lowpass_hz > 0:
        shape[freqs > lowpass_hz] = 0.0
    signal = (
        full_band if not (highpass_hz or lowpass_hz) else np.fft.irfft(spectrum * shape, n=length)
    )
    return signal / rms if rms > 0 else signal


def stereo(
    *,
    decorrelation: float = 1.0,
    decorrelation_highpass_hz: float = 200.0,
    balance_db: float = 0.0,
    amplitude: float = 0.5,
    seed: int = 0,
    seconds: float = SECONDS,
    rate: int = RATE,
    slope_db_per_octave: float = HEALTHY_SLOPE,
    band_gains: tuple[tuple[float, float, float], ...] = (),
    neutral_top: bool = True,
) -> np.ndarray:
    """A stereo pair with a controlled image.

    The uncorrelated part is high-passed so the bass stays coherent. Real
    mixes behave that way, and a fixture that did not would trip the
    low-end phase rule on every test that meant to be about something
    else.
    """
    # The neutral trim goes first so an explicit band_gain can override
    # it: a fixture that means to be bright says so and wins.
    gains = (NEUTRAL_TOP_TRIM, *band_gains) if neutral_top else band_gains
    shape = {
        "seconds": seconds,
        "rate": rate,
        "slope_db_per_octave": slope_db_per_octave,
        "band_gains": gains,
    }
    common = shaped_noise(seed=seed, **shape)
    left_only = shaped_noise(seed=seed + 101, highpass_hz=decorrelation_highpass_hz, **shape)
    right_only = shaped_noise(seed=seed + 202, highpass_hz=decorrelation_highpass_hz, **shape)
    left = common + decorrelation * left_only
    right = common + decorrelation * right_only
    left *= 10.0 ** (balance_db / 2.0 / 20.0)
    right *= 10.0 ** (-balance_db / 2.0 / 20.0)
    pair = np.stack([left, right], axis=1)
    peak = float(np.abs(pair).max())
    return pair * (amplitude / peak) if peak > 0 else pair


def add_bursts(
    samples: np.ndarray,
    *,
    low_hz: float,
    high_hz: float,
    gain_db: float,
    rate: int = RATE,
    period_seconds: float = 0.5,
    length_seconds: float = 0.06,
    seed: int = 7,
) -> np.ndarray:
    """Add short band-limited bursts.

    Sibilance and harshness are defined here by *spiking*, not by level,
    so a fixture for them has to vary over time. Raising a band steadily
    would produce a bright track, which is a different thing and must not
    trip the same flag.
    """
    out = samples.copy()
    burst_length = int(length_seconds * rate)
    source = shaped_noise(
        seconds=length_seconds,
        rate=rate,
        slope_db_per_octave=0.0,
        highpass_hz=low_hz,
        lowpass_hz=high_hz,
        seed=seed,
    )[:burst_length]
    envelope = np.hanning(burst_length)
    scale = 10.0 ** (gain_db / 20.0) * float(np.abs(samples).max())
    burst = source * envelope * scale
    if samples.ndim > 1:
        # A different noise seed per channel, at identical gain. Scaling
        # one channel instead would decorrelate the bursts by making the
        # fixture lopsided, and it would then trip the balance rule as
        # well as the sibilance one.
        channels = [
            shaped_noise(
                seconds=length_seconds,
                rate=rate,
                slope_db_per_octave=0.0,
                highpass_hz=low_hz,
                lowpass_hz=high_hz,
                seed=seed + 11 * index,
            )[:burst_length]
            * envelope
            * scale
            for index in range(samples.shape[1])
        ]
        burst = np.stack(channels, axis=1)
    step = int(period_seconds * rate)
    for start in range(step, out.shape[0] - burst_length, step):
        out[start : start + burst_length] += burst
    peak = float(np.abs(out).max())
    limit = float(np.abs(samples).max())
    return out * (limit / peak) if peak > limit else out


def write_wav(path: Path, samples: np.ndarray, *, rate: int = RATE, bit_depth: int = 24) -> Path:
    """Write float samples as PCM WAV, clipping rather than wrapping."""
    if samples.ndim == 1:
        samples = samples[:, None]
    clipped = np.clip(samples, -1.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(samples.shape[1])
        handle.setsampwidth(bit_depth // 8)
        handle.setframerate(rate)
        if bit_depth == 16:
            handle.writeframes((clipped * 32767.0).astype("<i2").tobytes())
        elif bit_depth == 24:
            as_int = (clipped * 8388607.0).astype(np.int32)
            packed = as_int.astype("<i4").tobytes()
            triples = bytearray()
            for offset in range(0, len(packed), 4):
                triples += packed[offset : offset + 3]
            handle.writeframes(bytes(triples))
        else:
            raise ValueError(f"unsupported bit depth for fixtures: {bit_depth}")
    return path


@pytest.fixture
def healthy_stereo() -> np.ndarray:
    """A mix with nothing wrong with it. Must produce NO_ACTION."""
    return stereo()


@pytest.fixture
def healthy_path(tmp_path: Path, healthy_stereo: np.ndarray) -> Path:
    return write_wav(tmp_path / "healthy.wav", healthy_stereo)


@pytest.fixture
def dull_stereo() -> np.ndarray:
    """A healthy low end with the top rolled off: dark, and dark only.

    Steepening the overall slope instead would also concentrate energy in
    the bass and flatten the waveform, so the fixture would trip the
    low-end and transient rules too and stop being a test about darkness.
    """
    return stereo(band_gains=((4_000.0, 24_000.0, -24.0),))


@pytest.fixture
def muddy_stereo() -> np.ndarray:
    return stereo(band_gains=((150.0, 400.0, 9.0),))


@pytest.fixture
def sibilant_stereo() -> np.ndarray:
    return add_bursts(stereo(), low_hz=6_000.0, high_hz=9_000.0, gain_db=6.0)


@pytest.fixture
def narrow_stereo() -> np.ndarray:
    return stereo(decorrelation=0.02)
