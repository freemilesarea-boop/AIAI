"""Objective measurement of a generated master.

Analysis exists before correction, and deliberately so: the human report
that drove this phase ("dull, flat, narrow") is a perception, and a
perception is not a specification. Nothing here decides what to change.
It only establishes what is measurably true about a file, so that a
correction can later be justified by a number and audited against it.

Three commitments shape the implementation:

*Nothing is invented.* Bands above Nyquist are absent, not empty. Mono
files have no stereo metrics rather than fabricated ones. Loudness is
``None`` when ffmpeg could not measure it.

*Time matters.* A single FFT over a four-minute song is dominated by
whichever section is longest. Every spectral figure is therefore also
reported as P10/P50/P90 across the track, so a bright chorus and a dull
intro stay distinguishable from a uniformly dull master.

*Cost is bounded.* Frames are consumed in blocks and reduced immediately;
the full spectrogram is never held in memory. Analysis of a four-minute
master stays in single-digit megabytes, because this has to run on the
same laptop that runs inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from luber_audio_finishing.audiofile import LoadedAudio, load_audio
from luber_audio_finishing.bands import BAND_EDGES, BandCoverage, band_coverage
from luber_audio_finishing.loudness import UNMEASURED, LoudnessMeasurement, measure_loudness

#: 4096 at 48 kHz gives 11.7 Hz bins and an 85 ms window. The low bands
#: are coarse at this size — SUB spans roughly three bins — which is
#: accepted because doubling the window to fix it would halve the time
#: resolution that the transient and sibilance measurements depend on.
FRAME_SIZE = 4096
HOP_SIZE = 2048
#: Frames reduced per iteration. Caps peak memory independent of length.
BLOCK_FRAMES = 128

#: Below this, content is subsonic and excluded from every ratio.
ANALYSIS_LOW_HZ = 20.0

#: Frames more than this far below the loudest frame are excluded from
#: every per-frame distribution. Without the gate, a fade-out and the
#: silence around an intro dominate the low percentiles and make every
#: track look as if its high frequencies vanish in places — an artefact
#: of measuring near-silence, not a property of the music.
ACTIVITY_GATE_DB = 40.0

#: Short windows for level statistics: long enough to have an RMS,
#: short enough that a snare is not averaged into the bar around it.
LEVEL_WINDOW_SECONDS = 0.05
SILENCE_THRESHOLD_DBFS = -60.0

CLIPPING_THRESHOLD = 0.999
NEAR_CLIPPING_THRESHOLD = 0.99

#: Reference band for every "relative to the body of the mix" ratio.
#:
#: Deliberately the MID band exactly, so it overlaps none of the bands
#: measured against it. An earlier 300 Hz-3 kHz reference shared 300-400
#: with the low-mid band and 2.5-3 kHz with the harshness band, and a
#: ratio whose numerator also sits in its denominator saturates: thick
#: 300-400 Hz content raised both sides and read as balanced, and a
#: harshness burst stopped registering past about 13 dB no matter how
#: loud it got.
BODY_LOW_HZ, BODY_HIGH_HZ = 400.0, 2_000.0
SIBILANCE_LOW_HZ, SIBILANCE_HIGH_HZ = 6_000.0, 9_000.0
HARSHNESS_LOW_HZ, HARSHNESS_HIGH_HZ = 2_500.0, 5_000.0
AIR_RATIO_LOW_HZ, AIR_RATIO_HIGH_HZ = 10_000.0, 16_000.0
LOW_MID_RATIO_LOW_HZ, LOW_MID_RATIO_HIGH_HZ = 150.0, 400.0

#: Slope is fitted above the fundamental region, where a mix's spectrum
#: is approximately a straight line on a log-log plot.
SLOPE_LOW_HZ, SLOPE_HIGH_HZ = 200.0, 16_000.0

#: Correlation and width in the region where mono compatibility and
#: playback-system behaviour actually matter.
LOW_STEREO_HZ = 120.0
HIGH_STEREO_HZ = 2_000.0

_EPS = 1e-20


def _db(value: float) -> float:
    """Power ratio to dB, floored rather than infinite for silence."""
    return 10.0 * float(np.log10(max(value, _EPS)))


def _amplitude_db(value: float) -> float:
    return 20.0 * float(np.log10(max(value, _EPS)))


def _percentiles(values: np.ndarray) -> tuple[float, float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    p10, p50, p90 = np.percentile(values, [10, 50, 90])
    return (float(p10), float(p50), float(p90))


@dataclass(frozen=True)
class Distribution:
    """A metric's spread over the track, not just its average."""

    p10: float
    p50: float
    p90: float
    mean: float

    @property
    def spread(self) -> float:
        """P90 - P10: how much the metric moves between sections."""
        return self.p90 - self.p10


def _distribution(values: np.ndarray) -> Distribution:
    p10, p50, p90 = _percentiles(values)
    mean = float(values.mean()) if values.size else float("nan")
    return Distribution(p10=p10, p50=p50, p90=p90, mean=mean)


@dataclass(frozen=True)
class TechnicalProperties:
    duration_seconds: float
    sample_rate: int
    channels: int
    bit_depth: int | None
    frames: int

    @property
    def nyquist_hz(self) -> float:
        return self.sample_rate / 2.0


@dataclass(frozen=True)
class LevelMetrics:
    """Amplitude-domain facts. These decide output safety."""

    peak_dbfs: float
    rms_dbfs: float
    crest_factor_db: float
    #: Largest absolute DC offset across channels, in sample units.
    dc_offset: float
    clipped_samples: int
    near_clipped_samples: int
    #: Fraction of short windows quieter than -60 dBFS.
    silence_ratio: float
    #: Crest factor measured per 50 ms window; flatness shows up here.
    short_window_crest_db: Distribution


@dataclass(frozen=True)
class BandMeasurement:
    name: str
    low_hz: float
    high_hz: float
    #: Absent when the band lies entirely above Nyquist.
    energy_db: float | None
    #: Share of total analysed energy, 0-1. ``None`` when absent.
    share: float | None
    truncated_by_nyquist: bool


@dataclass(frozen=True)
class FrequencyMetrics:
    bands: tuple[BandMeasurement, ...]
    spectral_centroid_hz: Distribution
    spectral_rolloff85_hz: Distribution
    spectral_bandwidth_hz: Distribution
    spectral_flatness: Distribution
    #: Fitted 200 Hz-16 kHz slope. More negative = darker.
    spectral_slope_db_per_octave: float
    #: Energy relative to the 300 Hz-3 kHz body of the mix, in dB.
    air_ratio_db: Distribution
    low_mid_ratio_db: Distribution
    presence_ratio_db: Distribution
    #: Where the 150-400 Hz energy actually concentrates. A mud
    #: correction aimed at a fixed 250 Hz would miss most songs; this is
    #: what makes the cut follow the track instead of a preset.
    low_mid_peak_hz: float

    def band(self, name: str) -> BandMeasurement | None:
        for measurement in self.bands:
            if measurement.name == name:
                return measurement
        return None


@dataclass(frozen=True)
class SibilanceMetrics:
    """Evidence about harsh events, not a verdict about them.

    6-9 kHz energy is sibilance, cymbals, string noise and synth texture
    all at once, so absolute level is weak evidence. What separates
    sibilance is that it *spikes*: the metric that matters is how far the
    loudest frames rise above the track's own typical level in that band.
    """

    sibilance_ratio_db: Distribution
    #: P90 - P50 of the ratio. High means the band spikes rather than sits.
    sibilance_peak_excess_db: float
    harshness_ratio_db: Distribution
    harshness_peak_excess_db: float


@dataclass(frozen=True)
class TransientMetrics:
    """Whether the signal moves, described without naming instruments."""

    spectral_flux: Distribution
    onset_rate_per_second: float
    #: Fraction of frames whose flux clears the track's own onset
    #: threshold. Low values mean a sustained, unarticulated master.
    transient_density: float


@dataclass(frozen=True)
class StereoMetrics:
    """Stereo facts, or explicit absence of them for mono sources."""

    is_stereo: bool
    lr_balance_db: float | None
    mid_energy_db: float | None
    side_energy_db: float | None
    #: Side minus mid, in dB. The width proxy: higher is wider.
    side_to_mid_db: float | None
    #: Normalised width, side / (mid + side), 0-1, measured **above**
    #: 120 Hz. Side energy below that is not image width — it is bass
    #: that will partially cancel in mono, and the engine's own
    #: correction for it is to remove that energy. Counting it here
    #: would make a track measure narrower after being repaired.
    width: float | None
    #: The same ratio over the whole spectrum, reported for completeness
    #: and used by nothing.
    full_band_width: float | None
    correlation: float | None
    low_band_correlation: float | None
    low_band_side_to_mid_db: float | None
    high_band_side_to_mid_db: float | None


@dataclass(frozen=True)
class SpatialProxies:
    """Named proxies, and named as proxies for a reason.

    Depth and space are not directly measurable from a stereo mixdown
    without assumptions this engine does not get to make. What is
    measurable is decorrelation between channels and how quickly energy
    decays after peaks. Both correlate loosely with perceived space and
    both are confounded — a wide synth pad and a reverberant room look
    alike here, as do a long note and a long tail.

    Nothing in the decision engine reads these fields. They are recorded
    so that a later phase can test them against listening results
    instead of assuming them.
    """

    stereo_decorrelation: float | None
    high_band_decorrelation: float | None
    #: Median post-peak decay of the energy envelope, dB per second.
    envelope_decay_db_per_second: float
    #: Always true in p14-v1: see the class docstring.
    drives_no_processing: bool = True


@dataclass(frozen=True)
class AudioAnalysis:
    """Everything measured about one file."""

    path: Path
    technical: TechnicalProperties
    level: LevelMetrics
    loudness: LoudnessMeasurement
    frequency: FrequencyMetrics
    sibilance: SibilanceMetrics
    transient: TransientMetrics
    stereo: StereoMetrics
    spatial: SpatialProxies
    #: Bands the sample rate cannot represent, for the report.
    absent_bands: tuple[str, ...] = field(default_factory=tuple)


def _bin_slice(freqs: np.ndarray, low: float, high: float) -> slice:
    start = int(np.searchsorted(freqs, low, side="left"))
    stop = int(np.searchsorted(freqs, high, side="right"))
    return slice(start, max(stop, start + 1))


def _band_ratio_db(power: np.ndarray, target: slice, reference: slice) -> np.ndarray:
    """Per-frame band ratio in dB, floored so silence cannot produce -inf."""
    numerator = power[:, target].sum(axis=1) + _EPS
    denominator = power[:, reference].sum(axis=1) + _EPS
    return np.asarray(10.0 * np.log10(numerator / denominator), dtype=np.float64)


class _SpectralAccumulator:
    """Reduces STFT blocks to fixed-size statistics as they are produced."""

    def __init__(self, sample_rate: int) -> None:
        self.freqs = np.fft.rfftfreq(FRAME_SIZE, 1.0 / sample_rate)
        self.mean_power = np.zeros(self.freqs.size, dtype=np.float64)
        self.cross_real = np.zeros(self.freqs.size, dtype=np.float64)
        self.power_left = np.zeros(self.freqs.size, dtype=np.float64)
        self.power_right = np.zeros(self.freqs.size, dtype=np.float64)
        self.frame_count = 0

        self._analysis = _bin_slice(self.freqs, ANALYSIS_LOW_HZ, sample_rate / 2.0)
        self._body = _bin_slice(self.freqs, BODY_LOW_HZ, BODY_HIGH_HZ)
        self._sibilance = _bin_slice(self.freqs, SIBILANCE_LOW_HZ, SIBILANCE_HIGH_HZ)
        self._harshness = _bin_slice(self.freqs, HARSHNESS_LOW_HZ, HARSHNESS_HIGH_HZ)
        self._air = _bin_slice(self.freqs, AIR_RATIO_LOW_HZ, AIR_RATIO_HIGH_HZ)
        self._low_mid = _bin_slice(self.freqs, LOW_MID_RATIO_LOW_HZ, LOW_MID_RATIO_HIGH_HZ)
        self._presence = _bin_slice(self.freqs, 2_000.0, 5_000.0)

        self.centroid: list[np.ndarray] = []
        self.rolloff: list[np.ndarray] = []
        self.bandwidth: list[np.ndarray] = []
        self.flatness: list[np.ndarray] = []
        self.flux: list[np.ndarray] = []
        self.air_ratio: list[np.ndarray] = []
        self.low_mid_ratio: list[np.ndarray] = []
        self.presence_ratio: list[np.ndarray] = []
        self.sibilance_ratio: list[np.ndarray] = []
        self.harshness_ratio: list[np.ndarray] = []
        self.frame_level_db: list[np.ndarray] = []
        self._previous_magnitude: np.ndarray | None = None

    def add(self, magnitude: np.ndarray, left: np.ndarray | None, right: np.ndarray | None) -> None:
        power = magnitude**2
        self.mean_power += power.sum(axis=0)
        self.frame_count += magnitude.shape[0]

        if left is not None and right is not None:
            self.power_left += (np.abs(left) ** 2).sum(axis=0)
            self.power_right += (np.abs(right) ** 2).sum(axis=0)
            self.cross_real += np.real(left * np.conj(right)).sum(axis=0)

        window = magnitude[:, self._analysis]
        freqs = self.freqs[self._analysis]
        total = window.sum(axis=1) + _EPS
        self.frame_level_db.append(10.0 * np.log10(power[:, self._analysis].sum(axis=1) + _EPS))
        centroid = (window * freqs[None, :]).sum(axis=1) / total
        self.centroid.append(centroid)

        cumulative = np.cumsum(window, axis=1) / total[:, None]
        self.rolloff.append(freqs[np.argmax(cumulative >= 0.85, axis=1)])

        deviation = freqs[None, :] - centroid[:, None]
        self.bandwidth.append(np.sqrt((window * deviation**2).sum(axis=1) / total))

        band_power = power[:, self._analysis] + _EPS
        geometric = np.exp(np.log(band_power).mean(axis=1))
        self.flatness.append(geometric / band_power.mean(axis=1))

        # Positive spectral flux, normalised by frame magnitude so it
        # describes change in spectral shape rather than change in level.
        # The very first frame has no predecessor and is recorded as zero
        # so that flux stays aligned with every other per-frame array —
        # they are masked together by the activity gate below.
        previous = self._previous_magnitude
        if previous is None:
            rise = np.maximum(magnitude[1:] - magnitude[:-1], 0.0).sum(axis=1)
            flux = rise / (magnitude[1:].sum(axis=1) + _EPS)
            self.flux.append(np.concatenate([np.zeros(1), flux]))
        else:
            reference = np.vstack([previous, magnitude[:-1]])
            rise = np.maximum(magnitude - reference, 0.0).sum(axis=1)
            self.flux.append(rise / (magnitude.sum(axis=1) + _EPS))
        self._previous_magnitude = magnitude[-1:]

        self.air_ratio.append(_band_ratio_db(power, self._air, self._body))
        self.low_mid_ratio.append(_band_ratio_db(power, self._low_mid, self._body))
        self.presence_ratio.append(_band_ratio_db(power, self._presence, self._body))
        self.sibilance_ratio.append(_band_ratio_db(power, self._sibilance, self._body))
        self.harshness_ratio.append(_band_ratio_db(power, self._harshness, self._body))

    def averaged_power(self) -> np.ndarray:
        if self.frame_count == 0:
            return self.mean_power
        return self.mean_power / self.frame_count


def _joined(chunks: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float64)


def _activity_mask(frame_level_db: np.ndarray) -> np.ndarray:
    """Which frames carry enough signal to describe the music.

    Falls back to keeping everything when the gate would leave nothing,
    so a uniformly quiet file still produces measurements rather than
    empty distributions.
    """
    if frame_level_db.size == 0:
        return np.ones(0, dtype=bool)
    mask = frame_level_db >= (float(frame_level_db.max()) - ACTIVITY_GATE_DB)
    return mask if bool(mask.any()) else np.ones(frame_level_db.size, dtype=bool)


def _block_bounds(signal: np.ndarray) -> list[tuple[int, int]]:
    """Block boundaries over the frame grid, not the frames themselves."""
    count = 1 + max(0, (signal.size - FRAME_SIZE) // HOP_SIZE)
    return [(start, min(start + BLOCK_FRAMES, count)) for start in range(0, count, BLOCK_FRAMES)]


def _stack_frames(signal: np.ndarray, first: int, last: int, window: np.ndarray) -> np.ndarray:
    offsets = np.arange(first, last) * HOP_SIZE
    index = offsets[:, None] + np.arange(FRAME_SIZE)[None, :]
    return signal[index] * window[None, :]


def _analyse_spectrum(audio: LoadedAudio) -> _SpectralAccumulator:
    mono = audio.mono()
    if mono.size < FRAME_SIZE:
        mono = np.pad(mono, (0, FRAME_SIZE - mono.size))
    window = np.hanning(FRAME_SIZE)
    accumulator = _SpectralAccumulator(audio.sample_rate)

    left: np.ndarray | None = None
    right: np.ndarray | None = None
    if audio.is_stereo:
        left = audio.samples[:, 0]
        right = audio.samples[:, 1]
        if left.size < FRAME_SIZE:
            left = np.pad(left, (0, FRAME_SIZE - left.size))
            right = np.pad(right, (0, FRAME_SIZE - right.size))

    for first, last in _block_bounds(mono):
        magnitude = np.abs(np.fft.rfft(_stack_frames(mono, first, last, window), axis=1))
        if left is None or right is None:
            accumulator.add(magnitude, None, None)
            continue
        spectrum_left = np.fft.rfft(_stack_frames(left, first, last, window), axis=1)
        spectrum_right = np.fft.rfft(_stack_frames(right, first, last, window), axis=1)
        accumulator.add(magnitude, spectrum_left, spectrum_right)
    return accumulator


def _band_measurements(
    averaged_power: np.ndarray, freqs: np.ndarray, coverage: tuple[BandCoverage, ...]
) -> tuple[tuple[BandMeasurement, ...], tuple[str, ...]]:
    # Shares are taken over the banded range only, so they sum to 1 even
    # at 48 kHz where content exists above the highest band's 20 kHz top.
    banded_top = min(BAND_EDGES[-1][2], float(freqs[-1]))
    analysed = averaged_power[_bin_slice(freqs, ANALYSIS_LOW_HZ, banded_top)].sum()
    measurements: list[BandMeasurement] = []
    absent: list[str] = []
    for band in coverage:
        if band.is_absent or band.usable_high_hz is None:
            absent.append(band.name)
            measurements.append(
                BandMeasurement(
                    name=band.name,
                    low_hz=band.low_hz,
                    high_hz=band.high_hz,
                    energy_db=None,
                    share=None,
                    truncated_by_nyquist=False,
                )
            )
            continue
        energy = float(averaged_power[_bin_slice(freqs, band.low_hz, band.usable_high_hz)].sum())
        measurements.append(
            BandMeasurement(
                name=band.name,
                low_hz=band.low_hz,
                high_hz=band.high_hz,
                energy_db=_db(energy),
                share=float(energy / analysed) if analysed > 0 else 0.0,
                truncated_by_nyquist=band.is_partial,
            )
        )
    return tuple(measurements), tuple(absent)


def _spectral_peak_hz(
    averaged_power: np.ndarray, freqs: np.ndarray, low: float, high: float
) -> float:
    """Frequency of greatest average energy inside a range.

    Smoothed over three bins so a single resonant partial does not decide
    where a broad, low-Q correction is centred.
    """
    window = _bin_slice(freqs, low, high)
    band = averaged_power[window]
    if band.size == 0:
        return float("nan")
    if band.size >= 3:
        band = np.convolve(band, np.ones(3) / 3.0, mode="same")
    return float(freqs[window][int(np.argmax(band))])


def _spectral_slope(averaged_power: np.ndarray, freqs: np.ndarray, nyquist: float) -> float:
    """dB per octave, fitted over the region a mix behaves smoothly in."""
    high = min(SLOPE_HIGH_HZ, nyquist * 0.9)
    if high <= SLOPE_LOW_HZ:
        return float("nan")
    window = _bin_slice(freqs, SLOPE_LOW_HZ, high)
    band_freqs = freqs[window]
    band_power = averaged_power[window]
    usable = band_freqs > 0
    if usable.sum() < 8:
        return float("nan")
    octaves = np.log2(band_freqs[usable])
    levels = 10.0 * np.log10(band_power[usable] + _EPS)
    slope, _ = np.polyfit(octaves, levels, 1)
    return float(slope)


def _level_metrics(audio: LoadedAudio) -> LevelMetrics:
    samples = audio.samples
    peak = float(np.abs(samples).max()) if samples.size else 0.0
    rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
    dc = float(np.abs(samples.mean(axis=0)).max()) if samples.size else 0.0

    window = max(1, int(LEVEL_WINDOW_SECONDS * audio.sample_rate))
    mono = audio.mono()
    usable = (mono.size // window) * window
    if usable >= window:
        blocks = mono[:usable].reshape(-1, window)
        block_rms = np.sqrt((blocks**2).mean(axis=1))
        block_peak = np.abs(blocks).max(axis=1)
        loud = block_rms > 10 ** (SILENCE_THRESHOLD_DBFS / 20.0)
        silence_ratio = float(1.0 - loud.mean())
        crest = 20.0 * np.log10((block_peak[loud] + _EPS) / (block_rms[loud] + _EPS))
    else:
        silence_ratio = 0.0
        crest = np.zeros(0, dtype=np.float64)

    return LevelMetrics(
        peak_dbfs=_amplitude_db(peak),
        rms_dbfs=_amplitude_db(rms),
        crest_factor_db=_amplitude_db(peak) - _amplitude_db(rms),
        dc_offset=dc,
        clipped_samples=int((np.abs(samples) >= CLIPPING_THRESHOLD).sum()),
        near_clipped_samples=int((np.abs(samples) >= NEAR_CLIPPING_THRESHOLD).sum()),
        silence_ratio=silence_ratio,
        short_window_crest_db=_distribution(crest),
    )


def _stereo_metrics(
    audio: LoadedAudio, accumulator: _SpectralAccumulator
) -> tuple[StereoMetrics, SpatialProxies]:
    envelope_decay = _envelope_decay(audio)
    if not audio.is_stereo:
        # Truthful absence. A mono file has no balance, no width and no
        # correlation, and reporting 1.0 for correlation would read as
        # "perfectly mono-compatible stereo", which is a different claim.
        return (
            StereoMetrics(
                is_stereo=False,
                lr_balance_db=None,
                mid_energy_db=None,
                side_energy_db=None,
                side_to_mid_db=None,
                width=None,
                full_band_width=None,
                correlation=None,
                low_band_correlation=None,
                low_band_side_to_mid_db=None,
                high_band_side_to_mid_db=None,
            ),
            SpatialProxies(
                stereo_decorrelation=None,
                high_band_decorrelation=None,
                envelope_decay_db_per_second=envelope_decay,
            ),
        )

    freqs = accumulator.freqs
    full = _bin_slice(freqs, ANALYSIS_LOW_HZ, float(freqs[-1]))
    low = _bin_slice(freqs, ANALYSIS_LOW_HZ, LOW_STEREO_HZ)
    high = _bin_slice(freqs, HIGH_STEREO_HZ, float(freqs[-1]))

    def correlation(window: slice) -> float:
        left = float(accumulator.power_left[window].sum())
        right = float(accumulator.power_right[window].sum())
        cross = float(accumulator.cross_real[window].sum())
        denominator = float(np.sqrt(left * right))
        return cross / denominator if denominator > _EPS else 0.0

    def mid_side(window: slice) -> tuple[float, float]:
        # |L+R|^2/4 and |L-R|^2/4 expand into the same three sums, so
        # mid/side energy comes free from the correlation accumulators.
        left = float(accumulator.power_left[window].sum())
        right = float(accumulator.power_right[window].sum())
        cross = float(accumulator.cross_real[window].sum())
        return ((left + right + 2 * cross) / 4.0, (left + right - 2 * cross) / 4.0)

    mid, side = mid_side(full)
    low_mid, low_side = mid_side(low)
    high_mid, high_side = mid_side(high)
    # The image band: everything the width correction can legitimately act on.
    image = _bin_slice(freqs, LOW_STEREO_HZ, float(freqs[-1]))
    image_mid, image_side = mid_side(image)

    left_rms = float(np.sqrt(np.mean(audio.samples[:, 0] ** 2)))
    right_rms = float(np.sqrt(np.mean(audio.samples[:, 1] ** 2)))

    stereo = StereoMetrics(
        is_stereo=True,
        lr_balance_db=_amplitude_db(left_rms) - _amplitude_db(right_rms),
        mid_energy_db=_db(mid),
        side_energy_db=_db(side),
        side_to_mid_db=_db(side) - _db(mid),
        width=(
            float(image_side / (image_mid + image_side)) if (image_mid + image_side) > _EPS else 0.0
        ),
        full_band_width=float(side / (mid + side)) if (mid + side) > _EPS else 0.0,
        correlation=correlation(full),
        low_band_correlation=correlation(low),
        low_band_side_to_mid_db=_db(low_side) - _db(low_mid),
        high_band_side_to_mid_db=_db(high_side) - _db(high_mid),
    )
    spatial = SpatialProxies(
        stereo_decorrelation=1.0 - abs(correlation(full)),
        high_band_decorrelation=1.0 - abs(correlation(high)),
        envelope_decay_db_per_second=envelope_decay,
    )
    return stereo, spatial


def _envelope_decay(audio: LoadedAudio) -> float:
    """Median decay rate after energy peaks, in dB per second.

    A proxy for how wet or sustained a mix is, and a weak one: a held
    pad and a reverb tail decay alike. Recorded, never acted on.
    """
    window = max(1, int(0.02 * audio.sample_rate))
    mono = audio.mono()
    usable = (mono.size // window) * window
    if usable < window * 8:
        return float("nan")
    blocks = mono[:usable].reshape(-1, window)
    level = 20.0 * np.log10(np.sqrt((blocks**2).mean(axis=1)) + _EPS)
    step = window / audio.sample_rate
    # Peaks are local maxima; decay is measured over the following 100 ms.
    lookahead = max(1, int(0.1 / step))
    if level.size <= lookahead + 2:
        return float("nan")
    interior = level[1:-1]
    is_peak = (interior >= level[:-2]) & (interior > level[2:])
    indices = np.flatnonzero(is_peak) + 1
    # Only peaks with a full lookahead window after them can be measured.
    indices = indices[indices + lookahead < level.size]
    if indices.size == 0:
        return float("nan")
    drops = level[indices] - level[indices + lookahead]
    return float(np.median(drops) / (lookahead * step))


def _transient_metrics(flux: np.ndarray, duration: float) -> TransientMetrics:
    if flux.size == 0 or duration <= 0:
        return TransientMetrics(
            spectral_flux=_distribution(flux),
            onset_rate_per_second=float("nan"),
            transient_density=float("nan"),
        )
    # The threshold is relative to the track's own flux distribution, so
    # a sparse ballad and a dense mix are each judged against themselves
    # rather than against an absolute number that would suit neither.
    quiet, loud = np.percentile(flux, [60, 95])
    threshold = quiet + 0.5 * (loud - quiet)
    above = flux > threshold
    peaks = above[1:-1] & (flux[1:-1] >= flux[:-2]) & (flux[1:-1] > flux[2:])
    return TransientMetrics(
        spectral_flux=_distribution(flux),
        onset_rate_per_second=float(peaks.sum() / duration),
        transient_density=float(above.mean()),
    )


def analyze_audio(path: Path, *, measure_r128: bool = True) -> AudioAnalysis:
    """Measure a file end to end.

    ``measure_r128`` exists so batch analysis can skip the one step that
    costs a subprocess per file; skipping it yields ``None`` loudness
    rather than an estimate.
    """
    audio = load_audio(path)
    accumulator = _analyse_spectrum(audio)
    averaged = accumulator.averaged_power()
    freqs = accumulator.freqs
    coverage = band_coverage(audio.sample_rate)
    bands, absent = _band_measurements(averaged, freqs, coverage)

    active = _activity_mask(_joined(accumulator.frame_level_db))

    def join(chunks: list[np.ndarray]) -> np.ndarray:
        values = _joined(chunks)
        return values[active] if values.size == active.size else values

    frequency = FrequencyMetrics(
        bands=bands,
        spectral_centroid_hz=_distribution(join(accumulator.centroid)),
        spectral_rolloff85_hz=_distribution(join(accumulator.rolloff)),
        spectral_bandwidth_hz=_distribution(join(accumulator.bandwidth)),
        spectral_flatness=_distribution(join(accumulator.flatness)),
        spectral_slope_db_per_octave=_spectral_slope(averaged, freqs, audio.sample_rate / 2.0),
        air_ratio_db=_distribution(join(accumulator.air_ratio)),
        low_mid_ratio_db=_distribution(join(accumulator.low_mid_ratio)),
        presence_ratio_db=_distribution(join(accumulator.presence_ratio)),
        low_mid_peak_hz=_spectral_peak_hz(
            averaged, freqs, LOW_MID_RATIO_LOW_HZ, LOW_MID_RATIO_HIGH_HZ
        ),
    )

    sibilance_ratio = _distribution(join(accumulator.sibilance_ratio))
    harshness_ratio = _distribution(join(accumulator.harshness_ratio))
    sibilance = SibilanceMetrics(
        sibilance_ratio_db=sibilance_ratio,
        sibilance_peak_excess_db=sibilance_ratio.p90 - sibilance_ratio.p50,
        harshness_ratio_db=harshness_ratio,
        harshness_peak_excess_db=harshness_ratio.p90 - harshness_ratio.p50,
    )

    stereo, spatial = _stereo_metrics(audio, accumulator)
    return AudioAnalysis(
        path=path,
        technical=TechnicalProperties(
            duration_seconds=audio.duration_seconds,
            sample_rate=audio.sample_rate,
            channels=audio.channels,
            bit_depth=audio.bit_depth,
            frames=audio.frames,
        ),
        level=_level_metrics(audio),
        loudness=measure_loudness(path) if measure_r128 else UNMEASURED,
        frequency=frequency,
        sibilance=sibilance,
        transient=_transient_metrics(join(accumulator.flux), audio.duration_seconds),
        stereo=stereo,
        spatial=spatial,
        absent_bands=absent,
    )
