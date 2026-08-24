"""Measured audio features, checked against signals whose answer is known.

Every test here builds a signal with a property somebody can state in a
sentence — a pure tone at 12 kHz is bright, a click train at 2 Hz has a
steady pulse — and asserts the measure reports it. That is the only way
to know a DSP function does what its name says rather than merely
returning a plausible number.
"""

import math

import numpy as np
import pytest

from luber_dataset.audio_features import (
    HIGH_BAND_HZ,
    AudioFeatureError,
    analyse_signal,
    onset_times,
    read_wav_mono,
)

RATE = 48_000


def _tone(hz: float, seconds: float = 4.0, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(RATE * seconds), dtype=np.float32) / RATE
    return (amplitude * np.sin(2 * math.pi * hz * t)).astype(np.float32)


def _clicks(per_second: float, seconds: float = 8.0, jitter: float = 0.0) -> np.ndarray:
    signal = np.zeros(int(RATE * seconds), dtype=np.float32)
    rng = np.random.default_rng(0)
    step = RATE / per_second
    position = RATE * 0.1
    while position < signal.size - 512:
        index = int(position)
        # A short decaying burst, so there is spectral flux to detect.
        burst = np.exp(-np.arange(512) / 60.0).astype(np.float32)
        noise = rng.standard_normal(512).astype(np.float32)
        signal[index : index + 512] += burst * noise * 0.8
        position += step * (1.0 + (rng.uniform(-jitter, jitter) if jitter else 0.0))
    return np.clip(signal, -1.0, 1.0)


class TestHighEnd:
    def test_a_bright_tone_reports_high_frequency_energy(self):
        bright = analyse_signal(_tone(12_000), RATE)
        assert bright.high_frequency_energy_ratio > 0.9

    def test_a_dull_tone_reports_almost_none(self):
        dull = analyse_signal(_tone(200), RATE)
        assert dull.high_frequency_energy_ratio < 0.01

    def test_the_centroid_follows_the_tone(self):
        low = analyse_signal(_tone(200), RATE).spectral_centroid_hz
        high = analyse_signal(_tone(12_000), RATE).spectral_centroid_hz
        assert low < 1_000 < high
        assert abs(high - 12_000) < 2_000

    def test_the_high_band_level_separates_bright_from_dull(self):
        bright = analyse_signal(_tone(12_000), RATE).high_band_rms_db
        dull = analyse_signal(_tone(200), RATE).high_band_rms_db
        assert bright > dull + 20

    def test_the_band_edge_is_where_it_says_it_is(self):
        below = analyse_signal(_tone(HIGH_BAND_HZ * 0.5), RATE)
        above = analyse_signal(_tone(HIGH_BAND_HZ * 1.5), RATE)
        assert below.high_frequency_energy_ratio < 0.05
        assert above.high_frequency_energy_ratio > 0.9


class TestRhythm:
    def test_a_steady_click_train_reports_its_tempo(self):
        features = analyse_signal(_clicks(2.0), RATE)
        assert features.tempo_bpm is not None
        # 2 clicks a second is 120 BPM; a harmonic of it is acceptable.
        assert any(abs(features.tempo_bpm - t) < 8 for t in (120.0, 60.0, 240.0))

    def test_onsets_are_counted_at_roughly_the_right_rate(self):
        features = analyse_signal(_clicks(2.0), RATE)
        assert 1.0 < features.onset_density_per_second < 4.0

    def test_a_steady_pulse_scores_higher_than_a_jittery_one(self):
        steady = analyse_signal(_clicks(2.0, jitter=0.0), RATE)
        loose = analyse_signal(_clicks(2.0, jitter=0.6), RATE)
        assert steady.beat_stability > loose.beat_stability

    def test_silence_has_no_tempo_and_no_pulse(self):
        features = analyse_signal(np.zeros(RATE * 4, dtype=np.float32), RATE)
        assert features.tempo_bpm is None
        assert features.beat_stability == 0.0
        assert features.onset_density_per_second == 0.0

    def test_tempo_consistency_needs_more_than_one_segment(self):
        short = analyse_signal(_clicks(2.0, seconds=5.0), RATE)
        assert short.tempo_consistency == 0.0


class TestArrangement:
    def test_broadband_noise_is_denser_than_a_single_tone(self):
        rng = np.random.default_rng(1)
        noise = (rng.standard_normal(RATE * 4) * 0.2).astype(np.float32)
        assert (
            analyse_signal(noise, RATE).layer_density
            > analyse_signal(_tone(1_000), RATE).layer_density
        )

    def test_two_tones_are_denser_than_one(self):
        one = analyse_signal(_tone(1_000), RATE).layer_density
        two = analyse_signal(_tone(1_000) + _tone(5_000), RATE).layer_density
        assert two > one

    def test_silence_reports_nothing_rather_than_a_number(self):
        features = analyse_signal(np.zeros(RATE * 2, dtype=np.float32), RATE)
        assert features.layer_density == 0.0
        assert features.active_band_fraction == 0.0


class TestBoundaries:
    def test_audio_shorter_than_one_frame_returns_empty_measures(self):
        features = analyse_signal(np.zeros(100, dtype=np.float32), RATE)
        assert features.rms_db == -120.0
        assert features.tempo_bpm is None

    def test_the_dictionary_carries_every_measure(self):
        payload = analyse_signal(_clicks(2.0), RATE).to_dict()
        for name in (
            "high_frequency_energy_ratio",
            "spectral_centroid_hz",
            "high_band_rms_db",
            "transient_density_per_second",
            "onset_density_per_second",
            "beat_stability",
            "tempo_consistency",
            "tempo_bpm",
            "drum_bass_alignment",
            "layer_density",
        ):
            assert name in payload, name

    def test_onset_times_are_seconds_inside_the_signal(self):
        seconds = 8.0
        times = onset_times(_clicks(2.0, seconds=seconds), RATE)
        assert times
        assert all(0.0 <= value <= seconds for value in times)
        assert list(times) == sorted(times)


class TestReading:
    def test_a_non_sixteen_bit_file_is_refused_rather_than_guessed_at(self, tmp_path):
        import wave

        path = tmp_path / "eight_bit.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(1)
            handle.setframerate(RATE)
            handle.writeframes(b"\x80" * 1000)
        with pytest.raises(AudioFeatureError, match="16-bit"):
            read_wav_mono(path)

    def test_stereo_is_mixed_to_mono(self, tmp_path):
        import wave

        path = tmp_path / "stereo.wav"
        left = (_tone(1_000, seconds=1.0) * 32767).astype("<i2")
        stereo = np.empty(left.size * 2, dtype="<i2")
        stereo[0::2] = left
        stereo[1::2] = left
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(RATE)
            handle.writeframes(stereo.tobytes())
        signal, rate, channels = read_wav_mono(path)
        assert channels == 2
        assert rate == RATE
        assert signal.size == left.size
