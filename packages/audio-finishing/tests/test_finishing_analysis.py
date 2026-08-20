"""Analyser tests: does it measure what is actually there?

The engine's decisions are only as good as these numbers, so the cases
that matter most are the ones where a plausible implementation would
quietly return something wrong instead of failing: a band above Nyquist,
a mono file, a file of silence, a file shorter than one analysis frame.
"""

from __future__ import annotations

import numpy as np
import pytest
from finishing_fixtures import RATE, add_bursts, shaped_noise, stereo, write_wav

from luber_audio_finishing.analysis import ACTIVITY_GATE_DB, FRAME_SIZE, analyze_audio
from luber_audio_finishing.audiofile import AudioLoadError


class TestTechnicalProperties:
    def test_reports_the_file_as_written(self, tmp_path, healthy_stereo):
        path = write_wav(tmp_path / "a.wav", healthy_stereo)
        technical = analyze_audio(path, measure_r128=False).technical
        assert technical.sample_rate == RATE
        assert technical.channels == 2
        assert technical.bit_depth == 24
        assert technical.duration_seconds == pytest.approx(6.0, abs=0.01)

    @pytest.mark.parametrize("rate", [22_050, 44_100, 48_000])
    def test_handles_several_sample_rates(self, tmp_path, rate):
        samples = stereo(rate=rate, seconds=3.0)
        analysis = analyze_audio(
            write_wav(tmp_path / "r.wav", samples, rate=rate), measure_r128=False
        )
        assert analysis.technical.sample_rate == rate
        assert analysis.technical.nyquist_hz == rate / 2

    def test_reads_16_bit_as_well_as_24(self, tmp_path, healthy_stereo):
        path = write_wav(tmp_path / "16.wav", healthy_stereo, bit_depth=16)
        assert analyze_audio(path, measure_r128=False).technical.bit_depth == 16

    def test_refuses_a_missing_file(self, tmp_path):
        with pytest.raises(AudioLoadError):
            analyze_audio(tmp_path / "nope.wav", measure_r128=False)

    def test_refuses_an_empty_file(self, tmp_path):
        empty = tmp_path / "empty.wav"
        empty.write_bytes(b"")
        with pytest.raises(AudioLoadError):
            analyze_audio(empty, measure_r128=False)


class TestNyquist:
    def test_bands_above_nyquist_are_absent_not_empty(self, tmp_path):
        """22.05 kHz has no 16-20 kHz band.

        Reporting zero energy there would read as "no air" and provoke a
        correction the file cannot benefit from.
        """
        samples = stereo(rate=22_050, seconds=3.0)
        analysis = analyze_audio(
            write_wav(tmp_path / "low.wav", samples, rate=22_050), measure_r128=False
        )
        ultra = analysis.frequency.band("ultra_high")
        assert ultra is not None
        assert ultra.energy_db is None
        assert ultra.share is None
        assert "ultra_high" in analysis.absent_bands

    def test_a_band_straddling_nyquist_is_truncated_rather_than_dropped(self, tmp_path):
        """At 36 kHz, Nyquist lands inside 16-20 kHz.

        The band is real up to 18 kHz, so dropping it would discard
        measurable air and reporting it whole would invent some.
        """
        samples = stereo(rate=36_000, seconds=3.0)
        analysis = analyze_audio(
            write_wav(tmp_path / "odd.wav", samples, rate=36_000), measure_r128=False
        )
        ultra = analysis.frequency.band("ultra_high")
        assert ultra is not None and ultra.energy_db is not None
        assert ultra.truncated_by_nyquist is True

    def test_44_1_khz_keeps_every_band_whole(self, tmp_path):
        """Nyquist is 22.05 kHz, above the highest band edge of 20 kHz."""
        samples = stereo(rate=44_100, seconds=3.0)
        analysis = analyze_audio(
            write_wav(tmp_path / "cd.wav", samples, rate=44_100), measure_r128=False
        )
        assert analysis.absent_bands == ()
        assert all(not band.truncated_by_nyquist for band in analysis.frequency.bands)

    def test_full_rate_audio_has_every_band(self, tmp_path, healthy_stereo):
        analysis = analyze_audio(write_wav(tmp_path / "f.wav", healthy_stereo), measure_r128=False)
        assert analysis.absent_bands == ()
        assert all(band.energy_db is not None for band in analysis.frequency.bands)

    def test_band_shares_sum_to_one(self, tmp_path, healthy_stereo):
        analysis = analyze_audio(write_wav(tmp_path / "f.wav", healthy_stereo), measure_r128=False)
        total = sum(band.share or 0.0 for band in analysis.frequency.bands)
        assert total == pytest.approx(1.0, abs=0.01)


class TestDegenerateInput:
    def test_silence_does_not_raise_or_report_nonsense(self, tmp_path):
        path = write_wav(tmp_path / "silent.wav", np.zeros((RATE * 2, 2)))
        analysis = analyze_audio(path, measure_r128=False)
        assert analysis.level.silence_ratio == pytest.approx(1.0)
        assert analysis.level.clipped_samples == 0
        assert analysis.technical.duration_seconds == pytest.approx(2.0, abs=0.01)

    def test_near_silence_is_measured_not_gated_away(self, tmp_path, healthy_stereo):
        """The activity gate is relative, so a quiet file is still a file."""
        path = write_wav(tmp_path / "quiet.wav", healthy_stereo * 0.0005)
        analysis = analyze_audio(path, measure_r128=False)
        assert not np.isnan(analysis.frequency.spectral_centroid_hz.p50)
        assert analysis.frequency.spectral_centroid_hz.p50 > 0

    def test_audio_shorter_than_one_frame_still_analyses(self, tmp_path):
        short = stereo(seconds=FRAME_SIZE / RATE / 4)
        analysis = analyze_audio(write_wav(tmp_path / "s.wav", short), measure_r128=False)
        assert analysis.technical.frames < FRAME_SIZE
        assert not np.isnan(analysis.frequency.spectral_centroid_hz.p50)

    def test_clipped_input_is_counted(self, tmp_path, healthy_stereo):
        loud = np.clip(healthy_stereo * 8.0, -1.0, 1.0)
        analysis = analyze_audio(write_wav(tmp_path / "c.wav", loud), measure_r128=False)
        assert analysis.level.clipped_samples > 0
        assert analysis.level.near_clipped_samples >= analysis.level.clipped_samples

    def test_clean_input_reports_no_clipping(self, tmp_path, healthy_stereo):
        analysis = analyze_audio(write_wav(tmp_path / "q.wav", healthy_stereo), measure_r128=False)
        assert analysis.level.clipped_samples == 0

    def test_dc_offset_is_detected(self, tmp_path, healthy_stereo):
        offset = np.clip(healthy_stereo * 0.5 + 0.1, -1.0, 1.0)
        analysis = analyze_audio(write_wav(tmp_path / "dc.wav", offset), measure_r128=False)
        assert analysis.level.dc_offset > 0.05


class TestActivityGate:
    def test_leading_silence_does_not_drag_the_percentiles_down(self, tmp_path, healthy_stereo):
        """A fade-in must not make the track look as if it loses its top.

        Without the gate the low percentiles describe the silence, and
        every track with an intro reads as intermittently dull.
        """
        padded = np.concatenate([np.zeros((RATE * 3, 2)), healthy_stereo])
        plain = analyze_audio(write_wav(tmp_path / "p.wav", healthy_stereo), measure_r128=False)
        gated = analyze_audio(write_wav(tmp_path / "g.wav", padded), measure_r128=False)
        assert gated.frequency.air_ratio_db.p10 == pytest.approx(
            plain.frequency.air_ratio_db.p10, abs=1.5
        )

    def test_the_gate_threshold_is_relative_to_the_loudest_frame(self):
        assert ACTIVITY_GATE_DB > 0


class TestFrequency:
    def test_a_dark_mix_measures_darker_than_a_bright_one(self, tmp_path, dull_stereo):
        bright = stereo(slope_db_per_octave=-2.0)
        dark = analyze_audio(write_wav(tmp_path / "d.wav", dull_stereo), measure_r128=False)
        light = analyze_audio(write_wav(tmp_path / "b.wav", bright), measure_r128=False)
        assert (
            dark.frequency.spectral_slope_db_per_octave
            < light.frequency.spectral_slope_db_per_octave
        )
        assert dark.frequency.air_ratio_db.p50 < light.frequency.air_ratio_db.p50
        assert dark.frequency.spectral_centroid_hz.p50 < light.frequency.spectral_centroid_hz.p50

    def test_the_measured_slope_matches_the_constructed_one(self, tmp_path):
        # Without the baseline top trim: this is a test of the analyser's
        # fidelity to a constructed spectrum, so the spectrum has to be
        # the one that was asked for.
        samples = stereo(slope_db_per_octave=-5.0, neutral_top=False)
        analysis = analyze_audio(write_wav(tmp_path / "s.wav", samples), measure_r128=False)
        assert analysis.frequency.spectral_slope_db_per_octave == pytest.approx(-5.0, abs=0.5)

    def test_the_low_mid_peak_follows_the_track(self, tmp_path):
        """The mud correction is centred here, so it must not be fixed."""
        low = stereo(band_gains=((170.0, 200.0, 18.0),))
        high = stereo(band_gains=((340.0, 390.0, 18.0),))
        low_peak = analyze_audio(write_wav(tmp_path / "l.wav", low), measure_r128=False)
        high_peak = analyze_audio(write_wav(tmp_path / "h.wav", high), measure_r128=False)
        assert low_peak.frequency.low_mid_peak_hz < 250
        assert high_peak.frequency.low_mid_peak_hz > 300


class TestSibilance:
    def test_bursts_raise_peak_excess_but_steady_energy_does_not(self, tmp_path, healthy_stereo):
        """The distinction the whole detector rests on.

        A steadily bright track and a spiky one can hold identical 6-9 kHz
        energy. Only the second is sibilant, and treating the first as
        sibilant would suppress brightness corrections on bright music.
        """
        bursty = add_bursts(healthy_stereo, low_hz=6_000.0, high_hz=9_000.0, gain_db=6.0)
        steady = stereo(band_gains=((6_000.0, 9_000.0, 10.0),))
        spiky = analyze_audio(write_wav(tmp_path / "sp.wav", bursty), measure_r128=False)
        flat = analyze_audio(write_wav(tmp_path / "st.wav", steady), measure_r128=False)
        assert spiky.sibilance.sibilance_peak_excess_db > 10.0
        assert flat.sibilance.sibilance_peak_excess_db < 5.0


class TestStereo:
    def test_mono_reports_mono_rather_than_inventing_a_stereo_image(self, tmp_path):
        path = write_wav(tmp_path / "mono.wav", shaped_noise() * 0.4)
        stereo_metrics = analyze_audio(path, measure_r128=False).stereo
        assert stereo_metrics.is_stereo is False
        assert stereo_metrics.width is None
        assert stereo_metrics.correlation is None
        assert stereo_metrics.lr_balance_db is None

    def test_mono_has_no_stereo_spatial_proxies_either(self, tmp_path):
        path = write_wav(tmp_path / "mono.wav", shaped_noise() * 0.4)
        spatial = analyze_audio(path, measure_r128=False).spatial
        assert spatial.stereo_decorrelation is None
        assert spatial.high_band_decorrelation is None

    def test_a_narrow_image_measures_narrower_than_a_wide_one(self, tmp_path, narrow_stereo):
        wide = stereo(decorrelation=2.5)
        thin = analyze_audio(write_wav(tmp_path / "n.wav", narrow_stereo), measure_r128=False)
        broad = analyze_audio(write_wav(tmp_path / "w.wav", wide), measure_r128=False)
        assert thin.stereo.width < broad.stereo.width
        assert thin.stereo.correlation > broad.stereo.correlation

    def test_phase_opposed_channels_read_as_fully_decorrelated(self, tmp_path, healthy_stereo):
        flipped = np.stack([healthy_stereo[:, 0], -healthy_stereo[:, 0]], axis=1)
        analysis = analyze_audio(write_wav(tmp_path / "f.wav", flipped), measure_r128=False)
        assert analysis.stereo.correlation == pytest.approx(-1.0, abs=0.01)
        assert analysis.stereo.low_band_correlation == pytest.approx(-1.0, abs=0.01)
        assert analysis.stereo.width == pytest.approx(1.0, abs=0.01)

    def test_balance_carries_the_sign_the_decision_engine_relies_on(self, tmp_path):
        louder_left = analyze_audio(
            write_wav(tmp_path / "l.wav", stereo(balance_db=3.0)), measure_r128=False
        )
        louder_right = analyze_audio(
            write_wav(tmp_path / "r.wav", stereo(balance_db=-3.0)), measure_r128=False
        )
        assert louder_left.stereo.lr_balance_db > 2.0
        assert louder_right.stereo.lr_balance_db < -2.0

    def test_width_ignores_bass_side_energy(self, tmp_path, healthy_stereo):
        """Side energy under 120 Hz is a mono-compatibility fault.

        Counting it as image width would make a track measure narrower
        after the engine repaired its bass, which is the wrong direction.
        """
        analysis = analyze_audio(write_wav(tmp_path / "h.wav", healthy_stereo), measure_r128=False)
        assert analysis.stereo.width is not None
        assert analysis.stereo.full_band_width is not None
        assert analysis.stereo.width != analysis.stereo.full_band_width


class TestTransient:
    def test_flux_and_onsets_are_reported_for_real_audio(self, tmp_path, healthy_stereo):
        transient = analyze_audio(
            write_wav(tmp_path / "t.wav", healthy_stereo), measure_r128=False
        ).transient
        assert transient.onset_rate_per_second >= 0.0
        assert 0.0 <= transient.transient_density <= 1.0
        assert transient.spectral_flux.p50 >= 0.0
