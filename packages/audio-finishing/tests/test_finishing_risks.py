"""Risk evaluation tests: does a flag mean what its name says?

Thresholds are claims about audio, and the way they fail is by firing on
material that is merely different rather than defective. Each test here
pairs a signal that should trip a flag with one that should not.
"""

from __future__ import annotations

import numpy as np
import pytest
from finishing_fixtures import add_bursts, shaped_noise, stereo, write_wav

from luber_audio_finishing.analysis import analyze_audio
from luber_audio_finishing.risks import (
    LOW_END_EXCESS_SHARE,
    STEREO_NARROW_WIDTH,
    STEREO_WIDE_WIDTH,
    RiskFlag,
    evaluate_risks,
)


def flags_for(tmp_path, samples, name="x.wav"):
    analysis = analyze_audio(write_wav(tmp_path / name, samples), measure_r128=False)
    return {finding.flag for finding in evaluate_risks(analysis)}


class TestNoFalsePositives:
    def test_healthy_audio_raises_nothing(self, tmp_path, healthy_stereo):
        assert flags_for(tmp_path, healthy_stereo) == set()

    def test_a_bright_mix_is_not_called_harsh(self, tmp_path):
        """Steady high-frequency energy is brightness, not harshness."""
        bright = stereo(band_gains=((2_500.0, 12_000.0, 8.0),))
        raised = flags_for(tmp_path, bright)
        assert RiskFlag.HARSHNESS_RISK not in raised
        assert RiskFlag.SIBILANCE_RISK not in raised

    def test_an_ordinary_low_end_is_not_called_excessive(self, tmp_path, healthy_stereo):
        """Music concentrates energy low; the threshold is for the tail.

        The corpus-scale version of this claim is asserted against the
        committed baseline in the benchmark tests, where it belongs:
        power-law noise keeps rising all the way to 20 Hz and so carries
        far more sub energy than any real mix, which makes it the wrong
        instrument for calibrating a low-end threshold.
        """
        analysis = analyze_audio(write_wav(tmp_path / "b.wav", healthy_stereo), measure_r128=False)
        sub = analysis.frequency.band("sub")
        bass = analysis.frequency.band("bass")
        assert (sub.share + bass.share) < LOW_END_EXCESS_SHARE


class TestFrequencyRisks:
    def test_a_rolled_off_top_raises_the_high_frequency_flags(self, tmp_path, dull_stereo):
        raised = flags_for(tmp_path, dull_stereo)
        assert RiskFlag.HIGH_FREQUENCY_DEFICIT in raised
        assert RiskFlag.AIR_DEFICIT in raised

    def test_thick_low_mids_raise_mud(self, tmp_path, muddy_stereo):
        assert RiskFlag.LOW_MID_MUD in flags_for(tmp_path, muddy_stereo)

    def test_a_dominant_low_end_raises_excess(self, tmp_path):
        assert RiskFlag.LOW_END_EXCESS in flags_for(
            tmp_path, stereo(band_gains=((20.0, 150.0, 14.0),))
        )

    def test_recessed_upper_mids_raise_a_presence_deficit(self, tmp_path):
        assert RiskFlag.PRESENCE_DEFICIT in flags_for(
            tmp_path, stereo(band_gains=((2_000.0, 5_000.0, -22.0),))
        )


class TestSpikeRisks:
    def test_bursts_at_6_to_9_khz_raise_sibilance(self, tmp_path, sibilant_stereo):
        assert RiskFlag.SIBILANCE_RISK in flags_for(tmp_path, sibilant_stereo)

    def test_bursts_in_the_upper_mids_raise_harshness(self, tmp_path, healthy_stereo):
        """Bursts sit at 3.5-5 kHz, clear of the 300 Hz-3 kHz body band.

        The harshness band and the body band overlap between 2.5 and
        3 kHz, so a burst spanning the overlap raises its own denominator
        and the ratio saturates near 13 dB no matter how loud it gets.
        Real harsh material concentrates higher up, which is where the
        flag fired on seven of the forty baseline tracks.
        """
        harsh = add_bursts(healthy_stereo, low_hz=3_500.0, high_hz=5_000.0, gain_db=10.0)
        assert RiskFlag.HARSHNESS_RISK in flags_for(tmp_path, harsh)


class TestStereoRisks:
    def test_a_nearly_mono_image_raises_narrowness(self, tmp_path, narrow_stereo):
        assert RiskFlag.STEREO_TOO_NARROW in flags_for(tmp_path, narrow_stereo)

    def test_phase_opposed_channels_raise_the_low_end_phase_risk(self, tmp_path, healthy_stereo):
        flipped = np.stack([healthy_stereo[:, 0], -healthy_stereo[:, 0]], axis=1)
        raised = flags_for(tmp_path, flipped)
        assert RiskFlag.LOW_END_PHASE_RISK in raised
        assert RiskFlag.STEREO_TOO_WIDE in raised

    def test_an_off_centre_image_raises_imbalance(self, tmp_path):
        assert RiskFlag.STEREO_IMBALANCE in flags_for(tmp_path, stereo(balance_db=2.0))

    def test_a_mono_file_raises_no_stereo_risks(self, tmp_path):
        raised = flags_for(tmp_path, shaped_noise() * 0.4)
        stereo_flags = {
            RiskFlag.STEREO_TOO_NARROW,
            RiskFlag.STEREO_TOO_WIDE,
            RiskFlag.STEREO_IMBALANCE,
            RiskFlag.LOW_END_PHASE_RISK,
        }
        assert raised & stereo_flags == set()

    def test_the_width_thresholds_bracket_the_corpus(self):
        assert STEREO_NARROW_WIDTH < STEREO_WIDE_WIDTH


class TestSafetyRisks:
    def test_clipping_is_flagged(self, tmp_path, healthy_stereo):
        clipped = np.clip(healthy_stereo * 8.0, -1.0, 1.0)
        assert RiskFlag.CLIPPING_PRESENT in flags_for(tmp_path, clipped)

    def test_dc_offset_is_flagged(self, tmp_path, healthy_stereo):
        offset = np.clip(healthy_stereo * 0.5 + 0.05, -1.0, 1.0)
        assert RiskFlag.DC_OFFSET_PRESENT in flags_for(tmp_path, offset)


class TestEvidence:
    def test_every_finding_carries_the_number_that_raised_it(self, tmp_path, dull_stereo):
        analysis = analyze_audio(write_wav(tmp_path / "d.wav", dull_stereo), measure_r128=False)
        findings = evaluate_risks(analysis)
        assert findings
        for finding in findings:
            assert finding.metric
            assert finding.detail
            assert not np.isnan(finding.value)
            assert not np.isnan(finding.threshold)

    def test_findings_are_stable_for_the_same_analysis(self, tmp_path, muddy_stereo):
        analysis = analyze_audio(write_wav(tmp_path / "m.wav", muddy_stereo), measure_r128=False)
        assert evaluate_risks(analysis) == evaluate_risks(analysis)

    def test_margin_reports_distance_past_the_threshold(self, tmp_path, muddy_stereo):
        analysis = analyze_audio(write_wav(tmp_path / "m.wav", muddy_stereo), measure_r128=False)
        mud = next(f for f in evaluate_risks(analysis) if f.flag == RiskFlag.LOW_MID_MUD)
        assert mud.margin == pytest.approx(mud.value - mud.threshold)
        assert mud.margin > 0
