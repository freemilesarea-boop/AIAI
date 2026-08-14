"""Loader, band, version-stamp and serialisation tests.

Small pieces, but each one can fail silently: a WAV decoded at the wrong
bit depth still produces plausible audio, a NaN written into JSON still
produces a file, and a version stamp that does not round-trip still
looks like it works until the day something has to be reprocessed.
"""

from __future__ import annotations

import json
import wave
from itertools import pairwise

import numpy as np
import pytest
from conftest import RATE, shaped_noise, stereo, write_wav

from luber_audio_finishing.analysis import analyze_audio
from luber_audio_finishing.audiofile import AudioLoadError, load_audio
from luber_audio_finishing.bands import BAND_EDGES, BAND_NAMES, band_coverage
from luber_audio_finishing.decision import FinishingDecisionEngine
from luber_audio_finishing.loudness import UNMEASURED
from luber_audio_finishing.report import build_report
from luber_audio_finishing.risks import RiskFlag
from luber_audio_finishing.serialize import analysis_to_dict, plan_to_dict, report_to_dict
from luber_audio_finishing.version import (
    FINISHING_VERSION,
    finishing_stamp,
    parse_finishing_stamp,
)


class TestLoader:
    @pytest.mark.parametrize("bit_depth", [16, 24])
    def test_round_trips_pcm_at_full_scale(self, tmp_path, bit_depth):
        samples = stereo(seconds=0.5, amplitude=0.9)
        loaded = load_audio(write_wav(tmp_path / "a.wav", samples, bit_depth=bit_depth))
        assert loaded.channels == 2
        assert loaded.bit_depth == bit_depth
        assert np.abs(loaded.samples - samples).max() < 0.001

    def test_mono_stays_one_channel(self, tmp_path):
        loaded = load_audio(write_wav(tmp_path / "m.wav", shaped_noise(seconds=0.5) * 0.3))
        assert loaded.channels == 1
        assert loaded.is_stereo is False

    def test_mono_downmix_averages_the_channels(self, tmp_path):
        samples = stereo(seconds=0.5)
        loaded = load_audio(write_wav(tmp_path / "s.wav", samples))
        assert np.abs(loaded.mono() - samples.mean(axis=1)).max() < 0.001

    def test_a_non_audio_file_is_rejected(self, tmp_path):
        junk = tmp_path / "junk.wav"
        junk.write_bytes(b"not a wav file at all, not even close")
        with pytest.raises(AudioLoadError):
            load_audio(junk)

    def test_a_zero_length_wav_is_rejected(self, tmp_path):
        path = tmp_path / "empty.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(3)
            handle.setframerate(RATE)
            handle.writeframes(b"")
        with pytest.raises(AudioLoadError):
            load_audio(path)


class TestBands:
    def test_the_bands_are_contiguous(self):
        """Gaps would make band shares silently fail to sum to one."""
        for (_, _, previous_high), (_, low, _) in pairwise(BAND_EDGES):
            assert low == previous_high

    def test_every_band_is_present_at_48_khz(self):
        assert all(not band.is_absent for band in band_coverage(48_000))

    def test_the_top_band_disappears_below_its_own_floor(self):
        coverage = {band.name: band for band in band_coverage(22_050)}
        assert coverage["ultra_high"].is_absent
        assert coverage["air"].is_partial

    def test_names_match_the_edges(self):
        assert BAND_NAMES == tuple(name for name, _, _ in BAND_EDGES)


class TestVersionStamp:
    def test_a_stamp_round_trips(self):
        assert parse_finishing_stamp(finishing_stamp()) == FINISHING_VERSION

    def test_foreign_metadata_is_not_mistaken_for_a_stamp(self):
        assert parse_finishing_stamp("encoded by something else") is None
        assert parse_finishing_stamp(None) is None
        assert parse_finishing_stamp("") is None


class TestReport:
    def test_a_report_pairs_measurements_with_risks(self, tmp_path, muddy_stereo):
        report = build_report(write_wav(tmp_path / "m.wav", muddy_stereo), measure_r128=False)
        assert report.has(RiskFlag.LOW_MID_MUD)
        assert report.finding(RiskFlag.LOW_MID_MUD) is not None
        assert report.finding(RiskFlag.CLIPPING_PRESENT) is None
        assert report.technical.channels == 2
        assert report.loudness == UNMEASURED


class TestSerialisation:
    def test_an_analysis_serialises_to_strict_json(self, tmp_path, healthy_stereo):
        payload = analysis_to_dict(
            analyze_audio(write_wav(tmp_path / "h.wav", healthy_stereo), measure_r128=False)
        )
        # allow_nan=False is the point: NaN tokens are not valid JSON and
        # break strict parsers downstream.
        json.dumps(payload, allow_nan=False)

    def test_unmeasurable_values_become_null_rather_than_zero(self, tmp_path):
        """ "Not measurable" and "measured as zero" must stay distinct."""
        payload = analysis_to_dict(
            analyze_audio(write_wav(tmp_path / "z.wav", np.zeros((RATE, 2))), measure_r128=False)
        )
        assert payload["spatial"]["envelope_decay_db_per_second"] is None

    def test_no_filesystem_path_is_serialised(self, tmp_path, healthy_stereo):
        """Committed benchmark records must not carry machine paths."""
        payload = analysis_to_dict(
            analyze_audio(write_wav(tmp_path / "h.wav", healthy_stereo), measure_r128=False)
        )
        assert payload["path"] == "h.wav"
        assert "/" not in json.dumps(payload["path"])

    def test_a_report_serialises_with_its_risk_flags(self, tmp_path, muddy_stereo):
        payload = report_to_dict(
            build_report(write_wav(tmp_path / "m.wav", muddy_stereo), measure_r128=False)
        )
        json.dumps(payload, allow_nan=False)
        assert any(item["flag"] == "LOW_MID_MUD" for item in payload["risk_flags"])

    def test_a_plan_serialises_its_whole_decision_trail(self, tmp_path, dull_stereo):
        analysis = analyze_audio(write_wav(tmp_path / "d.wav", dull_stereo), measure_r128=False)
        payload = plan_to_dict(FinishingDecisionEngine().plan(analysis))
        json.dumps(payload, allow_nan=False)
        assert payload["finishing_version"] == FINISHING_VERSION
        assert payload["no_action"] is False
        assert payload["actions"] and payload["risks"] and payload["deferred"]

    def test_no_action_is_written_explicitly(self, tmp_path, healthy_stereo):
        """An empty action list alone cannot say whether the engine ran."""
        analysis = analyze_audio(write_wav(tmp_path / "h.wav", healthy_stereo), measure_r128=False)
        payload = plan_to_dict(FinishingDecisionEngine().plan(analysis))
        assert payload["no_action"] is True
        assert payload["actions"] == []
