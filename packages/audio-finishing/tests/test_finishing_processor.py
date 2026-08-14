"""Processor tests: is the rendered file safe, and is the raw one intact?

These use real ffmpeg on real files. Mocking the renderer would test the
command string rather than the audio, and the properties that matter —
no clipping, unchanged duration, an untouched source — are properties of
the audio.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from conftest import add_bursts, stereo, write_wav

from luber_audio_finishing.analysis import analyze_audio
from luber_audio_finishing.audiofile import load_audio
from luber_audio_finishing.decision import ActionKind, FinishingDecisionEngine
from luber_audio_finishing.processor import (
    AlreadyFinishedError,
    FinishingError,
    build_filter_graph,
    finish_audio,
    read_finishing_stamp,
)
from luber_audio_finishing.risks import STEREO_IMBALANCE_DB
from luber_audio_finishing.version import FINISHING_VERSION

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe required to render audio",
)


class TestFilterGraph:
    def test_no_action_produces_an_empty_graph(self, tmp_path, healthy_stereo):
        analysis = analyze_audio(write_wav(tmp_path / "h.wav", healthy_stereo), measure_r128=False)
        assert build_filter_graph(FinishingDecisionEngine().plan(analysis)) == ""

    def test_cuts_are_placed_before_boosts(self, tmp_path, muddy_stereo):
        """So a boost lifts audio that has already lost its excess."""
        dark_and_muddy = stereo(band_gains=((150.0, 400.0, 9.0), (4_000.0, 24_000.0, -24.0)))
        analysis = analyze_audio(write_wav(tmp_path / "m.wav", dark_and_muddy), measure_r128=False)
        graph = build_filter_graph(FinishingDecisionEngine().plan(analysis))
        assert "equalizer" in graph and "highshelf" in graph
        assert graph.index("equalizer") < graph.index("highshelf")

    def test_the_stereo_stage_only_filters_the_side_channel(self, tmp_path, narrow_stereo):
        """The mid is the mono sum and must survive untouched.

        A crossover that mono-ised the low band and summed it back was
        tried first and rotated phase enough to inflate the sample peak
        by up to 3 dB, which the level stage then had to give away.
        """
        analysis = analyze_audio(write_wav(tmp_path / "n.wav", narrow_stereo), measure_r128=False)
        graph = build_filter_graph(FinishingDecisionEngine().plan(analysis))
        assert "[mid]" in graph and "[side]" in graph
        mid_stage = graph[graph.index("[ms_a]") : graph.index("[mid]")]
        assert "highpass" not in mid_stage and "volume" not in mid_stage

    def test_the_level_stage_comes_last(self, tmp_path, dull_stereo):
        analysis = analyze_audio(write_wav(tmp_path / "d.wav", dull_stereo), measure_r128=False)
        plan = FinishingDecisionEngine().plan(analysis)
        graph = build_filter_graph(plan, output_gain_db=-1.5)
        assert graph.endswith("volume=-1.5000dB")


class TestRendering:
    def test_a_finished_file_never_clips(self, tmp_path, dull_stereo):
        result = finish_audio(write_wav(tmp_path / "d.wav", dull_stereo), tmp_path / "out.wav")
        assert result.changed
        assert result.finished_analysis.level.clipped_samples == 0

    def test_the_output_stays_under_the_ceiling(self, tmp_path, dull_stereo):
        result = finish_audio(write_wav(tmp_path / "d.wav", dull_stereo), tmp_path / "out.wav")
        ceiling = result.plan.output_ceiling_dbfs
        assert result.finished_analysis.level.peak_dbfs <= ceiling + 0.1

    def test_duration_is_preserved(self, tmp_path, dull_stereo):
        result = finish_audio(write_wav(tmp_path / "d.wav", dull_stereo), tmp_path / "out.wav")
        assert result.finished_analysis.technical.duration_seconds == pytest.approx(
            result.source_analysis.technical.duration_seconds, abs=0.01
        )

    def test_sample_rate_and_channels_are_preserved(self, tmp_path, dull_stereo):
        result = finish_audio(write_wav(tmp_path / "d.wav", dull_stereo), tmp_path / "out.wav")
        assert result.finished_analysis.technical.sample_rate == 48_000
        assert result.finished_analysis.technical.channels == 2

    def test_the_output_is_finite(self, tmp_path, dull_stereo):
        result = finish_audio(write_wav(tmp_path / "d.wav", dull_stereo), tmp_path / "out.wav")
        assert np.all(np.isfinite(load_audio(result.output_path).samples))

    def test_bit_depth_is_never_reduced(self, tmp_path, dull_stereo):
        result = finish_audio(write_wav(tmp_path / "d.wav", dull_stereo), tmp_path / "out.wav")
        assert result.finished_analysis.technical.bit_depth == 24

    def test_the_plan_actually_moved_the_measurement_it_cited(self, tmp_path, dull_stereo):
        result = finish_audio(write_wav(tmp_path / "d.wav", dull_stereo), tmp_path / "out.wav")
        assert result.plan.action(ActionKind.HIGH_SHELF_LIFT) is not None
        assert (
            result.finished_analysis.frequency.air_ratio_db.p50
            > result.source_analysis.frequency.air_ratio_db.p50
        )

    def test_dynamics_are_not_crushed(self, tmp_path, dull_stereo):
        """No limiter runs in p14-v1, so crest factor barely moves."""
        result = finish_audio(write_wav(tmp_path / "d.wav", dull_stereo), tmp_path / "out.wav")
        assert result.finished_analysis.level.crest_factor_db == pytest.approx(
            result.source_analysis.level.crest_factor_db, abs=1.5
        )

    def test_the_output_is_not_left_off_centre(self, tmp_path, narrow_stereo):
        """The mid/side stage can shift balance; the engine must undo it."""
        result = finish_audio(write_wav(tmp_path / "n.wav", narrow_stereo), tmp_path / "out.wav")
        balance = result.finished_analysis.stereo.lr_balance_db
        assert abs(balance) <= STEREO_IMBALANCE_DB


class TestRawPreservation:
    def test_the_source_file_is_byte_identical_afterwards(self, tmp_path, dull_stereo):
        """Non-negotiable: the raw generation is the only copy there is."""
        source = write_wav(tmp_path / "raw.wav", dull_stereo)
        before = source.read_bytes()
        finish_audio(source, tmp_path / "out.wav")
        assert source.read_bytes() == before

    def test_writing_over_the_source_is_refused(self, tmp_path, dull_stereo):
        source = write_wav(tmp_path / "raw.wav", dull_stereo)
        with pytest.raises(FinishingError, match="overwrite"):
            finish_audio(source, source)

    def test_a_missing_source_is_refused(self, tmp_path):
        with pytest.raises(FinishingError):
            finish_audio(tmp_path / "nope.wav", tmp_path / "out.wav")


class TestNoAction:
    def test_healthy_audio_writes_nothing_at_all(self, tmp_path, healthy_stereo):
        """The raw master is already the deliverable.

        Re-encoding it into an identical "finished" file would only give
        the two copies a chance to drift apart.
        """
        result = finish_audio(write_wav(tmp_path / "h.wav", healthy_stereo), tmp_path / "out.wav")
        assert result.changed is False
        assert result.output_path is None
        assert not (tmp_path / "out.wav").exists()
        assert result.output_gain_db == 0.0


class TestIdempotency:
    def test_a_finished_file_is_stamped_with_the_engine_version(self, tmp_path, dull_stereo):
        result = finish_audio(write_wav(tmp_path / "d.wav", dull_stereo), tmp_path / "out.wav")
        assert read_finishing_stamp(result.output_path) == FINISHING_VERSION

    def test_a_raw_file_carries_no_stamp(self, tmp_path, dull_stereo):
        assert read_finishing_stamp(write_wav(tmp_path / "d.wav", dull_stereo)) is None

    def test_finishing_a_finished_file_is_refused(self, tmp_path, dull_stereo):
        """Otherwise corrections stack on corrections.

        The second pass would measure audio the first pass already
        changed, and the result would depend on how many times it ran.
        """
        first = finish_audio(write_wav(tmp_path / "d.wav", dull_stereo), tmp_path / "one.wav")
        with pytest.raises(AlreadyFinishedError):
            finish_audio(first.output_path, tmp_path / "two.wav")

    def test_the_same_source_produces_the_same_plan_and_the_same_bytes(self, tmp_path, dull_stereo):
        source = write_wav(tmp_path / "d.wav", dull_stereo)
        first = finish_audio(source, tmp_path / "a.wav")
        second = finish_audio(source, tmp_path / "b.wav")
        assert first.plan == second.plan
        assert first.output_gain_db == second.output_gain_db
        assert first.output_path.read_bytes() == second.output_path.read_bytes()


class TestContradictoryRender:
    def test_a_dull_sibilant_track_is_not_brightened_recklessly(self, tmp_path, dull_stereo):
        """End to end, not just in the plan.

        The suppression rule is only worth anything if the rendered audio
        reflects it, so the 6-9 kHz band is measured on the actual output.
        """
        both = add_bursts(dull_stereo, low_hz=6_000.0, high_hz=9_000.0, gain_db=8.0)
        result = finish_audio(write_wav(tmp_path / "b.wav", both), tmp_path / "out.wav")
        if not result.changed:
            return
        moved = (
            result.finished_analysis.sibilance.sibilance_ratio_db.p90
            - result.source_analysis.sibilance.sibilance_ratio_db.p90
        )
        assert moved <= 1.5
