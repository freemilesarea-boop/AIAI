"""Decision engine tests: is every correction earned, bounded and sane?

The tests that matter most are the ones about restraint. An engine that
corrects everything it can measure is easy to write and would ruin most
of the catalogue, so the invariants here are mostly about what does
*not* happen: healthy audio is left alone, ceilings hold, and a track
that is both dull and sibilant does not get brightened into pain.
"""

from __future__ import annotations

import pytest
from conftest import add_bursts, stereo, write_wav

from luber_audio_finishing.analysis import analyze_audio
from luber_audio_finishing.decision import (
    MAX_HIGH_SHELF_DB,
    MAX_WIDTH_ADJUST_DB,
    MIN_ACTION_DB,
    SIBILANCE_SHELF_CEILING_DB,
    ActionKind,
    FinishingDecisionEngine,
)
from luber_audio_finishing.risks import RiskFlag
from luber_audio_finishing.version import FINISHING_VERSION


def plan_for(tmp_path, samples, name="x.wav"):
    analysis = analyze_audio(write_wav(tmp_path / name, samples), measure_r128=False)
    return FinishingDecisionEngine().plan(analysis)


class TestHealthyAudio:
    def test_healthy_audio_produces_no_action(self, tmp_path, healthy_stereo):
        """The invariant the whole engine is judged on.

        If a well-balanced mix provokes corrections, the thresholds
        describe a taste rather than a defect, and every song in the
        catalogue gets reshaped toward it.
        """
        plan = plan_for(tmp_path, healthy_stereo)
        assert plan.is_no_action
        assert plan.actions == ()
        assert plan.risks == ()

    def test_a_bright_mix_is_also_left_alone(self, tmp_path):
        """Bright is not a defect; only dull and harsh are."""
        plan = plan_for(tmp_path, stereo(slope_db_per_octave=-2.0))
        assert plan.is_no_action

    def test_no_action_still_records_the_engine_version(self, tmp_path, healthy_stereo):
        assert plan_for(tmp_path, healthy_stereo).finishing_version == FINISHING_VERSION


class TestDullAudio:
    def test_a_dull_track_gets_a_high_shelf(self, tmp_path, dull_stereo):
        plan = plan_for(tmp_path, dull_stereo)
        assert RiskFlag.HIGH_FREQUENCY_DEFICIT in plan.flags
        shelf = plan.action(ActionKind.HIGH_SHELF_LIFT)
        assert shelf is not None
        assert shelf.gain_db is not None and shelf.gain_db > 0

    def test_the_lift_never_exceeds_the_ceiling(self, tmp_path):
        """Even audio with essentially no top end.

        A proportional rule with no ceiling would ask for 10 dB here.
        """
        plan = plan_for(tmp_path, stereo(band_gains=((3_000.0, 24_000.0, -40.0),)))
        shelf = plan.action(ActionKind.HIGH_SHELF_LIFT)
        assert shelf is not None and shelf.gain_db is not None
        assert shelf.gain_db <= MAX_HIGH_SHELF_DB
        assert shelf.clamped is True
        assert shelf.requested_gain_db > shelf.gain_db

    def test_the_correction_is_partial_not_total(self, tmp_path, dull_stereo):
        """Full correction would impose one spectral opinion on every song."""
        plan = plan_for(tmp_path, dull_stereo)
        shelf = plan.action(ActionKind.HIGH_SHELF_LIFT)
        assert shelf is not None and shelf.requested_gain_db is not None
        deficit = shelf.threshold - shelf.measured_value
        assert shelf.requested_gain_db < deficit


class TestContradictoryRisks:
    def test_a_dull_and_sibilant_track_is_barely_brightened(self, tmp_path, dull_stereo):
        """The contradiction that matters, and it is not hypothetical.

        Eight of the forty baseline tracks are both dark and spiky at
        6-9 kHz. Answering the darkness at full strength would trade a
        fixable dullness for an unfixable harshness.
        """
        both = add_bursts(dull_stereo, low_hz=6_000.0, high_hz=9_000.0, gain_db=8.0)
        plan = plan_for(tmp_path, both)
        assert RiskFlag.SIBILANCE_RISK in plan.flags
        assert RiskFlag.HIGH_FREQUENCY_DEFICIT in plan.flags

        shelf = plan.action(ActionKind.HIGH_SHELF_LIFT)
        if shelf is not None:
            assert shelf.gain_db is not None
            assert shelf.gain_db <= SIBILANCE_SHELF_CEILING_DB
            assert shelf.ceiling_db == SIBILANCE_SHELF_CEILING_DB
            assert any("6-9 kHz" in note for note in shelf.notes)

    def test_sibilance_alone_provokes_nothing(self, tmp_path, sibilant_stereo):
        """A risk is not an instruction.

        Dynamic de-essing is deferred, so a sibilant but otherwise
        healthy track is measured, flagged, and left alone.
        """
        plan = plan_for(tmp_path, sibilant_stereo)
        assert RiskFlag.SIBILANCE_RISK in plan.flags
        assert plan.action(ActionKind.HIGH_SHELF_LIFT) is None

    def test_harshness_blocks_the_presence_lift_entirely(self, tmp_path):
        """2.5-5 kHz and 2-5 kHz are the same region.

        Lifting presence on a track that already spikes there is lifting
        exactly what hurts, so the rule does not fire at all.
        """
        harsh_and_dull = add_bursts(
            stereo(band_gains=((2_000.0, 24_000.0, -16.0),)),
            low_hz=2_500.0,
            high_hz=5_000.0,
            gain_db=10.0,
        )
        plan = plan_for(tmp_path, harsh_and_dull)
        if RiskFlag.HARSHNESS_RISK in plan.flags:
            assert plan.action(ActionKind.PRESENCE_LIFT) is None


class TestMuddyAudio:
    def test_a_muddy_track_gets_a_low_mid_cut(self, tmp_path, muddy_stereo):
        plan = plan_for(tmp_path, muddy_stereo)
        assert RiskFlag.LOW_MID_MUD in plan.flags
        cut = plan.action(ActionKind.LOW_MID_CUT)
        assert cut is not None and cut.gain_db is not None
        assert cut.gain_db < 0

    def test_the_cut_is_centred_on_this_track_not_on_a_preset(self, tmp_path):
        """Two tracks thick in different places get different cuts.

        The upper case needs more gain to trip the same flag: a -4 dB per
        octave spectrum already holds far less energy at 250-400 Hz than
        at 150-240, so an equal boost there moves the band average less.
        """
        low = plan_for(tmp_path, stereo(band_gains=((150.0, 240.0, 16.0),)), "low.wav")
        high = plan_for(tmp_path, stereo(band_gains=((250.0, 400.0, 20.0),)), "high.wav")
        low_cut = low.action(ActionKind.LOW_MID_CUT)
        high_cut = high.action(ActionKind.LOW_MID_CUT)
        assert low_cut is not None and high_cut is not None
        assert low_cut.frequency_hz < high_cut.frequency_hz

    def test_the_cut_is_broad(self, tmp_path, muddy_stereo):
        """A narrow notch would leave a hole where warmth used to be."""
        cut = plan_for(tmp_path, muddy_stereo).action(ActionKind.LOW_MID_CUT)
        assert cut is not None and cut.q is not None
        assert cut.q < 1.5


class TestStereoDecisions:
    def test_a_narrow_track_is_widened(self, tmp_path, narrow_stereo):
        plan = plan_for(tmp_path, narrow_stereo)
        assert RiskFlag.STEREO_TOO_NARROW in plan.flags
        width = plan.action(ActionKind.STEREO_WIDTH)
        assert width is not None and width.gain_db is not None
        assert 0 < width.gain_db <= MAX_WIDTH_ADJUST_DB

    def test_widening_always_brings_bass_mono_with_it(self, tmp_path, narrow_stereo):
        """The one stereo fault that survives into mono as missing bass.

        A side-channel boost lifts every frequency, bass included, so the
        mono stage is not optional whenever the width stage runs.
        """
        plan = plan_for(tmp_path, narrow_stereo)
        assert plan.action(ActionKind.STEREO_WIDTH) is not None
        assert plan.action(ActionKind.LOW_FREQUENCY_MONO) is not None

    def test_out_of_phase_bass_is_summed_to_mono(self, tmp_path, healthy_stereo):
        import numpy as np

        flipped = np.stack([healthy_stereo[:, 0], -healthy_stereo[:, 0]], axis=1)
        plan = plan_for(tmp_path, flipped)
        assert RiskFlag.LOW_END_PHASE_RISK in plan.flags
        mono = plan.action(ActionKind.LOW_FREQUENCY_MONO)
        assert mono is not None
        assert mono.frequency_hz == pytest.approx(120.0)

    def test_an_off_centre_image_is_recentred(self, tmp_path):
        plan = plan_for(tmp_path, stereo(balance_db=2.5))
        assert RiskFlag.STEREO_IMBALANCE in plan.flags
        balance = plan.action(ActionKind.BALANCE_CORRECTION)
        assert balance is not None and balance.gain_db is not None
        # Left was louder, so the correction must be negative.
        assert balance.gain_db < 0

    def test_a_mono_file_gets_no_stereo_corrections(self, tmp_path):
        from conftest import shaped_noise

        plan = plan_for(tmp_path, shaped_noise() * 0.4)
        assert plan.action(ActionKind.STEREO_WIDTH) is None
        assert plan.action(ActionKind.BALANCE_CORRECTION) is None
        assert plan.action(ActionKind.LOW_FREQUENCY_MONO) is None


class TestAuditability:
    def test_every_action_carries_its_evidence(self, tmp_path, dull_stereo):
        for action in plan_for(tmp_path, dull_stereo).actions:
            assert action.metric
            assert action.reason
            if action.gain_db is not None:
                assert action.ceiling_db is not None
                assert abs(action.gain_db) <= abs(action.ceiling_db) + 1e-9
                assert abs(action.gain_db) >= MIN_ACTION_DB

    def test_every_trigger_corresponds_to_a_raised_flag(self, tmp_path, dull_stereo):
        plan = plan_for(tmp_path, dull_stereo)
        for action in plan.actions:
            if action.trigger is not None:
                assert action.trigger in plan.flags

    def test_deferrals_are_recorded_with_reasons(self, tmp_path, healthy_stereo):
        """ "Chose not to" and "forgot" look identical without this."""
        plan = plan_for(tmp_path, healthy_stereo)
        areas = {item.area for item in plan.deferred}
        assert {"transient shaping", "spatial and reverb", "dynamic harshness control"} <= areas
        assert all(len(item.reason) > 40 for item in plan.deferred)

    def test_planning_is_deterministic(self, tmp_path, dull_stereo):
        path = write_wav(tmp_path / "d.wav", dull_stereo)
        analysis = analyze_audio(path, measure_r128=False)
        engine = FinishingDecisionEngine()
        first, second = engine.plan(analysis), engine.plan(analysis)
        assert first == second
