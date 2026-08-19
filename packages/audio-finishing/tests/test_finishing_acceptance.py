"""Acceptance tests: does the engine refuse its own bad work?

The question these ask is narrower than "is the output good". It is
"is the output better than the raw master", which is the only question
the engine is entitled to answer, because the raw master is always
available and always a legitimate deliverable.

Most of these construct a finished analysis by hand rather than by
rendering. That is deliberate: the failures worth testing — a shelf that
darkened the track, a correction that bought air with sibilance, a
render that came back louder — are ones the current filter chain does
not produce. Waiting for a real render to misbehave would mean the
guard is only tested once it is already too late.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from conftest import stereo, write_wav

from luber_audio_finishing.acceptance import (
    MAX_LOUDNESS_INCREASE_LU,
    MIN_BELL_EFFICACY_FRACTION,
    MIN_EFFICACY_FRACTION,
    AcceptanceOutcome,
    CheckKind,
    adjudicate,
)
from luber_audio_finishing.analysis import analyze_audio
from luber_audio_finishing.decision import (
    ActionKind,
    FinishingAction,
    FinishingPlan,
)
from luber_audio_finishing.risks import RiskFlag
from luber_audio_finishing.version import FINISHING_VERSION


@pytest.fixture
def source(tmp_path):
    return analyze_audio(write_wav(tmp_path / "src.wav", stereo()), measure_r128=False)


def plan_with(*actions: FinishingAction) -> FinishingPlan:
    return FinishingPlan(
        finishing_version=FINISHING_VERSION,
        actions=actions,
        risks=(),
        deferred=(),
    )


def shelf_lift(gain_db: float = 2.0) -> FinishingAction:
    return FinishingAction(
        kind=ActionKind.HIGH_SHELF_LIFT,
        trigger=RiskFlag.AIR_DEFICIT,
        reason="test",
        metric="frequency.air_ratio_db.p50",
        measured_value=-31.0,
        threshold=-30.0,
        gain_db=gain_db,
        requested_gain_db=gain_db,
        ceiling_db=3.0,
        clamped=False,
    )


def with_air(analysis, delta_db: float):
    """The same analysis with its air ratio moved."""
    frequency = analysis.frequency
    air = frequency.air_ratio_db
    return replace(
        analysis,
        frequency=replace(
            frequency,
            air_ratio_db=replace(
                air,
                p10=air.p10 + delta_db,
                p50=air.p50 + delta_db,
                p90=air.p90 + delta_db,
                mean=air.mean + delta_db,
            ),
        ),
    )


class TestAnUnchangedRenderIsNotAccepted:
    def test_a_lift_that_did_nothing_is_rejected(self, source):
        """A stage that costs a filter and delivers nothing is not free."""
        verdict = adjudicate(plan_with(shelf_lift(2.0)), source, source)
        assert verdict.outcome is AcceptanceOutcome.REJECTED
        assert any(check.kind is CheckKind.EFFICACY for check in verdict.failures)

    def test_a_lift_that_darkened_the_track_is_rejected(self, source):
        """The failure mode safety checks cannot see.

        A mis-signed shelf is exactly as peak-safe, exactly as long and
        exactly as stereo as a correct one.
        """
        finished = with_air(source, -2.0)
        verdict = adjudicate(plan_with(shelf_lift(2.0)), source, finished)
        assert verdict.outcome is AcceptanceOutcome.REJECTED
        assert any("wrong direction" in r or "requested" in r for r in verdict.reasons)

    def test_a_token_move_is_rejected(self, source):
        """Right direction, wrong magnitude, still not worth a stage."""
        finished = with_air(source, 2.0 * MIN_EFFICACY_FRACTION * 0.5)
        verdict = adjudicate(plan_with(shelf_lift(2.0)), source, finished)
        assert verdict.outcome is AcceptanceOutcome.REJECTED

    def test_a_working_lift_is_accepted(self, source):
        finished = with_air(source, 2.0)
        verdict = adjudicate(plan_with(shelf_lift(2.0)), source, finished)
        assert verdict.outcome is AcceptanceOutcome.ACCEPTED
        assert verdict.failures == ()


class TestRegressions:
    def test_buying_air_with_sibilance_is_rejected(self, source):
        """Measured on a real master, not imagined.

        The Phase 14 shelf moved one baseline track's sibilance peak
        excess from 15.68 to 16.03 dB. Neither value crossed a threshold,
        so nothing caught it. This is the check that watches the trade
        rather than the thresholds.
        """
        finished = with_air(source, 2.0)
        sibilance = finished.sibilance
        finished = replace(
            finished,
            sibilance=replace(
                sibilance,
                sibilance_peak_excess_db=sibilance.sibilance_peak_excess_db + 3.0,
            ),
        )
        verdict = adjudicate(plan_with(shelf_lift(2.0)), source, finished)
        assert verdict.outcome is AcceptanceOutcome.REJECTED
        assert any("sibilance" in reason for reason in verdict.reasons)

    def test_losing_mono_compatibility_is_rejected(self, source):
        """Worse on every phone, in exchange for a wider image."""
        finished = with_air(source, 2.0)
        finished = replace(
            finished,
            stereo=replace(finished.stereo, low_band_correlation=0.2),
        )
        verdict = adjudicate(plan_with(shelf_lift(2.0)), source, finished)
        assert verdict.outcome is AcceptanceOutcome.REJECTED
        assert any("mono" in reason for reason in verdict.reasons)

    def test_a_louder_render_is_rejected(self, tmp_path):
        """A louder file wins comparisons for the wrong reason.

        Loudness is measured here rather than stubbed, so the check is
        exercised against a real R128 reading.
        """
        source = analyze_audio(write_wav(tmp_path / "s.wav", stereo()))
        louder = replace(
            source,
            loudness=replace(
                source.loudness,
                integrated_lufs=(source.loudness.integrated_lufs or -14.0)
                + MAX_LOUDNESS_INCREASE_LU
                + 1.0,
            ),
        )
        louder = with_air(louder, 2.0)
        verdict = adjudicate(plan_with(shelf_lift(2.0)), source, louder)
        assert verdict.outcome is AcceptanceOutcome.REJECTED
        assert any("louder" in reason for reason in verdict.reasons)

    def test_a_quieter_render_is_allowed(self, tmp_path):
        """Peak safety legitimately costs level; that is not a regression."""
        source = analyze_audio(write_wav(tmp_path / "s.wav", stereo()))
        quieter = replace(
            source,
            loudness=replace(
                source.loudness,
                integrated_lufs=(source.loudness.integrated_lufs or -14.0) - 1.0,
            ),
        )
        verdict = adjudicate(plan_with(shelf_lift(2.0)), source, with_air(quieter, 2.0))
        assert verdict.outcome is AcceptanceOutcome.ACCEPTED


class TestSafetyStillApplies:
    def test_clipping_is_rejected_however_effective(self, source):
        """Effectiveness never buys a pass on safety."""
        finished = with_air(source, 2.0)
        finished = replace(finished, level=replace(finished.level, clipped_samples=64))
        verdict = adjudicate(plan_with(shelf_lift(2.0)), source, finished)
        assert verdict.outcome is AcceptanceOutcome.REJECTED
        assert any(check.kind is CheckKind.SAFETY for check in verdict.failures)

    def test_a_changed_duration_is_rejected(self, source):
        finished = with_air(source, 2.0)
        finished = replace(
            finished,
            technical=replace(
                finished.technical,
                duration_seconds=finished.technical.duration_seconds + 0.5,
            ),
        )
        verdict = adjudicate(plan_with(shelf_lift(2.0)), source, finished)
        assert verdict.outcome is AcceptanceOutcome.REJECTED
        assert any("duration" in reason for reason in verdict.reasons)

    def test_a_changed_channel_count_is_rejected(self, source):
        finished = with_air(source, 2.0)
        finished = replace(finished, technical=replace(finished.technical, channels=1))
        verdict = adjudicate(plan_with(shelf_lift(2.0)), source, finished)
        assert verdict.outcome is AcceptanceOutcome.REJECTED


class TestTheVerdictIsReadable:
    def test_every_check_is_recorded_not_only_the_failures(self, source):
        """ "Why was this accepted?" has to be answerable too."""
        verdict = adjudicate(plan_with(shelf_lift(2.0)), source, with_air(source, 2.0))
        assert len(verdict.checks) > 5
        assert all(check.detail for check in verdict.checks)

    def test_all_checks_run_even_after_one_fails(self, source):
        """A verdict that stops at the first objection hides the rest."""
        finished = with_air(source, -2.0)
        finished = replace(finished, level=replace(finished.level, clipped_samples=8))
        verdict = adjudicate(plan_with(shelf_lift(2.0)), source, finished)
        kinds = {check.kind for check in verdict.failures}
        assert CheckKind.SAFETY in kinds
        assert CheckKind.EFFICACY in kinds

    def test_a_no_op_plan_has_nothing_to_prove(self, source):
        """No actions means no promises, so only safety and regression apply."""
        verdict = adjudicate(plan_with(), source, source)
        assert verdict.outcome is AcceptanceOutcome.ACCEPTED
        assert not any(check.kind is CheckKind.EFFICACY for check in verdict.checks)


class TestEfficacyIsJudgedPerFilterShape:
    """A bell and a shelf cannot be held to the same standard.

    How much of a filter's gain reaches the metric depends on how well
    its shape overlaps the band the metric averages. A shelf moves the
    whole band; a bell centred at the edge of one moves it barely at
    all. Measured on the corpus, shelves delivered 0.90-1.34 of their
    gain and the low-mid bell delivered 0.06-0.72 — so a single floor
    either lets broken shelves through or rejects working bells.
    """

    def low_mid_cut(self, gain_db: float = -2.0) -> FinishingAction:
        return FinishingAction(
            kind=ActionKind.LOW_MID_CUT,
            trigger=RiskFlag.LOW_MID_MUD,
            reason="test",
            metric="frequency.low_mid_ratio_db.p50",
            measured_value=9.0,
            threshold=5.5,
            gain_db=gain_db,
            requested_gain_db=gain_db,
            ceiling_db=2.5,
            clamped=False,
            frequency_hz=380.0,
            q=0.9,
        )

    def with_low_mid(self, analysis, delta_db: float):
        frequency = analysis.frequency
        ratio = frequency.low_mid_ratio_db
        return replace(
            analysis,
            frequency=replace(
                frequency,
                low_mid_ratio_db=replace(
                    ratio,
                    p10=ratio.p10 + delta_db,
                    p50=ratio.p50 + delta_db,
                    p90=ratio.p90 + delta_db,
                    mean=ratio.mean + delta_db,
                ),
            ),
        )

    def test_the_bell_floor_is_lower_than_the_shelf_floor(self):
        assert MIN_BELL_EFFICACY_FRACTION < MIN_EFFICACY_FRACTION

    def test_an_edge_centred_bell_that_acted_is_accepted(self, source):
        """The real case: 06db2d47 delivered 49% and was right to.

        Under a single 0.5 floor this render was rejected by one
        percentage point, and the raw master shipped instead of a
        correction that had genuinely worked.
        """
        delivered = -2.0 * 0.49
        verdict = adjudicate(
            plan_with(self.low_mid_cut(-2.0)), source, self.with_low_mid(source, delivered)
        )
        assert verdict.outcome is AcceptanceOutcome.ACCEPTED, verdict.summary()

    def test_a_bell_that_achieved_nothing_is_still_rejected(self, source):
        """The floor is lower, not absent. 0.06 of a request is nothing."""
        delivered = -2.0 * 0.06
        verdict = adjudicate(
            plan_with(self.low_mid_cut(-2.0)), source, self.with_low_mid(source, delivered)
        )
        assert verdict.outcome is AcceptanceOutcome.REJECTED

    def test_a_bell_moving_the_wrong_way_is_rejected(self, source):
        """Direction is never excused, whatever the shape."""
        verdict = adjudicate(
            plan_with(self.low_mid_cut(-2.0)), source, self.with_low_mid(source, +1.5)
        )
        assert verdict.outcome is AcceptanceOutcome.REJECTED

    def test_a_shelf_delivering_the_same_share_is_rejected(self, source):
        """The same 0.49 that passes for a bell fails for a shelf."""
        verdict = adjudicate(plan_with(shelf_lift(2.0)), source, with_air(source, 2.0 * 0.49))
        assert verdict.outcome is AcceptanceOutcome.REJECTED
