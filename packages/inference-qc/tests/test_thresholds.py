"""Every number restated here still agrees with the one it came from.

Four thresholds are copied rather than imported. Phase 26's evaluation
suite and the Phase 5 benchmark both hold numbers this engine needs, and
importing either would drag the training registry or a benchmark harness
into the runtime generation path.

Copying is a real cost: two places can drift, and a drift in a rejection
threshold is a change in what gets delivered that nobody reviewed. This
module is what makes the drift a test failure instead. It is the only
reason the copy is acceptable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from luber_inference_qc import thresholds as t

BENCHMARK = Path(__file__).resolve().parents[3] / "benchmarks" / "music_quality" / "scripts"


@pytest.fixture(scope="module")
def phase5():
    """The Phase 5 benchmark's metrics module, which is not a package."""
    sys.path.insert(0, str(BENCHMARK))
    try:
        from bench import metrics

        return metrics
    finally:
        sys.path.remove(str(BENCHMARK))


def test_the_duration_tolerances_are_phase_26s_and_phase_5s(phase5):
    from luber_evaluation.suite import DEFAULT_DURATION_TOLERANCE_RELATIVE

    assert t.DURATION_SOFT_TOLERANCE_RATIO == DEFAULT_DURATION_TOLERANCE_RELATIVE
    assert t.DURATION_HARD_TOLERANCE_RATIO == phase5.DURATION_TOLERANCE_RATIO


def test_the_bpm_confidence_floor_is_phase_26s():
    from luber_evaluation.suite import DEFAULT_BPM_CONFIDENCE_FLOOR

    assert t.BPM_CONFIDENCE_FLOOR == DEFAULT_BPM_CONFIDENCE_FLOOR


def test_the_silence_and_clipping_ratios_are_phase_5s(phase5):
    assert t.SILENCE_PEAK_RATIO == phase5.SILENCE_PEAK_RATIO
    assert t.NEAR_SILENCE_PEAK_RATIO == phase5.NEAR_SILENCE_PEAK_RATIO
    assert t.EXCESSIVE_SILENCE_RATIO == phase5.EXCESSIVE_SILENCE_RATIO
    assert t.SEVERE_CLIPPING_SAMPLE_RATIO == phase5.CLIPPING_SAMPLE_RATIO


def test_the_finishing_thresholds_are_phase_22s():
    from luber_audio_finishing import risks

    assert t.DC_OFFSET_LIMIT == risks.DC_OFFSET_LIMIT
    assert t.PHASE_RISK_CORRELATION == risks.BROADBAND_PHASE_CORRELATION
    assert t.NARROW_STEREO_WIDTH == risks.STEREO_NARROW_WIDTH
    assert t.CHANNEL_IMBALANCE_DB == risks.STEREO_IMBALANCE_DB
    assert t.HARSHNESS_PEAK_EXCESS_DB == risks.HARSHNESS_PEAK_EXCESS_DB
    assert t.SIBILANCE_PEAK_EXCESS_DB == risks.SIBILANCE_PEAK_EXCESS_DB


# ── the ones this phase decided, and the shape they have to keep ─────


def test_rejection_is_always_looser_than_the_finding_that_precedes_it():
    """A candidate must never be rejected for something Phase 22 merely
    records — the finishing engine's job is to repair those."""
    assert t.DC_OFFSET_SEVERE > t.DC_OFFSET_LIMIT
    assert t.CHANNEL_IMBALANCE_SEVERE_DB > t.CHANNEL_IMBALANCE_DB
    assert t.PHASE_UNSAFE_CORRELATION < t.PHASE_RISK_CORRELATION
    assert t.DURATION_HARD_TOLERANCE_RATIO > t.DURATION_SOFT_TOLERANCE_RATIO
    assert t.BPM_HARD_TOLERANCE_RATIO > t.BPM_SOFT_TOLERANCE_RATIO
    assert t.NEAR_SILENCE_PEAK_RATIO > t.SILENCE_PEAK_RATIO


def test_the_spectral_rule_sits_clear_of_the_corpus_rather_than_midway():
    """Measured over the raw corpus: real songs top out at 0.805 band
    concentration, and a degenerate tone reaches 0.967. The threshold is
    set near the tone because the cost of being wrong is a song thrown
    away, not a defect delivered."""
    assert 0.85 <= t.SPECTRAL_CONCENTRATION_SHARE < 0.96


def test_the_key_floor_is_above_phase_23s_recording_floor():
    """Phase 23's floor is for *recording* an estimate. Acting on one
    needs more."""
    from luber_dataset.factory.musical import MIN_KEY_CONFIDENCE

    assert t.KEY_CONFIDENCE_FLOOR > MIN_KEY_CONFIDENCE
