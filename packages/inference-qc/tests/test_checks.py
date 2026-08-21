"""What QC rejects, and — more importantly — what it does not.

The rejections are the easy half. A silent file is silent and nobody
argues. The half that matters is the other one: a dark master, a narrow
mix, a long fade to nothing, a mono file. Every one of those is a
production decision somebody made on purpose, and every one of them
looks, to a naive threshold, like a defect. If this engine rejects them
it is not protecting users from bad generations, it is destroying good
ones and charging for the retry.

So the acceptance table below is longer than the rejection table, and it
is the one to read first.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import qc_fixtures as fx

from luber_inference_qc import (
    CandidateGeneration,
    CandidateStatus,
    Finding,
    RequestExpectation,
    Severity,
    judge,
)
from luber_inference_qc import thresholds as t
from luber_inference_qc.checks import check_bpm
from luber_inference_qc.measurement import measure


def _judge(path, **expectation) -> CandidateGeneration:
    candidate = CandidateGeneration(
        candidate_id="cand",
        generation_id="gen",
        attempt_index=0,
        request_sha256="digest",
    )
    judge(candidate, path, RequestExpectation(**expectation))
    return candidate


def _critical(candidate: CandidateGeneration) -> set[str]:
    return {finding.code for finding in candidate.critical_findings}


# ── what must never be rejected ──────────────────────────────────────

ACCEPTED = [
    ("a plausible generation", fx.healthy, 12.0),
    ("a dark master", fx.dark_but_valid, 12.0),
    ("a bright master", fx.bright_but_valid, 12.0),
    ("a mix with no stereo width", fx.mono_as_stereo, 12.0),
    ("a genuinely mono file", fx.true_mono, 12.0),
    ("a long fade to near-nothing", fx.quiet_outro, 20.0),
    ("a master right at the ceiling", fx.peak_overshoot, 12.0),
]


@pytest.mark.parametrize(
    ("description", "make", "seconds"), ACCEPTED, ids=[case[0] for case in ACCEPTED]
)
def test_a_production_decision_is_not_a_defect(audio_dir, description, make, seconds):
    candidate = _judge(make(audio_dir / "case.wav"), duration_seconds=seconds)
    assert candidate.status == CandidateStatus.ELIGIBLE.value, (
        f"{description} was rejected for {sorted(_critical(candidate))}"
    )


# ── one thing wrong, each ────────────────────────────────────────────

REJECTED = [
    ("digital silence", fx.silent, 12.0, Finding.SILENT_OUTPUT),
    ("audible only as noise", fx.near_silent, 12.0, Finding.NEAR_SILENT),
    ("content that stops dead", fx.early_collapse, 40.0, Finding.EARLY_COLLAPSE),
    ("distortion in the samples", fx.severely_clipped, 12.0, Finding.SEVERE_CLIPPING),
    ("channels inverted against each other", fx.anti_phase, 12.0, Finding.PHASE_UNSAFE),
    ("one tone rather than a mix", fx.spectral_collapse, 12.0, Finding.SPECTRAL_COLLAPSE),
    ("a channel effectively absent", fx.channel_imbalance, 12.0, Finding.CHANNEL_IMBALANCE),
    ("an offset eating the headroom", fx.dc_offset, 12.0, Finding.DC_OFFSET),
    ("three seconds when thirty were asked for", fx.truncated, 30.0, Finding.DURATION_SHORT),
]


@pytest.mark.parametrize(
    ("description", "make", "seconds", "expected"), REJECTED, ids=[case[0] for case in REJECTED]
)
def test_a_broken_candidate_is_rejected_for_the_right_reason(
    audio_dir, description, make, seconds, expected
):
    candidate = _judge(make(audio_dir / "case.wav"), duration_seconds=seconds)
    assert candidate.status == CandidateStatus.REJECTED.value, f"{description} was accepted"
    assert expected.value in _critical(candidate)


def test_a_file_that_is_not_audio_is_rejected_without_a_measurement(audio_dir):
    candidate = _judge(fx.undecodable(audio_dir / "broken.wav"), duration_seconds=12.0)
    assert _critical(candidate) == {Finding.INVALID_AUDIO.value}
    # Nothing measured means nothing to score, and nothing invented.
    assert candidate.technical_selection_score is None
    assert candidate.duration_seconds is None


def test_an_empty_file_is_rejected(audio_dir):
    candidate = _judge(fx.empty(audio_dir / "empty.wav"), duration_seconds=12.0)
    assert candidate.status == CandidateStatus.REJECTED.value


def test_a_quiet_track_and_a_gappy_one_get_different_codes(audio_dir):
    """One code covering both would make a trace unreadable.

    A file mastered 40 dB down and a file with long structured gaps are
    different problems with different fixes, and the second one is not a
    rejection at all — rejecting it needs the positional evidence the
    collapse detector provides.
    """
    quiet = _judge(fx.near_silent(audio_dir / "quiet.wav"), duration_seconds=12.0)
    assert Finding.NEAR_SILENT.value in _critical(quiet)
    assert Finding.EXCESSIVE_SILENCE.value not in quiet.finding_codes()

    gappy = _judge(fx.early_collapse(audio_dir / "gappy.wav", 40.0, 12.0), duration_seconds=40.0)
    excessive = next(
        finding for finding in gappy.findings if finding.code == Finding.EXCESSIVE_SILENCE.value
    )
    assert excessive.severity == Severity.MAJOR.value
    assert excessive.code not in _critical(gappy)


# ── duration: the difference between wrong and a near miss ───────────


def test_a_near_miss_on_duration_is_recorded_and_not_rejected(audio_dir):
    """12s against a 12.5s request. Recorded, ranked on, delivered."""
    candidate = _judge(fx.healthy(audio_dir / "near.wav", 12.0), duration_seconds=12.5)
    assert candidate.status == CandidateStatus.ELIGIBLE.value


def test_a_duration_far_past_the_request_is_rejected(audio_dir):
    candidate = _judge(fx.healthy(audio_dir / "long.wav", 30.0), duration_seconds=12.0)
    assert Finding.DURATION_LONG.value in _critical(candidate)


def test_no_duration_expectation_means_no_duration_finding(audio_dir):
    """An edit states no duration, so QC may not invent one to fail."""
    candidate = _judge(fx.healthy(audio_dir / "any.wav", 3.0))
    assert candidate.status == CandidateStatus.ELIGIBLE.value
    assert (
        not {Finding.DURATION_SHORT.value, Finding.DURATION_LONG.value} & candidate.finding_codes()
    )


# ── control adherence ────────────────────────────────────────────────


def test_a_tempo_that_matches_the_request_is_not_a_mismatch(audio_dir):
    candidate = _judge(fx.at_tempo(audio_dir / "beat.wav", 120.0), duration_seconds=16.0, bpm=120)
    assert Finding.CONTROL_BPM_MISMATCH.value not in candidate.finding_codes()


def test_a_tempo_nowhere_near_the_request_is_a_mismatch(audio_dir):
    candidate = _judge(fx.at_tempo(audio_dir / "beat.wav", 120.0), duration_seconds=16.0, bpm=170)
    assert Finding.CONTROL_BPM_MISMATCH.value in candidate.finding_codes()
    # Measurably wrong, and still deliverable: a song at the wrong tempo
    # is a song. It loses a ranking; it is not thrown away.
    assert candidate.status == CandidateStatus.ELIGIBLE.value


def test_an_estimate_below_the_confidence_floor_is_never_called_a_mismatch(audio_dir):
    """Two guesses disagreeing is not evidence that the request was missed.

    Phase 23's estimator reports a tempo for material with no pulse at
    all, so the comparison is gated on confidence. The gate is exercised
    directly rather than through a fixture chosen for having a weak
    estimate — a fixture's confidence is a property of the fixture, and
    one that drifted would silently stop testing this.
    """
    measurement = measure(fx.at_tempo(audio_dir / "beat.wav", 120.0), measure_musical=True)
    unsure = replace(measurement, bpm_confidence=t.BPM_CONFIDENCE_FLOOR - 0.01)

    findings = check_bpm(unsure, RequestExpectation(bpm=170))

    assert [finding.code for finding in findings] == [Finding.CONTROL_NOT_MEASURABLE.value]
    assert findings[0].not_measurable is True
    assert findings[0].severity == Severity.INFO.value


def test_asking_for_nothing_measurable_produces_no_control_findings(audio_dir):
    candidate = _judge(fx.healthy(audio_dir / "plain.wav"), duration_seconds=12.0)
    assert Finding.CONTROL_BPM_MISMATCH.value not in candidate.finding_codes()
    assert Finding.CONTROL_KEY_MISMATCH.value not in candidate.finding_codes()


# ── vocals: the honest answer is that this repository cannot tell ────


def test_a_vocal_request_is_reported_as_unknown_rather_than_guessed(audio_dir):
    """No validated detector exists, so no verdict is fabricated."""
    candidate = _judge(
        fx.healthy(audio_dir / "song.wav"), duration_seconds=12.0, instrumental=False
    )
    assert Finding.CONTROL_VOCAL_UNKNOWN.value in candidate.finding_codes()
    assert Finding.CONTROL_VOCAL_MISMATCH.value not in candidate.finding_codes()


def test_an_unknown_vocal_verdict_never_rejects(audio_dir):
    candidate = _judge(fx.healthy(audio_dir / "song.wav"), duration_seconds=12.0, instrumental=True)
    assert candidate.status == CandidateStatus.ELIGIBLE.value
    unknown = next(
        finding
        for finding in candidate.findings
        if finding.code == Finding.CONTROL_VOCAL_UNKNOWN.value
    )
    assert unknown.severity != Severity.CRITICAL.value
    assert unknown.not_measurable is True
