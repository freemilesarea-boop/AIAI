"""The order of the four steps, which is the safety property.

Measure, check, gate, score — and score *only* what survived the gate.
A caller that scored first and gated second would produce a number for a
broken candidate, and a number that exists is a number something will
eventually compare.
"""

from __future__ import annotations

import qc_fixtures as fx

from luber_inference_qc import (
    CandidateGeneration,
    CandidateStatus,
    MeasurementCache,
    RequestExpectation,
    judge,
)
from luber_inference_qc.scoring import WEIGHTS
from luber_inference_qc.versions import QC_ENGINE_VERSION


def _candidate(index: int = 0) -> CandidateGeneration:
    return CandidateGeneration(
        candidate_id=f"cand_{index:02d}",
        generation_id="gen",
        attempt_index=index,
        request_sha256="digest",
    )


def test_an_eligible_candidate_is_scored_with_its_parts_shown(audio_dir):
    candidate = _candidate()
    judge(candidate, fx.healthy(audio_dir / "ok.wav"), RequestExpectation(duration_seconds=12.0))

    assert candidate.status == CandidateStatus.ELIGIBLE.value
    assert candidate.technical_selection_score is not None
    assert 0.0 <= candidate.technical_selection_score <= 1.0
    # Every weighted component is present, so a ranking can be argued
    # with rather than only read.
    assert set(candidate.score_components) == set(WEIGHTS)


def test_a_rejected_candidate_is_given_no_score_at_all(audio_dir):
    """Not a low score. None. There is nothing for a later comparison
    to pick up and treat as a number."""
    candidate = _candidate()
    judge(
        candidate, fx.silent(audio_dir / "nothing.wav"), RequestExpectation(duration_seconds=12.0)
    )

    assert candidate.status == CandidateStatus.REJECTED.value
    assert candidate.technical_selection_score is None
    assert candidate.score_components == {}


def test_nothing_here_claims_to_judge_whether_the_song_is_good(audio_dir):
    """The line the whole package is built to keep visible."""
    candidate = _candidate()
    judge(candidate, fx.healthy(audio_dir / "ok.wav"), RequestExpectation(duration_seconds=12.0))

    forbidden = ("quality", "musical", "naturalness", "melody", "commercial", "taste")
    for name in candidate.score_components:
        assert not any(word in name for word in forbidden), name


def test_the_measurement_records_what_it_measured(audio_dir):
    candidate = _candidate()
    measurement = judge(
        candidate, fx.healthy(audio_dir / "ok.wav", 12.0), RequestExpectation(duration_seconds=12.0)
    )

    assert measurement is not None
    assert candidate.raw_sha256 == measurement.sha256
    assert candidate.sample_rate == fx.SAMPLE_RATE
    assert candidate.channels == fx.CHANNELS
    assert 11.5 < (candidate.duration_seconds or 0) < 12.5
    assert (candidate.qc_seconds or 0) > 0


def test_a_decode_failure_returns_no_measurement_to_derive_findings_from(audio_dir):
    candidate = _candidate()
    assert judge(candidate, fx.undecodable(audio_dir / "bad.wav"), RequestExpectation()) is None
    assert candidate.status == CandidateStatus.REJECTED.value


# ── the cache ────────────────────────────────────────────────────────


def test_the_same_bytes_are_not_measured_twice(audio_dir):
    """Re-measuring the winner before delivery would double the QC cost
    of every generation for an answer already computed."""
    cache = MeasurementCache()
    path = fx.healthy(audio_dir / "ok.wav")
    expectation = RequestExpectation(duration_seconds=12.0)

    first = _candidate(0)
    judge(first, path, expectation, cache=cache)
    assert len(cache) == 1

    second = _candidate(1)
    judge(second, path, expectation, cache=cache, sha256=first.raw_sha256)

    assert len(cache) == 1
    assert second.technical_selection_score == first.technical_selection_score


def test_the_cache_key_carries_the_engine_version(audio_dir):
    """A changed threshold must not be answered from a cache written
    before it changed."""
    cache = MeasurementCache()
    candidate = _candidate()
    judge(candidate, fx.healthy(audio_dir / "ok.wav"), RequestExpectation(), cache=cache)
    assert candidate.raw_sha256 is not None
    assert QC_ENGINE_VERSION in cache._key(candidate.raw_sha256)
