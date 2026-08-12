"""Global verdicts and two-stage triage.

The Phase 5 evaluator rejected a whole baseline without per-track
scoring. These tests pin the mechanism that records that honestly —
and, importantly, that it never manufactures per-track numbers.
"""

import json
from pathlib import Path

import pytest

from bench.verdict import (
    HUMAN_FAILURE_TAGS,
    TRIAGE_DETAIL_THRESHOLD,
    TRIAGE_SCALE,
    GlobalVerdict,
    VerdictError,
    VerdictStore,
    validate_failure_tags,
    validate_triage,
    validate_verdict,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _verdict(**overrides) -> GlobalVerdict:
    base = dict(
        baseline_id="LUBER_BASELINE_P5_V1",
        evaluator="product-owner",
        recorded_at="2026-08-12",
        tracks_reviewed=26,
        tracks_accepted=0,
        tracks_rejected=26,
        overall_score=2,
        commercially_usable=False,
        reason="quality below detailed-scoring threshold",
        findings=["excessive_sibilance"],
    )
    base.update(overrides)
    return GlobalVerdict(**base)  # type: ignore[arg-type]


# ── the recorded Phase 5 verdict ──────────────────────────────────────


def test_the_real_phase5_verdict_is_recorded_and_intact():
    """The shipped record must say exactly what the evaluator said."""
    path = REPO_ROOT / "benchmarks" / "music_quality" / "listening" / "verdicts.jsonl"
    entries = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    verdict = next(v for v in entries if v["baseline_id"] == "LUBER_BASELINE_P5_V1")

    assert verdict["overall_score"] == 2
    assert verdict["tracks_reviewed"] == 26
    assert verdict["tracks_accepted"] == 0
    assert verdict["tracks_rejected"] == 26
    assert verdict["commercially_usable"] is False
    assert verdict["reason"] == "quality below detailed-scoring threshold"
    assert set(verdict["findings"]) == {
        "frequency_balance_poor",
        "instrument_fidelity_poor",
        "audio_texture_poor",
        "excessive_high_frequency",
        "excessive_sibilance",
        "korean_lyric_sentence_omission",
        "lyric_line_skipping",
        "unwanted_trot_vocal_character",
        "contemporary_vocal_style_miss",
    }


def test_no_per_track_scores_were_fabricated():
    """26 tracks were rejected; there must not be 26 invented score rows."""
    path = REPO_ROOT / "benchmarks" / "music_quality" / "listening" / "scores.jsonl"
    entries = (
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if path.is_file()
        else []
    )
    # Only genuinely submitted evaluations may exist, and far fewer than
    # the 26 reviewed tracks.
    assert len(entries) < 26
    for entry in entries:
        assert entry.get("evaluator")
        assert entry.get("scored_at")


# ── verdict validation ────────────────────────────────────────────────


def test_valid_verdict_passes():
    validate_verdict(_verdict())


def test_counts_must_reconcile():
    with pytest.raises(VerdictError, match="must equal reviewed"):
        validate_verdict(_verdict(tracks_accepted=5, tracks_rejected=26))


def test_score_range_enforced():
    for bad in (0, 11, -3):
        with pytest.raises(VerdictError, match="outside 1-10"):
            validate_verdict(_verdict(overall_score=bad))


def test_reason_is_required():
    with pytest.raises(VerdictError, match="state its reason"):
        validate_verdict(_verdict(reason="   "))


def test_cannot_claim_usable_while_rejecting_everything():
    """An internally contradictory verdict must not be storable."""
    with pytest.raises(VerdictError, match="no track was accepted"):
        validate_verdict(_verdict(commercially_usable=True))


def test_negative_counts_rejected():
    with pytest.raises(VerdictError):
        validate_verdict(_verdict(tracks_accepted=-1, tracks_rejected=27))


def test_acceptance_rate():
    assert _verdict().acceptance_rate == 0.0
    assert _verdict(tracks_accepted=13, tracks_rejected=13).acceptance_rate == 0.5


# ── two-stage triage ──────────────────────────────────────────────────


def test_triage_scale_covers_1_to_10():
    assert set(TRIAGE_SCALE) == set(range(1, 11))
    assert TRIAGE_SCALE[1] == "unusable"
    assert TRIAGE_SCALE[8] == "commercially releasable"


def test_low_score_does_not_require_the_full_rubric():
    """The point of triage: do not make evaluators grade unusable audio."""
    validate_triage(2, detailed_scores=None, reject_tags=["EXCESSIVE_SIBILANCE"])


def test_low_score_still_requires_a_reason():
    with pytest.raises(VerdictError, match="at least one reject reason"):
        validate_triage(2, detailed_scores=None, reject_tags=[])


def test_score_at_threshold_requires_detailed_rubric():
    with pytest.raises(VerdictError, match="full rubric is required"):
        validate_triage(TRIAGE_DETAIL_THRESHOLD, detailed_scores=None)
    validate_triage(TRIAGE_DETAIL_THRESHOLD, detailed_scores={"harmony": 5})


def test_high_score_requires_detailed_rubric():
    with pytest.raises(VerdictError, match="full rubric is required"):
        validate_triage(8, detailed_scores=None, reject_tags=["MIX_MUDDY"])


def test_triage_score_range_enforced():
    with pytest.raises(VerdictError, match="outside 1-10"):
        validate_triage(0, reject_tags=["MIX_MUDDY"])


# ── failure tags ──────────────────────────────────────────────────────


def test_phase5_human_findings_have_tags():
    """Every human finding must be expressible as a tag."""
    for tag in (
        "FREQUENCY_BALANCE_BAD",
        "HIGH_END_OVERBOOST",
        "EXCESSIVE_SIBILANCE",
        "INSTRUMENT_FIDELITY_LOW",
        "TEXTURE_LOW_QUALITY",
        "KOREAN_LINE_OMISSION",
        "LYRIC_LINE_SKIP",
        "TROT_LIKE_VOCAL",
        "VOCAL_STYLE_OUTDATED",
    ):
        assert tag in HUMAN_FAILURE_TAGS


def test_new_tags_extend_rather_than_replace_the_artifact_taxonomy():
    from bench.scoring import ARTIFACT_TAGS

    validate_failure_tags(["TROT_LIKE_VOCAL", "MIX_MUDDY"])
    assert not set(HUMAN_FAILURE_TAGS) & set(ARTIFACT_TAGS)


def test_unknown_tags_rejected():
    with pytest.raises(VerdictError, match="unknown failure tags"):
        validate_failure_tags(["SOUNDS_A_BIT_OFF"])


# ── store ─────────────────────────────────────────────────────────────


def test_store_roundtrip_and_latest(tmp_path):
    store = VerdictStore(tmp_path / "v.jsonl")
    store.append(_verdict(overall_score=2))
    store.append(_verdict(overall_score=4, tracks_accepted=1, tracks_rejected=25))
    assert store.latest_for("LUBER_BASELINE_P5_V1")["overall_score"] == 4
    assert store.latest_for("NOPE") is None


def test_store_refuses_an_invalid_verdict(tmp_path):
    store = VerdictStore(tmp_path / "v.jsonl")
    with pytest.raises(VerdictError):
        store.append(_verdict(overall_score=99))
    assert store.load() == []


def test_store_preserves_korean_notes(tmp_path):
    store = VerdictStore(tmp_path / "v.jsonl")
    store.append(_verdict(notes="뽕끼 있는 보컬"))
    assert "뽕끼" in store.load()[0]["notes"]
