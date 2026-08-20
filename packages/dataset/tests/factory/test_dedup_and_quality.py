"""Deduplication, fingerprinting, musical estimation, and QC tiers.

The dedup tests are written around one asymmetry: a false merge deletes
a distinct track silently, a false split leaves one extra. So the tests
that matter most are the ones asserting the factory *refuses* to decide
— that a near match goes to review rather than being folded away.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import RATE, pulse, requires_ffmpeg, tone, transcode, write_wav

from luber_audio_finishing import load_audio
from luber_dataset.factory.audio_analysis import TechnicalAnalysis
from luber_dataset.factory.config import DedupThresholds, QualityThresholds
from luber_dataset.factory.decoder import DecodeResult, DecodeStatus
from luber_dataset.factory.dedup import (
    DedupDecision,
    DuplicateType,
    _Candidate,
    analyse_duplicates,
    compute_fingerprint,
    similarity,
)
from luber_dataset.factory.musical import (
    MIN_KEY_CONFIDENCE,
    estimate_key,
    estimate_tempo,
)
from luber_dataset.factory.musical import (
    analyse as analyse_music,
)
from luber_dataset.factory.quality import QualityTier, evaluate, meets_tier


def fingerprint_of(path: Path) -> str | None:
    loaded = load_audio(path)
    return compute_fingerprint(loaded.mono(), loaded.sample_rate)


class TestFingerprint:
    def test_the_same_audio_fingerprints_identically(self, tmp_path: Path):
        samples = tone(seed=7)
        one = write_wav(tmp_path / "a.wav", samples)
        two = write_wav(tmp_path / "b.wav", samples)
        assert fingerprint_of(one) == fingerprint_of(two)

    def test_different_audio_fingerprints_differently(self, tmp_path: Path):
        a = write_wav(tmp_path / "a.wav", tone(frequency=220.0, seed=1))
        b = write_wav(tmp_path / "b.wav", tone(frequency=550.0, seed=2))
        assert similarity(fingerprint_of(a), fingerprint_of(b)) < 0.9

    @requires_ffmpeg
    def test_a_lossless_transcode_is_identical(self, tmp_path: Path):
        """FLAC is the same samples in another container, and scores 1.0.

        This is the only re-encode the factory merges automatically.
        """
        source = write_wav(tmp_path / "source.wav", tone(seed=3))
        encoded = transcode(source, tmp_path / "same.flac")
        assert similarity(fingerprint_of(source), fingerprint_of(encoded)) == 1.0

    @requires_ffmpeg
    def test_a_lossy_re_encode_lands_in_the_review_band(self, tmp_path: Path):
        """Detected, and deliberately not merged.

        Measured across three signals and four lossy settings, same-audio
        scores span 0.879-0.991 while unrelated pairs reach 0.900 — the
        ranges overlap, so no threshold separates them. A lossy pair is
        therefore reported to a person rather than decided here.
        """
        thresholds = DedupThresholds()
        source = write_wav(tmp_path / "source.wav", tone(seed=3))
        encoded = transcode(source, tmp_path / "same.mp3", "-codec:a", "libmp3lame", "-b:a", "192k")
        score = similarity(fingerprint_of(source), fingerprint_of(encoded))
        assert score is not None
        assert score >= thresholds.near_audio_similarity, f"missed entirely: {score}"
        assert score < thresholds.exact_audio_similarity, f"would auto-merge: {score}"

    def test_a_volume_change_does_not_change_it(self, tmp_path: Path):
        """The bits are second differences, so level cancels out."""
        samples = tone(seed=5)
        quiet = write_wav(tmp_path / "quiet.wav", samples * 0.5)
        loud = write_wav(tmp_path / "loud.wav", samples)
        score = similarity(fingerprint_of(quiet), fingerprint_of(loud))
        assert score is not None and score > 0.95, score

    def test_audio_too_short_gets_no_fingerprint(self):
        """Silence beats a meaningless hash that would collide."""
        assert compute_fingerprint(np.zeros(1000), RATE) is None

    def test_a_missing_fingerprint_is_not_a_negative(self):
        """`None` means unknown, and unknown must not read as different."""
        assert similarity(None, "abcd") is None
        assert similarity("abcd", None) is None


class TestDuplicateDetection:
    def candidate(self, name: str, digest: str, fingerprint: str | None, duration=180.0):
        return _Candidate(
            track_id=f"trk_{name}",
            sha256=digest,
            source_path=f"/library/{name}.wav",
            fingerprint=fingerprint,
            duration_seconds=duration,
        )

    def test_identical_bytes_form_one_canonical_track(self):
        items = [
            self.candidate("a", "hash1", "aa" * 200),
            self.candidate("b", "hash1", "aa" * 200),
        ]
        records = analyse_duplicates(items, DedupThresholds())
        assert len({r.canonical_track_id for r in records.values()}) == 1
        assert records["trk_a"].duplicate_type == DuplicateType.EXACT_FILE.value

    def test_every_source_path_is_retained(self):
        """Nothing is deleted, and nothing is forgotten."""
        items = [
            self.candidate("a", "hash1", "aa" * 200),
            self.candidate("b", "hash1", "aa" * 200),
        ]
        records = analyse_duplicates(items, DedupThresholds())
        assert records["trk_a"].all_source_paths == ["/library/a.wav", "/library/b.wav"]

    def test_the_canonical_choice_is_deterministic(self):
        items = [
            self.candidate("z", "hash1", "aa" * 200),
            self.candidate("a", "hash1", "aa" * 200),
        ]
        first = analyse_duplicates(items, DedupThresholds())
        second = analyse_duplicates(list(reversed(items)), DedupThresholds())
        assert first["trk_a"].canonical_track_id == second["trk_a"].canonical_track_id

    def test_near_identical_audio_is_merged(self):
        matching = "ff" * 200
        items = [
            self.candidate("a", "hash1", matching),
            self.candidate("b", "hash2", matching),
        ]
        records = analyse_duplicates(items, DedupThresholds())
        assert records["trk_b"].duplicate_type == DuplicateType.EXACT_AUDIO.value
        assert records["trk_b"].dedup_decision == DedupDecision.MERGED.value

    def test_a_near_match_goes_to_review_not_to_a_merge(self):
        """The refusal that matters. Uncertainty is not a decision."""
        left = np.zeros(1600, dtype=np.uint8)
        right = left.copy()
        right[:80] = 1  # 95% agreement: suspicious, not certain
        items = [
            self.candidate("a", "h1", np.packbits(left).tobytes().hex()),
            self.candidate("b", "h2", np.packbits(right).tobytes().hex()),
        ]
        records = analyse_duplicates(items, DedupThresholds())
        assert records["trk_b"].dedup_decision == DedupDecision.REVIEW_REQUIRED.value
        assert records["trk_b"].duplicate_type == DuplicateType.NEAR_AUDIO.value
        assert records["trk_b"].canonical_track_id == "trk_b", "a review is not a merge"

    def test_different_durations_are_never_paired(self):
        """Two songs can share a spectrum; rarely at the same length."""
        matching = "ff" * 200
        items = [
            self.candidate("a", "h1", matching, duration=180.0),
            self.candidate("b", "h2", matching, duration=240.0),
        ]
        records = analyse_duplicates(items, DedupThresholds())
        assert records["trk_b"].dedup_decision == DedupDecision.KEEP.value

    def test_an_unknown_duration_does_not_clear_the_gate(self):
        matching = "ff" * 200
        items = [
            self.candidate("a", "h1", matching, duration=None),
            self.candidate("b", "h2", matching, duration=180.0),
        ]
        records = analyse_duplicates(items, DedupThresholds())
        assert records["trk_b"].dedup_decision == DedupDecision.KEEP.value

    def test_unrelated_tracks_are_left_alone(self):
        items = [
            self.candidate("a", "h1", "00" * 200),
            self.candidate("b", "h2", "ff" * 200),
        ]
        records = analyse_duplicates(items, DedupThresholds())
        assert all(r.dedup_decision == DedupDecision.KEEP.value for r in records.values())


class TestMusicalAnalysis:
    @pytest.mark.parametrize("bpm", [90, 120, 140])
    def test_a_known_tempo_is_recovered(self, bpm: int):
        """The estimator earns its place by finding what it wasn't told."""
        samples = pulse(bpm=bpm)[:, 0]
        estimated, confidence = estimate_tempo(samples, RATE)
        assert estimated is not None and confidence is not None
        assert abs(estimated - bpm) < 4, f"{bpm} -> {estimated}"
        assert confidence > 0.5

    def test_the_octave_error_is_resolved(self):
        """Unweighted autocorrelation reports 120 BPM as 60."""
        estimated, _ = estimate_tempo(pulse(bpm=120)[:, 0], RATE)
        assert estimated is not None
        assert estimated > 90, f"reported {estimated}: half-tempo octave error"

    def test_a_known_key_is_recovered(self):
        t = np.arange(int(20 * RATE)) / RATE
        signal = np.zeros_like(t)
        for midi in (60, 64, 67, 72, 62, 65, 69, 71):  # C major material
            signal += np.sin(2 * np.pi * 440.0 * 2 ** ((midi - 69) / 12) * t)
        key, mode, confidence = estimate_key(signal / np.abs(signal).max(), RATE)
        assert key == "C"
        assert mode == "major"
        assert confidence is not None and confidence >= MIN_KEY_CONFIDENCE

    def test_structure_is_never_invented(self):
        """The honesty test. No segmenter exists, so none is claimed."""
        result = analyse_music(tone()[:, 0], RATE)
        assert result.estimated_structure is None
        assert result.structure_status == "UNAVAILABLE"
        assert "estimated_structure" in result.unavailable

    def test_a_downbeat_is_not_claimed_from_a_period(self):
        """Tempo gives the beat period, not its phase."""
        result = analyse_music(pulse(bpm=120)[:, 0], RATE)
        assert result.estimated_downbeat_seconds is None
        assert "estimated_downbeat_seconds" in result.unavailable

    def test_silence_produces_no_tempo_rather_than_a_guess(self):
        result = analyse_music(np.zeros(RATE * 5), RATE)
        assert result.bpm is None
        assert "bpm" in result.unavailable


class TestQualityControl:
    def valid(self, **overrides) -> TechnicalAnalysis:
        base = {
            "duration_seconds": 180.0,
            "sample_rate": 44_100,
            "channels": 2,
            "peak_dbfs": -1.0,
            "rms_dbfs": -14.0,
            "crest_factor_db": 13.0,
            "dc_offset": 0.0001,
            "silence_ratio": 0.02,
            "clipping_sample_ratio": 0.0,
            "integrated_lufs": -14.0,
            "phase_correlation": 0.8,
            "high_frequency_cutoff_hz": 20_000.0,
        }
        return TechnicalAnalysis(**{**base, **overrides})

    def decoded(self) -> DecodeResult:
        return DecodeResult(status=DecodeStatus.VALID, duration_seconds=180.0)

    def test_a_healthy_track_reaches_tier_a(self):
        result = evaluate(self.decoded(), self.valid(), QualityThresholds())
        assert result.quality_flags == []
        assert result.quality_tier == QualityTier.A.value
        assert result.quality_score == 1.0

    def test_a_corrupt_file_is_rejected_whatever_else_holds(self):
        result = evaluate(
            DecodeResult(status=DecodeStatus.INVALID, decode_error="broken"),
            self.valid(),
            QualityThresholds(),
        )
        assert "CORRUPT" in result.quality_flags
        assert result.quality_tier == QualityTier.REJECT.value

    @pytest.mark.parametrize(
        ("overrides", "flag"),
        [
            ({"duration_seconds": 5.0}, "TOO_SHORT"),
            ({"duration_seconds": 5_000.0}, "TOO_LONG"),
            ({"sample_rate": 22_050}, "LOW_SAMPLE_RATE"),
            ({"channels": 1}, "MONO"),
            ({"clipping_sample_ratio": 0.02}, "CLIPPING"),
            ({"crest_factor_db": 3.0}, "LOW_DYNAMIC_RANGE"),
            ({"dc_offset": 0.2}, "DC_OFFSET"),
            ({"integrated_lufs": -2.0}, "EXTREME_LOUDNESS"),
            ({"silence_ratio": 0.9}, "EXTREME_SILENCE"),
            ({"phase_correlation": -0.5}, "PHASE_RISK"),
            ({"high_frequency_cutoff_hz": 11_000.0}, "SUSPICIOUS_BANDWIDTH"),
        ],
    )
    def test_each_condition_raises_its_own_flag(self, overrides: dict, flag: str):
        result = evaluate(self.decoded(), self.valid(**overrides), QualityThresholds())
        assert flag in result.quality_flags

    def test_a_flag_is_not_automatically_fatal(self):
        """Plenty of excellent recordings are mono."""
        result = evaluate(self.decoded(), self.valid(channels=1), QualityThresholds())
        assert "MONO" in result.quality_flags
        assert result.quality_tier != QualityTier.REJECT.value

    def test_thresholds_are_configurable(self):
        lenient = QualityThresholds(min_sample_rate=8_000)
        result = evaluate(self.decoded(), self.valid(sample_rate=22_050), lenient)
        assert "LOW_SAMPLE_RATE" not in result.quality_flags

    def test_nothing_is_rejected_without_a_reason(self):
        result = evaluate(self.decoded(), self.valid(duration_seconds=2.0), QualityThresholds())
        assert result.quality_tier == QualityTier.REJECT.value
        assert result.reasons, "a rejection must say why"

    def test_a_near_duplicate_is_flagged_but_not_rejected(self):
        result = evaluate(self.decoded(), self.valid(), QualityThresholds(), near_duplicate=True)
        assert "NEAR_DUPLICATE" in result.quality_flags
        assert result.quality_tier != QualityTier.REJECT.value

    def test_tier_ordering(self):
        assert meets_tier("A", "B")
        assert meets_tier("B", "B")
        assert not meets_tier("C", "B")
        assert not meets_tier("REJECT", "C")
