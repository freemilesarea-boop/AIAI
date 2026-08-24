"""Tiering: a ranking inside a library, and never a judgement of music.

Two things matter here. The scoring has to actually track the features
it claims to — a brighter population should rank higher on HIGH_END —
and the VOCAL axis has to stay unmeasured, because nothing in this
repository can tell a sung note from a lead synth.
"""

import pytest

from luber_dataset.audio_features import AudioFeatures
from luber_dataset.quality_tiers import (
    AXIS_FEATURES,
    QualityAxis,
    QualityTier,
    classify_population,
    score_population,
    tier_summary,
)


def _features(**overrides) -> AudioFeatures:
    base = {
        "duration_seconds": 180.0,
        "sample_rate": 48_000,
        "channels": 2,
        "high_frequency_energy_ratio": 0.01,
        "spectral_centroid_hz": 800.0,
        "high_band_rms_db": -40.0,
        "rms_db": -16.0,
        "transient_density_per_second": 3.0,
        "onset_density_per_second": 5.0,
        "beat_stability": 0.4,
        "tempo_consistency": 0.9,
        "tempo_bpm": 90.0,
        "drum_bass_alignment": 0.2,
        "layer_density": 0.35,
        "active_band_fraction": 0.2,
    }
    base.update(overrides)
    return AudioFeatures(**base)


def _population(count: int = 20):
    return [
        (
            f"t{i:02d}",
            _features(
                high_frequency_energy_ratio=0.001 * (i + 1),
                spectral_centroid_hz=300.0 + 60.0 * i,
                high_band_rms_db=-60.0 + i,
                beat_stability=0.2 + 0.02 * i,
                tempo_consistency=0.7 + 0.01 * i,
                drum_bass_alignment=0.05 * (i + 1) / 4,
                layer_density=0.25 + 0.01 * i,
                onset_density_per_second=4.0 + 0.1 * i,
                active_band_fraction=0.05 * i,
            ),
        )
        for i in range(count)
    ]


class TestScoring:
    def test_a_brighter_track_ranks_higher_on_high_end(self):
        scores = score_population(_population())
        assert scores["t19"].high_end > scores["t00"].high_end

    def test_a_steadier_track_ranks_higher_on_rhythm(self):
        scores = score_population(_population())
        assert scores["t19"].rhythm > scores["t00"].rhythm

    def test_a_busier_track_ranks_higher_on_arrangement(self):
        scores = score_population(_population())
        assert scores["t19"].arrangement > scores["t00"].arrangement

    def test_every_score_is_a_rank_inside_the_unit_interval(self):
        for score in score_population(_population()).values():
            for value in (score.high_end, score.rhythm, score.arrangement, score.combined):
                assert 0.0 <= value <= 1.0

    def test_a_population_of_one_sits_in_the_middle(self):
        """A single track is neither above nor below anything."""
        scores = score_population([("only", _features())])
        assert scores["only"].combined == pytest.approx(0.5)

    def test_an_empty_population_scores_nothing(self):
        assert score_population([]) == {}


class TestTheVocalAxis:
    def test_it_is_never_given_a_number(self):
        scores = score_population(_population())
        for score in scores.values():
            assert score.vocal is None
            assert score.to_dict()["VOCAL"] is None

    def test_the_dictionary_says_why(self):
        payload = next(iter(score_population(_population()).values())).to_dict()
        assert "not scored" in payload["vocal_note"]

    def test_no_measurable_axis_feeds_it(self):
        assert QualityAxis.VOCAL.value not in AXIS_FEATURES

    def test_it_still_exists_in_the_vocabulary(self):
        """The listening evaluation is organised around it."""
        assert QualityAxis.VOCAL.value == "VOCAL"


class TestTiers:
    def test_the_best_material_lands_in_tier_a(self):
        assignments = {a.item_id: a for a in classify_population(_population())}
        assert assignments["t19"].tier == QualityTier.TIER_A.value
        assert assignments["t00"].tier == QualityTier.TIER_C.value

    def test_every_track_gets_a_tier(self):
        assignments = classify_population(_population())
        assert len(assignments) == 20
        assert all(a.tier in {t.value for t in QualityTier} for a in assignments)

    def test_the_thresholds_are_reported_with_the_decision(self):
        """A chosen cutoff must travel with the thing it decided."""
        assignment = classify_population(_population())[0]
        assert "tier A at" in assignment.detail
        assert "not a judgement of the music" in assignment.detail

    def test_a_tier_is_relative_to_the_population_it_was_ranked_in(self):
        small = classify_population(_population(4))
        assert any(a.tier == QualityTier.TIER_A.value for a in small)
        assert "population of 4" in small[0].detail

    def test_the_summary_counts_by_tier_and_group(self):
        population = _population()
        groups = {f"t{i:02d}": ("POP" if i % 2 else "Lofi") for i in range(20)}
        summary = tier_summary(classify_population(population, groups=groups))
        assert sum(summary["by_tier"].values()) == 20
        assert set(summary["by_group"]) == {"POP", "Lofi"}

    def test_no_tier_name_claims_quality(self):
        forbidden = ("GOOD", "BEST", "HIGH_QUALITY", "PREMIUM")
        for tier in QualityTier:
            assert not any(word in tier.value for word in forbidden)
