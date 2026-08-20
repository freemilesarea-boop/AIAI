"""Distributions, missingness, concentration, and what a finding needs.

The tests that matter most here are negative: that a share never hides
its denominator, and that no gap is reported without a target declaring
one. Both failures produce output that looks like analysis and is not.
"""

from __future__ import annotations

import pytest
from conftest import dominated_by_one_artist, record, sparse_genre

from luber_dataset.factory.intelligence import concentration, distributions, findings, targets
from luber_dataset.factory.intelligence import profile as profile_module
from luber_dataset.factory.intelligence.schemas import Severity, TrackView


def views(records, *, min_confidence: float = 0.55) -> list[TrackView]:
    return [TrackView(r, min_confidence=min_confidence) for r in records]


def build(records, **kwargs) -> profile_module.DatasetProfile:
    return profile_module.build(views(records), population="test", **kwargs)


class TestDistributions:
    def test_counts_and_shares_over_known_values(self):
        profile = build(
            [record("a", language="ko"), record("b", language="ko"), record("c", language="en")]
        )
        language = profile.categorical["language"]
        assert language.known_count == 3
        assert language.share("ko") == pytest.approx(2 / 3)

    def test_unknowns_are_excluded_from_the_denominator(self):
        """The failure this whole layer exists to prevent.

        Two labelled tracks out of twenty is not "100% pop"; it is 100%
        of the tenth of the corpus anyone has looked at.
        """
        profile = build(sparse_genre(total=20, labelled=2))
        genre = profile.categorical["genre"]
        assert genre.known_count == 2
        assert genre.unknown_count == 18
        assert genre.share("pop") == 1.0, "share is over known values"
        assert genre.coverage == pytest.approx(0.1), "and coverage says how few those are"

    def test_unknown_is_never_a_category(self):
        profile = build(sparse_genre(total=10, labelled=1))
        assert "unknown" not in {b.label for b in profile.categorical["genre"].buckets}

    def test_duration_weighting_differs_from_counting(self):
        """A hundred sketches and ten long pieces are not equal exposure."""
        profile = build(
            [
                record("short1", language="ko", duration=30.0),
                record("short2", language="ko", duration=30.0),
                record("long", language="en", duration=600.0),
            ]
        )
        language = profile.categorical["language"]
        assert language.share("ko") == pytest.approx(2 / 3)
        assert language.share_by_duration("ko") == pytest.approx(60 / 660)

    def test_the_source_of_each_value_is_recorded(self):
        profile = build(
            [
                record("a", artist="From Sidecar"),
                record("b", embedded={"artist": "From Tag"}),
            ]
        )
        assert profile.categorical["artist"].source_breakdown == {"USER": 1, "EMBEDDED": 1}

    def test_a_sidecar_outranks_a_container_tag(self):
        view = TrackView(record("a", artist="Declared", embedded={"artist": "Tagged"}))
        assert view.artist().value == "Declared"
        assert view.artist().source == "USER"

    def test_numeric_quantiles(self):
        profile = build([record(f"t{i}", duration=float(i * 10 + 10)) for i in range(10)])
        summary = profile.numeric["duration_seconds"]
        assert summary.minimum == 10.0
        assert summary.maximum == 100.0
        assert summary.median == pytest.approx(55.0)

    def test_buckets_are_ordered_deterministically(self):
        records = [record(f"t{i}", language="ko" if i < 3 else "en") for i in range(5)]
        first = [b.label for b in build(records).categorical["language"].buckets]
        second = [b.label for b in build(list(reversed(records))).categorical["language"].buckets]
        assert first == second == ["ko", "en"]


class TestConfidenceGates:
    def test_a_low_confidence_tempo_is_not_a_fact(self):
        """Phase 23 reports a tempo for material with no pulse at all."""
        profile = build([record("a", bpm=50.0, bpm_confidence=0.2)])
        tempo = profile.categorical["tempo_bucket"]
        assert tempo.known_count == 0
        assert tempo.low_confidence_count == 1

    def test_a_confident_tempo_is_bucketed(self):
        profile = build([record("a", bpm=120.0, bpm_confidence=0.95)])
        assert profile.categorical["tempo_bucket"].share("110-130") == 1.0

    def test_the_gate_is_configurable(self):
        records = [record("a", bpm=120.0, bpm_confidence=0.4)]
        assert (
            profile_module.build(views(records, min_confidence=0.9), population="t")
            .categorical["tempo_bucket"]
            .known_count
            == 0
        )
        assert (
            profile_module.build(views(records, min_confidence=0.3), population="t")
            .categorical["tempo_bucket"]
            .known_count
            == 1
        )


class TestConcentration:
    def test_one_artist_dominating_is_visible_in_every_metric(self):
        profile = build(dominated_by_one_artist(total=20, dominant=12))
        metrics = profile.concentration["artist"]
        assert metrics.top1_label == "Dominant Artist"
        assert metrics.top1_share == pytest.approx(0.6)
        assert metrics.hhi > 0.35
        assert metrics.effective_categories < 3.0

    def test_effective_count_is_not_the_labelled_count(self):
        """The number that matters: how many it behaves as having."""
        profile = build(dominated_by_one_artist(total=20, dominant=12))
        metrics = profile.concentration["artist"]
        assert metrics.category_count == 9
        assert metrics.effective_categories < metrics.category_count / 2

    def test_an_even_distribution_scores_as_even(self):
        profile = build([record(f"t{i}", artist=f"Artist {i}") for i in range(10)])
        metrics = profile.concentration["artist"]
        assert metrics.hhi == pytest.approx(0.1)
        assert metrics.effective_categories == pytest.approx(10.0)
        assert metrics.normalized_entropy == pytest.approx(1.0)

    def test_duration_weighted_concentration_can_disagree(self):
        profile = build(
            [
                record("a", artist="Short", duration=10.0),
                record("b", artist="Short", duration=10.0),
                record("c", artist="Long", duration=1000.0),
            ]
        )
        assert profile.concentration["artist"].top1_label == "Short"
        assert profile.concentration_by_duration["artist"].top1_label == "Long"

    def test_a_single_category_reports_zero_normalized_entropy(self):
        """log(1) is 0; reporting a ratio would be a division by zero."""
        profile = build([record(f"t{i}", artist="Only") for i in range(4)])
        assert profile.concentration["artist"].normalized_entropy == 0.0

    def test_long_tail_partitioning(self):
        records = [record(f"t{i}", artist="Head") for i in range(10)]
        records.extend(record(f"s{i}", artist=f"Tail {i}") for i in range(10))
        tail = profile_module.build(views(records), population="t").long_tail["artist"]
        assert "Head" in tail.head_categories
        assert len(tail.singletons) == 10
        assert len(tail.rare_categories) == 10


class TestFamilyPressure:
    def test_a_large_family_is_measured(self):
        from conftest import one_big_duplicate_family

        profile = build(one_big_duplicate_family(family_size=20, others=10))
        pressure = profile.family_pressure
        assert pressure.largest_family == 20
        assert pressure.total_tracks == 30
        assert pressure.effective_families < 5.0

    def test_solo_tracks_count_as_families_of_one(self):
        """Otherwise a deduplicated corpus looks like it has no families."""
        profile = build([record(f"t{i}") for i in range(5)])
        assert profile.family_pressure.unique_families == 5
        assert profile.family_pressure.largest_family == 1

    def test_tracks_over_the_cap_are_counted(self):
        from conftest import one_big_duplicate_family

        profile = build(one_big_duplicate_family(family_size=20, others=2), duplicate_family_cap=1)
        assert profile.family_pressure.tracks_over_cap == 19


class TestSyntheticShare:
    def test_it_comes_from_provenance_only(self):
        profile = build(
            [
                record("a", source_type="AI_GENERATED"),
                record("b", source_type="USER_ORIGINAL"),
            ]
        )
        assert profile.synthetic_share_by_count == pytest.approx(0.5)

    def test_self_model_counts_as_synthetic(self):
        profile = build([record("a", source_type="SELF_MODEL_OUTPUT")])
        assert profile.synthetic_share_by_count == 1.0


class TestFindingsRequireATarget:
    def test_the_neutral_profile_invents_no_gaps(self):
        """A gap without a declared target is an invented objective."""
        profile = build([record(f"t{i}", language="ko", artist=f"A{i}") for i in range(10)])
        result = findings.evaluate(profile, targets.neutral())
        assert not [f for f in result if f.code.startswith("NEED_MORE_")]

    def test_a_declared_minimum_produces_a_gap(self):
        profile = build(
            [record(f"t{i}", language="ko" if i else "en", artist=f"A{i}") for i in range(10)]
        )
        target = targets.TargetProfile(
            name="T", shares={"language": {"en": targets.Range(minimum=0.5)}}
        )
        gaps = [f for f in findings.evaluate(profile, target) if f.code == "NEED_MORE_LANGUAGE_EN"]
        assert len(gaps) == 1
        assert gaps[0].current_share == pytest.approx(0.1)
        assert gaps[0].target_range is not None

    def test_a_declared_maximum_produces_overrepresentation(self):
        profile = build([record(f"t{i}", language="ko", artist=f"A{i}") for i in range(10)])
        target = targets.TargetProfile(
            name="T", shares={"language": {"ko": targets.Range(maximum=0.5)}}
        )
        over = [
            f for f in findings.evaluate(profile, target) if f.code == "LANGUAGE_OVERREPRESENTED"
        ]
        assert len(over) == 1

    def test_a_target_on_a_sparse_dimension_is_not_evaluated(self):
        """A confident finding computed from two tracks is worse than none."""
        profile = build(sparse_genre(total=20, labelled=2))
        target = targets.TargetProfile(
            name="T", shares={"genre": {"rock": targets.Range(minimum=0.3)}}
        )
        result = findings.evaluate(profile, target)
        assert [f for f in result if f.code == "NOT_ASSESSABLE" and f.dimension == "genre"]
        assert not [f for f in result if f.code.startswith("NEED_MORE_")]

    def test_every_finding_carries_its_denominator(self):
        profile = build(dominated_by_one_artist())
        for finding in findings.evaluate(profile, targets.neutral()):
            if finding.current_share is not None:
                assert finding.known_denominator is not None, finding.code

    def test_artist_dominance_is_detected_without_any_target(self):
        """Domination is a problem under every objective."""
        profile = build(dominated_by_one_artist(total=20, dominant=16))
        codes = {f.code for f in findings.evaluate(profile, targets.neutral())}
        assert findings.ONE_ARTIST_DOMINATES in codes

    def test_findings_are_ordered_worst_first_and_deterministically(self):
        profile = build(dominated_by_one_artist(total=20, dominant=18))
        first = [f.code for f in findings.evaluate(profile, targets.neutral())]
        second = [f.code for f in findings.evaluate(profile, targets.neutral())]
        assert first == second
        severities = [f.severity for f in findings.evaluate(profile, targets.neutral())]
        order = {Severity.CRITICAL.value: 0, Severity.WARNING.value: 1, Severity.INFO.value: 2}
        assert severities == sorted(severities, key=lambda s: order[s])

    def test_low_metadata_coverage_is_reported(self):
        profile = build(sparse_genre(total=20, labelled=1))
        codes = {
            f.dimension
            for f in findings.evaluate(profile, targets.neutral())
            if f.code == findings.METADATA_COVERAGE_LOW
        }
        assert "genre" in codes


class TestTargetProfiles:
    def test_the_neutral_profile_declares_no_shares(self):
        assert targets.neutral().shares == {}

    def test_built_in_profiles_validate(self):
        for name in targets.BUILT_IN:
            targets.validate(targets.by_name(name))

    def test_a_profile_cannot_constrain_an_unmeasurable_dimension(self):
        with pytest.raises(targets.ProfileError, match="does not provide"):
            targets.validate(
                targets.TargetProfile(name="X", shares={"trot_style": {"no": targets.Range()}})
            )

    def test_impossible_minimums_are_refused(self):
        with pytest.raises(targets.ProfileError, match="no dataset can satisfy"):
            targets.validate(
                targets.TargetProfile(
                    name="X",
                    shares={
                        "language": {
                            "ko": targets.Range(minimum=0.7),
                            "en": targets.Range(minimum=0.7),
                        }
                    },
                )
            )

    def test_a_range_with_min_above_max_is_refused(self):
        with pytest.raises(ValueError, match="exceeds maximum"):
            targets.Range(minimum=0.8, maximum=0.2)

    def test_a_future_profile_says_what_it_is_waiting_for(self):
        with pytest.raises(targets.ProfileError, match="not available yet"):
            targets.by_name("MODERN_NON_TROT")

    def test_unknown_profile_keys_are_refused(self):
        with pytest.raises(targets.ProfileError, match="unrecognised"):
            targets.load({"name": "X", "langauges": {}})

    def test_a_profile_digest_is_stable_and_sensitive(self):
        first = targets.korean_pop()
        assert first.digest() == targets.korean_pop().digest()
        assert first.digest() != targets.global_pop().digest()

    def test_a_profile_round_trips_through_json(self):
        loaded = targets.load(targets.korean_pop().to_dict())
        assert loaded.digest() == targets.korean_pop().digest()


class TestBucketing:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(10, "<30s"), (45, "30-60s"), (90, "60-120s"), (200, "180-240s"), (500, ">360s")],
    )
    def test_duration_buckets(self, seconds: float, expected: str):
        assert distributions.duration_bucket(seconds) == expected

    @pytest.mark.parametrize(
        ("bpm", "expected"), [(60, "<70"), (80, "70-90"), (120, "110-130"), (200, ">180")]
    )
    def test_tempo_buckets(self, bpm: float, expected: str):
        assert distributions.tempo_bucket(bpm) == expected


class TestConcentrationMath:
    def test_hhi_and_effective_count_are_reciprocal(self):
        distribution = distributions.CategoricalDistribution(dimension="x")
        distribution.known_count = 4
        distribution.buckets = [
            distributions.Bucket(label=str(i), count=1, share_by_count=0.25) for i in range(4)
        ]
        metrics = concentration.measure(distribution)
        assert metrics.hhi == pytest.approx(0.25)
        assert metrics.effective_categories == pytest.approx(4.0)

    def test_an_empty_distribution_is_not_an_error(self):
        metrics = concentration.measure(distributions.CategoricalDistribution(dimension="x"))
        assert metrics.top1_share == 0.0
        assert metrics.effective_categories == 0.0
