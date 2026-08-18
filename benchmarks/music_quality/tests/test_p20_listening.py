"""The rules the listening tool enforces on a human's behalf.

A baseline is only worth having if the scores in it mean one thing. That
requires refusing three specific inputs: a score outside the anchors, a
dimension that does not apply to the case, and a tag that is not in the
taxonomy. Each of those, accepted quietly, produces a number that looks
like data and is not.

The rubric documents are the specification. These tests also check the
module has not drifted from them, because two sources of truth for what
"the rubric" means is how a frozen baseline stops being frozen.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.p20_rubric import (
    ALL_DIMENSIONS,
    ARTIFACT_TAGS,
    DIMENSION_GROUPS,
    KOREAN_DIMENSIONS,
    VOCAL_DIMENSIONS,
    P20ScoreError,
    expected_dimensions,
    validate_scores,
    validate_tags,
)

BENCH = Path(__file__).resolve().parents[1]
RUBRIC = BENCH / "listening" / "RUBRIC_P20.md"
TAXONOMY = BENCH / "listening" / "TAXONOMY.md"


def full(dimensions) -> dict[str, int]:
    return {name: 7 for name in dimensions}


class TestApplicableDimensions:
    def test_an_instrumental_is_not_asked_about_a_voice(self):
        dims = expected_dimensions(instrumental=True, korean=False, duration_seconds=60)
        assert not set(dims) & set(VOCAL_DIMENSIONS)
        assert not set(dims) & set(KOREAN_DIMENSIONS)

    def test_a_non_korean_vocal_gets_vocal_but_not_korean(self):
        dims = expected_dimensions(instrumental=False, korean=False, duration_seconds=60)
        assert set(VOCAL_DIMENSIONS) <= set(dims)
        assert not set(dims) & set(KOREAN_DIMENSIONS)

    def test_a_korean_vocal_gets_both(self):
        dims = expected_dimensions(instrumental=False, korean=True, duration_seconds=60)
        assert set(VOCAL_DIMENSIONS) <= set(dims)
        assert set(KOREAN_DIMENSIONS) <= set(dims)

    def test_long_form_coherence_is_only_asked_when_there_is_long_form(self):
        short = expected_dimensions(instrumental=True, korean=False, duration_seconds=60)
        long_ = expected_dimensions(instrumental=True, korean=False, duration_seconds=120)
        assert "long_form_coherence" not in short
        assert "long_form_coherence" in long_

    def test_the_trot_measurement_is_present_on_every_vocal_case(self):
        """The load-bearing dimension of this phase."""
        dims = expected_dimensions(instrumental=False, korean=True, duration_seconds=60)
        assert "trot_absence" in dims


class TestScoreValidation:
    def test_a_complete_in_range_set_is_accepted(self):
        dims = expected_dimensions(instrumental=True, korean=False, duration_seconds=60)
        assert validate_scores(
            full(dims), instrumental=True, korean=False, duration_seconds=60
        ) == full(dims)

    @pytest.mark.parametrize("value", [0, 11, -3, 100])
    def test_a_score_outside_the_anchors_is_refused(self, value):
        dims = expected_dimensions(instrumental=True, korean=False, duration_seconds=60)
        scores = full(dims) | {"melody_quality": value}
        with pytest.raises(P20ScoreError, match="outside 1-10"):
            validate_scores(scores, instrumental=True, korean=False, duration_seconds=60)

    def test_a_missing_score_is_refused_rather_than_defaulted(self):
        """A filled-in average would distort more than an absent score."""
        dims = expected_dimensions(instrumental=True, korean=False, duration_seconds=60)
        scores = full(dims)
        del scores["melody_quality"]
        with pytest.raises(P20ScoreError, match="missing"):
            validate_scores(scores, instrumental=True, korean=False, duration_seconds=60)

    def test_a_vocal_score_on_an_instrumental_is_refused(self):
        dims = expected_dimensions(instrumental=True, korean=False, duration_seconds=60)
        scores = full(dims) | {"vocal_timbre": 6}
        with pytest.raises(P20ScoreError, match="do not apply"):
            validate_scores(scores, instrumental=True, korean=False, duration_seconds=60)

    def test_an_invented_dimension_is_refused(self):
        dims = expected_dimensions(instrumental=True, korean=False, duration_seconds=60)
        scores = full(dims) | {"vibes": 9}
        with pytest.raises(P20ScoreError, match="unknown dimension"):
            validate_scores(scores, instrumental=True, korean=False, duration_seconds=60)

    def test_a_non_numeric_score_is_refused(self):
        dims = expected_dimensions(instrumental=True, korean=False, duration_seconds=60)
        scores = full(dims) | {"melody_quality": "good"}
        with pytest.raises(P20ScoreError):
            validate_scores(scores, instrumental=True, korean=False, duration_seconds=60)


class TestTagValidation:
    def test_known_tags_pass(self):
        assert validate_tags(["VOCAL_TROT_STYLE", "MELODY_TROT_LIKE"]) == [
            "VOCAL_TROT_STYLE",
            "MELODY_TROT_LIKE",
        ]

    def test_an_unknown_tag_is_refused(self):
        """A typo silently becoming a category makes frequencies useless."""
        with pytest.raises(P20ScoreError, match="unknown artifact tag"):
            validate_tags(["VOCAL_TROT_STYL"])

    def test_duplicates_collapse_so_frequency_counts_stay_honest(self):
        assert validate_tags(["SIBILANCE", "SIBILANCE"]) == ["SIBILANCE"]

    def test_no_tags_is_valid(self):
        assert validate_tags([]) == []


class TestVocabularyMatchesTheSpecification:
    def test_every_tag_in_the_module_appears_in_the_taxonomy(self):
        text = TAXONOMY.read_text()
        missing = [tag for tag in ARTIFACT_TAGS if tag not in text]
        assert not missing, f"tags absent from TAXONOMY.md: {missing}"

    def test_every_dimension_in_the_module_appears_in_the_rubric(self):
        text = RUBRIC.read_text()
        missing = [d for d in ALL_DIMENSIONS if d not in text]
        assert not missing, f"dimensions absent from RUBRIC_P20.md: {missing}"

    def test_the_groups_cover_every_dimension_exactly_once(self):
        grouped = [name for _, names in DIMENSION_GROUPS for name in names]
        assert sorted(grouped) == sorted(ALL_DIMENSIONS)
        assert len(grouped) == len(set(grouped))


class TestBlindness:
    def test_the_page_template_exposes_no_identifying_field(self):
        """The listener must not be told what they are listening to."""
        source = (
            Path(__file__).resolve().parents[3] / "scripts" / "benchmark" / "p20_listening.py"
        ).read_text()
        # The rendered template, not the module's own prose.
        template = source.split('PAGE = """')[1].split('"""')[0]
        for leak in ("seed", "generation_id", "model", "acestep", "lufs", "finished"):
            assert leak not in template.lower(), f"page template mentions {leak}"

    def test_scores_are_recorded_as_blind(self):
        source = (
            Path(__file__).resolve().parents[3] / "scripts" / "benchmark" / "p20_listening.py"
        ).read_text()
        assert "blind=True" in source
