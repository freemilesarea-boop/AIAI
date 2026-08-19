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

import importlib.util
import json
import re
import sys
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


def load_listening_tool():
    """Import the server script by path; it is a script, not a package."""
    path = Path(__file__).resolve().parents[3] / "scripts" / "benchmark" / "p20_listening.py"
    spec = importlib.util.spec_from_file_location("p20_listening", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["p20_listening"] = module
    spec.loader.exec_module(module)
    return module


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


class TestScorePersistenceUX:
    """The interface guarantees that stop a listening session vanishing.

    A 41-field form rejected at the bottom of the page is how twelve
    tracks of work disappear without anyone noticing — the failure this
    hardening exists to remove. These assert the guarantees rather than
    the styling: the required set is per-track, the save control is
    disabled until it is satisfied, an invalid submit is reported at the
    top, and nothing typed is discarded before persistence is confirmed.
    """

    def page(self):
        module = load_listening_tool()
        item = {
            "benchmark_id": "TROT-01",
            "prompt": "p",
            "lyrics": "[Verse]\n가사",
            "instrumental": False,
            "korean": True,
            "duration": 60.0,
            "_generation_id": "x",
            "_source": Path("/dev/null"),
        }
        return module.render(item, 0, 12, completed=0), module

    def test_the_required_dimensions_are_published_for_the_counter(self):
        html, _ = self.page()
        payload = re.search(r'id="required-dims">(.*?)</script>', html, re.S).group(1)
        assert len(json.loads(payload)) == 41

    def test_the_required_count_follows_the_case_not_a_constant(self):
        """An instrumental needs far fewer scores than a Korean vocal."""
        module = load_listening_tool()
        base = {
            "benchmark_id": "GEN-10",
            "prompt": "p",
            "lyrics": "",
            "_generation_id": "x",
            "_source": Path("/dev/null"),
        }
        instrumental = module.render(
            {**base, "instrumental": True, "korean": False, "duration": 60.0}, 0, 12, completed=0
        )
        korean = module.render(
            {**base, "instrumental": False, "korean": True, "duration": 60.0}, 0, 12, completed=0
        )

        def count(page: str) -> int:
            payload = re.search(r'id="required-dims">(.*?)</script>', page, re.S)
            assert payload is not None
            return len(json.loads(payload.group(1)))

        assert count(instrumental) < count(korean)

    def test_save_starts_disabled(self):
        html, _ = self.page()
        assert 'id="save" disabled' in html

    def test_the_counter_and_save_sit_in_a_sticky_bar(self):
        """Both must be reachable without scrolling the whole form."""
        html, _ = self.page()
        assert 'class="bar"' in html
        assert "position:sticky" in html
        assert 'id="counter"' in html

    def test_every_section_carries_its_own_completion_badge(self):
        html, _ = self.page()
        assert len(re.findall(r"data-section=", html)) >= 6

    def test_the_error_region_is_above_the_form(self):
        html, _ = self.page()
        assert html.index('id="topbanner"') < html.index('id="scoreform"')

    def test_native_validation_is_disabled_in_favour_of_the_aggregated_error(self):
        """The browser's tooltip lands wherever it likes, often off-screen."""
        html, _ = self.page()
        assert "novalidate" in html

    def test_the_draft_key_is_namespaced_by_baseline_and_track(self):
        html, module = self.page()
        assert f'data-baseline="{module.BASELINE_ID}"' in html
        assert 'data-benchmark="TROT-01"' in html
        assert "luber.p20h.draft" in html

    def test_the_draft_is_cleared_only_after_a_confirmed_save(self):
        html, _ = self.page()
        script = html[html.index("<script>") :]
        assert script.index("response.ok") < script.index("removeItem(draftKey)")

    def test_a_failed_save_keeps_the_draft(self):
        html, _ = self.page()
        assert "Your draft is preserved" in html

    def test_the_unload_flush_cannot_resurrect_a_saved_draft(self):
        """The bug this guard exists for: the flush re-wrote the draft
        from the still-populated form as the page navigated away."""
        html, _ = self.page()
        assert "if (persisted) return;" in html

    def test_success_names_the_running_total(self):
        html, _ = self.page()
        assert "Score saved —" in html

    def test_no_audio_or_model_identity_reaches_the_draft(self):
        html, _ = self.page()
        script = html[html.index("<script>") :]
        for forbidden in ("generation_id", "audio", "model", "seed"):
            assert f'"{forbidden}"' not in script


class TestProgressTruthfulness:
    def test_completed_counts_persisted_scores_not_queue_position(self):
        """Skipping a track must not read as having finished it."""
        module = load_listening_tool()
        item = {
            "benchmark_id": "KO-01",
            "prompt": "p",
            "lyrics": "[Verse]\n가사",
            "instrumental": False,
            "korean": True,
            "duration": 30.0,
            "_generation_id": "x",
            "_source": Path("/dev/null"),
        }
        # Third in the queue because two were skipped, but nothing saved.
        html = module.render(item, 2, 12, completed=0)
        assert "Track 3 / 12 · Completed 0 / 12" in html

    def test_the_session_end_reports_the_real_total(self):
        module = load_listening_tool()
        assert "Completed 5 / 12" in module.render(None, 12, 12, completed=5)


class TestDuplicateProtection:
    def test_the_handler_refuses_a_second_save_for_one_track(self):
        source = (
            Path(__file__).resolve().parents[3] / "scripts" / "benchmark" / "p20_listening.py"
        ).read_text()
        # Refusal happens before the append, so the store is never touched.
        guard = source.index("if benchmark_id in self._scored():")
        append = source.index("self.store.append(")
        assert guard < append
        assert "409" in source[guard:append]
