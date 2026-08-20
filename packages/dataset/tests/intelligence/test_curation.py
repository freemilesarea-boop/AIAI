"""Curation end to end: the hard gate, selection, weights, and the lock.

The rights tests are the ones that must never be allowed to rot. Every
other property here can be re-run and corrected; audio trained on
without permission cannot be untrained, and curation is the layer most
likely to try — it exists to improve balance, and the tempting way to
improve balance is to reach for material that is barred.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import (
    dominated_by_one_artist,
    mixed_rights,
    one_big_duplicate_family,
    record,
    write_manifest,
)

from luber_dataset.factory.intelligence import (
    curation,
    drift,
    human_report,
    reports,
    sampling,
    targets,
)
from luber_dataset.factory.intelligence import profile as profile_module
from luber_dataset.factory.intelligence.schemas import CurationAction, TrackView


def manifest_of(tmp_path: Path, records, name: str = "dataset_manifest.jsonl") -> Path:
    return write_manifest(tmp_path / name, records)


def views(records):
    return [TrackView(r, min_confidence=0.55) for r in records]


class TestInputContract:
    def test_the_source_manifest_is_never_written_to(self, tmp_path: Path):
        path = manifest_of(tmp_path, dominated_by_one_artist())
        before = path.read_bytes()
        curation.curate(path)
        assert path.read_bytes() == before

    def test_the_manifest_digest_is_recorded(self, tmp_path: Path):
        path = manifest_of(tmp_path, dominated_by_one_artist())
        result = curation.curate(path)
        assert len(result.source_manifest_sha256) == 64

    def test_the_factory_schema_version_is_carried_through(self, tmp_path: Path):
        result = curation.curate(manifest_of(tmp_path, [record("a")]))
        assert result.factory_schema_version == "luber-dataset-factory/1"

    def test_a_malformed_manifest_line_is_refused(self, tmp_path: Path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"track_id": "a"}\nnot json\n', encoding="utf-8")
        with pytest.raises(ValueError, match="not valid JSON"):
            curation.curate(path)

    def test_a_non_record_line_is_refused(self, tmp_path: Path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"no_track_id": true}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="not a manifest record"):
            curation.curate(path)


class TestRightsAreAHardGate:
    """Curation happens after eligibility, and cannot reach back past it."""

    def result(self, tmp_path: Path, **kwargs):
        return curation.curate(manifest_of(tmp_path, mixed_rights()), **kwargs)

    def test_unknown_rights_are_never_selected(self, tmp_path: Path):
        result = self.result(tmp_path)
        assert "trk_unknown" not in result.selection.selected_ids

    def test_denied_rights_are_never_selected(self, tmp_path: Path):
        assert "trk_denied" not in self.result(tmp_path).selection.selected_ids

    def test_restricted_status_is_never_selected(self, tmp_path: Path):
        """Even with permission TRUE: RESTRICTED means restricted."""
        assert "trk_restricted" not in self.result(tmp_path).selection.selected_ids

    def test_a_hard_block_is_never_selected(self, tmp_path: Path):
        assert "trk_selfmodel" not in self.result(tmp_path).selection.selected_ids

    def test_cleared_tracks_are_selected(self, tmp_path: Path):
        """The gate must not simply reject everything."""
        selected = set(self.result(tmp_path).selection.selected_ids)
        assert {"trk_ok1", "trk_ok2"} <= selected

    def test_exclusion_records_the_rights_reason(self, tmp_path: Path):
        decision = self.result(tmp_path).selection.decisions["trk_unknown"]
        assert decision.action == CurationAction.EXCLUDE_POLICY.value
        assert "RIGHTS_NOT_PERMITTED" in decision.reasons

    def test_no_target_profile_can_admit_barred_material(self, tmp_path: Path):
        """The temptation this gate exists to resist.

        A profile demanding 90% Korean, in a corpus whose only Korean
        material has unknown rights, must fail to reach its target
        rather than reach it.
        """
        records = [
            record("trk_en", language="en", artist="A"),
            record(
                "trk_ko_barred",
                language="ko",
                artist="B",
                permission="UNKNOWN",
                rights_status="UNKNOWN",
                training_eligible=False,
            ),
        ]
        target = targets.TargetProfile(
            name="GREEDY", shares={"language": {"ko": targets.Range(minimum=0.9)}}
        )
        result = curation.curate(manifest_of(tmp_path, records), target=target)
        assert "trk_ko_barred" not in result.selection.selected_ids

    def test_rights_are_absent_from_the_scoring_components(self, tmp_path: Path):
        """If rights were a weight, enough quality could outvote them."""
        result = self.result(tmp_path)
        components = result.selection.decisions["trk_ok1"].components.to_dict()
        for name in components:
            assert "right" not in name
            assert "provenance" not in name
            assert "permission" not in name


class TestEvaluationProtection:
    def test_an_evaluation_only_track_never_enters_training(self, tmp_path: Path):
        path = manifest_of(tmp_path, [record("trk_bench"), record("trk_normal")])
        config = curation.CurationConfig(evaluation_only=("trk_bench",))
        result = curation.curate(path, config=config)
        assert "trk_bench" not in result.selection.selected_ids
        assert "trk_normal" in result.selection.selected_ids

    def test_the_reason_is_recorded(self, tmp_path: Path):
        path = manifest_of(tmp_path, [record("trk_bench")])
        result = curation.curate(
            path, config=curation.CurationConfig(evaluation_only=("trk_bench",))
        )
        decision = result.selection.decisions["trk_bench"]
        assert decision.action == CurationAction.HOLDOUT.value
        assert "EVALUATION_ONLY" in decision.reasons


class TestSplitProtection:
    def test_validation_and_test_are_left_alone(self, tmp_path: Path):
        records = [
            record("trk_train", split="TRAIN"),
            record("trk_val", split="VALIDATION"),
            record("trk_test", split="TEST"),
        ]
        result = curation.curate(manifest_of(tmp_path, records))
        assert result.selection.selected_ids == ["trk_train"]
        for held in ("trk_val", "trk_test"):
            assert result.selection.decisions[held].action == CurationAction.HOLDOUT.value

    def test_no_track_changes_split(self, tmp_path: Path):
        """Curation selects from a frozen split; it does not reassign."""
        records = [record("a", split="TRAIN"), record("b", split="TEST")]
        result = curation.curate(manifest_of(tmp_path, records))
        for curated in result.curated_records:
            original = next(r for r in records if r["track_id"] == curated["track_id"])
            assert curated["split"] == original["split"]

    def test_validation_and_test_receive_no_sampling_weight(self, tmp_path: Path):
        records = [record("trk_train"), record("trk_val", split="VALIDATION")]
        result = curation.curate(manifest_of(tmp_path, records))
        assert "trk_val" not in result.sampling_plan.weights


class TestDuplicatePressure:
    def test_a_family_is_capped(self, tmp_path: Path):
        result = curation.curate(
            manifest_of(tmp_path, one_big_duplicate_family(family_size=20, others=5))
        )
        selected = set(result.selection.selected_ids)
        family = {f"trk_fam{i:03d}" for i in range(20)}
        assert len(selected & family) == 1, "one representative per family by default"

    def test_the_excluded_copies_say_why(self, tmp_path: Path):
        result = curation.curate(
            manifest_of(tmp_path, one_big_duplicate_family(family_size=5, others=2))
        )
        dropped = [
            d
            for d in result.selection.decisions.values()
            if d.action == CurationAction.EXCLUDE_DUPLICATE_PRESSURE.value
        ]
        assert dropped
        assert all("DUPLICATE_FAMILY_CAP" in d.reasons for d in dropped)

    def test_a_review_required_duplicate_is_not_collapsed(self, tmp_path: Path):
        """Uncertainty goes to a person, not to a decision."""
        records = [record("trk_a", dedup_decision="REVIEW_REQUIRED", duplicate_group="g")]
        result = curation.curate(manifest_of(tmp_path, records))
        assert result.selection.decisions["trk_a"].action == CurationAction.REVIEW.value

    def test_the_cap_is_configurable(self, tmp_path: Path):
        target = targets.TargetProfile(
            name="T", selection=targets.SelectionLimits(max_records_per_duplicate_family=3)
        )
        result = curation.curate(
            manifest_of(tmp_path, one_big_duplicate_family(family_size=10, others=0)),
            target=target,
        )
        assert len(result.selection.selected_ids) == 3


class TestConcentrationCaps:
    def test_an_artist_cap_downsamples_the_dominant_artist(self, tmp_path: Path):
        target = targets.TargetProfile(
            name="T", selection=targets.SelectionLimits(max_tracks_per_artist=3)
        )
        result = curation.curate(
            manifest_of(tmp_path, dominated_by_one_artist(total=20, dominant=12)), target=target
        )
        dominant = {
            track_id for track_id in result.selection.selected_ids if track_id.startswith("trk_dom")
        }
        assert len(dominant) == 3

    def test_minority_material_survives_the_cap(self, tmp_path: Path):
        """Downsampling the head must not also cost the tail."""
        target = targets.TargetProfile(
            name="T", selection=targets.SelectionLimits(max_tracks_per_artist=3)
        )
        result = curation.curate(
            manifest_of(tmp_path, dominated_by_one_artist(total=20, dominant=12)), target=target
        )
        minority = {
            track_id for track_id in result.selection.selected_ids if track_id.startswith("trk_oth")
        }
        assert len(minority) == 8

    def test_an_unknown_artist_is_never_capped(self, tmp_path: Path):
        """Grouping unknowns would treat "nobody knows" as one artist."""
        target = targets.TargetProfile(
            name="T", selection=targets.SelectionLimits(max_tracks_per_artist=1)
        )
        records = [record(f"t{i}") for i in range(5)]
        result = curation.curate(manifest_of(tmp_path, records), target=target)
        assert len(result.selection.selected_ids) == 5

    def test_the_cap_keeps_the_best_scoring_members(self, tmp_path: Path):
        target = targets.TargetProfile(
            name="T", selection=targets.SelectionLimits(max_tracks_per_artist=1)
        )
        records = [
            record("trk_low", artist="Same", quality_tier="C", quality_score=0.5),
            record("trk_high", artist="Same", quality_tier="A", quality_score=1.0),
        ]
        result = curation.curate(manifest_of(tmp_path, records), target=target)
        assert result.selection.selected_ids == ["trk_high"]


class TestScoring:
    def test_every_component_is_stored(self, tmp_path: Path):
        result = curation.curate(manifest_of(tmp_path, [record("a", artist="A", language="ko")]))
        components = result.selection.decisions["a"].components.to_dict()
        for name in (
            "quality",
            "coverage_contribution",
            "metadata_completeness",
            "source_diversity",
            "duplicate_pressure",
        ):
            assert name in components

    def test_a_score_is_bounded(self, tmp_path: Path):
        result = curation.curate(manifest_of(tmp_path, dominated_by_one_artist()))
        for decision in result.selection.decisions.values():
            assert 0.0 <= decision.score <= 1.0

    def test_a_rare_tier_b_can_beat_a_redundant_tier_a(self, tmp_path: Path):
        """The trade-off the coverage weight exists to make possible.

        Nine identical-region Tier A tracks and one Tier B track in an
        empty region: the B track is worth more to the dataset.
        """
        records = [
            record(f"trk_common{i}", artist=f"A{i}", language="ko", quality_tier="A")
            for i in range(9)
        ]
        records.append(record("trk_rare", artist="Rare", language="en", quality_tier="B"))
        result = curation.curate(manifest_of(tmp_path, records))
        rare = result.selection.decisions["trk_rare"].score
        common = result.selection.decisions["trk_common0"].score
        assert rare > common, f"rare B {rare} did not beat redundant A {common}"

    def test_unknown_metadata_earns_no_rarity_bonus(self, tmp_path: Path):
        """Otherwise "unlabelled" becomes the most valuable property."""
        records = [
            record("trk_known", artist="A", language="ko", vocal_class="VOCAL", bpm=120.0),
            record("trk_unknown", artist="B"),
        ]
        result = curation.curate(manifest_of(tmp_path, records))
        known = result.selection.decisions["trk_known"].components
        unknown = result.selection.decisions["trk_unknown"].components
        assert known.metadata_completeness > unknown.metadata_completeness

    def test_weights_must_sum_to_one(self, tmp_path: Path):
        config = curation.CurationConfig(scoring_weights={"quality": 0.5})
        with pytest.raises(ValueError, match=r"sum to 1\.0"):
            curation.curate(manifest_of(tmp_path, [record("a")]), config=config)


class TestSamplingWeights:
    def test_weights_are_bounded(self, tmp_path: Path):
        target = targets.korean_pop()
        records = [
            record(f"t{i}", language="en", artist=f"A{i}", vocal_class="VOCAL") for i in range(19)
        ]
        records.append(record("trk_ko", language="ko", artist="K", vocal_class="VOCAL"))
        result = curation.curate(manifest_of(tmp_path, records), target=target)
        assert result.sampling_plan.bounded
        for weight in result.sampling_plan.weights.values():
            assert weight <= result.sampling_plan.max_weight

    def test_a_scarce_targeted_category_is_lifted(self, tmp_path: Path):
        target = targets.korean_pop()
        records = [
            record(f"t{i}", language="en", artist=f"A{i}", vocal_class="VOCAL") for i in range(19)
        ]
        records.append(record("trk_ko", language="ko", artist="K", vocal_class="VOCAL"))
        result = curation.curate(manifest_of(tmp_path, records), target=target)
        assert result.sampling_plan.weights["trk_ko"] > 1.0

    def test_nothing_is_duplicated_to_rebalance(self, tmp_path: Path):
        """A weight, never a copy. Copies teach memorisation."""
        records = [record(f"t{i}", language="en", artist=f"A{i}") for i in range(5)]
        result = curation.curate(manifest_of(tmp_path, records))
        ids = [r["track_id"] for r in result.curated_records]
        assert len(ids) == len(set(ids))

    def test_an_untargeted_track_is_neutral(self, tmp_path: Path):
        result = curation.curate(manifest_of(tmp_path, [record("a", language="ko", artist="A")]))
        assert result.sampling_plan.weights["a"] == 1.0

    def test_the_cap_is_configurable_and_must_contain_one(self, tmp_path: Path):
        with pytest.raises(ValueError, match=r"contain 1\.0"):
            sampling.build(
                [], profile_module.build([], population="t"), targets.neutral(), max_weight=0.5
            )


class TestDeterminism:
    def test_two_runs_agree_on_everything(self, tmp_path: Path):
        path = manifest_of(tmp_path, dominated_by_one_artist(total=30, dominant=18))
        first = curation.curate(path, target=targets.korean_pop())
        second = curation.curate(path, target=targets.korean_pop())

        assert first.canonical_digest() == second.canonical_digest()
        assert first.selection.selected_ids == second.selection.selected_ids
        assert first.sampling_plan.weights == second.sampling_plan.weights
        for track_id, decision in first.selection.decisions.items():
            assert decision.score == second.selection.decisions[track_id].score

    def test_record_order_does_not_matter(self, tmp_path: Path):
        records = dominated_by_one_artist(total=20, dominant=12)
        forward = curation.curate(manifest_of(tmp_path, records, "a.jsonl"))
        backward = curation.curate(manifest_of(tmp_path, list(reversed(records)), "b.jsonl"))
        assert forward.selection.selected_ids == backward.selection.selected_ids

    def test_a_different_profile_changes_the_answer(self, tmp_path: Path):
        """Determinism must not mean insensitivity."""
        path = manifest_of(tmp_path, dominated_by_one_artist())
        neutral = curation.curate(path, target=targets.neutral())
        capped = curation.curate(
            path,
            target=targets.TargetProfile(
                name="C", selection=targets.SelectionLimits(max_tracks_per_artist=2)
            ),
        )
        assert neutral.selection.selected_ids != capped.selection.selected_ids

    def test_the_config_digest_is_stable_and_sensitive(self):
        assert curation.CurationConfig().digest() == curation.CurationConfig().digest()
        assert curation.CurationConfig(seed=1).digest() != curation.CurationConfig(seed=2).digest()


class TestArtifacts:
    def curated(self, tmp_path: Path):
        path = manifest_of(tmp_path, dominated_by_one_artist(total=20, dominant=12))
        result = curation.curate(path, target=targets.korean_pop())
        output = tmp_path / "curated"
        return result, output

    def test_the_curated_manifest_is_separate(self, tmp_path: Path):
        result, output = self.curated(tmp_path)
        reports.write_curated_manifest(output, result)
        assert (output / "curated_manifest.jsonl").is_file()
        assert (tmp_path / "dataset_manifest.jsonl").is_file()

    def test_each_curated_record_keeps_its_original_identity(self, tmp_path: Path):
        result, output = self.curated(tmp_path)
        reports.write_curated_manifest(output, result)
        rows = [
            json.loads(line)
            for line in (output / "curated_manifest.jsonl").read_text().splitlines()
        ]
        for row in rows:
            assert "track_id" in row
            assert "source" in row and "provenance" in row
            assert row["curation_action"]
            assert row["curation_schema_version"] == "luber-dataset-curation/1"
            assert "score_components" in row

    def test_the_summary_reports_before_and_after(self, tmp_path: Path):
        result, _ = self.curated(tmp_path)
        summary = reports.build_summary(result)
        assert summary["before"]["track_count"] >= summary["after"]["track_count"]
        assert "excluded_by_reason" in summary
        assert summary["source_manifest_sha256"]

    def test_the_wishlist_only_asks_for_declared_targets(self, tmp_path: Path):
        path = manifest_of(
            tmp_path, [record(f"t{i}", language="ko", artist=f"A{i}") for i in range(10)]
        )
        neutral = curation.curate(path, target=targets.neutral())
        assert reports.build_wishlist(neutral) == []

    def test_the_wishlist_estimates_hours_from_the_declared_range(self, tmp_path: Path):
        records = [
            record(f"t{i}", language="ko", artist=f"A{i}", vocal_class="VOCAL") for i in range(19)
        ]
        records.append(record("trk_en", language="en", artist="E", vocal_class="VOCAL"))
        result = curation.curate(manifest_of(tmp_path, records), target=targets.korean_pop())
        entries = [e for e in reports.build_wishlist(result) if e["target"] == "en"]
        assert entries and entries[0]["estimated_hours_needed"] is not None

    def test_the_review_queue_puts_rights_first(self, tmp_path: Path):
        result, _ = self.curated(tmp_path)
        existing = [
            {"track_id": "trk_dom000", "reason": "LANGUAGE_UNCERTAIN"},
            {"track_id": "trk_dom001", "reason": "RIGHTS_UNKNOWN"},
        ]
        ordered = reports.build_review_queue(result, existing)
        assert ordered[0]["reason"] == "RIGHTS_UNKNOWN"
        assert ordered[0]["why_this_order"]

    def test_the_human_report_answers_its_questions(self, tmp_path: Path):
        result, _ = self.curated(tmp_path)
        text = human_report.render(result)
        for heading in (
            "Top 10 risks",
            "What dominates this dataset?",
            "What is missing?",
            "What is uncertain?",
            "What cannot be assessed, and why",
            "What should be added?",
            "What should be reduced?",
        ):
            assert heading in text, heading

    def test_the_report_names_what_it_cannot_assess(self, tmp_path: Path):
        result, _ = self.curated(tmp_path)
        text = human_report.render(result)
        assert "Song structure" in text
        assert "Transcripts" in text
        assert "no validated detector" in text


class TestCurationLock:
    def build(self, tmp_path: Path):
        path = manifest_of(tmp_path, dominated_by_one_artist())
        result = curation.curate(path, target=targets.korean_pop())
        output = tmp_path / "curated"
        reports.write_curated_manifest(output, result)
        weights = reports.write_sampling_weights(output, result)
        lock = reports.freeze(output, result, curation_id="cur-1", weights_digest=weights)
        return path, output, result, lock

    def test_the_lock_records_every_input_digest(self, tmp_path: Path):
        _, _, _, lock = self.build(tmp_path)
        for value in (
            lock.source_manifest_sha256,
            lock.target_profile_sha256,
            lock.config_sha256,
            lock.curated_manifest_sha256,
            lock.distribution_summary_digest,
        ):
            assert len(value) == 64

    def test_a_curation_verifies_against_its_own_lock(self, tmp_path: Path):
        path, output, _, _ = self.build(tmp_path)
        fresh = curation.curate(path, target=targets.korean_pop())
        assert reports.verify(output / "curation_lock.json", fresh) == []

    def test_a_changed_profile_fails_verification(self, tmp_path: Path):
        path, output, _, _ = self.build(tmp_path)
        other = curation.curate(path, target=targets.global_pop())
        problems = reports.verify(output / "curation_lock.json", other)
        assert any("target profile" in problem for problem in problems)

    def test_a_changed_manifest_fails_verification(self, tmp_path: Path):
        _path, output, _, _ = self.build(tmp_path)
        extra = dominated_by_one_artist()
        extra.append(record("trk_new", artist="New"))
        changed = curation.curate(
            manifest_of(tmp_path, extra, "changed.jsonl"), target=targets.korean_pop()
        )
        problems = reports.verify(output / "curation_lock.json", changed)
        assert any("source manifest" in problem for problem in problems)

    def test_the_lock_is_stable_across_reruns(self, tmp_path: Path):
        path, _, _, first = self.build(tmp_path)
        second_result = curation.curate(path, target=targets.korean_pop())
        assert first.curated_manifest_sha256 == second_result.canonical_digest()


class TestDrift:
    def test_a_language_shift_is_reported(self, tmp_path: Path):
        before = profile_module.build(
            views([record(f"t{i}", language="ko", artist=f"A{i}") for i in range(10)]),
            population="a",
        )
        after = profile_module.build(
            views(
                [
                    record(f"t{i}", language="ko" if i < 5 else "en", artist=f"A{i}")
                    for i in range(10)
                ]
            ),
            population="b",
        )
        report = drift.compare(before, after)
        moved = report.dimensions["language"].moved
        assert "en" in moved
        assert moved["en"][2] == pytest.approx(0.5)

    def test_a_collapse_in_diversity_is_called_out(self, tmp_path: Path):
        before = profile_module.build(
            views([record(f"t{i}", artist=f"A{i}") for i in range(10)]), population="a"
        )
        after = profile_module.build(
            views([record(f"t{i}", artist="Only") for i in range(10)]), population="b"
        )
        report = drift.compare(before, after)
        assert any("diversity collapsed" in note for note in report.notes)

    def test_identical_datasets_show_no_movement(self, tmp_path: Path):
        records = [record(f"t{i}", language="ko", artist=f"A{i}") for i in range(10)]
        profile = profile_module.build(views(records), population="a")
        report = drift.compare(profile, profile)
        assert all(not d.moved for d in report.dimensions.values())
        assert report.notes == []

    def test_the_markdown_renders(self, tmp_path: Path):
        profile = profile_module.build(
            views([record("a", artist="A", language="ko")]), population="a"
        )
        text = drift.render_markdown(drift.compare(profile, profile))
        assert "# Dataset drift" in text
