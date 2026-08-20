"""The rights gate, sidecar validation, eligibility, and splitting.

The rights tests exist because this is the one part of the factory whose
failure cannot be undone. Audio trained on without permission cannot be
untrained, so the assertion that matters is negative: UNKNOWN does not
become TRUE, through any path, under any configuration.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import tone, write_sidecar, write_wav

from luber_dataset.factory.classification import assess_language, assess_text, assess_vocals
from luber_dataset.factory.config import SplitConfig
from luber_dataset.factory.dedup import DedupDecision, DedupRecord
from luber_dataset.factory.metadata import Sidecar, SidecarError, load_sidecar
from luber_dataset.factory.provenance import (
    RightsStatus,
    SourceType,
    TrainingPermission,
    resolve,
)
from luber_dataset.factory.quality import QualityAssessment, QualityTier
from luber_dataset.factory.splitting import (
    RIGHTS_HARD_BLOCK,
    RIGHTS_UNKNOWN,
    Split,
    assess_eligibility,
    assign_splits,
    build_groups,
    group_key,
    verify_no_leakage,
)


def good_quality() -> QualityAssessment:
    return QualityAssessment(quality_flags=[], quality_score=1.0, quality_tier=QualityTier.A.value)


def clean_dedup() -> DedupRecord:
    return DedupRecord(canonical_track_id="trk_x", dedup_decision=DedupDecision.KEEP.value)


class TestSidecarValidation:
    def test_a_valid_sidecar_is_read(self, tmp_path: Path):
        audio = write_wav(tmp_path / "t.wav", tone())
        write_sidecar(audio, title="Song", artist="Me", rights_status="USER_OWNED")
        sidecar = load_sidecar(audio)
        assert sidecar is not None
        assert sidecar.get("title") == "Song"

    def test_no_sidecar_is_not_an_error(self, tmp_path: Path):
        assert load_sidecar(write_wav(tmp_path / "t.wav", tone())) is None

    def test_an_unknown_field_is_refused(self, tmp_path: Path):
        """A typo that silently does nothing is the worst outcome."""
        audio = write_wav(tmp_path / "t.wav", tone())
        write_sidecar(audio, commercial_trainng_allowed="true")
        with pytest.raises(SidecarError, match="unrecognised"):
            load_sidecar(audio)

    def test_malformed_json_is_refused_rather_than_ignored(self, tmp_path: Path):
        audio = write_wav(tmp_path / "t.wav", tone())
        audio.with_suffix(".json").write_text("{not json", encoding="utf-8")
        with pytest.raises(SidecarError, match="not valid JSON"):
            load_sidecar(audio)

    def test_a_bad_permission_value_is_refused(self, tmp_path: Path):
        audio = write_wav(tmp_path / "t.wav", tone())
        write_sidecar(audio, commercial_training_allowed="probably")
        with pytest.raises(SidecarError, match="never as permission"):
            load_sidecar(audio)

    @pytest.mark.parametrize(
        ("supplied", "expected"),
        [
            ("true", "TRUE"),
            (True, "TRUE"),
            ("false", "FALSE"),
            (False, "FALSE"),
            ("unknown", "UNKNOWN"),
        ],
    )
    def test_permission_is_tri_state_not_truthiness(self, tmp_path: Path, supplied, expected):
        """`bool("false")` is True, and that must never grant a right."""
        audio = write_wav(tmp_path / "t.wav", tone())
        write_sidecar(audio, commercial_training_allowed=supplied)
        sidecar = load_sidecar(audio)
        assert sidecar is not None
        assert sidecar.get("commercial_training_allowed") == expected


class TestRightsGate:
    def test_no_sidecar_means_unknown(self, tmp_path: Path):
        provenance = resolve(tmp_path / "music" / "song.wav", None)
        assert provenance.rights_status == RightsStatus.UNKNOWN.value
        assert provenance.commercial_training_allowed == TrainingPermission.UNKNOWN.value
        assert not provenance.training_permitted

    def test_unknown_never_becomes_true(self, tmp_path: Path):
        """The rule the whole module exists for."""
        provenance = resolve(tmp_path / "originals" / "my song.wav", None)
        assert provenance.commercial_training_allowed != TrainingPermission.TRUE.value

    def test_a_folder_name_grants_nothing(self, tmp_path: Path):
        """`originals/` is a filing decision, not a licence."""
        for folder in ("originals", "제작음원", "my_music", "public_domain"):
            provenance = resolve(tmp_path / folder / "song.wav", None)
            assert not provenance.training_permitted, folder

    def test_an_operator_statement_can_permit(self, tmp_path: Path):
        sidecar = Sidecar(
            path="s.json",
            fields={"rights_status": "USER_OWNED", "commercial_training_allowed": "TRUE"},
        )
        provenance = resolve(tmp_path / "song.wav", sidecar)
        assert provenance.training_permitted

    def test_permission_without_a_status_is_downgraded(self, tmp_path: Path):
        """An incoherent claim resolves against permission, and says so."""
        sidecar = Sidecar(path="s.json", fields={"commercial_training_allowed": "TRUE"})
        provenance = resolve(tmp_path / "song.wav", sidecar)
        assert provenance.commercial_training_allowed == TrainingPermission.UNKNOWN.value
        assert "downgraded" in provenance.provenance_notes
        assert not provenance.training_permitted

    def test_self_model_output_is_hard_blocked(self, tmp_path: Path):
        """Training ACE-Step on ACE-Step output teaches it its own faults."""
        provenance = resolve(tmp_path / "ace-step" / "out.wav", None)
        assert "SELF_MODEL_OUTPUT" in provenance.hard_blocks
        assert provenance.source_type == SourceType.SELF_MODEL_OUTPUT.value

    def test_a_hard_block_cannot_be_argued_away(self, tmp_path: Path):
        """Not even by an explicit, well-formed operator permission."""
        sidecar = Sidecar(
            path="s.json",
            fields={
                "rights_status": "USER_OWNED",
                "commercial_training_allowed": "TRUE",
                "source_type": "USER_ORIGINAL",
            },
        )
        provenance = resolve(tmp_path / "acestep" / "out.wav", sidecar)
        assert provenance.hard_blocks
        assert not provenance.training_permitted

    def test_unlawful_acquisition_is_hard_blocked(self, tmp_path: Path):
        sidecar = Sidecar(
            path="s.json",
            fields={
                "rights_status": "USER_OWNED",
                "commercial_training_allowed": "TRUE",
                "source": "ripped from a CD",
            },
        )
        provenance = resolve(tmp_path / "song.wav", sidecar)
        assert any("UNLAWFUL" in block for block in provenance.hard_blocks)
        assert not provenance.training_permitted

    def test_the_source_of_every_field_is_recorded(self, tmp_path: Path):
        """An inference must never be mistaken for an assertion."""
        sidecar = Sidecar(
            path="s.json",
            fields={"rights_status": "LICENSED", "commercial_training_allowed": "TRUE"},
        )
        provenance = resolve(tmp_path / "song.wav", sidecar)
        assert provenance.field_sources["rights_status"] == "USER"
        assert provenance.field_sources["commercial_training_allowed"] == "USER"


class TestEligibility:
    def kwargs(self, **overrides):
        base = {
            "decoded": True,
            "quality": good_quality(),
            "provenance": resolve(Path("/music/song.wav"), None),
            "dedup": clean_dedup(),
            "min_tier": "B",
            "include_rights_unknown": False,
        }
        return {**base, **overrides}

    def test_unknown_rights_block_training_by_default(self):
        result = assess_eligibility(**self.kwargs())
        assert not result.training_eligible
        assert RIGHTS_UNKNOWN in result.eligibility_reasons

    def test_analysis_is_still_allowed(self):
        """Measuring a file is not using it, and the operator needs the
        inventory in order to decide anything."""
        assert assess_eligibility(**self.kwargs()).analysis_eligible

    def test_the_override_admits_unknown_rights(self):
        result = assess_eligibility(**self.kwargs(include_rights_unknown=True))
        assert result.training_eligible
        assert RIGHTS_UNKNOWN in result.eligibility_reasons, "the reason is still recorded"

    def test_the_override_cannot_clear_a_hard_block(self):
        """An override that waived everything would be an off switch."""
        blocked = resolve(Path("/ace-step/out.wav"), None)
        result = assess_eligibility(**self.kwargs(provenance=blocked, include_rights_unknown=True))
        assert not result.training_eligible
        assert RIGHTS_HARD_BLOCK in result.eligibility_reasons

    def test_the_override_cannot_clear_a_quality_rejection(self):
        rejected = QualityAssessment(
            quality_flags=["CORRUPT"], quality_score=0.0, quality_tier=QualityTier.REJECT.value
        )
        result = assess_eligibility(**self.kwargs(quality=rejected, include_rights_unknown=True))
        assert not result.training_eligible

    def test_a_review_required_duplicate_is_excluded(self):
        pending = DedupRecord(
            canonical_track_id="trk_x", dedup_decision=DedupDecision.REVIEW_REQUIRED.value
        )
        result = assess_eligibility(**self.kwargs(dedup=pending, include_rights_unknown=True))
        assert not result.training_eligible
        assert "NEAR_DUPLICATE_REVIEW_REQUIRED" in result.eligibility_reasons

    def test_every_refusal_is_recorded(self):
        result = assess_eligibility(**self.kwargs(decoded=False))
        assert "DECODE_FAILED" in result.eligibility_reasons


class TestDeterministicSplitting:
    def groups(self, count: int = 200) -> dict[str, list[str]]:
        return {f"group-{i}": [f"trk_{i}"] for i in range(count)}

    def test_the_same_inputs_give_the_same_split(self):
        config = SplitConfig()
        assert assign_splits(self.groups(), config) == assign_splits(self.groups(), config)

    def test_a_different_seed_gives_a_different_split(self):
        first = assign_splits(self.groups(), SplitConfig(seed=1))
        second = assign_splits(self.groups(), SplitConfig(seed=2))
        assert first != second

    def test_input_order_does_not_matter(self):
        """Hashing, not shuffling: adding a track cannot move another."""
        groups = self.groups(50)
        reversed_groups = dict(reversed(list(groups.items())))
        config = SplitConfig()
        assert assign_splits(groups, config) == assign_splits(reversed_groups, config)

    def test_adding_a_track_does_not_move_the_others(self):
        config = SplitConfig()
        before = assign_splits(self.groups(100), config)
        after = assign_splits(self.groups(101), config)
        for track_id, split in before.items():
            assert after[track_id] == split

    def test_the_proportions_are_roughly_honoured(self):
        assignment = assign_splits(self.groups(2000), SplitConfig())
        train = sum(1 for s in assignment.values() if s == Split.TRAIN.value)
        assert 0.86 < train / len(assignment) < 0.94

    def test_a_duplicate_group_never_straddles_splits(self):
        """The leak that makes evaluation results *better* and so hides."""
        groups = {f"dup:{i}": [f"trk_{i}a", f"trk_{i}b", f"trk_{i}c"] for i in range(300)}
        assignment = assign_splits(groups, SplitConfig())
        assert verify_no_leakage(groups, assignment) == []

    def test_an_album_never_straddles_splits(self):
        groups = {f"aa:artist{i}|album": [f"trk_{i}_{n}" for n in range(6)] for i in range(200)}
        assignment = assign_splits(groups, SplitConfig())
        assert verify_no_leakage(groups, assignment) == []

    def test_leakage_detection_can_actually_fail(self):
        """A check that cannot fail proves nothing."""
        groups = {"g": ["a", "b"]}
        assert verify_no_leakage(groups, {"a": "TRAIN", "b": "TEST"}) == ["g"]


class TestGrouping:
    def test_a_duplicate_group_is_the_strongest_signal(self):
        key = group_key(
            track_id="trk_a",
            duplicate_group_id="dup_1",
            artist="Someone",
            album="Record",
            parent_directory="folder",
        )
        assert key.startswith("dup:")

    def test_artist_and_album_group_together(self):
        common = {"duplicate_group_id": None, "artist": "A", "album": "B", "parent_directory": "x"}
        assert group_key(track_id="t1", **common) == group_key(track_id="t2", **common)

    def test_grouping_is_case_and_form_insensitive(self):
        """macOS hands back decomposed Korean; a sidecar supplies NFC."""
        left = group_key(
            track_id="a",
            duplicate_group_id=None,
            artist="Artist",
            album="Album",
            parent_directory=None,
        )
        right = group_key(
            track_id="b",
            duplicate_group_id=None,
            artist="  ARTIST ",
            album="album",
            parent_directory=None,
        )
        assert left == right

    def test_a_lone_track_groups_on_itself(self):
        key = group_key(
            track_id="trk_a",
            duplicate_group_id=None,
            artist=None,
            album=None,
            parent_directory=None,
        )
        assert key == "t:trk_a"

    def test_groups_build_from_pairs(self):
        assert build_groups([("a", "g1"), ("b", "g1"), ("c", "g2")]) == {
            "g1": ["a", "b"],
            "g2": ["c"],
        }


class TestClassificationHonesty:
    def test_vocals_are_uncertain_without_a_statement(self):
        """No validated detector exists, so none is claimed."""
        result = assess_vocals(None)
        assert result.vocal_class == "UNCERTAIN"
        assert result.vocal_source == "NONE"
        assert "no validated vocal detector" in result.reason

    def test_an_operator_statement_is_authoritative(self):
        sidecar = Sidecar(path="s.json", fields={"vocal_type": "instrumental"})
        result = assess_vocals(sidecar)
        assert result.vocal_class == "INSTRUMENTAL"
        assert result.vocal_source == "USER"

    def test_gender_is_only_ever_the_operators_word(self):
        assert assess_vocals(None).vocal_gender is None
        sidecar = Sidecar(path="s.json", fields={"vocal_type": "female"})
        result = assess_vocals(sidecar)
        assert result.vocal_gender == "female"
        assert result.vocal_gender_source == "USER"

    def test_an_unrecognised_vocal_type_does_not_force_a_bucket(self):
        sidecar = Sidecar(path="s.json", fields={"vocal_type": "whistling"})
        assert assess_vocals(sidecar).vocal_class == "UNCERTAIN"

    def test_language_is_unknown_without_evidence(self):
        result = assess_language(None, {})
        assert result.language == "unknown"
        assert "folder name is not evidence" in result.reason

    def test_language_comes_from_a_statement(self):
        sidecar = Sidecar(path="s.json", fields={"language": "ko"})
        result = assess_language(sidecar, {})
        assert result.language == "ko"
        assert result.language_source == "USER"

    def test_language_can_be_read_from_supplied_lyrics(self):
        """A fact about the text, never a guess about the audio."""
        result = assess_language(None, {}, lyrics="사랑은 언제나 그렇게 찾아오는 거야")
        assert result.language == "ko"
        assert "not analysis of the audio" in result.reason

    def test_lyrics_are_never_generated(self):
        result = assess_text(None, None, {})
        assert result.lyrics is None
        assert result.transcript is None
        assert any("absent rather than generated" in note for note in result.notes)

    def test_supplied_lyrics_are_kept_with_their_source(self):
        sidecar = Sidecar(path="s.json", fields={"lyrics": "[Verse]\nline"})
        result = assess_text(sidecar, None, {})
        assert result.lyrics == "[Verse]\nline"
        assert result.lyrics_source == "USER"
