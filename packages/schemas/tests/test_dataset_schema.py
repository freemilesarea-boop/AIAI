"""The gates that stop a file becoming training data by accident.

Most of these assert a refusal. That asymmetry is the point: training on
material the project has no right to, or on the benchmark it is measured
against, are both mistakes that are invisible afterwards — the model is
already trained and the score already meaningless. A refusal is cheap.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from luber_schemas import (
    DataRights,
    DatasetItem,
    DatasetManifest,
    DataSplit,
    QualityTier,
    SourceType,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def item(**overrides) -> DatasetItem:
    base = {
        "item_id": "song-1",
        "audio_path": "korean/song-1.wav",
        "sha256": DIGEST_A,
        "source_type": SourceType.USER_PROVIDED,
        "ingested_at": "2026-08-16T00:00:00Z",
        "duration_seconds": 180.0,
        "sample_rate": 48_000,
        "channels": 2,
    }
    base.update(overrides)
    return DatasetItem(**base)


class TestPaths:
    def test_an_absolute_path_is_refused(self):
        with pytest.raises(ValidationError, match="relative to the dataset root"):
            item(audio_path="/Users/someone/music/song.wav")

    def test_a_home_relative_path_is_refused(self):
        with pytest.raises(ValidationError):
            item(audio_path="~/music/song.wav")

    def test_a_path_escaping_the_root_is_refused(self):
        with pytest.raises(ValidationError, match="escape"):
            item(audio_path="../../etc/passwd")

    def test_a_relative_path_is_accepted(self):
        assert item(audio_path="ko/ballad/01.wav").audio_path == "ko/ballad/01.wav"


class TestHashes:
    def test_a_malformed_digest_is_refused(self):
        with pytest.raises(ValidationError, match="sha256"):
            item(sha256="not-a-hash")

    def test_uppercase_is_refused_so_comparison_stays_trivial(self):
        with pytest.raises(ValidationError):
            item(sha256="A" * 64)

    def test_the_pcm_digest_is_optional(self):
        assert item(pcm_sha256=None).pcm_sha256 is None


class TestRightsGate:
    @pytest.mark.parametrize(
        "rights",
        [DataRights.UNKNOWN, DataRights.DO_NOT_TRAIN, DataRights.REFERENCE_ONLY],
    )
    def test_unestablished_or_forbidden_rights_cannot_reach_train(self, rights):
        with pytest.raises(ValidationError, match="may not be used for training"):
            item(rights=rights, quality_tier=QualityTier.GOLD, split=DataSplit.TRAIN)

    @pytest.mark.parametrize(
        "rights",
        [
            DataRights.OWNED,
            DataRights.LICENSED_FOR_TRAINING,
            DataRights.PUBLIC_DOMAIN,
            DataRights.AI_GENERATED_ALLOWED,
        ],
    )
    def test_established_rights_may_train(self, rights):
        assert item(
            rights=rights, quality_tier=QualityTier.GOLD, split=DataSplit.TRAIN
        ).is_trainable

    def test_unknown_is_the_default(self):
        """A file arrives with no provenance until someone supplies it."""
        assert item().rights is DataRights.UNKNOWN
        assert item().is_trainable is False

    def test_reference_only_material_is_never_trainable(self):
        """Analysis material must not drift into the training set."""
        reference = item(rights=DataRights.REFERENCE_ONLY, quality_tier=QualityTier.GOLD)
        assert reference.is_trainable is False


class TestQualityGate:
    def test_a_reject_item_cannot_reach_train(self):
        with pytest.raises(ValidationError, match="REJECT"):
            item(rights=DataRights.OWNED, quality_tier=QualityTier.REJECT, split=DataSplit.TRAIN)

    def test_rights_and_quality_are_independent_axes(self):
        """Perfectly licensed and still too poor to train on."""
        allowed_but_bad = item(rights=DataRights.OWNED, quality_tier=QualityTier.REJECT)
        assert allowed_but_bad.is_trainable is False


class TestSplitGate:
    def test_evaluation_only_material_is_not_trainable(self):
        held_back = item(
            rights=DataRights.OWNED,
            quality_tier=QualityTier.GOLD,
            split=DataSplit.EVALUATION_ONLY,
        )
        assert held_back.is_trainable is False

    @pytest.mark.parametrize("split", [DataSplit.VALIDATION, DataSplit.TEST])
    def test_validation_and_test_are_not_training_data(self, split):
        assert not item(
            rights=DataRights.OWNED, quality_tier=QualityTier.GOLD, split=split
        ).is_trainable


class TestVocalConsistency:
    def test_an_instrumental_cannot_carry_lyrics(self):
        with pytest.raises(ValidationError, match="instrumental"):
            item(instrumental=True, lyrics="[Verse]\n노래")

    def test_a_vocal_item_without_lyrics_is_allowed(self):
        """Unannotated, not invalid — it simply cannot reach GOLD later."""
        assert item(instrumental=False, lyrics=None).lyrics is None

    def test_korean_metadata_round_trips(self):
        korean = item(
            language="ko",
            lyrics="[Verse]\n작은 방 안에 햇살이 들어와",
            lyrics_source="written for this dataset",
            genre_tags=["kpop", "ballad"],
        )
        assert korean.language == "ko"
        assert "햇살" in korean.lyrics
        assert korean.genre_tags == ["kpop", "ballad"]


class TestManifest:
    def manifest(self, items):
        return DatasetManifest(dataset_name="t", created_at="2026-08-16T00:00:00Z", items=items)

    def test_duplicate_item_ids_are_refused(self):
        with pytest.raises(ValidationError, match="item_id"):
            self.manifest([item(), item(audio_path="other.wav")])

    def test_duplicate_paths_are_refused(self):
        with pytest.raises(ValidationError, match="audio_path"):
            self.manifest([item(), item(item_id="song-2")])

    def test_the_same_audio_in_two_splits_is_refused(self):
        """Leakage detected by content, not by filename.

        The same recording under two names in TRAIN and TEST is how a
        benchmark quietly stops measuring anything.
        """
        train = item(
            item_id="a",
            audio_path="a.wav",
            sha256=DIGEST_A,
            rights=DataRights.OWNED,
            quality_tier=QualityTier.GOLD,
            split=DataSplit.TRAIN,
        )
        leaked = item(
            item_id="b",
            audio_path="b.wav",
            sha256=DIGEST_A,
            split=DataSplit.TEST,
        )
        with pytest.raises(ValidationError, match="more than one split"):
            self.manifest([train, leaked])

    def test_leakage_is_caught_through_the_decoded_pcm_too(self):
        """A re-encode has a different file hash and the same samples."""
        train = item(
            item_id="a",
            audio_path="a.wav",
            sha256=DIGEST_A,
            pcm_sha256=DIGEST_B,
            rights=DataRights.OWNED,
            quality_tier=QualityTier.GOLD,
            split=DataSplit.TRAIN,
        )
        reencoded = item(
            item_id="b",
            audio_path="b.flac",
            sha256="c" * 64,
            pcm_sha256=DIGEST_B,
            split=DataSplit.TEST,
        )
        with pytest.raises(ValidationError, match="more than one split"):
            self.manifest([train, reencoded])

    def test_distinct_audio_across_splits_is_fine(self):
        train = item(
            item_id="a",
            audio_path="a.wav",
            sha256=DIGEST_A,
            rights=DataRights.OWNED,
            quality_tier=QualityTier.GOLD,
            split=DataSplit.TRAIN,
        )
        test = item(item_id="b", audio_path="b.wav", sha256=DIGEST_B, split=DataSplit.TEST)
        assert len(self.manifest([train, test]).items) == 2

    def test_trainable_and_quarantined_partition_the_manifest(self):
        good = item(
            item_id="a",
            audio_path="a.wav",
            sha256=DIGEST_A,
            rights=DataRights.OWNED,
            quality_tier=QualityTier.GOLD,
            split=DataSplit.TRAIN,
        )
        unknown = item(item_id="b", audio_path="b.wav", sha256=DIGEST_B)
        manifest = self.manifest([good, unknown])
        assert [i.item_id for i in manifest.trainable_items()] == ["a"]
        assert [i.item_id for i in manifest.quarantined_items()] == ["b"]


class TestSyntheticAccounting:
    def test_generated_audio_is_flagged_however_it_is_labelled(self):
        by_source = item(source_type=SourceType.LUBER_GENERATED)
        by_rights = item(rights=DataRights.AI_GENERATED_ALLOWED)
        assert by_source.is_synthetic and by_rights.is_synthetic

    def test_human_material_is_not_flagged(self):
        assert item(source_type=SourceType.USER_PROVIDED).is_synthetic is False

    def test_the_synthetic_share_of_training_data_is_measurable(self):
        """Model collapse is a risk you can only manage if you count it."""
        synthetic = item(
            item_id="s",
            audio_path="s.wav",
            sha256=DIGEST_A,
            source_type=SourceType.LUBER_GENERATED,
            rights=DataRights.AI_GENERATED_ALLOWED,
            quality_tier=QualityTier.SILVER,
            split=DataSplit.TRAIN,
        )
        human = item(
            item_id="h",
            audio_path="h.wav",
            sha256=DIGEST_B,
            source_type=SourceType.USER_PROVIDED,
            rights=DataRights.OWNED,
            quality_tier=QualityTier.GOLD,
            split=DataSplit.TRAIN,
        )
        manifest = DatasetManifest(
            dataset_name="t", created_at="2026-08-16T00:00:00Z", items=[synthetic, human]
        )
        assert manifest.synthetic_fraction() == 0.5

    def test_an_empty_training_set_reports_zero_rather_than_dividing(self):
        empty = DatasetManifest(dataset_name="t", created_at="2026-08-16T00:00:00Z", items=[])
        assert empty.synthetic_fraction() == 0.0
