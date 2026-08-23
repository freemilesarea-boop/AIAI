"""Train / validation / evaluation splits, and the contamination they must not have.

The failure these guard against does not raise, does not log, and makes
the numbers look better: a track in both the training set and the
evaluation set turns a generalisation measurement into a memorisation
measurement. So exclusivity is asserted from both ends — the builder's
construction and the independent leakage report — and the tests that
matter most are the ones that deliberately contaminate a split and
check that something notices.
"""

import pytest

from luber_dataset.splits import (
    EVALUATION,
    TRAIN,
    VALIDATION,
    ExperimentSplits,
    Split,
    SplitError,
    SplitMember,
    build_experiment_splits,
    leakage_report,
)


def _track(index: int, *, group: str, trainable: bool = True, digest: str | None = None) -> dict:
    return {
        "track_id": f"track-{index:03d}",
        "audio_sha256": digest or f"{index:064x}",
        "source_group": group,
        "duration_seconds": 100.0 + index,
        "training_allowed": trainable,
    }


def _library(counts: dict[str, int]) -> dict:
    tracks, index = [], 0
    for group, count in counts.items():
        for _ in range(count):
            index += 1
            tracks.append(_track(index, group=group))
    return {"dataset_id": "LIB", "content_hash": "c" * 64, "tracks": tracks}


def _build(library: dict, *, seed: int = 36, sizes=(24, 4, 4)) -> ExperimentSplits:
    train, validation, evaluation = sizes
    return build_experiment_splits(
        library,
        train_size=train,
        validation_size=validation,
        evaluation_size=evaluation,
        seed=seed,
    )


class TestItIsDeterministic:
    def test_the_same_library_and_seed_give_the_same_allocation(self):
        library = _library({"A": 64, "B": 64})
        assert _build(library).digest() == _build(library).digest()

    def test_manifest_order_does_not_change_the_allocation(self):
        """Selection follows the audio, not the order it was written in."""
        library = _library({"A": 64, "B": 64})
        shuffled = {**library, "tracks": list(reversed(library["tracks"]))}
        assert _build(library).to_dict() == _build(shuffled).to_dict()

    def test_a_different_seed_gives_a_different_allocation(self):
        library = _library({"A": 64, "B": 64})
        assert _build(library, seed=36).digest() != _build(library, seed=37).digest()

    def test_the_digest_is_over_audio_not_track_ids(self):
        library = _library({"A": 64, "B": 64})
        renamed = {
            **library,
            "tracks": [{**t, "track_id": t["track_id"].upper()} for t in library["tracks"]],
        }
        assert _build(library).digest() == _build(renamed).digest()


class TestExclusivity:
    def test_no_track_holds_two_roles(self):
        splits = _build(_library({"A": 64, "B": 64}))
        ids = [member.track_id for split in splits.splits for member in split.members]
        assert len(ids) == len(set(ids)) == 32

    def test_no_audio_digest_appears_in_two_splits(self):
        splits = _build(_library({"A": 64, "B": 64}))
        digests = [m.audio_sha256 for split in splits.splits for m in split.members]
        assert len(digests) == len(set(digests))

    def test_a_library_holding_the_same_audio_twice_places_it_once(self):
        """Two paths, one recording. It cannot end up on both sides."""
        library = _library({"A": 40, "B": 40})
        library["tracks"].append(
            {**library["tracks"][0], "track_id": "a-second-filing-of-the-same-audio"}
        )
        splits = _build(library)
        shared = library["tracks"][0]["audio_sha256"]
        placements = [
            split.name
            for split in splits.splits
            for member in split.members
            if member.audio_sha256 == shared
        ]
        assert len(placements) <= 1

    def test_the_requested_sizes_are_what_is_produced(self):
        splits = _build(_library({"A": 64, "B": 64}))
        assert (
            len(splits.train.members),
            len(splits.validation.members),
            len(splits.evaluation.members),
        ) == (24, 4, 4)


class TestItStratifies:
    def test_every_split_draws_from_every_group(self):
        splits = _build(_library({"A": 64, "B": 64}))
        for split in splits.splits:
            assert set(split.group_distribution) == {"A", "B"}

    def test_an_unbalanced_library_still_fills_the_splits(self):
        splits = _build(_library({"A": 3, "B": 60}))
        assert splits.track_count == 32


class TestItRefusesWhatItCannotDoHonestly:
    def test_it_refuses_when_there_are_not_enough_candidates(self):
        with pytest.raises(SplitError, match="only 10"):
            _build(_library({"A": 10}))

    def test_barred_tracks_are_not_candidates(self):
        library = _library({"A": 40})
        for track in library["tracks"][:20]:
            track["training_allowed"] = False
        with pytest.raises(SplitError, match="only 20"):
            _build(library)

    def test_a_track_without_a_content_digest_is_refused_not_skipped(self):
        library = _library({"A": 64})
        library["tracks"][0]["audio_sha256"] = ""
        with pytest.raises(SplitError, match="content identity"):
            _build(library)

    def test_a_split_of_nothing_is_refused(self):
        with pytest.raises(SplitError, match="not a split"):
            _build(_library({"A": 64}), sizes=(0, 0, 0))


class TestTheLeakageReport:
    def test_a_clean_allocation_passes(self):
        report = leakage_report(_build(_library({"A": 64, "B": 64})))
        assert report.passed
        assert not report.findings

    def _contaminate(self, splits: ExperimentSplits, *, target: str) -> ExperimentSplits:
        """Put a training track into another split, as a slip would."""
        stolen = splits.train.members[0]
        current = getattr(splits, target)
        poisoned = Split(name=current.name, members=(*current.members, stolen))
        return ExperimentSplits(
            dataset_id=splits.dataset_id,
            library_content_hash=splits.library_content_hash,
            seed=splits.seed,
            train=splits.train if target != "train" else poisoned,
            validation=splits.validation if target != "validation" else poisoned,
            evaluation=splits.evaluation if target != "evaluation" else poisoned,
        )

    @pytest.mark.parametrize("target", ["validation", "evaluation"])
    def test_a_training_track_in_a_held_out_split_fails(self, target):
        report = leakage_report(
            self._contaminate(_build(_library({"A": 64, "B": 64})), target=target)
        )
        assert not report.passed
        kinds = {finding.kind for finding in report.findings}
        assert "AUDIO_DIGEST_COLLISION" in kinds
        assert "TRACK_ID_COLLISION" in kinds

    def test_the_same_audio_under_a_new_id_still_fails(self):
        """The check a track-id comparison alone would miss."""
        splits = _build(_library({"A": 64, "B": 64}))
        stolen = splits.train.members[0]
        renamed = SplitMember(
            track_id="looks-like-a-different-track",
            audio_sha256=stolen.audio_sha256,
            source_group=stolen.source_group,
            duration_seconds=stolen.duration_seconds,
        )
        poisoned = ExperimentSplits(
            dataset_id=splits.dataset_id,
            library_content_hash=splits.library_content_hash,
            seed=splits.seed,
            train=splits.train,
            validation=splits.validation,
            evaluation=Split(name=EVALUATION, members=(*splits.evaluation.members, renamed)),
        )
        report = leakage_report(poisoned)
        assert not report.passed
        assert {f.kind for f in report.findings} == {"AUDIO_DIGEST_COLLISION"}

    def test_a_duplicate_inside_one_split_is_reported(self):
        splits = _build(_library({"A": 64, "B": 64}))
        doubled = Split(name=TRAIN, members=(*splits.train.members, splits.train.members[0]))
        report = leakage_report(
            ExperimentSplits(
                dataset_id=splits.dataset_id,
                library_content_hash=splits.library_content_hash,
                seed=splits.seed,
                train=doubled,
                validation=splits.validation,
                evaluation=splits.evaluation,
            )
        )
        assert not report.passed
        assert {f.kind for f in report.findings} == {"DUPLICATE_WITHIN_SPLIT"}

    def test_it_names_which_splits_and_which_identities(self):
        report = leakage_report(
            self._contaminate(_build(_library({"A": 64, "B": 64})), target="evaluation")
        )
        finding = report.findings[0]
        assert {finding.left, finding.right} == {TRAIN, EVALUATION}
        assert finding.identities
        assert VALIDATION not in finding.left


class TestSplitDigests:
    def test_each_split_has_its_own_digest(self):
        splits = _build(_library({"A": 64, "B": 64}))
        digests = {split.digest() for split in splits.splits}
        assert len(digests) == 3

    def test_a_split_digest_does_not_depend_on_member_order(self):
        splits = _build(_library({"A": 64, "B": 64}))
        reversed_split = Split(
            name=splits.train.name, members=tuple(reversed(splits.train.members))
        )
        assert reversed_split.digest() == splits.train.digest()

    def test_moving_one_track_changes_the_overall_digest(self):
        library = _library({"A": 64, "B": 64})
        assert _build(library).digest() != _build(library, sizes=(23, 5, 4)).digest()
