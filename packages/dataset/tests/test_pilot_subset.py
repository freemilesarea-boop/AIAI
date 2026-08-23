"""Choosing which few tracks a bounded pilot trains on.

The property under test is boring and load-bearing: the same library
and the same size always give the same tracks. A pilot whose subset
depends on directory iteration order is not an experiment anyone can
repeat, and the measurements taken on it describe nothing in
particular.
"""

import pytest

from luber_dataset import (
    PILOT_SUBSET_MAX,
    PILOT_SUBSET_MIN,
    SubsetError,
    select_pilot_subset,
)


def _track(index: int, *, group: str, trainable: bool = True) -> dict:
    return {
        "track_id": f"track-{index:03d}",
        "audio_sha256": f"{index:064x}",
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
    return {
        "dataset_id": "LIB",
        "content_hash": "c" * 64,
        "tracks": tracks,
    }


class TestItIsDeterministic:
    def test_the_same_library_and_size_give_the_same_subset(self):
        library = _library({"A": 5, "B": 5})
        first = select_pilot_subset(library, size=4)
        second = select_pilot_subset(library, size=4)
        assert first.track_ids == second.track_ids
        assert first.digest() == second.digest()

    def test_shuffling_the_manifest_does_not_change_the_subset(self):
        """Selection follows the audio, not the order it was listed in."""
        library = _library({"A": 5, "B": 5})
        shuffled = dict(library)
        shuffled["tracks"] = list(reversed(library["tracks"]))
        assert (
            select_pilot_subset(library, size=4).track_ids
            == select_pilot_subset(shuffled, size=4).track_ids
        )

    def test_the_digest_covers_the_audio_and_not_the_track_ids(self):
        library = _library({"A": 4})
        renamed = {
            **library,
            "tracks": [{**t, "track_id": t["track_id"].upper()} for t in library["tracks"]],
        }
        assert (
            select_pilot_subset(library, size=3).digest()
            == select_pilot_subset(renamed, size=3).digest()
        )

    def test_different_audio_gives_a_different_digest(self):
        a = select_pilot_subset(_library({"A": 4}), size=3)
        b = select_pilot_subset(_library({"A": 5}), size=4)
        assert a.digest() != b.digest()


class TestItStratifies:
    def test_it_draws_from_every_group_rather_than_the_first_one(self):
        subset = select_pilot_subset(_library({"A": 10, "B": 10}), size=4)
        assert subset.group_distribution == {"A": 2, "B": 2}

    def test_an_exhausted_group_does_not_block_the_subset(self):
        subset = select_pilot_subset(_library({"A": 1, "B": 10}), size=4)
        assert subset.group_distribution == {"A": 1, "B": 3}

    def test_it_can_be_restricted_to_named_groups(self):
        subset = select_pilot_subset(_library({"A": 10, "B": 10}), size=3, groups=("B",))
        assert subset.group_distribution == {"B": 3}

    def test_naming_a_group_that_is_not_there_is_an_error(self):
        with pytest.raises(SubsetError, match="not in the library"):
            select_pilot_subset(_library({"A": 10}), size=3, groups=("C",))


class TestItStaysBounded:
    @pytest.mark.parametrize("size", [0, PILOT_SUBSET_MIN - 1, PILOT_SUBSET_MAX + 1, 500])
    def test_it_refuses_a_size_outside_the_bounded_range(self, size):
        with pytest.raises(SubsetError, match="bounded range"):
            select_pilot_subset(_library({"A": 600}), size=size)

    def test_it_refuses_when_there_are_not_enough_candidates(self):
        with pytest.raises(SubsetError, match="only 2"):
            select_pilot_subset(_library({"A": 2}), size=3)


class TestItRespectsTheRightsVerdict:
    def test_a_track_not_cleared_for_training_is_not_a_candidate(self):
        library = _library({"A": 4})
        library["tracks"][0]["training_allowed"] = False
        subset = select_pilot_subset(library, size=3)
        assert library["tracks"][0]["track_id"] not in subset.track_ids

    def test_barred_tracks_do_not_count_towards_the_size(self):
        library = _library({"A": 4})
        for track in library["tracks"][:2]:
            track["training_allowed"] = False
        with pytest.raises(SubsetError, match="only 2"):
            select_pilot_subset(library, size=3)

    def test_a_track_without_a_content_digest_is_refused_not_skipped(self):
        """No identity means no selection, and silence would hide it."""
        library = _library({"A": 4})
        library["tracks"][0]["audio_sha256"] = ""
        with pytest.raises(SubsetError, match="content identity"):
            select_pilot_subset(library, size=3)
