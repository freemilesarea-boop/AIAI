"""Tracks are split first, then windowed. The order is the safety property.

Window a library and split the windows afterwards and two views of the
same recording can land on opposite sides of a held-out boundary. The
summary table looks identical either way; the measurement is worthless.
These tests assert the correct order holds and demonstrate what the
wrong order would have produced.
"""

from luber_dataset.splits import build_experiment_splits, leakage_report
from luber_dataset.windows import plan_windows


def _library(count: int = 128) -> dict:
    tracks = []
    for index in range(1, count + 1):
        tracks.append(
            {
                "track_id": f"track-{index:03d}",
                "audio_sha256": f"{index:064x}",
                "source_group": "POP" if index % 2 else "Lofi",
                # Long enough that most tracks yield several windows,
                # which is when leakage would actually bite.
                "duration_seconds": 130.0 + (index % 9) * 32.0,
                "training_allowed": True,
            }
        )
    return {"dataset_id": "LIB", "content_hash": "c" * 64, "tracks": tracks}


def _split_then_window(seed: int = 37):
    library = _library()
    splits = build_experiment_splits(
        library, train_size=64, validation_size=8, evaluation_size=8, seed=seed
    )
    by_id = {t["track_id"]: t for t in library["tracks"]}
    return {
        split.name: plan_windows(
            [by_id[m.track_id] for m in split.members], split=split.name, seed=seed
        )
        for split in splits.splits
    }


class TestTracksAreSplitFirst:
    def test_no_source_track_appears_in_two_splits(self):
        manifests = _split_then_window()
        seen: dict[str, str] = {}
        for name, manifest in manifests.items():
            for track_id in manifest.track_ids:
                assert track_id not in seen, f"{track_id} in {seen.get(track_id)} and {name}"
                seen[track_id] = name

    def test_no_audio_digest_appears_in_two_splits(self):
        manifests = _split_then_window()
        seen: dict[str, str] = {}
        for name, manifest in manifests.items():
            for window in manifest.windows:
                previous = seen.setdefault(window.audio_sha256, name)
                assert previous == name

    def test_every_window_of_a_track_stays_with_that_track(self):
        """A four-window track contributes four windows to one split."""
        manifests = _split_then_window()
        for manifest in manifests.values():
            for track_id, count in manifest.windows_per_track.items():
                owned = [w for w in manifest.windows if w.track_id == track_id]
                assert len(owned) == count

    def test_the_track_split_itself_is_clean(self):
        library = _library()
        splits = build_experiment_splits(
            library, train_size=64, validation_size=8, evaluation_size=8, seed=37
        )
        assert leakage_report(splits).passed


class TestWhatTheWrongOrderWouldDo:
    def test_windowing_first_would_put_one_recording_in_two_splits(self):
        """Not a test of our code — a demonstration of why the order matters.

        Windows are given distinct ids by construction, so a splitter
        that treats them as independent items has no way to know two of
        them are the same song, and will happily separate them.
        """
        library = _library()
        everything = plan_windows(library["tracks"], split="ALL", seed=37)
        multi = [t for t, n in everything.windows_per_track.items() if n > 1]
        assert multi, "the fixture must contain multi-window tracks for this to mean anything"

        as_items = {
            "dataset_id": "LIB",
            "content_hash": "c" * 64,
            "tracks": [
                {
                    # Each window pretending to be its own track, which
                    # is exactly the mistake.
                    "track_id": w.window_id,
                    "audio_sha256": w.window_id.replace("-", "").ljust(64, "0")[:64],
                    "source_group": w.source_group,
                    "duration_seconds": w.duration_seconds,
                    "training_allowed": True,
                }
                for w in everything.windows
            ],
        }
        window_splits = build_experiment_splits(
            as_items, train_size=64, validation_size=8, evaluation_size=8, seed=37
        )
        origin = {w.window_id: w.track_id for w in everything.windows}
        placement: dict[str, set[str]] = {}
        for split in window_splits.splits:
            for member in split.members:
                placement.setdefault(origin[member.track_id], set()).add(split.name)
        assert any(len(names) > 1 for names in placement.values()), (
            "windowing before splitting should be able to separate one recording; "
            "if this ever stops being true the demonstration needs rewriting"
        )

    def test_and_our_order_does_not(self):
        manifests = _split_then_window()
        placement: dict[str, set[str]] = {}
        for name, manifest in manifests.items():
            for window in manifest.windows:
                placement.setdefault(window.track_id, set()).add(name)
        assert all(len(names) == 1 for names in placement.values())


class TestDeterminism:
    def test_the_same_seed_gives_the_same_windows_in_the_same_splits(self):
        first, second = _split_then_window(37), _split_then_window(37)
        assert {k: v.digest() for k, v in first.items()} == {
            k: v.digest() for k, v in second.items()
        }

    def test_a_different_seed_moves_tracks_but_keeps_them_whole(self):
        other = _split_then_window(38)
        placement: dict[str, set[str]] = {}
        for name, manifest in other.items():
            for window in manifest.windows:
                placement.setdefault(window.track_id, set()).add(name)
        assert all(len(names) == 1 for names in placement.values())
