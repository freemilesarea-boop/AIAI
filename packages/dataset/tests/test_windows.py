"""Multi-window coverage: deterministic, bounded, and traceable to a track.

Phase 36 trained on the first two minutes of every song because Metal
keeps an allocator working set per tensor shape. These tests hold the
two things that let Phase 37 see more of each song without giving that
back: every window is the same length, and every window still knows
which recording it came from.
"""

import itertools

import pytest

from luber_dataset.windows import (
    DEFAULT_WINDOW_FRAMES,
    LATENT_FRAMES_PER_SECOND,
    MAX_OVERLAP_FRACTION,
    MAX_WINDOWS_PER_TRACK,
    WindowError,
    eligible_tracks,
    plan_track_windows,
    plan_windows,
    sampling_weights,
    track_frames,
    window_count_for,
    window_offsets,
)

W = DEFAULT_WINDOW_FRAMES


def _track(index: int, seconds: float, *, group: str = "POP") -> dict:
    return {
        "track_id": f"track-{index:03d}",
        "audio_sha256": f"{index:064x}",
        "source_group": group,
        "duration_seconds": seconds,
        "training_allowed": True,
        "rights": {"basis": "OPERATOR_AUTHORIZED_SCOPE"},
    }


class TestTheFrameRate:
    def test_it_matches_what_the_real_pipeline_produces(self):
        """Measured, not assumed: 178.80 s preprocesses to 4470 frames."""
        assert LATENT_FRAMES_PER_SECOND == 25.0
        assert track_frames(178.80) == 4470
        assert track_frames(121.80) == 3045
        assert track_frames(120.0) == W

    def test_a_partial_frame_is_not_a_frame(self):
        """Rounding up would put a window past the end of the audio."""
        assert track_frames(120.039) == W


class TestHowManyWindows:
    @pytest.mark.parametrize(
        ("frames", "expected"),
        [
            # n windows are admitted when the even spacing
            # (frames - W) / (n - 1) is at least half a window, which
            # puts the thresholds at W, 1.5W, 2W and 2.5W.
            (W - 1, 0),
            (W, 1),
            (int(W * 1.4), 1),
            (int(W * 1.5), 2),
            (int(W * 1.9), 2),
            (int(W * 2.0), 3),
            (int(W * 2.4), 3),
            (int(W * 2.5), 4),
            (int(W * 12), MAX_WINDOWS_PER_TRACK),
        ],
    )
    def test_the_count_follows_from_the_overlap_rule(self, frames, expected):
        assert window_count_for(frames) == expected

    def test_a_track_shorter_than_one_window_yields_none(self):
        """Padding it would train the model on inserted silence."""
        assert plan_track_windows(_track(1, 100.0), seed=37) == ()

    def test_no_track_ever_exceeds_the_cap(self):
        for seconds in (400.0, 1_200.0, 10_000.0):
            assert len(plan_track_windows(_track(1, seconds), seed=37)) <= MAX_WINDOWS_PER_TRACK

    def test_consecutive_windows_never_overlap_by_more_than_half(self):
        for seconds in (120.0, 150.0, 190.0, 245.0, 300.0, 377.5, 900.0):
            windows = plan_track_windows(_track(1, seconds), seed=37)
            starts = [w.start_frame for w in windows]
            for first, second in itertools.pairwise(starts):
                assert second - first >= W * MAX_OVERLAP_FRACTION

    def test_a_short_track_does_not_get_duplicate_windows(self):
        windows = plan_track_windows(_track(1, 140.0), seed=37)
        assert len(windows) == 1
        assert {w.start_frame for w in windows} == {0}

    def test_a_long_track_covers_more_than_its_opening(self):
        """The whole point of the phase."""
        windows = plan_track_windows(_track(1, 377.5), seed=37)
        assert len(windows) >= 3
        assert max(w.end_frame for w in windows) == track_frames(377.5)
        assert min(w.start_frame for w in windows) == 0


class TestOffsets:
    def test_windows_never_run_past_the_source(self):
        for seconds in (120.0, 190.0, 245.0, 377.5):
            frames = track_frames(seconds)
            for window in plan_track_windows(_track(1, seconds), seed=37):
                assert window.start_frame >= 0
                assert window.end_frame <= frames

    def test_the_first_window_starts_at_zero_and_the_last_ends_at_the_end(self):
        frames = track_frames(300.0)
        offsets = window_offsets(frames, 4)
        assert offsets[0] == 0
        assert offsets[-1] + W == frames

    def test_a_single_window_starts_at_zero(self):
        assert window_offsets(track_frames(130.0), 1) == (0,)

    def test_a_track_too_short_for_a_window_is_refused_not_padded(self):
        with pytest.raises(WindowError, match="cannot yield"):
            window_offsets(W - 1, 1)


class TestDeterminism:
    def test_the_same_track_and_seed_give_the_same_windows(self):
        first = plan_track_windows(_track(1, 245.0), seed=37)
        second = plan_track_windows(_track(1, 245.0), seed=37)
        assert [w.to_dict() for w in first] == [w.to_dict() for w in second]

    def test_window_ids_are_derived_from_audio_seed_and_index(self):
        a = plan_track_windows(_track(1, 245.0), seed=37)
        b = plan_track_windows(_track(1, 245.0), seed=38)
        assert [w.window_id for w in a] != [w.window_id for w in b]
        assert [w.start_frame for w in a] == [w.start_frame for w in b]

    def test_different_audio_never_shares_a_window_id(self):
        a = plan_track_windows(_track(1, 245.0), seed=37)
        b = plan_track_windows(_track(2, 245.0), seed=37)
        assert not ({w.window_id for w in a} & {w.window_id for w in b})

    def test_manifest_order_does_not_change_the_manifest(self):
        tracks = [_track(i, 150.0 + i * 30) for i in range(1, 9)]
        assert (
            plan_windows(tracks, split="TRAIN", seed=37).digest()
            == plan_windows(list(reversed(tracks)), split="TRAIN", seed=37).digest()
        )

    def test_a_track_without_a_digest_is_refused(self):
        broken = {**_track(1, 200.0), "audio_sha256": ""}
        with pytest.raises(WindowError, match="audio_sha256"):
            plan_track_windows(broken, seed=37)


class TestFixedShape:
    def test_every_window_in_a_manifest_is_the_same_length(self):
        tracks = [_track(i, 120.0 + i * 17.3) for i in range(1, 20)]
        manifest = plan_windows(tracks, split="TRAIN", seed=37)
        assert manifest.unique_latent_frames == (W,)

    def test_the_shape_count_is_one_however_varied_the_durations(self):
        """The Phase 36 OOM, guarded at the representation level."""
        tracks = [_track(i, 120.0 + i * 3.7) for i in range(1, 60)]
        manifest = plan_windows(tracks, split="TRAIN", seed=37)
        assert len(manifest.unique_latent_frames) == 1

    def test_a_custom_window_length_is_still_uniform(self):
        tracks = [_track(i, 200.0 + i * 11) for i in range(1, 10)]
        manifest = plan_windows(tracks, split="TRAIN", seed=37, window_frames=2000)
        assert manifest.unique_latent_frames == (2000,)


class TestProvenance:
    def test_a_window_knows_the_recording_it_came_from(self):
        window = plan_track_windows(_track(7, 245.0), seed=37)[1]
        assert window.track_id == "track-007"
        assert window.audio_sha256 == f"{7:064x}"
        assert window.source_group == "POP"
        assert window.authorization_basis == "OPERATOR_AUTHORIZED_SCOPE"

    def test_every_field_a_later_reader_needs_is_recorded(self):
        payload = plan_track_windows(_track(1, 245.0), seed=37)[0].to_dict()
        for field in (
            "window_id",
            "track_id",
            "audio_sha256",
            "window_index",
            "window_count",
            "position",
            "start_frame",
            "end_frame",
            "start_seconds",
            "end_seconds",
            "latent_frames",
            "track_frames",
            "experiment_seed",
            "source_group",
            "authorization_basis",
        ):
            assert field in payload, field

    def test_positions_are_named_and_spread(self):
        assert [w.position for w in plan_track_windows(_track(1, 130.0), seed=37)] == ["START"]
        assert [w.position for w in plan_track_windows(_track(1, 200.0), seed=37)] == [
            "START",
            "END",
        ]
        assert next(iter(plan_track_windows(_track(1, 300.0), seed=37))).position == "START"


class TestTrackWeighting:
    def test_every_track_carries_the_same_total_weight(self):
        tracks = [_track(1, 130.0), _track(2, 245.0), _track(3, 377.5)]
        manifest = plan_windows(tracks, split="TRAIN", seed=37)
        weights = sampling_weights(manifest)
        totals: dict[str, float] = {}
        for window in manifest.windows:
            totals[window.track_id] = totals.get(window.track_id, 0.0) + weights[window.window_id]
        assert all(abs(value - 1.0) < 1e-9 for value in totals.values())

    def test_a_four_window_track_does_not_get_four_times_the_influence(self):
        tracks = [_track(1, 130.0), _track(2, 500.0)]
        manifest = plan_windows(tracks, split="TRAIN", seed=37)
        weights = sampling_weights(manifest)
        long_windows = [w for w in manifest.windows if w.track_id == "track-002"]
        assert len(long_windows) > 1
        assert all(weights[w.window_id] < 1.0 for w in long_windows)


class TestEligibility:
    def test_short_tracks_are_reported_rather_than_dropped_silently(self):
        tracks = [_track(1, 100.0), _track(2, 200.0), _track(3, 90.0)]
        keep, drop = eligible_tracks(tracks)
        assert [t["track_id"] for t in keep] == ["track-002"]
        assert [t["track_id"] for t in drop] == ["track-001", "track-003"]

    def test_a_track_exactly_one_window_long_is_eligible(self):
        keep, drop = eligible_tracks([_track(1, 120.0)])
        assert len(keep) == 1 and not drop
