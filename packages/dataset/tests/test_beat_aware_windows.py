"""Beat-aware window starts and arrangement-weighted sampling.

Phase 37 spread window starts evenly, which can open a window halfway
through a snare hit: the model's first frames are the tail of an event
whose attack it never saw. Phase 38 nudges each start onto a nearby
onset — but only nearby, because the even spread is what decides which
*parts of the song* get covered and snapping must not undo that.
"""

import pytest

from luber_dataset.windows import (
    DEFAULT_WINDOW_FRAMES,
    MAX_BEAT_SNAP_FRAMES,
    Window,
    WindowError,
    WindowManifest,
    arrangement_weighted_sampling,
    beat_aware_offsets,
    rank_windows_by_arrangement,
    snap_to_beat,
    window_offsets,
)

W = DEFAULT_WINDOW_FRAMES


def _window(index: int, track: str, *, start: int = 0) -> Window:
    return Window(
        window_id=f"{track}-w{index}",
        track_id=track,
        audio_sha256=f"{abs(hash(track)) % 10**10:064d}",
        source_group="POP",
        window_index=index,
        window_count=4,
        position=f"BEAT_AWARE_{index}",
        start_frame=start,
        end_frame=start + W,
        latent_frames=W,
        track_frames=20_000,
        experiment_seed=38,
    )


class TestSnapping:
    def test_a_start_moves_onto_a_nearby_onset(self):
        frame, reason = snap_to_beat(1_000, [980, 1_500])
        assert frame == 980
        assert reason == "SNAPPED"

    def test_it_takes_the_nearest_one(self):
        frame, _ = snap_to_beat(1_000, [960, 1_010, 1_200])
        assert frame == 1_010

    def test_a_start_already_on_an_onset_does_not_move(self):
        frame, reason = snap_to_beat(1_000, [1_000, 1_020])
        assert frame == 1_000
        assert reason == "ALREADY_ON_ONSET"

    def test_a_distant_onset_does_not_drag_the_window(self):
        """The even spread decides coverage; snapping only fine-tunes."""
        frame, reason = snap_to_beat(1_000, [5_000])
        assert frame == 1_000
        assert reason == "NO_ONSET_IN_RANGE"

    def test_the_shift_is_bounded(self):
        frame, _ = snap_to_beat(1_000, [1_000 + MAX_BEAT_SNAP_FRAMES + 1])
        assert frame == 1_000

    def test_a_track_with_no_onsets_keeps_its_even_spacing(self):
        frame, reason = snap_to_beat(1_000, [])
        assert frame == 1_000
        assert reason == "NO_ONSETS"

    def test_it_never_snaps_past_a_supplied_limit(self):
        """A window may not start so late that it runs off the end."""
        frame, _ = snap_to_beat(1_000, [1_030], limit=1_010)
        assert frame == 1_000


class TestBeatAwareOffsets:
    def test_every_start_lands_on_an_onset_when_onsets_are_dense(self):
        frames = 20_000
        onsets = list(range(0, frames, 40))
        placed = beat_aware_offsets(frames, 4, onset_frames=onsets, window_frames=W)
        assert all(start in onsets for start, _ in placed)

    def test_windows_still_cover_the_whole_track(self):
        frames = 20_000
        onsets = list(range(0, frames, 40))
        placed = beat_aware_offsets(frames, 4, onset_frames=onsets, window_frames=W)
        starts = [start for start, _ in placed]
        assert starts[0] < W
        assert starts[-1] + W <= frames
        assert starts == sorted(starts)

    def test_no_window_runs_past_the_end_of_the_track(self):
        frames = 9_437
        onsets = list(range(0, frames, 7))
        for start, _ in beat_aware_offsets(frames, 3, onset_frames=onsets, window_frames=W):
            assert 0 <= start
            assert start + W <= frames

    def test_two_windows_never_collapse_onto_one_frame(self):
        # One onset that both windows would want.
        placed = beat_aware_offsets(W + 20, 2, onset_frames=[10], window_frames=W)
        starts = [start for start, _ in placed]
        assert len(set(starts)) == len(starts)
        assert any(reason == "COLLISION_KEPT_EVEN_SPACING" for _, reason in placed)

    def test_with_no_onsets_it_matches_the_even_spacing_exactly(self):
        frames = 20_000
        placed = beat_aware_offsets(frames, 4, onset_frames=[], window_frames=W)
        assert [start for start, _ in placed] == list(window_offsets(frames, 4, window_frames=W))

    def test_the_reason_for_every_placement_is_recorded(self):
        placed = beat_aware_offsets(20_000, 3, onset_frames=[0, 8_000], window_frames=W)
        assert all(isinstance(reason, str) and reason for _, reason in placed)


class TestArrangementWeighting:
    def _manifest(self) -> WindowManifest:
        return WindowManifest(
            split="TRAIN",
            window_frames=W,
            experiment_seed=38,
            windows=(
                _window(0, "long"),
                _window(1, "long"),
                _window(2, "long"),
                _window(3, "long"),
                _window(0, "short"),
            ),
        )

    def test_every_track_still_contributes_the_same_total(self):
        """The Phase 37 property, kept."""
        manifest = self._manifest()
        scores = {"long-w0": 0.9, "long-w1": 0.1, "long-w2": 0.5, "long-w3": 0.3, "short-w0": 0.2}
        weights = arrangement_weighted_sampling(manifest, scores)
        totals: dict[str, float] = {}
        for window in manifest.windows:
            totals[window.track_id] = totals.get(window.track_id, 0.0) + weights[window.window_id]
        assert all(abs(value - 1.0) < 1e-9 for value in totals.values())

    def test_a_busier_window_gets_more_of_its_track_share(self):
        manifest = self._manifest()
        scores = {"long-w0": 0.9, "long-w1": 0.1, "long-w2": 0.5, "long-w3": 0.3, "short-w0": 0.2}
        weights = arrangement_weighted_sampling(manifest, scores)
        assert weights["long-w0"] > weights["long-w1"]

    def test_zero_emphasis_reproduces_the_flat_split(self):
        manifest = self._manifest()
        scores = {"long-w0": 0.9, "long-w1": 0.1, "long-w2": 0.5, "long-w3": 0.3, "short-w0": 0.2}
        weights = arrangement_weighted_sampling(manifest, scores, emphasis=0.0)
        assert weights["long-w0"] == pytest.approx(0.25)
        assert weights["short-w0"] == pytest.approx(1.0)

    def test_a_negative_emphasis_is_refused(self):
        with pytest.raises(WindowError, match="emptiest"):
            arrangement_weighted_sampling(self._manifest(), {}, emphasis=-1.0)

    def test_an_unscored_window_still_gets_a_share(self):
        weights = arrangement_weighted_sampling(self._manifest(), {})
        assert all(value > 0 for value in weights.values())

    def test_ranking_is_total_and_stable(self):
        windows = [_window(i, "t") for i in range(4)]
        scores = {"t-w0": 0.5, "t-w1": 0.5, "t-w2": 0.9, "t-w3": 0.1}
        first = rank_windows_by_arrangement(windows, scores)
        second = rank_windows_by_arrangement(list(reversed(windows)), scores)
        assert [w.window_id for w in first] == [w.window_id for w in second]
        assert first[0].window_id == "t-w2"
