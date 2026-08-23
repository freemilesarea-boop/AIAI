"""Deterministic fixed-length windows over variable-length tracks.

Phase 36 trained on the first 120 seconds of every track, because Metal
keeps an allocator working set per tensor shape and a dataset with 24
different latent lengths reached 29 GiB where four tracks at the same
maximum length peaked at 9.4. The fix was one fixed length; the cost was
that 95% of the library's music was never seen.

This module removes the cost without giving back the fix. Every window
is the *same* length, so the shape count stays at one; several windows
are taken from different positions in a long track, so the model sees
more than its opening two minutes.

Three properties the rest of the phase depends on:

**Deterministic.** A window is a pure function of the track digest, the
experiment seed and the window index. No random crop per epoch: the same
manifest always describes the same audio, and an experiment that cannot
be re-derived cannot be compared with another one.

**Bounded overlap.** Windows are spread evenly across a track, and the
count is chosen so consecutive windows never share more than half their
length. Two windows that are 90% the same are not two views of a song,
they are one view and a rounding error.

**Traceable.** A window is not a new song. Every one carries the id and
the digest of the track it came from, its index, its offset, and the
authorisation the track was admitted under, so a split can be checked
for leakage at the level that matters — the source recording.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

#: Measured from the real preprocessing pipeline, not assumed: a 178.80 s
#: track preprocesses to exactly 4470 latent frames.
LATENT_FRAMES_PER_SECOND = 25.0

#: The fixed window Phase 36 established and Phase 37 keeps.
DEFAULT_WINDOW_FRAMES = 3000

#: A track never yields more than this many windows, however long it is.
#: Four positions across a song is coverage; more is the same song
#: counted repeatedly.
MAX_WINDOWS_PER_TRACK = 4

#: Consecutive windows may share at most half their length. This is what
#: sets the duration thresholds: n windows are admitted only when the
#: track is long enough to space them at least this far apart.
MAX_OVERLAP_FRACTION = 0.5

#: Names for where a window sits, in the order they are assigned.
WINDOW_POSITIONS: tuple[str, ...] = ("START", "EARLY_MIDDLE", "MIDDLE", "LATE", "END")


class WindowError(RuntimeError):
    """Raised when a windowing request cannot be satisfied honestly."""


@dataclass(frozen=True)
class Window:
    """One fixed-length view of one track."""

    window_id: str
    track_id: str
    audio_sha256: str
    source_group: str
    window_index: int
    window_count: int
    position: str
    start_frame: int
    end_frame: int
    latent_frames: int
    track_frames: int
    experiment_seed: int
    authorization_basis: str = ""

    @property
    def start_seconds(self) -> float:
        return self.start_frame / LATENT_FRAMES_PER_SECOND

    @property
    def end_seconds(self) -> float:
        return self.end_frame / LATENT_FRAMES_PER_SECOND

    @property
    def duration_seconds(self) -> float:
        return self.latent_frames / LATENT_FRAMES_PER_SECOND

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "track_id": self.track_id,
            "audio_sha256": self.audio_sha256,
            "source_group": self.source_group,
            "window_index": self.window_index,
            "window_count": self.window_count,
            "position": self.position,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_seconds": round(self.start_seconds, 3),
            "end_seconds": round(self.end_seconds, 3),
            "latent_frames": self.latent_frames,
            "duration_seconds": round(self.duration_seconds, 3),
            "track_frames": self.track_frames,
            "experiment_seed": self.experiment_seed,
            "authorization_basis": self.authorization_basis,
        }


@dataclass(frozen=True)
class WindowManifest:
    """Every window in one split, and what it was derived from."""

    split: str
    window_frames: int
    experiment_seed: int
    windows: tuple[Window, ...] = ()

    @property
    def track_ids(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for window in self.windows:
            seen.setdefault(window.track_id, None)
        return tuple(seen)

    @property
    def track_count(self) -> int:
        return len(self.track_ids)

    @property
    def windows_per_track(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for window in self.windows:
            counts[window.track_id] = counts.get(window.track_id, 0) + 1
        return counts

    @property
    def group_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for window in self.windows:
            counts[window.source_group] = counts.get(window.source_group, 0) + 1
        return counts

    @property
    def unique_latent_frames(self) -> tuple[int, ...]:
        """Every distinct tensor length. On MPS this must have one entry."""
        return tuple(sorted({window.latent_frames for window in self.windows}))

    @property
    def total_duration_seconds(self) -> float:
        return sum(window.duration_seconds for window in self.windows)

    def digest(self) -> str:
        """Identity of the whole manifest, over audio and offsets."""
        running = hashlib.sha256()
        running.update(f"{self.split}:{self.window_frames}:{self.experiment_seed}".encode())
        for window in sorted(self.windows, key=lambda item: item.window_id):
            running.update(b"\0")
            running.update(
                f"{window.audio_sha256}:{window.start_frame}:{window.latent_frames}".encode()
            )
        return running.hexdigest()

    def to_dict(self) -> dict[str, Any]:
        per_track = self.windows_per_track
        return {
            "split": self.split,
            "window_frames": self.window_frames,
            "experiment_seed": self.experiment_seed,
            "manifest_digest": self.digest(),
            "track_count": self.track_count,
            "window_count": len(self.windows),
            "windows_per_track": per_track,
            "average_windows_per_track": (
                round(len(self.windows) / self.track_count, 3) if self.track_count else 0.0
            ),
            "maximum_windows_per_track": max(per_track.values()) if per_track else 0,
            "unique_latent_frames": list(self.unique_latent_frames),
            "group_distribution": self.group_distribution,
            "total_duration_seconds": round(self.total_duration_seconds, 3),
            "windows": [window.to_dict() for window in self.windows],
        }


def track_frames(duration_seconds: float) -> int:
    """Latent frames a track of this length preprocesses to.

    Rounded down: a partial frame is not a frame, and claiming one would
    put a window one sample past the end of the audio.
    """
    return int(duration_seconds * LATENT_FRAMES_PER_SECOND)


def window_count_for(frames: int, *, window_frames: int = DEFAULT_WINDOW_FRAMES) -> int:
    """How many windows a track of *frames* supports.

    Derived from the overlap rule rather than chosen by hand. ``n``
    windows spread evenly across a track are spaced ``(frames -
    window) / (n - 1)`` apart, and ``n`` is admitted only while that
    spacing is at least half a window. A track shorter than one window
    supports none: padding it would train the model on silence somebody
    inserted.
    """
    if window_frames <= 0:
        raise WindowError("a window of no frames is not a window")
    if frames < window_frames:
        return 0
    minimum_stride = window_frames * MAX_OVERLAP_FRACTION
    count = 1
    while count < MAX_WINDOWS_PER_TRACK:
        candidate = count + 1
        stride = (frames - window_frames) / (candidate - 1)
        if stride < minimum_stride:
            break
        count = candidate
    return count


def window_offsets(
    frames: int, count: int, *, window_frames: int = DEFAULT_WINDOW_FRAMES
) -> tuple[int, ...]:
    """Where each window starts. Evenly spread, never past the end."""
    if count <= 0:
        return ()
    if frames < window_frames:
        raise WindowError(f"a {frames}-frame track cannot yield a {window_frames}-frame window")
    last = frames - window_frames
    if count == 1:
        return (0,)
    return tuple(round(last * index / (count - 1)) for index in range(count))


def _window_id(audio_sha256: str, seed: int, index: int) -> str:
    """A window's name, derived from what defines it.

    The digest of the audio, the experiment seed and the index — so the
    same three inputs always name the same window, and two windows of
    different audio can never collide.
    """
    digest = hashlib.sha256(f"{audio_sha256}:{seed}:{index}".encode()).hexdigest()
    return f"{audio_sha256[:16]}-w{index}-{digest[:8]}"


def plan_track_windows(
    track: dict[str, Any],
    *,
    seed: int,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    max_windows: int = MAX_WINDOWS_PER_TRACK,
) -> tuple[Window, ...]:
    """Every window one track yields. Empty if it is shorter than one."""
    digest = str(track.get("audio_sha256", ""))
    if len(digest) != 64:
        raise WindowError(
            f"track {track.get('track_id', '?')!r} has no usable audio_sha256; "
            "a window is named after its audio and cannot be named without it"
        )
    frames = track_frames(float(track.get("duration_seconds", 0.0)))
    count = min(max_windows, window_count_for(frames, window_frames=window_frames))
    if count == 0:
        return ()

    offsets = window_offsets(frames, count, window_frames=window_frames)
    positions = _positions_for(count)
    return tuple(
        Window(
            window_id=_window_id(digest, seed, index),
            track_id=str(track["track_id"]),
            audio_sha256=digest,
            source_group=str(track.get("source_group", "")),
            window_index=index,
            window_count=count,
            position=positions[index],
            start_frame=start,
            end_frame=start + window_frames,
            latent_frames=window_frames,
            track_frames=frames,
            experiment_seed=int(seed),
            authorization_basis=str(
                (track.get("rights") or {}).get("basis", "")
                if isinstance(track.get("rights"), dict)
                else ""
            ),
        )
        for index, start in enumerate(offsets)
    )


def _positions_for(count: int) -> tuple[str, ...]:
    """Names spread across the song for however many windows there are."""
    if count == 1:
        return ("START",)
    if count == 2:
        return ("START", "END")
    if count == 3:
        return ("START", "MIDDLE", "END")
    return ("START", "EARLY_MIDDLE", "LATE", "END")


def plan_windows(
    tracks: Iterable[dict[str, Any]],
    *,
    split: str,
    seed: int,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    max_windows: int = MAX_WINDOWS_PER_TRACK,
) -> WindowManifest:
    """Window every track in one split. Tracks, never a mixed library.

    Takes the tracks of a *single* split by design. Windowing a library
    and splitting the windows afterwards would let two views of the same
    recording land on both sides of a held-out boundary, which is the
    one mistake this phase cannot make.
    """
    windows: list[Window] = []
    for track in tracks:
        windows.extend(
            plan_track_windows(
                track, seed=seed, window_frames=window_frames, max_windows=max_windows
            )
        )
    windows.sort(key=lambda item: (item.audio_sha256, item.window_index))
    return WindowManifest(
        split=split,
        window_frames=window_frames,
        experiment_seed=int(seed),
        windows=tuple(windows),
    )


def eligible_tracks(
    tracks: Iterable[dict[str, Any]], *, window_frames: int = DEFAULT_WINDOW_FRAMES
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split a library into tracks that can yield a window and those that cannot.

    Returned rather than filtered silently: a track dropped for being
    shorter than the window is a track the experiment did not train on,
    and that belongs in the report.
    """
    keep: list[dict[str, Any]] = []
    drop: list[dict[str, Any]] = []
    for track in tracks:
        frames = track_frames(float(track.get("duration_seconds", 0.0)))
        (keep if frames >= window_frames else drop).append(track)
    return keep, drop


def sampling_weights(manifest: WindowManifest) -> dict[str, float]:
    """Per-window weights that make every *track* count the same.

    A four-window track and a one-window track are one recording each.
    Weighting windows equally would give the long one four times the
    influence for no reason but its length, so each window carries
    ``1 / windows_in_its_track`` and every track sums to one.
    """
    per_track = manifest.windows_per_track
    return {window.window_id: 1.0 / per_track[window.track_id] for window in manifest.windows}


__all__ = [
    "DEFAULT_WINDOW_FRAMES",
    "LATENT_FRAMES_PER_SECOND",
    "MAX_OVERLAP_FRACTION",
    "MAX_WINDOWS_PER_TRACK",
    "WINDOW_POSITIONS",
    "Window",
    "WindowError",
    "WindowManifest",
    "eligible_tracks",
    "plan_track_windows",
    "plan_windows",
    "sampling_weights",
    "track_frames",
    "window_count_for",
    "window_offsets",
]
