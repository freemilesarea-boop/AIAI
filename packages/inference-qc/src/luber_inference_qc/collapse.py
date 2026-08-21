"""Where the music actually stops, as distinct from where the file does.

This detector is new. The Phase 29 brief asks to reuse the Phase 22
early-collapse detector, and there is not one: the finishing engine
measures a whole-file ``silence_ratio``, which cannot tell a track with a
quiet outro apart from one that ends at 110 seconds and pads the
remaining two minutes with digital silence. Both produce the same ratio
and only one is a failure.

The measurement is positional. Short RMS windows, a floor relative to
the track's own loud material, and the last window above that floor is
the *effective content end*. Everything after it is trailing silence.

Three decisions worth stating.

**The floor is relative, not absolute.** A quiet song is not a collapsed
one. Anchoring to a high percentile of the track's own window energy
means a track mastered 20 dB quieter than another is judged the same way.

**A gap in the middle is not a collapse.** Songs have breakdowns and
false endings. Only trailing silence counts, so a four-bar drop out at
2:00 in a 4:00 track produces no finding.

**Requested duration is compared separately from file duration.** A
provider that returns a 110-second file for a 240-second request has a
duration failure; one that returns a 240-second file whose last 130
seconds are silent has a collapse. They have different causes and the
trace should not conflate them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from luber_audio_finishing import LoadedAudio

#: Window length for the energy envelope. Long enough that a single
#: percussive gap does not read as silence, short enough to locate an
#: ending within a musically meaningful margin.
WINDOW_SECONDS = 0.25

#: How far below the track's own loud material a window has to sit to
#: count as silence. 45 dB is well below any musical dynamic — a real
#: outro fades over a few seconds and stays above this until it stops.
RELATIVE_FLOOR_DB = -45.0

#: An absolute backstop for material that is loud throughout. Windows
#: below this are silent whatever the relative floor says.
ABSOLUTE_FLOOR_DBFS = -70.0

#: The percentile of window energy taken as "the track's loud material".
#: Not the maximum: one clipped transient would raise the floor for the
#: whole track.
REFERENCE_PERCENTILE = 90.0

#: Trailing silence shorter than this is an ending, not a collapse.
MINIMUM_TRAILING_SILENCE_SECONDS = 8.0

_EPS = 1e-12


@dataclass(frozen=True)
class CollapseMeasurement:
    """Where content ends, and how much file follows it."""

    #: Seconds of audio in the file.
    file_duration_seconds: float
    #: Last instant carrying content above the floor.
    content_end_seconds: float
    #: File duration minus content end.
    trailing_silence_seconds: float
    #: Trailing silence as a share of the file.
    trailing_silence_ratio: float
    #: Content end over file duration. 1.0 means content runs to the end.
    content_ratio: float
    #: The floor this was judged against, for the trace.
    floor_dbfs: float
    #: True when no window anywhere exceeded the floor.
    entirely_silent: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_duration_seconds": round(self.file_duration_seconds, 3),
            "content_end_seconds": round(self.content_end_seconds, 3),
            "trailing_silence_seconds": round(self.trailing_silence_seconds, 3),
            "trailing_silence_ratio": round(self.trailing_silence_ratio, 4),
            "content_ratio": round(self.content_ratio, 4),
            "floor_dbfs": round(self.floor_dbfs, 2),
            "entirely_silent": self.entirely_silent,
        }


def _window_rms_dbfs(audio: LoadedAudio) -> np.ndarray:
    """Per-window RMS in dBFS over the mono sum."""
    mono = audio.mono()
    size = max(1, int(WINDOW_SECONDS * audio.sample_rate))
    usable = (mono.size // size) * size
    if usable == 0:
        # Shorter than one window: treat the whole thing as one.
        rms = float(np.sqrt(np.mean(np.square(mono)) + _EPS)) if mono.size else 0.0
        return np.asarray([20.0 * np.log10(rms + _EPS)], dtype=np.float64)
    frames = mono[:usable].reshape(-1, size)
    rms = np.sqrt(np.mean(np.square(frames), axis=1) + _EPS)
    return np.asarray(20.0 * np.log10(rms + _EPS), dtype=np.float64)


def measure_collapse(audio: LoadedAudio) -> CollapseMeasurement:
    """Find the effective end of content.

    Returns a measurement, never a verdict — whether the trailing
    silence constitutes a failure depends on what was requested, and
    that comparison belongs in the checks.
    """
    duration = audio.duration_seconds
    windows = _window_rms_dbfs(audio)
    window_seconds = min(WINDOW_SECONDS, duration) if duration else WINDOW_SECONDS

    reference = float(np.percentile(windows, REFERENCE_PERCENTILE))
    floor = max(reference + RELATIVE_FLOOR_DB, ABSOLUTE_FLOOR_DBFS)

    above = np.nonzero(windows > floor)[0]
    if above.size == 0:
        return CollapseMeasurement(
            file_duration_seconds=duration,
            content_end_seconds=0.0,
            trailing_silence_seconds=duration,
            trailing_silence_ratio=1.0,
            content_ratio=0.0,
            floor_dbfs=floor,
            entirely_silent=True,
        )

    # The end of the last window carrying content, capped at the file
    # length so a partial final window cannot report content past the end.
    content_end = min(duration, float(above[-1] + 1) * window_seconds)
    trailing = max(0.0, duration - content_end)
    return CollapseMeasurement(
        file_duration_seconds=duration,
        content_end_seconds=content_end,
        trailing_silence_seconds=trailing,
        trailing_silence_ratio=(trailing / duration) if duration else 0.0,
        content_ratio=(content_end / duration) if duration else 0.0,
        floor_dbfs=floor,
        entirely_silent=False,
    )


__all__ = [
    "ABSOLUTE_FLOOR_DBFS",
    "MINIMUM_TRAILING_SILENCE_SECONDS",
    "RELATIVE_FLOOR_DB",
    "WINDOW_SECONDS",
    "CollapseMeasurement",
    "measure_collapse",
]
