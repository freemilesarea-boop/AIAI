"""Time windows, and the discipline of saying which one a number came from.

A rate without a window is not a fact. "Failure rate 4%" invites the
reader to supply their own period, and they will supply the one that
makes the number mean what they already believed.

So every window here is a half-open interval `[start, end)` in UTC, and
it travels with every aggregate, finding, incident and report. Half-open
because adjacent windows must tile without double-counting the instant
they share: a generation at exactly 12:00 belongs to the window starting
at 12:00 and to no other.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any


class WindowSize(StrEnum):
    """The named windows the dashboard and CLI offer."""

    FIVE_MINUTES = "5m"
    ONE_HOUR = "1h"
    ONE_DAY = "24h"
    SEVEN_DAYS = "7d"


DURATIONS: dict[str, timedelta] = {
    WindowSize.FIVE_MINUTES.value: timedelta(minutes=5),
    WindowSize.ONE_HOUR.value: timedelta(hours=1),
    WindowSize.ONE_DAY.value: timedelta(hours=24),
    WindowSize.SEVEN_DAYS.value: timedelta(days=7),
}


def duration_of(size: str) -> timedelta:
    key = size.strip().lower()
    if key not in DURATIONS:
        raise ValueError(f"unknown window size {size!r}. Known: {', '.join(DURATIONS)}")
    return DURATIONS[key]


@dataclass(frozen=True, order=True)
class TimeWindow:
    """A half-open interval in UTC, `[start, end)`."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("a window must be timezone-aware; store UTC")
        if self.end <= self.start:
            raise ValueError(f"a window must end after it starts: {self.start} → {self.end}")

    @classmethod
    def ending_at(cls, end: datetime, size: str) -> TimeWindow:
        """The named window immediately before *end*."""
        finish = end.astimezone(UTC)
        return cls(finish - duration_of(size), finish)

    @classmethod
    def of(cls, start: datetime, end: datetime) -> TimeWindow:
        """An arbitrary interval, for a query nobody anticipated."""
        return cls(start.astimezone(UTC), end.astimezone(UTC))

    @property
    def duration(self) -> timedelta:
        return self.end - self.start

    def contains(self, moment: datetime) -> bool:
        aware = moment if moment.tzinfo else moment.replace(tzinfo=UTC)
        return self.start <= aware.astimezone(UTC) < self.end

    def shifted(self, delta: timedelta) -> TimeWindow:
        return TimeWindow(self.start + delta, self.end + delta)

    def preceding(self, span: timedelta, *, gap: timedelta = timedelta(0)) -> TimeWindow:
        """The interval of length *span* ending *gap* before this one starts.

        ``gap`` is how a baseline avoids learning the incident it is
        supposed to detect. A rolling baseline that runs right up to the
        current window absorbs the first minutes of a regression and
        then reports that nothing changed.
        """
        end = self.start - gap
        return TimeWindow(end - span, end)

    def buckets(self, step: timedelta) -> list[TimeWindow]:
        """Tile this window into consecutive intervals of length *step*.

        The last one is truncated to the window's end rather than
        overhanging it: a trend point covering time that has not happened
        yet would show a cliff at the right-hand edge of every chart.
        """
        if step <= timedelta(0):
            raise ValueError("bucket step must be positive")
        out: list[TimeWindow] = []
        cursor = self.start
        while cursor < self.end:
            nxt = min(cursor + step, self.end)
            out.append(TimeWindow(cursor, nxt))
            cursor = nxt
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "duration_seconds": self.duration.total_seconds(),
        }

    def label(self) -> str:
        return f"{self.start.isoformat()} → {self.end.isoformat()}"


#: How finely to slice each named window for a trend chart.
#:
#: Chosen so a chart has enough points to show a shape and few enough
#: that each point has samples in it. Twelve five-minute points across an
#: hour is a trend; sixty one-minute points across an hour is mostly
#: noise with a line through it.
TREND_STEPS: dict[str, timedelta] = {
    WindowSize.FIVE_MINUTES.value: timedelta(seconds=30),
    WindowSize.ONE_HOUR.value: timedelta(minutes=5),
    WindowSize.ONE_DAY.value: timedelta(hours=1),
    WindowSize.SEVEN_DAYS.value: timedelta(hours=6),
}


def trend_step(size: str) -> timedelta:
    key = size.strip().lower()
    if key not in TREND_STEPS:
        raise ValueError(f"unknown window size {size!r}")
    return TREND_STEPS[key]


def step_for(window: TimeWindow, *, target_points: int = 24) -> timedelta:
    """A sensible bucket for an arbitrary window.

    Used when a caller supplied their own start and end, where no named
    step applies. Aims at *target_points* buckets and never returns less
    than a second.
    """
    seconds = max(1.0, window.duration.total_seconds() / max(1, target_points))
    return timedelta(seconds=seconds)


__all__ = [
    "DURATIONS",
    "TREND_STEPS",
    "TimeWindow",
    "WindowSize",
    "duration_of",
    "step_for",
    "trend_step",
]
