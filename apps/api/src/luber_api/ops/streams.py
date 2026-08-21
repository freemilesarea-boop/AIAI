"""Metrics and logs, at a size a browser can hold.

Both of these grow without bound during a run, and both are read on a
poll. That combination is what makes naive versions of them unusable:
a thirty-hour training run writes hundreds of thousands of metric events
and a log measured in hundreds of megabytes, and a console that fetched
all of either every few seconds would spend the run's whole duration
re-downloading its own history.

**Metrics are thinned, and the response says so.** A chart 900 pixels
wide cannot show 400,000 points, so a series over the cap is sampled to
an even stride with the first and last points kept — the last one
because it is the number the operator is actually reading, and losing it
to a stride boundary would make the chart disagree with the summary
beside it. ``sampled`` travels with the series, so the chart can say it
is a summary rather than implying it drew every step.

**Logs are incremental, using Phase 27's cursor.** The browser sends the
offset it was given and receives only what arrived since. That is
already the protocol between the control plane and a worker, and
inventing a second one here would mean two things that could disagree
about what "read so far" means. The first read of a large file starts
from the tail rather than the beginning, and says so, because an
operator opening a failed run wants the end of it.
"""

from __future__ import annotations

from pathlib import Path

from luber_api.ops.redaction import redact_text
from luber_api.ops.schemas import LogView, MetricPoint, MetricSeries
from luber_training.metrics import MetricEvent, MetricSource, iter_metrics
from luber_training.remote.streams import read_log

#: Points per series in one response. Chosen for a chart, not for a
#: dataset: beyond roughly this many the line is denser than the pixels
#: available and the extra points cost bytes without showing anything.
DEFAULT_METRIC_POINTS = 600

#: Bytes of log per read. Large enough that a normal poll returns
#: everything new in one go, small enough that a single response stays
#: something a browser can render.
DEFAULT_LOG_BYTES = 96 * 1024

#: Metric names that describe the machine rather than the model. Split
#: out so the training charts are not diluted by GPU temperature, and so
#: the telemetry panel can say "unavailable" honestly when a run
#: produced none.
TELEMETRY_METRICS: frozenset[str] = frozenset(
    {
        "gpu_memory_mb",
        "gpu_utilization_percent",
        "gpu_power_watts",
        "cpu_percent",
        "ram_mb",
        "disk_free_mb",
    }
)


def _sample(points: list[MetricPoint], limit: int) -> tuple[list[MetricPoint], bool]:
    """Thin *points* to at most *limit*, keeping the ends.

    An even stride rather than a rolling average: averaging would hide
    the loss spike that is often the only interesting thing in the
    series, and a console that smoothed away the spike would be lying
    about a run in the direction of it looking fine.
    """
    if len(points) <= limit:
        return points, False
    stride = len(points) / float(limit)
    kept = [points[int(index * stride)] for index in range(limit)]
    if kept[-1] is not points[-1]:
        kept[-1] = points[-1]
    return kept, True


def _sources_path(run_directory: Path | None, worker_directory: Path | None) -> list[Path]:
    """Where this run's metrics might be, nearest first.

    The control plane's own file first: it is what the registry cites
    and what survives the worker being torn down. The worker's file is
    consulted only when the local one holds nothing, which is the case
    while a remote run is still in flight and nothing has collected it
    yet.
    """
    candidates: list[Path] = []
    if run_directory is not None:
        candidates.append(run_directory / "metrics.jsonl")
    if worker_directory is not None:
        candidates.append(worker_directory / "metrics" / "metrics.jsonl")
    return [path for path in candidates if path.is_file()]


def load_metrics(
    run_directory: Path | None, worker_directory: Path | None = None
) -> list[MetricEvent]:
    """Every metric event recorded for a run, deduplicated by identity.

    Two sources can legitimately hold the same event — the worker wrote
    it and a collection copied it — and Phase 27 already defines what
    makes two events one event. Reusing that identity here means the
    console counts a step once however many times it was transferred.
    """
    from luber_training.remote.streams import metric_identity

    seen: set[tuple[str, int | None, str, str]] = set()
    events: list[MetricEvent] = []
    for path in _sources_path(run_directory, worker_directory):
        for event in iter_metrics(path):
            identity = metric_identity(event)
            if identity in seen:
                continue
            seen.add(identity)
            events.append(event)
    return events


def build_series(
    events: list[MetricEvent],
    *,
    names: frozenset[str] | None = None,
    exclude: frozenset[str] | None = None,
    limit: int = DEFAULT_METRIC_POINTS,
) -> list[MetricSeries]:
    """Group events into chartable series.

    Only metrics that exist become series. There is no list of expected
    charts rendered empty: a panel for validation loss on a trainer that
    computes none would suggest the number is coming, and it is not.
    """
    grouped: dict[str, list[MetricEvent]] = {}
    for event in events:
        if names is not None and event.metric_name not in names:
            continue
        if exclude is not None and event.metric_name in exclude:
            continue
        grouped.setdefault(event.metric_name, []).append(event)

    series: list[MetricSeries] = []
    for name, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda item: item.step if item.step is not None else -1)
        points = [
            MetricPoint(
                step=event.step,
                epoch=event.epoch,
                value=event.value,
                timestamp=event.timestamp,
            )
            for event in ordered
        ]
        kept, sampled = _sample(points, limit)
        series.append(
            MetricSeries(
                metric_name=name,
                unit=next((event.unit for event in ordered if event.unit), ""),
                sources=sorted({event.source for event in ordered}),
                points=kept,
                total_points=len(points),
                sampled=sampled,
                last_value=points[-1].value if points else None,
            )
        )
    return series


def latest_value(events: list[MetricEvent], metric_name: str) -> float | None:
    """The most recent value of one metric, by step then by arrival.

    Simulated values are not excluded here — a dry run's chart is a real
    chart of simulated numbers — but the series carries its sources, so
    nothing downstream can present SIMULATED as a measurement without
    having chosen to ignore the label.
    """
    candidates = [event for event in events if event.metric_name == metric_name]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item.step if item.step is not None else -1, item.timestamp))
    return candidates[-1].value


def has_real_measurements(events: list[MetricEvent]) -> bool:
    return any(event.source != MetricSource.SIMULATED.value for event in events)


def _log_paths(
    run_directory: Path | None, worker_directory: Path | None, stream: str
) -> list[Path]:
    name = "trainer.stderr.log" if stream == "stderr" else "trainer.stdout.log"
    candidates: list[Path] = []
    if worker_directory is not None:
        candidates.append(worker_directory / "logs" / name)
    if run_directory is not None:
        candidates.append(run_directory / "logs" / name)
    return [path for path in candidates if path.is_file()]


def read_stream(
    run_directory: Path | None,
    worker_directory: Path | None,
    *,
    stream: str = "stdout",
    offset: int | None = None,
    limit: int = DEFAULT_LOG_BYTES,
    unavailable_reason: str | None = None,
) -> LogView:
    """One incremental read, redacted before it leaves this process.

    ``offset=None`` means "start where an operator would want to start":
    at the beginning for a short file, and at the tail for a long one.
    An explicit offset is honoured exactly, which is what makes both
    polling forward and loading older work through the same endpoint.
    """
    paths = _log_paths(run_directory, worker_directory, stream)
    if not paths:
        return LogView(
            available=False,
            unavailable_reason=(
                unavailable_reason or "No log file has been written for this run on this machine."
            ),
            stream="stderr" if stream == "stderr" else "stdout",
        )

    path = paths[0]
    size = path.stat().st_size
    from_tail = False
    if offset is None:
        if size > limit:
            offset = size - limit
            from_tail = True
        else:
            offset = 0

    chunk = read_log(path, offset=offset, limit=limit, stream=stream)
    return LogView(
        available=True,
        stream="stderr" if stream == "stderr" else "stdout",
        offset=chunk.offset,
        next_offset=chunk.next_offset,
        size_bytes=chunk.size_bytes,
        eof=chunk.eof,
        truncated=chunk.truncated,
        text=redact_text(chunk.text),
        from_tail=from_tail,
    )


def tail_lines(
    run_directory: Path | None,
    worker_directory: Path | None,
    *,
    stream: str = "stderr",
    lines: int = 30,
) -> list[str]:
    """The last few lines, for a failure panel.

    Step 46: an operator should not have to open a log viewer to learn
    why a run failed. Reading a bounded tail rather than the file keeps
    that cheap even when the file is large.
    """
    view = read_stream(run_directory, worker_directory, stream=stream, offset=None, limit=16 * 1024)
    if not view.available or not view.text:
        return []
    return [line for line in view.text.splitlines() if line.strip()][-lines:]


__all__ = [
    "DEFAULT_LOG_BYTES",
    "DEFAULT_METRIC_POINTS",
    "TELEMETRY_METRICS",
    "build_series",
    "has_real_measurements",
    "latest_value",
    "load_metrics",
    "read_stream",
    "tail_lines",
]
