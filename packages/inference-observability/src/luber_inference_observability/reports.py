"""Two renderings of the same facts: one for a machine, one for a person.

The JSON is the record. The Markdown is what somebody reads at 2am, and
it is written for that reader: counts beside every percentage, windows
on every number, and no sentence claiming to know why anything changed.

Both come from the same computation. A report that recomputed its own
numbers would eventually disagree with the dashboard, and the resulting
argument about which is right is time nobody has during an incident.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from luber_inference_observability.aggregation import Metric
from luber_inference_observability.dimensions import Segment
from luber_inference_observability.incidents import InferenceIncident
from luber_inference_observability.queries import (
    coverage,
    evaluate_segments,
    summary,
    top_segments,
)
from luber_inference_observability.regressions import DEFAULT_RULES, regressions
from luber_inference_observability.storage import ObservationStore
from luber_inference_observability.versions import PHASE29_BOUNDARY_COMMIT, version_block
from luber_inference_observability.windows import TimeWindow


def health_report(
    store: ObservationStore,
    *,
    window: TimeWindow,
    incidents: Sequence[InferenceIncident] = (),
    generated_at: datetime | None = None,
    segment: Segment | None = None,
) -> dict[str, Any]:
    """Everything worth saying about one window, as data."""
    rows = list(store.select(window, segment=segment))
    overview = summary(store, window=window, segment=segment)
    findings = evaluate_segments(
        store, current=window, by=("provider_revision",), rules=DEFAULT_RULES
    )
    worst = top_segments(
        store,
        window=window,
        by=("provider", "duration_bucket"),
        metric=Metric.GENERATION_FAILURE_RATE.value,
    )
    active = [item for item in incidents if item.active]

    return {
        **version_block(),
        "generated_at": (generated_at or datetime.now(UTC)).astimezone(UTC).isoformat(),
        "window": window.to_dict(),
        "segment": (segment or Segment()).to_dict(),
        "sample_count": len(rows),
        "coverage": coverage(rows, window),
        "overview": overview["overview"],
        "counters": overview["counters"],
        "rates": overview["rates"],
        "averages": overview["averages"],
        "latency": overview["latency"],
        "findings": overview["findings"],
        "data_quality": overview["data_quality"],
        "regressions": [item.to_dict() for item in regressions(findings)],
        "most_affected_segments": worst["segments"],
        "segments_below_minimum": worst["segments_below_minimum"],
        "incidents": {
            "open": len(active),
            "total": len(incidents),
            "items": [item.to_dict(evidence_limit=3) for item in active],
        },
        "automatic_remediation": (
            "none — this system detects and explains; every action is an operator's"
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    """The same report, for a human under time pressure.

    Ordered by what somebody needs first: is anything on fire, then what
    changed, then the numbers behind it.
    """
    window = report["window"]
    lines: list[str] = [
        "# Inference health report",
        "",
        f"**Window** {window['start']} → {window['end']}  ",
        f"**Generated** {report['generated_at']}  ",
        f"**Observations** {report['sample_count']:,}",
        "",
    ]

    cover = report.get("coverage") or {}
    if cover.get("partial"):
        lines += [
            "> **Partial data.** " + str(cover.get("note", "")),
            "",
        ]

    incidents = report.get("incidents") or {}
    lines += ["## Incidents", ""]
    if not incidents.get("open"):
        lines += ["No open incidents.", ""]
    else:
        lines += [f"{incidents['open']} open.", ""]
        for item in incidents.get("items", []):
            lines.append(
                f"- **{item['severity']}** `{item['finding_type']}` — {item['segment_label']}  "
            )
            lines.append(
                f"  first seen {item['first_seen']}, last seen {item['last_seen']}, "
                f"{item['occurrence_count']} occurrences"
            )
            for evidence in item.get("evidence", [])[-1:]:
                lines.append(f"  {evidence['explanation']}")
        lines.append("")

    lines += ["## Regressions this window", ""]
    found = report.get("regressions") or []
    if not found:
        lines += ["Nothing crossed a threshold.", ""]
    else:
        for item in found:
            lines.append(f"- **{item['severity']}** {item['explanation']}")
            lines.append(
                f"  threshold: {item['threshold_crossed']}; "
                f"recommended: {', '.join(item['recommendations']) or 'none'}"
            )
        lines.append("")

    lines += ["## Health", "", "| Metric | Value | Counts |", "| --- | --- | --- |"]
    for name, rate in sorted((report.get("overview") or {}).items()):
        if rate["status"] == "NO_DATA":
            lines.append(f"| {name} | NO_DATA | 0 samples |")
        else:
            lines.append(
                f"| {name} | {rate['percent']:.2f}% | {rate['numerator']}/{rate['denominator']} |"
            )
    lines.append("")

    counters = report.get("counters") or {}
    lines += [
        "## Volume",
        "",
        f"- Requests: {counters.get('generation_requests', 0):,}",
        f"- Completed: {counters.get('completed_generations', 0):,}",
        f"- Failed: {counters.get('failed_generations', 0):,}",
        f"- Cancelled: {counters.get('cancelled_generations', 0):,}",
        f"- Provider calls: {counters.get('provider_calls', 0):,}",
        f"- Quality retries: {counters.get('quality_retries', 0):,}",
        f"- Retry exhaustions: {counters.get('retry_exhaustions', 0):,}",
        "",
    ]

    lines += [
        "## Latency",
        "",
        "| Stage | P50 | P95 | P99 | Samples |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, dist in sorted((report.get("latency") or {}).items()):
        if dist["status"] == "NO_DATA":
            lines.append(f"| {name} | NO_DATA | NO_DATA | NO_DATA | 0 |")
        else:
            lines.append(
                f"| {name} | {_s(dist['p50'])} | {_s(dist['p95'])} | "
                f"{_s(dist['p99'])} | {dist['count']:,} |"
            )
    lines.append("")

    critical = (report.get("findings") or {}).get("critical") or {}
    soft = (report.get("findings") or {}).get("soft") or {}
    lines += ["## QC findings", ""]
    if critical:
        lines += ["**Rejections**", ""]
        lines += [f"- {code}: {count:,}" for code, count in sorted(critical.items())]
        lines.append("")
    else:
        lines += ["No rejections.", ""]
    if soft:
        # Kept under its own heading so an advisory can never be read as
        # a failure. A harshness proxy and invalid audio are not the
        # same news and must not appear in the same list.
        lines += ["**Advisories on delivered audio** (not failures)", ""]
        lines += [f"- {code}: {count:,}" for code, count in sorted(soft.items())]
        lines.append("")

    worst = report.get("most_affected_segments") or []
    if worst:
        lines += [
            "## Most affected segments",
            "",
            "| Segment | Failure rate | Counts |",
            "| --- | --- | --- |",
        ]
        for item in worst:
            value = item["value"]
            rendered = "NO_DATA" if value is None else f"{value * 100:.2f}%"
            lines.append(
                f"| {item['segment_label']} | {rendered} | "
                f"{item['numerator']}/{item['denominator']} |"
            )
        lines.append("")
    below = report.get("segments_below_minimum") or 0
    if below:
        lines += [
            f"_{below} segment(s) had too few samples to rank and are not shown._",
            "",
        ]

    quality = report.get("data_quality") or {}
    if quality:
        lines += ["## Telemetry problems", ""]
        lines += [f"- {issue}: {count:,}" for issue, count in sorted(quality.items())]
        lines.append("")

    lines += [
        "---",
        "",
        "No action was taken automatically. This system detects and explains; disabling a "
        "provider, changing a policy or altering a QC threshold are operator decisions.",
        "",
        f"Candidate-trace boundary: `{PHASE29_BOUNDARY_COMMIT}`.",
        "",
    ]
    return "\n".join(lines)


def _s(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}s"


def revision_report(comparison: dict[str, Any]) -> str:
    """A provider revision comparison, rendered."""
    lines = [
        "# Provider revision comparison",
        "",
        f"**Window** {comparison['window']['start']} → {comparison['window']['end']}  ",
        f"**A** `{comparison['left_revision']}`  ",
        f"**B** `{comparison['right_revision']}`  ",
        f"**Status** {comparison['status']}",
        "",
    ]
    if not comparison.get("sufficient_data"):
        lines += [
            f"Not enough data to compare: each side needs at least "
            f"{comparison['minimum_samples']} observations. "
            f"A has {comparison['left']['sample_count']}, "
            f"B has {comparison['right']['sample_count']}.",
            "",
        ]
        return "\n".join(lines)

    lines += ["| Metric | A | B | Δ |", "| --- | --- | --- | --- |"]
    for metric, delta in sorted((comparison.get("deltas") or {}).items()):
        if delta["status"] != "OK":
            lines.append(f"| {metric} | NO_DATA | NO_DATA | — |")
            continue
        lines.append(
            f"| {metric} | {delta['before']:.4f} | {delta['after']:.4f} | "
            f"{delta['absolute_delta']:+.4f} |"
        )
    lines += ["", f"_{comparison['caveat']}_", ""]
    return "\n".join(lines)


__all__ = ["health_report", "render_markdown", "revision_report"]
