"""Comparing two datasets, because they change one acquisition at a time.

A corpus is not built once. Material arrives, sidecars get written,
thresholds get tuned, and each change moves the distributions a little.
Individually those moves are invisible; cumulatively they decide what
the model learns. This answers "what actually changed" between two
manifests.

Direction is reported rather than judged, with one exception. Most
movements are neither good nor bad without a target — Korean rising from
30% to 45% is progress under one profile and overshoot under another —
so the diff states the delta and lets the profile decide. The exception
is concentration: effective artist count falling is a regression under
every objective, and saying so is not an opinion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from luber_dataset.factory.intelligence.profile import DatasetProfile

#: A share moving less than this is noise from a handful of tracks.
MATERIAL_SHARE_DELTA = 0.02
#: Effective-category changes below this are rounding.
MATERIAL_EFFECTIVE_DELTA = 0.5

#: Dimensions worth diffing. Everything else is either derived from
#: these or too sparse to compare meaningfully.
COMPARED_DIMENSIONS: tuple[str, ...] = (
    "quality_tier",
    "language",
    "vocal_class",
    "tempo_bucket",
    "duration_bucket",
    "genre",
    "source_type",
    "artist",
)


@dataclass
class DimensionDrift:
    dimension: str
    added_categories: list[str] = field(default_factory=list)
    removed_categories: list[str] = field(default_factory=list)
    #: category -> (before, after, delta)
    moved: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    coverage_before: float = 0.0
    coverage_after: float = 0.0

    @property
    def coverage_delta(self) -> float:
        return self.coverage_after - self.coverage_before

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "added_categories": self.added_categories,
            "removed_categories": self.removed_categories,
            "coverage_before": round(self.coverage_before, 6),
            "coverage_after": round(self.coverage_after, 6),
            "coverage_delta": round(self.coverage_delta, 6),
            "moved": {
                category: {
                    "before": round(before, 6),
                    "after": round(after, 6),
                    "delta": round(delta, 6),
                }
                for category, (before, after, delta) in sorted(self.moved.items())
            },
        }


@dataclass
class DriftReport:
    label_a: str
    label_b: str
    track_count_before: int = 0
    track_count_after: int = 0
    hours_before: float = 0.0
    hours_after: float = 0.0
    dimensions: dict[str, DimensionDrift] = field(default_factory=dict)
    concentration: dict[str, dict[str, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "label_a": self.label_a,
            "label_b": self.label_b,
            "track_count_before": self.track_count_before,
            "track_count_after": self.track_count_after,
            "track_count_delta": self.track_count_after - self.track_count_before,
            "hours_before": round(self.hours_before, 4),
            "hours_after": round(self.hours_after, 4),
            "hours_delta": round(self.hours_after - self.hours_before, 4),
            "dimensions": {
                name: value.to_dict() for name, value in sorted(self.dimensions.items())
            },
            "concentration": {
                name: {k: round(v, 6) for k, v in sorted(values.items())}
                for name, values in sorted(self.concentration.items())
            },
            "notes": self.notes,
        }


def compare(
    before: DatasetProfile, after: DatasetProfile, *, label_a: str = "A", label_b: str = "B"
) -> DriftReport:
    """Everything that moved between two profiles."""
    report = DriftReport(
        label_a=label_a,
        label_b=label_b,
        track_count_before=before.track_count,
        track_count_after=after.track_count,
        hours_before=before.total_hours,
        hours_after=after.total_hours,
    )

    for dimension in COMPARED_DIMENSIONS:
        left = before.categorical.get(dimension)
        right = after.categorical.get(dimension)
        if left is None or right is None:
            continue
        drift = DimensionDrift(
            dimension=dimension,
            coverage_before=left.coverage,
            coverage_after=right.coverage,
        )
        left_labels = {bucket.label for bucket in left.buckets}
        right_labels = {bucket.label for bucket in right.buckets}
        drift.added_categories = sorted(right_labels - left_labels)
        drift.removed_categories = sorted(left_labels - right_labels)

        for label in sorted(left_labels | right_labels):
            before_share = left.share(label)
            after_share = right.share(label)
            delta = after_share - before_share
            if abs(delta) >= MATERIAL_SHARE_DELTA:
                drift.moved[label] = (before_share, after_share, delta)
        report.dimensions[dimension] = drift

    for dimension in ("artist", "source_reference", "genre", "language", "duplicate_family"):
        left_metrics = before.concentration.get(dimension)
        right_metrics = after.concentration.get(dimension)
        if left_metrics is None or right_metrics is None:
            continue
        report.concentration[dimension] = {
            "top1_share_before": left_metrics.top1_share,
            "top1_share_after": right_metrics.top1_share,
            "top1_share_delta": right_metrics.top1_share - left_metrics.top1_share,
            "effective_before": left_metrics.effective_categories,
            "effective_after": right_metrics.effective_categories,
            "effective_delta": (
                right_metrics.effective_categories - left_metrics.effective_categories
            ),
        }
        # The one directional judgement in the module, and it holds under
        # every training objective.
        drop = left_metrics.effective_categories - right_metrics.effective_categories
        if drop >= MATERIAL_EFFECTIVE_DELTA:
            report.notes.append(
                f"{dimension} diversity collapsed: effective categories "
                f"{left_metrics.effective_categories:.1f} -> "
                f"{right_metrics.effective_categories:.1f}"
            )

    synthetic_delta = after.synthetic_share_by_count - before.synthetic_share_by_count
    if abs(synthetic_delta) >= MATERIAL_SHARE_DELTA:
        report.notes.append(
            f"declared synthetic share moved {before.synthetic_share_by_count:.1%} -> "
            f"{after.synthetic_share_by_count:.1%}"
        )
    return report


def render_markdown(report: DriftReport) -> str:
    lines = [
        "# Dataset drift",
        "",
        f"Comparing **{report.label_a}** → **{report.label_b}**.",
        "",
        f"- Tracks: {report.track_count_before} → {report.track_count_after} "
        f"({report.track_count_after - report.track_count_before:+d})",
        f"- Hours: {report.hours_before:.2f} → {report.hours_after:.2f} "
        f"({report.hours_after - report.hours_before:+.2f})",
        "",
    ]

    if report.notes:
        lines.append("## Notable")
        lines.append("")
        lines.extend(f"- {note}" for note in report.notes)
        lines.append("")

    lines.append("## Distribution shifts")
    lines.append("")
    any_movement = False
    for name, drift in sorted(report.dimensions.items()):
        if not drift.moved and not drift.added_categories and not drift.removed_categories:
            continue
        any_movement = True
        lines.append(f"### {name}")
        lines.append("")
        lines.append(
            f"Coverage {drift.coverage_before:.1%} → {drift.coverage_after:.1%} "
            f"({drift.coverage_delta:+.1%})"
        )
        lines.append("")
        if drift.moved:
            lines.append("| category | before | after | delta |")
            lines.append("|---|---|---|---|")
            for category, (before, after, delta) in sorted(
                drift.moved.items(), key=lambda item: -abs(item[1][2])
            ):
                lines.append(f"| {category} | {before:.1%} | {after:.1%} | {delta:+.1%} |")
            lines.append("")
        if drift.added_categories:
            lines.append(f"Added: {', '.join(drift.added_categories)}")
            lines.append("")
        if drift.removed_categories:
            lines.append(f"Removed: {', '.join(drift.removed_categories)}")
            lines.append("")
    if not any_movement:
        lines.append(
            f"_Nothing moved by more than {MATERIAL_SHARE_DELTA:.0%} in any compared dimension._"
        )
        lines.append("")

    lines.append("## Concentration")
    lines.append("")
    lines.append("| dimension | top-1 before | top-1 after | effective before | effective after |")
    lines.append("|---|---|---|---|---|")
    for name, values in sorted(report.concentration.items()):
        lines.append(
            f"| {name} | {values['top1_share_before']:.1%} | "
            f"{values['top1_share_after']:.1%} | {values['effective_before']:.1f} | "
            f"{values['effective_after']:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)
