"""Baseline report generation from the JSONL result and score stores."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from bench.scoring import (
    MAX_ARTIFACT_RATE,
    MAX_TECHNICAL_FAILURE_RATE,
    QUALITY_TARGETS,
    RUBRIC_DIMENSIONS,
    meets_targets,
)

TECHNICAL_FAILURE_FLAGS = {
    "SILENT_OUTPUT",
    "CORRUPTED_AUDIO",
    "GENERATION_FAILED",
    "INVALID_DURATION",
}


@dataclass
class Summary:
    total: int
    completed: int
    failed: int
    technical_failures: int

    @property
    def success_rate(self) -> float:
        return self.completed / self.total if self.total else 0.0

    @property
    def technical_failure_rate(self) -> float:
        return self.technical_failures / self.total if self.total else 0.0


def summarize(records: list[dict[str, Any]]) -> Summary:
    completed = [r for r in records if r.get("status") == "COMPLETED"]
    technical = [
        r
        for r in records
        if set(_flags(r)) & TECHNICAL_FAILURE_FLAGS or r.get("status") != "COMPLETED"
    ]
    return Summary(
        total=len(records),
        completed=len(completed),
        failed=len(records) - len(completed),
        technical_failures=len(technical),
    )


def _flags(record: dict[str, Any]) -> list[str]:
    metrics = record.get("metrics") or {}
    flags = metrics.get("flags") if isinstance(metrics, dict) else None
    return list(flags) if isinstance(flags, list) else []


def average_scores(scores: list[dict[str, Any]]) -> dict[str, float]:
    """Mean per rubric dimension across all submitted evaluations."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for entry in scores:
        for dimension, value in (entry.get("scores") or {}).items():
            if dimension in RUBRIC_DIMENSIONS and isinstance(value, int):
                buckets[dimension].append(value)
    return {d: round(statistics.mean(v), 2) for d, v in buckets.items() if v}


def artifact_frequency(scores: list[dict[str, Any]]) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for entry in scores:
        for tag in entry.get("artifact_tags") or []:
            counter[str(tag)] += 1
    return counter.most_common()


def group_average(
    records: list[dict[str, Any]], scores: list[dict[str, Any]], key: str, dimension: str
) -> dict[str, float]:
    """Mean of one rubric dimension grouped by a record field."""
    by_id = {r["benchmark_id"]: r for r in records if r.get("benchmark_id")}
    buckets: dict[str, list[int]] = defaultdict(list)
    for entry in scores:
        record = by_id.get(entry.get("benchmark_id"))
        if record is None:
            continue
        value = (entry.get("scores") or {}).get(dimension)
        if isinstance(value, int):
            buckets[str(record.get(key))].append(value)
    return {k: round(statistics.mean(v), 2) for k, v in sorted(buckets.items()) if v}


def seed_variance(records: list[dict[str, Any]], scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Spread of overall quality across seeds for the same prompt."""
    by_id = {r["benchmark_id"]: r for r in records if r.get("benchmark_id")}
    per_prompt: dict[str, list[int]] = defaultdict(list)
    for entry in scores:
        record = by_id.get(entry.get("benchmark_id"))
        if record is None:
            continue
        value = (entry.get("scores") or {}).get("overall_musical_quality")
        if isinstance(value, int):
            per_prompt[str(record.get("prompt_id"))].append(value)
    out: dict[str, Any] = {}
    for prompt_id, values in sorted(per_prompt.items()):
        if len(values) < 2:
            continue
        out[prompt_id] = {
            "n": len(values),
            "best": max(values),
            "median": round(statistics.median(values), 2),
            "worst": min(values),
            "spread": max(values) - min(values),
        }
    return out


def _table(rows: list[tuple[str, str]], headers: tuple[str, str]) -> str:
    if not rows:
        return "_No data._\n"
    out = [f"| {headers[0]} | {headers[1]} |", "|---|---|"]
    out += [f"| {a} | {b} |" for a, b in rows]
    return "\n".join(out) + "\n"


def render_verdict_section(verdict: dict[str, Any]) -> str:
    """Render the human verdict. This leads the report when present.

    A global verdict outranks every objective measurement below it: the
    machine can only report that audio decoded correctly, while this is
    a person saying whether the music is worth anything.
    """
    lines: list[str] = []
    add = lines.append
    score = verdict.get("overall_score")
    usable = verdict.get("commercially_usable")
    reviewed = verdict.get("tracks_reviewed", 0)
    rejected = verdict.get("tracks_rejected", 0)

    add("## HUMAN LISTENING VERDICT\n")
    add(f"**HUMAN LISTENING BASELINE: {score} / 10**\n")
    add(f"**COMMERCIAL RELEASE READINESS: {'PASS' if usable else 'FAIL'}**\n")
    add(f"**{rejected} / {reviewed} REJECTED**\n")
    add("**SUNO 4.5 PARITY: NOT ACHIEVED**\n")
    add(f"- Evaluator: {verdict.get('evaluator')}")
    add(f"- Recorded: {verdict.get('recorded_at')}")
    add(f"- Reason detailed scoring was skipped: _{verdict.get('reason')}_")
    add("")
    findings = verdict.get("findings") or []
    if findings:
        add("### Findings\n")
        for finding in findings:
            add(f"- `{finding}`")
        add("")
    if verdict.get("notes"):
        add(f"> {verdict['notes']}\n")
    add(
        "_No per-track scores were fabricated to fill this section. The "
        "evaluator rejected the set before per-dimension scoring became "
        "informative, and the record reflects exactly that._\n"
    )
    return "\n".join(lines)


def render_report(
    *,
    records: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    baseline_id: str,
    benchmark_version: str,
    ace_step_version: str,
    ace_step_commit: str,
    hardware: str,
    notes: str = "",
    verdict: dict[str, Any] | None = None,
) -> str:
    summary = summarize(records)
    completed = [r for r in records if r.get("status") == "COMPLETED"]
    averages = average_scores(scores)
    targets = meets_targets(averages)

    durations = [float(r["generation_seconds"]) for r in completed if r.get("generation_seconds")]
    rtfs = [float(r["real_time_factor"]) for r in completed if r.get("real_time_factor")]

    def count(pred: Any) -> int:
        return sum(1 for r in records if pred(r))

    korean_vocal = count(
        lambda r: r.get("language") == "ko" and r.get("vocal_gender") != "instrumental"
    )
    english_vocal = count(
        lambda r: r.get("language") == "en" and r.get("vocal_gender") != "instrumental"
    )
    instrumental = count(lambda r: r.get("vocal_gender") == "instrumental")
    long_form = count(lambda r: int(r.get("duration_requested") or 0) >= 180)

    lines: list[str] = []
    add = lines.append

    add(f"# Phase 5 Baseline Report — {baseline_id}\n")
    add(
        "Measurement only. No training, no prompt tuning, no configuration "
        "changes were made in response to these results.\n"
    )

    if verdict is not None:
        add(render_verdict_section(verdict))
        add("---\n")

    add("## Executive Summary\n")
    if verdict is not None:
        add(
            f"- **Human listening baseline: {verdict.get('overall_score')}/10 — "
            f"{verdict.get('tracks_rejected')}/{verdict.get('tracks_reviewed')} tracks "
            "rejected, not commercially usable.**"
        )
    add(f"- Generations attempted: **{summary.total}**")
    add(f"- Completed: **{summary.completed}** ({summary.success_rate:.0%})")
    add(
        f"- Technical failure rate: **{summary.technical_failure_rate:.1%}** "
        f"(gate: <{MAX_TECHNICAL_FAILURE_RATE:.0%})"
    )
    add(f"- Human evaluations recorded: **{len(scores)}**")
    if averages.get("overall_musical_quality") is not None:
        add(f"- Average overall musical quality: **{averages['overall_musical_quality']}/10**")
    if averages.get("commercial_release_readiness") is not None:
        add(
            "- Average commercial release readiness: "
            f"**{averages['commercial_release_readiness']}/10**"
        )
    add("")
    if notes:
        add(notes + "\n")

    add("## Benchmark Configuration\n")
    add(f"- Baseline id: `{baseline_id}`")
    add(f"- Benchmark version: `{benchmark_version}`")
    add(f"- ACE-Step version: `{ace_step_version}` @ `{ace_step_commit}`")
    configs = sorted({str(r.get("configuration_id")) for r in records})
    add(f"- Configurations: {', '.join(f'`{c}`' for c in configs)}")
    models = sorted({str(r.get("model")) for r in records})
    add(f"- Models: {', '.join(f'`{m}`' for m in models)}")
    add("")

    add("## Hardware\n")
    add(hardware + "\n")

    add("## Generation Counts\n")
    add(f"- Korean vocal: **{korean_vocal}**")
    add(f"- English vocal: **{english_vocal}**")
    add(f"- Instrumental: **{instrumental}**")
    add(f"- Long-form (>=180s): **{long_form}**")
    add("")

    add("## Generation Speed\n")
    if durations:
        add(
            f"- Wall-clock per generation: median **{statistics.median(durations):.1f}s**, "
            f"min {min(durations):.1f}s, max {max(durations):.1f}s"
        )
    if rtfs:
        add(
            f"- Real-time factor: median **{statistics.median(rtfs):.2f}x** "
            f"(wall-clock seconds per second of audio)"
        )
    add("")

    add("## Technical Failure Flags\n")
    flag_counter: Counter[str] = Counter()
    for r in records:
        for flag in _flags(r):
            flag_counter[flag] += 1
    add(_table([(f, str(c)) for f, c in flag_counter.most_common()], ("Flag", "Count")))

    add("## Genre Breakdown (overall musical quality)\n")
    add(
        _table(
            [
                (g, str(v))
                for g, v in group_average(
                    records, scores, "genre", "overall_musical_quality"
                ).items()
            ],
            ("Genre", "Mean"),
        )
    )

    add("## Language Breakdown\n")
    add(
        _table(
            [
                (g, str(v))
                for g, v in group_average(
                    records, scores, "language", "overall_musical_quality"
                ).items()
            ],
            ("Language", "Mean overall"),
        )
    )
    add(
        _table(
            [
                (g, str(v))
                for g, v in group_average(
                    records, scores, "language", "lyrics_pronunciation"
                ).items()
            ],
            ("Language", "Mean pronunciation"),
        )
    )

    add("## Duration Breakdown\n")
    add(
        _table(
            [
                (f"{d}s", str(v))
                for d, v in group_average(
                    records, scores, "duration_requested", "overall_musical_quality"
                ).items()
            ],
            ("Requested duration", "Mean overall"),
        )
    )

    add("## Human Scores vs Internal Quality Gate\n")
    rows = []
    for dimension, target in QUALITY_TARGETS.items():
        measured = averages.get(dimension)
        outcome = "PASS" if targets.get(dimension) else "MISS"
        rows.append(
            (
                dimension,
                f"{measured if measured is not None else 'n/a'} / {target} → **{outcome}**",
            )
        )
    add(_table(rows, ("Dimension", "Measured / Target")))

    add("### All rubric dimensions\n")
    add(
        _table(
            [(d, str(averages[d])) for d in RUBRIC_DIMENSIONS if d in averages],
            ("Dimension", "Mean"),
        )
    )

    add("## Artifact Frequency\n")
    artifacts = artifact_frequency(scores)
    scored = len(scores) or 1
    add(
        _table(
            [(tag, f"{count} ({count / scored:.0%} of scored tracks)") for tag, count in artifacts],
            ("Artifact", "Count"),
        )
    )
    add(f"_Artifact-rate gate: <{MAX_ARTIFACT_RATE:.0%} of tracks with an obvious artifact._\n")

    add("## Seed Variance\n")
    variance = seed_variance(records, scores)
    add(
        _table(
            [
                (
                    p,
                    f"best {v['best']} / median {v['median']} / worst {v['worst']} "
                    f"(spread {v['spread']})",
                )
                for p, v in variance.items()
            ],
            ("Prompt", "Overall quality across seeds"),
        )
    )

    return "\n".join(lines)
