"""The report a person reads before deciding to spend GPU money.

Organised around the questions somebody actually asks, in the order they
ask them, rather than around the modules that produced the answers. The
section nobody would think to request — *what cannot be assessed* — is
given equal weight, because a confident report that quietly omits its
own blind spots is worse than no report.

Numbers here always travel with their denominator. "60% pop" from a
corpus that is 10% labelled is not a fact about the corpus, and the
report is the last place that distinction can still be made before
somebody acts on it.
"""

from __future__ import annotations

from datetime import UTC, datetime

from luber_dataset.factory.intelligence.curation import CurationResult
from luber_dataset.factory.intelligence.profile import DatasetProfile
from luber_dataset.factory.intelligence.reports import build_wishlist
from luber_dataset.factory.intelligence.schemas import Severity

#: How many risks the report leads with. Ten is the brief the report was
#: written to answer; beyond that a reader stops triaging and starts
#: skimming.
TOP_RISKS = 10


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _table(rows: list[list[str]], header: list[str]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _dominance_lines(profile: DatasetProfile) -> list[str]:
    lines: list[str] = []
    rows: list[list[str]] = []
    for name in ("artist", "source_reference", "source_type", "language", "genre"):
        metrics = profile.concentration.get(name)
        distribution = profile.categorical.get(name)
        if metrics is None or distribution is None or metrics.known_count == 0:
            continue
        rows.append(
            [
                name,
                str(metrics.top1_label),
                _pct(metrics.top1_share),
                _pct(metrics.top5_share),
                f"{metrics.effective_categories:.1f}",
                f"{metrics.known_count} ({_pct(distribution.coverage)} coverage)",
            ]
        )
    if rows:
        lines.extend(
            _table(
                rows,
                ["dimension", "largest", "top-1", "top-5", "effective count", "known of"],
            )
        )
    else:
        lines.append("_No dimension has enough known values to describe dominance._")
    return lines


def render(result: CurationResult) -> str:
    eligible = result.eligible_profile
    selected = result.selected_profile
    corpus = result.corpus_profile
    lines: list[str] = []

    lines.append("# Dataset curation report")
    lines.append("")
    lines.append(f"- Generated: {datetime.now(UTC).isoformat()}")
    lines.append(f"- Target profile: **{result.target_profile.name}**")
    lines.append(f"- Source manifest: `{result.source_manifest_sha256[:16]}…`")
    lines.append(f"- Engine: `{result.config.digest()[:16]}…` config")
    lines.append("")

    if corpus and eligible and selected:
        lines.append(
            f"The manifest holds **{corpus.track_count} tracks / "
            f"{corpus.total_hours:.2f} h**. "
            f"**{eligible.track_count}** of them ({eligible.total_hours:.2f} h) are "
            f"training-eligible, and curation selected **{selected.track_count}** "
            f"({selected.total_hours:.2f} h)."
        )
        lines.append("")

    # ── 1. top risks ─────────────────────────────────────────────────
    lines.append(f"## Top {TOP_RISKS} risks if we train on this now")
    lines.append("")
    ranked = [f for f in result.findings if f.severity != Severity.INFO.value][:TOP_RISKS]
    if not ranked:
        lines.append(
            "_No critical or warning findings. That is a statement about the checks "
            "that ran, not a guarantee — see what cannot be assessed, below._"
        )
    else:
        for index, finding in enumerate(ranked, start=1):
            lines.append(f"{index}. **{finding.code}** ({finding.severity}) — {finding.detail}.")
            if finding.recommended_action:
                lines.append(f"   - Action: {finding.recommended_action}")
            if finding.known_denominator is not None:
                lines.append(f"   - Measured over {finding.known_denominator} known values")
    lines.append("")

    # ── 2. what dominates ────────────────────────────────────────────
    lines.append("## What dominates this dataset?")
    lines.append("")
    if eligible:
        lines.extend(_dominance_lines(eligible))
        lines.append("")
        pressure = eligible.family_pressure
        if pressure.total_tracks:
            lines.append(
                f"Duplicate families: **{pressure.unique_families}** families across "
                f"{pressure.total_tracks} tracks; the largest holds "
                f"{pressure.largest_family}. The corpus behaves as though it has "
                f"{pressure.effective_families:.1f} independent families."
            )
            lines.append("")

    # ── 3. what is missing ───────────────────────────────────────────
    lines.append("## What is missing?")
    lines.append("")
    gaps = [f for f in result.findings if f.code.startswith("NEED_MORE_")]
    if not gaps:
        lines.append(
            "_No gaps. A gap only exists relative to a declared target, and the "
            f"**{result.target_profile.name}** profile "
            + (
                "declares none — it detects domination only."
                if not result.target_profile.shares
                else "found every declared minimum satisfied."
            )
        )
    else:
        for finding in gaps:
            bounds = finding.target_range
            lines.append(
                f"- **{finding.dimension} = "
                f"{finding.code.replace(f'NEED_MORE_{finding.dimension.upper()}_', '').lower()}**: "
                f"{_pct(finding.current_share)} against a minimum of "
                f"{_pct(bounds[0] if bounds else None)}"
            )
    lines.append("")

    # ── 4. what is uncertain ─────────────────────────────────────────
    lines.append("## What is uncertain?")
    lines.append("")
    if eligible:
        rows = [
            [
                score.dimension,
                str(score.known),
                str(score.unknown),
                f"{score.missing_percentage:.0f}%",
                str(score.low_confidence),
            ]
            for _, score in sorted(eligible.completeness.items())
        ]
        lines.extend(_table(rows, ["dimension", "known", "unknown", "missing", "low confidence"]))
    lines.append("")

    # ── 5. what cannot be assessed ───────────────────────────────────
    lines.append("## What cannot be assessed, and why")
    lines.append("")
    unassessable = [f for f in result.findings if f.code == "NOT_ASSESSABLE"]
    if unassessable:
        for finding in unassessable:
            lines.append(f"- **{finding.dimension}** — {finding.detail}")
    lines.append(
        "- **Song structure** — no segmenter exists; Phase 23 records "
        "`structure_status: UNAVAILABLE` rather than guessing."
    )
    lines.append(
        "- **Transcripts** — no speech recogniser is configured, so lyrics exist only "
        "where an operator supplied them."
    )
    lines.append(
        "- **Vocal class** — no validated detector. Every classified track was "
        "classified by a human writing a sidecar."
    )
    lines.append(
        "- **Trot vs. modern style** — nothing in the manifest distinguishes them, so "
        "no profile can target it yet."
    )
    lines.append("")

    # ── 6. what to add, what to reduce ───────────────────────────────
    lines.append("## What should be added?")
    lines.append("")
    wishlist = build_wishlist(result)
    if not wishlist:
        lines.append("_Nothing the current target profile can justify asking for._")
    else:
        rows = [
            [
                str(entry["dimension"]),
                str(entry["target"]),
                _pct(entry.get("current_share")),
                _pct(entry.get("minimum_share")),
                (
                    f"{entry['estimated_hours_needed']:.1f} h"
                    if entry.get("estimated_hours_needed") is not None
                    else "not derivable"
                ),
                str(entry["priority"]),
            ]
            for entry in wishlist
        ]
        lines.extend(
            _table(rows, ["dimension", "target", "current", "minimum", "needed", "priority"])
        )
    lines.append("")

    lines.append("## What should be reduced?")
    lines.append("")
    reductions = [
        f
        for f in result.findings
        if f.code.endswith("_OVERREPRESENTED") or f.code.endswith("_DOMINATES")
    ]
    if not reductions:
        lines.append("_Nothing exceeds a declared ceiling._")
    else:
        for finding in reductions:
            lines.append(
                f"- **{finding.dimension}** — {finding.detail}. "
                f"{finding.affected_tracks} tracks, {finding.affected_hours:.2f} h."
            )
    lines.append("")

    # ── 7. what curation did ─────────────────────────────────────────
    lines.append("## What curation did")
    lines.append("")
    actions: dict[str, int] = {}
    for decision in result.selection.decisions.values():
        actions[decision.action] = actions.get(decision.action, 0) + 1
    lines.extend(
        _table(
            [[action, str(count)] for action, count in sorted(actions.items())],
            ["action", "tracks"],
        )
    )
    lines.append("")
    if result.selection.excluded_by_reason:
        lines.append("Exclusion reasons:")
        lines.append("")
        lines.extend(
            _table(
                [
                    [reason, str(count)]
                    for reason, count in sorted(result.selection.excluded_by_reason.items())
                ],
                ["reason", "tracks"],
            )
        )
        lines.append("")

    plan = result.sampling_plan
    lines.append(
        f"Sampling weights: {len(plan.weights)} tracks weighted, bounded to "
        f"[{plan.min_weight}, {plan.max_weight}] — "
        + ("all within bounds." if plan.bounded else "**OUT OF BOUNDS**.")
    )
    lines.append("")
    lines.append(
        "_Rights are a hard gate applied before any of this. No score, weight or "
        "target can admit a track whose training permission is not TRUE._"
    )
    lines.append("")
    return "\n".join(lines)
