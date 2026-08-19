"""JSON-safe views of analyses and plans.

Finishing decisions have to be auditable months later, when the audio is
the only thing left and nobody remembers why a shelf was applied. These
functions produce the record that makes that possible: every measurement
that existed, every decision taken, and the engine version that connects
them.

``NaN`` becomes ``None``. JSON has no NaN, and the alternatives — writing
bare ``NaN`` tokens that strict parsers reject, or substituting 0.0 —
either break the file or turn "not measurable" into "measured as zero".
"""

from __future__ import annotations

import math
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from luber_audio_finishing.acceptance import AcceptanceVerdict
from luber_audio_finishing.analysis import AudioAnalysis, Distribution
from luber_audio_finishing.decision import FinishingPlan
from luber_audio_finishing.report import AudioAnalysisReport


def _clean(value: Any) -> Any:
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, Path):
        # Only the file name: absolute paths are machine-specific and
        # must not end up in committed benchmark records.
        return value.name
    if isinstance(value, Distribution):
        return {
            "p10": _clean(value.p10),
            "p50": _clean(value.p50),
            "p90": _clean(value.p90),
            "mean": _clean(value.mean),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _clean(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    return value


def analysis_to_dict(analysis: AudioAnalysis) -> dict[str, Any]:
    """Flatten an analysis into a JSON-serialisable record."""
    return {field.name: _clean(getattr(analysis, field.name)) for field in fields(analysis)}


def report_to_dict(report: AudioAnalysisReport) -> dict[str, Any]:
    """An analysis plus the risks it raised, as one record."""
    payload = analysis_to_dict(report.analysis)
    payload["risk_flags"] = [_clean(finding) for finding in report.risk_flags]
    return payload


def plan_to_dict(plan: FinishingPlan) -> dict[str, Any]:
    """The full decision trail: risks, actions, ceilings, deferrals.

    ``no_action`` is written explicitly rather than left implicit in an
    empty action list, because "the engine decided to do nothing" and
    "the engine did not run" must not look the same in a record.
    """
    return {
        "finishing_version": plan.finishing_version,
        "no_action": plan.is_no_action,
        "output_ceiling_dbfs": plan.output_ceiling_dbfs,
        "match_source_loudness": plan.match_source_loudness,
        "risks": [_clean(finding) for finding in plan.risks],
        "actions": [_clean(action) for action in plan.actions],
        "deferred": [_clean(item) for item in plan.deferred],
        # Corrections the rules called for and the engine declined. Kept
        # because an empty action list has two very different meanings —
        # nothing was wrong, or something was wrong and was left alone —
        # and only this tells them apart.
        "suppressed": [_clean(item) for item in plan.suppressed],
    }


def verdict_to_dict(verdict: AcceptanceVerdict) -> dict[str, Any]:
    """Why a render was kept or thrown away.

    Every check is recorded, not only the failures. A verdict that lists
    two objections tells you what went wrong; one that lists eighteen
    passes and two objections also tells you what was examined, which is
    what makes a later "why was this accepted?" answerable.
    """
    return {
        "outcome": verdict.outcome.value,
        "accepted": verdict.accepted,
        "summary": verdict.summary(),
        "failed_checks": [check.name for check in verdict.failures],
        "checks": [
            {
                "kind": check.kind.value,
                "name": check.name,
                "passed": check.passed,
                "detail": check.detail,
                "source_value": _clean(check.source_value),
                "finished_value": _clean(check.finished_value),
                "tolerance": _clean(check.tolerance),
            }
            for check in verdict.checks
        ],
    }
