"""Rebuilding evaluation objects from the files that recorded them.

Every command after `run start` — compare, qualify, verify, rank — works
from artifacts on disk rather than from memory, and this is where those
files become objects again.

The reconstruction is strict in one specific way: it rebuilds only the
fields a class declares and ignores derived keys such as
``advisory_score`` or ``pareto``. Those are computed views, and reading
them back as though they were state would let a hand-edited file assert
a summary that its own metrics contradict. Everything derived is
derived again.

What this module deliberately does not do is repair. A truncated
``metrics.jsonl`` line, a metric carrying a value it should not, a
policy naming an unknown field — all raise. A verification pass that
silently dropped the damaged records would report a clean evaluation
built from fewer measurements than the one that ran.
"""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

from luber_evaluation.comparison import CandidateComparison, MetricComparison
from luber_evaluation.metrics import Aggregate, MetricResult
from luber_evaluation.qualification import (
    Coverage,
    GateOutcome,
    HardGate,
    QualificationDecision,
    QualificationPolicy,
)
from luber_evaluation.runner import EvaluationRun
from luber_evaluation.schemas import (
    CandidateLineage,
    EvaluationCase,
    GenerationSpec,
    ModelRef,
    PromotionReview,
    SampleProvenance,
)
from luber_evaluation.suite import BenchmarkIdentity, EvaluationSuite

T = TypeVar("T")


class DeserialisationError(ValueError):
    """Raised when a recorded artifact cannot be turned back into itself."""


def _known(cls: type[Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Only the fields *cls* declares, so derived keys are dropped."""
    if not is_dataclass(cls):  # pragma: no cover - programming error
        raise DeserialisationError(f"{cls!r} is not a dataclass")
    names = {field.name for field in fields(cls)}
    return {key: value for key, value in payload.items() if key in names}


def build(cls: type[T], payload: dict[str, Any]) -> T:
    """Rebuild a flat dataclass, reporting *which* record failed."""
    try:
        return cls(**_known(cls, payload))
    except (TypeError, ValueError) as exc:
        raise DeserialisationError(f"cannot rebuild {cls.__name__}: {exc}") from exc


# ── suite ────────────────────────────────────────────────────────────
def case_from_dict(payload: dict[str, Any]) -> EvaluationCase:
    data = _known(EvaluationCase, payload)
    data["spec"] = build(GenerationSpec, payload.get("spec", {}))
    data["applicable_metrics"] = tuple(payload.get("applicable_metrics", ()))
    data["tags"] = tuple(payload.get("tags", ()))
    return build(EvaluationCase, data)


def suite_from_dict(payload: dict[str, Any]) -> EvaluationSuite:
    data = _known(EvaluationSuite, payload)
    data["cases"] = [case_from_dict(case) for case in payload.get("cases", [])]
    data["seeds"] = tuple(int(seed) for seed in payload.get("seeds", ()))
    benchmark = payload.get("benchmark")
    data["benchmark"] = build(BenchmarkIdentity, benchmark) if benchmark else None
    return build(EvaluationSuite, data)


# ── policy ───────────────────────────────────────────────────────────
def policy_from_dict(payload: dict[str, Any]) -> QualificationPolicy:
    data = _known(QualificationPolicy, payload)
    data["hard_gates"] = tuple(build(HardGate, gate) for gate in payload.get("hard_gates", ()))
    data["never_regress"] = tuple(payload.get("never_regress", ()))
    return build(QualificationPolicy, data)


# ── run ──────────────────────────────────────────────────────────────
def run_from_dict(payload: dict[str, Any], suite: EvaluationSuite) -> EvaluationRun:
    """Rebuild a run against a suite loaded separately.

    The suite is passed in rather than read from the run record because
    the run stores only its id and digest. Verification then has two
    independent things to compare, which is the whole point of storing
    the digest.
    """
    data = _known(EvaluationRun, payload)
    data["suite"] = suite
    data["baseline"] = build(ModelRef, payload.get("baseline", {}))
    data["candidate"] = build(ModelRef, payload.get("candidate", {}))
    data["lineage"] = build(CandidateLineage, payload.get("lineage", {}))
    data["seeds"] = tuple(int(seed) for seed in payload.get("seeds", ()))
    cost = payload.get("cost", {})
    data["wall_seconds"] = float(cost.get("wall_seconds", 0.0))
    data["gpu_seconds"] = cost.get("gpu_seconds")
    data["generated_audio_seconds"] = float(cost.get("generated_audio_seconds", 0.0))
    return build(EvaluationRun, data)


# ── metrics and comparison ───────────────────────────────────────────
def metric_from_dict(payload: dict[str, Any]) -> MetricResult:
    return build(MetricResult, payload)


def aggregate_from_dict(payload: dict[str, Any]) -> Aggregate:
    data = _known(Aggregate, payload)
    data["failed_cases"] = list(payload.get("failed_cases", []))
    return build(Aggregate, data)


def aggregates_from_dict(payload: dict[str, Any]) -> dict[str, Aggregate]:
    return {name: aggregate_from_dict(value) for name, value in payload.items()}


def comparison_from_dict(payload: dict[str, Any]) -> CandidateComparison:
    data = _known(CandidateComparison, payload)
    data["metrics"] = {
        name: build(MetricComparison, value) for name, value in payload.get("metrics", {}).items()
    }
    return build(CandidateComparison, data)


def decision_from_dict(payload: dict[str, Any]) -> QualificationDecision:
    data = _known(QualificationDecision, payload)
    data["gate_outcomes"] = [build(GateOutcome, item) for item in payload.get("gate_outcomes", [])]
    return build(QualificationDecision, data)


def coverage_from_dict(payload: dict[str, Any]) -> Coverage:
    return build(Coverage, payload)


def sample_from_dict(payload: dict[str, Any]) -> SampleProvenance:
    return build(SampleProvenance, payload)


def promotion_review_from_dict(payload: dict[str, Any]) -> PromotionReview:
    return build(PromotionReview, payload)


# ── files ────────────────────────────────────────────────────────────
def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DeserialisationError(f"{path} does not exist")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DeserialisationError(f"{path} does not hold an object")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every record, or an error naming the line that broke.

    Skipping a malformed line would quietly shrink the evidence an
    evaluation rests on while leaving its coverage figures unchanged.
    """
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DeserialisationError(f"{path}:{number} is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise DeserialisationError(f"{path}:{number} does not hold an object")
        rows.append(payload)
    return rows


def read_metrics(path: Path, *, side: str | None = None) -> list[MetricResult]:
    """Metric records, optionally only one side's.

    ``metrics.jsonl`` holds both sides in one file with a ``side`` key,
    so that no reader can ever pair a baseline file with a candidate
    file from a different run.
    """
    results: list[MetricResult] = []
    for row in read_jsonl(path):
        if side is not None and row.get("side") != side:
            continue
        results.append(metric_from_dict(row))
    return results


__all__ = [
    "DeserialisationError",
    "aggregate_from_dict",
    "aggregates_from_dict",
    "build",
    "case_from_dict",
    "comparison_from_dict",
    "coverage_from_dict",
    "decision_from_dict",
    "metric_from_dict",
    "policy_from_dict",
    "promotion_review_from_dict",
    "read_json",
    "read_jsonl",
    "read_metrics",
    "run_from_dict",
    "sample_from_dict",
    "suite_from_dict",
]
