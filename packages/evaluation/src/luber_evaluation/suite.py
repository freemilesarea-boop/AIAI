"""Evaluation suites, and the frozen benchmark they are built from.

A suite is a versioned, hashed set of cases. Two evaluations are only
comparable if they ran the same suite, so the digest is what an
evaluation record cites — and changing a case means a new suite version
rather than an edit, because a silently-changed case makes yesterday's
result incomparable while the columns still line up.

The P20 benchmark is **read-only input**. Its 28 prompts, their lyrics
and their durations are frozen; this module verifies the file's digest
before using it and never writes to it. The Phase 20H rubric, taxonomy
and human score store are equally untouched — and the score store
currently holds nothing, which is recorded as a fact rather than
quietly filled in.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from luber_evaluation.metrics import CATALOGUE, MeasurementMode
from luber_evaluation.schemas import (
    CaseType,
    EvaluationCase,
    GenerationSpec,
    digest_of,
)

SUITE_SCHEMA_VERSION = "luber-evaluation-suite/1"

#: The frozen benchmark, as committed. Verified before every use.
P20_RELATIVE_PATH = "benchmarks/music_quality/prompts/BENCHMARK_P20.json"
P20_EXPECTED_SHA256 = "78dcb22ad9a1de8800667aa9befae13cf2653b0063e6cff7fb16b4a80dfffe13"

#: How far a duration may drift before it counts as wrong. Taken from
#: the P20 objective baseline, where generated durations landed within
#: a small fraction of the request; deliberately not a made-up 0.1 s.
DEFAULT_DURATION_TOLERANCE_RELATIVE = 0.05

#: Tempo detection is only trusted above this. Phase 23 reports a BPM
#: for material with no pulse at all, so an ungated adherence metric
#: would compare two guesses.
DEFAULT_BPM_CONFIDENCE_FLOOR = 0.55


class SuiteError(ValueError):
    """Raised when a suite cannot be built or verified."""


class BenchmarkIntegrityError(SuiteError):
    """Raised when frozen benchmark content does not match its digest."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_p20(repository_root: Path, *, expected: str = P20_EXPECTED_SHA256) -> str:
    """Confirm the frozen benchmark is exactly what it was.

    Raises rather than warning. An evaluation run against a modified
    benchmark produces numbers that look like every other run's and mean
    something different, which is worse than no run at all.
    """
    path = repository_root / P20_RELATIVE_PATH
    if not path.is_file():
        raise BenchmarkIntegrityError(f"the frozen P20 benchmark is missing at {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise BenchmarkIntegrityError(
            f"the P20 benchmark digest is {actual[:16]}… but {expected[:16]}… was "
            "expected; the frozen benchmark must not be modified"
        )
    return actual


@dataclass
class BenchmarkIdentity:
    """What benchmark content an evaluation used."""

    name: str
    path: str
    sha256: str
    case_count: int
    #: Recorded because it is a real and load-bearing fact: the Phase
    #: 20H human baseline has never been scored.
    human_scores_recorded: int = 0
    human_score_store: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "sha256": self.sha256,
            "case_count": self.case_count,
            "human_scores_recorded": self.human_scores_recorded,
            "human_score_store": self.human_score_store,
        }


def p20_identity(repository_root: Path) -> BenchmarkIdentity:
    """Identity of the frozen benchmark, including its empty score store."""
    path = repository_root / P20_RELATIVE_PATH
    digest = verify_p20(repository_root)
    payload = json.loads(path.read_text(encoding="utf-8"))

    store = repository_root / "benchmarks/music_quality/results/p20_human_baseline.jsonl"
    recorded = 0
    if store.is_file():
        recorded = sum(1 for line in store.read_text(encoding="utf-8").splitlines() if line.strip())

    return BenchmarkIdentity(
        name=str(payload.get("benchmark_version", "BENCHMARK_P20")),
        path=P20_RELATIVE_PATH,
        sha256=digest,
        case_count=len(payload.get("prompts", [])),
        human_scores_recorded=recorded,
        human_score_store=str(store.relative_to(repository_root)) if store.is_file() else "absent",
    )


def _case_type_for(prompt: dict[str, Any]) -> str:
    """Classify a P20 prompt using only what the record states."""
    if prompt.get("vocal_gender") == "instrumental":
        return CaseType.INSTRUMENTAL.value
    if float(prompt.get("duration", 0)) >= 120:
        return CaseType.LONG_FORM.value
    language = str(prompt.get("language", ""))
    if language == "ko":
        return CaseType.KOREAN_VOCAL.value
    if language == "en":
        return CaseType.ENGLISH_VOCAL.value
    return CaseType.TEXT_TO_MUSIC.value


def _applicable_metrics(prompt: dict[str, Any]) -> tuple[str, ...]:
    """Which metrics this case can support.

    Lyric metrics are attached to cases that *have* lyrics even though
    nothing can measure them today. That is deliberate: the case says
    what would be checkable, and the runner records NOT_MEASURABLE with
    the reason, so the gap is visible rather than absent.
    """
    names: list[str] = [
        "generation_success_rate",
        "generation_failure_rate",
        "invalid_audio_rate",
        "silent_output_rate",
        "early_collapse_rate",
        "wrong_duration_rate",
        "duration_absolute_error_seconds",
        "duration_relative_error",
        "clipping_sample_ratio",
        "silence_ratio",
        "peak_dbfs",
        "true_peak_dbtp",
        "integrated_lufs",
        "crest_factor_db",
        "spectral_centroid_hz",
        "high_frequency_energy_ratio",
        "stereo_width",
        "phase_correlation",
        "sample_rate",
        "channels",
    ]
    if prompt.get("lyrics"):
        names += ["lyric_line_coverage", "lyric_word_coverage"]
    if prompt.get("vocal_gender") == "instrumental":
        names.append("instrumental_adherence")
    if prompt.get("language") == "ko" and prompt.get("lyrics"):
        names += ["korean_pronunciation", "trot_style_absence"]
    if prompt.get("vocal_gender") != "instrumental":
        names.append("vocal_naturalness")
    return tuple(sorted(set(names)))


def load_p20_cases(repository_root: Path) -> list[EvaluationCase]:
    """Build evaluation cases from the frozen benchmark. Read-only."""
    path = repository_root / P20_RELATIVE_PATH
    verify_p20(repository_root)
    payload = json.loads(path.read_text(encoding="utf-8"))

    cases: list[EvaluationCase] = []
    for prompt in payload.get("prompts", []):
        cases.append(
            EvaluationCase(
                case_id=str(prompt["prompt_id"]),
                case_type=_case_type_for(prompt),
                spec=GenerationSpec(
                    prompt=str(prompt.get("prompt", "")),
                    lyrics=str(prompt.get("lyrics", "") or ""),
                    duration_seconds=float(prompt.get("duration", 0)),
                    language=str(prompt.get("language", "unknown")),
                    vocal_gender=str(prompt.get("vocal_gender", "unknown") or "unknown"),
                    task=CaseType.TEXT_TO_MUSIC.value,
                ),
                applicable_metrics=_applicable_metrics(prompt),
                origin=f"{P20_RELATIVE_PATH}#{prompt['prompt_id']}",
                tags=tuple(sorted(str(t) for t in prompt.get("tests", []) or [])),
            )
        )
    return sorted(cases, key=lambda case: case.case_id)


@dataclass
class EvaluationSuite:
    """A versioned, hashable set of cases and the rules for running them."""

    suite_id: str
    suite_version: str
    description: str = ""
    cases: list[EvaluationCase] = field(default_factory=list)
    #: Seeds every case is generated at. More than one, because a single
    #: seed compares two draws rather than two models.
    seeds: tuple[int, ...] = (11, 23, 37)
    mode: str = "RAW_MODEL"
    duration_tolerance_relative: float = DEFAULT_DURATION_TOLERANCE_RELATIVE
    bpm_confidence_floor: float = DEFAULT_BPM_CONFIDENCE_FLOOR
    benchmark: BenchmarkIdentity | None = None
    schema_version: str = SUITE_SCHEMA_VERSION

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(sorted(case.case_id for case in self.cases))

    def case(self, case_id: str) -> EvaluationCase:
        for candidate in self.cases:
            if candidate.case_id == case_id:
                return candidate
        raise SuiteError(f"case {case_id!r} is not in suite {self.suite_id}")

    def required_metrics(self) -> tuple[str, ...]:
        """Every automatic metric any case in the suite supports."""
        names: set[str] = set()
        for case in self.cases:
            for name in case.applicable_metrics:
                spec = CATALOGUE.get(name)
                if spec and spec.mode == MeasurementMode.AUTOMATIC.value:
                    names.add(name)
        return tuple(sorted(names))

    def human_required_metrics(self) -> tuple[str, ...]:
        names: set[str] = set()
        for case in self.cases:
            for name in case.applicable_metrics:
                spec = CATALOGUE.get(name)
                if spec and spec.mode == MeasurementMode.HUMAN_REQUIRED.value:
                    names.add(name)
        return tuple(sorted(names))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
            "description": self.description,
            "mode": self.mode,
            "seeds": list(self.seeds),
            "duration_tolerance_relative": self.duration_tolerance_relative,
            "bpm_confidence_floor": self.bpm_confidence_floor,
            "benchmark": self.benchmark.to_dict() if self.benchmark else None,
            "cases": [case.to_dict() for case in sorted(self.cases, key=lambda c: c.case_id)],
        }

    def digest(self) -> str:
        return digest_of(self.to_dict())


def build_p20_suite(
    repository_root: Path,
    *,
    suite_version: str = "1",
    seeds: tuple[int, ...] = (11, 23, 37),
    mode: str = "RAW_MODEL",
) -> EvaluationSuite:
    """The standard suite: every frozen P20 case, at several seeds."""
    identity = p20_identity(repository_root)
    return EvaluationSuite(
        suite_id="P20_FULL",
        suite_version=suite_version,
        description=(
            "Every case in the frozen P20 benchmark, evaluated on raw model output. "
            "The benchmark is read-only input and is never modified."
        ),
        cases=load_p20_cases(repository_root),
        seeds=seeds,
        mode=mode,
        benchmark=identity,
    )


def experiment_suite(
    base: EvaluationSuite,
    extra_cases: list[EvaluationCase],
    *,
    suite_id: str,
    suite_version: str = "1",
    description: str = "",
) -> EvaluationSuite:
    """A suite that adds experiment-specific cases to a global base.

    The added cases must not collide with benchmark ids: a suite where
    an experiment silently replaced a frozen case would produce results
    labelled P20 that are not.
    """
    existing = set(base.case_ids)
    collisions = sorted(case.case_id for case in extra_cases if case.case_id in existing)
    if collisions:
        raise SuiteError(
            f"experiment cases collide with frozen benchmark ids: {', '.join(collisions)}"
        )
    return EvaluationSuite(
        suite_id=suite_id,
        suite_version=suite_version,
        description=description or f"{base.suite_id} plus experiment-specific cases",
        cases=[*base.cases, *extra_cases],
        seeds=base.seeds,
        mode=base.mode,
        duration_tolerance_relative=base.duration_tolerance_relative,
        bpm_confidence_floor=base.bpm_confidence_floor,
        benchmark=base.benchmark,
    )


def smoke_suite(*, seeds: tuple[int, ...] = (11,)) -> EvaluationSuite:
    """A tiny synthetic suite for infrastructure tests.

    Deliberately does not read the frozen benchmark: an infrastructure
    test should not depend on the benchmark being present, and should
    never be able to touch it.
    """
    cases = [
        EvaluationCase(
            case_id="SYN-KO-01",
            case_type=CaseType.KOREAN_VOCAL.value,
            spec=GenerationSpec(
                prompt="synthetic korean vocal case",
                lyrics="[Verse]\n가사",
                duration_seconds=60.0,
                language="ko",
                vocal_gender="female",
            ),
            applicable_metrics=(
                "generation_success_rate",
                "invalid_audio_rate",
                "duration_absolute_error_seconds",
                "clipping_sample_ratio",
                "silence_ratio",
                "phase_correlation",
                "korean_pronunciation",
                "vocal_naturalness",
            ),
            origin="synthetic",
        ),
        EvaluationCase(
            case_id="SYN-INST-01",
            case_type=CaseType.INSTRUMENTAL.value,
            spec=GenerationSpec(
                prompt="synthetic instrumental case",
                duration_seconds=60.0,
                language="unknown",
                vocal_gender="instrumental",
            ),
            applicable_metrics=(
                "generation_success_rate",
                "invalid_audio_rate",
                "duration_absolute_error_seconds",
                "clipping_sample_ratio",
                "silence_ratio",
                "phase_correlation",
            ),
            origin="synthetic",
        ),
    ]
    return EvaluationSuite(
        suite_id="SYNTHETIC_SMOKE",
        suite_version="1",
        description="synthetic cases for infrastructure tests; no benchmark content",
        cases=cases,
        seeds=seeds,
    )
