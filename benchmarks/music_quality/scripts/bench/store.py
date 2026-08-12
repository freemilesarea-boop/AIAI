"""JSONL result store for benchmark runs and human scores.

JSONL because runs are append-only, resumable, diffable, and small
enough to commit — only metadata and scores are stored here, never
audio. Each generation is one line keyed by ``benchmark_id``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GenerationRecord:
    """One benchmark generation: what was asked, what came back."""

    benchmark_id: str
    benchmark_version: str
    prompt_id: str
    genre: str
    language: str
    vocal_gender: str
    duration_requested: int
    lyrics_structure: str
    prompt_style: str
    # Both prompts are recorded so a quality problem can be attributed to
    # the compiler or to the model, not guessed at.
    prompt: str
    compiled_prompt: str | None
    lyrics: str
    model: str
    model_version: str | None
    lm_enabled: bool
    thinking_enabled: bool
    inference_steps: int
    seed: int | None
    configuration_id: str
    runtime_backend: str
    generation_id: str | None = None
    generation_started_at: str | None = None
    generation_completed_at: str | None = None
    generation_seconds: float | None = None
    output_sha256: str | None = None
    output_duration: float | None = None
    output_path: str | None = None
    preview_sha256: str | None = None
    status: str = "PENDING"
    error_code: str | None = None
    reference_system: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    real_time_factor: float | None = None
    disk_free_gb_before: float | None = None
    swap_used_gb: float | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class ResultStore:
    """Append-only JSONL store with resume support."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> list[dict[str, Any]]:
        if not self._path.is_file():
            return []
        records: list[dict[str, Any]] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A truncated final line (killed mid-write) must not make
                # the whole run unresumable.
                continue
        return records

    def completed_ids(self) -> set[str]:
        """Benchmark ids that finished successfully and need no rerun."""
        return {
            str(r["benchmark_id"])
            for r in self.load()
            if r.get("status") == "COMPLETED" and r.get("benchmark_id")
        }

    def all_ids(self) -> set[str]:
        return {str(r["benchmark_id"]) for r in self.load() if r.get("benchmark_id")}

    def append(self, record: GenerationRecord) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(record.to_json() + "\n")


@dataclass
class ScoreRecord:
    """One human evaluation of one benchmark generation."""

    benchmark_id: str
    evaluator: str
    scored_at: str
    blind: bool
    scores: dict[str, int]
    artifact_tags: list[str] = field(default_factory=list)
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class ScoreStore:
    """Append-only JSONL store of human listening scores."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> list[dict[str, Any]]:
        if not self._path.is_file():
            return []
        out: list[dict[str, Any]] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def append(self, record: ScoreRecord) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(record.to_json() + "\n")
