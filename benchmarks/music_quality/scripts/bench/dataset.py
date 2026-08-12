"""Benchmark prompt dataset loading and validation.

The dataset is version-controlled and frozen: prompts are never edited
to chase a better score. Changing any prompt requires a new
``benchmark_version``, and results are only ever compared within the
same version.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

VALID_GENRES = {
    "KPOP",
    "BALLAD",
    "RNB",
    "HIPHOP",
    "ROCK",
    "INDIE",
    "JAZZ",
    "LOFI",
    "EDM",
    "ACOUSTIC",
}
VALID_VOCALS = {"female", "male", "instrumental"}
VALID_STYLES = {"simple", "detailed"}
VALID_STRUCTURES = {"simple", "complex", "none"}
#: Minimum share of the dataset that must be Korean-language vocal music.
MIN_KOREAN_VOCAL_RATIO = 0.40
MIN_PROMPTS_PER_GENRE = 6


class DatasetError(Exception):
    """Raised when the benchmark prompt dataset is invalid."""


@dataclass(frozen=True)
class BenchmarkPrompt:
    prompt_id: str
    genre: str
    language: str
    vocal_gender: str
    prompt_style: str
    structure: str
    prompt: str
    lyrics: str

    @property
    def is_instrumental(self) -> bool:
        return self.vocal_gender == "instrumental"

    @property
    def is_korean_vocal(self) -> bool:
        return self.language == "ko" and not self.is_instrumental


@dataclass(frozen=True)
class BenchmarkDataset:
    benchmark_version: str
    prompts: tuple[BenchmarkPrompt, ...]

    def by_id(self, prompt_id: str) -> BenchmarkPrompt:
        for p in self.prompts:
            if p.prompt_id == prompt_id:
                return p
        raise DatasetError(f"unknown prompt_id: {prompt_id}")

    def korean_vocal_ratio(self) -> float:
        if not self.prompts:
            return 0.0
        return sum(1 for p in self.prompts if p.is_korean_vocal) / len(self.prompts)


def load_dataset(path: Path) -> BenchmarkDataset:
    """Load and fully validate a benchmark prompt dataset."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetError(f"dataset not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetError(f"dataset is not valid JSON: {exc}") from exc

    version = raw.get("benchmark_version")
    if not version:
        raise DatasetError("dataset is missing benchmark_version")

    entries = raw.get("prompts")
    if not isinstance(entries, list) or not entries:
        raise DatasetError("dataset has no prompts")

    prompts: list[BenchmarkPrompt] = []
    required = (
        "prompt_id",
        "genre",
        "language",
        "vocal_gender",
        "prompt_style",
        "structure",
        "prompt",
        "lyrics",
    )
    for index, entry in enumerate(entries):
        missing = [field for field in required if field not in entry]
        if missing:
            raise DatasetError(f"prompt #{index} is missing fields: {', '.join(missing)}")
        prompts.append(BenchmarkPrompt(**{field: entry[field] for field in required}))

    _validate(prompts)
    return BenchmarkDataset(benchmark_version=str(version), prompts=tuple(prompts))


def _validate(prompts: list[BenchmarkPrompt]) -> None:
    ids = [p.prompt_id for p in prompts]
    duplicates = sorted({i for i, count in Counter(ids).items() if count > 1})
    if duplicates:
        raise DatasetError(f"duplicate prompt_id values: {', '.join(duplicates)}")

    for p in prompts:
        if p.genre not in VALID_GENRES:
            raise DatasetError(f"{p.prompt_id}: unknown genre {p.genre!r}")
        if p.vocal_gender not in VALID_VOCALS:
            raise DatasetError(f"{p.prompt_id}: unknown vocal_gender {p.vocal_gender!r}")
        if p.prompt_style not in VALID_STYLES:
            raise DatasetError(f"{p.prompt_id}: unknown prompt_style {p.prompt_style!r}")
        if p.structure not in VALID_STRUCTURES:
            raise DatasetError(f"{p.prompt_id}: unknown structure {p.structure!r}")
        if not p.prompt.strip():
            raise DatasetError(f"{p.prompt_id}: empty prompt")
        # A vocal track without lyrics would silently become instrumental
        # upstream, which would quietly corrupt the vocal benchmark.
        if not p.is_instrumental and not p.lyrics.strip():
            raise DatasetError(f"{p.prompt_id}: vocal prompt has no lyrics")
        if p.is_instrumental and p.lyrics.strip():
            raise DatasetError(f"{p.prompt_id}: instrumental prompt must have empty lyrics")

    per_genre = Counter(p.genre for p in prompts)
    thin = sorted(g for g, c in per_genre.items() if c < MIN_PROMPTS_PER_GENRE)
    if thin:
        raise DatasetError(
            f"genres with fewer than {MIN_PROMPTS_PER_GENRE} prompts: {', '.join(thin)}"
        )

    korean = sum(1 for p in prompts if p.is_korean_vocal) / len(prompts)
    if korean < MIN_KOREAN_VOCAL_RATIO:
        raise DatasetError(
            f"Korean vocal share {korean:.0%} is below the required {MIN_KOREAN_VOCAL_RATIO:.0%}"
        )
