"""Prompt dataset validation, including the shipped v1 dataset."""

import json
from pathlib import Path

import pytest

from bench.dataset import (
    MIN_KOREAN_VOCAL_RATIO,
    MIN_PROMPTS_PER_GENRE,
    DatasetError,
    load_dataset,
)

BENCH_ROOT = Path(__file__).resolve().parents[1]
V1 = BENCH_ROOT / "prompts" / "BENCHMARK_V1.json"


def _write(tmp_path: Path, prompts: list[dict], version: str = "test") -> Path:
    path = tmp_path / "ds.json"
    path.write_text(
        json.dumps({"benchmark_version": version, "prompts": prompts}), encoding="utf-8"
    )
    return path


def _prompt(**overrides) -> dict:
    base = {
        "prompt_id": "KPOP-01",
        "genre": "KPOP",
        "language": "ko",
        "vocal_gender": "female",
        "prompt_style": "simple",
        "structure": "simple",
        "prompt": "bright K-pop",
        "lyrics": "[Verse]\n가사",
    }
    base.update(overrides)
    return base


def _full_valid_set() -> list[dict]:
    """Six prompts per genre, Korean vocal share above the floor."""
    genres = ["KPOP", "BALLAD", "RNB", "HIPHOP", "ROCK", "INDIE", "JAZZ", "LOFI", "EDM", "ACOUSTIC"]
    out = []
    for g in genres:
        for i in range(MIN_PROMPTS_PER_GENRE):
            out.append(_prompt(prompt_id=f"{g}-{i:02d}", genre=g))
    return out


# ── the shipped dataset ───────────────────────────────────────────────


def test_shipped_v1_dataset_is_valid():
    dataset = load_dataset(V1)
    assert dataset.benchmark_version == "v1"
    assert len(dataset.prompts) >= 60


def test_shipped_v1_meets_korean_requirement():
    dataset = load_dataset(V1)
    assert dataset.korean_vocal_ratio() >= MIN_KOREAN_VOCAL_RATIO


def test_shipped_v1_covers_every_genre_and_type():
    dataset = load_dataset(V1)
    genres = {p.genre for p in dataset.prompts}
    assert len(genres) == 10
    assert {p.vocal_gender for p in dataset.prompts} == {"female", "male", "instrumental"}
    assert {p.prompt_style for p in dataset.prompts} == {"simple", "detailed"}
    assert "complex" in {p.structure for p in dataset.prompts}
    assert {p.language for p in dataset.prompts} >= {"ko", "en"}


def test_shipped_v1_complex_structures_use_full_section_set():
    dataset = load_dataset(V1)
    complex_prompts = [p for p in dataset.prompts if p.structure == "complex"]
    assert complex_prompts
    for p in complex_prompts:
        for section in ("[Verse", "[Chorus]", "[Bridge]", "[Outro]"):
            assert section in p.lyrics, f"{p.prompt_id} missing {section}"


def test_lookup_by_id_and_unknown_id():
    dataset = load_dataset(V1)
    assert dataset.by_id("KPOP-01").genre == "KPOP"
    with pytest.raises(DatasetError, match="unknown prompt_id"):
        dataset.by_id("NOPE-99")


# ── validation rules ──────────────────────────────────────────────────


def test_duplicate_prompt_ids_rejected(tmp_path):
    prompts = _full_valid_set()
    prompts.append(_prompt(prompt_id="KPOP-00", genre="KPOP"))
    with pytest.raises(DatasetError, match="duplicate prompt_id"):
        load_dataset(_write(tmp_path, prompts))


def test_vocal_prompt_without_lyrics_rejected(tmp_path):
    prompts = _full_valid_set()
    prompts[0]["lyrics"] = "   "
    with pytest.raises(DatasetError, match="no lyrics"):
        load_dataset(_write(tmp_path, prompts))


def test_instrumental_with_lyrics_rejected(tmp_path):
    prompts = _full_valid_set()
    prompts[0]["vocal_gender"] = "instrumental"
    with pytest.raises(DatasetError, match="must have empty lyrics"):
        load_dataset(_write(tmp_path, prompts))


def test_thin_genre_rejected(tmp_path):
    prompts = [p for p in _full_valid_set() if not p["prompt_id"].startswith("JAZZ-05")]
    with pytest.raises(DatasetError, match="fewer than"):
        load_dataset(_write(tmp_path, prompts))


def test_insufficient_korean_share_rejected(tmp_path):
    prompts = [dict(p, language="en") for p in _full_valid_set()]
    with pytest.raises(DatasetError, match="Korean vocal share"):
        load_dataset(_write(tmp_path, prompts))


def test_unknown_genre_rejected(tmp_path):
    prompts = _full_valid_set()
    prompts[0]["genre"] = "POLKA"
    with pytest.raises(DatasetError, match="unknown genre"):
        load_dataset(_write(tmp_path, prompts))


def test_missing_field_rejected(tmp_path):
    prompts = _full_valid_set()
    del prompts[0]["structure"]
    with pytest.raises(DatasetError, match="missing fields"):
        load_dataset(_write(tmp_path, prompts))


def test_missing_file_and_bad_json(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(DatasetError, match="not valid JSON"):
        load_dataset(bad)


def test_missing_version_rejected(tmp_path):
    path = tmp_path / "v.json"
    path.write_text(json.dumps({"prompts": _full_valid_set()}), encoding="utf-8")
    with pytest.raises(DatasetError, match="benchmark_version"):
        load_dataset(path)
