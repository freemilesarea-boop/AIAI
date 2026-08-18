"""The benchmark suite and the ingestion tool.

Two things are defended. The benchmark file is a frozen instrument, so
what is asserted is the properties that make a comparison valid — no
duplicate ids, expected lyrics preserved verbatim, no artist names, the
anti-trot probes actually asking for what they claim. And the ingestion
tool must be incapable of the two mistakes that cannot be undone:
mutating somebody's audio, and letting unestablished provenance reach a
training manifest.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import wave
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = REPO_ROOT / "benchmarks" / "music_quality" / "prompts" / "BENCHMARK_P20.json"
MANIFEST = REPO_ROOT / "benchmarks" / "music_quality" / "p20_baseline_manifest.json"
INGEST = REPO_ROOT / "scripts" / "dataset" / "ingest.py"

sys.path.insert(0, str(REPO_ROOT / "packages" / "schemas" / "src"))
from luber_schemas.dataset import DataRights, DatasetManifest, QualityTier  # noqa: E402


@pytest.fixture(scope="module")
def suite() -> dict:
    return json.loads(BENCHMARK.read_text())


def load_ingest():
    spec = importlib.util.spec_from_file_location("ingest", INGEST)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Registered before exec: @dataclass resolves annotations through
    # sys.modules, and a module missing from it fails with an opaque
    # AttributeError inside dataclasses itself.
    sys.modules["ingest"] = module
    spec.loader.exec_module(module)
    return module


def write_wav(path: Path, *, seconds: float = 0.5, seed: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(48_000 * seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(48_000)
        handle.writeframes(bytes([(i * seed) % 251 for i in range(frames * 4)]))
    return path


class TestBenchmarkIntegrity:
    def test_the_suite_is_the_size_a_listener_can_finish(self, suite):
        assert 24 <= len(suite["prompts"]) <= 36

    def test_prompt_ids_are_unique(self, suite):
        ids = [p["prompt_id"] for p in suite["prompts"]]
        assert len(ids) == len(set(ids))

    def test_every_required_set_is_represented(self, suite):
        sets = {p["set"] for p in suite["prompts"]}
        assert {"GEN", "TROT", "KO", "LONG"} <= sets

    def test_long_form_covers_the_durations_that_expose_drift(self, suite):
        durations = {p["duration"] for p in suite["prompts"]}
        assert {30, 60, 120, 180} <= durations

    def test_vocal_cases_carry_expected_lyrics(self, suite):
        """Scoring omission needs something to compare against."""
        for case in suite["prompts"]:
            if case["vocal_gender"] != "instrumental":
                assert case["lyrics"].strip(), f"{case['prompt_id']} has no expected lyrics"

    def test_instrumental_cases_carry_none(self, suite):
        for case in suite["prompts"]:
            if case["vocal_gender"] == "instrumental":
                assert case["lyrics"] == ""

    def test_the_korean_stress_set_is_large_enough_to_generalise(self, suite):
        korean = [p for p in suite["prompts"] if p["set"] == "KO"]
        assert 8 <= len(korean) <= 12
        assert all(p["language"] == "ko" for p in korean)

    def test_anti_trot_probes_actually_ask_for_restraint(self, suite):
        """A probe that does not ask cannot prove the model ignored it."""
        for case in (p for p in suite["prompts"] if p["set"] == "TROT"):
            prompt = case["prompt"].lower()
            assert "trot" in prompt, f"{case['prompt_id']} does not mention trot"
            assert any(w in prompt for w in ("no ", "minimal", "restrained")), case["prompt_id"]

    def test_no_artist_names_or_commercial_references(self, suite):
        text = BENCHMARK.read_text().lower()
        for banned in ("bts", "blackpink", "iu ", "taylor swift", "spotify", "suno"):
            assert banned not in text

    def test_the_suite_declares_that_it_scores_raw_output(self, suite):
        assert "RAW" in suite["raw_only"]


class TestBaselineManifest:
    def test_the_manifest_pins_both_codebases(self):
        manifest = json.loads(MANIFEST.read_text())
        assert len(manifest["luber"]["git_commit"]) == 40
        assert len(manifest["engine"]["git_commit"]) == 40

    def test_it_hashes_the_benchmark_so_edits_are_detectable(self):
        manifest = json.loads(MANIFEST.read_text())
        actual = hashlib.sha256(BENCHMARK.read_bytes()).hexdigest()
        assert manifest["benchmark"]["sha256"] == actual, (
            "the benchmark changed after the baseline was frozen; "
            "a new benchmark_version is required, not an edit"
        )

    def test_human_scores_are_declared_pending_not_invented(self):
        manifest = json.loads(MANIFEST.read_text())
        assert "PENDING" in manifest["human_scores"]

    def test_it_records_that_raw_output_is_the_subject(self):
        assert "RAW" in json.loads(MANIFEST.read_text())["measures"]


class TestIngestionSafety:
    def test_source_files_are_never_mutated(self, tmp_path):
        """The property that makes it safe to point at real music."""
        module = load_ingest()
        audio = write_wav(tmp_path / "a.wav")
        before = (audio.read_bytes(), audio.stat().st_mtime_ns)

        module.scan(
            tmp_path,
            rights=DataRights.OWNED,
            source_type=module.SourceType.USER_PROVIDED,
            rights_note="",
        )

        assert audio.read_bytes() == before[0]
        assert audio.stat().st_mtime_ns == before[1]

    def test_nothing_is_written_without_an_explicit_flag(self, tmp_path):
        write_wav(tmp_path / "a.wav")
        result = subprocess.run(
            [sys.executable, str(INGEST), "--dir", str(tmp_path), "--name", "t"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0
        assert "dry run" in result.stdout
        assert list(tmp_path.glob("*.json")) == []

    def test_unknown_rights_produce_nothing_trainable(self, tmp_path):
        module = load_ingest()
        write_wav(tmp_path / "a.wav")
        items, report = module.scan(
            tmp_path,
            rights=DataRights.UNKNOWN,
            source_type=module.SourceType.OTHER,
            rights_note="",
        )
        assert report.eligible == 0
        assert all(not i.is_trainable for i in items)

    def test_scanned_items_arrive_as_candidates_not_training_data(self, tmp_path):
        """Even with perfect rights, tier and split stay human decisions."""
        module = load_ingest()
        write_wav(tmp_path / "a.wav")
        items, _ = module.scan(
            tmp_path,
            rights=DataRights.OWNED,
            source_type=module.SourceType.LUBER_GENERATED,
            rights_note="ours",
        )
        assert items[0].quality_tier is QualityTier.REJECT
        assert items[0].is_trainable is False

    def test_paths_recorded_are_relative_to_the_scanned_directory(self, tmp_path):
        module = load_ingest()
        write_wav(tmp_path / "nested" / "deep" / "a.wav")
        items, _ = module.scan(
            tmp_path,
            rights=DataRights.UNKNOWN,
            source_type=module.SourceType.OTHER,
            rights_note="",
        )
        assert items[0].audio_path == "nested/deep/a.wav"
        assert str(tmp_path) not in items[0].audio_path

    def test_an_identical_file_under_two_names_is_reported(self, tmp_path):
        module = load_ingest()
        write_wav(tmp_path / "a.wav", seed=3)
        (tmp_path / "copy.wav").write_bytes((tmp_path / "a.wav").read_bytes())
        _, report = module.scan(
            tmp_path,
            rights=DataRights.UNKNOWN,
            source_type=module.SourceType.OTHER,
            rights_note="",
        )
        assert len(report.duplicates) == 1

    def test_a_file_that_will_not_decode_is_reported_not_skipped(self, tmp_path):
        module = load_ingest()
        (tmp_path / "broken.wav").write_bytes(b"this is not audio")
        _, report = module.scan(
            tmp_path,
            rights=DataRights.UNKNOWN,
            source_type=module.SourceType.OTHER,
            rights_note="",
        )
        assert report.discovered == 1
        assert len(report.invalid) == 1
        assert report.valid == 0

    def test_a_written_manifest_validates_and_holds_no_absolute_path(self, tmp_path):
        write_wav(tmp_path / "a.wav")
        out = tmp_path / "out" / "manifest.json"
        result = subprocess.run(
            [
                sys.executable,
                str(INGEST),
                "--dir",
                str(tmp_path),
                "--name",
                "t",
                "--rights",
                "OWNED",
                "--source-type",
                "LUBER_GENERATED",
                "--write",
                str(out),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        manifest = DatasetManifest.model_validate_json(out.read_text())
        assert manifest.items
        assert all(not i.audio_path.startswith("/") for i in manifest.items)
        assert "/Users/" not in out.read_text()

    def test_the_tool_states_that_near_duplicates_are_not_detected(self, tmp_path):
        """It must not imply a guarantee it does not implement."""
        write_wav(tmp_path / "a.wav")
        result = subprocess.run(
            [sys.executable, str(INGEST), "--dir", str(tmp_path), "--name", "t"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert "NOT implemented" in result.stdout

    def test_a_directory_must_be_named_explicitly(self):
        """No default path: it cannot sweep a home folder by accident."""
        result = subprocess.run(
            [sys.executable, str(INGEST), "--name", "t"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode != 0
        assert "--dir" in result.stderr
