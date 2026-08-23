"""Staging the three splits, and leaving the source exactly as found.

The source root is the one thing here that cannot be regenerated. These
tests hold the promise the script makes about it, and the other promise
staging exists for: three directories, so a training run pointed at one
of them cannot reach an evaluation track by mistake.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "dataset" / "stage_experiment_splits.py"


def _load():
    spec = importlib.util.spec_from_file_location("stage_experiment_splits_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stage = _load()


def _fixture(tmp_path: Path, counts=(3, 2, 2)):
    source = tmp_path / "source"
    payload_splits, sources, paths = {}, {}, []
    index = 0
    for key, name, count in zip(
        ("train", "validation", "evaluation"),
        ("TRAIN", "VALIDATION", "EVALUATION"),
        counts,
        strict=True,
    ):
        tracks = []
        for _ in range(count):
            index += 1
            group = "POP" if index % 2 else "Lofi"
            payload = bytes([index]) * (2048 + index)
            origin = source / group / f"t{index}.wav"
            origin.parent.mkdir(parents=True, exist_ok=True)
            origin.write_bytes(payload)
            paths.append(origin)
            digest = hashlib.sha256(payload).hexdigest()
            track_id = digest[:16]
            tracks.append(
                {
                    "track_id": track_id,
                    "audio_sha256": digest,
                    "source_group": group,
                    "duration_seconds": 120.0,
                }
            )
            sources[track_id] = str(origin)
        payload_splits[key] = {"name": name, "track_count": count, "tracks": tracks}

    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"splits_digest": "d" * 64, **payload_splits}), encoding="utf-8")
    splits.with_suffix(".paths.json").write_text(
        json.dumps({"note": "local", "tracks": sources}), encoding="utf-8"
    )
    return splits, source, paths


def _run(splits: Path, output: Path) -> int:
    return stage.main(
        ["--splits", str(splits), "--output", str(output), "--scope", "/authorized/**"]
    )


class TestItSeparatesTheSplitsOnDisk:
    def test_each_split_gets_its_own_directory(self, tmp_path, capsys):
        splits, _, _ = _fixture(tmp_path)
        assert _run(splits, tmp_path / "staged") == 0
        capsys.readouterr()
        for key, count in (("train", 3), ("validation", 2), ("evaluation", 2)):
            assert len(list((tmp_path / "staged" / key).glob("*.wav"))) == count

    def test_no_evaluation_track_appears_in_the_training_directory(self, tmp_path, capsys):
        splits, _, _ = _fixture(tmp_path)
        assert _run(splits, tmp_path / "staged") == 0
        capsys.readouterr()
        payload = json.loads(splits.read_text(encoding="utf-8"))
        held = {t["track_id"] for t in payload["evaluation"]["tracks"]}
        staged = {p.stem for p in (tmp_path / "staged" / "train").glob("*.wav")}
        assert not (held & staged)


class TestItLeavesTheSourceAlone:
    def test_every_source_file_is_byte_identical_afterwards(self, tmp_path, capsys):
        splits, _, paths = _fixture(tmp_path)
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
        assert _run(splits, tmp_path / "staged") == 0
        capsys.readouterr()
        for path, (payload, mtime) in before.items():
            assert path.read_bytes() == payload
            assert path.stat().st_mtime_ns == mtime

    def test_no_sidecar_is_written_into_the_source_tree(self, tmp_path, capsys):
        splits, source, _ = _fixture(tmp_path)
        assert _run(splits, tmp_path / "staged") == 0
        capsys.readouterr()
        assert {p.suffix for p in source.rglob("*") if p.is_file()} == {".wav"}

    def test_the_report_records_that_the_source_was_unchanged(self, tmp_path, capsys):
        splits, _, _ = _fixture(tmp_path)
        assert _run(splits, tmp_path / "staged") == 0
        capsys.readouterr()
        report = json.loads(
            (tmp_path / "staged" / "staging_report.json").read_text(encoding="utf-8")
        )
        assert report["source_unchanged"] is True
        assert report["staged_counts"] == {"train": 3, "validation": 2, "evaluation": 2}


class TestItRefusesAChangedSource:
    def test_a_source_that_no_longer_matches_the_split_aborts(self, tmp_path, capsys):
        splits, _, paths = _fixture(tmp_path)
        paths[0].write_bytes(b"different audio entirely")
        assert _run(splits, tmp_path / "staged") == 2
        assert "no longer hashes" in capsys.readouterr().err

    def test_a_missing_source_aborts_rather_than_staging_fewer(self, tmp_path, capsys):
        splits, _, paths = _fixture(tmp_path)
        paths[0].unlink()
        assert _run(splits, tmp_path / "staged") == 2
        assert "no longer at its recorded path" in capsys.readouterr().err


class TestTheSidecars:
    def test_they_record_operator_authorisation_and_the_split(self, tmp_path, capsys):
        splits, _, _ = _fixture(tmp_path)
        assert _run(splits, tmp_path / "staged") == 0
        capsys.readouterr()
        for path in (tmp_path / "staged" / "evaluation").glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["rights_status"] == "OPERATOR_AUTHORIZED"
            assert "EVALUATION" in payload["notes"]
            assert "none is claimed here" in payload["notes"]
