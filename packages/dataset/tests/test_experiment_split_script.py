"""The split-building script, end to end.

It writes the file the trainer will be gated against, so the thing worth
testing is that it refuses to write a bad one — and that what it writes
is what the gate reads.
"""

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "dataset" / "build_experiment_splits.py"


def _load():
    spec = importlib.util.spec_from_file_location("build_experiment_splits_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load()


def _library(tmp_path: Path, count: int = 128) -> Path:
    tracks = []
    for index in range(1, count + 1):
        tracks.append(
            {
                "track_id": f"track-{index:03d}",
                "audio_sha256": f"{index:064x}",
                "source_group": "A" if index % 2 else "B",
                "duration_seconds": 100.0 + index,
                "training_allowed": True,
            }
        )
    path = tmp_path / "library.json"
    path.write_text(
        json.dumps({"dataset_id": "LIB", "content_hash": "c" * 64, "tracks": tracks}),
        encoding="utf-8",
    )
    path.with_suffix(".paths.json").write_text(
        json.dumps(
            {
                "note": "local",
                "tracks": {t["track_id"]: f"/src/{t['track_id']}.wav" for t in tracks},
            }
        ),
        encoding="utf-8",
    )
    return path


def _run(manifest: Path, out: Path, *extra: str) -> int:
    return builder.main(
        [
            "--manifest",
            str(manifest),
            "--train",
            "24",
            "--validation",
            "4",
            "--evaluation",
            "4",
            "--seed",
            "36",
            "--output",
            str(out),
            *extra,
        ]
    )


class TestItWritesAGatedSplit:
    def test_it_writes_the_requested_sizes(self, tmp_path, capsys):
        out = tmp_path / "splits.json"
        assert _run(_library(tmp_path), out) == 0
        capsys.readouterr()
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["train"]["track_count"] == 24
        assert payload["validation"]["track_count"] == 4
        assert payload["evaluation"]["track_count"] == 4

    def test_the_gate_passes_on_what_was_written(self, tmp_path, capsys):
        from luber_training.gates import split_leakage_gate

        out = tmp_path / "splits.json"
        assert _run(_library(tmp_path), out) == 0
        capsys.readouterr()
        assert split_leakage_gate(json.loads(out.read_text(encoding="utf-8"))).passed

    def test_it_reports_both_the_report_and_the_gate(self, tmp_path, capsys):
        """Two checks that must agree, and both are printed."""
        assert _run(_library(tmp_path), tmp_path / "splits.json") == 0
        printed = capsys.readouterr().out
        assert "leakage report: PASS" in printed
        assert "leakage gate  : PASS" in printed

    def test_the_path_map_is_local_only_and_the_split_holds_no_paths(self, tmp_path, capsys):
        out = tmp_path / "splits.json"
        assert _run(_library(tmp_path), out) == 0
        capsys.readouterr()
        assert "/src/" not in out.read_text(encoding="utf-8")
        side = json.loads(out.with_suffix(".paths.json").read_text(encoding="utf-8"))
        assert "never commit" in side["note"]
        assert len(side["tracks"]) == 32

    def test_it_is_reproducible(self, tmp_path, capsys):
        manifest = _library(tmp_path)
        first, second = tmp_path / "a.json", tmp_path / "b.json"
        assert _run(manifest, first) == 0
        assert _run(manifest, second) == 0
        capsys.readouterr()
        assert (
            json.loads(first.read_text(encoding="utf-8"))["splits_digest"]
            == json.loads(second.read_text(encoding="utf-8"))["splits_digest"]
        )


class TestItRefusesAnImpossibleRequest:
    def test_too_few_candidates_raises_rather_than_writing_a_short_split(self, tmp_path, capsys):
        import pytest

        from luber_dataset.splits import SplitError

        out = tmp_path / "splits.json"
        with pytest.raises(SplitError):
            _run(_library(tmp_path, count=10), out)
        assert not out.exists()
