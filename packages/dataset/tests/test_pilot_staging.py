"""Staging a pilot subset out of a read-only authorised root.

The source material is the one thing in this pipeline that cannot be
regenerated, so the staging step copies rather than moves and proves
afterwards that it did. These tests hold that promise, and they hold
the other one staging exists for: the operator sidecars the factory
needs are written next to the copies, never into the source.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "dataset" / "stage_pilot_subset.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("stage_pilot_subset_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stage = _load_script()

SCOPE = "/authorized/root/**"


def _library(tmp_path: Path, count: int = 4) -> tuple[Path, Path, list[Path]]:
    """A manifest, a path map and the source files they describe."""
    source_root = tmp_path / "source"
    tracks, sources, paths = [], {}, []
    for index in range(count):
        group = "POP" if index % 2 else "Lofi"
        payload = bytes([index]) * (4096 + index)
        origin = source_root / group / f"track{index}.wav"
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
                "duration_seconds": 120.0 + index,
                "training_allowed": True,
            }
        )
        sources[track_id] = str(origin)

    manifest = tmp_path / "library.json"
    manifest.write_text(
        json.dumps({"dataset_id": "LIB", "content_hash": "a" * 64, "tracks": tracks}),
        encoding="utf-8",
    )
    manifest.with_suffix(".paths.json").write_text(
        json.dumps({"note": "local", "authorization_scope": SCOPE, "tracks": sources}),
        encoding="utf-8",
    )
    return manifest, source_root, paths


def _run(manifest: Path, output: Path, size: int = 3) -> int:
    return stage.main(["--manifest", str(manifest), "--size", str(size), "--output", str(output)])


class TestItCopiesAndLeavesTheSourceAlone:
    def test_the_source_files_are_byte_identical_afterwards(self, tmp_path, capsys):
        manifest, _, paths = _library(tmp_path)
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}

        assert _run(manifest, tmp_path / "staged") == 0
        capsys.readouterr()

        for path, (payload, mtime) in before.items():
            assert path.read_bytes() == payload
            assert path.stat().st_mtime_ns == mtime

    def test_no_sidecar_is_written_into_the_source_tree(self, tmp_path, capsys):
        manifest, source_root, _ = _library(tmp_path)
        assert _run(manifest, tmp_path / "staged") == 0
        capsys.readouterr()
        assert [p.suffix for p in source_root.rglob("*") if p.is_file()] == [".wav"] * 4

    def test_the_staged_copy_matches_the_manifest_digest(self, tmp_path, capsys):
        manifest, _, _ = _library(tmp_path)
        staged = tmp_path / "staged"
        assert _run(manifest, staged) == 0
        capsys.readouterr()

        subset = json.loads((staged / "pilot_subset.json").read_text(encoding="utf-8"))
        for track in subset["tracks"]:
            copy = staged / f"{track['track_id']}.wav"
            assert hashlib.sha256(copy.read_bytes()).hexdigest() == track["audio_sha256"]
        assert subset["source_unchanged"] is True


class TestItRefusesAChangedSource:
    def test_a_source_that_no_longer_hashes_to_the_manifest_aborts(self, tmp_path, capsys):
        """A subset is only the locked subset while the bytes still are."""
        manifest, _, paths = _library(tmp_path)
        paths[0].write_bytes(b"different audio entirely")

        assert _run(manifest, tmp_path / "staged") == 2
        assert "no longer hashes" in capsys.readouterr().err

    def test_a_missing_source_aborts_rather_than_staging_fewer_tracks(self, tmp_path, capsys):
        manifest, _, paths = _library(tmp_path)
        paths[0].unlink()

        assert _run(manifest, tmp_path / "staged") == 2
        assert "no longer at its recorded path" in capsys.readouterr().err


class TestTheSidecarSaysWhatItHas:
    def test_it_records_operator_authorisation_and_not_ownership(self, tmp_path, capsys):
        manifest, _, _ = _library(tmp_path)
        staged = tmp_path / "staged"
        assert _run(manifest, staged) == 0
        capsys.readouterr()

        sidecars = sorted(staged.glob("*.json"))
        assert sidecars, "expected a sidecar beside each staged copy"
        for path in sidecars:
            if path.name == "pilot_subset.json":
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["rights_status"] == "OPERATOR_AUTHORIZED"
            assert payload["source_type"] == "UNKNOWN"
            assert SCOPE in payload["notes"]
            assert "none is claimed here" in payload["notes"]

    def test_the_operator_folder_label_becomes_a_genre_not_a_rights_claim(self, tmp_path, capsys):
        manifest, _, _ = _library(tmp_path)
        staged = tmp_path / "staged"
        assert _run(manifest, staged) == 0
        capsys.readouterr()

        genres = {
            json.loads(p.read_text(encoding="utf-8"))["genre"]
            for p in staged.glob("*.json")
            if p.name != "pilot_subset.json"
        }
        assert genres <= {"pop", "lofi"}


class TestItIsRepeatable:
    def test_staging_twice_selects_the_same_tracks(self, tmp_path, capsys):
        manifest, _, _ = _library(tmp_path)
        first_dir, second_dir = tmp_path / "one", tmp_path / "two"
        assert _run(manifest, first_dir) == 0
        assert _run(manifest, second_dir) == 0
        capsys.readouterr()

        first = json.loads((first_dir / "pilot_subset.json").read_text(encoding="utf-8"))
        second = json.loads((second_dir / "pilot_subset.json").read_text(encoding="utf-8"))
        assert first["subset_digest"] == second["subset_digest"]
        assert first["tracks"] == second["tracks"]

    @pytest.mark.parametrize("size", [3, 4])
    def test_it_stages_exactly_the_requested_number(self, tmp_path, capsys, size):
        manifest, _, _ = _library(tmp_path)
        staged = tmp_path / f"staged-{size}"
        assert _run(manifest, staged, size=size) == 0
        capsys.readouterr()
        assert len(list(staged.glob("*.wav"))) == size
