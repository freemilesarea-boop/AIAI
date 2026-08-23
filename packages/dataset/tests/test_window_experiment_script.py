"""The Phase 37 build script: split tracks, then window, then gate.

It writes the manifests the trainer is handed, so what matters is that a
contaminated or multi-shape result never reaches disk in the first
place.
"""

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "dataset" / "build_window_experiment.py"


def _load():
    spec = importlib.util.spec_from_file_location("build_window_experiment_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load()


def _library(tmp_path: Path, count: int = 128) -> Path:
    tracks, sources = [], {}
    for index in range(1, count + 1):
        track_id = f"track-{index:03d}"
        tracks.append(
            {
                "track_id": track_id,
                "audio_sha256": f"{index:064x}",
                "source_group": "POP" if index % 2 else "Lofi",
                # A spread that produces one-, two- and multi-window
                # tracks, plus a few too short for any window at all.
                "duration_seconds": (95.0 if index % 20 == 0 else 130.0 + (index % 11) * 28.0),
                "training_allowed": True,
            }
        )
        sources[track_id] = f"/src/{track_id}.wav"
    path = tmp_path / "library.json"
    path.write_text(
        json.dumps({"dataset_id": "LIB", "content_hash": "c" * 64, "tracks": tracks}),
        encoding="utf-8",
    )
    path.with_suffix(".paths.json").write_text(
        json.dumps({"note": "local", "tracks": sources}), encoding="utf-8"
    )
    return path


def _run(manifest: Path, out: Path, **sizes) -> int:
    return builder.main(
        [
            "--manifest",
            str(manifest),
            "--train",
            str(sizes.get("train", 64)),
            "--validation",
            str(sizes.get("validation", 8)),
            "--evaluation",
            str(sizes.get("evaluation", 8)),
            "--seed",
            "37",
            "--output",
            str(out),
        ]
    )


class TestWhatItWrites:
    def test_it_writes_a_manifest_for_every_split(self, tmp_path, capsys):
        out = tmp_path / "exp"
        assert _run(_library(tmp_path), out) == 0
        capsys.readouterr()
        for name in (
            "splits.json",
            "windows_train.json",
            "windows_validation.json",
            "windows_evaluation.json",
            "reserve.json",
        ):
            assert (out / name).is_file(), name

    def test_the_track_counts_are_the_ones_requested(self, tmp_path, capsys):
        out = tmp_path / "exp"
        assert _run(_library(tmp_path), out) == 0
        capsys.readouterr()
        splits = json.loads((out / "splits.json").read_text(encoding="utf-8"))
        assert splits["train"]["track_count"] == 64
        assert splits["validation"]["track_count"] == 8
        assert splits["evaluation"]["track_count"] == 8

    def test_long_tracks_contribute_more_than_one_window(self, tmp_path, capsys):
        out = tmp_path / "exp"
        assert _run(_library(tmp_path), out) == 0
        capsys.readouterr()
        train = json.loads((out / "windows_train.json").read_text(encoding="utf-8"))
        assert train["window_count"] > train["track_count"]
        assert train["maximum_windows_per_track"] > 1

    def test_every_window_across_every_split_is_one_shape(self, tmp_path, capsys):
        out = tmp_path / "exp"
        assert _run(_library(tmp_path), out) == 0
        capsys.readouterr()
        shapes = set()
        for name in ("train", "validation", "evaluation"):
            payload = json.loads((out / f"windows_{name}.json").read_text(encoding="utf-8"))
            shapes.update(payload["unique_latent_frames"])
        assert shapes == {3000}
        assert json.loads((out / "splits.json").read_text(encoding="utf-8"))[
            "unique_latent_shapes"
        ] == [3000]

    def test_sampling_weights_equalise_track_influence(self, tmp_path, capsys):
        out = tmp_path / "exp"
        assert _run(_library(tmp_path), out) == 0
        capsys.readouterr()
        train = json.loads((out / "windows_train.json").read_text(encoding="utf-8"))
        weights = train["sampling_weights"]
        totals: dict[str, float] = {}
        for window in train["windows"]:
            totals[window["track_id"]] = (
                totals.get(window["track_id"], 0.0) + weights[window["window_id"]]
            )
        assert all(abs(v - 1.0) < 1e-9 for v in totals.values())


class TestItReservesWhatItDoesNotUse:
    def test_unused_authorised_tracks_are_recorded_and_untouched(self, tmp_path, capsys):
        out = tmp_path / "exp"
        assert _run(_library(tmp_path), out) == 0
        capsys.readouterr()
        reserve = json.loads((out / "reserve.json").read_text(encoding="utf-8"))
        splits = json.loads((out / "splits.json").read_text(encoding="utf-8"))
        used = {
            m["track_id"]
            for key in ("train", "validation", "evaluation")
            for m in splits[key]["tracks"]
        }
        assert reserve["reserved_track_count"] == 128 - len(used)
        assert not (set(reserve["track_ids"]) & used)


class TestItRefusesBadInput:
    def test_short_tracks_are_reported_not_silently_dropped(self, tmp_path, capsys):
        assert _run(_library(tmp_path), tmp_path / "exp") == 0
        printed = capsys.readouterr().out
        assert "too short" in printed

    def test_it_is_reproducible(self, tmp_path, capsys):
        manifest = _library(tmp_path)
        first, second = tmp_path / "a", tmp_path / "b"
        assert _run(manifest, first) == 0
        assert _run(manifest, second) == 0
        capsys.readouterr()
        for name in ("windows_train.json", "windows_validation.json", "windows_evaluation.json"):
            a = json.loads((first / name).read_text(encoding="utf-8"))
            b = json.loads((second / name).read_text(encoding="utf-8"))
            assert a["manifest_digest"] == b["manifest_digest"]

    def test_no_window_of_one_track_reaches_two_splits(self, tmp_path, capsys):
        out = tmp_path / "exp"
        assert _run(_library(tmp_path), out) == 0
        capsys.readouterr()
        placement: dict[str, set[str]] = {}
        for name in ("train", "validation", "evaluation"):
            payload = json.loads((out / f"windows_{name}.json").read_text(encoding="utf-8"))
            for window in payload["windows"]:
                placement.setdefault(window["track_id"], set()).add(name)
        assert all(len(names) == 1 for names in placement.values())
