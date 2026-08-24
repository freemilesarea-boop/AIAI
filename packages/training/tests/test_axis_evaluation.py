"""The four-axis listening set: one question per axis, four models each.

HIGH-END, RHYTHM, ARRANGEMENT and VOCAL are the four things the operator
is asked about, so the set has to give each one something to bite on
while keeping the prompt identical across every model — otherwise the
comparison measures prompts rather than adapters.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "training" / "build_axis_eval_spec.py"


def _load():
    spec = importlib.util.spec_from_file_location("build_axis_eval_spec_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load()


def _splits() -> dict:
    def track(index: int) -> dict:
        return {
            "track_id": f"track-{index:03d}",
            "audio_sha256": f"{index:064x}",
            "source_group": "POP",
            "duration_seconds": 200.0,
        }

    return {
        "train": {"name": "TRAIN", "tracks": [track(i) for i in range(1, 51)]},
        "validation": {"name": "VALIDATION", "tracks": [track(i) for i in range(51, 59)]},
        "evaluation": {"name": "EVALUATION", "tracks": [track(i) for i in range(59, 67)]},
    }


class TestTheAxes:
    def test_all_four_axes_are_represented(self):
        items = builder.build_items(_splits(), duration=30.0)
        assert {item["axis"] for item in items} == {
            "HIGH_END",
            "RHYTHM",
            "ARRANGEMENT",
            "VOCAL",
        }

    def test_each_axis_prompt_mentions_what_it_asks_about(self):
        assert "top end" in builder.AXIS_PROMPTS["HIGH_END"]
        assert "groove" in builder.AXIS_PROMPTS["RHYTHM"]
        assert "layered" in builder.AXIS_PROMPTS["ARRANGEMENT"]
        assert "vocal" in builder.AXIS_PROMPTS["VOCAL"]

    def test_the_axes_are_spread_evenly_over_the_evaluation_tracks(self):
        items = builder.build_items(_splits(), duration=30.0)
        counts: dict[str, int] = {}
        for item in items:
            counts[item["axis"]] = counts.get(item["axis"], 0) + 1
        assert max(counts.values()) - min(counts.values()) <= 1

    def test_the_vocal_axis_asks_for_vocals_not_an_instrumental(self):
        items = [i for i in builder.build_items(_splits(), duration=30.0) if i["axis"] == "VOCAL"]
        assert items
        for item in items:
            assert item["instrumental"] is False


class TestIsolation:
    def test_no_item_derives_from_a_training_track(self):
        splits = _splits()
        items = builder.build_items(splits, duration=30.0)
        training = {t["track_id"] for t in splits["train"]["tracks"]}
        assert not ({item["source_track"] for item in items} & training)

    def test_no_item_derives_from_a_validation_track(self):
        splits = _splits()
        items = builder.build_items(splits, duration=30.0)
        held = {t["track_id"] for t in splits["validation"]["tracks"]}
        assert not ({item["source_track"] for item in items} & held)

    def test_every_evaluation_track_contributes(self):
        splits = _splits()
        items = builder.build_items(splits, duration=30.0)
        assert {item["source_track"] for item in items} == {
            t["track_id"] for t in splits["evaluation"]["tracks"]
        }


class TestDeterminism:
    def test_the_same_split_gives_the_same_set(self):
        assert builder.build_items(_splits(), duration=30.0) == builder.build_items(
            _splits(), duration=30.0
        )

    def test_seeds_come_from_the_audio_digest(self):
        digest = "ab" * 32
        assert builder.seed_for(digest, 0) == builder.seed_for(digest, 0)
        assert builder.seed_for(digest, 0) != builder.seed_for("cd" * 32, 0)

    @pytest.mark.parametrize("digest", ["f" * 64, "0" * 64, "a1b2c3d4" * 8])
    def test_seeds_stay_inside_a_signed_integer(self, digest):
        assert 0 <= builder.seed_for(digest, 0) <= builder.SEED_MASK

    def test_the_spec_round_trips_through_json(self):
        items = builder.build_items(_splits(), duration=30.0)
        assert json.loads(json.dumps(items)) == items


class TestFourWayComparison:
    def test_a_four_sided_spec_names_base_and_three_adapters(self, tmp_path):
        out = tmp_path / "spec.json"
        splits_path = tmp_path / "splits.json"
        splits_path.write_text(json.dumps(_splits()), encoding="utf-8")
        code = builder.main(
            [
                "--splits",
                str(splits_path),
                "--project-root",
                "/trainer",
                "--output-root",
                str(tmp_path / "out"),
                "--manifest",
                str(tmp_path / "out" / "manifest.json"),
                "--out",
                str(out),
                "--side",
                "base=",
                "--side",
                "phase36=/adapters/p36",
                "--side",
                "phase37=/adapters/p37",
                "--side",
                "phase38=/adapters/p38",
            ]
        )
        assert code == 0
        spec = json.loads(out.read_text(encoding="utf-8"))
        assert [s["name"] for s in spec["sides"]] == [
            "base",
            "phase36",
            "phase37",
            "phase38",
        ]
        assert spec["sides"][0]["lora_path"] is None
        adapters = [s["lora_path"] for s in spec["sides"][1:]]
        assert len(set(adapters)) == 3

    def test_every_side_shares_one_prompt_and_one_seed_per_item(self, tmp_path):
        out = tmp_path / "spec.json"
        splits_path = tmp_path / "splits.json"
        splits_path.write_text(json.dumps(_splits()), encoding="utf-8")
        builder.main(
            [
                "--splits",
                str(splits_path),
                "--project-root",
                "/trainer",
                "--output-root",
                str(tmp_path / "out"),
                "--manifest",
                str(tmp_path / "out" / "manifest.json"),
                "--out",
                str(out),
                "--side",
                "base=",
                "--side",
                "phase38=/adapters/p38",
            ]
        )
        spec = json.loads(out.read_text(encoding="utf-8"))
        # The question lives on the item; a side carries only its adapter.
        for side in spec["sides"]:
            assert set(side) == {"name", "lora_path"}
        for item in spec["items"]:
            assert "caption" in item and "seed" in item
