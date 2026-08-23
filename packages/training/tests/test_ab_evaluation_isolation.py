"""The A/B listening set, and where its questions come from.

An evaluation whose prompts came from the training split measures
recall. The property under test is that they cannot: the spec builder
reads the evaluation split and nothing else, and the same split always
produces the same set.
"""

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "training" / "build_ab_spec.py"


def _load():
    spec = importlib.util.spec_from_file_location("build_ab_spec_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load()


def _splits() -> dict:
    def track(index: int, group: str) -> dict:
        return {
            "track_id": f"track-{index:03d}",
            "audio_sha256": f"{index:064x}",
            "source_group": group,
            "duration_seconds": 120.0,
        }

    return {
        "train": {"name": "TRAIN", "tracks": [track(i, "A") for i in range(1, 25)]},
        "validation": {"name": "VALIDATION", "tracks": [track(i, "A") for i in range(25, 29)]},
        "evaluation": {
            "name": "EVALUATION",
            "tracks": [track(29, "A"), track(30, "A"), track(31, "B"), track(32, "B")],
        },
    }


class TestItDrawsOnlyFromTheEvaluationSplit:
    def test_no_pair_comes_from_a_training_track(self):
        splits = _splits()
        spec = builder.build_spec(splits)
        training_ids = {t["track_id"] for t in splits["train"]["tracks"]}
        for pair in spec["pairs"]:
            assert pair["id"].rsplit("-", 1)[0] not in training_ids

    def test_no_pair_comes_from_a_validation_track(self):
        splits = _splits()
        spec = builder.build_spec(splits)
        validation_ids = {t["track_id"] for t in splits["validation"]["tracks"]}
        for pair in spec["pairs"]:
            assert pair["id"].rsplit("-", 1)[0] not in validation_ids

    def test_every_evaluation_track_is_represented(self):
        splits = _splits()
        spec = builder.build_spec(splits)
        represented = {pair["id"].rsplit("-", 1)[0] for pair in spec["pairs"]}
        assert represented == {t["track_id"] for t in splits["evaluation"]["tracks"]}

    def test_an_empty_evaluation_split_produces_no_pairs(self):
        splits = _splits()
        splits["evaluation"]["tracks"] = []
        assert builder.build_spec(splits)["pairs"] == []


class TestItIsDeterministic:
    def test_the_same_split_gives_the_same_set(self):
        first = builder.build_spec(_splits())
        second = builder.build_spec(_splits())
        assert first["pairs"] == second["pairs"]

    def test_seeds_come_from_the_audio_digest(self):
        digest = "ab" * 32
        assert builder.seeds_for(digest) == builder.seeds_for(digest)
        assert builder.seeds_for(digest) != builder.seeds_for("cd" * 32)

    def test_seeds_stay_inside_a_signed_integer(self):
        for seed in builder.seeds_for("f" * 64):
            assert 0 <= seed <= 0x7FFFFFFF

    def test_the_set_is_the_size_the_phase_asked_for(self):
        pairs = builder.build_spec(_splits())["pairs"]
        assert 5 <= len(pairs) <= 10


class TestBothSidesAreAskedTheSameQuestion:
    def test_a_pair_carries_one_caption_and_one_seed_for_both_sides(self):
        for pair in builder.build_spec(_splits())["pairs"]:
            assert set(pair) == {"id", "caption", "seed", "source_group"}

    def test_the_caption_is_the_operator_group_label(self):
        pairs = builder.build_spec(_splits())["pairs"]
        assert {pair["caption"] for pair in pairs} == {"a", "b"}

    def test_the_spec_round_trips_through_json(self):
        """The generator writes it; the trainer's interpreter reads it."""
        spec = builder.build_spec(_splits(), lora_path="/adapters/x")
        assert json.loads(json.dumps(spec)) == spec
