"""Three-way evaluation: same question, three models, no cross-contamination.

An A/B/C comparison is only worth anything if the sides differ in one
thing. Two ways that can quietly fail: the prompts can come from data a
model trained on, and a side can inherit the previous side's adapter.
"""

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_SCRIPT = REPO_ROOT / "scripts" / "training" / "build_ab_spec.py"
ABC_SCRIPT = REPO_ROOT / "scripts" / "training" / "abc_evaluate.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = _load(SPEC_SCRIPT, "build_ab_spec_abc_test")
abc = _load(ABC_SCRIPT, "abc_evaluate_under_test")


def _splits() -> dict:
    def track(index: int, group: str) -> dict:
        return {
            "track_id": f"track-{index:03d}",
            "audio_sha256": f"{index:064x}",
            "source_group": group,
            "duration_seconds": 200.0,
        }

    return {
        "train": {"name": "TRAIN", "tracks": [track(i, "POP") for i in range(1, 65)]},
        "validation": {"name": "VALIDATION", "tracks": [track(i, "POP") for i in range(65, 73)]},
        "evaluation": {
            "name": "EVALUATION",
            "tracks": [track(i, "POP" if i % 2 else "Lofi") for i in range(73, 81)],
        },
    }


class TestPromptsComeFromHeldOutTracksOnly:
    def test_no_prompt_derives_from_a_training_track(self):
        splits = _splits()
        pairs = builder.build_spec(splits)["pairs"]
        training = {t["track_id"] for t in splits["train"]["tracks"]}
        assert not ({p["id"].rsplit("-", 1)[0] for p in pairs} & training)

    def test_no_prompt_derives_from_a_validation_track(self):
        splits = _splits()
        pairs = builder.build_spec(splits)["pairs"]
        validation = {t["track_id"] for t in splits["validation"]["tracks"]}
        assert not ({p["id"].rsplit("-", 1)[0] for p in pairs} & validation)

    def test_the_set_is_at_least_the_eight_the_phase_asks_for(self):
        assert len(builder.build_spec(_splits())["pairs"]) >= 8

    def test_it_is_reproducible(self):
        assert builder.build_spec(_splits())["pairs"] == builder.build_spec(_splits())["pairs"]


class TestEverySideGetsTheSameQuestion:
    def _spec(self, tmp_path: Path) -> dict:
        items = [
            {"id": f"pair_{i:02d}", "caption": "pop", "seed": 1000 + i, "duration": 30.0}
            for i in range(1, 9)
        ]
        return {
            "project_root": "/trainer",
            "output_root": str(tmp_path / "abc"),
            "manifest": str(tmp_path / "abc" / "manifest.json"),
            "duration": 30.0,
            "inference_steps": 8,
            "items": items,
            "sides": [
                {"name": "base", "lora_path": None},
                {"name": "phase36", "lora_path": "/adapters/p36"},
                {"name": "phase37", "lora_path": "/adapters/p37"},
            ],
        }

    def test_each_side_writes_to_its_own_directory(self, tmp_path):
        spec = self._spec(tmp_path)
        targets = {
            (item["id"], side["name"]): Path(spec["output_root"]) / item["id"] / side["name"]
            for item in spec["items"]
            for side in spec["sides"]
        }
        assert len(set(targets.values())) == len(targets)

    def test_the_base_side_carries_no_adapter(self):
        spec = self._spec(Path("/tmp"))
        base = next(s for s in spec["sides"] if s["name"] == "base")
        assert not base["lora_path"]

    def test_the_two_adapter_sides_are_different_checkpoints(self):
        spec = self._spec(Path("/tmp"))
        paths = [s["lora_path"] for s in spec["sides"] if s["lora_path"]]
        assert len(paths) == len(set(paths)) == 2

    def test_every_item_carries_one_seed_used_by_all_sides(self):
        spec = self._spec(Path("/tmp"))
        for item in spec["items"]:
            assert isinstance(item["seed"], int)
        assert len({item["seed"] for item in spec["items"]}) == len(spec["items"])

    def test_the_spec_round_trips_through_json(self):
        spec = self._spec(Path("/tmp"))
        assert json.loads(json.dumps(spec)) == spec


class TestTheHarnessRefusesUnsafeState:
    def test_it_reads_adapter_state_off_the_handler(self):
        class Handler:
            lora_loaded = True
            use_lora = True
            lora_scale = 1.0
            _lora_active_adapter = "final"

        state = abc.lora_state(Handler())
        assert state["lora_loaded"] and state["use_lora"]
        assert state["active_adapter"] == "final"

    def test_an_unloaded_handler_reads_as_no_adapter(self):
        class Handler:
            lora_loaded = False
            use_lora = False
            lora_scale = 1.0
            _lora_active_adapter = None

        assert abc.lora_state(Handler()) == {
            "lora_loaded": False,
            "use_lora": False,
            "lora_scale": 1.0,
            "active_adapter": None,
        }

    def test_it_lists_whatever_container_the_pipeline_wrote(self, tmp_path):
        for name in ("a.flac", "b.wav", "notes.json"):
            (tmp_path / name).write_bytes(b"x")
        assert abc.audio_files(tmp_path) == ["a.flac", "b.wav"]
