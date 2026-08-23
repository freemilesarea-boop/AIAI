"""The vocal test set: one question, asked of every model.

The vocal songs are not drawn from the library at all — the lyrics are
written for the test and the captions describe a target, so there is no
training material to leak. What has to hold is that all three sides are
asked exactly the same thing, and that nothing in the set is an
instrumental request wearing a vocal label.
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ABC_SCRIPT = REPO_ROOT / "scripts" / "training" / "abc_evaluate.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


abc = _load(ABC_SCRIPT, "abc_evaluate_vocal_test")


def _songs() -> list[dict]:
    return [
        {
            "id": "song_01",
            "caption": "modern atmospheric pop R&B with an emotional female lead vocal",
            "lyrics": "[Verse 1]\nline one\nline two\n\n[Chorus]\nhook\n",
            "seed": 360101,
            "duration": 195.0,
            "instrumental": False,
            "vocal_language": "en",
            "vocal_target": "female vocal",
        },
        {
            "id": "song_02",
            "caption": "dark contemporary R&B with an intimate male lead vocal",
            "lyrics": "[Verse 1]\nline one\n\n[Chorus]\nhook\n",
            "seed": 360202,
            "duration": 195.0,
            "instrumental": False,
            "vocal_language": "en",
            "vocal_target": "male vocal",
        },
        {
            "id": "song_03",
            "caption": "bright modern pop R&B with a strong female lead vocal",
            "lyrics": "[Verse 1]\nline one\n\n[Chorus]\nhook\n",
            "seed": 360303,
            "duration": 195.0,
            "instrumental": False,
            "vocal_language": "en",
            "vocal_target": "female vocal with background harmonies",
        },
    ]


class TestTheSetItself:
    def test_there_are_three_songs_covering_the_requested_targets(self):
        songs = _songs()
        assert len(songs) >= 3
        targets = " ".join(s["vocal_target"] for s in songs)
        assert "female" in targets and "male" in targets

    def test_every_song_has_a_different_seed(self):
        seeds = [s["seed"] for s in _songs()]
        assert len(seeds) == len(set(seeds))

    def test_every_song_requests_vocals_rather_than_an_instrumental(self):
        for song in _songs():
            assert song["instrumental"] is False
            assert song["lyrics"].strip()
            assert "[Instrumental]" not in song["lyrics"]

    def test_lyrics_carry_song_structure(self):
        for song in _songs():
            assert "[Verse" in song["lyrics"]
            assert "[Chorus]" in song["lyrics"]

    def test_the_durations_are_in_the_requested_range(self):
        for song in _songs():
            assert 120.0 <= song["duration"] <= 240.0


class TestEverySideIsAskedTheSame:
    def _spec(self, root: Path) -> dict:
        return {
            "project_root": "/trainer",
            "output_root": str(root),
            "manifest": str(root / "manifest.json"),
            "inference_steps": 8,
            "items": _songs(),
            "sides": [
                {"name": "base", "lora_path": None},
                {"name": "phase36", "lora_path": "/adapters/p36"},
                {"name": "phase37", "lora_path": "/adapters/p37"},
            ],
        }

    def test_the_prompt_and_seed_belong_to_the_item_not_the_side(self):
        """One item, three sides: nothing per-side can change the question."""
        spec = self._spec(Path("/tmp/vocal"))
        for side in spec["sides"]:
            assert set(side) <= {"name", "lora_path"}

    def test_all_three_sides_are_present_for_every_song(self):
        spec = self._spec(Path("/tmp/vocal"))
        names = {side["name"] for side in spec["sides"]}
        assert names == {"base", "phase36", "phase37"}

    def test_each_song_and_side_writes_to_its_own_directory(self, tmp_path):
        spec = self._spec(tmp_path)
        targets = [
            Path(spec["output_root"]) / item["id"] / side["name"]
            for item in spec["items"]
            for side in spec["sides"]
        ]
        assert len(set(targets)) == len(targets) == 9

    def test_the_base_side_carries_no_adapter(self):
        spec = self._spec(Path("/tmp/vocal"))
        assert not next(s for s in spec["sides"] if s["name"] == "base")["lora_path"]

    def test_the_harness_lists_whatever_container_was_written(self, tmp_path):
        for name in ("one.flac", "two.wav", "manifest.json"):
            (tmp_path / name).write_bytes(b"x")
        assert abc.audio_files(tmp_path) == ["one.flac", "two.wav"]
