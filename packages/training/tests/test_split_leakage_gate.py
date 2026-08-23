"""The gate that stops a contaminated experiment before it starts.

It reads the split manifest as written to disk rather than the builder's
in-memory result, because "the builder is correct" and "the file is
correct" are two different claims and only the second one is what the
trainer will be handed.
"""

import json
from pathlib import Path

from luber_dataset.splits import build_experiment_splits
from luber_training.entities import FailureCode
from luber_training.gates import split_leakage_gate


def _library(counts: dict[str, int]) -> dict:
    tracks, index = [], 0
    for group, count in counts.items():
        for _ in range(count):
            index += 1
            tracks.append(
                {
                    "track_id": f"track-{index:03d}",
                    "audio_sha256": f"{index:064x}",
                    "source_group": group,
                    "duration_seconds": 100.0 + index,
                    "training_allowed": True,
                }
            )
    return {"dataset_id": "LIB", "content_hash": "c" * 64, "tracks": tracks}


def _payload(**sizes) -> dict:
    splits = build_experiment_splits(
        _library({"A": 64, "B": 64}),
        train_size=sizes.get("train", 24),
        validation_size=sizes.get("validation", 4),
        evaluation_size=sizes.get("evaluation", 4),
        seed=36,
    )
    return splits.to_dict()


class TestItPassesACleanSplit:
    def test_a_clean_manifest_passes_and_records_the_digests(self):
        result = split_leakage_gate(_payload())
        assert result.passed
        assert result.evidence["TRAIN"] != result.evidence["EVALUATION"]
        assert result.evidence["splits_digest"]

    def test_it_survives_a_round_trip_through_json(self):
        """The file is what it gates, so the file is what it is given."""
        payload = json.loads(json.dumps(_payload()))
        assert split_leakage_gate(payload).passed


class TestItBlocksContamination:
    def test_a_training_track_in_the_evaluation_split_fails(self):
        payload = _payload()
        payload["evaluation"]["tracks"].append(payload["train"]["tracks"][0])
        result = split_leakage_gate(payload)
        assert not result.passed
        assert result.failure_code == FailureCode.EVALUATION_LEAKAGE.value
        assert result.offending_count >= 1

    def test_a_training_track_in_the_validation_split_fails(self):
        payload = _payload()
        payload["validation"]["tracks"].append(payload["train"]["tracks"][0])
        assert not split_leakage_gate(payload).passed

    def test_the_same_audio_under_a_different_id_still_fails(self):
        payload = _payload()
        stolen = dict(payload["train"]["tracks"][0])
        stolen["track_id"] = "renamed-but-the-same-recording"
        payload["evaluation"]["tracks"].append(stolen)
        result = split_leakage_gate(payload)
        assert not result.passed
        kinds = {f["kind"] for f in result.evidence["findings"]}
        assert "AUDIO_DIGEST_COLLISION" in kinds

    def test_an_empty_split_fails_rather_than_passing_vacuously(self):
        payload = _payload()
        payload["validation"]["tracks"] = []
        result = split_leakage_gate(payload)
        assert not result.passed
        assert "empty" in result.detail

    def test_an_unreadable_manifest_fails_closed(self):
        result = split_leakage_gate({"train": {"tracks": [{"audio_sha256": object()}]}})
        assert not result.passed


class TestItRunsOnWhatIsOnDisk:
    def test_the_gate_reads_the_written_file(self, tmp_path: Path):
        path = tmp_path / "splits.json"
        path.write_text(json.dumps(_payload()), encoding="utf-8")
        assert split_leakage_gate(json.loads(path.read_text(encoding="utf-8"))).passed

        poisoned = json.loads(path.read_text(encoding="utf-8"))
        poisoned["evaluation"]["tracks"].append(poisoned["train"]["tracks"][1])
        path.write_text(json.dumps(poisoned), encoding="utf-8")
        assert not split_leakage_gate(json.loads(path.read_text(encoding="utf-8"))).passed
