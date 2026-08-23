"""Resume: the checkpoint reopens, the provenance still matches, the step advances.

A run that never resumed has not shown that a longer one could survive
an interruption. Phase 36 additionally needs the resumed checkpoint's
provenance to verify — a checkpoint that reloads but cannot say what
made it is not a resumable artifact, it is a directory of tensors.
"""

import json

import pytest

from luber_training.canary import inspect_checkpoint
from luber_training.checkpoint_provenance import (
    CHECKPOINT_PROVENANCE_NAME,
    CheckpointProvenance,
    verify_checkpoint_provenance,
    write_checkpoint_provenance,
)


def _provenance(**overrides) -> CheckpointProvenance:
    base = {
        "experiment_id": "exp_1",
        "run_id": "run_1",
        "checkpoint_path": "output/checkpoints/epoch_10_loss_1.0",
        "epoch": 10,
        "step": 60,
        "base_model_id": "mdl_1",
        "dataset_id": "DS_1",
        "dataset_lock_sha256": "a" * 64,
        "curation_id": "CUR_1",
        "curation_lock_sha256": "b" * 64,
        "train_split_digest": "c" * 64,
        "validation_split_digest": "d" * 64,
        "evaluation_split_digest": "e" * 64,
        "config_digest": "f" * 64,
        "lora_rank": 16,
        "precision": "bf16",
        "device": "MPS",
        "optimizer": "adamw",
        "learning_rate": 1e-4,
        "seed": 42,
        "code_commit": "0" * 40,
    }
    base.update(overrides)
    return CheckpointProvenance(**base)


def _checkpoint(tmp_path, name: str, **overrides):
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_model.safetensors").write_bytes(b"weights")
    (directory / "training_state.pt").write_bytes(b"state")
    write_checkpoint_provenance(directory, _provenance(**overrides))
    return directory


class TestASegmentBoundaryCheckpoint:
    def test_it_carries_the_step_it_stopped_at(self, tmp_path):
        directory = _checkpoint(tmp_path, "epoch_10", epoch=10, step=60)
        recorded = json.loads((directory / CHECKPOINT_PROVENANCE_NAME).read_text(encoding="utf-8"))
        assert recorded["epoch"] == 10
        assert recorded["step"] == 60

    def test_the_resumed_checkpoint_advances_the_step(self, tmp_path):
        first = _checkpoint(tmp_path, "epoch_10", epoch=10, step=60)
        second = _checkpoint(tmp_path, "epoch_20", epoch=20, step=120)
        source = json.loads((first / CHECKPOINT_PROVENANCE_NAME).read_text(encoding="utf-8"))
        final = json.loads((second / CHECKPOINT_PROVENANCE_NAME).read_text(encoding="utf-8"))
        assert final["step"] > source["step"]

    def test_both_checkpoints_cite_the_same_data(self, tmp_path):
        """A resume that quietly changed dataset is a different run."""
        first = _checkpoint(tmp_path, "epoch_10", epoch=10, step=60)
        second = _checkpoint(tmp_path, "epoch_20", epoch=20, step=120)
        for field in ("train_split_digest", "dataset_lock_sha256", "config_digest"):
            a = json.loads((first / CHECKPOINT_PROVENANCE_NAME).read_text(encoding="utf-8"))
            b = json.loads((second / CHECKPOINT_PROVENANCE_NAME).read_text(encoding="utf-8"))
            assert a[field] == b[field]


class TestVerificationAtResumeTime:
    def test_a_checkpoint_whose_data_disagrees_is_a_mismatch_not_a_pass(self, tmp_path):
        directory = _checkpoint(tmp_path, "epoch_10", train_split_digest="9" * 64)
        verdict = verify_checkpoint_provenance(directory, expected={"train_split_digest": "c" * 64})
        assert not verdict.ok
        assert verdict.mismatches

    def test_a_checkpoint_with_matching_expectations_verifies(self, tmp_path):
        directory = _checkpoint(tmp_path, "epoch_10")
        verdict = verify_checkpoint_provenance(
            directory,
            expected={
                "train_split_digest": "c" * 64,
                "config_digest": "f" * 64,
                "run_id": "run_1",
            },
        )
        assert verdict.ok

    def test_a_checkpoint_whose_provenance_was_deleted_fails(self, tmp_path):
        directory = _checkpoint(tmp_path, "epoch_10")
        (directory / CHECKPOINT_PROVENANCE_NAME).unlink()
        assert not verify_checkpoint_provenance(directory).ok
        assert "no provenance record was written beside this checkpoint" in (
            inspect_checkpoint(directory).problems
        )

    @pytest.mark.parametrize("field", ["train_split_digest", "config_digest", "code_commit"])
    def test_a_provenance_with_a_blanked_field_fails_verification(self, tmp_path, field):
        directory = _checkpoint(tmp_path, "epoch_10")
        payload = json.loads((directory / CHECKPOINT_PROVENANCE_NAME).read_text(encoding="utf-8"))
        payload[field] = ""
        (directory / CHECKPOINT_PROVENANCE_NAME).write_text(json.dumps(payload), encoding="utf-8")
        verdict = verify_checkpoint_provenance(directory)
        assert not verdict.ok
        assert field in verdict.missing_fields


class TestBaseModelFreeze:
    def test_an_unchanged_base_digest_reads_as_preserved(self):
        from luber_training.experiment import ParameterUpdateEvidence

        evidence = ParameterUpdateEvidence(
            base_model_digest_before="a" * 64, base_model_digest_after="a" * 64
        )
        assert evidence.base_model_preserved is True

    def test_a_changed_base_digest_reads_as_not_preserved(self):
        from luber_training.experiment import ParameterUpdateEvidence

        evidence = ParameterUpdateEvidence(
            base_model_digest_before="a" * 64, base_model_digest_after="b" * 64
        )
        assert evidence.base_model_preserved is False

    def test_an_unmeasured_base_digest_is_unknown_not_preserved(self):
        """Nobody measured it is not the same as nothing changed."""
        from luber_training.experiment import ParameterUpdateEvidence

        assert ParameterUpdateEvidence().base_model_preserved is None
