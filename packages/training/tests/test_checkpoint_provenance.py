"""Checkpoint provenance: present, whole, and matching.

Phase 35B's pilot produced sound checkpoints that the integrity check
still called incomplete, because nothing beside them said what they
were. The fix is only worth having if it fails loudly when the record
is missing or has holes in it — a provenance file that exists and says
nothing is worse than none, because the first looks like a pass.
"""

import json

import pytest

from luber_training.checkpoint_provenance import (
    CHECKPOINT_PROVENANCE_NAME,
    REQUIRED_FIELDS,
    REQUIRED_NUMERIC_FIELDS,
    CheckpointProvenance,
    ProvenanceError,
    read_checkpoint_provenance,
    verify_checkpoint_provenance,
    write_checkpoint_provenance,
)


def _provenance(**overrides) -> CheckpointProvenance:
    base = {
        "experiment_id": "exp_1",
        "run_id": "run_1",
        "checkpoint_path": "output/checkpoints/epoch_24_loss_1.0",
        "epoch": 24,
        "step": 24,
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
        "code_commit": "0123456789abcdef",
    }
    base.update(overrides)
    return CheckpointProvenance(**base)


class TestWriting:
    def test_it_writes_a_record_that_verifies(self, tmp_path):
        write_checkpoint_provenance(tmp_path, _provenance())
        verdict = verify_checkpoint_provenance(tmp_path)
        assert verdict.ok
        assert verdict.present and verdict.complete
        assert not verdict.missing_fields

    def test_the_record_lands_under_a_predictable_name(self, tmp_path):
        path = write_checkpoint_provenance(tmp_path, _provenance())
        assert path.name == CHECKPOINT_PROVENANCE_NAME
        assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == "run_1"

    @pytest.mark.parametrize("field", REQUIRED_FIELDS)
    def test_it_refuses_to_write_a_record_missing_a_required_field(self, tmp_path, field):
        with pytest.raises(ProvenanceError, match=field):
            write_checkpoint_provenance(tmp_path, _provenance(**{field: ""}))
        assert not (tmp_path / CHECKPOINT_PROVENANCE_NAME).exists()

    @pytest.mark.parametrize("field", REQUIRED_NUMERIC_FIELDS)
    def test_a_negative_numeric_field_is_refused(self, tmp_path, field):
        with pytest.raises(ProvenanceError, match=field):
            write_checkpoint_provenance(tmp_path, _provenance(**{field: -1}))

    def test_the_artifact_class_is_recorded_and_never_empty(self, tmp_path):
        write_checkpoint_provenance(tmp_path, _provenance())
        recorded = read_checkpoint_provenance(tmp_path)
        assert recorded["artifact_class"] == [
            "EXPERIMENTAL",
            "NON_PRODUCTION",
            "NEVER_AUTO_PROMOTE",
        ]

    def test_an_absent_base_model_digest_stays_absent(self, tmp_path):
        """Unmeasured is not 'unchanged', and the record must not blur them."""
        write_checkpoint_provenance(tmp_path, _provenance(base_model_digest=None))
        assert read_checkpoint_provenance(tmp_path)["base_model_digest"] is None


class TestVerifying:
    def test_a_checkpoint_with_no_record_fails(self, tmp_path):
        verdict = verify_checkpoint_provenance(tmp_path)
        assert not verdict.present and not verdict.ok
        assert "no provenance record" in verdict.detail

    def test_an_unreadable_record_fails(self, tmp_path):
        (tmp_path / CHECKPOINT_PROVENANCE_NAME).write_text("{not json", encoding="utf-8")
        verdict = verify_checkpoint_provenance(tmp_path)
        assert verdict.present and not verdict.ok
        assert "unreadable" in verdict.detail

    def test_a_record_with_holes_in_it_fails(self, tmp_path):
        payload = _provenance().to_dict()
        payload["train_split_digest"] = ""
        payload["dataset_lock_sha256"] = ""
        (tmp_path / CHECKPOINT_PROVENANCE_NAME).write_text(json.dumps(payload), encoding="utf-8")
        verdict = verify_checkpoint_provenance(tmp_path)
        assert not verdict.ok
        assert set(verdict.missing_fields) == {"train_split_digest", "dataset_lock_sha256"}

    def test_expectations_that_disagree_are_reported_as_mismatches(self, tmp_path):
        """A checkpoint from the wrong dataset is a different failure from one with no dataset."""
        write_checkpoint_provenance(tmp_path, _provenance())
        verdict = verify_checkpoint_provenance(tmp_path, expected={"train_split_digest": "9" * 64})
        assert verdict.complete
        assert not verdict.ok
        assert verdict.mismatches and "train_split_digest" in verdict.mismatches[0]

    def test_expectations_that_agree_pass(self, tmp_path):
        write_checkpoint_provenance(tmp_path, _provenance())
        verdict = verify_checkpoint_provenance(
            tmp_path,
            expected={"train_split_digest": "c" * 64, "run_id": "run_1"},
        )
        assert verdict.ok

    def test_a_legacy_canary_record_is_present_but_never_complete(self, tmp_path):
        """It predates these fields and is not pretended to answer them."""
        (tmp_path / "luber_canary_provenance.json").write_text(
            json.dumps({"plan_digest": "z" * 64, "schema_version": "luber-canary/1"}),
            encoding="utf-8",
        )
        verdict = verify_checkpoint_provenance(tmp_path)
        assert verdict.present
        assert verdict.legacy
        assert not verdict.complete
        assert not verdict.ok

    def test_the_current_record_wins_over_a_legacy_one(self, tmp_path):
        (tmp_path / "luber_canary_provenance.json").write_text("{}", encoding="utf-8")
        write_checkpoint_provenance(tmp_path, _provenance())
        verdict = verify_checkpoint_provenance(tmp_path)
        assert verdict.ok and not verdict.legacy


class TestItTiesACheckpointToItsData:
    def test_every_identity_a_later_reader_needs_is_recorded(self, tmp_path):
        """The point of the file: what data, what config, what code."""
        write_checkpoint_provenance(tmp_path, _provenance())
        recorded = read_checkpoint_provenance(tmp_path)
        for field in (
            "dataset_id",
            "dataset_lock_sha256",
            "curation_id",
            "curation_lock_sha256",
            "train_split_digest",
            "validation_split_digest",
            "evaluation_split_digest",
            "config_digest",
            "code_commit",
        ):
            assert recorded[field], field

    def test_checkpoint_inspection_uses_the_same_judgement(self, tmp_path):
        from luber_training.canary import inspect_checkpoint

        (tmp_path / "adapter.safetensors").write_bytes(b"not really weights")
        assert "no provenance record was written beside this checkpoint" in (
            inspect_checkpoint(tmp_path).problems
        )

        write_checkpoint_provenance(tmp_path, _provenance())
        integrity = inspect_checkpoint(tmp_path)
        assert integrity.provenance_present
        assert integrity.provenance_verdict is not None
        assert integrity.provenance_verdict.ok
        assert not any("provenance" in problem for problem in integrity.problems)
