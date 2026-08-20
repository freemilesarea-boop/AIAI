"""Config strictness, and the five gates that stand between data and a GPU.

The gate tests are the ones that must never be allowed to rot. Audio
trained on without permission cannot be untrained, a leaked benchmark
silently stops measuring generalisation, and both failures are invisible
in a loss curve.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import build_locked_dataset, manifest_record

from luber_training import config as config_module
from luber_training.config import ConfigError, TrainingConfig, from_dict, preset, validate
from luber_training.entities import FailureCode
from luber_training.gates import (
    GateInputs,
    curation_lock_gate,
    dataset_lock_gate,
    leakage_gate,
    rights_gate,
    run_all,
    self_generated_gate,
)


def curated(records):
    from conftest import curated_record

    return [curated_record(record) for record in records]


class TestTrainingConfig:
    def test_the_same_config_hashes_the_same(self):
        assert TrainingConfig().digest() == TrainingConfig().digest()

    def test_a_changed_field_changes_the_hash(self):
        assert TrainingConfig().digest() != TrainingConfig(rank=32).digest()

    def test_presets_validate(self):
        for name in config_module.PRESETS:
            validate(preset(name))

    def test_smoke_is_deliberately_useless_for_training(self):
        """It proves the plumbing; it must not look like a real run."""
        smoke = preset("SMOKE")
        assert smoke.epochs == 1
        assert smoke.rank < preset("LORA_SMALL").rank

    def test_an_unknown_field_is_refused(self):
        """A parameter the trainer ignores must not travel silently."""
        with pytest.raises(ConfigError, match="unrecognised"):
            from_dict({"learning_rate": 1e-4, "max_steps": 1000})

    @pytest.mark.parametrize("absent", ["max_steps", "validation_interval", "checkpoint_interval"])
    def test_fields_the_trainer_lacks_do_not_exist(self, absent: str):
        """Audited: the installed parser has no such flag."""
        assert absent not in TrainingConfig.__dataclass_fields__

    def test_checkpoint_field_is_named_for_epochs(self):
        """`--save-every` counts epochs, and the name has to say so."""
        assert "checkpoint_every_epochs" in TrainingConfig.__dataclass_fields__

    def test_full_finetune_is_not_offered(self):
        """The installed trainer has no entry point for it."""
        assert not hasattr(config_module.TrainingStrategy, "FULL")
        with pytest.raises(ConfigError, match="not accepted by the trainer"):
            validate(TrainingConfig(strategy="FULL"))

    @pytest.mark.parametrize(
        ("field", "value"),
        [("optimizer_type", "lion"), ("scheduler_type", "magic"), ("precision", "fp8")],
    )
    def test_values_the_trainer_rejects_are_refused(self, field: str, value: str):
        with pytest.raises(ConfigError, match="not accepted by the trainer"):
            validate(TrainingConfig(**{field: value}))

    def test_nonsense_numbers_are_refused(self):
        with pytest.raises(ConfigError, match="must be positive"):
            validate(TrainingConfig(epochs=0))
        with pytest.raises(ConfigError, match=r"dropout must be in \[0, 1\)"):
            validate(TrainingConfig(dropout=1.5))

    def test_a_config_for_another_trainer_is_refused(self):
        """Re-audit before training against a different ACE-Step tree."""
        with pytest.raises(ConfigError, match="re-audit"):
            validate(TrainingConfig(ace_step_commit="deadbeef" * 5))


class TestDatasetLockGate:
    def test_a_matching_dataset_passes(self, tmp_path: Path):
        dataset, _curation = build_locked_dataset(tmp_path, [manifest_record("trk_a")])
        result = dataset_lock_gate(
            dataset / "dataset_lock.json", dataset / "dataset_manifest.jsonl"
        )
        assert result.passed, result.detail

    def test_a_modified_manifest_fails(self, tmp_path: Path):
        """The corrupt-lock scenario: data changed after the freeze."""
        dataset, _ = build_locked_dataset(tmp_path, [manifest_record("trk_a")])
        path = dataset / "dataset_manifest.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows.append(manifest_record("trk_sneaked"))
        path.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
        )
        result = dataset_lock_gate(dataset / "dataset_lock.json", path)
        assert not result.passed
        assert result.failure_code == FailureCode.DATASET_LOCK_INVALID.value

    def test_a_missing_lock_fails(self, tmp_path: Path):
        assert not dataset_lock_gate(tmp_path / "nope.json", tmp_path / "also.jsonl").passed


class TestCurationLockGate:
    def test_a_matching_curation_passes(self, tmp_path: Path):
        dataset, curation = build_locked_dataset(tmp_path, [manifest_record("trk_a")])
        result = curation_lock_gate(
            curation / "curation_lock.json",
            curation / "curated_manifest.jsonl",
            dataset / "dataset_lock.json",
        )
        assert result.passed, result.detail

    def test_a_curation_from_a_different_dataset_fails(self, tmp_path: Path):
        """The mistake that is otherwise invisible.

        Both artifacts are internally consistent; they simply do not
        belong together. Only the recorded dataset-lock digest catches
        it.
        """
        _first, curation = build_locked_dataset(tmp_path / "a", [manifest_record("trk_a")])
        second, _ = build_locked_dataset(tmp_path / "b", [manifest_record("trk_b")])
        result = curation_lock_gate(
            curation / "curation_lock.json",
            curation / "curated_manifest.jsonl",
            second / "dataset_lock.json",
        )
        assert not result.passed
        assert "different dataset lock" in result.detail

    def test_a_modified_curated_manifest_fails(self, tmp_path: Path):
        dataset, curation = build_locked_dataset(tmp_path, [manifest_record("trk_a")])
        path = curation / "curated_manifest.jsonl"
        path.write_text(path.read_text().replace('"KEEP"', '"KEEP_PRIORITY"'), encoding="utf-8")
        result = curation_lock_gate(
            curation / "curation_lock.json", path, dataset / "dataset_lock.json"
        )
        assert not result.passed


class TestRightsGate:
    def test_permitted_tracks_pass(self):
        assert rights_gate(curated([manifest_record("trk_a")])).passed

    @pytest.mark.parametrize(
        "overrides",
        [
            {"permission": "UNKNOWN", "rights_status": "UNKNOWN"},
            {"permission": "FALSE"},
            {"rights_status": "RESTRICTED"},
            {"hard_blocks": ["SELF_MODEL_OUTPUT"]},
        ],
    )
    def test_forbidden_provenance_fails(self, overrides: dict):
        result = rights_gate(curated([manifest_record("trk_bad", **overrides)]))
        assert not result.passed
        assert result.failure_code == FailureCode.RIGHTS_GATE_FAILED.value
        assert "trk_bad" in result.offending_ids

    def test_the_gate_reads_provenance_not_the_eligibility_flag(self):
        """A permissive curation must not be able to admit a track.

        Phase 23 can build with `include_rights_unknown`, which marks
        unknown-rights material eligible for *analysis*. Production
        training must not reach that path, so the gate re-derives from
        the provenance block itself.
        """
        record = manifest_record(
            "trk_flagged", permission="UNKNOWN", rights_status="UNKNOWN", training_eligible=True
        )
        assert not rights_gate(curated([record])).passed

    def test_unselected_tracks_are_not_judged(self):
        """A track curation excluded is not going to be trained on."""
        from conftest import curated_record

        record = curated_record(
            manifest_record("trk_bad", permission="FALSE"), action="EXCLUDE_POLICY"
        )
        assert rights_gate([record]).passed

    def test_no_override_parameter_exists(self):
        """Structural: the function has no way to be told to allow it."""
        import inspect

        signature = inspect.signature(rights_gate)
        assert list(signature.parameters) == ["curated_records"]


class TestLeakageGate:
    def test_clean_training_data_passes(self):
        assert leakage_gate(curated([manifest_record("trk_a")])).passed

    def test_an_evaluation_only_id_fails(self):
        result = leakage_gate(
            curated([manifest_record("trk_bench")]),
            evaluation_only_ids=frozenset({"trk_bench"}),
        )
        assert not result.passed
        assert result.failure_code == FailureCode.EVALUATION_LEAKAGE.value

    def test_leakage_is_caught_by_digest_not_only_by_id(self):
        """A benchmark track copied under another id is the same audio.

        Checking ids alone would miss it, and the benchmark would quietly
        stop measuring generalisation.
        """
        import hashlib

        digest = hashlib.sha256(b"benchmark-audio").hexdigest()
        record = manifest_record("trk_renamed", sha256=digest)
        result = leakage_gate(curated([record]), evaluation_only_digests=frozenset({digest}))
        assert not result.passed
        assert "trk_renamed" in result.offending_ids

    def test_a_validation_split_track_cannot_train(self):
        result = leakage_gate(curated([manifest_record("trk_v", split="VALIDATION")]))
        assert not result.passed
        assert "SPLIT_VALIDATION" in json.dumps(result.evidence)

    def test_a_test_split_track_cannot_train(self):
        assert not leakage_gate(curated([manifest_record("trk_t", split="TEST")])).passed


class TestSelfGeneratedGate:
    def test_human_material_passes(self):
        assert self_generated_gate(curated([manifest_record("trk_a")])).passed

    def test_self_model_output_is_blocked_by_default(self):
        result = self_generated_gate(
            curated([manifest_record("trk_self", source_type="SELF_MODEL_OUTPUT")])
        )
        assert not result.passed
        assert result.failure_code == FailureCode.SELF_GENERATED_BLOCKED.value

    def test_it_can_be_admitted_by_explicit_policy(self):
        result = self_generated_gate(
            curated([manifest_record("trk_self", source_type="SELF_MODEL_OUTPUT")]),
            allow_self_generated=True,
        )
        assert result.passed

    def test_indeterminate_provenance_blocks_rather_than_guesses(self):
        """Unknown origin is how self-generated audio gets in unnoticed."""
        record = manifest_record("trk_mystery")
        record["provenance"]["source_type"] = ""
        result = self_generated_gate(curated([record]))
        assert not result.passed
        assert result.evidence["reason"] == "INDETERMINATE_PROVENANCE"

    def test_indeterminate_is_not_cleared_by_the_allow_flag(self):
        """The flag admits *known* self-generated audio, not unknowns."""
        record = manifest_record("trk_mystery")
        record["provenance"]["source_type"] = ""
        assert not self_generated_gate(curated([record]), allow_self_generated=True).passed

    def test_third_party_ai_audio_is_not_self_model_output(self):
        """AI_GENERATED with cleared rights is legitimate material."""
        result = self_generated_gate(
            curated([manifest_record("trk_ai", source_type="AI_GENERATED")])
        )
        assert result.passed
        assert result.evidence["synthetic_total"] == 1


class TestGateBattery:
    def inputs(self, tmp_path: Path, records, **kwargs) -> GateInputs:
        dataset, curation = build_locked_dataset(tmp_path, records)
        return GateInputs(
            dataset_lock_path=dataset / "dataset_lock.json",
            dataset_manifest_path=dataset / "dataset_manifest.jsonl",
            curation_lock_path=curation / "curation_lock.json",
            curated_manifest_path=curation / "curated_manifest.jsonl",
            **kwargs,
        )

    def test_a_clean_dataset_clears_every_gate(self, tmp_path: Path):
        report = run_all(self.inputs(tmp_path, [manifest_record(f"trk_{i}") for i in range(3)]))
        assert report.passed, report.to_dict()
        assert len(report.results) == 5

    def test_the_first_failure_names_the_code(self, tmp_path: Path):
        report = run_all(self.inputs(tmp_path, [manifest_record("trk_bad", permission="FALSE")]))
        assert not report.passed
        assert report.failure_code() == FailureCode.RIGHTS_GATE_FAILED.value

    def test_later_gates_do_not_run_on_an_unverified_manifest(self, tmp_path: Path):
        """Answering a rights question from a file nobody verified would
        be answering the right question from the wrong data."""
        inputs = self.inputs(tmp_path, [manifest_record("trk_a")])
        inputs.dataset_manifest_path.write_text("", encoding="utf-8")
        report = run_all(inputs)
        assert not report.passed
        skipped = [r for r in report.results if r.evidence.get("skipped")]
        assert len(skipped) == 3
