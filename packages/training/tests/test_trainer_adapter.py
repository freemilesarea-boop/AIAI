"""The trainer adapter, the command compiler, and shell safety.

Two properties matter here. The compiled command must contain only flags
the installed trainer actually has — checked against the parser source,
not against memory — and no operator-supplied string may ever reach a
shell.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from conftest import curated_record, manifest_record

from luber_training.config import TrainingConfig, preset
from luber_training.entities import TrainingDatasetRef
from luber_training.ids import EntityKind, new_id
from luber_training.plan import TrainingPlan, default_requirements
from luber_training.trainer_adapter import (
    INSTRUMENTAL_MARKER,
    AdapterError,
    build_dataset,
    compile_command,
    to_trainer_sample,
    validate_dataset,
)

#: The installed parser, read at test time. If ACE-Step is not present
#: the contract tests skip rather than assert against a guess.
ACE_STEP_ARGS = Path.home() / "ace-step-1.5" / "acestep" / "training_v2" / "cli" / "args.py"
requires_ace_step = pytest.mark.skipif(
    not ACE_STEP_ARGS.is_file(), reason="the pinned ACE-Step tree is not installed"
)


def a_plan(config: TrainingConfig | None = None) -> TrainingPlan:
    config = config or preset("LORA_STANDARD")
    return TrainingPlan(
        plan_id=new_id(EntityKind.PLAN),
        run_id=new_id(EntityKind.RUN),
        experiment_id=new_id(EntityKind.EXPERIMENT),
        base_model_id=new_id(EntityKind.MODEL),
        base_model_upstream_commit="6d467e4b5081ccb0abf1ec1bf4fdf9051a2d34b0",
        dataset_ref=TrainingDatasetRef(
            dataset_id="ds-1",
            dataset_lock_sha256="a" * 64,
            curation_id="cur-1",
            curation_lock_sha256="b" * 64,
            curated_manifest_sha256="c" * 64,
            manifest_artifact_ref="curation://cur-1/curated_manifest",
        ),
        config=config,
        execution_backend="remote-gpu",
        requirements=default_requirements(config),
    )


class TestDatasetAdapter:
    def test_a_record_becomes_a_trainer_sample(self):
        sample = to_trainer_sample(manifest_record("trk_a"))
        assert sample.filename == "trk_a.wav"
        assert sample.caption
        assert sample.duration > 0

    def test_supplied_lyrics_are_carried_through(self):
        sample = to_trainer_sample(manifest_record("trk_a", lyrics="[Verse]\nsomething"))
        assert sample.lyrics == "[Verse]\nsomething"

    def test_a_declared_instrumental_gets_the_marker(self):
        record = manifest_record("trk_i", vocal_class="INSTRUMENTAL", lyrics=None)
        assert to_trainer_sample(record).lyrics == INSTRUMENTAL_MARKER

    def test_an_uncertain_vocal_is_not_labelled_instrumental(self):
        """The trainer defaults absent lyrics to "[Instrumental]".

        A vocal track silently labelled instrumental is a training-data
        error invisible in the loss curve, so an unknown stays empty
        rather than taking the default.
        """
        record = manifest_record("trk_u", vocal_class="UNCERTAIN", lyrics=None)
        assert to_trainer_sample(record).lyrics == ""

    def test_the_caption_invents_nothing(self):
        record = manifest_record("trk_bare", language="unknown", vocal_class="UNCERTAIN")
        record["metadata"]["sidecar"] = {}
        record["music"]["bpm_confidence"] = None
        record["music"]["key_confidence"] = None
        caption = to_trainer_sample(record).caption
        assert caption == "unlabelled music"

    def test_a_low_confidence_key_is_not_forwarded(self):
        """The fixture's key confidence is 0.4, below what Phase 23 trusts."""
        record = manifest_record("trk_a")
        record["music"]["key_confidence"] = None
        assert to_trainer_sample(record).keyscale == ""

    def test_a_record_without_a_filename_is_refused(self):
        record = manifest_record("trk_a")
        record["source"]["source_filename"] = ""
        with pytest.raises(AdapterError, match="indexes samples by filename"):
            to_trainer_sample(record)

    def test_only_selected_records_are_adapted(self):
        records = [
            curated_record(manifest_record("trk_in"), action="KEEP"),
            curated_record(manifest_record("trk_out"), action="DOWNSAMPLE"),
        ]
        dataset = build_dataset(records)
        assert [sample.filename for sample in dataset.samples] == ["trk_in.wav"]

    def test_the_dataset_is_deterministic(self):
        records = [curated_record(manifest_record(f"trk_{i}")) for i in range(5)]
        first = build_dataset(records).to_dict()
        second = build_dataset(list(reversed(records))).to_dict()
        assert first == second

    def test_duplicate_filenames_are_reported(self):
        """The loader indexes by name, so two are one."""
        records = [
            curated_record(manifest_record("trk_a")),
            curated_record(manifest_record("trk_a")),
        ]
        problems = validate_dataset(build_dataset(records))
        assert any("duplicate filename" in problem for problem in problems)

    def test_an_empty_dataset_is_reported(self):
        assert any("no samples" in problem for problem in validate_dataset(build_dataset([])))

    def test_the_canonical_manifest_is_not_reshaped(self):
        """The adapter reads; it never writes back to the manifest."""
        record = manifest_record("trk_a")
        before = dict(record)
        to_trainer_sample(record)
        assert record == before


class TestCommandCompiler:
    def test_it_produces_argv_not_a_shell_string(self):
        command = compile_command(a_plan(), trainer_root="/opt/ace-step")
        assert isinstance(command.argv, list)
        assert all(isinstance(part, str) for part in command.argv)
        assert command.argv[1] == "train.py"
        assert command.argv[2] == "fixed"

    def test_it_carries_no_secrets(self):
        command = compile_command(a_plan(), trainer_root="/opt/ace-step")
        joined = " ".join(command.argv).lower()
        for marker in ("password", "token", "secret", "api_key", "ssh", "begin rsa"):
            assert marker not in joined
        assert command.required_env == ()

    def test_paths_are_placeholders_not_this_machine(self):
        """A plan compiled on a Mac has to run on a Linux box."""
        command = compile_command(a_plan(), trainer_root="/opt/ace-step")
        joined = " ".join(command.argv)
        assert "${LUBER_DATASET_DIR}" in joined
        assert "/Users/" not in joined

    def test_the_same_plan_compiles_identically(self):
        plan = a_plan()
        assert (
            compile_command(plan, trainer_root="/opt/x").argv
            == compile_command(plan, trainer_root="/opt/x").argv
        )

    def test_boolean_flags_use_the_negative_form_explicitly(self):
        """`--gradient-checkpointing` defaults ON upstream.

        Omitting it would silently keep the trainer's default rather
        than honour the config.
        """
        config = preset("LORA_STANDARD").with_overrides(gradient_checkpointing=False)
        argv = compile_command(a_plan(config), trainer_root="/opt/x").argv
        assert "--no-gradient-checkpointing" in argv
        assert "--gradient-checkpointing" not in argv

    def test_lora_parameters_are_emitted(self):
        argv = compile_command(a_plan(), trainer_root="/opt/x").argv
        for flag in ("--rank", "--alpha", "--dropout", "--target-modules", "--attention-type"):
            assert flag in argv

    def test_ddp_only_when_multiple_devices(self):
        single = compile_command(a_plan(), trainer_root="/opt/x").argv
        assert single[single.index("--strategy") + 1] == "auto"
        many = compile_command(
            a_plan(preset("LORA_STANDARD").with_overrides(num_devices=4)), trainer_root="/opt/x"
        ).argv
        assert many[many.index("--strategy") + 1] == "ddp"

    @requires_ace_step
    def test_every_emitted_flag_exists_in_the_installed_parser(self):
        """The contract check. A flag the trainer lacks is a silent no-op."""
        source = ACE_STEP_ARGS.read_text(encoding="utf-8")
        declared = set(re.findall(r'"(--[a-z0-9-]+)"', source))
        # BooleanOptionalAction declares `--x` and accepts `--no-x`.
        declared |= {flag.replace("--", "--no-", 1) for flag in declared}
        # `--learning-rate` is declared as an alias on the same call.
        declared.add("--learning-rate")

        argv = compile_command(a_plan(), trainer_root="/opt/x").argv
        emitted = {part for part in argv if part.startswith("--")}
        unknown = sorted(emitted - declared)
        assert not unknown, f"flags absent from the installed trainer: {unknown}"

    @requires_ace_step
    def test_the_trainer_has_no_max_steps_flag(self):
        """Guards the config schema against drifting back."""
        source = ACE_STEP_ARGS.read_text(encoding="utf-8")
        assert '"--max-steps"' not in source


class TestShellSafety:
    @pytest.mark.parametrize(
        "hostile",
        [
            'x"; rm -rf ~; echo "',
            "$(whoami)",
            "`id`",
            "../../etc/passwd",
            "a && curl evil.example",
            "| tee /tmp/x",
        ],
    )
    def test_a_hostile_path_stays_one_argv_element(self, hostile: str):
        """No shell is involved, so metacharacters are inert text."""
        plan = a_plan()
        plan.dataset_dir = hostile
        command = compile_command(plan, trainer_root="/opt/x")
        assert command.argv.count(hostile) == 1
        assert hostile in command.argv

    def test_control_characters_are_refused(self):
        """They corrupt logs and any future shell transport."""
        plan = a_plan()
        plan.dataset_dir = "dataset\nrm -rf /"
        with pytest.raises(AdapterError, match="control character"):
            compile_command(plan, trainer_root="/opt/x")

    def test_a_hostile_trainer_root_is_refused(self):
        with pytest.raises(AdapterError, match="control character"):
            compile_command(a_plan(), trainer_root="/opt/x\n/evil")

    def test_display_quoting_is_for_reading_only(self):
        """`display()` is never executed; it exists so a human can read it."""
        plan = a_plan()
        plan.output_dir = "a b; c"
        command = compile_command(plan, trainer_root="/opt/x")
        rendered = command.display()
        assert "'a b; c'" in rendered
        assert "a b; c" in command.argv


class TestPlanImmutabilityAndSecrets:
    def test_a_plan_hash_ignores_compile_time(self):
        first, second = a_plan(), a_plan()
        second.run_id = first.run_id
        second.experiment_id = first.experiment_id
        second.base_model_id = first.base_model_id
        second.compiled_at = "1999-01-01T00:00:00+00:00"
        assert first.digest() == second.digest()

    def test_a_changed_config_changes_the_hash(self):
        first = a_plan()
        second = a_plan(preset("LORA_HIGH_QUALITY"))
        second.run_id = first.run_id
        second.experiment_id = first.experiment_id
        second.base_model_id = first.base_model_id
        assert first.digest() != second.digest()

    def test_a_plan_holds_references_never_paths_as_identity(self):
        plan = a_plan()
        assert plan.dataset_ref.dataset_lock_sha256
        assert plan.dataset_ref.manifest_artifact_ref.startswith("curation://")

    def test_secret_refs_are_names(self):
        plan = a_plan()
        plan.secret_refs = ("prod-ssh-key", "hf-token-name")
        payload = plan.to_dict()
        assert payload["secret_refs"] == ["prod-ssh-key", "hf-token-name"]
        assert "BEGIN" not in str(payload)

    def test_vram_is_an_explicit_unknown(self):
        plan = a_plan()
        assert plan.requirements.minimum_vram_mb is None
        assert any("no VRAM figure" in note for note in plan.requirements.unknown_requirements)
