"""The bounded canary: its ceilings, its rights gate, its checkpoints.

Two kinds of test here.

Most of them never start a process. They exercise the bounds, the
dataset authorisation and the checkpoint contract, and they are the ones
that matter for the claim "a canary cannot become a training run" —
because that claim is about what the code will refuse, not about what
happened to run today.

A few of them do start the installed trainer, and skip out loud when
there is not one. Those are the ones that make the canary evidence
rather than architecture. They are bounded by the same envelope as
everything else: two synthetic samples, one epoch, a wall clock.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
from preflight_fixtures import a_plan

from luber_hardware import ComputeDevice, ExecutionLocation
from luber_training.canary import (
    CANARY_MAX_EPOCHS,
    CANARY_MAX_OPTIMIZER_STEPS,
    CANARY_MAX_RESUME_EPOCHS,
    CANARY_MAX_SAMPLES,
    SYNTHETIC_FIXTURE_TYPE,
    CanaryBoundsError,
    CanaryEnvelope,
    CanaryMode,
    CanaryStatus,
    ace_step_canary,
    bound_plan,
    default_workspace,
    inspect_checkpoint,
    latest_checkpoint,
    orchestration_canary,
    verify_canary_dataset,
    within,
    write_provenance,
)
from luber_training.config import Precision, TrainingConfig
from luber_training.gates import GateReport, GateResult

TRAINER_ROOT = Path.home() / "ace-step-1.5"
TRAINER_PYTHON = TRAINER_ROOT / ".venv" / "bin" / "python"
MODEL_DIR = TRAINER_ROOT / "checkpoints"

needs_trainer = pytest.mark.skipif(
    not (
        TRAINER_ROOT.is_dir()
        and (TRAINER_ROOT / "train.py").is_file()
        and TRAINER_PYTHON.is_file()
        and (MODEL_DIR / "acestep-v15-turbo").is_dir()
    ),
    reason=(
        "no ACE-Step installation with base weights was found at ~/ace-step-1.5. These "
        "tests run the real trainer and are skipped rather than faked."
    ),
)


def _fixture_dataset(directory: Path, samples: int = 2) -> Path:
    """A dataset directory shaped like upstream's fixture generator.

    Written here rather than generated, so the authorisation tests run
    without torch. The files are empty markers: nothing loads them.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(samples):
        (directory / f"test_{index:04d}.pt").write_bytes(b"")
    (directory / "manifest.json").write_text(
        json.dumps({"metadata": {"type": SYNTHETIC_FIXTURE_TYPE, "num_samples": samples}}),
        encoding="utf-8",
    )
    return directory


@pytest.fixture
def workspace():
    """A canary workspace beneath the trainer root, removed afterwards.

    Beneath the trainer root because ACE-Step refuses a dataset
    anywhere else; removed because what it leaves behind is an adapter
    that learned noise.
    """
    directory = default_workspace(TRAINER_ROOT, "pytest-canary")
    yield directory
    shutil.rmtree(directory, ignore_errors=True)


def _passing_gates() -> GateReport:
    return GateReport(
        results=[
            GateResult(name=name, passed=True)
            for name in ("dataset_lock", "curation_lock", "rights", "evaluation_leakage")
        ]
    )


# ── 20-22. the bounds are structural ─────────────────────────────────


class TestEnvelopeBounds:
    def test_the_step_ceiling_is_enforced_on_construction(self):
        with pytest.raises(CanaryBoundsError, match="optimizer step"):
            CanaryEnvelope(max_optimizer_steps=CANARY_MAX_OPTIMIZER_STEPS + 1)

    def test_the_sample_ceiling_is_enforced_on_construction(self):
        with pytest.raises(CanaryBoundsError, match="sample"):
            CanaryEnvelope(max_samples=CANARY_MAX_SAMPLES + 1)

    def test_the_epoch_ceiling_is_enforced_on_construction(self):
        with pytest.raises(CanaryBoundsError, match="epoch"):
            CanaryEnvelope(max_epochs=CANARY_MAX_EPOCHS + 1)

    def test_a_resume_canary_may_use_exactly_one_more_epoch(self):
        assert CanaryEnvelope(max_epochs=CANARY_MAX_RESUME_EPOCHS, resume=True).max_epochs == 2
        with pytest.raises(CanaryBoundsError):
            CanaryEnvelope(max_epochs=CANARY_MAX_RESUME_EPOCHS + 1, resume=True)

    def test_the_wall_clock_has_a_ceiling(self):
        with pytest.raises(CanaryBoundsError, match="run for up to"):
            CanaryEnvelope(wall_clock_seconds=999_999.0)

    def test_a_canary_cannot_become_full_training(self):
        """The length is derived from the envelope, never accepted."""
        envelope = CanaryEnvelope(max_samples=2)
        bounded = envelope.bound_config(TrainingConfig(epochs=60, checkpoint_every_epochs=10))
        assert bounded.epochs == CANARY_MAX_EPOCHS
        assert bounded.checkpoint_every_epochs == 1
        assert envelope.upper_bound_steps(bounded) <= CANARY_MAX_OPTIMIZER_STEPS

    def test_a_config_that_would_exceed_the_step_ceiling_is_refused(self):
        """Batch size cannot be used to smuggle in more steps."""
        envelope = CanaryEnvelope(max_samples=4, max_optimizer_steps=2)
        with pytest.raises(CanaryBoundsError, match="optimizer step"):
            envelope.bound_config(TrainingConfig(batch_size=1))

    def test_the_bounded_plan_is_a_different_plan(self):
        """A canary's checkpoint must not carry the run's identity."""
        plan = a_plan(device=ComputeDevice.MPS.value)
        envelope = CanaryEnvelope(max_samples=2)
        bounded = bound_plan(
            plan,
            envelope,
            dataset_dir=Path("/tmp/ds"),
            output_dir=Path("/tmp/out"),
            model_dir=Path("/tmp/model"),
        )
        assert bounded.digest() != plan.digest()
        assert bounded.config.epochs == CANARY_MAX_EPOCHS


# ── the rights gate is not relaxed for being small ───────────────────


class TestCanaryDataAuthorisation:
    def test_a_synthetic_fixture_is_permitted(self, tmp_path: Path):
        verdict = verify_canary_dataset(
            _fixture_dataset(tmp_path / "ds"), envelope=CanaryEnvelope(max_samples=2)
        )
        assert verdict.permitted
        assert verdict.kind == "SYNTHETIC"

    def test_material_with_no_provenance_is_refused(self, tmp_path: Path):
        directory = tmp_path / "ds"
        directory.mkdir()
        (directory / "a.pt").write_bytes(b"")
        verdict = verify_canary_dataset(directory, envelope=CanaryEnvelope(max_samples=2))
        assert not verdict.permitted
        assert verdict.kind == "UNAUTHORISED"
        assert "being small is not an authorisation" in verdict.detail

    def test_gate_cleared_material_is_permitted(self, tmp_path: Path):
        directory = tmp_path / "ds"
        directory.mkdir()
        (directory / "a.pt").write_bytes(b"")
        verdict = verify_canary_dataset(
            directory, envelope=CanaryEnvelope(max_samples=2), gate_report=_passing_gates()
        )
        assert verdict.permitted
        assert verdict.kind == "GATE_CLEARED"

    def test_failing_gates_are_still_a_refusal(self, tmp_path: Path):
        directory = tmp_path / "ds"
        directory.mkdir()
        (directory / "a.pt").write_bytes(b"")
        failing = GateReport(
            results=[GateResult(name="rights", passed=False, detail="not permitted")]
        )
        verdict = verify_canary_dataset(
            directory, envelope=CanaryEnvelope(max_samples=2), gate_report=failing
        )
        assert not verdict.permitted

    def test_a_directory_bigger_than_the_envelope_is_refused(self, tmp_path: Path):
        verdict = verify_canary_dataset(
            _fixture_dataset(tmp_path / "ds", samples=4), envelope=CanaryEnvelope(max_samples=2)
        )
        assert not verdict.permitted
        assert verdict.kind == "OVER_BOUND"

    def test_an_empty_directory_is_refused(self, tmp_path: Path):
        (tmp_path / "ds").mkdir()
        verdict = verify_canary_dataset(tmp_path / "ds", envelope=CanaryEnvelope(max_samples=2))
        assert not verdict.permitted
        assert verdict.kind == "EMPTY"


# ── 23-24. checkpoint integrity and provenance ───────────────────────


class TestCheckpointIntegrity:
    def test_a_missing_checkpoint_is_not_ok(self, tmp_path: Path):
        integrity = inspect_checkpoint(tmp_path / "nothing")
        assert not integrity.ok
        assert "does not exist" in integrity.problems[0]

    def test_a_checkpoint_without_provenance_is_not_ok(self, tmp_path: Path):
        directory = tmp_path / "ckpt"
        directory.mkdir()
        (directory / "adapter_model.safetensors").write_bytes(b"x" * 16)
        integrity = inspect_checkpoint(directory)
        assert not integrity.ok
        assert any("provenance" in problem for problem in integrity.problems)

    def test_provenance_traces_the_plan_device_and_precision(self, tmp_path: Path):
        plan = a_plan(
            device=ComputeDevice.MPS.value,
            config=TrainingConfig(epochs=1, precision=Precision.BF16.value),
        )
        directory = tmp_path / "ckpt"
        directory.mkdir()
        (directory / "adapter_model.safetensors").write_bytes(b"x" * 16)
        path = write_provenance(
            directory,
            plan=plan,
            envelope=CanaryEnvelope(max_samples=2),
            mode=CanaryMode.ACE_STEP.value,
            execution_location=ExecutionLocation.LOCAL.value,
            execution_device=ComputeDevice.MPS.value,
            resolved_precision=Precision.BF16.value,
            dataset_kind="SYNTHETIC",
            steps=1,
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["plan_digest"] == plan.digest()
        assert payload["execution_device"] == ComputeDevice.MPS.value
        assert payload["resolved_precision"] == Precision.BF16.value
        assert payload["optimizer"] == plan.config.optimizer_type
        assert payload["dataset"]["curated_manifest_sha256"]
        assert payload["ace_step_commit"] == plan.config.ace_step_commit
        assert payload["optimizer_steps"] == 1
        assert "must never be promoted" in payload["warning"]

    def test_provenance_from_a_different_plan_is_a_problem(self, tmp_path: Path):
        plan = a_plan(device=ComputeDevice.CPU.value)
        directory = tmp_path / "ckpt"
        directory.mkdir()
        (directory / "adapter_model.safetensors").write_bytes(b"x" * 16)
        write_provenance(
            directory,
            plan=plan,
            envelope=CanaryEnvelope(max_samples=2),
            mode=CanaryMode.ACE_STEP.value,
            execution_location=ExecutionLocation.LOCAL.value,
            execution_device=ComputeDevice.CPU.value,
            resolved_precision="fp32",
            dataset_kind="SYNTHETIC",
            steps=1,
        )
        integrity = inspect_checkpoint(directory, expected_plan_digest="f" * 64)
        assert any("provenance cites plan" in problem for problem in integrity.problems)

    def test_reopening_is_unknown_without_an_interpreter(self, tmp_path: Path):
        """Nobody tried is not the same as it worked."""
        directory = tmp_path / "ckpt"
        directory.mkdir()
        (directory / "adapter_model.safetensors").write_bytes(b"x" * 16)
        integrity = inspect_checkpoint(directory, python_executable=None)
        assert integrity.reopened is None
        assert not integrity.ok

    def test_the_latest_checkpoint_is_read_from_the_trainers_own_layout(self, tmp_path: Path):
        output = tmp_path / "output"
        for epoch in (1, 2):
            (output / "checkpoints" / f"epoch_{epoch}_loss_1.0000").mkdir(parents=True)
        assert latest_checkpoint(output) is not None
        assert latest_checkpoint(output).name == "epoch_2_loss_1.0000"

    def test_no_checkpoint_directory_is_none(self, tmp_path: Path):
        assert latest_checkpoint(tmp_path / "nothing") is None


# ── the orchestration canary ─────────────────────────────────────────


class TestOrchestrationCanary:
    def test_it_compiles_a_bounded_command_and_starts_nothing(self):
        result = orchestration_canary(
            a_plan(device=ComputeDevice.MPS.value),
            CanaryEnvelope(max_samples=2),
            trainer_root="/opt/ace-step",
            python_executable="/opt/py",
            execution_location=ExecutionLocation.LOCAL.value,
        )
        assert result.status == CanaryStatus.PASSED.value
        assert result.mode == CanaryMode.ORCHESTRATION.value
        assert result.steps == 0
        assert "Nothing was executed" in result.detail
        assert "--epochs" in result.command
        assert result.command[result.command.index("--epochs") + 1] == str(CANARY_MAX_EPOCHS)

    def test_the_command_carries_the_flag_the_trainer_needs_to_start(self):
        """Without `--yes` the trainer exits 0 having trained nothing."""
        result = orchestration_canary(
            a_plan(device=ComputeDevice.CPU.value),
            CanaryEnvelope(max_samples=1),
            trainer_root="/opt/ace-step",
            execution_location=ExecutionLocation.LOCAL.value,
        )
        assert result.command.index("--yes") < result.command.index("fixed")

    def test_it_records_the_ceilings_it_ran_under(self):
        result = orchestration_canary(
            a_plan(device=ComputeDevice.CPU.value),
            CanaryEnvelope(max_samples=1),
            trainer_root="/opt/ace-step",
            execution_location=ExecutionLocation.LOCAL.value,
        )
        ceilings = result.to_dict()["envelope"]["ceilings"]
        assert ceilings["optimizer_steps"] == CANARY_MAX_OPTIMIZER_STEPS
        assert ceilings["samples"] == CANARY_MAX_SAMPLES


# ── the trainer's own path constraint ────────────────────────────────


class TestTrainerPathSafety:
    def test_within_answers_the_question_the_trainer_asks(self, tmp_path: Path):
        root = tmp_path / "trainer"
        (root / "inside").mkdir(parents=True)
        assert within(root / "inside", root)
        assert not within(tmp_path / "outside", root)

    def test_a_dataset_outside_the_trainer_root_is_blocked_before_loading(self, tmp_path: Path):
        """The trainer refuses this *after* loading a 2.4B model."""
        trainer = tmp_path / "trainer"
        (trainer / "checkpoints").mkdir(parents=True)
        (trainer / "train.py").write_text("", encoding="utf-8")
        interpreter = tmp_path / "python"
        interpreter.write_text("", encoding="utf-8")
        outside = _fixture_dataset(tmp_path / "elsewhere")

        result = ace_step_canary(
            a_plan(device=ComputeDevice.CPU.value),
            CanaryEnvelope(max_samples=2),
            trainer_root=trainer,
            python_executable=interpreter,
            model_dir=trainer / "checkpoints",
            workspace=tmp_path / "workspace",
            execution_location=ExecutionLocation.LOCAL.value,
            dataset_dir=outside,
        )
        assert result.status == CanaryStatus.BLOCKED.value
        assert "outside the trainer's working directory" in result.detail

    def test_the_default_workspace_is_inside_the_trainer_root(self, tmp_path: Path):
        assert within(default_workspace(tmp_path, "x").parent.parent, tmp_path)


class TestCanaryBlockedReasons:
    def test_a_missing_trainer_is_blocked_not_failed(self, tmp_path: Path):
        result = ace_step_canary(
            a_plan(device=ComputeDevice.CPU.value),
            CanaryEnvelope(max_samples=1),
            trainer_root=tmp_path / "absent",
            python_executable=tmp_path / "absent-python",
            model_dir=tmp_path / "absent-model",
            workspace=tmp_path / "workspace",
            execution_location=ExecutionLocation.LOCAL.value,
        )
        assert result.status == CanaryStatus.BLOCKED.value
        assert "no trainer is installed" in result.detail

    def test_a_missing_model_root_is_blocked_and_nothing_is_downloaded(self, tmp_path: Path):
        trainer = tmp_path / "trainer"
        trainer.mkdir()
        (trainer / "train.py").write_text("", encoding="utf-8")
        interpreter = tmp_path / "python"
        interpreter.write_text("", encoding="utf-8")
        result = ace_step_canary(
            a_plan(device=ComputeDevice.CPU.value),
            CanaryEnvelope(max_samples=1),
            trainer_root=trainer,
            python_executable=interpreter,
            model_dir=tmp_path / "absent-model",
            workspace=tmp_path / "workspace",
            execution_location=ExecutionLocation.LOCAL.value,
        )
        assert result.status == CanaryStatus.BLOCKED.value
        assert "never downloads" in result.detail


# ── 25. the real trainer, bounded ────────────────────────────────────


@needs_trainer
class TestRealTrainerCanary:
    """Runs the installed ACE-Step trainer. Two samples, one epoch.

    Skipped where there is no trainer rather than replaced by something
    that looks like one. A synthetic stand-in passing here would tell us
    nothing about ACE-Step, which is the only thing these tests exist to
    establish.
    """

    def test_bf16_on_this_machine_trains_and_writes_a_reopenable_checkpoint(self, workspace):
        plan = a_plan(
            device=ComputeDevice.MPS.value
            if os.uname().sysname == "Darwin"
            else ComputeDevice.CPU.value,
            config=TrainingConfig(epochs=1, rank=4, alpha=8, precision=Precision.BF16.value),
        )
        envelope = CanaryEnvelope(max_samples=2, resume=True, wall_clock_seconds=1500.0)
        result = ace_step_canary(
            plan,
            envelope,
            trainer_root=TRAINER_ROOT,
            python_executable=TRAINER_PYTHON,
            model_dir=MODEL_DIR,
            workspace=workspace,
            execution_location=ExecutionLocation.LOCAL.value,
            resolved_precision=Precision.BF16.value,
        )
        assert result.status == CanaryStatus.PASSED.value, result.detail
        assert result.dataset_kind == "SYNTHETIC"
        assert result.checkpoint is not None
        assert result.checkpoint["ok"]
        assert result.checkpoint["reopened"]
        assert result.checkpoint["non_zero_parameters"] > 0
        assert result.checkpoint["provenance_plan_digest"] == result.plan_digest
        # Resume is only a claim if the step counter moved.
        assert result.resume is not None and result.resume["ok"]
        assert result.resume["second_step"] > result.resume["first_step"]
        assert (result.steps or 0) <= envelope.max_optimizer_steps
