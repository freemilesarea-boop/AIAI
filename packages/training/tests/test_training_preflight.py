"""The execution-readiness gate: what it clears and what it refuses.

Every CUDA case here runs against a **fixture** machine — a literal
capability describing hardware nobody in this project owns. That is
enough to test the logic and is not evidence about NVIDIA hardware, and
the distinction is stated in the fixture module as well as here.

The tests that matter most are the refusals. A preflight that says READY
too easily is worse than none, because the whole point of it is to be
the thing an operator trusts before spending a GPU day.
"""

from __future__ import annotations

import dataclasses

import pytest
from preflight_fixtures import (
    MANIFEST_DIGEST,
    MEASURED_AT,
    NO_BF16,
    a_plan,
    a_request,
    a_worker,
    apple_capability,
    cpu_capability,
    cuda_capability,
    good_dataset,
    good_remote,
    good_storage,
    good_trainer,
)

from luber_hardware import ComputeDevice, ExecutionLocation
from luber_training.capacity import (
    CapacityEvidence,
    CapacityReport,
    EvidenceSource,
    capacity_report,
    required_disk_evidence,
    training_memory_requirement,
)
from luber_training.config import Optimizer, Precision, TrainingConfig
from luber_training.entities import WorkerClass
from luber_training.gates import GateReport, GateResult
from luber_training.preflight import (
    UNTRAINABLE_PRECISION,
    BlockingReason,
    CheckStatus,
    PreflightIntent,
    PreflightStatus,
    evaluate,
)


def _reasons(result) -> set[str]:
    """The taxonomy entries this result cites, whatever the wording."""
    return {
        check.reason
        for check in result.checks
        if check.reason is not None and (check.blocks or check.unverifies)
    }


def _check(result, name):
    return next(check for check in result.checks if check.name == name)


# ── 1-3. the three devices, each cleared on its own machine ──────────


class TestReadyCases:
    def test_cpu_preflight_is_ready(self):
        result = evaluate(a_request(plan=a_plan(device=ComputeDevice.CPU.value)))
        assert result.status == PreflightStatus.READY.value, result.blocking_reasons + list(
            result.unverified
        )
        assert result.execution_device == ComputeDevice.CPU.value
        assert result.torch_device == "cpu"

    def test_mps_fixture_preflight_is_ready(self):
        """A fixture Apple machine, at a precision the trainer can train in."""
        plan = a_plan(
            device=ComputeDevice.MPS.value,
            config=TrainingConfig(epochs=1, precision=Precision.BF16.value),
        )
        result = evaluate(a_request(plan=plan, capability=apple_capability()))
        assert result.status == PreflightStatus.READY.value, result.blocking_reasons
        assert result.execution_device == ComputeDevice.MPS.value
        assert result.resolved_precision == Precision.BF16.value

    def test_cuda_fixture_preflight_is_ready(self):
        """Fixture-tested CUDA logic. No NVIDIA hardware was involved."""
        plan = a_plan(device=ComputeDevice.CUDA.value)
        result = evaluate(
            a_request(
                plan=plan,
                capability=cuda_capability(),
                location=ExecutionLocation.REMOTE.value,
                worker=a_worker(),
            )
        )
        assert result.status == PreflightStatus.READY.value, result.blocking_reasons
        assert result.execution_device == ComputeDevice.CUDA.value


# ── 4-6. device, precision and optimizer refusals ────────────────────


class TestHardwareRefusals:
    def test_missing_requested_device_blocks(self):
        """CUDA on a Mac is BLOCKED, never quietly moved to MPS."""
        plan = a_plan(device=ComputeDevice.CUDA.value)
        result = evaluate(a_request(plan=plan, capability=apple_capability()))
        assert result.status == PreflightStatus.BLOCKED.value
        assert BlockingReason.DEVICE_UNAVAILABLE.value in _reasons(result)

    def test_unsupported_precision_blocks(self):
        plan = a_plan(
            device=ComputeDevice.CUDA.value,
            config=TrainingConfig(epochs=1, precision=Precision.BF16.value),
        )
        result = evaluate(
            a_request(
                plan=plan,
                capability=cuda_capability(precision=NO_BF16),
                location=ExecutionLocation.REMOTE.value,
                worker=a_worker(),
            )
        )
        assert result.status == PreflightStatus.BLOCKED.value
        assert BlockingReason.PRECISION_UNSUPPORTED.value in _reasons(result)

    def test_unsupported_optimizer_blocks(self):
        """`adamw8bit` off CUDA is a refusal, not a substitution.

        ACE-Step catches the bitsandbytes ImportError and uses AdamW,
        which would train something other than what the plan records.
        """
        plan = a_plan(
            device=ComputeDevice.MPS.value,
            config=TrainingConfig(
                epochs=1,
                precision=Precision.BF16.value,
                optimizer_type=Optimizer.ADAMW_8BIT.value,
            ),
        )
        result = evaluate(a_request(plan=plan, capability=apple_capability()))
        assert result.status == PreflightStatus.BLOCKED.value
        assert BlockingReason.OPTIMIZER_UNSUPPORTED.value in _reasons(result)

    def test_fp16_on_mps_blocks_because_the_trainer_cannot_train_in_it(self):
        """Measured, not inferred: fp16 on MPS dies at the first clip.

        The hardware holds fp16 tensors fine — Phase 32 measured that.
        The trainer loads the model in fp16 and drives Fabric at
        `16-mixed`, and torch refuses to unscale fp16 gradients.
        """
        assert (ComputeDevice.MPS.value, Precision.FP16.value) in UNTRAINABLE_PRECISION
        plan = a_plan(
            device=ComputeDevice.MPS.value,
            config=TrainingConfig(epochs=1, precision=Precision.FP16.value),
        )
        result = evaluate(a_request(plan=plan, capability=apple_capability()))
        assert result.status == PreflightStatus.BLOCKED.value
        assert _check(result, "hardware.trainer_precision").status == CheckStatus.FAIL.value

    def test_auto_on_mps_blocks_for_the_same_reason(self):
        """`auto` resolves to fp16 on MPS, so it hits the same wall."""
        plan = a_plan(
            device=ComputeDevice.MPS.value,
            config=TrainingConfig(epochs=1, precision=Precision.AUTO.value),
        )
        result = evaluate(a_request(plan=plan, capability=apple_capability()))
        assert result.status == PreflightStatus.BLOCKED.value
        assert BlockingReason.PRECISION_UNSUPPORTED.value in _reasons(result)


# ── 7-8. the trainer itself ──────────────────────────────────────────


class TestTrainerRefusals:
    def test_missing_trainer_blocks(self):
        result = evaluate(
            a_request(trainer=good_trainer(trainer_root_present=False, entrypoint_present=False))
        )
        assert result.status == PreflightStatus.BLOCKED.value
        assert BlockingReason.TRAINER_UNAVAILABLE.value in _reasons(result)

    def test_missing_optimizer_dependency_blocks(self):
        plan = a_plan(
            device=ComputeDevice.CUDA.value,
            config=TrainingConfig(epochs=1, optimizer_type=Optimizer.PRODIGY.value),
        )
        result = evaluate(
            a_request(
                plan=plan,
                capability=cuda_capability(),
                location=ExecutionLocation.REMOTE.value,
                worker=a_worker(),
                trainer=good_trainer(missing_packages=("prodigyopt",)),
            )
        )
        assert result.status == PreflightStatus.BLOCKED.value
        assert BlockingReason.DEPENDENCY_MISSING.value in _reasons(result)

    def test_a_command_the_installed_trainer_rejects_blocks(self):
        result = evaluate(
            a_request(
                trainer=good_trainer(
                    command_accepted=False,
                    command_detail="unrecognized arguments: --yes",
                )
            )
        )
        assert result.status == PreflightStatus.BLOCKED.value
        assert BlockingReason.TRAINER_UNAVAILABLE.value in _reasons(result)

    def test_a_different_ace_step_revision_blocks(self):
        result = evaluate(a_request(trainer=good_trainer(observed_ace_step_commit="0" * 40)))
        assert result.status == PreflightStatus.BLOCKED.value
        assert _check(result, "trainer.runtime_identity").status == CheckStatus.FAIL.value


# ── 9-13. plan and data ──────────────────────────────────────────────


class TestPlanAndData:
    def test_an_unsupported_plan_schema_blocks(self):
        plan = a_plan(device=ComputeDevice.CPU.value, schema_version="luber-training-plan/99")
        result = evaluate(a_request(plan=plan))
        assert result.status == PreflightStatus.BLOCKED.value
        assert BlockingReason.PLAN_INVALID.value in _reasons(result)

    def test_a_plan_with_no_device_is_unverified_not_ready(self):
        """A plan that never named a device has not been placed."""
        result = evaluate(a_request(plan=a_plan(device=None)))
        assert result.status != PreflightStatus.READY.value
        assert _check(result, "plan.execution_device").status == CheckStatus.UNKNOWN.value

    def test_a_self_contradicting_plan_blocks(self):
        plan = a_plan(device=ComputeDevice.MPS.value)
        broken = dataclasses.replace(plan.requirements, requires_cuda=True)
        result = evaluate(
            a_request(
                plan=dataclasses.replace(plan, requirements=broken),
                capability=apple_capability(),
            )
        )
        assert result.status == PreflightStatus.BLOCKED.value
        assert _check(result, "plan.coherence").status == CheckStatus.FAIL.value

    def test_dataset_not_ready_blocks(self):
        result = evaluate(a_request(dataset=good_dataset(eligible_sample_count=0)))
        assert result.status == PreflightStatus.BLOCKED.value
        assert BlockingReason.DATASET_NOT_READY.value in _reasons(result)

    def test_a_rights_failure_blocks(self):
        report = GateReport(
            results=[
                GateResult(name="dataset_lock", passed=True),
                GateResult(name="curation_lock", passed=True),
                GateResult(
                    name="rights",
                    passed=False,
                    detail="2 of 4 selected tracks are not permitted for training",
                    failure_code="RIGHTS_GATE_FAILED",
                ),
                GateResult(name="evaluation_leakage", passed=True),
                GateResult(name="self_generated", passed=True),
            ]
        )
        result = evaluate(a_request(gate_report=report))
        assert result.status == PreflightStatus.BLOCKED.value
        assert BlockingReason.RIGHTS_BLOCKED.value in _reasons(result)

    def test_no_gate_report_is_unverified_never_ready(self):
        """A run whose rights nobody established cannot be READY."""
        result = evaluate(a_request(with_gates=False))
        assert result.status != PreflightStatus.READY.value

    def test_evaluation_only_material_blocks(self):
        result = evaluate(a_request(dataset=good_dataset(evaluation_only_count=1)))
        assert result.status == PreflightStatus.BLOCKED.value
        assert BlockingReason.EVALUATION_LEAKAGE.value in _reasons(result)

    def test_manifest_drift_blocks(self):
        result = evaluate(a_request(dataset=good_dataset(observed_manifest_sha256="d" * 64)))
        assert result.status == PreflightStatus.BLOCKED.value
        assert BlockingReason.MANIFEST_DRIFT.value in _reasons(result)
        assert MANIFEST_DIGEST[:16] in _check(result, "dataset.drift").detail


# ── 14-16. remote workers ────────────────────────────────────────────


class TestRemote:
    def _remote(self, **kwargs):
        return a_request(
            plan=a_plan(device=ComputeDevice.CUDA.value),
            capability=cuda_capability(),
            location=ExecutionLocation.REMOTE.value,
            **kwargs,
        )

    def test_a_stale_capability_report_blocks(self):
        result = evaluate(
            self._remote(worker=a_worker(), remote=good_remote(capability_age_seconds=999_999.0))
        )
        assert result.status == PreflightStatus.BLOCKED.value
        assert BlockingReason.WORKER_STALE.value in _reasons(result)

    def test_a_worker_that_never_reported_a_time_is_unverified(self):
        """Policy: an unreadable timestamp is UNKNOWN, not stale.

        The two need different words. Stale means we know it is old;
        this means we do not know how old it is, and neither is a pass.
        """
        result = evaluate(
            self._remote(worker=a_worker(), remote=good_remote(capability_age_seconds=None))
        )
        assert result.status == PreflightStatus.UNVERIFIED.value
        assert _check(result, "remote.capability_freshness").status == CheckStatus.UNKNOWN.value

    def test_a_missing_remote_worker_blocks(self):
        result = evaluate(self._remote(worker=None))
        assert result.status == PreflightStatus.BLOCKED.value
        assert BlockingReason.WORKER_UNAVAILABLE.value in _reasons(result)

    def test_a_development_only_worker_cannot_take_a_cuda_plan(self):
        result = evaluate(
            self._remote(worker=a_worker(worker_class=WorkerClass.DEVELOPMENT_ONLY.value))
        )
        assert result.status == PreflightStatus.BLOCKED.value
        assert BlockingReason.WORKER_UNAVAILABLE.value in _reasons(result)

    def test_an_unreachable_worker_blocks(self):
        result = evaluate(self._remote(worker=a_worker(), remote=good_remote(reachable=False)))
        assert result.status == PreflightStatus.BLOCKED.value
        assert BlockingReason.REMOTE_UNREACHABLE.value in _reasons(result)

    def test_remote_checks_do_not_apply_to_a_local_run(self):
        result = evaluate(a_request())
        assert _check(result, "remote.transport").status == CheckStatus.NOT_APPLICABLE.value


# ── 17-19. storage and capacity ──────────────────────────────────────


class TestStorageAndCapacity:
    def test_unwritable_storage_blocks(self):
        result = evaluate(a_request(storage=good_storage(output_writable=False)))
        assert result.status == PreflightStatus.BLOCKED.value
        assert BlockingReason.STORAGE_UNAVAILABLE.value in _reasons(result)

    def test_a_dataset_outside_the_trainer_root_blocks(self):
        """The trainer refuses it, after loading the model."""
        result = evaluate(a_request(storage=good_storage(dataset_within_trainer_root=False)))
        assert result.status == PreflightStatus.BLOCKED.value
        assert _check(result, "storage.trainer_path_safety").status == CheckStatus.FAIL.value

    def test_insufficient_known_disk_blocks(self):
        capacity = CapacityReport(
            device=ComputeDevice.CPU.value,
            evidence=(required_disk_evidence(200 * 1024 * 1024, checkpoints_expected=4),),
        )
        result = evaluate(a_request(storage=good_storage(free_disk_mb=10), capacity=capacity))
        assert result.status == PreflightStatus.BLOCKED.value
        assert BlockingReason.INSUFFICIENT_DISK.value in _reasons(result)

    def test_an_unknown_disk_requirement_does_not_block_twice(self):
        """One gap is reported once, as capacity evidence."""
        capacity = capacity_report(cpu_capability(), device=ComputeDevice.CPU.value, free_disk_mb=5)
        result = evaluate(a_request(capacity=capacity, storage=good_storage(free_disk_mb=5)))
        assert _check(result, "storage.free_disk").status == CheckStatus.PASS.value
        assert not _check(result, "storage.free_disk").mandatory

    def test_unknown_capacity_leaves_full_training_unverified(self):
        """The honest state of this project: nobody measured it."""
        capacity = capacity_report(
            cpu_capability(), device=ComputeDevice.CPU.value, free_disk_mb=200_000
        )
        result = evaluate(a_request(intent=PreflightIntent.FULL_TRAINING.value, capacity=capacity))
        assert result.status == PreflightStatus.UNVERIFIED.value
        assert BlockingReason.CAPACITY_UNVERIFIED.value in _reasons(result)
        assert result.capacity_status == EvidenceSource.UNKNOWN.value

    def test_a_canary_may_be_ready_with_the_same_unknown(self):
        """A bounded canary does not need a production memory figure."""
        capacity = capacity_report(
            cpu_capability(), device=ComputeDevice.CPU.value, free_disk_mb=200_000
        )
        result = evaluate(a_request(intent=PreflightIntent.CANARY.value, capacity=capacity))
        assert result.status == PreflightStatus.READY.value

    def test_unified_memory_is_not_reported_as_vram(self):
        report = capacity_report(apple_capability(), device=ComputeDevice.MPS.value)
        evidence = report.by_name("device_memory_mb")
        assert evidence is not None
        assert evidence.unified_memory is True
        assert "not dedicated VRAM" in evidence.detail

    def test_cuda_memory_is_not_marked_unified(self):
        report = capacity_report(cuda_capability(), device=ComputeDevice.CUDA.value)
        evidence = report.by_name("device_memory_mb")
        assert evidence is not None
        assert evidence.unified_memory is False

    def test_the_training_requirement_is_unknown_everywhere(self):
        assert training_memory_requirement().source == EvidenceSource.UNKNOWN.value

    def test_evidence_sources_never_mix(self):
        report = capacity_report(
            apple_capability(),
            device=ComputeDevice.MPS.value,
            free_disk_mb=1000,
            checkpoint_bytes=5 * 1024 * 1024,
        )
        sources = report.sources()
        assert "device_memory_mb" in sources[EvidenceSource.MEASURED.value]
        assert "required_disk_mb" in sources[EvidenceSource.ESTIMATED.value]
        assert "training_memory_requirement_mb" in sources[EvidenceSource.UNKNOWN.value]

    def test_a_measured_figure_must_have_a_value(self):
        with pytest.raises(ValueError, match="must have a value"):
            CapacityEvidence(name="x", source=EvidenceSource.MEASURED.value)

    def test_an_estimate_must_state_its_derivation(self):
        with pytest.raises(ValueError, match="derivation"):
            CapacityEvidence(name="x", source=EvidenceSource.ESTIMATED.value, value_mb=1)


# ── 26-30. identity, independence and determinism ────────────────────


class TestIdentityAndDeterminism:
    def test_the_plan_digest_is_unchanged_by_runtime_measurement(self):
        """Hardware facts belong to the preflight, never to the plan."""
        plan = a_plan(device=ComputeDevice.MPS.value)
        before = plan.digest()
        first = evaluate(a_request(plan=plan, capability=apple_capability()))
        second = evaluate(
            a_request(plan=plan, capability=apple_capability(memory_mb=131072, label="other"))
        )
        assert plan.digest() == before
        assert first.plan_digest == second.plan_digest == before

    def test_mps_and_cuda_remain_distinct_plans(self):
        assert (
            a_plan(device=ComputeDevice.MPS.value).digest()
            != a_plan(device=ComputeDevice.CUDA.value).digest()
        )

    def test_location_and_device_are_independent_axes(self):
        """The same device, in two locations, is the same plan."""
        plan = a_plan(device=ComputeDevice.CPU.value)
        local = evaluate(a_request(plan=plan, location=ExecutionLocation.LOCAL.value))
        remote = evaluate(
            a_request(
                plan=plan,
                location=ExecutionLocation.REMOTE.value,
                worker=a_worker(cuda=None),
            )
        )
        assert local.plan_digest == remote.plan_digest
        assert local.execution_location != remote.execution_location
        assert local.execution_device == remote.execution_device

    def test_no_silent_fallback_anywhere_in_a_refusal(self):
        """A refused CUDA request never comes back as another device."""
        plan = a_plan(device=ComputeDevice.CUDA.value)
        result = evaluate(a_request(plan=plan, capability=apple_capability()))
        assert result.execution_device == ComputeDevice.CUDA.value
        assert result.status == PreflightStatus.BLOCKED.value

    def test_identical_evidence_produces_an_identical_verdict(self):
        first = evaluate(a_request())
        second = evaluate(a_request())
        assert first.digest() == second.digest()
        assert first.measured_at == second.measured_at == MEASURED_AT

    def test_the_digest_ignores_when_it_was_taken(self):
        first = evaluate(a_request())
        later = dataclasses.replace(first, measured_at="2027-01-01T00:00:00+00:00")
        assert first.digest() == later.digest()

    def test_unverified_is_not_ready(self):
        result = evaluate(a_request(trainer=good_trainer(torch_importable=None)))
        assert result.status == PreflightStatus.UNVERIFIED.value
        assert not result.ready

    def test_a_blocked_result_names_a_machine_readable_reason(self):
        result = evaluate(a_request(storage=good_storage(checkpoint_writable=False)))
        assert result.blocking_reasons
        assert all(
            reason.split(":")[0] in {item.value for item in BlockingReason}
            for reason in result.blocking_reasons
        )

    def test_the_result_serialises_and_carries_its_note(self):
        payload = evaluate(a_request()).to_dict()
        assert payload["status"] == PreflightStatus.READY.value
        assert "UNVERIFIED is not READY" in payload["note"]
        assert isinstance(payload["checks"], list)
