"""Where a workload lands, and the job that must not move.

The expensive mistakes here are asymmetric. Sending preprocessing to a
rented GPU wastes money and somebody notices the bill. Sending heavy
training to a Mac because the GPU box was offline produces a *model* —
it completes, it looks like every other checkpoint, and nothing records
that it trained somewhere nobody chose. That asymmetry is why most of
this file is about refusals.
"""

from __future__ import annotations

from hardware_fixtures import gpu_target, mac_target

from luber_hardware import (
    ComputeDevice,
    ComputePreference,
    ExecutionLocation,
    ExecutionTarget,
    PlacementOutcome,
    PlacementPolicy,
    PlacementRequest,
    WorkloadClass,
    place,
    planned_cuda_worker,
    planned_mac_mini_24gb,
)

# ── the topology this project is planning ────────────────────────────


def test_heavy_training_prefers_the_remote_gpu():
    decision = place(
        PlacementRequest(workload=WorkloadClass.HEAVY_TRAINING.value),
        [mac_target(), gpu_target()],
    )

    assert decision.placed
    assert decision.execution_location == ExecutionLocation.REMOTE.value
    assert decision.compute_device == ComputeDevice.CUDA.value
    assert decision.worker_id == "worker-1", "the decision names the machine Phase 27 will call"


def test_a_cuda_job_with_no_cuda_worker_is_blocked_not_moved_to_the_mac():
    """The assertion this phase exists for.

    A Mac is present, it can train, and it is still the wrong answer.
    Running here would produce a model from a plan that asked for
    something else.
    """
    decision = place(PlacementRequest(workload=WorkloadClass.HEAVY_TRAINING.value), [mac_target()])

    assert decision.outcome == PlacementOutcome.BLOCKED_NO_COMPATIBLE_EXECUTION_TARGET.value
    assert decision.compute_device is None
    assert "allow_local_fallback is false" in decision.reason


def test_local_fallback_happens_only_when_somebody_asked_for_it():
    """Explicit, per request, and visible in the decision."""
    decision = place(
        PlacementRequest(workload=WorkloadClass.HEAVY_TRAINING.value, allow_local_fallback=True),
        [mac_target()],
    )

    assert decision.placed
    assert decision.execution_location == ExecutionLocation.LOCAL.value
    assert decision.compute_device == ComputeDevice.MPS.value


def test_cuda_required_refuses_everything_else():
    decision = place(
        PlacementRequest(
            workload=WorkloadClass.LIGHT_FINE_TUNE.value,
            policy=PlacementPolicy.CUDA_REQUIRED.value,
        ),
        [mac_target()],
    )

    assert decision.outcome == PlacementOutcome.BLOCKED_NO_COMPATIBLE_EXECUTION_TARGET.value


# ── work that should stay off the GPU ────────────────────────────────


def test_preprocessing_stays_local_and_on_the_cpu():
    """`luber_dataset` is numpy and ffmpeg. It has no tensors in it.

    Sending it to a rented GPU would spend GPU-hours on work a CPU does
    at the same speed, and the only reason anybody would is that a GPU
    was available.
    """
    decision = place(
        PlacementRequest(workload=WorkloadClass.PREPROCESS.value),
        [mac_target(), gpu_target()],
    )

    assert decision.placed
    assert decision.execution_location == ExecutionLocation.LOCAL.value
    assert decision.compute_device == ComputeDevice.CPU.value


def test_evaluation_does_not_require_cuda():
    """Phase 26 drives a server over HTTP and computes in pure Python.

    Whichever device *that server* uses is the server's business.
    """
    decision = place(PlacementRequest(workload=WorkloadClass.EVALUATION.value), [mac_target()])

    assert decision.placed
    assert decision.compute_device == ComputeDevice.CPU.value


def test_light_fine_tuning_may_use_apple_silicon():
    decision = place(PlacementRequest(workload=WorkloadClass.LIGHT_FINE_TUNE.value), [mac_target()])

    assert decision.placed
    assert decision.compute_device == ComputeDevice.MPS.value
    assert decision.precision == "fp16", "the trainer's own MPS default"


def test_checkpoint_validation_runs_wherever_it_is_asked():
    """Loading a checkpoint needs a device that holds tensors, which is
    every device. This is what makes 'train remotely, inspect locally' a
    topology rather than a wish."""
    decision = place(
        PlacementRequest(workload=WorkloadClass.CHECKPOINT_VALIDATION.value), [mac_target()]
    )

    assert decision.placed


# ── the decision has to be readable later ────────────────────────────


def test_a_decision_carries_its_reasoning_and_its_gaps():
    decision = place(
        PlacementRequest(workload=WorkloadClass.HEAVY_TRAINING.value),
        [mac_target(), gpu_target()],
    )

    payload = decision.to_dict()
    assert payload["execution_placement_policy_version"]
    assert payload["capability_digest"]
    assert payload["memory"]["verdict"] == "UNKNOWN"
    assert any("no memory requirement has been measured" in item for item in decision.unknowns)


def test_a_refusal_names_every_target_it_looked_at():
    decision = place(
        PlacementRequest(workload=WorkloadClass.HEAVY_TRAINING.value),
        [mac_target(), ExecutionTarget("mac-2", mac_target().capability)],
    )

    assert set(decision.considered) == {"mac", "mac-2"}


def test_no_targets_is_its_own_answer():
    decision = place(PlacementRequest(), [])

    assert decision.outcome == PlacementOutcome.BLOCKED_NO_TARGETS.value


# ── planned profiles never beat real hardware ────────────────────────


def test_a_planned_profile_loses_to_a_machine_that_exists():
    """Planning profiles answer "could this work". They must never win a
    placement away from hardware somebody has actually probed."""
    planned = ExecutionTarget("planned-mini", planned_mac_mini_24gb())
    decision = place(
        PlacementRequest(workload=WorkloadClass.LIGHT_FINE_TUNE.value),
        [planned, mac_target()],
    )

    assert decision.target_name == "mac"
    assert not decision.planned_target


def test_a_planned_target_is_flagged_when_it_is_the_only_one():
    decision = place(
        PlacementRequest(workload=WorkloadClass.LIGHT_FINE_TUNE.value),
        [ExecutionTarget("planned-mini", planned_mac_mini_24gb())],
    )

    assert decision.placed
    assert decision.planned_target
    assert any("planned profile" in item for item in decision.unknowns)


def test_a_planned_cuda_worker_carries_no_invented_vram():
    """No GPU model, no memory figure, unless a caller stated one."""
    capability = planned_cuda_worker()

    assert capability.cuda_device_memory_mb is None
    assert "H100" not in (capability.cuda_device_name or "")
    assert capability.planned


# ── determinism ──────────────────────────────────────────────────────


def test_placement_is_stable_across_target_ordering():
    """Same machines, same answer, whatever order they arrive in."""
    targets = [mac_target(), gpu_target()]
    first = place(PlacementRequest(workload=WorkloadClass.HEAVY_TRAINING.value), targets)
    second = place(
        PlacementRequest(workload=WorkloadClass.HEAVY_TRAINING.value), list(reversed(targets))
    )

    assert first.target_name == second.target_name
    assert first.compute_device == second.compute_device


def test_an_explicit_preference_survives_placement():
    decision = place(
        PlacementRequest(
            workload=WorkloadClass.LIGHT_FINE_TUNE.value,
            preference=ComputePreference.CPU.value,
        ),
        [mac_target()],
    )

    assert decision.compute_device == ComputeDevice.CPU.value
