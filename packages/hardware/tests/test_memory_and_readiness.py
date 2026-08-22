"""Memory nobody has measured, and a readiness view that cannot lie.

Two failure modes are guarded here. A memory model that turns "we have
not measured this" into a pass, and a readiness view that disagrees with
the scheduler — an operator plans against readiness, so a row saying a
Mac can take heavy training would be planning material contradicted by
every actual attempt.
"""

from __future__ import annotations

from hardware_fixtures import apple_machine, gpu_target, mac_target, torchless_machine

from luber_hardware import (
    DEFAULT_HEADROOM_FRACTION,
    LOCAL_TRAINING_CONCURRENCY,
    ComputeDevice,
    ExecutionLocation,
    ExecutionTarget,
    MemoryVerdict,
    TargetStatus,
    WorkloadClass,
    assess,
    budget_for,
    readiness,
)

# ── memory ───────────────────────────────────────────────────────────


def test_no_measurement_is_unknown_and_unknown_does_not_pass():
    """Nothing in this project has measured what the 2B DiT needs.

    Writing "18.4 GB" would invent the number every scheduling decision
    depends on. UNKNOWN is the honest answer and it is carried on the
    decision rather than collapsed into a boolean.
    """
    verdict = assess(budget_for(24576))

    assert verdict.verdict == MemoryVerdict.UNKNOWN.value
    assert not verdict.blocks, "unknown must not block every placement"
    assert "This is not a pass" in verdict.reason


def test_a_machine_never_gets_to_use_all_of_its_memory():
    """24 GB of unified memory is not 24 GB of PyTorch.

    Apple shares it with the GPU and, on the planned topology, with the
    API, Postgres and Redis at the same time.
    """
    budget = budget_for(24576)

    assert budget.usable_mb() is not None
    assert budget.usable_mb() < 24576
    assert budget.reserved_mb() == int(24576 * DEFAULT_HEADROOM_FRACTION)


def test_a_control_plane_machine_reserves_more():
    shared = budget_for(24576, shared_with_control_plane=True)
    dedicated = budget_for(24576)

    assert shared.usable_mb() < dedicated.usable_mb()
    assert "control plane" in shared.note


def test_a_small_machine_reserves_the_floor_rather_than_a_percentage():
    """A percentage of a small machine reserves too little to run an OS."""
    budget = budget_for(4096)

    assert budget.reserved_mb() == 4096
    assert budget.usable_mb() == 0


def test_an_estimate_that_fits_is_known_safe_and_one_that_does_not_is_blocked():
    budget = budget_for(24576)
    usable = budget.usable_mb()

    assert assess(budget, usable - 1).verdict == MemoryVerdict.KNOWN_SAFE.value
    too_large = assess(budget, usable + 1)
    assert too_large.verdict == MemoryVerdict.LIKELY_TOO_LARGE.value
    assert too_large.blocks


def test_an_unreported_machine_size_is_unknown_not_infinite():
    assert assess(budget_for(None), 8192).verdict == MemoryVerdict.UNKNOWN.value


def test_local_training_concurrency_is_bounded_at_one():
    """The Mac mini is the 24/7 control plane before it is a trainer."""
    assert LOCAL_TRAINING_CONCURRENCY == 1


# ── readiness ────────────────────────────────────────────────────────


def test_readiness_shows_a_remote_cuda_row_even_with_no_gpu():
    """A missing row reads as "we didn't check". A NOT_CONNECTED row
    reads as "there isn't one yet", which is true and more useful."""
    report = readiness([mac_target()])

    assert report.status_of("UNKNOWN", ComputeDevice.CUDA.value) == (
        TargetStatus.NOT_CONNECTED.value
    )
    assert "no NVIDIA worker has been registered" in report.render()


def test_readiness_never_claims_a_mac_can_take_heavy_training():
    """It must agree with the scheduler, which refuses exactly this."""
    report = readiness([mac_target()])

    assert not report.can_run(WorkloadClass.HEAVY_TRAINING.value)
    assert report.can_run(WorkloadClass.LIGHT_FINE_TUNE.value)
    assert "none can take HEAVY_TRAINING" in report.summary


def test_a_connected_gpu_worker_changes_the_summary():
    report = readiness([mac_target(), gpu_target()])

    assert report.can_run(WorkloadClass.HEAVY_TRAINING.value)
    assert report.status_of("gpu-1", ComputeDevice.CUDA.value) == TargetStatus.READY.value


def test_an_unprobed_interpreter_is_not_reported_as_broken_hardware():
    """ "Nobody could look" and "the hardware is not there" are different
    answers, and only one of them is somebody's problem."""
    report = readiness([ExecutionTarget("control-plane", torchless_machine())])

    assert report.status_of("control-plane", ComputeDevice.MPS.value) == (
        TargetStatus.UNPROBED.value
    )
    assert "no torch" in report.render()


def test_a_torch_without_metal_reads_as_unavailable_rather_than_unprobed():
    machine = apple_machine(mps=False)
    machine = type(machine)(**{**machine.__dict__, "mps_built": False})
    report = readiness([ExecutionTarget("mac", machine)])

    assert report.status_of("mac", ComputeDevice.MPS.value) == TargetStatus.NOT_AVAILABLE.value
    assert "built without the Metal backend" in report.render()


def test_a_linux_worker_grows_no_permanent_mps_row():
    """A row saying "MPS: not available" forever on a machine that could
    never have it is noise pretending to be a finding."""
    report = readiness([gpu_target()])

    rows = [view for view in report.targets if view.name == "gpu-1"]
    assert all(view.device != ComputeDevice.MPS.value for view in rows)


def test_each_ready_row_lists_what_it_can_actually_be_asked_to_do():
    report = readiness([mac_target()])

    mps = next(
        view
        for view in report.targets
        if view.device == ComputeDevice.MPS.value and view.name == "mac"
    )
    assert WorkloadClass.LIGHT_FINE_TUNE.value in mps.workloads
    assert WorkloadClass.HEAVY_TRAINING.value not in mps.workloads
    assert WorkloadClass.PREPROCESS.value not in mps.workloads, "preprocessing is CPU work"


def test_readiness_carries_measured_precisions_only():
    report = readiness([mac_target()])
    mps = next(
        view
        for view in report.targets
        if view.device == ComputeDevice.MPS.value and view.name == "mac"
    )

    assert set(mps.precisions) == {"fp32", "fp16", "bf16"}

    unprobed = readiness([ExecutionTarget("control-plane", torchless_machine())])
    row = next(
        view
        for view in unprobed.targets
        if view.device == ComputeDevice.CPU.value and view.name == "control-plane"
    )
    assert row.precisions == (), "nothing measured, nothing claimed"


def test_readiness_is_serialisable_for_an_operator_view():
    payload = readiness([mac_target(), gpu_target()]).to_dict()

    assert payload["capability_schema_version"]
    assert payload["targets"]
    assert all("status" in row for row in payload["targets"])
    locations = {row["location"] for row in payload["targets"]}
    assert locations <= {ExecutionLocation.LOCAL.value, ExecutionLocation.REMOTE.value}
