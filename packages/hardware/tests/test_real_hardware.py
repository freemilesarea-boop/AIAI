"""Tests that ask the actual machine, and skip out loud when they cannot.

Every test here needs a Python with `torch` installed. LUBER's own
environment does not have one — the control plane never imports torch —
so on most machines and in CI these skip, and the skip reason says which
interpreter was looked for.

That is the design, not a compromise. A test suite that required torch
would make CI carry a large dependency for the sake of hardware CI does
not have; a suite that faked it would report PASS for something that
never ran. Skipping with a reason is the only honest third option, and
the reason is what stops a green run being read as verification.

Where a torch interpreter *is* present, these do real work: a real MPS
probe, a real eight-step training loop, a real checkpoint written on one
device and loaded on another.

To run them against a specific interpreter:

    LUBER_TORCH_PYTHON=/path/to/python uv run pytest packages/hardware
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from hardware_fixtures import torch_interpreter

from luber_hardware import (
    ComputeDevice,
    DeviceRequest,
    DeviceResolver,
    ProbeError,
    _smoke,
    probe_machine,
    probe_this_process,
)

INTERPRETER = torch_interpreter()

needs_torch = pytest.mark.skipif(
    INTERPRETER is None,
    reason=(
        "no interpreter with torch was found. Looked at $LUBER_TORCH_PYTHON, this "
        "process, and ~/ace-step-1.5/.venv/bin/python. Set LUBER_TORCH_PYTHON to run "
        "these against a real runtime."
    ),
)


def _smoke_result(devices: list[str]) -> dict:
    """Run the smoke in the torch interpreter and read its JSON."""
    assert INTERPRETER is not None
    completed = subprocess.run(
        [INTERPRETER, str(Path(_smoke.__file__).resolve()), *devices],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return json.loads(completed.stdout)


# ── the probe tells the truth ────────────────────────────────────────


def test_the_probe_never_invents_an_accelerator():
    """The one thing a capability probe must never do.

    Run everywhere, including where torch is absent: a machine with no
    CUDA must not report CUDA, whatever else is unknown.
    """
    capability = probe_this_process()

    assert capability.cuda_available is not True or capability.torch_installed, (
        "CUDA cannot be reported available by an interpreter that has no torch"
    )
    if not capability.torch_installed:
        assert capability.mps_available is None, "unmeasured is None, never False"
        assert capability.devices() == (ComputeDevice.CPU.value,)


def test_darwin_alone_does_not_imply_mps():
    """`sys.platform == "darwin"` says nothing about the Metal backend.

    A torch built without it, or an Intel Mac, is Darwin with no MPS.
    Only torch can answer, and where torch is absent the answer is
    UNKNOWN rather than True.
    """
    capability = probe_this_process()

    if capability.apple_silicon and not capability.torch_installed:
        assert capability.mps_available is None


def test_the_report_contains_nothing_personal():
    """Structural: there is no field for a hostname or a home directory.

    Asserted anyway against the rendered document, because a later field
    could change that and this is where it would be caught.
    """
    payload = json.dumps(probe_this_process().to_dict())
    home = str(Path.home())

    for needle in (home, Path.home().name, "Users/"):
        assert needle not in payload, f"the capability report leaked {needle!r}"


def test_probing_a_missing_interpreter_fails_rather_than_guessing():
    with pytest.raises(ProbeError):
        probe_machine("/nonexistent/python")


# ── the real machine ─────────────────────────────────────────────────


@needs_torch
def test_a_torch_interpreter_reports_a_usable_device():
    capability = probe_machine(INTERPRETER)

    assert capability.torch_installed
    assert capability.torch_version
    assert ComputeDevice.CPU.value in capability.devices()
    # Every device it claims must carry a measured precision table.
    for device in capability.devices():
        assert capability.precision_support.get(device) is not None


@needs_torch
def test_auto_selects_the_best_device_this_machine_actually_has():
    capability = probe_machine(INTERPRETER)
    decision = DeviceResolver(capability).resolve(DeviceRequest())

    assert decision.resolved
    assert decision.device == capability.devices()[0]


@needs_torch
def test_a_tiny_training_run_completes_on_the_cpu():
    """Forward, backward, AdamW, scheduler, gradient clip. Eight steps.

    A toy network on synthetic noise — not ACE-Step, not a DiT, no music
    anywhere in it. It proves the mechanism works, and nothing about the
    model.
    """
    result = _smoke_result(["cpu"])["results"]["cpu"]

    assert result["ok"]
    assert result["training"]["steps"] == 8
    assert result["training"]["loss_changed"], "the optimizer moved the weights"
    assert result["training"]["parameters_finite"]


@needs_torch
def test_a_tiny_training_run_completes_on_mps_where_mps_exists():
    payload = _smoke_result([])
    if ComputeDevice.MPS.value.lower() not in payload["devices"]:
        pytest.skip("this machine's torch reports no MPS backend")

    result = payload["results"]["mps"]

    assert result["ok"], result.get("error")
    assert result["training"]["loss_changed"]
    assert result["training"]["parameters_finite"]


@needs_torch
def test_a_checkpoint_written_on_one_device_loads_on_the_others():
    """The claim behind "train on the GPU box, inspect on the Mac".

    `map_location` is the whole question: a checkpoint holds tensors
    that remember where they were, and loading one elsewhere fails
    unless the loader is told where to put them. Optimizer state is
    included and a step is taken after loading, because state that
    loads but cannot be stepped is not portable.
    """
    payload = _smoke_result([])
    for device, result in payload["results"].items():
        for target, answer in result["checkpoint"]["loads"].items():
            assert answer["ok"], f"{device} → {target}: {answer.get('error')}"
            assert answer["model"]
            assert answer["optimizer"]
            assert answer["adapter"], "a LoRA-shaped tensor pair must move too"


@needs_torch
def test_a_bounded_memory_allocation_works_and_is_released():
    """64 MiB, allocated and freed. Deliberately not a search for the
    ceiling: exhausting a machine to find out how much it has would swap
    the control plane out and measure the day rather than the hardware.
    """
    payload = _smoke_result([])
    for device, result in payload["results"].items():
        memory = result["memory"]
        assert memory["allocated"], f"{device}: {memory.get('error')}"
        assert memory["released"]


@needs_torch
def test_the_benchmark_reports_this_machine_and_claims_nothing_else():
    """A sanity measurement, printed for the record.

    It exists to catch "MPS is not actually being used" and "this path
    is broken". It is not a ranking, and nothing may be extrapolated
    from it to different hardware.
    """
    payload = _smoke_result([])
    for device, result in payload["results"].items():
        benchmark = result["benchmark"]
        assert benchmark["matmul_ms"] > 0
        assert benchmark["forward_backward_ms"] > 0
        print(
            f"\n{device}: matmul({benchmark['matmul_size']}) "
            f"{benchmark['matmul_ms']:.3f}ms, "
            f"fwd+bwd {benchmark['forward_backward_ms']:.3f}ms "
            f"[this machine only — not comparable to other hardware]"
        )
