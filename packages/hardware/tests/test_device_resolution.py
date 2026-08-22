"""What device a request gets, and the refusals that matter more.

Most of this file is about things the resolver will *not* do. A device
resolver that picks something reasonable when it cannot honour a request
is the failure mode worth testing: the run completes, the checkpoint
looks normal, and the only evidence that it trained on the wrong thing
is a wall-clock time nobody compares.
"""

from __future__ import annotations

import pytest
from hardware_fixtures import (
    ALL_PRECISION,
    NO_BF16,
    apple_machine,
    cuda_machine,
    torchless_machine,
)

from luber_hardware import (
    ComputeDevice,
    ComputePreference,
    DeviceOutcome,
    DeviceRequest,
    DeviceResolver,
    MpsFallbackPolicy,
    Precision,
    WorkloadClass,
)

# ── AUTO ─────────────────────────────────────────────────────────────


def test_auto_prefers_cuda_then_mps_then_cpu():
    """The same order the installed trainer's own `resolve_gpu` uses.

    Matching it is not deference — it is what stops LUBER reporting one
    device while the trainer selects another for the same run.
    """
    assert _auto(cuda_machine()) == ComputeDevice.CUDA.value
    assert _auto(apple_machine()) == ComputeDevice.MPS.value
    assert _auto(apple_machine(mps=False)) == ComputeDevice.CPU.value


def test_auto_is_deterministic():
    """Same capability, same answer, every time and in every process.

    A placement that depended on dictionary ordering would be
    reproducible by accident, which is worse than not being reproducible
    at all.
    """
    machine = apple_machine()
    answers = {DeviceResolver(machine).resolve(DeviceRequest()).device for _ in range(25)}
    assert answers == {ComputeDevice.MPS.value}


def test_auto_records_what_it_passed_over():
    """An operator asking "why not the GPU" gets the answer from the
    decision rather than by re-running anything."""
    decision = DeviceResolver(apple_machine()).resolve(DeviceRequest())

    assert decision.device == ComputeDevice.MPS.value
    assert ComputeDevice.CUDA.value in decision.considered
    assert "CUDA: not available" in decision.reason


# ── explicit requests are never downgraded ───────────────────────────


def test_requesting_cuda_without_cuda_is_blocked_not_downgraded():
    """The single most important assertion in this package.

    Somebody who named CUDA had a reason. Handing them a CPU produces a
    run that finishes, looks like every other run, and took nine hours.
    """
    decision = DeviceResolver(apple_machine()).resolve(
        DeviceRequest(preference=ComputePreference.CUDA.value)
    )

    assert decision.outcome == DeviceOutcome.BLOCKED_DEVICE_UNAVAILABLE.value
    assert decision.device is None
    assert "not downgraded to CPU" in decision.reason


def test_requesting_mps_where_mps_exists_gets_mps():
    decision = DeviceResolver(apple_machine()).resolve(
        DeviceRequest(preference=ComputePreference.MPS.value)
    )

    assert decision.device == ComputeDevice.MPS.value
    assert decision.torch_device == "mps"


def test_a_torch_built_without_metal_says_so():
    """ "Unavailable" and "not built" are different problems.

    One is a machine without the hardware; the other is an installation
    somebody can fix. An operator reading "no MPS" learns nothing about
    which they have.
    """
    machine = apple_machine(mps=False)
    machine = type(machine)(**{**machine.__dict__, "mps_built": False})

    decision = DeviceResolver(machine).resolve(
        DeviceRequest(preference=ComputePreference.MPS.value)
    )

    assert "built without the Metal backend" in decision.reason


def test_an_interpreter_without_torch_cannot_confirm_any_accelerator():
    """The control plane's own environment, described honestly.

    It must not report "no GPU" — it does not know. It reports that it
    could not look, and says where to look instead.
    """
    decision = DeviceResolver(torchless_machine()).resolve(
        DeviceRequest(preference=ComputePreference.CUDA.value)
    )

    assert decision.outcome == DeviceOutcome.BLOCKED_DEVICE_UNAVAILABLE.value
    assert "has no torch" in decision.reason
    assert "Probe the interpreter that runs training" in decision.reason


# ── precision ────────────────────────────────────────────────────────


def test_auto_precision_mirrors_the_installed_trainer():
    """bf16 on CUDA, fp16 on MPS, fp32 on CPU.

    Read from `training_v2/fixed_lora_module.py` at the pinned commit.
    LUBER does not get an opinion here: this is what will happen, and
    reporting anything else would be reporting a run that will not
    occur.
    """
    assert _auto_precision(cuda_machine()) == Precision.BF16.value
    assert _auto_precision(apple_machine()) == Precision.FP16.value
    assert _auto_precision(apple_machine(mps=False)) == Precision.FP32.value


def test_an_unsupported_precision_is_blocked():
    """Upstream passes an explicit precision straight through without
    checking. Somebody has to check, and it is cheaper here than in the
    first epoch."""
    decision = DeviceResolver(cuda_machine(precision=NO_BF16)).resolve(
        DeviceRequest(preference=ComputePreference.CUDA.value, precision=Precision.BF16.value)
    )

    assert decision.outcome == DeviceOutcome.BLOCKED_PRECISION_UNSUPPORTED.value
    assert "the arithmetic failed" in decision.reason


def test_an_unmeasured_precision_is_refused_by_default_and_recorded():
    """A machine nobody probed cannot satisfy a precision request.

    Same rule Phase 27's preflight applies to VRAM: unmeasured is not
    satisfied. The caller can opt in, and the decision still says the
    figure was never measured.
    """
    machine = apple_machine()
    machine = type(machine)(**{**machine.__dict__, "precision_support": {}})

    strict = DeviceResolver(machine).resolve(
        DeviceRequest(preference=ComputePreference.MPS.value, precision=Precision.BF16.value)
    )
    assert strict.outcome == DeviceOutcome.BLOCKED_PRECISION_UNSUPPORTED.value
    assert strict.precision_unverified

    relaxed = DeviceResolver(machine).resolve(
        DeviceRequest(
            preference=ComputePreference.MPS.value,
            precision=Precision.BF16.value,
            allow_unverified_precision=True,
        )
    )
    assert relaxed.resolved
    assert relaxed.precision_unverified, "accepted, and the gap is still on the record"


def test_auto_precision_survives_an_unprobed_machine():
    """`auto` is what the trainer would pick anyway.

    Refusing it for want of a measurement would block every unprobed
    target from running at all, which helps nobody.
    """
    machine = apple_machine()
    machine = type(machine)(**{**machine.__dict__, "precision_support": {}})

    decision = DeviceResolver(machine).resolve(
        DeviceRequest(preference=ComputePreference.MPS.value)
    )

    assert decision.resolved
    assert decision.precision == Precision.FP16.value
    assert decision.precision_unverified


def test_an_unknown_precision_name_is_refused():
    decision = DeviceResolver(apple_machine()).resolve(DeviceRequest(precision="fp8"))

    assert decision.outcome == DeviceOutcome.BLOCKED_PRECISION_UNSUPPORTED.value
    assert "unknown precision" in decision.reason


# ── MPS fallback ─────────────────────────────────────────────────────


def test_an_operation_needing_cuda_blocks_mps_under_the_strict_policy():
    """`adamw8bit` is the live example and it fails *soft* upstream.

    ACE-Step's `optim.py` catches the bitsandbytes ImportError, logs a
    warning and trains with AdamW. The run succeeds having used a
    different optimizer than its own plan records — so LUBER refuses
    before the run rather than discovering it in a log.
    """
    decision = DeviceResolver(apple_machine()).resolve(
        DeviceRequest(
            preference=ComputePreference.MPS.value,
            required_operations=("adamw8bit",),
            mps_fallback=MpsFallbackPolicy.STRICT.value,
        )
    )

    assert decision.outcome == DeviceOutcome.BLOCKED_MPS_FALLBACK_REQUIRED.value
    assert "bitsandbytes" in decision.reason


def test_training_refuses_cpu_fallback_even_when_the_policy_permits_it():
    """Permission to fall back is not permission to train differently.

    For inference a CPU fallback is a slower answer to the same
    question. For training it changes what was learned, so the strict
    rule applies to training whatever the policy says.
    """
    decision = DeviceResolver(apple_machine()).resolve(
        DeviceRequest(
            workload=WorkloadClass.HEAVY_TRAINING.value,
            preference=ComputePreference.MPS.value,
            required_operations=("adamw8bit",),
            mps_fallback=MpsFallbackPolicy.ALLOW_CPU_FALLBACK.value,
        )
    )

    assert decision.outcome == DeviceOutcome.BLOCKED_MPS_FALLBACK_REQUIRED.value


def test_a_permitted_fallback_is_recorded_rather_than_silent():
    """The whole difference between this and `PYTORCH_ENABLE_MPS_FALLBACK`.

    The environment variable makes the fallback invisible. Here it lands
    on the decision, where somebody reading the run can see it.
    """
    decision = DeviceResolver(apple_machine()).resolve(
        DeviceRequest(
            workload=WorkloadClass.INFERENCE.value,
            preference=ComputePreference.MPS.value,
            required_operations=("adamw8bit",),
            mps_fallback=MpsFallbackPolicy.ALLOW_CPU_FALLBACK.value,
        )
    )

    assert decision.resolved
    assert decision.fallback_used
    assert any("CPU fallback in use" in item for item in decision.limitations)


def test_a_device_outside_the_allowed_set_is_refused():
    """A policy that permits only CPU gets only CPU, whatever is here."""
    decision = DeviceResolver(cuda_machine(), allowed=(ComputeDevice.CPU.value,)).resolve(
        DeviceRequest(preference=ComputePreference.CUDA.value)
    )

    assert decision.outcome == DeviceOutcome.BLOCKED_DEVICE_UNAVAILABLE.value
    assert "not permitted" in decision.reason


def test_nothing_permitted_is_a_named_refusal():
    decision = DeviceResolver(apple_machine(), allowed=()).resolve(DeviceRequest())

    assert decision.outcome == DeviceOutcome.BLOCKED_NO_COMPATIBLE_DEVICE.value


# ── helpers ──────────────────────────────────────────────────────────


def _auto(machine) -> str | None:
    return DeviceResolver(machine).resolve(DeviceRequest()).device


def _auto_precision(machine) -> str | None:
    return DeviceResolver(machine).resolve(DeviceRequest()).precision


@pytest.mark.parametrize("precision", [Precision.FP32.value, Precision.FP16.value])
def test_every_measured_precision_resolves(precision: str):
    decision = DeviceResolver(apple_machine(precision=ALL_PRECISION)).resolve(
        DeviceRequest(preference=ComputePreference.MPS.value, precision=precision)
    )

    assert decision.resolved
    assert decision.precision == precision
