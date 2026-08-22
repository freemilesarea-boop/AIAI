"""The vocabulary: where something runs, and what it runs on.

Two axes, kept apart on purpose.

**Where** is `ExecutionLocation` — this machine, or one reached over
SSH. **What on** is `ComputeDevice` — the CPU, Apple's Metal backend, or
an NVIDIA GPU.

Collapsing them into one enum is the tempting simplification and it is
wrong here. This repository already has a *local* backend that uses no
accelerator at all (the dry run) and a *remote* backend whose device is
whatever the worker reported — which could be CPU, on a machine that has
never had a GPU. `LOCAL` does not mean CPU and `REMOTE` does not mean
CUDA, and a single enum would encode both falsehoods permanently.

Precision values are lowercase to match `luber_training.config.Precision`
exactly. A training config's `precision` string is passed to this
package unchanged; two spellings of "bf16" across a seam is a bug
waiting for the day somebody compares them.
"""

from __future__ import annotations

from enum import StrEnum


class ExecutionLocation(StrEnum):
    """Which machine runs the work."""

    LOCAL = "LOCAL"
    REMOTE = "REMOTE"


class ComputeDevice(StrEnum):
    """Which compute unit does the arithmetic."""

    CPU = "CPU"
    #: Apple's Metal Performance Shaders backend. Not "GPU": a Mac has a
    #: GPU whether or not this torch can reach it, and the distinction
    #: is the whole question.
    MPS = "MPS"
    CUDA = "CUDA"


class ComputePreference(StrEnum):
    """What the caller asked for.

    Distinct from `ComputeDevice` because `AUTO` is a request and never
    an answer. A decision that recorded `AUTO` as the selected device
    would be unreadable a month later.
    """

    AUTO = "AUTO"
    CPU = "CPU"
    MPS = "MPS"
    CUDA = "CUDA"


class Precision(StrEnum):
    """Numeric precision, spelled as the training config spells it."""

    AUTO = "auto"
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"


class MpsFallbackPolicy(StrEnum):
    """What happens when MPS cannot do something.

    PyTorch offers `PYTORCH_ENABLE_MPS_FALLBACK=1`, which silently runs
    unsupported operations on the CPU. For inference that is a
    convenience. For *training* it is a trap: a run that quietly moved
    half its steps to the CPU would take hours longer for reasons
    nothing recorded, and the operator would conclude that Apple silicon
    is slow rather than that one operator was missing.

    So the default is STRICT, and fallback — where a workload permits it
    at all — is recorded on the decision rather than left to an
    environment variable nobody set deliberately.
    """

    #: An unsupported operation is a refusal, before the run starts.
    STRICT = "MPS_STRICT"
    #: CPU fallback is permitted, and every use of it is recorded.
    ALLOW_CPU_FALLBACK = "MPS_ALLOW_CPU_FALLBACK"


#: The environment variable that enables PyTorch's own MPS fallback.
#:
#: Named here so there is exactly one spelling of it in the repository,
#: and so the audit question "does LUBER depend on this?" has a single
#: place to look. The answer today is no: nothing sets it, and
#: `MpsFallbackPolicy.STRICT` is what the training path uses.
PYTORCH_MPS_FALLBACK_ENV = "PYTORCH_ENABLE_MPS_FALLBACK"


#: Device a torch device string maps to, for reading a decision back.
TORCH_DEVICE_STRING: dict[str, str] = {
    ComputeDevice.CPU.value: "cpu",
    ComputeDevice.MPS.value: "mps",
    ComputeDevice.CUDA.value: "cuda",
}


def torch_device_string(device: str) -> str:
    """The string the trainer's `--device` flag expects."""
    try:
        return TORCH_DEVICE_STRING[device]
    except KeyError:
        raise ValueError(
            f"unknown compute device {device!r}. Known: " + ", ".join(sorted(TORCH_DEVICE_STRING))
        ) from None


__all__ = [
    "PYTORCH_MPS_FALLBACK_ENV",
    "ComputeDevice",
    "ComputePreference",
    "ExecutionLocation",
    "MpsFallbackPolicy",
    "Precision",
    "torch_device_string",
]
