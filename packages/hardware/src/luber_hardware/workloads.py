"""What kind of work is being placed.

Only classes this repository actually distinguishes. A taxonomy with
entries nothing produces is a taxonomy that gets ignored, so each one
below corresponds to real code and a real placement difference.

The ordering that matters is not size but *what breaks if it goes to
the wrong place*. Preprocessing on a rented GPU wastes money.
Heavy training on a laptop wastes a day. A CUDA-required job quietly
moved to MPS produces a model somebody will ship.
"""

from __future__ import annotations

from enum import StrEnum


class WorkloadClass(StrEnum):
    """The kinds of work LUBER places."""

    #: Dataset building, audio decode, resample, manifest generation.
    #: `luber_dataset` and `luber_audio_utils`: numpy and ffmpeg, no
    #: torch at all. Sending this to a rented GPU spends GPU-hours on
    #: work a CPU does at the same speed.
    PREPROCESS = "PREPROCESS"

    #: Phase 26 evaluation. Drives a running ACE-Step server over HTTP
    #: and computes metrics in pure Python — whichever device *that
    #: server* uses is the server's business, not this decision's.
    EVALUATION = "EVALUATION"

    #: Music generation for the product. Runs through the ARQ generation
    #: worker against a provider; there is no remote generation path
    #: today, and this class does not create one.
    INFERENCE = "INFERENCE"

    #: Small LoRA/LoKr runs whose purpose is to find out whether the
    #: pipeline works. Apple silicon is a legitimate host for these.
    LIGHT_FINE_TUNE = "LIGHT_FINE_TUNE"

    #: Real adapter training. Prefers a CUDA machine; a policy may
    #: require one.
    HEAVY_TRAINING = "HEAVY_TRAINING"

    #: Loading a checkpoint to inspect or verify it. Needs a device that
    #: can hold tensors and nothing more, which is every device.
    CHECKPOINT_VALIDATION = "CHECKPOINT_VALIDATION"


#: Workloads that consult a compute device at all.
#:
#: The other three are Python and ffmpeg. Recording that here rather
#: than in each caller is what keeps a future reader from "improving"
#: preprocessing onto a GPU because a GPU was available.
DEVICE_BOUND: frozenset[str] = frozenset(
    {
        WorkloadClass.LIGHT_FINE_TUNE.value,
        WorkloadClass.HEAVY_TRAINING.value,
        WorkloadClass.CHECKPOINT_VALIDATION.value,
        WorkloadClass.INFERENCE.value,
    }
)

#: Workloads that train. These are the ones where a silent device
#: substitution changes what was learned rather than how fast it was.
TRAINING_WORKLOADS: frozenset[str] = frozenset(
    {WorkloadClass.LIGHT_FINE_TUNE.value, WorkloadClass.HEAVY_TRAINING.value}
)


def is_training(workload: str) -> bool:
    return workload in TRAINING_WORKLOADS


def uses_device(workload: str) -> bool:
    return workload in DEVICE_BOUND


__all__ = [
    "DEVICE_BOUND",
    "TRAINING_WORKLOADS",
    "WorkloadClass",
    "is_training",
    "uses_device",
]
