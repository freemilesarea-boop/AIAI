"""Failure codes, in the operator's language and the system's.

A closed vocabulary is only useful if somebody can act on it. Phase 25
chose codes that are honest — ``UNKNOWN`` stays ``UNKNOWN`` — and this
module adds the sentence an operator needs beside each one. The raw code
never disappears: it is what they will grep the logs for, and a console
that showed only prose would make that search harder.

Two entries carry most of the weight of this phase.

``WORKER_LOST`` must not read as "training failed". It means contact was
lost, which is a statement about the connection and not about the
trainer. A rented GPU may still be burning money, and an operator told
"failed" will retry — producing a second trainer writing into the same
checkpoint directory, which is how two well-formed artifacts become one
worthless one. So the guidance says reconcile first, in those words.

``OOM`` is claimed only where the classification was definitive. Phase 27
raises it on an explicit CUDA out-of-memory message and never on a
SIGKILL, because the kernel OOM killer and ``kill -9`` are
indistinguishable at the exit code. Where the evidence is weaker the
console says so rather than sending the next experiment chasing a memory
problem that never existed.
"""

from __future__ import annotations

from dataclasses import dataclass

from luber_training.entities import FailureCode


@dataclass(frozen=True)
class FailureMeaning:
    headline: str
    guidance: str
    #: Whether the classification is definitive enough to act on
    #: without further evidence.
    confident: bool = True


_MEANINGS: dict[str, FailureMeaning] = {
    FailureCode.DATASET_LOCK_INVALID.value: FailureMeaning(
        headline="The dataset lock did not verify",
        guidance=(
            "The manifest no longer matches the digest the lock recorded. Rebuild the dataset "
            "or select the build the lock was written for; nothing here should be forced."
        ),
    ),
    FailureCode.CURATION_LOCK_INVALID.value: FailureMeaning(
        headline="The curation lock did not verify",
        guidance=(
            "The curated manifest has changed since the lock was written — a curation "
            "regenerated between validation and dispatch does this. Re-curate and create a "
            "new run."
        ),
    ),
    FailureCode.RIGHTS_GATE_FAILED.value: FailureMeaning(
        headline="Rights are not clear for every selected track",
        guidance=(
            "One or more tracks in the curated manifest are not cleared for training. This "
            "gate has no override: resolve the rights record, re-curate, and create a new run."
        ),
    ),
    FailureCode.EVALUATION_LEAKAGE.value: FailureMeaning(
        headline="Evaluation material appears in the training set",
        guidance=(
            "Training on a track the model is later judged on makes the evaluation "
            "meaningless. Remove the flagged ids from the curation and rebuild."
        ),
    ),
    FailureCode.SELF_GENERATED_BLOCKED.value: FailureMeaning(
        headline="Self-generated audio is present in the training set",
        guidance=(
            "Training on this system's own output compounds its artifacts. Exclude the "
            "flagged ids, or re-run with the self-generated allowance set deliberately."
        ),
    ),
    FailureCode.ENVIRONMENT_INVALID.value: FailureMeaning(
        headline="The worker environment is not what the plan requires",
        guidance=(
            "Compare the run's environment lock with the worker's capability report. A "
            "protocol, torch or ACE-Step mismatch is fixed on the worker, not in the plan."
        ),
    ),
    FailureCode.INSUFFICIENT_HARDWARE.value: FailureMeaning(
        headline="The worker cannot hold this plan",
        guidance=(
            "The reported capability is below what the plan asks for. Choose a worker whose "
            "capability was measured, or reduce the configuration in a new run."
        ),
    ),
    FailureCode.CODE_VERSION_DIRTY.value: FailureMeaning(
        headline="The repository revision could not be identified",
        guidance=(
            "A run from an unidentified revision cannot be reproduced. Commit or stash the "
            "working tree and create a new run."
        ),
    ),
    FailureCode.WORKER_LOST.value: FailureMeaning(
        headline="Worker connection lost",
        guidance=(
            "The remote trainer may still be running. Reconcile before doing anything else: "
            "launching a retry now can put two trainers in one checkpoint directory, and the "
            "artifacts they produce are individually well-formed and jointly worthless."
        ),
        # We know contact stopped. We do not know the trainer did.
        confident=False,
    ),
    FailureCode.TRAINER_CRASH.value: FailureMeaning(
        headline="The trainer exited with a non-zero status",
        guidance=(
            "Read the tail of stderr below. The exit code and the last log lines are the "
            "evidence; the code above is only the category."
        ),
    ),
    FailureCode.OOM.value: FailureMeaning(
        headline="CUDA out of memory",
        guidance=(
            "The trainer said so explicitly. Compare the worker's VRAM with the batch size, "
            "gradient accumulation and rank below, then create a new run with a changed "
            "configuration — nothing here edits a run in place."
        ),
    ),
    FailureCode.CHECKPOINT_WRITE_FAILED.value: FailureMeaning(
        headline="A checkpoint could not be written",
        guidance=(
            "Usually disk. Check free space on the worker's checkpoint root; any checkpoint "
            "already finalised is unaffected and remains READY."
        ),
    ),
    FailureCode.CANCELLED_BY_OPERATOR.value: FailureMeaning(
        headline="Cancelled by an operator",
        guidance=(
            "Metrics, logs and any completed checkpoints were kept. A cancelled run is part "
            "of the experiment's history."
        ),
    ),
    FailureCode.UNKNOWN.value: FailureMeaning(
        headline="The cause was not established",
        guidance=(
            "The system declined to guess. The logs and the last metric below are what there "
            "is; a more specific code would have been invented rather than observed."
        ),
        confident=False,
    ),
}

_UNRECOGNISED = FailureMeaning(
    headline="Unrecognised failure code",
    guidance=(
        "This build does not know this code. It is shown exactly as recorded rather than "
        "mapped onto something that looks familiar."
    ),
    confident=False,
)


def meaning_for(code: str | None) -> FailureMeaning | None:
    if not code:
        return None
    return _MEANINGS.get(code, _UNRECOGNISED)


__all__ = ["FailureMeaning", "meaning_for"]
