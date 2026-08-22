"""The gate immediately before a trainer starts, and its vocabulary.

Phase 25 asks *may* this be trained — rights, locks, leakage. Phase 27
asks whether the artifacts arrived intact on the worker. This module
asks the question between them, which nothing owned before: **can the
selected machine actually execute this exact plan, and can we prove it
before the expensive part begins?**

Three statuses, and the third is the one that carries the phase.

``READY`` — every mandatory check is proven satisfied for the stated
intent. Nothing is assumed and nothing was skipped.

``BLOCKED`` — at least one check found a definite incompatibility. The
run must not start, and the reason is machine-readable so a console can
route it rather than printing a sentence.

``UNVERIFIED`` — a mandatory check could not be performed. Nobody
measured the thing; nobody probed the machine; the capability report is
older than the policy allows. **UNVERIFIED is not READY.** It is not a
warning, it is not amber-but-fine, and no caller in this repository may
treat it as a pass.

Two design decisions are worth stating.

**Checks are pure functions over evidence.** Every filesystem read,
subprocess and probe happens in :mod:`luber_training.preflight_collect`
and arrives here as a frozen dataclass. That is what makes a preflight
deterministic — the same evidence produces the same result, byte for
byte, including its digest — and what makes the CUDA path testable on a
machine with no NVIDIA hardware.

**Intent decides what is mandatory.** A bounded canary and a
production-scale run need different proofs. Nobody has measured what a
real LUBER training workload needs on any device, so a full-training
preflight is UNVERIFIED on that ground alone — which is the true answer
and is exactly why the canary intent exists to be answerable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from luber_hardware import (
    AUTO_BY_DEVICE,
    CUDA_ONLY_OPERATIONS,
    ComputeDevice,
    ExecutionLocation,
    MachineCapability,
    resolve_precision,
    torch_device_string,
)
from luber_training.capacity import (
    DEVICE_MEMORY,
    REQUIRED_DISK,
    TRAINING_MEMORY_REQUIREMENT,
    CapacityReport,
    EvidenceSource,
)
from luber_training.capacity_policy import CapacityDecision, CapacityQualification
from luber_training.config import Optimizer, Precision
from luber_training.entities import TrainingWorker, WorkerClass
from luber_training.gates import GateReport
from luber_training.plan import TRAINING_PLAN_SCHEMA_VERSION, TrainingPlan

TRAINING_PREFLIGHT_SCHEMA_VERSION = "luber-training-preflight/1"

#: Bump when the same evidence would produce a different verdict.
TRAINING_PREFLIGHT_POLICY_VERSION = "training-preflight-v1"

#: Plan schema versions this preflight understands. A plan written by a
#: newer LUBER is refused rather than read optimistically: the fields it
#: added are exactly the ones a stale reader would ignore.
SUPPORTED_PLAN_SCHEMA_VERSIONS: frozenset[str] = frozenset({TRAINING_PLAN_SCHEMA_VERSION})

#: How old a worker capability report may be and still be evidence.
#:
#: Twelve hours. A rented GPU is returned, resized, or has its driver
#: updated between sessions, and a capability report from last week
#: describes a machine that may no longer exist. Not a measurement —
#: a policy — so it is named, configurable per call, and stated in the
#: result rather than applied silently.
DEFAULT_CAPABILITY_MAX_AGE_SECONDS = 12 * 60 * 60

#: The package each optional optimizer needs, in the trainer's own
#: environment.
#:
#: This matters more than it looks. ACE-Step's ``build_optimizer``
#: catches ``ImportError`` for every one of these and **logs a warning
#: and uses AdamW instead**. A run configured for Prodigy on a machine
#: without `prodigyopt` therefore trains with a different optimizer than
#: its own plan records, and the only evidence is one line in a log
#: nobody read. Verified here instead.
OPTIMIZER_PACKAGES: dict[str, str] = {
    Optimizer.ADAMW_8BIT.value: "bitsandbytes",
    Optimizer.ADAFACTOR.value: "transformers",
    Optimizer.PRODIGY.value: "prodigyopt",
}


#: Device and precision pairs the installed trainer cannot train in.
#:
#: Not a capability of the hardware — Phase 32 measured fp16 tensors
#: working on this machine's MPS backend, and they do. It is a property
#: of the *trainer*: `fixed_lora_module` loads the model in fp16 and
#: drives Lightning Fabric at `16-mixed`, and Fabric's GradScaler
#: refuses to unscale gradients that are themselves fp16. The run dies
#: at the first gradient clip, after the 2.4B model has been loaded.
#:
#: Every entry must name how it is known. This one was measured, on this
#: machine, by a Phase 33 canary — not inferred from reading the code.
UNTRAINABLE_PRECISION: dict[tuple[str, str], str] = {
    (ComputeDevice.MPS.value, Precision.FP16.value): (
        "the installed trainer loads the model in fp16 on MPS and runs Fabric at "
        "'16-mixed'; torch then raises 'Attempting to unscale FP16 gradients.' at the "
        "first gradient clip and no step completes. Measured on Apple M4 Pro, torch "
        "2.10.0, ACE-Step 6d467e4b, by a Phase 33 bounded canary. Request bf16 or fp32 "
        "explicitly — 'auto' resolves to fp16 here and hits the same wall"
    ),
}


class PreflightIntent(StrEnum):
    """What the preflight is clearing the way for.

    The intents differ in what they are allowed to leave unmeasured, not
    in how honest they are. Both report the same unknowns; only one of
    them can reach READY while the memory requirement is unmeasured.
    """

    #: A bounded canary: a handful of samples, one epoch, a hard step
    #: ceiling. Capacity for a *production* workload is irrelevant to it
    #: because it will not run one.
    CANARY = "CANARY"

    #: A real training run. An unmeasured memory requirement makes this
    #: UNVERIFIED, which is the honest state of this project today.
    FULL_TRAINING = "FULL_TRAINING"


class PreflightStatus(StrEnum):
    READY = "READY"
    BLOCKED = "BLOCKED"
    UNVERIFIED = "UNVERIFIED"


class CheckStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    #: Could not be established. Never a soft pass.
    UNKNOWN = "UNKNOWN"
    #: This check does not apply to this contract — a remote check on a
    #: local run. Distinct from PASS so a reader can tell "we looked and
    #: it was fine" from "there was nothing to look at".
    NOT_APPLICABLE = "NOT_APPLICABLE"


class BlockingReason(StrEnum):
    """Why a preflight refused, in a closed machine-readable vocabulary.

    Closed, and deliberately not collapsed into ``FAILED``. Each value
    names something an operator does differently: a rights refusal ends
    the run, a stale worker means re-probe, an unwritable directory
    means fix the disk. One generic code would make the console a
    paragraph of prose.
    """

    PLAN_INVALID = "PLAN_INVALID"
    DATASET_NOT_READY = "DATASET_NOT_READY"
    RIGHTS_BLOCKED = "RIGHTS_BLOCKED"
    MANIFEST_DRIFT = "MANIFEST_DRIFT"
    EVALUATION_LEAKAGE = "EVALUATION_LEAKAGE"
    DEVICE_UNAVAILABLE = "DEVICE_UNAVAILABLE"
    PRECISION_UNSUPPORTED = "PRECISION_UNSUPPORTED"
    OPTIMIZER_UNSUPPORTED = "OPTIMIZER_UNSUPPORTED"
    TRAINER_UNAVAILABLE = "TRAINER_UNAVAILABLE"
    DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
    WORKER_UNAVAILABLE = "WORKER_UNAVAILABLE"
    WORKER_STALE = "WORKER_STALE"
    REMOTE_UNREACHABLE = "REMOTE_UNREACHABLE"
    STORAGE_UNAVAILABLE = "STORAGE_UNAVAILABLE"
    INSUFFICIENT_DISK = "INSUFFICIENT_DISK"
    CAPACITY_UNVERIFIED = "CAPACITY_UNVERIFIED"
    CANARY_FAILED = "CANARY_FAILED"
    CHECKPOINT_FAILED = "CHECKPOINT_FAILED"
    RESUME_FAILED = "RESUME_FAILED"


#: Which group each check belongs to, for the per-area statuses an
#: operator console renders. Derived from the check name's prefix so a
#: new check cannot be added without landing in a group.
GROUPS: tuple[str, ...] = (
    "plan",
    "dataset",
    "trainer",
    "hardware",
    "storage",
    "remote",
    "capacity",
    "canary",
    "checkpoint",
)


@dataclass(frozen=True)
class PreflightCheck:
    """One question, its answer, and whether the answer may stop a run."""

    name: str
    status: str
    detail: str = ""
    #: The taxonomy entry a failure or an unknown maps to. ``None`` only
    #: for a check that passed or does not apply.
    reason: str | None = None
    #: Whether this check's outcome decides the overall status. A
    #: non-mandatory check is recorded and reported and never blocks.
    mandatory: bool = True

    @property
    def blocks(self) -> bool:
        return self.mandatory and self.status == CheckStatus.FAIL.value

    @property
    def unverifies(self) -> bool:
        return self.mandatory and self.status == CheckStatus.UNKNOWN.value

    @property
    def group(self) -> str:
        head = self.name.split(".", 1)[0]
        return head if head in GROUPS else "plan"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "reason": self.reason,
            "mandatory": self.mandatory,
            "group": self.group,
        }


# ── evidence ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DatasetEvidence:
    """What is known about the data this plan would train on.

    Every field defaults to ``None`` — nobody looked — because a
    preflight run without a dataset root is a legitimate thing to do and
    must come back UNVERIFIED rather than pretending the data is fine.
    """

    manifest_present: bool | None = None
    #: Whether the locks verify. Normally supplied by the Phase 25 gates
    #: rather than recomputed here.
    locks_verified: bool | None = None
    eligible_sample_count: int | None = None
    evaluation_only_count: int | None = None
    #: The curated manifest digest as it is on disk right now, and the
    #: one the plan cited when it was compiled. Different means the
    #: build moved under the plan.
    observed_manifest_sha256: str | None = None
    referenced_files_present: bool | None = None
    missing_file_count: int | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_present": self.manifest_present,
            "locks_verified": self.locks_verified,
            "eligible_sample_count": self.eligible_sample_count,
            "evaluation_only_count": self.evaluation_only_count,
            "observed_manifest_sha256": self.observed_manifest_sha256,
            "referenced_files_present": self.referenced_files_present,
            "missing_file_count": self.missing_file_count,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class TrainerEvidence:
    """What is known about the trainer the plan would invoke."""

    trainer_root_present: bool | None = None
    entrypoint_present: bool | None = None
    interpreter_present: bool | None = None
    torch_importable: bool | None = None
    #: The ACE-Step revision installed, and the one the config was
    #: written against.
    observed_ace_step_commit: str | None = None
    #: Whether the compiled argv is accepted by the installed parser.
    #: ``False`` means the command would not even start.
    command_accepted: bool | None = None
    command_detail: str = ""
    #: Packages the plan needs that the trainer's interpreter does not
    #: have. Named, because ACE-Step substitutes a different optimizer
    #: rather than failing when one is missing.
    missing_packages: tuple[str, ...] = ()
    #: Packages nobody could check — an interpreter that could not be
    #: reached at all.
    unchecked_packages: tuple[str, ...] = ()
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "trainer_root_present": self.trainer_root_present,
            "entrypoint_present": self.entrypoint_present,
            "interpreter_present": self.interpreter_present,
            "torch_importable": self.torch_importable,
            "observed_ace_step_commit": self.observed_ace_step_commit,
            "command_accepted": self.command_accepted,
            "command_detail": self.command_detail,
            "missing_packages": list(self.missing_packages),
            "unchecked_packages": list(self.unchecked_packages),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class StorageEvidence:
    """Whether the run has somewhere to read from and write to."""

    dataset_readable: bool | None = None
    output_writable: bool | None = None
    checkpoint_writable: bool | None = None
    free_disk_mb: int | None = None
    #: Whether the dataset directory resolves inside the trainer's own
    #: working directory. ACE-Step validates `--dataset-dir` against the
    #: working directory at import time and refuses anything outside it
    #: — after the model has loaded, which is the expensive place to
    #: find out.
    dataset_within_trainer_root: bool | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_readable": self.dataset_readable,
            "output_writable": self.output_writable,
            "checkpoint_writable": self.checkpoint_writable,
            "free_disk_mb": self.free_disk_mb,
            "dataset_within_trainer_root": self.dataset_within_trainer_root,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class RemoteEvidence:
    """What is known about reaching and trusting a remote worker."""

    #: Seconds since the worker's capability report was written.
    #: ``None`` means it has never reported, or the timestamp could not
    #: be read — both of which are UNKNOWN, never fresh.
    capability_age_seconds: float | None = None
    reachable: bool | None = None
    transport: str | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_age_seconds": self.capability_age_seconds,
            "reachable": self.reachable,
            "transport": self.transport,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CanaryEvidence:
    """The outcome of a bounded canary, when one has been run.

    Supplied by :mod:`luber_training.canary`. A preflight never runs a
    canary itself — it reports one — because a check that could start a
    trainer would be a check nobody could safely run twice.
    """

    #: One of ``PASSED``, ``FAILED``, ``BLOCKED``, ``NOT_RUN``.
    status: str = "NOT_RUN"
    mode: str | None = None
    detail: str = ""
    steps: int | None = None
    checkpoint_ok: bool | None = None
    checkpoint_detail: str = ""
    resume_ok: bool | None = None
    resume_detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "mode": self.mode,
            "detail": self.detail,
            "steps": self.steps,
            "checkpoint_ok": self.checkpoint_ok,
            "checkpoint_detail": self.checkpoint_detail,
            "resume_ok": self.resume_ok,
            "resume_detail": self.resume_detail,
        }


@dataclass(frozen=True)
class PreflightRequest:
    """Everything a preflight reads. No I/O happens after this is built."""

    plan: TrainingPlan
    capability: MachineCapability
    execution_location: str = ExecutionLocation.LOCAL.value
    intent: str = PreflightIntent.CANARY.value
    worker: TrainingWorker | None = None
    gate_report: GateReport | None = None
    dataset: DatasetEvidence = field(default_factory=DatasetEvidence)
    trainer: TrainerEvidence = field(default_factory=TrainerEvidence)
    storage: StorageEvidence = field(default_factory=StorageEvidence)
    remote: RemoteEvidence = field(default_factory=RemoteEvidence)
    canary: CanaryEvidence = field(default_factory=CanaryEvidence)
    capacity: CapacityReport | None = None
    #: Phase 34's verdict on whether measured evidence permits a run of
    #: this configuration. ``None`` means nobody asked the qualifier,
    #: which is not the same as its having said no.
    capacity_decision: CapacityDecision | None = None
    capability_max_age_seconds: float = DEFAULT_CAPABILITY_MAX_AGE_SECONDS
    #: Fixed by the caller so an identical evidence set produces an
    #: identical result. Defaulting to "now" inside would make every
    #: preflight differ from every other one.
    measured_at: str | None = None


@dataclass(frozen=True)
class TrainingPreflightResult:
    """Whether this machine may start this plan, and what is unproven."""

    status: str
    intent: str
    run_id: str
    plan_id: str
    plan_digest: str
    execution_location: str | None
    execution_device: str | None
    torch_device: str | None
    resolved_precision: str | None
    optimizer: str
    worker_identity: str | None
    target_label: str
    capability_digest: str | None
    hardware_snapshot: dict[str, Any] = field(default_factory=dict)
    capacity: dict[str, Any] = field(default_factory=dict)
    checks: tuple[PreflightCheck, ...] = ()
    blocking_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    unverified: tuple[str, ...] = ()
    dataset_status: str = CheckStatus.UNKNOWN.value
    dependency_status: str = CheckStatus.UNKNOWN.value
    storage_status: str = CheckStatus.UNKNOWN.value
    checkpoint_status: str = CheckStatus.NOT_APPLICABLE.value
    canary_status: str = "NOT_RUN"
    capacity_status: str = EvidenceSource.UNKNOWN.value
    #: Phase 34's qualification, where one was supplied.
    capacity_qualification: str = CapacityQualification.UNVERIFIED.value
    capacity_decision: dict[str, Any] = field(default_factory=dict)
    measured_at: str = ""
    schema_version: str = TRAINING_PREFLIGHT_SCHEMA_VERSION
    policy_version: str = TRAINING_PREFLIGHT_POLICY_VERSION

    @property
    def ready(self) -> bool:
        return self.status == PreflightStatus.READY.value

    def group_status(self, group: str) -> str:
        """The worst outcome among a group's mandatory checks."""
        relevant = [check for check in self.checks if check.group == group]
        if not relevant:
            return CheckStatus.NOT_APPLICABLE.value
        if any(check.blocks for check in relevant):
            return CheckStatus.FAIL.value
        if any(check.unverifies for check in relevant):
            return CheckStatus.UNKNOWN.value
        if all(check.status == CheckStatus.NOT_APPLICABLE.value for check in relevant):
            return CheckStatus.NOT_APPLICABLE.value
        return CheckStatus.PASS.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_version": self.policy_version,
            "status": self.status,
            "intent": self.intent,
            "run_id": self.run_id,
            "plan_id": self.plan_id,
            "plan_digest": self.plan_digest,
            "execution_location": self.execution_location,
            "execution_device": self.execution_device,
            "torch_device": self.torch_device,
            "resolved_precision": self.resolved_precision,
            "optimizer": self.optimizer,
            "worker_identity": self.worker_identity,
            "target_label": self.target_label,
            "capability_digest": self.capability_digest,
            "hardware_snapshot": self.hardware_snapshot,
            "capacity": self.capacity,
            "checks": [check.to_dict() for check in self.checks],
            "blocking_reasons": list(self.blocking_reasons),
            "warnings": list(self.warnings),
            "unverified": list(self.unverified),
            "dataset_status": self.dataset_status,
            "dependency_status": self.dependency_status,
            "storage_status": self.storage_status,
            "checkpoint_status": self.checkpoint_status,
            "canary_status": self.canary_status,
            "capacity_status": self.capacity_status,
            "capacity_qualification": self.capacity_qualification,
            "capacity_decision": self.capacity_decision,
            "measured_at": self.measured_at,
            "note": (
                "UNVERIFIED is not READY. A mandatory check nobody could perform is "
                "reported as unperformed, and a run that requires it does not start."
            ),
        }

    def digest(self) -> str:
        """A fingerprint of the verdict, excluding when it was taken.

        Used to assert that the same evidence produces the same answer.
        The timestamp is excluded for the same reason a plan excludes
        its compile time: it is real information and it is not part of
        what was decided.
        """
        payload = {
            key: value for key, value in self.to_dict().items() if key not in {"measured_at"}
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            )
        ).hexdigest()

    def render(self) -> str:
        lines = [
            f"{self.status}  ({self.intent})  {self.execution_location} + "
            f"{self.execution_device}  precision={self.resolved_precision}"
        ]
        for reason in self.blocking_reasons:
            lines.append(f"  BLOCKED  {reason}")
        for item in self.unverified:
            lines.append(f"  UNVERIFIED  {item}")
        for warning in self.warnings:
            lines.append(f"  warning  {warning}")
        return "\n".join(lines)


# ── the checks ───────────────────────────────────────────────────────


def _plan_checks(request: PreflightRequest) -> list[PreflightCheck]:
    plan = request.plan
    requirements = plan.requirements
    checks: list[PreflightCheck] = []

    checks.append(
        PreflightCheck(
            name="plan.schema",
            status=(
                CheckStatus.PASS.value
                if plan.schema_version in SUPPORTED_PLAN_SCHEMA_VERSIONS
                else CheckStatus.FAIL.value
            ),
            detail=(
                plan.schema_version
                if plan.schema_version in SUPPORTED_PLAN_SCHEMA_VERSIONS
                else (
                    f"plan schema {plan.schema_version!r} is not one this build reads "
                    f"({', '.join(sorted(SUPPORTED_PLAN_SCHEMA_VERSIONS))}); a plan from a "
                    "newer LUBER is refused rather than read with its new fields ignored"
                )
            ),
            reason=(
                None
                if plan.schema_version in SUPPORTED_PLAN_SCHEMA_VERSIONS
                else BlockingReason.PLAN_INVALID.value
            ),
        )
    )

    digest = plan.digest()
    cited = bool(plan.run_id)
    checks.append(
        PreflightCheck(
            name="plan.digest",
            status=CheckStatus.PASS.value if cited else CheckStatus.FAIL.value,
            detail=(
                f"{digest[:16]} recomputed from the plan's own content"
                if cited
                else "the plan names no run, so nothing can cite it"
            ),
            reason=None if cited else BlockingReason.PLAN_INVALID.value,
        )
    )

    device = requirements.execution_device
    known_devices = {item.value for item in ComputeDevice}
    if device is None:
        checks.append(
            PreflightCheck(
                name="plan.execution_device",
                status=CheckStatus.UNKNOWN.value,
                detail=(
                    "the plan names no execution device, so the device it would run on is "
                    "derived from requires_cuda rather than chosen. Compile the plan against "
                    "a placement before starting it"
                ),
                reason=BlockingReason.PLAN_INVALID.value,
            )
        )
    elif device not in known_devices:
        checks.append(
            PreflightCheck(
                name="plan.execution_device",
                status=CheckStatus.FAIL.value,
                detail=f"{device!r} is not a compute device",
                reason=BlockingReason.PLAN_INVALID.value,
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="plan.execution_device",
                status=CheckStatus.PASS.value,
                detail=f"{device} (torch: {torch_device_string(device)})",
            )
        )

    contradictions = requirements.contradictions()
    checks.append(
        PreflightCheck(
            name="plan.coherence",
            status=CheckStatus.PASS.value if not contradictions else CheckStatus.FAIL.value,
            detail=(
                "location and device are independent and the requirement set agrees with itself"
                if not contradictions
                else "; ".join(contradictions)
            ),
            reason=None if not contradictions else BlockingReason.PLAN_INVALID.value,
        )
    )

    location = request.execution_location
    legal_location = location in {item.value for item in ExecutionLocation}
    checks.append(
        PreflightCheck(
            name="plan.execution_location",
            status=CheckStatus.PASS.value if legal_location else CheckStatus.FAIL.value,
            detail=(
                f"{location}; a location does not imply a device and a device does not "
                "imply a location"
                if legal_location
                else f"{location!r} is not an execution location"
            ),
            reason=None if legal_location else BlockingReason.PLAN_INVALID.value,
        )
    )

    precision = plan.config.precision
    supported = requirements.supported_precision
    precision_in_plan = precision == Precision.AUTO.value or precision in supported
    checks.append(
        PreflightCheck(
            name="plan.precision",
            status=CheckStatus.PASS.value if precision_in_plan else CheckStatus.FAIL.value,
            detail=(
                f"{precision} is within the plan's supported set"
                if precision_in_plan
                else f"{precision} is not in the plan's supported set {', '.join(supported)}"
            ),
            reason=None if precision_in_plan else BlockingReason.PRECISION_UNSUPPORTED.value,
        )
    )
    return checks


def _dataset_checks(request: PreflightRequest) -> list[PreflightCheck]:
    evidence = request.dataset
    plan = request.plan
    checks: list[PreflightCheck] = []
    report = request.gate_report

    if report is None:
        checks.append(
            PreflightCheck(
                name="dataset.gates",
                status=CheckStatus.UNKNOWN.value,
                detail=(
                    "no gate report was supplied, so rights, locks and leakage are "
                    "unestablished for this run"
                ),
                reason=BlockingReason.DATASET_NOT_READY.value,
            )
        )
    else:
        for result in report.results:
            failed = not result.passed
            reason = {
                "rights": BlockingReason.RIGHTS_BLOCKED.value,
                "evaluation_leakage": BlockingReason.EVALUATION_LEAKAGE.value,
                "dataset_lock": BlockingReason.DATASET_NOT_READY.value,
                "curation_lock": BlockingReason.DATASET_NOT_READY.value,
                "self_generated": BlockingReason.DATASET_NOT_READY.value,
            }.get(result.name, BlockingReason.DATASET_NOT_READY.value)
            checks.append(
                PreflightCheck(
                    name=f"dataset.gate.{result.name}",
                    status=CheckStatus.PASS.value if result.passed else CheckStatus.FAIL.value,
                    detail=result.detail,
                    reason=reason if failed else None,
                )
            )

    if evidence.manifest_present is None:
        checks.append(
            PreflightCheck(
                name="dataset.manifest",
                status=CheckStatus.UNKNOWN.value,
                detail="nobody looked for the curated manifest this plan cites",
                reason=BlockingReason.DATASET_NOT_READY.value,
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="dataset.manifest",
                status=(
                    CheckStatus.PASS.value if evidence.manifest_present else CheckStatus.FAIL.value
                ),
                detail=evidence.detail
                or (
                    "the curated manifest is present"
                    if evidence.manifest_present
                    else "the curated manifest this plan cites is not on this machine"
                ),
                reason=None
                if evidence.manifest_present
                else BlockingReason.DATASET_NOT_READY.value,
            )
        )

    count = evidence.eligible_sample_count
    if count is None:
        checks.append(
            PreflightCheck(
                name="dataset.samples",
                status=CheckStatus.UNKNOWN.value,
                detail="nobody counted the eligible training samples",
                reason=BlockingReason.DATASET_NOT_READY.value,
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="dataset.samples",
                status=CheckStatus.PASS.value if count > 0 else CheckStatus.FAIL.value,
                detail=(
                    f"{count} eligible training sample(s)"
                    if count > 0
                    else "the selection contains no eligible training samples"
                ),
                reason=None if count > 0 else BlockingReason.DATASET_NOT_READY.value,
            )
        )

    leaked = evidence.evaluation_only_count
    checks.append(
        PreflightCheck(
            name="dataset.evaluation_only",
            status=(
                CheckStatus.UNKNOWN.value
                if leaked is None
                else CheckStatus.PASS.value
                if leaked == 0
                else CheckStatus.FAIL.value
            ),
            detail=(
                "nobody checked the selection for evaluation-only material"
                if leaked is None
                else "no evaluation-only material in the selection"
                if leaked == 0
                else f"{leaked} evaluation-only sample(s) are in the training selection"
            ),
            reason=(
                BlockingReason.DATASET_NOT_READY.value
                if leaked is None
                else None
                if leaked == 0
                else BlockingReason.EVALUATION_LEAKAGE.value
            ),
        )
    )

    observed = evidence.observed_manifest_sha256
    expected = plan.dataset_ref.curated_manifest_sha256
    if observed is None:
        checks.append(
            PreflightCheck(
                name="dataset.drift",
                status=CheckStatus.UNKNOWN.value,
                detail=(
                    "the curated manifest on disk was not re-hashed, so whether the build "
                    "moved since the plan was compiled is unknown"
                ),
                reason=BlockingReason.MANIFEST_DRIFT.value,
            )
        )
    else:
        matched = observed == expected
        checks.append(
            PreflightCheck(
                name="dataset.drift",
                status=CheckStatus.PASS.value if matched else CheckStatus.FAIL.value,
                detail=(
                    f"the curated manifest still hashes to {observed[:16]}, as the plan cites"
                    if matched
                    else (
                        f"the plan cites {expected[:16]} and the manifest on disk hashes to "
                        f"{observed[:16]}; the build moved under the plan and this is a new run, "
                        "not an edited one"
                    )
                ),
                reason=None if matched else BlockingReason.MANIFEST_DRIFT.value,
            )
        )

    present = evidence.referenced_files_present
    if present is None and request.execution_location == ExecutionLocation.REMOTE.value:
        # The manifest holds the paths of the machine that built it. A
        # control plane dispatching to a rented Linux box cannot see
        # them, and reporting them missing from here would be a false
        # refusal. Phase 27's worker-side preflight is what establishes
        # that the files arrived, and it does so where it can look.
        checks.append(
            PreflightCheck(
                name="dataset.files",
                status=CheckStatus.NOT_APPLICABLE.value,
                detail=(
                    "the training files live on the worker; the worker's own preflight "
                    "re-hashes every staged artifact before the trainer starts"
                ),
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="dataset.files",
                status=(
                    CheckStatus.UNKNOWN.value
                    if present is None
                    else CheckStatus.PASS.value
                    if present
                    else CheckStatus.FAIL.value
                ),
                detail=(
                    "the files the trainer will open were not checked from here"
                    if present is None
                    else "every referenced file the trainer will open is present"
                    if present
                    else f"{evidence.missing_file_count or 'some'} referenced file(s) are missing"
                ),
                reason=(
                    BlockingReason.DATASET_NOT_READY.value
                    if present is None or not present
                    else None
                ),
            )
        )
    return checks


def _trainer_checks(request: PreflightRequest) -> list[PreflightCheck]:
    evidence = request.trainer
    plan = request.plan
    checks: list[PreflightCheck] = []

    for name, value, label in (
        ("trainer.root", evidence.trainer_root_present, "the trainer installation"),
        ("trainer.entrypoint", evidence.entrypoint_present, "the trainer entrypoint (train.py)"),
        ("trainer.interpreter", evidence.interpreter_present, "the configured interpreter"),
    ):
        checks.append(
            PreflightCheck(
                name=name,
                status=(
                    CheckStatus.UNKNOWN.value
                    if value is None
                    else CheckStatus.PASS.value
                    if value
                    else CheckStatus.FAIL.value
                ),
                detail=(
                    f"{label} was not looked for from here"
                    if value is None
                    else f"{label} is present"
                    if value
                    else f"{label} is not present where the deployment says it is"
                ),
                reason=None if value else BlockingReason.TRAINER_UNAVAILABLE.value,
            )
        )

    checks.append(
        PreflightCheck(
            name="trainer.torch",
            status=(
                CheckStatus.UNKNOWN.value
                if evidence.torch_importable is None
                else CheckStatus.PASS.value
                if evidence.torch_importable
                else CheckStatus.FAIL.value
            ),
            detail=(
                "nobody asked the trainer's interpreter whether it can import torch"
                if evidence.torch_importable is None
                else "torch imports in the trainer's interpreter"
                if evidence.torch_importable
                else "torch does not import in the trainer's interpreter"
            ),
            reason=None if evidence.torch_importable else BlockingReason.DEPENDENCY_MISSING.value,
        )
    )

    expected_commit = plan.config.ace_step_commit
    observed_commit = evidence.observed_ace_step_commit
    if observed_commit is None:
        checks.append(
            PreflightCheck(
                name="trainer.runtime_identity",
                status=CheckStatus.UNKNOWN.value,
                detail=(
                    "the installed ACE-Step revision could not be read, so the trainer this "
                    f"config was audited against ({expected_commit[:12]}) cannot be confirmed"
                ),
                reason=BlockingReason.TRAINER_UNAVAILABLE.value,
            )
        )
    else:
        matched = observed_commit == expected_commit
        checks.append(
            PreflightCheck(
                name="trainer.runtime_identity",
                status=CheckStatus.PASS.value if matched else CheckStatus.FAIL.value,
                detail=(
                    f"ACE-Step {observed_commit[:12]}, as the config was audited against"
                    if matched
                    else (
                        f"the config targets ACE-Step {expected_commit[:12]} and the machine "
                        f"has {observed_commit[:12]}; the flags this plan compiles to were "
                        "read from a different tree"
                    )
                ),
                reason=None if matched else BlockingReason.TRAINER_UNAVAILABLE.value,
            )
        )

    checks.append(
        PreflightCheck(
            name="trainer.command",
            status=(
                CheckStatus.UNKNOWN.value
                if evidence.command_accepted is None
                else CheckStatus.PASS.value
                if evidence.command_accepted
                else CheckStatus.FAIL.value
            ),
            detail=evidence.command_detail
            or (
                "the compiled command was not offered to the installed parser"
                if evidence.command_accepted is None
                else "the installed trainer accepts every flag this plan compiles to"
                if evidence.command_accepted
                else "the installed trainer rejects the compiled command"
            ),
            reason=None if evidence.command_accepted else BlockingReason.TRAINER_UNAVAILABLE.value,
        )
    )

    required_package = OPTIMIZER_PACKAGES.get(plan.config.optimizer_type)
    if required_package is None:
        checks.append(
            PreflightCheck(
                name="trainer.optimizer_dependency",
                status=CheckStatus.PASS.value,
                detail=f"{plan.config.optimizer_type} needs no optional package",
            )
        )
    elif required_package in evidence.missing_packages:
        checks.append(
            PreflightCheck(
                name="trainer.optimizer_dependency",
                status=CheckStatus.FAIL.value,
                detail=(
                    f"{plan.config.optimizer_type} needs {required_package}, which the "
                    "trainer's interpreter does not have. ACE-Step catches the ImportError "
                    "and uses AdamW instead, so this run would train with an optimizer its "
                    "own plan does not record"
                ),
                reason=BlockingReason.DEPENDENCY_MISSING.value,
            )
        )
    elif required_package in evidence.unchecked_packages or evidence.torch_importable is None:
        checks.append(
            PreflightCheck(
                name="trainer.optimizer_dependency",
                status=CheckStatus.UNKNOWN.value,
                detail=(
                    f"nobody could check whether {required_package} is installed, and a "
                    "missing one is silently substituted rather than raised"
                ),
                reason=BlockingReason.DEPENDENCY_MISSING.value,
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="trainer.optimizer_dependency",
                status=CheckStatus.PASS.value,
                detail=f"{required_package} is importable in the trainer's interpreter",
            )
        )
    return checks


def _hardware_checks(request: PreflightRequest) -> list[PreflightCheck]:
    plan = request.plan
    capability = request.capability
    device = plan.requirements.execution_device
    checks: list[PreflightCheck] = []

    if device is None or device not in {item.value for item in ComputeDevice}:
        # The plan check already refused. Reporting a device question
        # about a device nobody named would be noise.
        checks.append(
            PreflightCheck(
                name="hardware.device",
                status=CheckStatus.UNKNOWN.value,
                detail="the plan names no resolvable device, so no machine can be checked for it",
                reason=BlockingReason.DEVICE_UNAVAILABLE.value,
            )
        )
        return checks

    present = capability.has_device(device)
    checks.append(
        PreflightCheck(
            name="hardware.device",
            status=CheckStatus.PASS.value if present else CheckStatus.FAIL.value,
            detail=(
                f"{capability.label} offers {device}"
                if present
                else (
                    f"{capability.label} does not offer {device}. This is not downgraded to "
                    "another device: a named device is a statement about what the run needs"
                )
            ),
            reason=None if present else BlockingReason.DEVICE_UNAVAILABLE.value,
        )
    )

    decision = resolve_precision(
        capability,
        device=device,
        requested=plan.config.precision,
        allow_unverified=False,
    )
    if not decision.resolved:
        status = CheckStatus.UNKNOWN.value if decision.unverified else CheckStatus.FAIL.value
        checks.append(
            PreflightCheck(
                name="hardware.precision",
                status=status,
                detail=decision.reason,
                reason=BlockingReason.PRECISION_UNSUPPORTED.value,
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="hardware.precision",
                status=(
                    CheckStatus.UNKNOWN.value if decision.unverified else CheckStatus.PASS.value
                ),
                detail=decision.reason,
                reason=(
                    BlockingReason.PRECISION_UNSUPPORTED.value if decision.unverified else None
                ),
            )
        )

    # What the *trainer* will do with this precision, as distinct from
    # what the hardware can hold. `auto` is resolved first, because the
    # combination that fails is the one `auto` picks.
    effective = plan.config.precision
    if effective == Precision.AUTO.value:
        effective = AUTO_BY_DEVICE.get(device, Precision.FP32.value)
    obstacle = UNTRAINABLE_PRECISION.get((device, effective))
    checks.append(
        PreflightCheck(
            name="hardware.trainer_precision",
            status=CheckStatus.PASS.value if obstacle is None else CheckStatus.FAIL.value,
            detail=(
                f"the trainer trains in {effective} on {device}"
                + (
                    " (resolved from 'auto')"
                    if plan.config.precision == Precision.AUTO.value
                    else ""
                )
                if obstacle is None
                else obstacle
            ),
            reason=None if obstacle is None else BlockingReason.PRECISION_UNSUPPORTED.value,
        )
    )

    optimizer = plan.config.optimizer_type
    if optimizer in CUDA_ONLY_OPERATIONS and device != ComputeDevice.CUDA.value:
        checks.append(
            PreflightCheck(
                name="hardware.optimizer",
                status=CheckStatus.FAIL.value,
                detail=(
                    f"{optimizer} is a CUDA-only path and this plan runs on {device}. "
                    "The trainer would substitute AdamW and train something other than "
                    "what the plan says"
                ),
                reason=BlockingReason.OPTIMIZER_UNSUPPORTED.value,
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="hardware.optimizer",
                status=CheckStatus.PASS.value,
                detail=f"{optimizer} is legal on {device}",
            )
        )

    gpu_count = capability.cuda_device_count
    needed = plan.requirements.minimum_gpu_count
    if device == ComputeDevice.CUDA.value:
        if gpu_count is None:
            checks.append(
                PreflightCheck(
                    name="hardware.device_count",
                    status=CheckStatus.UNKNOWN.value,
                    detail=f"{capability.label} has not reported how many GPUs it has",
                    reason=BlockingReason.DEVICE_UNAVAILABLE.value,
                )
            )
        else:
            checks.append(
                PreflightCheck(
                    name="hardware.device_count",
                    status=(
                        CheckStatus.PASS.value if gpu_count >= needed else CheckStatus.FAIL.value
                    ),
                    detail=f"{gpu_count} GPU(s) present, {needed} required",
                    reason=(
                        None if gpu_count >= needed else BlockingReason.DEVICE_UNAVAILABLE.value
                    ),
                )
            )
    else:
        checks.append(
            PreflightCheck(
                name="hardware.device_count",
                status=CheckStatus.NOT_APPLICABLE.value,
                detail=f"{device} is a single-device backend on this target",
            )
        )
    return checks


def _storage_checks(request: PreflightRequest) -> list[PreflightCheck]:
    evidence = request.storage
    checks: list[PreflightCheck] = []

    for name, value, label in (
        ("storage.dataset_readable", evidence.dataset_readable, "the training input"),
        ("storage.output_writable", evidence.output_writable, "the output directory"),
        (
            "storage.checkpoint_writable",
            evidence.checkpoint_writable,
            "the checkpoint directory",
        ),
    ):
        checks.append(
            PreflightCheck(
                name=name,
                status=(
                    CheckStatus.UNKNOWN.value
                    if value is None
                    else CheckStatus.PASS.value
                    if value
                    else CheckStatus.FAIL.value
                ),
                detail=(
                    f"{label} was not checked from here"
                    if value is None
                    else f"{label} is usable"
                    if value
                    else f"{label} is not usable"
                ),
                reason=None if value else BlockingReason.STORAGE_UNAVAILABLE.value,
            )
        )

    inside = evidence.dataset_within_trainer_root
    checks.append(
        PreflightCheck(
            name="storage.trainer_path_safety",
            status=(
                CheckStatus.UNKNOWN.value
                if inside is None
                else CheckStatus.PASS.value
                if inside
                else CheckStatus.FAIL.value
            ),
            detail=(
                "nobody compared the dataset directory with the trainer's working directory"
                if inside is None
                else "the dataset directory is inside the trainer's working directory"
                if inside
                else (
                    "the dataset directory is outside the trainer's working directory. "
                    "ACE-Step's path_safety refuses --dataset-dir anywhere else and does "
                    "so after loading the model, so this fails expensively rather than "
                    "quickly"
                )
            ),
            reason=None if inside else BlockingReason.STORAGE_UNAVAILABLE.value,
            # Not mandatory when nobody looked from here: a control
            # plane dispatching to a worker cannot compare two paths it
            # cannot see, and the worker checks them where it can.
            mandatory=inside is not None,
        )
    )

    required = None if request.capacity is None else request.capacity.by_name(REQUIRED_DISK)
    free = evidence.free_disk_mb
    if free is None:
        checks.append(
            PreflightCheck(
                name="storage.free_disk",
                status=CheckStatus.UNKNOWN.value,
                detail="free disk could not be measured on the execution target",
                reason=BlockingReason.INSUFFICIENT_DISK.value,
            )
        )
    elif required is None or not required.known or required.value_mb is None:
        checks.append(
            PreflightCheck(
                name="storage.free_disk",
                status=CheckStatus.PASS.value,
                detail=(
                    f"{free} MB free. No disk requirement has been derived for this "
                    "configuration, so this is reported rather than compared — the "
                    "requirement is UNKNOWN and is recorded as capacity evidence"
                ),
                # Not mandatory: an unmeasured requirement is already
                # carried as UNKNOWN capacity evidence, and blocking
                # twice for one gap would report the same hole as two.
                mandatory=False,
            )
        )
    elif free < required.value_mb:
        checks.append(
            PreflightCheck(
                name="storage.free_disk",
                status=CheckStatus.FAIL.value,
                detail=(f"{free} MB free, {required.value_mb} MB required ({required.derivation})"),
                reason=BlockingReason.INSUFFICIENT_DISK.value,
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="storage.free_disk",
                status=CheckStatus.PASS.value,
                detail=f"{free} MB free, {required.value_mb} MB required",
            )
        )
    return checks


def _remote_checks(request: PreflightRequest) -> list[PreflightCheck]:
    """Everything that only matters when the work leaves this machine."""
    if request.execution_location != ExecutionLocation.REMOTE.value:
        return [
            PreflightCheck(
                name="remote.transport",
                status=CheckStatus.NOT_APPLICABLE.value,
                detail="this run executes locally; there is no worker to reach",
            )
        ]

    worker = request.worker
    evidence = request.remote
    checks: list[PreflightCheck] = []

    if worker is None:
        checks.append(
            PreflightCheck(
                name="remote.worker",
                status=CheckStatus.FAIL.value,
                detail="this run is remote and no worker is assigned to it",
                reason=BlockingReason.WORKER_UNAVAILABLE.value,
            )
        )
        return checks

    checks.append(
        PreflightCheck(
            name="remote.worker",
            status=CheckStatus.PASS.value,
            detail=f"{worker.name} ({worker.worker_class})",
        )
    )

    device = request.plan.requirements.execution_device
    if device == ComputeDevice.CUDA.value:
        eligible = worker.worker_class == WorkerClass.GPU_TRAINING_READY.value
        checks.append(
            PreflightCheck(
                name="remote.worker_class",
                status=CheckStatus.PASS.value if eligible else CheckStatus.FAIL.value,
                detail=(
                    f"{worker.name} was verified GPU_TRAINING_READY by a probe"
                    if eligible
                    else (
                        f"{worker.name} is {worker.worker_class}; only a worker a probe "
                        "verified as CUDA-capable may take a CUDA plan"
                    )
                ),
                reason=None if eligible else BlockingReason.WORKER_UNAVAILABLE.value,
            )
        )
        reported = worker.capabilities.cuda_available
        checks.append(
            PreflightCheck(
                name="remote.device_identity",
                status=(
                    CheckStatus.UNKNOWN.value
                    if reported is None
                    else CheckStatus.PASS.value
                    if reported
                    else CheckStatus.FAIL.value
                ),
                detail=(
                    f"{worker.name} has never reported CUDA availability"
                    if reported is None
                    else (
                        f"{worker.name} reports CUDA "
                        f"({worker.capabilities.gpu_model or 'model unreported'})"
                    )
                    if reported
                    else f"{worker.name} reports no CUDA"
                ),
                reason=None if reported else BlockingReason.DEVICE_UNAVAILABLE.value,
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="remote.worker_class",
                status=CheckStatus.NOT_APPLICABLE.value,
                detail=f"a {device} plan does not require a CUDA-verified worker",
            )
        )

    age = evidence.capability_age_seconds
    limit = request.capability_max_age_seconds
    if age is None:
        checks.append(
            PreflightCheck(
                name="remote.capability_freshness",
                status=CheckStatus.UNKNOWN.value,
                detail=(
                    f"{worker.name} has no readable capability timestamp, so whether its "
                    "report still describes the machine is unknown"
                ),
                reason=BlockingReason.WORKER_STALE.value,
            )
        )
    elif age > limit:
        checks.append(
            PreflightCheck(
                name="remote.capability_freshness",
                status=CheckStatus.FAIL.value,
                detail=(
                    f"{worker.name} was last probed {int(age)}s ago and the policy allows "
                    f"{int(limit)}s. Re-probe it: a rented machine can be returned, resized "
                    "or have its driver changed between sessions"
                ),
                reason=BlockingReason.WORKER_STALE.value,
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="remote.capability_freshness",
                status=CheckStatus.PASS.value,
                detail=f"probed {int(age)}s ago, within the {int(limit)}s policy",
            )
        )

    reachable = evidence.reachable
    checks.append(
        PreflightCheck(
            name="remote.transport",
            status=(
                CheckStatus.UNKNOWN.value
                if reachable is None
                else CheckStatus.PASS.value
                if reachable
                else CheckStatus.FAIL.value
            ),
            detail=evidence.detail
            or (
                "no transport to the worker is configured on this machine, so reachability "
                "could not be established"
                if reachable is None
                else f"the worker answered over {evidence.transport or 'the configured transport'}"
                if reachable
                else "the worker did not answer"
            ),
            reason=(
                BlockingReason.REMOTE_UNREACHABLE.value
                if reachable is None or not reachable
                else None
            ),
        )
    )
    return checks


def _capacity_checks(request: PreflightRequest) -> list[PreflightCheck]:
    report = request.capacity
    if report is None:
        return [
            PreflightCheck(
                name="capacity.evidence",
                status=CheckStatus.UNKNOWN.value,
                detail="no capacity evidence was collected for this target",
                reason=BlockingReason.CAPACITY_UNVERIFIED.value,
                mandatory=request.intent == PreflightIntent.FULL_TRAINING.value,
            )
        ]

    checks: list[PreflightCheck] = []
    device_memory = report.by_name(DEVICE_MEMORY)
    if device_memory is None or not device_memory.known:
        checks.append(
            PreflightCheck(
                name="capacity.device_memory",
                status=CheckStatus.UNKNOWN.value,
                detail="the target has not reported how much memory the selected device has",
                reason=BlockingReason.CAPACITY_UNVERIFIED.value,
            )
        )
    else:
        checks.append(
            PreflightCheck(
                name="capacity.device_memory",
                status=CheckStatus.PASS.value,
                detail=device_memory.detail,
            )
        )

    checks.append(_qualification_check(request, report))
    return checks


def _qualification_check(
    request: PreflightRequest, report: CapacityReport | None
) -> PreflightCheck:
    """Whether measured evidence permits a run of this configuration.

    Phase 33 could only ever answer UNKNOWN here, because nothing had
    measured what a run costs. Phase 34 gives the question an answer
    when — and only when — an applicable, completed, representative
    profile exists whose peak satisfies the capacity policy.

    Mandatory for FULL_TRAINING and informational for a canary. A canary
    runs bounded on synthetic tensors; requiring a production memory
    figure before it could start would block the one run that produces
    the figure.
    """
    mandatory = request.intent == PreflightIntent.FULL_TRAINING.value
    decision = request.capacity_decision

    if decision is None:
        requirement = None if report is None else report.by_name(TRAINING_MEMORY_REQUIREMENT)
        return PreflightCheck(
            name="capacity.training_requirement",
            status=CheckStatus.UNKNOWN.value,
            detail=(
                (requirement.detail if requirement is not None else "")
                or "no memory requirement has been measured for this configuration"
            ),
            reason=BlockingReason.CAPACITY_UNVERIFIED.value,
            mandatory=mandatory,
        )

    if decision.qualification == CapacityQualification.INSUFFICIENT.value:
        return PreflightCheck(
            name="capacity.training_requirement",
            status=CheckStatus.FAIL.value,
            detail=(
                "measured evidence says this configuration cannot satisfy the capacity "
                "policy on this machine: " + "; ".join(decision.reasons[:3])
            ),
            # A definite refusal, so it blocks whatever the intent. A
            # canary whose own configuration is known not to fit should
            # not be started either.
            reason=BlockingReason.CAPACITY_UNVERIFIED.value,
        )

    if decision.qualification == CapacityQualification.UNVERIFIED.value:
        return PreflightCheck(
            name="capacity.training_requirement",
            status=CheckStatus.UNKNOWN.value,
            detail=(
                "no applicable memory profile qualifies this configuration: "
                + (decision.applicability_detail or "; ".join(decision.reasons[:2]))
            ),
            reason=BlockingReason.CAPACITY_UNVERIFIED.value,
            mandatory=mandatory,
        )

    return PreflightCheck(
        name="capacity.training_requirement",
        status=CheckStatus.PASS.value,
        detail=(
            f"{decision.qualification} against policy {decision.policy_version}, from "
            f"profile {decision.profile_id}: " + "; ".join(decision.reasons[:2])
        ),
    )


def _canary_checks(request: PreflightRequest) -> list[PreflightCheck]:
    evidence = request.canary
    if evidence.status == "NOT_RUN":
        return [
            PreflightCheck(
                name="canary.run",
                status=CheckStatus.NOT_APPLICABLE.value,
                detail="no canary has been run for this plan",
            )
        ]

    checks: list[PreflightCheck] = [
        PreflightCheck(
            name="canary.run",
            status=(
                CheckStatus.PASS.value
                if evidence.status == "PASSED"
                else CheckStatus.UNKNOWN.value
                if evidence.status == "BLOCKED"
                else CheckStatus.FAIL.value
            ),
            detail=evidence.detail,
            reason=None if evidence.status == "PASSED" else BlockingReason.CANARY_FAILED.value,
        )
    ]
    if evidence.checkpoint_ok is not None:
        checks.append(
            PreflightCheck(
                name="checkpoint.integrity",
                status=(
                    CheckStatus.PASS.value if evidence.checkpoint_ok else CheckStatus.FAIL.value
                ),
                detail=evidence.checkpoint_detail,
                reason=None if evidence.checkpoint_ok else BlockingReason.CHECKPOINT_FAILED.value,
            )
        )
    if evidence.resume_ok is not None:
        checks.append(
            PreflightCheck(
                name="checkpoint.resume",
                status=CheckStatus.PASS.value if evidence.resume_ok else CheckStatus.FAIL.value,
                detail=evidence.resume_detail,
                reason=None if evidence.resume_ok else BlockingReason.RESUME_FAILED.value,
            )
        )
    return checks


# ── the whole thing ──────────────────────────────────────────────────


def evaluate(request: PreflightRequest) -> TrainingPreflightResult:
    """Every check, against evidence somebody else collected."""
    checks: list[PreflightCheck] = []
    checks.extend(_plan_checks(request))
    checks.extend(_dataset_checks(request))
    checks.extend(_trainer_checks(request))
    checks.extend(_hardware_checks(request))
    checks.extend(_storage_checks(request))
    checks.extend(_remote_checks(request))
    checks.extend(_capacity_checks(request))
    checks.extend(_canary_checks(request))

    blocking = tuple(
        f"{check.reason or BlockingReason.PLAN_INVALID.value}: {check.name}: {check.detail}"
        for check in checks
        if check.blocks
    )
    unverified = tuple(
        f"{check.reason or BlockingReason.CAPACITY_UNVERIFIED.value}: {check.name}: {check.detail}"
        for check in checks
        if check.unverifies
    )
    warnings = tuple(
        f"{check.name}: {check.detail}"
        for check in checks
        if not check.mandatory
        and check.status in (CheckStatus.FAIL.value, CheckStatus.UNKNOWN.value)
    )

    if blocking:
        status = PreflightStatus.BLOCKED.value
    elif unverified:
        status = PreflightStatus.UNVERIFIED.value
    else:
        status = PreflightStatus.READY.value

    plan = request.plan
    device = plan.requirements.execution_device
    precision = resolve_precision(
        request.capability,
        device=device or ComputeDevice.CPU.value,
        requested=plan.config.precision,
        allow_unverified=True,
    )
    capacity = request.capacity
    result = TrainingPreflightResult(
        status=status,
        intent=request.intent,
        run_id=plan.run_id,
        plan_id=plan.plan_id,
        plan_digest=plan.digest(),
        execution_location=request.execution_location,
        execution_device=device,
        torch_device=(
            torch_device_string(device)
            if device in {item.value for item in ComputeDevice}
            else None
        ),
        resolved_precision=precision.precision,
        optimizer=plan.config.optimizer_type,
        worker_identity=None if request.worker is None else request.worker.worker_id,
        target_label=request.capability.label,
        capability_digest=request.capability.digest(),
        hardware_snapshot=_hardware_snapshot(request.capability, device),
        capacity={} if capacity is None else capacity.to_dict(),
        checks=tuple(checks),
        blocking_reasons=blocking,
        warnings=warnings,
        unverified=unverified,
        canary_status=request.canary.status,
        capacity_qualification=(
            CapacityQualification.UNVERIFIED.value
            if request.capacity_decision is None
            else request.capacity_decision.qualification
        ),
        capacity_decision=(
            {} if request.capacity_decision is None else request.capacity_decision.to_dict()
        ),
        capacity_status=(
            EvidenceSource.UNKNOWN.value
            if capacity is None or capacity.any_unknown
            else EvidenceSource.MEASURED.value
        ),
        measured_at=request.measured_at or datetime.now(UTC).isoformat(),
    )
    # The per-area statuses are derived from the same checks rather than
    # tracked separately, so a console row can never disagree with the
    # verdict it sits beside.
    return replace(
        result,
        dataset_status=result.group_status("dataset"),
        dependency_status=result.group_status("trainer"),
        storage_status=result.group_status("storage"),
        checkpoint_status=result.group_status("checkpoint"),
    )


def _hardware_snapshot(capability: MachineCapability, device: str | None) -> dict[str, Any]:
    """The hardware facts worth carrying on a preflight record.

    A subset rather than the whole capability report: this record is
    about a decision, and a reader who wants the full machine has the
    capability digest to look it up by. Nothing here identifies a host.
    """
    return {
        "label": capability.label,
        "system": capability.system,
        "architecture": capability.architecture,
        "apple_silicon": capability.apple_silicon,
        "cpu_model": capability.cpu_model,
        "cpu_count": capability.cpu_count,
        "memory_total_mb": capability.memory_total_mb,
        "python_version": capability.python_version,
        "torch_version": capability.torch_version,
        "mps_built": capability.mps_built,
        "mps_available": capability.mps_available,
        "cuda_available": capability.cuda_available,
        "cuda_version": capability.cuda_version,
        "cuda_device_name": capability.cuda_device_name,
        "cuda_device_count": capability.cuda_device_count,
        "cuda_device_memory_mb": capability.cuda_device_memory_mb,
        "devices": list(capability.devices()),
        "selected_device": device,
        "capability_digest": capability.digest(),
    }


__all__ = [
    "DEFAULT_CAPABILITY_MAX_AGE_SECONDS",
    "GROUPS",
    "OPTIMIZER_PACKAGES",
    "SUPPORTED_PLAN_SCHEMA_VERSIONS",
    "TRAINING_PREFLIGHT_POLICY_VERSION",
    "TRAINING_PREFLIGHT_SCHEMA_VERSION",
    "UNTRAINABLE_PRECISION",
    "BlockingReason",
    "CanaryEvidence",
    "CheckStatus",
    "DatasetEvidence",
    "PreflightCheck",
    "PreflightIntent",
    "PreflightRequest",
    "PreflightStatus",
    "RemoteEvidence",
    "StorageEvidence",
    "TrainerEvidence",
    "TrainingPreflightResult",
    "evaluate",
]
