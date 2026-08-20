"""Execution backends: where orchestration stops and hardware begins.

The whole point of the abstraction is that nothing above it knows about
CUDA, SSH, RunPod or any other vendor. Orchestration decides *what* to
train; a backend decides *how* and *where*. Swapping a rented A100 for a
different provider must not touch a gate, a registry or a plan.

Two implementations here, and they are honest about which is which.

**LocalDryRunBackend** validates orchestration without training. It
walks the real lifecycle, emits metrics clearly marked ``SIMULATED``,
and produces no model. Where a test needs an artifact it registers one
of kind ``MOCK`` — a distinct kind rather than a flag, so no query for a
real checkpoint can return it by accident. A dry run that produced
something indistinguishable from a trained adapter would be the single
most dangerous thing in this package.

**RemoteGpuBackend** is a contract with no implementation. Phase 25
connects to nothing. The class exists so the interface is settled before
a provider is chosen, and every method raises rather than pretending.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from luber_training.entities import RunStatus, TrainingWorker, WorkerClass
from luber_training.metrics import MetricEvent, MetricSource
from luber_training.plan import TrainingPlan

#: Backend identifiers recorded on a run.
DRY_RUN = "dry-run"
REMOTE_GPU = "remote-gpu"


@dataclass
class EnvironmentCheck:
    """Whether a backend can execute a plan, and what is unknown.

    ``unknown`` is a first-class outcome, not a soft pass. A preflight
    that turned "nobody measured VRAM" into a tick would be the same
    failure as inventing the number.
    """

    ok: bool
    problems: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "problems": self.problems, "unknown": self.unknown}


@dataclass
class BackendStatus:
    status: str
    detail: str = ""
    exit_code: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "detail": self.detail, "exit_code": self.exit_code}


class TrainingExecutionBackend(ABC):
    """The contract every backend honours."""

    name: str = "abstract"

    @abstractmethod
    def validate_environment(self, plan: TrainingPlan, worker: TrainingWorker) -> EnvironmentCheck:
        """Can this worker execute this plan?"""

    @abstractmethod
    def prepare_run(self, plan: TrainingPlan, worker: TrainingWorker) -> None:
        """Stage whatever the run needs before it starts."""

    @abstractmethod
    def start(self, plan: TrainingPlan, worker: TrainingWorker) -> BackendStatus:
        """Begin execution. Must be safe to call only once per run."""

    @abstractmethod
    def status(self, plan: TrainingPlan) -> BackendStatus:
        """Current execution state, as the backend understands it."""

    @abstractmethod
    def cancel(self, plan: TrainingPlan) -> BackendStatus:
        """Request graceful termination. Never deletes outputs."""

    @abstractmethod
    def collect_metrics(self, plan: TrainingPlan) -> list[MetricEvent]:
        """Metrics produced since the last collection."""

    @abstractmethod
    def collect_checkpoints(self, plan: TrainingPlan) -> list[dict[str, Any]]:
        """Checkpoint descriptors the backend has finished writing."""

    @abstractmethod
    def cleanup(self, plan: TrainingPlan) -> None:
        """Release transient resources. Never removes checkpoints or logs."""


def capability_check(plan: TrainingPlan, worker: TrainingWorker) -> EnvironmentCheck:
    """Match a plan's requirements against a worker's reported facts.

    Shared by every backend because the question is the same everywhere
    and having two answers would be worse than having none.

    An unmeasured capability is reported as unknown and does **not**
    pass. A worker that has never been probed cannot satisfy a CUDA
    requirement by virtue of nobody having checked.
    """
    requirements = plan.requirements
    capabilities = worker.capabilities
    problems: list[str] = []
    unknown: list[str] = []

    if requirements.requires_cuda:
        if worker.worker_class == WorkerClass.DEVELOPMENT_ONLY.value:
            problems.append(f"worker {worker.name} is DEVELOPMENT_ONLY and the plan requires CUDA")
        if capabilities.cuda_available is None:
            problems.append(
                f"worker {worker.name} has never reported CUDA availability; "
                "run the capability probe before scheduling a CUDA plan"
            )
        elif not capabilities.cuda_available:
            problems.append(f"worker {worker.name} reports no CUDA")

    if capabilities.gpu_count is None:
        if requirements.minimum_gpu_count > 0 and requirements.requires_cuda:
            problems.append(f"worker {worker.name} has not reported a GPU count")
    elif capabilities.gpu_count < requirements.minimum_gpu_count:
        problems.append(
            f"plan needs {requirements.minimum_gpu_count} GPU(s); worker reports "
            f"{capabilities.gpu_count}"
        )

    if requirements.minimum_vram_mb is None:
        unknown.append(
            "minimum_vram_mb is UNKNOWN_REQUIREMENT: no VRAM figure has been measured, "
            "so memory sufficiency cannot be checked"
        )
    elif capabilities.vram_total_mb is None:
        unknown.append(f"worker {worker.name} has not reported VRAM")
    elif capabilities.vram_total_mb < requirements.minimum_vram_mb:
        problems.append(
            f"plan needs {requirements.minimum_vram_mb} MB VRAM; worker reports "
            f"{capabilities.vram_total_mb} MB"
        )

    precision = plan.config.precision
    if precision != "auto" and precision not in requirements.supported_precision:
        problems.append(f"precision {plan.config.precision} is not in the plan's supported set")
    if plan.config.precision == "bf16" and capabilities.bf16_supported is False:
        problems.append(f"worker {worker.name} reports no bf16 support")

    return EnvironmentCheck(ok=not problems, problems=problems, unknown=unknown)


class LocalDryRunBackend(TrainingExecutionBackend):
    """Exercises the orchestration lifecycle without training anything.

    Deterministic: the same plan produces the same sequence of states
    and the same simulated metrics, so tests assert behaviour rather
    than tolerate noise.
    """

    name = DRY_RUN

    def __init__(self, *, steps: int = 5, step_delay_seconds: float = 0.0) -> None:
        self.steps = steps
        self.step_delay_seconds = step_delay_seconds
        self._state: dict[str, str] = {}
        self._cancelled: set[str] = set()

    def validate_environment(self, plan: TrainingPlan, worker: TrainingWorker) -> EnvironmentCheck:
        """A dry run needs no hardware — but it must not launder one.

        If the plan requires CUDA, the same capability check runs as for
        a real backend. Otherwise "it passed on dry-run" would become
        evidence that a development Mac could take a GPU job.
        """
        if plan.requirements.requires_cuda:
            return capability_check(plan, worker)
        return EnvironmentCheck(
            ok=True,
            problems=[],
            unknown=["this is a dry run: no training occurs and no hardware is exercised"],
        )

    def prepare_run(self, plan: TrainingPlan, worker: TrainingWorker) -> None:
        self._state[plan.run_id] = RunStatus.QUEUED.value

    def start(self, plan: TrainingPlan, worker: TrainingWorker) -> BackendStatus:
        self._state[plan.run_id] = RunStatus.RUNNING.value
        return BackendStatus(status=RunStatus.RUNNING.value, detail="dry run started")

    def status(self, plan: TrainingPlan) -> BackendStatus:
        if plan.run_id in self._cancelled:
            return BackendStatus(status=RunStatus.CANCELLED.value, detail="cancelled by operator")
        return BackendStatus(status=self._state.get(plan.run_id, RunStatus.DRAFT.value))

    def cancel(self, plan: TrainingPlan) -> BackendStatus:
        self._cancelled.add(plan.run_id)
        self._state[plan.run_id] = RunStatus.CANCELLED.value
        return BackendStatus(status=RunStatus.CANCELLED.value, detail="cancellation requested")

    def collect_metrics(self, plan: TrainingPlan) -> list[MetricEvent]:
        """Synthetic infrastructure metrics, every one marked SIMULATED.

        No loss values. A simulated `train_loss` would sit in the same
        column as a real one and eventually be plotted next to it, and
        there is no honest number to put there — nothing computed a
        gradient.
        """
        events: list[MetricEvent] = []
        for step in range(1, self.steps + 1):
            if plan.run_id in self._cancelled:
                break
            if self.step_delay_seconds:
                time.sleep(self.step_delay_seconds)
            events.append(
                MetricEvent(
                    run_id=plan.run_id,
                    step=step,
                    epoch=1,
                    metric_name="step_time_seconds",
                    value=0.0,
                    unit="seconds",
                    source=MetricSource.SIMULATED.value,
                )
            )
            events.append(
                MetricEvent(
                    run_id=plan.run_id,
                    step=step,
                    epoch=1,
                    metric_name="samples_per_second",
                    value=0.0,
                    unit="samples/s",
                    source=MetricSource.SIMULATED.value,
                )
            )
        return events

    def collect_checkpoints(self, plan: TrainingPlan) -> list[dict[str, Any]]:
        """Nothing. A dry run trains nothing and so produces nothing.

        A caller that needs an artifact registers a MOCK checkpoint
        deliberately and by name. It is not this backend's business to
        hand one out.
        """
        return []

    def cleanup(self, plan: TrainingPlan) -> None:
        self._state.pop(plan.run_id, None)


class RemoteGpuBackendNotImplementedError(NotImplementedError):
    """Raised by every RemoteGpuBackend method.

    Phase 25 establishes the contract and connects to nothing. A
    partially-working remote backend would be worse than none: it would
    invite someone to try it.
    """


class RemoteGpuBackend(TrainingExecutionBackend):
    """The contract a future GPU host will satisfy.

    Deliberately provider-neutral. RunPod, Vast.ai, Lambda, AWS and a
    bare SSH box differ in provisioning and billing, not in what
    orchestration needs from them: validate, stage, start, poll, cancel,
    collect, clean up. Binding those seven verbs now means choosing a
    provider later is a configuration decision rather than a rewrite.

    Implementing it will require, at minimum:

    * a transport (SSH or a provider API) resolved from a
      ``credential_ref``, never from a value stored here;
    * artifact transfer in both directions — dataset out, checkpoints
      and metrics back;
    * a heartbeat, so a vanished worker becomes ``LOST`` rather than
      silently ``RUNNING`` forever;
    * graceful cancellation that terminates the trainer without
      destroying completed checkpoints or logs.
    """

    name = REMOTE_GPU

    def __init__(self, *, host_ref: str, credential_ref: str) -> None:
        # References only. If this constructor ever accepts a key or a
        # token, the secrets boundary has been broken.
        self.host_ref = host_ref
        self.credential_ref = credential_ref

    def _unimplemented(self, what: str) -> BackendStatus:
        raise RemoteGpuBackendNotImplementedError(
            f"RemoteGpuBackend.{what} is a Phase 25 contract with no implementation; "
            "no remote host is contacted in this phase"
        )

    def validate_environment(self, plan: TrainingPlan, worker: TrainingWorker) -> EnvironmentCheck:
        # The one method with real behaviour: capability matching is
        # provider-independent arithmetic over reported facts, and
        # having it work lets the scheduler reject a mismatch before any
        # transport exists.
        return capability_check(plan, worker)

    def prepare_run(self, plan: TrainingPlan, worker: TrainingWorker) -> None:
        self._unimplemented("prepare_run")

    def start(self, plan: TrainingPlan, worker: TrainingWorker) -> BackendStatus:
        return self._unimplemented("start")

    def status(self, plan: TrainingPlan) -> BackendStatus:
        return self._unimplemented("status")

    def cancel(self, plan: TrainingPlan) -> BackendStatus:
        return self._unimplemented("cancel")

    def collect_metrics(self, plan: TrainingPlan) -> list[MetricEvent]:
        self._unimplemented("collect_metrics")
        return []

    def collect_checkpoints(self, plan: TrainingPlan) -> list[dict[str, Any]]:
        self._unimplemented("collect_checkpoints")
        return []

    def cleanup(self, plan: TrainingPlan) -> None:
        self._unimplemented("cleanup")
