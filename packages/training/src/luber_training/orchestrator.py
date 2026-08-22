"""The service layer: creates entities, runs gates, drives the lifecycle.

Everything an operator does goes through here, so three rules live here
rather than being repeated at every call site.

**Gates run before QUEUED, never after.** A run that has not cleared
every gate cannot reach a backend. There is no code path from DRAFT to
STARTING that skips validation, which is what makes "no override exists"
a structural claim rather than a promise.

**Starting is idempotent.** Launching an already-running run is not an
error to swallow and not a second trainer — it is a transition the state
machine does not have, so the second call returns the existing state and
changes nothing.

**Nothing is edited to look tidy.** A failed run stays failed; a retry
is a new run citing its parent. Immutable fields are enforced once a run
has started, so the plan hash keeps meaning what it meant.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from luber_hardware import MachineCapability
from luber_training import registry as registry_module
from luber_training.backends import DRY_RUN, EnvironmentCheck, TrainingExecutionBackend
from luber_training.capacity import CapacityReport
from luber_training.capacity_policy import CapacityDecision
from luber_training.config import TrainingConfig
from luber_training.config import validate as validate_config
from luber_training.entities import (
    IMMUTABLE_AFTER_START,
    CandidateStatus,
    Checkpoint,
    CheckpointKind,
    CheckpointStatus,
    EvaluationCandidate,
    Experiment,
    ExperimentStatus,
    FailureCode,
    ModelBaseline,
    RunStatus,
    TrainingDatasetRef,
    TrainingRun,
    TrainingWorker,
    WorkerStatus,
    derive_experiment_status,
    now,
)
from luber_training.gates import GateInputs, GateReport, run_all
from luber_training.ids import EntityKind, new_id
from luber_training.metrics import MetricEvent, MetricWriter
from luber_training.plan import (
    CodeVersion,
    TrainingPlan,
    capture_code_version,
    capture_environment,
    default_requirements,
)
from luber_training.preflight import (
    DEFAULT_CAPABILITY_MAX_AGE_SECONDS,
    CanaryEvidence,
    DatasetEvidence,
    PreflightIntent,
    PreflightRequest,
    RemoteEvidence,
    StorageEvidence,
    TrainerEvidence,
    TrainingPreflightResult,
)
from luber_training.preflight import (
    evaluate as evaluate_preflight,
)
from luber_training.registry import Registry

RUNS_DIRECTORY_NAME = "training_runs"

#: Phase 33's records, beside the plan and the environment lock.
TRAINING_PREFLIGHT_NAME = "training_preflight.json"
CANARY_RECORD_NAME = "canary.json"

#: Phase 34's. A memory profile is stored twice on purpose: once beside
#: the run it was taken for, and once in the registry's own profile
#: directory where the qualifier looks for evidence about a
#: configuration regardless of which run produced it.
MEMORY_PROFILE_RECORD_NAME = "training_memory_profile.json"


class OrchestrationError(RuntimeError):
    """Raised when an operation is not legal in the current state."""


@dataclass
class PreflightReport:
    """What is verified, what failed, and what nobody knows.

    ``unknown`` never becomes a pass. A preflight that reported UNKNOWN
    as green would be exactly the fabrication this phase is meant to
    avoid, and the operator reading it before renting a GPU is the
    person it would mislead.
    """

    run_id: str
    checks: dict[str, str] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    gate_report: dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "ok": self.ok,
            "checks": dict(sorted(self.checks.items())),
            "problems": self.problems,
            "unknown": self.unknown,
            "gates": self.gate_report,
            "generated_at": self.generated_at,
        }


class Orchestrator:
    """Operator-facing operations over the training registry."""

    def __init__(
        self,
        registry: Registry,
        *,
        artifacts_root: Path | None = None,
        repository_root: Path | None = None,
    ) -> None:
        self.registry = registry
        self.artifacts_root = Path(artifacts_root or registry.root / RUNS_DIRECTORY_NAME)
        self.repository_root = Path(repository_root or Path.cwd())

    # ── baselines ────────────────────────────────────────────────────
    def register_baseline(self, baseline: ModelBaseline) -> ModelBaseline:
        self.registry.write("models", baseline.model_id, baseline.to_dict())
        self.registry.append_audit(
            registry_module.BASELINE_REGISTERED,
            baseline.model_id,
            "model",
            upstream_commit=baseline.upstream_commit,
            stage=baseline.stage,
        )
        return baseline

    def get_baseline(self, model_id: str) -> ModelBaseline:
        payload = self.registry.read("models", model_id)
        return ModelBaseline(**payload)

    # ── experiments ──────────────────────────────────────────────────
    def create_experiment(
        self,
        *,
        name: str,
        hypothesis: str,
        base_model_id: str,
        description: str = "",
        operator: str = "",
        tags: list[str] | None = None,
        status: str = ExperimentStatus.DRAFT.value,
        blocked_reason: str = "",
    ) -> Experiment:
        self.get_baseline(base_model_id)  # refuse an experiment on an unknown model
        experiment = Experiment(
            experiment_id=new_id(EntityKind.EXPERIMENT),
            name=name,
            hypothesis=hypothesis,
            base_model_id=base_model_id,
            description=description,
            operator=operator,
            tags=list(tags or []),
            status=status,
            blocked_reason=blocked_reason,
        )
        self.registry.write("experiments", experiment.experiment_id, experiment.to_dict())
        self.registry.append_audit(
            registry_module.EXPERIMENT_CREATED,
            experiment.experiment_id,
            "experiment",
            name=name,
            base_model_id=base_model_id,
        )
        return experiment

    def get_experiment(self, experiment_id: str) -> Experiment:
        return Experiment(**self.registry.read("experiments", experiment_id))

    def refresh_experiment_status(self, experiment_id: str) -> Experiment:
        """Derive status from runs. Never rewrites the runs themselves."""
        experiment = self.get_experiment(experiment_id)
        runs = self.registry.find("runs", experiment_id=experiment_id)
        statuses = [str(run.get("status")) for run in runs]
        derived = derive_experiment_status(experiment.status, statuses)
        if derived != experiment.status:
            experiment.status = derived
            self.registry.write("experiments", experiment_id, experiment.to_dict(), overwrite=True)
            self.registry.append_audit(
                registry_module.EXPERIMENT_UPDATED, experiment_id, "experiment", status=derived
            )
        return experiment

    # ── workers ──────────────────────────────────────────────────────
    def register_worker(self, worker: TrainingWorker) -> TrainingWorker:
        self.registry.write("workers", worker.worker_id, worker.to_dict())
        self.registry.append_audit(
            registry_module.WORKER_REGISTERED,
            worker.worker_id,
            "worker",
            worker_class=worker.worker_class,
            backend_type=worker.backend_type,
        )
        return worker

    def get_worker(self, worker_id: str) -> TrainingWorker:
        from luber_training.entities import WorkerCapabilities

        payload = self.registry.read("workers", worker_id)
        capabilities = payload.pop("capabilities", {}) or {}
        return TrainingWorker(**payload, capabilities=WorkerCapabilities(**capabilities))

    def record_heartbeat(self, worker_id: str) -> TrainingWorker:
        worker = self.get_worker(worker_id)
        worker.last_heartbeat = now()
        if worker.status in (WorkerStatus.OFFLINE.value, WorkerStatus.LOST.value):
            worker.status = WorkerStatus.ONLINE.value
        self.registry.write("workers", worker_id, worker.to_dict(), overwrite=True)
        return worker

    def mark_worker_lost(self, worker_id: str) -> list[str]:
        """A vanished worker, and the runs that lose contact with it.

        Runs go to LOST rather than FAILED. We know we stopped hearing
        from the worker; we do **not** know training stopped, and a
        remote process may still be burning money. Saying FAILED would
        assert something nobody checked.
        """
        worker = self.get_worker(worker_id)
        worker.status = WorkerStatus.LOST.value
        self.registry.write("workers", worker_id, worker.to_dict(), overwrite=True)

        affected: list[str] = []
        for payload in self.registry.find("runs", worker_id=worker_id):
            run = self._run_from_dict(payload)
            if run.status == RunStatus.RUNNING.value:
                self.transition_run(
                    run.run_id,
                    RunStatus.LOST.value,
                    error_code=FailureCode.WORKER_LOST.value,
                    error_message="the worker stopped reporting; remote state is unknown",
                )
                affected.append(run.run_id)
        return affected

    def busy_run_count(self, worker_id: str) -> int:
        active = {RunStatus.STARTING.value, RunStatus.RUNNING.value}
        return sum(
            1
            for payload in self.registry.find("runs", worker_id=worker_id)
            if str(payload.get("status")) in active
        )

    # ── runs ─────────────────────────────────────────────────────────
    def create_run(
        self,
        *,
        experiment_id: str,
        dataset_ref: TrainingDatasetRef,
        config: TrainingConfig,
        execution_backend: str = DRY_RUN,
        worker_id: str | None = None,
        parent_run_id: str | None = None,
        resume_from_checkpoint_id: str | None = None,
    ) -> TrainingRun:
        experiment = self.get_experiment(experiment_id)
        validate_config(config)

        if resume_from_checkpoint_id is not None:
            self._validate_resume(resume_from_checkpoint_id, experiment)

        run = TrainingRun(
            run_id=new_id(EntityKind.RUN),
            experiment_id=experiment_id,
            base_model_id=experiment.base_model_id,
            dataset_ref=dataset_ref,
            config=config,
            execution_backend=execution_backend,
            worker_id=worker_id,
            parent_run_id=parent_run_id,
            resume_from_checkpoint_id=resume_from_checkpoint_id,
        )
        run.output_directory = str(self.artifacts_root / experiment_id / run.run_id)
        self.registry.write("runs", run.run_id, run.to_dict())
        self.registry.append_audit(
            registry_module.RUN_CREATED,
            run.run_id,
            "run",
            experiment_id=experiment_id,
            config_sha256=run.config_sha256,
            backend=execution_backend,
            parent_run_id=parent_run_id,
        )
        self.refresh_experiment_status(experiment_id)
        return run

    def _validate_resume(self, checkpoint_id: str, experiment: Experiment) -> None:
        """Refuse arbitrary checkpoint mixing.

        A checkpoint is only a valid starting point if it is READY, is
        real weights rather than a placeholder, and came from a run on
        the same base model. Resuming across model families produces an
        adapter shaped for weights it was never trained against.
        """
        checkpoint = self.get_checkpoint(checkpoint_id)
        if checkpoint.status != CheckpointStatus.READY.value:
            raise OrchestrationError(
                f"checkpoint {checkpoint_id} is {checkpoint.status}; only READY "
                "checkpoints may be resumed from"
            )
        if not checkpoint.is_real_model:
            raise OrchestrationError(
                f"checkpoint {checkpoint_id} is a {checkpoint.kind} artifact and contains "
                "no trained weights; it cannot be resumed from"
            )
        source_run = self.get_run(checkpoint.run_id)
        if source_run.base_model_id != experiment.base_model_id:
            raise OrchestrationError(
                f"checkpoint {checkpoint_id} was trained on {source_run.base_model_id} but "
                f"this experiment uses {experiment.base_model_id}"
            )

    def get_run(self, run_id: str) -> TrainingRun:
        return self._run_from_dict(self.registry.read("runs", run_id))

    def _run_from_dict(self, payload: dict[str, Any]) -> TrainingRun:
        data = dict(payload)
        data.pop("config_sha256", None)
        config = TrainingConfig(**data.pop("config"))
        dataset_ref = TrainingDatasetRef(**data.pop("dataset_ref"))
        return TrainingRun(**data, config=config, dataset_ref=dataset_ref)

    def save_run(self, run: TrainingRun) -> TrainingRun:
        """Persist a run, refusing edits to frozen fields.

        The immutability check compares against what is on disk, so it
        catches a mutation made anywhere — not only one made through a
        setter this class controls.
        """
        with self.registry.lock():
            if self.registry.exists("runs", run.run_id):
                stored = self._run_from_dict(self.registry.read("runs", run.run_id))
                if stored.has_started:
                    # A field still unset may be filled in once — the
                    # plan hash is assigned at compile time, which
                    # happens after QUEUED. What is forbidden is
                    # *changing* a value that has already been recorded
                    # and cited.
                    changed = [
                        name
                        for name in sorted(IMMUTABLE_AFTER_START)
                        if _field_value(stored, name) is not None
                        and _field_value(stored, name) != _field_value(run, name)
                    ]
                    if changed:
                        raise OrchestrationError(
                            f"run {run.run_id} has started; {', '.join(changed)} may not "
                            "change. Create a new run instead."
                        )
            self.registry.write("runs", run.run_id, run.to_dict(), overwrite=True)
        return run

    def transition_run(
        self,
        run_id: str,
        status: str,
        *,
        error_code: str | None = None,
        error_message: str | None = None,
        worker_id: str | None = None,
    ) -> TrainingRun:
        """Move a run, honouring the state machine.

        Illegal transitions raise. That is what makes idempotent launch
        possible: the second `start` does not need a special case
        because DRAFT→RUNNING simply is not a transition.
        """
        with self.registry.lock():
            run = self.get_run(run_id)
            if status == run.status:
                return run
            if not run.can_transition_to(status):
                raise OrchestrationError(f"run {run_id} cannot move {run.status} -> {status}")

            run.status = status
            stamp = now()
            if status == RunStatus.QUEUED.value:
                run.queued_at = stamp
            elif status == RunStatus.RUNNING.value:
                run.started_at = stamp
            elif status == RunStatus.COMPLETED.value:
                run.completed_at = stamp
            elif status == RunStatus.FAILED.value:
                run.failed_at = stamp
            elif status == RunStatus.CANCELLED.value:
                run.cancelled_at = stamp

            if error_code is not None:
                run.error_code = error_code
            if error_message is not None:
                run.error_message = _sanitize(error_message)
            if worker_id is not None:
                run.worker_id = worker_id

            self.registry.write("runs", run_id, run.to_dict(), overwrite=True)

        event = {
            RunStatus.QUEUED.value: registry_module.RUN_QUEUED,
            RunStatus.RUNNING.value: registry_module.RUN_STARTED,
            RunStatus.COMPLETED.value: registry_module.RUN_COMPLETED,
            RunStatus.FAILED.value: registry_module.RUN_FAILED,
            RunStatus.CANCELLED.value: registry_module.RUN_CANCELLED,
            RunStatus.LOST.value: registry_module.RUN_LOST,
        }.get(status)
        if event:
            self.registry.append_audit(event, run_id, "run", status=status, error_code=error_code)
        self.refresh_experiment_status(run.experiment_id)
        return run

    # ── validation and preflight ─────────────────────────────────────
    def validate_run(
        self, run_id: str, gate_inputs: GateInputs, *, worker_id: str | None = None
    ) -> tuple[TrainingRun, GateReport]:
        """Run every gate. On success the run reaches QUEUED.

        A failure moves the run to FAILED with the gate's own code, so
        the reason survives in the registry rather than only in whatever
        terminal the operator was looking at.
        """
        run = self.get_run(run_id)
        if run.status != RunStatus.DRAFT.value:
            raise OrchestrationError(f"run {run_id} is {run.status}, not DRAFT")

        self.transition_run(run_id, RunStatus.VALIDATING.value)
        report = run_all(gate_inputs)
        self.registry.append_audit(
            registry_module.RUN_VALIDATED, run_id, "run", passed=report.passed
        )

        if not report.passed:
            failure = report.first_failure
            self.registry.append_audit(
                registry_module.RUN_BLOCKED,
                run_id,
                "run",
                failure_code=report.failure_code(),
                gate=failure.name if failure else None,
            )
            run = self.transition_run(
                run_id,
                RunStatus.FAILED.value,
                error_code=report.failure_code(),
                error_message=failure.detail if failure else "gate failure",
            )
            return run, report

        run = self.transition_run(run_id, RunStatus.QUEUED.value, worker_id=worker_id)
        return run, report

    def compile_plan(self, run_id: str, *, execution_device: str | None = None) -> TrainingPlan:
        """Build the immutable plan for a validated run.

        ``execution_device`` is Phase 32's placement, when one has been
        made. Omitted, the plan is compiled exactly as it was before —
        `requires_cuda` with no device named — so the digest of an
        existing run does not move. Supplied, it becomes part of the
        plan's identity, which is correct: a plan that says MPS is not
        the same plan as one that left the device implied.
        """
        run = self.get_run(run_id)
        baseline = self.get_baseline(run.base_model_id)
        if not baseline.supports(run.config.strategy):
            raise OrchestrationError(
                f"model {baseline.model_id} does not support {run.config.strategy}; "
                f"it supports {', '.join(baseline.training_strategy_support) or 'nothing'}"
            )

        plan = TrainingPlan(
            plan_id=new_id(EntityKind.PLAN),
            run_id=run.run_id,
            experiment_id=run.experiment_id,
            base_model_id=baseline.model_id,
            base_model_upstream_commit=baseline.upstream_commit,
            dataset_ref=run.dataset_ref,
            config=run.config,
            execution_backend=run.execution_backend,
            requirements=default_requirements(run.config, device=execution_device),
            resume_from_checkpoint_id=run.resume_from_checkpoint_id,
        )
        digest = plan.digest()
        if run.training_plan_sha256 is not None and run.training_plan_sha256 != digest:
            raise OrchestrationError(
                f"run {run.run_id} already cites plan {run.training_plan_sha256[:12]}; "
                f"recompiling produced {digest[:12]}. Create a new run."
            )
        run.training_plan_sha256 = digest
        self.save_run(run)
        return plan

    def preflight(
        self,
        run_id: str,
        plan: TrainingPlan,
        worker: TrainingWorker,
        backend: TrainingExecutionBackend,
        gate_report: GateReport,
        *,
        require_clean_repository: bool = True,
        minimum_free_disk_mb: int | None = None,
        disk_safety_margin: float = 1.5,
    ) -> PreflightReport:
        """Everything checkable before a real run starts."""
        report = PreflightReport(run_id=run_id, gate_report=gate_report.to_dict())

        for result in gate_report.results:
            report.checks[f"gate:{result.name}"] = "PASS" if result.passed else "FAIL"
            if not result.passed:
                report.problems.append(f"{result.name}: {result.detail}")

        code: CodeVersion = capture_code_version(self.repository_root)
        if code.commit is None:
            report.checks["code_version"] = "UNKNOWN"
            report.unknown.append("the repository commit could not be read")
            if require_clean_repository:
                report.problems.append("training requires an identifiable repository revision")
        elif code.dirty and require_clean_repository:
            report.checks["code_version"] = "FAIL"
            report.problems.append(
                f"the working tree is dirty at {code.commit[:12]}; a run from an "
                "unidentified revision cannot be reproduced"
            )
        else:
            report.checks["code_version"] = "PASS"

        environment: EnvironmentCheck = backend.validate_environment(plan, worker)
        report.checks["worker_environment"] = "PASS" if environment.ok else "FAIL"
        report.problems.extend(environment.problems)
        report.unknown.extend(environment.unknown)

        if worker.max_concurrent_runs <= self.busy_run_count(worker.worker_id):
            report.checks["worker_capacity"] = "FAIL"
            report.problems.append(
                f"worker {worker.name} already has {self.busy_run_count(worker.worker_id)} "
                f"active run(s) and allows {worker.max_concurrent_runs}"
            )
        else:
            report.checks["worker_capacity"] = "PASS"

        output = Path(self.get_run(run_id).output_directory or self.artifacts_root)
        try:
            output.mkdir(parents=True, exist_ok=True)
            probe = output / ".writable"
            probe.write_text("", encoding="utf-8")
            probe.unlink()
            report.checks["output_writable"] = "PASS"
        except OSError as exc:
            report.checks["output_writable"] = "FAIL"
            report.problems.append(f"output directory is not writable: {exc}")

        if minimum_free_disk_mb is None:
            report.checks["disk_capacity"] = "UNKNOWN"
            report.unknown.append(
                "checkpoint size has never been measured for any LUBER configuration, so "
                "the disk requirement is unknown; supply minimum_free_disk_mb to check it"
            )
        else:
            free = worker.capabilities.free_disk_mb
            required = int(minimum_free_disk_mb * disk_safety_margin)
            if free is None:
                report.checks["disk_capacity"] = "UNKNOWN"
                report.unknown.append(f"worker {worker.name} has not reported free disk")
            elif free < required:
                report.checks["disk_capacity"] = "FAIL"
                report.problems.append(
                    f"worker reports {free} MB free; {required} MB required "
                    f"({minimum_free_disk_mb} MB x {disk_safety_margin} margin)"
                )
            else:
                report.checks["disk_capacity"] = "PASS"

        return report

    # ── Phase 33: the execution-readiness gate ───────────────────────
    def training_preflight(
        self,
        run_id: str,
        plan: TrainingPlan,
        *,
        capability: MachineCapability,
        execution_location: str,
        intent: str = PreflightIntent.CANARY.value,
        worker: TrainingWorker | None = None,
        gate_report: GateReport | None = None,
        dataset: DatasetEvidence | None = None,
        trainer: TrainerEvidence | None = None,
        storage: StorageEvidence | None = None,
        remote: RemoteEvidence | None = None,
        canary: CanaryEvidence | None = None,
        capacity: CapacityReport | None = None,
        capacity_decision: CapacityDecision | None = None,
        capability_max_age_seconds: float | None = None,
        measured_at: str | None = None,
        write: bool = True,
    ) -> TrainingPreflightResult:
        """Can this machine execute this plan, and is it proven?

        Distinct from :meth:`preflight`, which is Phase 25's and stays
        exactly as it was. That one answers "is this run legitimate and
        does the worker look plausible"; this one answers "will the
        trainer start, on this device, at this precision, with this
        optimizer, and can we say so without guessing".

        Evidence is passed in rather than gathered here. Collecting it
        means subprocesses and filesystem reads that a browser-reachable
        process may not be able to perform, and a verdict that changed
        depending on who asked would not be a verdict.
        """
        request = PreflightRequest(
            plan=plan,
            capability=capability,
            execution_location=execution_location,
            intent=intent,
            worker=worker,
            gate_report=gate_report,
            dataset=dataset or DatasetEvidence(),
            trainer=trainer or TrainerEvidence(),
            storage=storage or StorageEvidence(),
            remote=remote or RemoteEvidence(),
            canary=canary or CanaryEvidence(),
            capacity=capacity,
            capacity_decision=capacity_decision,
            capability_max_age_seconds=(
                capability_max_age_seconds
                if capability_max_age_seconds is not None
                else DEFAULT_CAPABILITY_MAX_AGE_SECONDS
            ),
            measured_at=measured_at,
        )
        result = evaluate_preflight(request)
        if write:
            directory = Path(self.get_run(run_id).output_directory or self.artifacts_root / run_id)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / TRAINING_PREFLIGHT_NAME).write_text(
                json.dumps(result.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        self.registry.append_audit(
            registry_module.TRAINING_PREFLIGHT_RECORDED,
            run_id,
            "run",
            status=result.status,
            intent=result.intent,
            execution_device=result.execution_device,
        )
        return result

    def record_canary(self, run_id: str, result: dict[str, Any]) -> Path:
        """Store a canary's own record beside the run.

        Separate from the preflight file because they are separate
        claims: a canary is something that happened, a preflight is a
        verdict about what may happen next.
        """
        directory = Path(self.get_run(run_id).output_directory or self.artifacts_root / run_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / CANARY_RECORD_NAME
        path.write_text(
            json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self.registry.append_audit(
            registry_module.CANARY_RECORDED,
            run_id,
            "run",
            status=result.get("status"),
            mode=result.get("mode"),
        )
        return path

    def record_memory_profile(self, run_id: str, profile: dict[str, Any]) -> Path:
        """Store a memory profile beside the run it was measured for.

        Its own file rather than the canary's: a canary is a run that
        happened, a profile is a measurement of one, and a reader
        looking for "what did this cost" should not have to know that a
        canary produced it.
        """
        directory = Path(self.get_run(run_id).output_directory or self.artifacts_root / run_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / MEMORY_PROFILE_RECORD_NAME
        path.write_text(
            json.dumps(profile, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self.registry.append_audit(
            registry_module.MEMORY_PROFILE_RECORDED,
            run_id,
            "run",
            outcome=profile.get("outcome"),
            profile_id=profile.get("profile_id"),
            representativeness=profile.get("representativeness"),
        )
        return path

    # ── execution ────────────────────────────────────────────────────
    def start_run(
        self,
        run_id: str,
        plan: TrainingPlan,
        worker: TrainingWorker,
        backend: TrainingExecutionBackend,
    ) -> TrainingRun:
        """Start a queued run. Calling twice starts one trainer.

        The guard is the state machine rather than a flag: a second call
        finds the run already past QUEUED, and QUEUED is the only state
        a start may leave.
        """
        run = self.get_run(run_id)
        if run.status in (RunStatus.STARTING.value, RunStatus.RUNNING.value):
            return run
        if run.is_terminal:
            raise OrchestrationError(f"run {run_id} is {run.status} and cannot be started")
        if run.status != RunStatus.QUEUED.value:
            raise OrchestrationError(f"run {run_id} is {run.status}; validate it before starting")

        run = self.transition_run(run_id, RunStatus.STARTING.value, worker_id=worker.worker_id)
        self._write_run_artifacts(run, plan)
        backend.prepare_run(plan, worker)
        backend.start(plan, worker)
        return self.transition_run(run_id, RunStatus.RUNNING.value)

    def _write_run_artifacts(self, run: TrainingRun, plan: TrainingPlan) -> None:
        directory = Path(run.output_directory or self.artifacts_root / run.run_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "plan.json").write_text(
            json.dumps(plan.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        environment = capture_environment(
            self.repository_root, ace_step_commit=run.config.ace_step_commit
        )
        (directory / "environment_lock.json").write_text(
            json.dumps(environment.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (directory / "run.json").write_text(
            json.dumps(run.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (directory / "checkpoints").mkdir(exist_ok=True)
        (directory / "logs").mkdir(exist_ok=True)
        (directory / "artifacts").mkdir(exist_ok=True)

    def metric_writer(self, run_id: str) -> MetricWriter:
        run = self.get_run(run_id)
        directory = Path(run.output_directory or self.artifacts_root / run_id)
        return MetricWriter(directory / "metrics.jsonl")

    def record_metrics(self, run_id: str, events: list[MetricEvent]) -> int:
        self.metric_writer(run_id).append_many(events)
        return len(events)

    def cancel_run(
        self, run_id: str, plan: TrainingPlan, backend: TrainingExecutionBackend
    ) -> TrainingRun:
        """Cancel a run without destroying what it produced.

        Metrics, logs and any completed checkpoints survive. A cancelled
        run is part of the history of an experiment, not an embarrassment
        to be erased.
        """
        run = self.get_run(run_id)
        if run.is_terminal:
            return run
        backend.cancel(plan)
        return self.transition_run(
            run_id,
            RunStatus.CANCELLED.value,
            error_code=FailureCode.CANCELLED_BY_OPERATOR.value,
            error_message="cancelled by operator",
        )

    def complete_run(self, run_id: str) -> TrainingRun:
        return self.transition_run(run_id, RunStatus.COMPLETED.value)

    def fail_run(self, run_id: str, code: str, message: str) -> TrainingRun:
        return self.transition_run(
            run_id, RunStatus.FAILED.value, error_code=code, error_message=message
        )

    # ── checkpoints ──────────────────────────────────────────────────
    def register_checkpoint(self, checkpoint: Checkpoint) -> Checkpoint:
        self.registry.write("checkpoints", checkpoint.checkpoint_id, checkpoint.to_dict())
        self.registry.append_audit(
            registry_module.CHECKPOINT_REGISTERED,
            checkpoint.checkpoint_id,
            "checkpoint",
            run_id=checkpoint.run_id,
            kind=checkpoint.kind,
            status=checkpoint.status,
        )
        return checkpoint

    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint:
        return Checkpoint(**self.registry.read("checkpoints", checkpoint_id))

    def finalize_checkpoint_record(
        self, checkpoint_id: str, *, sha256: str, size_bytes: int, reference: str
    ) -> Checkpoint:
        """Mark a checkpoint READY only after its bytes are on disk.

        Nothing else in this class sets READY. A checkpoint reaches it
        by having been validated, hashed and atomically moved — which is
        the difference between a model and a directory that might be one.
        """
        with self.registry.lock():
            checkpoint = self.get_checkpoint(checkpoint_id)
            if checkpoint.status != CheckpointStatus.WRITING.value:
                raise OrchestrationError(
                    f"checkpoint {checkpoint_id} is {checkpoint.status}, not WRITING"
                )
            checkpoint.sha256 = sha256
            checkpoint.size_bytes = size_bytes
            checkpoint.reference = reference
            checkpoint.status = CheckpointStatus.READY.value
            checkpoint.finalized_at = now()
            self.registry.write("checkpoints", checkpoint_id, checkpoint.to_dict(), overwrite=True)
        self.registry.append_audit(
            registry_module.CHECKPOINT_FINALIZED,
            checkpoint_id,
            "checkpoint",
            run_id=checkpoint.run_id,
            sha256=sha256,
        )
        return checkpoint

    def run_checkpoints(self, run_id: str) -> list[Checkpoint]:
        return [
            Checkpoint(**payload) for payload in self.registry.find("checkpoints", run_id=run_id)
        ]

    # ── evaluation candidates ────────────────────────────────────────
    def create_candidate(self, checkpoint_id: str, *, notes: str = "") -> EvaluationCandidate:
        """Nominate a checkpoint for evaluation. Never promote it.

        A candidate is a request for evidence, not a claim of quality.
        Refusing MOCK artifacts here is what stops a dry run from ever
        producing something that could be evaluated as a model.
        """
        checkpoint = self.get_checkpoint(checkpoint_id)
        if checkpoint.status != CheckpointStatus.READY.value:
            raise OrchestrationError(
                f"checkpoint {checkpoint_id} is {checkpoint.status}; only READY "
                "checkpoints may become candidates"
            )
        if checkpoint.kind == CheckpointKind.MOCK.value:
            raise OrchestrationError(
                f"checkpoint {checkpoint_id} is a MOCK artifact and contains no trained "
                "weights; it can never become an evaluation candidate"
            )

        run = self.get_run(checkpoint.run_id)
        candidate = EvaluationCandidate(
            candidate_id=new_id(EntityKind.CANDIDATE),
            run_id=run.run_id,
            checkpoint_id=checkpoint_id,
            experiment_id=run.experiment_id,
            status=CandidateStatus.PENDING_EVALUATION.value,
            notes=notes,
        )
        self.registry.write("candidates", candidate.candidate_id, candidate.to_dict())
        self.registry.append_audit(
            registry_module.CANDIDATE_CREATED,
            candidate.candidate_id,
            "candidate",
            checkpoint_id=checkpoint_id,
            run_id=run.run_id,
        )
        return candidate

    # ── reproducibility ──────────────────────────────────────────────
    def run_bundle(self, run_id: str, plan: TrainingPlan | None = None) -> dict[str, Any]:
        """Everything needed to understand a run afterwards.

        References and digests rather than copies. The bundle is meant
        to be readable in a year and to point at the artifacts that
        explain what happened, not to duplicate them.
        """
        run = self.get_run(run_id)
        baseline = self.get_baseline(run.base_model_id)
        directory = Path(run.output_directory or self.artifacts_root / run_id)

        environment: dict[str, Any] = {}
        environment_path = directory / "environment_lock.json"
        if environment_path.is_file():
            environment = json.loads(environment_path.read_text(encoding="utf-8"))

        worker_payload: dict[str, Any] = {}
        if run.worker_id:
            try:
                worker_payload = self.get_worker(run.worker_id).to_dict()
            except Exception:
                worker_payload = {"worker_id": run.worker_id, "note": "worker record unavailable"}

        return {
            "schema_version": "luber-run-bundle/1",
            "run_id": run.run_id,
            "experiment_id": run.experiment_id,
            "status": run.status,
            "error_code": run.error_code,
            "model_baseline": {
                "model_id": baseline.model_id,
                "upstream_commit": baseline.upstream_commit,
                "identity_basis": baseline.identity_basis,
            },
            "dataset_ref": run.dataset_ref.to_dict(),
            "training_config_sha256": run.config_sha256,
            "training_config": run.config.to_dict(),
            "training_plan_sha256": run.training_plan_sha256,
            "training_plan": plan.to_dict() if plan is not None else None,
            "environment_lock": environment,
            "worker": worker_payload,
            "checkpoints": [checkpoint.to_dict() for checkpoint in self.run_checkpoints(run_id)],
            "lineage": {
                "parent_run_id": run.parent_run_id,
                "resume_from_checkpoint_id": run.resume_from_checkpoint_id,
            },
            "audit_events": self.registry.audit_events(run_id),
        }


def _field_value(run: TrainingRun, name: str) -> Any:
    value = getattr(run, name)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


#: Substrings that suggest a secret leaked into a message. Redaction is
#: crude on purpose — a false positive costs a few characters of a log
#: line, a false negative writes a credential to disk forever.
_SECRET_MARKERS: tuple[str, ...] = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "private key",
    "BEGIN RSA",
    "BEGIN OPENSSH",
)


def _sanitize(message: str) -> str:
    """Redact a diagnostic that looks like it carries a credential."""
    lowered = message.lower()
    if any(marker in lowered for marker in _SECRET_MARKERS):
        return "[redacted: the diagnostic matched a credential pattern]"
    return message
