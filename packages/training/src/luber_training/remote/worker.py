"""The program that runs on the GPU machine.

It is not a daemon. Each verb is a short-lived invocation that reads
state from files under the worker root, does one thing, writes the state
back, and exits — which is what lets it be driven over SSH without a
listening port, a service manager, or an open connection that a training
run would outlive.

That choice has one consequence worth naming: this process is usually
*not* the trainer's parent. It launches the trainer detached and exits;
a later invocation finds it again by PID. So it can always answer "is it
alive" and it cannot always answer "how did it exit" — and where it
cannot, it says UNKNOWN rather than guessing. A worker that reported
COMPLETED because it could no longer see a process would be the single
most dangerous behaviour available to it.

The worker decides nothing about legitimacy. It does not know which
dataset is allowed, which experiment is worth running, whether rights
are valid, or whether a checkpoint is worth promoting. Those are control
plane questions, answered before anything reaches this machine. What the
worker does is verify that what arrived is what was sent, refuse to
start when it is not, run the exact command it was given, and report
truthfully.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from luber_training.remote.capabilities import (
    CapabilityReport,
    WorkerClassification,
    probe,
    sample_telemetry,
)
from luber_training.remote.execution import (
    ExecutionState,
    TrainerProcess,
    classify_failure,
    process_alive,
)
from luber_training.remote.identity import (
    Heartbeat,
    RunLease,
    WorkerIdentity,
    host_fingerprint,
    load_or_create_identity,
)
from luber_training.remote.manifest import RemoteArtifactManifest
from luber_training.remote.paths import RemoteRoots, RunLayout, validate_relative
from luber_training.remote.preflight import PreflightReport, run_preflight
from luber_training.remote.protocol import (
    REMOTE_PROTOCOL_VERSION,
    TERMINAL_WORKER_STATES,
    WorkerHealth,
    WorkerState,
    now,
)
from luber_training.remote.result import RemoteResult, build_result, discover_checkpoints
from luber_training.remote.streams import LogChunk, read_log
from luber_training.remote.transport import ContentCache

WORKER_CONFIG_NAME = "worker.json"
IDENTITY_NAME = "identity.json"
HEARTBEAT_NAME = "heartbeat.json"


class WorkerError(RuntimeError):
    """Raised when the worker cannot honour a request."""


@dataclass
class WorkerConfig:
    """Where this machine keeps things, and what it is called.

    Written once at `init` and read by every later invocation. It holds
    paths and names — never a credential, because the worker is the far
    end of a connection and has no reason to authenticate outward.
    """

    worker_name: str
    roots: RemoteRoots
    trainer_root: str | None = None
    repository_root: str | None = None
    backend_type: str = "remote-gpu"
    #: Explicitly stated by the operator, not inferred. A machine does
    #: not become a training worker because a field looked promising.
    declared_classification: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_name": self.worker_name,
            "roots": self.roots.to_dict(),
            "trainer_root": self.trainer_root,
            "repository_root": self.repository_root,
            "backend_type": self.backend_type,
            "declared_classification": self.declared_classification,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> WorkerConfig:
        return cls(
            worker_name=str(payload["worker_name"]),
            roots=RemoteRoots(**payload["roots"]),
            trainer_root=payload.get("trainer_root"),
            repository_root=payload.get("repository_root"),
            backend_type=str(payload.get("backend_type", "remote-gpu")),
            declared_classification=payload.get("declared_classification"),
        )


class RemoteWorker:
    """One machine's view of its own work.

    Every method is safe to call from a fresh process. There is no
    in-memory state that matters: `self` holds paths, and everything
    else is read from and written to disk on each call.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ── configuration and identity ───────────────────────────────────
    @property
    def config_path(self) -> Path:
        return self.root / WORKER_CONFIG_NAME

    @property
    def identity_path(self) -> Path:
        return self.root / IDENTITY_NAME

    @property
    def heartbeat_path(self) -> Path:
        return self.root / HEARTBEAT_NAME

    def initialise(
        self,
        *,
        worker_name: str,
        roots: RemoteRoots | None = None,
        trainer_root: str | None = None,
        repository_root: str | None = None,
        backend_type: str = "remote-gpu",
    ) -> WorkerConfig:
        config = WorkerConfig(
            worker_name=worker_name,
            roots=roots or RemoteRoots.under(self.root),
            trainer_root=trainer_root,
            repository_root=repository_root,
            backend_type=backend_type,
        )
        self.config_path.write_text(
            json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for logical in ("code_root", "data_root", "run_root", "checkpoint_root", "cache_root"):
            Path(config.roots.get(logical)).mkdir(parents=True, exist_ok=True)
        return config

    def config(self) -> WorkerConfig:
        if not self.config_path.is_file():
            raise WorkerError(
                f"{self.root} has not been initialised as a worker root; run "
                "`python -m luber_training.remote init` first"
            )
        return WorkerConfig.from_dict(json.loads(self.config_path.read_text(encoding="utf-8")))

    def capabilities(self) -> CapabilityReport:
        config = self.config()
        return probe(
            trainer_root=Path(config.trainer_root) if config.trainer_root else None,
            repository_root=Path(config.repository_root) if config.repository_root else None,
            disk_path=Path(config.roots.run_root),
        )

    def identity(self, report: CapabilityReport | None = None) -> tuple[WorkerIdentity, bool]:
        """This worker's stable identity, and whether the host changed.

        A changed fingerprint means the machine underneath was rebuilt.
        The identity is not silently reissued — the control plane is
        told, and decides whether a rebuilt host keeps the id it
        verified last week.
        """
        config = self.config()
        report = report or self.capabilities()
        return load_or_create_identity(
            self.identity_path,
            worker_name=config.worker_name,
            backend_type=config.backend_type,
            fingerprint=host_fingerprint(report),
            capability_signature=report.signature(),
        )

    def cache(self) -> ContentCache:
        return ContentCache(Path(self.config().roots.cache_root))

    # ── run layout ───────────────────────────────────────────────────
    def layout(self, run_id: str) -> RunLayout:
        return RunLayout.for_run(Path(self.config().roots.run_root), validate_relative(run_id))

    def known_runs(self) -> list[str]:
        run_root = Path(self.config().roots.run_root)
        if not run_root.is_dir():
            return []
        return sorted(path.name for path in run_root.iterdir() if path.is_dir())

    def active_run(self) -> str | None:
        """The run this worker is currently executing, if any."""
        for run_id in self.known_runs():
            state = ExecutionState.read(self.layout(run_id))
            if state and state.state not in TERMINAL_WORKER_STATES | {WorkerState.IDLE.value}:
                return run_id
        return None

    # ── heartbeat ────────────────────────────────────────────────────
    def heartbeat(self) -> Heartbeat:
        """Report life, and what this machine believes it is doing.

        Written to disk as well as returned, so a control plane reading
        over a transport that cannot invoke commands — a shared
        filesystem, say — still sees it.
        """
        identity, changed = self.identity()
        active = self.active_run()
        state = WorkerState.IDLE.value
        detail = ""

        if active:
            execution = ExecutionState.read(self.layout(active))
            if execution:
                state = execution.state
                detail = execution.detail

        free_disk = None
        try:
            free_disk = int(shutil.disk_usage(self.config().roots.run_root).free / (1024 * 1024))
        except OSError:
            pass

        beat = Heartbeat(
            worker_id=identity.worker_id,
            worker_state=state,
            active_run_id=active,
            health=WorkerHealth.DEGRADED.value if changed else WorkerHealth.OK.value,
            uptime_seconds=_uptime(),
            free_disk_mb=free_disk,
            gpu=[reading.to_dict() for reading in sample_telemetry()],
            detail=(
                "the host fingerprint has changed since this worker id was minted; the "
                "machine may have been rebuilt"
                if changed
                else detail
            ),
        )
        self.heartbeat_path.write_text(
            json.dumps(beat.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return beat

    # ── receiving a run ──────────────────────────────────────────────
    def prepare(
        self, *, run_id: str, plan_sha256: str, manifest_sha256: str, worker_id: str | None = None
    ) -> ExecutionState:
        """Claim a run and make room for it. Idempotent by design.

        This is where double dispatch is stopped. An existing lease for
        the same run and the same plan returns the state that already
        exists; a *different* plan under the same run id is an integrity
        violation, because a run id has to mean exactly one training
        configuration or nothing downstream can be reasoned about.
        """
        identity, _ = self.identity()
        worker_id = worker_id or identity.worker_id
        layout = self.layout(run_id)
        layout.ensure()

        existing = ExecutionState.read(layout)
        if layout.lease_json.is_file():
            lease = RunLease.from_dict(json.loads(layout.lease_json.read_text(encoding="utf-8")))
            # Raises on a foreign worker or a different plan. Returning
            # the existing state is only correct when it is genuinely
            # the same work.
            lease.check_compatible(run_id=run_id, worker_id=worker_id, plan_sha256=plan_sha256)
            if existing is not None:
                return existing

        lease = RunLease.create(
            run_id=run_id, worker_id=worker_id, training_plan_sha256=plan_sha256
        )
        layout.lease_json.write_text(
            json.dumps(lease.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        state = existing or ExecutionState(run_id=run_id)
        state.state = WorkerState.RECEIVING.value
        state.training_plan_sha256 = plan_sha256
        state.manifest_sha256 = manifest_sha256
        state.lease_id = lease.lease_id
        state.detail = "awaiting artifacts"
        state.write(layout)
        return state

    def lease(self, run_id: str) -> RunLease | None:
        layout = self.layout(run_id)
        if not layout.lease_json.is_file():
            return None
        return RunLease.from_dict(json.loads(layout.lease_json.read_text(encoding="utf-8")))

    # ── preflight ────────────────────────────────────────────────────
    def preflight(
        self,
        run_id: str,
        *,
        minimum_free_disk_mb: int | None = None,
        require_code_match: bool = True,
    ) -> PreflightReport:
        layout = self.layout(run_id)
        state = ExecutionState.read(layout)
        if state is None:
            raise WorkerError(f"run {run_id} was never prepared on this worker")

        if not layout.manifest_json.is_file():
            raise WorkerError(f"run {run_id} has no artifact manifest; nothing was received")
        manifest = RemoteArtifactManifest.read(layout.manifest_json)

        plan: dict[str, Any] = {}
        from luber_training.remote.staging import PLAN_PATH

        plan_path = layout.root / PLAN_PATH
        if plan_path.is_file():
            plan = json.loads(plan_path.read_text(encoding="utf-8"))

        environment: dict[str, Any] | None = None
        if layout.environment_json.is_file():
            environment = json.loads(layout.environment_json.read_text(encoding="utf-8"))

        config = self.config()
        identity, _ = self.identity()
        state.state = WorkerState.PREFLIGHT.value
        state.write(layout)

        report = run_preflight(
            layout=layout,
            worker_id=identity.worker_id,
            plan=plan,
            manifest=manifest,
            capabilities=self.capabilities(),
            expected_plan_sha256=state.training_plan_sha256,
            expected_manifest_sha256=state.manifest_sha256,
            environment_lock=environment,
            trainer_root=Path(config.trainer_root) if config.trainer_root else None,
            minimum_free_disk_mb=minimum_free_disk_mb,
            require_code_match=require_code_match,
        )
        report.write(layout)

        state.state = WorkerState.READY.value if report.passed else WorkerState.PREFLIGHT.value
        state.detail = (
            "preflight passed"
            if report.passed
            else f"preflight {report.status}: {'; '.join(report.blocking_reasons[:3])}"
        )
        state.write(layout)
        return report

    # ── launching ────────────────────────────────────────────────────
    def start(
        self,
        run_id: str,
        *,
        argv: list[str] | None = None,
        working_directory: Path | None = None,
        environment: dict[str, str] | None = None,
    ) -> ExecutionState:
        """Launch the trainer. Calling twice launches one trainer.

        The idempotency guard is a live process, not a flag: if the
        recorded PID is alive, this returns the existing state. That is
        what makes the ambiguous-launch case safe — a control plane that
        never received the acknowledgement can call again, and the
        second call finds the first one's trainer.
        """
        layout = self.layout(run_id)
        state = ExecutionState.read(layout)
        if state is None:
            raise WorkerError(f"run {run_id} was never prepared on this worker")

        if state.process is not None and process_alive(state.process.pid):
            state.detail = "already running; this request started nothing"
            state.write(layout)
            return state
        if state.state in TERMINAL_WORKER_STATES:
            return state
        if state.state != WorkerState.READY.value:
            raise WorkerError(
                f"run {run_id} is {state.state}; a trainer only starts after preflight passes"
            )

        if argv is None:
            argv, working_directory = self._compile(run_id, layout)
        if working_directory is None:
            config = self.config()
            working_directory = Path(config.trainer_root or layout.root)

        state.state = WorkerState.STARTING.value
        state.started_at = state.started_at or now()
        state.write(layout)

        process = TrainerProcess(layout)
        record = process.launch(argv, working_directory=working_directory, environment=environment)
        state.process = record

        if record.exit_code is None:
            state.state = WorkerState.RUNNING.value
            state.detail = f"trainer running as pid {record.pid}"
            state.write(layout)
            return state

        # It finished inside the confirmation window. The exit code
        # decides which outcome that was: a short run that succeeded is
        # COMPLETED, and only a non-zero exit is a launch failure.
        # Treating every early exit as a failure would misreport any
        # genuinely fast run as broken.
        state.exit_code = record.exit_code
        if record.exit_code == 0:
            state.state = WorkerState.COMPLETED.value
            state.completed_at = now()
            state.detail = "the trainer exited cleanly"
        else:
            state.state = WorkerState.FAILED.value
            state.failed_at = now()
            code, detail = classify_failure(
                record.exit_code, stdout=layout.stdout_log, stderr=layout.stderr_log
            )
            state.failure_code = code
            state.detail = f"the trainer exited immediately: {detail}"

        state.write(layout)
        self.result(run_id)
        return state

    def _compile(self, run_id: str, layout: RunLayout) -> tuple[list[str], Path]:
        """Rebuild the trainer invocation from the plan that arrived.

        Uses Phase 25's compiler rather than assembling a command here.
        A second implementation would drift, and then the command an
        operator reviewed on the control plane would not be the command
        that ran.
        """
        from luber_training.entities import TrainingDatasetRef
        from luber_training.plan import HardwareRequirements, TrainingPlan
        from luber_training.remote.staging import PLAN_PATH
        from luber_training.trainer_adapter import compile_command

        config_data = self.config()
        if not config_data.trainer_root:
            raise WorkerError(
                "this worker has no trainer root configured, so no training command can be resolved"
            )

        plan_path = layout.root / PLAN_PATH
        if not plan_path.is_file():
            raise WorkerError(f"run {run_id} has no plan; nothing can be compiled")
        payload = json.loads(plan_path.read_text(encoding="utf-8"))

        requirements = payload.get("requirements") or {}
        plan = TrainingPlan(
            plan_id=str(payload.get("plan_id", "")),
            run_id=str(payload["run_id"]),
            experiment_id=str(payload.get("experiment_id", "")),
            base_model_id=str(payload.get("base_model_id", "")),
            base_model_upstream_commit=str(payload.get("base_model_upstream_commit", "")),
            dataset_ref=TrainingDatasetRef(**payload["dataset_ref"]),
            config=_config_from(payload["config"]),
            execution_backend=str(payload.get("execution_backend", "remote-gpu")),
            requirements=HardwareRequirements(
                requires_cuda=bool(requirements.get("requires_cuda", True)),
                minimum_vram_mb=requirements.get("minimum_vram_mb"),
                minimum_gpu_count=int(requirements.get("minimum_gpu_count", 1)),
                supported_precision=tuple(requirements.get("supported_precision", ())),
                # Read back rather than defaulted. A device dropped here
                # would compile to `--device cuda` on a plan that was
                # placed on something else, and the worker would train
                # on hardware the control plane did not choose.
                execution_device=requirements.get("execution_device"),
                unknown_requirements=tuple(requirements.get("unknown_requirements", ())),
            ),
            # The placeholders become this worker's real directories.
            # This is the only place logical roots turn into paths.
            dataset_dir=str(layout.dataset_dir),
            output_dir=str(layout.output_dir),
            # The trainer's `--checkpoint-dir` is the root it *reads
            # base model weights from* — `acestep-v15-turbo` and the
            # rest — not a place to write. Phase 33 found this pointed
            # at the run's own empty output directory, which would have
            # made the first real dispatch fail at model loading. It is
            # the worker's shared checkpoint root, which is what that
            # logical root has always been for.
            checkpoint_dir=config_data.roots.checkpoint_root,
        )

        command = compile_command(plan, trainer_root=config_data.trainer_root)
        return list(command.argv), Path(command.working_directory)

    # ── observing ────────────────────────────────────────────────────
    def status(self, run_id: str) -> ExecutionState:
        """The run's state, refreshed against the operating system.

        Where the trainer has gone but this process never owned it, the
        exit code is unknown and the state says so. A worker that turned
        "the process is no longer there" into COMPLETED would be
        inventing the single fact everything downstream depends on.
        """
        layout = self.layout(run_id)
        state = ExecutionState.read(layout)
        if state is None:
            raise WorkerError(f"run {run_id} is not present on this worker")

        if state.state not in (WorkerState.RUNNING.value, WorkerState.CANCELLING.value):
            return state
        if state.process is None:
            return state

        process = TrainerProcess(layout)
        if process.running(state.process):
            return state

        exit_code = process.reap(state.process)
        state.exit_code = exit_code
        state.process.finished_at = state.process.finished_at or now()

        if state.state == WorkerState.CANCELLING.value:
            state.state = WorkerState.CANCELLED.value
            state.cancelled_at = now()
            state.detail = "the trainer stopped after a cancellation request"
        elif exit_code is None:
            # The honest answer. This invocation did not launch the
            # trainer, so the kernel gave its status to a process that
            # has since exited, and nobody recorded it.
            state.state = WorkerState.FAILED.value
            state.failed_at = now()
            state.failure_code = None
            state.detail = (
                "the trainer is no longer running and its exit status is UNKNOWN: this "
                "worker process did not launch it, so the exit code was never collected. "
                "Read the logs before concluding anything about how it ended"
            )
        elif exit_code == 0:
            state.state = WorkerState.COMPLETED.value
            state.completed_at = now()
            state.detail = "the trainer exited cleanly"
        else:
            state.state = WorkerState.FAILED.value
            state.failed_at = now()
            code, detail = classify_failure(
                exit_code, stdout=layout.stdout_log, stderr=layout.stderr_log
            )
            state.failure_code = code
            state.detail = detail

        state.write(layout)
        if state.state in TERMINAL_WORKER_STATES:
            self.result(run_id)
        return state

    def logs(
        self, run_id: str, *, stream: str = "stdout", offset: int = 0, limit: int = 262_144
    ) -> LogChunk:
        layout = self.layout(run_id)
        path = layout.stderr_log if stream == "stderr" else layout.stdout_log
        return read_log(path, offset=offset, limit=limit, stream=stream)

    def metrics_path(self, run_id: str) -> Path:
        return self.layout(run_id).metrics_jsonl

    def checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        return [item.to_dict() for item in discover_checkpoints(self.layout(run_id))]

    def result(self, run_id: str) -> RemoteResult:
        layout = self.layout(run_id)
        state = ExecutionState.read(layout)
        if state is None:
            raise WorkerError(f"run {run_id} is not present on this worker")
        identity, _ = self.identity()
        result = build_result(
            layout=layout,
            worker_id=identity.worker_id,
            state=state,
            capability_signature=identity.capability_signature,
        )
        result.write(layout)
        return result

    # ── stopping ─────────────────────────────────────────────────────
    def cancel(self, run_id: str, *, grace_seconds: float = 60.0) -> ExecutionState:
        """Stop the trainer, preserving everything it produced.

        The completion race is resolved in favour of completion. If the
        trainer finished successfully before the signal landed, and that
        was durably recorded, the run stays COMPLETED — overwriting a
        real success with CANCELLED would discard a checkpoint that
        exists and was paid for.
        """
        layout = self.layout(run_id)
        state = ExecutionState.read(layout)
        if state is None:
            raise WorkerError(f"run {run_id} is not present on this worker")

        # Refresh first: this is where a trainer that finished a moment
        # ago is noticed.
        state = self.status(run_id)
        if state.state in TERMINAL_WORKER_STATES:
            state.detail = (
                f"cancellation requested, but the run had already reached {state.state}; "
                "the earlier outcome stands"
            )
            state.write(layout)
            return state

        if state.process is None:
            state.state = WorkerState.CANCELLED.value
            state.cancelled_at = now()
            state.detail = "cancelled before a trainer was launched"
            state.write(layout)
            self.result(run_id)
            return state

        state.state = WorkerState.CANCELLING.value
        state.detail = "SIGTERM sent to the trainer's process group"
        state.write(layout)

        process = TrainerProcess(layout)
        record = process.cancel(state.process, grace_seconds=grace_seconds)
        state.process = record
        process.reap(record)
        state.exit_code = record.exit_code

        state.state = WorkerState.CANCELLED.value
        state.cancelled_at = now()
        state.detail = f"the trainer was stopped with {record.cancel_signal}"
        state.write(layout)
        self.result(run_id)
        return state

    # ── housekeeping ─────────────────────────────────────────────────
    def cleanup(self, run_id: str, *, remove_dataset: bool = False) -> dict[str, Any]:
        """Remove scratch. Never remove evidence.

        Temporary files and partial transfers go. Logs, metrics,
        checkpoints and the result manifest stay, whatever the run's
        outcome — a failed run's logs are the only thing that explains
        it, and deleting them as part of "cleanup" would destroy the
        diagnosis with the mess.
        """
        layout = self.layout(run_id)
        removed: list[str] = []
        freed = 0

        if layout.temp_dir.is_dir():
            for path in sorted(layout.temp_dir.rglob("*")):
                if path.is_file():
                    freed += path.stat().st_size
                    removed.append(path.relative_to(layout.root).as_posix())
            shutil.rmtree(layout.temp_dir, ignore_errors=True)
            layout.temp_dir.mkdir(parents=True, exist_ok=True)

        for path in sorted(layout.root.rglob("*.luber-partial")):
            if path.is_file():
                freed += path.stat().st_size
                removed.append(path.relative_to(layout.root).as_posix())
                path.unlink()

        dataset_removed = False
        if remove_dataset and layout.dataset_dir.is_dir():
            state = ExecutionState.read(layout)
            if state is None or state.state not in TERMINAL_WORKER_STATES:
                raise WorkerError(f"run {run_id} has not finished; its dataset is still in use")
            for path in sorted(layout.dataset_dir.rglob("*")):
                if path.is_file():
                    freed += path.stat().st_size
            shutil.rmtree(layout.dataset_dir, ignore_errors=True)
            dataset_removed = True

        return {
            "run_id": run_id,
            "removed": removed[:50],
            "removed_count": len(removed),
            "freed_bytes": freed,
            "dataset_removed": dataset_removed,
            "preserved": ["logs", "metrics", "checkpoints", "remote_result.json", "status.json"],
        }

    # ── reporting ────────────────────────────────────────────────────
    def describe(self) -> dict[str, Any]:
        """Everything a control plane needs to register this worker."""
        report = self.capabilities()
        identity, fingerprint_changed = self.identity(report)
        config = self.config()
        classification = report.classify()
        return {
            "protocol_version": REMOTE_PROTOCOL_VERSION,
            "identity": identity.to_dict(),
            "host_fingerprint_changed": fingerprint_changed,
            "classification": classification,
            "declared_classification": config.declared_classification,
            "capability_signature": report.signature(),
            "capabilities": report.to_dict(),
            "roots": config.roots.to_dict(),
            "trainer_root": config.trainer_root,
            "runs": self.known_runs(),
            "active_run_id": self.active_run(),
            "note": (
                "CUDA_TRAINING is only reported when torch demonstrates CUDA on this "
                "machine. A host with no NVIDIA hardware reports DEVELOPMENT_ONLY, which "
                f"is what {WorkerClassification.DEVELOPMENT_ONLY.value} means."
            ),
        }


def _config_from(payload: dict[str, Any]) -> Any:
    """Rebuild a TrainingConfig from the plan that arrived.

    Only fields the dataclass declares are used. A plan written by a
    newer control plane may carry a field this build does not know, and
    refusing the whole run over an unrecognised key would be worse than
    training with this build's default for it — the plan hash check has
    already established the two sides agree about what was dispatched.
    """
    from dataclasses import fields

    from luber_training.config import TrainingConfig

    known = {field.name for field in fields(TrainingConfig)}
    arguments: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in known:
            continue
        arguments[key] = tuple(value) if key == "target_modules" else value
    return TrainingConfig(**arguments)


def _uptime() -> float | None:
    """Seconds since boot, where the platform exposes it.

    Linux has /proc/uptime; macOS does not, and rather than shelling out
    to sysctl for a number nothing depends on, this returns None. An
    absent uptime is an absent uptime.
    """
    try:
        with open("/proc/uptime", encoding="utf-8") as handle:
            return float(handle.read().split()[0])
    except (OSError, ValueError, IndexError):
        pass
    try:
        return float(time.time() - os.stat("/proc/1").st_ctime)
    except (OSError, ValueError):
        return None


__all__ = [
    "HEARTBEAT_NAME",
    "IDENTITY_NAME",
    "WORKER_CONFIG_NAME",
    "RemoteWorker",
    "WorkerConfig",
    "WorkerError",
]
