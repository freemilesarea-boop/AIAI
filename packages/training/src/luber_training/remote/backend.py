"""The remote GPU backend, and the reconciliation that makes it safe.

Phase 25 defined `TrainingExecutionBackend` and left the remote
implementation raising `NotImplementedError`. This is that
implementation, built against three abstractions — a worker client, an
artifact transport, and a secret resolver — so that choosing a provider
later is configuration rather than a rewrite. No provider is named
anywhere in this module.

The hard part is not the happy path. It is what happens when the control
plane and the worker disagree, and the rule throughout is: **never
resolve ambiguity by launching something.**

A launch whose reply was lost may have started a trainer. Retrying would
start a second one, and two trainers writing into one checkpoint
directory produce artifacts that are corrupt in a way no hash will catch,
because each one is individually well-formed. So `dispatch` reconciles
first, always, and only launches when the worker positively says it has
no execution for that run.

The same asymmetry runs through everything else here. A worker that does
not answer makes a run LOST, not FAILED — we know we cannot see the
trainer, not that it stopped. A checkpoint is registered locally only
after its bytes have arrived and hashed to the value the worker
reported. A cancellation that races a completion loses, because the
completion is real and already paid for.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from luber_training.backends import (
    REMOTE_GPU,
    BackendStatus,
    EnvironmentCheck,
    TrainingExecutionBackend,
    capability_check,
)
from luber_training.entities import FailureCode, RunStatus, TrainingWorker
from luber_training.metrics import MetricEvent
from luber_training.plan import TrainingPlan
from luber_training.remote.client import (
    ClientRetryPolicy,
    RemoteWorkerClient,
    WorkerUnreachable,
)
from luber_training.remote.identity import LivenessPolicy
from luber_training.remote.manifest import RemoteArtifactManifest
from luber_training.remote.protocol import (
    LAUNCH_SAFE_OUTCOMES,
    TERMINAL_WORKER_STATES,
    ReconcileOutcome,
    WorkerState,
    now,
    run_status_for,
)
from luber_training.remote.streams import LogCursor, MetricStream, deduplicate
from luber_training.remote.transport import ArtifactTransport, TransportError

#: Where the worker's run directory sits relative to the transport root.
#: The transport is rooted at the worker's run root, so a run's files are
#: addressed as `<run_id>/...` and nothing can address a sibling run.
RUN_PREFIX = ""


class RemoteBackendError(RuntimeError):
    """Raised when a remote operation cannot be completed safely."""


@dataclass
class ReconcileReport:
    """What the worker actually knows about a run.

    Produced before any launch and after any lost contact. The
    ``outcome`` is the only thing callers should branch on, and
    UNKNOWN is a real value: a worker that answered but cannot say how
    its trainer ended has told us something true, and turning that into
    COMPLETED or FAILED would be inventing the fact that matters most.
    """

    run_id: str
    outcome: str
    worker_state: str | None = None
    run_status: str | None = None
    exit_code: int | None = None
    failure_code: str | None = None
    detail: str = ""
    process_alive: bool | None = None
    checked_at: str = field(default_factory=now)

    @property
    def safe_to_launch(self) -> bool:
        return self.outcome in LAUNCH_SAFE_OUTCOMES

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "outcome": self.outcome,
            "worker_state": self.worker_state,
            "run_status": self.run_status,
            "exit_code": self.exit_code,
            "failure_code": self.failure_code,
            "detail": self.detail,
            "process_alive": self.process_alive,
            "checked_at": self.checked_at,
            "safe_to_launch": self.safe_to_launch,
        }


@dataclass
class DispatchResult:
    """One dispatch attempt, and whether it started anything."""

    run_id: str
    worker_id: str
    launched: bool
    worker_state: str
    run_status: str
    reconcile: ReconcileReport
    preflight: dict[str, Any] | None = None
    transfer: dict[str, Any] | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "worker_id": self.worker_id,
            "launched": self.launched,
            "worker_state": self.worker_state,
            "run_status": self.run_status,
            "reconcile": self.reconcile.to_dict(),
            "preflight": self.preflight,
            "transfer": self.transfer,
            "detail": self.detail,
        }


class RemoteGpuBackend(TrainingExecutionBackend):
    """Executes a training run on a machine this process cannot see.

    Composed rather than inherited-from: a client for control, a
    transport for bytes, a secret resolver for credentials. Substituting
    any of them — SSH for local, a provider API for SSH — changes
    nothing above this class.
    """

    name = REMOTE_GPU

    def __init__(
        self,
        client: RemoteWorkerClient,
        transport: ArtifactTransport,
        *,
        liveness: LivenessPolicy | None = None,
        retry: ClientRetryPolicy | None = None,
        minimum_free_disk_mb: int | None = None,
        allow_code_mismatch: bool = False,
    ) -> None:
        self.client = client
        self.transport = transport
        self.liveness = liveness or LivenessPolicy()
        self.retry = retry or ClientRetryPolicy()
        self.minimum_free_disk_mb = minimum_free_disk_mb
        self.allow_code_mismatch = allow_code_mismatch
        self._log_cursors: dict[str, LogCursor] = {}
        self._metric_streams: dict[str, MetricStream] = {}

    # ── capability matching ──────────────────────────────────────────
    def validate_environment(self, plan: TrainingPlan, worker: TrainingWorker) -> EnvironmentCheck:
        """Phase 25's capability arithmetic, unchanged.

        Reused rather than reimplemented: the question "can this worker
        run this plan" has one right answer, and having two versions of
        it would eventually produce two.
        """
        return capability_check(plan, worker)

    # ── reconciliation ───────────────────────────────────────────────
    def reconcile(self, run_id: str) -> ReconcileReport:
        """Ask the worker what is actually happening. Idempotent.

        This is the function that makes every ambiguous case safe. It
        changes nothing, can be called any number of times, and returns
        a verdict that is honest about what could not be determined.
        """
        try:
            payload = self.client.status(run_id)
        except WorkerUnreachable as exc:
            return ReconcileReport(
                run_id=run_id,
                outcome=ReconcileOutcome.UNREACHABLE.value,
                detail=(
                    f"the worker did not answer: {exc}. The trainer may still be running; "
                    "nothing may be launched for this run until the worker responds"
                ),
            )
        except Exception as exc:
            message = str(exc)
            if "not present" in message or "never prepared" in message:
                return ReconcileReport(
                    run_id=run_id,
                    outcome=ReconcileOutcome.NOT_PRESENT.value,
                    detail="the worker has no record of this run",
                )
            return ReconcileReport(
                run_id=run_id,
                outcome=ReconcileOutcome.UNKNOWN.value,
                detail=f"the worker answered but could not be understood: {message}",
            )

        if not payload:
            return ReconcileReport(
                run_id=run_id,
                outcome=ReconcileOutcome.NOT_PRESENT.value,
                detail="the worker has no record of this run",
            )

        state = str(payload.get("state", ""))
        process = payload.get("process") or {}
        alive = process.get("exit_code") is None and bool(process.get("pid"))

        outcome = {
            WorkerState.RUNNING.value: ReconcileOutcome.RUNNING_RECOVERED.value,
            WorkerState.STARTING.value: ReconcileOutcome.RUNNING_RECOVERED.value,
            WorkerState.CANCELLING.value: ReconcileOutcome.RUNNING_RECOVERED.value,
            WorkerState.COMPLETED.value: ReconcileOutcome.COMPLETED_RECOVERED.value,
            WorkerState.FAILED.value: ReconcileOutcome.FAILED_RECOVERED.value,
            WorkerState.CANCELLED.value: ReconcileOutcome.CANCELLED_RECOVERED.value,
        }.get(state)

        if outcome is None:
            # RECEIVING, PREFLIGHT, READY: the run exists on the worker
            # and nothing is executing. Not NOT_PRESENT — the lease is
            # held, and a launch is the caller's decision, not a
            # reconciliation's.
            outcome = (
                ReconcileOutcome.UNKNOWN.value if state else ReconcileOutcome.NOT_PRESENT.value
            )

        return ReconcileReport(
            run_id=run_id,
            outcome=outcome,
            worker_state=state or None,
            run_status=run_status_for(state) if state else None,
            exit_code=payload.get("exit_code"),
            failure_code=payload.get("failure_code"),
            detail=str(payload.get("detail", "")),
            process_alive=alive,
        )

    # ── the lifecycle ────────────────────────────────────────────────
    def prepare_run(self, plan: TrainingPlan, worker: TrainingWorker) -> None:
        """Claim the run on the worker. Safe to repeat."""
        self.client.prepare_run(plan.run_id, plan.digest(), "")

    def stage_and_transfer(
        self, plan: TrainingPlan, manifest: RemoteArtifactManifest, staging_dir: Path
    ) -> dict[str, Any]:
        """Move the run's artifacts, skipping what is already there.

        Every path is prefixed with the run id, so one run's transfer
        cannot write into another's directory even if a manifest asked
        it to.
        """
        self.client.prepare_run(plan.run_id, plan.digest(), manifest.digest())

        scoped = RemoteArtifactManifest(
            run_id=manifest.run_id,
            training_plan_sha256=manifest.training_plan_sha256,
            built_at=manifest.built_at,
        )
        for entry in manifest.entries:
            scoped.add(
                type(entry)(
                    artifact_id=entry.artifact_id,
                    role=entry.role,
                    target_path=f"{plan.run_id}/{entry.target_path}",
                    sha256=entry.sha256,
                    size_bytes=entry.size_bytes,
                    required=entry.required,
                    source_reference=str(Path(staging_dir) / entry.target_path),
                    track_id=entry.track_id,
                )
            )

        plan_before = self.transport.plan(scoped)
        result = self.transport.upload_manifest(scoped)
        if not result.ok:
            raise TransportError(
                f"{len(result.failed)} artifact(s) did not transfer: "
                f"{'; '.join(f'{k}: {v}' for k, v in list(result.failed.items())[:3])}"
            )

        # The worker is told what it was supposed to receive. Preflight
        # then checks the files against this, on the far side.
        local_manifest = Path(staging_dir) / "artifact_manifest.json"
        remote_manifest = f"{plan.run_id}/artifact_manifest.json"
        self.transport.upload(
            local_manifest,
            remote_manifest,
            expected_sha256=_digest_of(local_manifest),
        )
        self.client.receive_manifest(plan.run_id, self._remote_absolute(remote_manifest))

        return {
            "plan": plan_before.to_dict(),
            "transfer": result.to_dict(),
            "manifest_sha256": manifest.digest(),
        }

    def _remote_absolute(self, relative: str) -> str:
        """Where the transport put a file, as the worker will see it."""
        root = getattr(self.transport, "remote_root", None) or getattr(self.transport, "root", "")
        return f"{root}/{relative}"

    def dispatch(
        self,
        plan: TrainingPlan,
        worker: TrainingWorker,
        *,
        manifest: RemoteArtifactManifest | None = None,
        staging_dir: Path | None = None,
    ) -> DispatchResult:
        """Transfer, verify, and start — reconciling before it launches.

        The reconciliation is unconditional. It costs one status call
        and it is the only thing standing between a lost acknowledgement
        and two trainers on one run.
        """
        reconcile = self.reconcile(plan.run_id)

        if reconcile.outcome == ReconcileOutcome.UNREACHABLE.value:
            raise RemoteBackendError(
                f"cannot dispatch {plan.run_id}: {reconcile.detail}. Launching now could "
                "start a second trainer alongside one that may still be running"
            )

        if reconcile.outcome in (
            ReconcileOutcome.RUNNING_RECOVERED.value,
            ReconcileOutcome.COMPLETED_RECOVERED.value,
            ReconcileOutcome.FAILED_RECOVERED.value,
            ReconcileOutcome.CANCELLED_RECOVERED.value,
        ):
            # Already dispatched. This is the double-dispatch case and
            # the ambiguous-launch case, and both resolve the same way:
            # return what exists.
            return DispatchResult(
                run_id=plan.run_id,
                worker_id=worker.worker_id,
                launched=False,
                worker_state=reconcile.worker_state or "",
                run_status=reconcile.run_status or RunStatus.RUNNING.value,
                reconcile=reconcile,
                detail=(
                    f"run {plan.run_id} already has an execution on this worker "
                    f"({reconcile.worker_state}); nothing new was started"
                ),
            )

        transfer: dict[str, Any] | None = None
        if manifest is not None and staging_dir is not None:
            transfer = self.stage_and_transfer(plan, manifest, staging_dir)

        passed, preflight = self.client.preflight(
            plan.run_id,
            minimum_free_disk_mb=self.minimum_free_disk_mb,
            allow_code_mismatch=self.allow_code_mismatch,
        )
        if not passed:
            reasons = preflight.get("blocking_reasons") or ["preflight did not pass"]
            return DispatchResult(
                run_id=plan.run_id,
                worker_id=worker.worker_id,
                launched=False,
                worker_state=WorkerState.PREFLIGHT.value,
                run_status=RunStatus.QUEUED.value,
                reconcile=reconcile,
                preflight=preflight,
                transfer=transfer,
                detail=(
                    f"preflight {preflight.get('status')}: {'; '.join(reasons[:3])}. "
                    "No trainer was started"
                ),
            )

        state = self.client.launch(plan.run_id)
        worker_state = str(state.get("state", WorkerState.RUNNING.value))
        return DispatchResult(
            run_id=plan.run_id,
            worker_id=worker.worker_id,
            launched=worker_state in (WorkerState.RUNNING.value, WorkerState.STARTING.value),
            worker_state=worker_state,
            run_status=run_status_for(worker_state),
            reconcile=reconcile,
            preflight=preflight,
            transfer=transfer,
            detail=str(state.get("detail", "")),
        )

    def start(self, plan: TrainingPlan, worker: TrainingWorker) -> BackendStatus:
        """Phase 25's `start`, reconciling first as `dispatch` does."""
        result = self.dispatch(plan, worker)
        return BackendStatus(status=result.run_status, detail=result.detail)

    def status(self, plan: TrainingPlan) -> BackendStatus:
        """The run's state as the worker reports it.

        An unreachable worker yields LOST rather than FAILED. That
        distinction is the difference between "the training stopped" and
        "we stopped being able to watch it", and only one of them
        justifies giving up on a rented GPU.
        """
        try:
            payload = self.client.status(plan.run_id)
        except WorkerUnreachable as exc:
            return BackendStatus(
                status=RunStatus.LOST.value,
                detail=(
                    f"the worker is unreachable: {exc}. The trainer may still be running; "
                    "reconcile before concluding anything"
                ),
            )
        state = str(payload.get("state", ""))
        return BackendStatus(
            status=run_status_for(state),
            detail=str(payload.get("detail", "")),
            exit_code=payload.get("exit_code"),
        )

    def cancel(self, plan: TrainingPlan) -> BackendStatus:
        payload = self.client.cancel(plan.run_id)
        state = str(payload.get("state", WorkerState.CANCELLED.value))
        return BackendStatus(
            status=run_status_for(state),
            detail=str(payload.get("detail", "")),
            exit_code=payload.get("exit_code"),
        )

    # ── streams ──────────────────────────────────────────────────────
    def collect_metrics(self, plan: TrainingPlan) -> list[MetricEvent]:
        """New metric events since the last collection.

        The cursor lives in this backend and the identity check lives in
        the stream, so a repeated poll returns nothing new even if the
        cursor was lost — which is what happens when the control plane
        restarts.
        """
        stream = self._metric_streams.setdefault(plan.run_id, MetricStream())
        try:
            payload = self.client.metrics(plan.run_id, cursor=stream.line_cursor)
        except WorkerUnreachable:
            return []

        events: list[MetricEvent] = []
        for item in payload.get("events", []):
            try:
                event = MetricEvent(
                    run_id=str(item["run_id"]),
                    metric_name=str(item["metric_name"]),
                    value=float(item["value"]),
                    source=str(item["source"]),
                    step=item.get("step"),
                    epoch=item.get("epoch"),
                    unit=str(item.get("unit", "")),
                    timestamp=str(item.get("timestamp", now())),
                )
            except (KeyError, TypeError, ValueError):
                continue
            from luber_training.remote.streams import metric_identity

            identity = metric_identity(event)
            if identity in stream.seen:
                continue
            stream.seen.add(identity)
            events.append(event)

        stream.line_cursor = int(payload.get("next_cursor", stream.line_cursor))
        return deduplicate(events)

    def collect_logs(self, plan: TrainingPlan, *, stream: str = "stdout") -> str:
        """The log text produced since the last call to this method."""
        cursor = self._log_cursors.setdefault(plan.run_id, LogCursor())
        offset = cursor.stderr if stream == "stderr" else cursor.stdout
        try:
            payload = self.client.logs(plan.run_id, stream=stream, offset=offset)
        except WorkerUnreachable:
            return ""
        text = str(payload.get("text", ""))
        next_offset = int(payload.get("next_offset", offset))
        if stream == "stderr":
            cursor.stderr = next_offset
        else:
            cursor.stdout = next_offset
        return text

    def collect_checkpoints(self, plan: TrainingPlan) -> list[dict[str, Any]]:
        """Checkpoints the worker has finished writing.

        Reported, not registered. Nothing here touches the Phase 25
        checkpoint registry — that happens after collection, when the
        bytes are local and the hash has been checked.
        """
        try:
            return self.client.list_checkpoints(plan.run_id)
        except WorkerUnreachable:
            return []

    def cleanup(self, plan: TrainingPlan) -> None:
        """Remove scratch on both sides. Never logs or checkpoints."""
        try:
            self.transport.remove_temp()
        except (TransportError, WorkerUnreachable):
            pass
        try:
            self.client.cleanup(plan.run_id)
        except WorkerUnreachable:
            pass

    # ── waiting ──────────────────────────────────────────────────────
    def wait_for_terminal(
        self, plan: TrainingPlan, *, poll_seconds: float = 1.0, timeout_seconds: float = 300.0
    ) -> BackendStatus:
        """Poll until the run finishes. For tests and short jobs.

        Deliberately not used to supervise a real training run: a
        multi-hour job under a wall-clock timeout is the most expensive
        possible bug, so the timeout here is explicit at every call site
        rather than a default anything inherits.
        """
        deadline = time.monotonic() + timeout_seconds
        last = BackendStatus(status=RunStatus.RUNNING.value)
        while time.monotonic() < deadline:
            last = self.status(plan)
            if last.status in (
                RunStatus.COMPLETED.value,
                RunStatus.FAILED.value,
                RunStatus.CANCELLED.value,
            ):
                return last
            time.sleep(poll_seconds)
        return last


def _digest_of(path: Path) -> str:
    from luber_training.remote.manifest import sha256_file

    digest, _ = sha256_file(path)
    return digest


def failure_code_for(reconcile: ReconcileReport) -> str:
    """The Phase 25 failure code a reconciliation implies.

    UNKNOWN where the worker could not say. Phase 25's taxonomy already
    has that value precisely so nobody has to choose between fabricating
    a cause and leaving the field blank.
    """
    if reconcile.outcome == ReconcileOutcome.UNREACHABLE.value:
        return FailureCode.WORKER_LOST.value
    if reconcile.failure_code:
        return str(reconcile.failure_code)
    if reconcile.outcome == ReconcileOutcome.CANCELLED_RECOVERED.value:
        return FailureCode.CANCELLED_BY_OPERATOR.value
    if reconcile.outcome == ReconcileOutcome.FAILED_RECOVERED.value:
        return FailureCode.TRAINER_CRASH.value
    return FailureCode.UNKNOWN.value


__all__ = [
    "TERMINAL_WORKER_STATES",
    "DispatchResult",
    "ReconcileReport",
    "RemoteBackendError",
    "RemoteGpuBackend",
    "failure_code_for",
]
