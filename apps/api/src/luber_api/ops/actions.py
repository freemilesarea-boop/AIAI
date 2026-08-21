"""What the console is allowed to change, and what it refuses to pretend.

Everything here re-validates on the server. A disabled button is a
courtesy to the operator, not a control: the same request can arrive
from a stale tab, a double click, or curl, and each of those has to meet
the same answer. Step 66 in one line — and it is why every function
below re-reads the run before acting rather than trusting what the
caller believed its state to be.

Three refusals are load-bearing.

**Remote dispatch is not offered.** Reaching a rented GPU needs SSH
credentials, and Phase 27 put them in the operator's CLI on purpose.
Moving them into a process a browser can reach would be a change to that
security boundary made for the convenience of a button, so the console
says where dispatch happens instead of doing it.

**A cancellation the console cannot deliver is reported as a request.**
For the dry-run backend the cancel is real and immediate. For a remote
run the console records the intent and the run stays exactly as it was,
because a run shown CANCELLED is a GPU an operator believes they have
stopped paying for.

**Reconciliation reports; it does not tidy.** Where the worker's answer
maps onto a legal Phase 25 transition, the transition is made. Where it
does not — a worker saying RUNNING about a run the registry has already
written off as LOST — the finding is reported and the record is left
alone, because the state machine has no path back and forcing one would
mean the console rewriting history to look tidy.

**A retry is a new run.** There is no button that re-runs a failed run.
Phase 25 records lineage precisely so the third attempt can be traced to
the first without the first being edited.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from luber_api.ops.context import OpsConfigurationError, OpsContext
from luber_api.ops.readmodel import (
    CONTROL_PREFLIGHT_NAME,
    GATE_REPORT_NAME,
    OPS_ARTIFACT_DIR,
    RUN_CANCEL_REQUESTED,
    RUN_RECONCILED,
    RUN_VALIDATION_REQUESTED,
    OpsReadModel,
)
from luber_api.ops.schemas import ActionResponse
from luber_training.backends import DRY_RUN, REMOTE_GPU, LocalDryRunBackend
from luber_training.config import preset
from luber_training.entities import (
    Experiment,
    RunStatus,
    TrainingDatasetRef,
    TrainingRun,
)
from luber_training.gates import GateInputs
from luber_training.orchestrator import OrchestrationError


class ActionRefused(RuntimeError):
    """An action that is not legal in the current state.

    Carries the operator's sentence, not a stack trace. The endpoint
    turns it into a 409 — the request was well formed and the world
    said no, which is different from the request being wrong.
    """


@dataclass(frozen=True)
class ResolvedBuilds:
    """The two directories a run's gates read, found by identity.

    Located by matching the dataset and curation *ids* recorded on the
    run against the catalogue, never by a path travelling in a request.
    A build that has since been removed is a refusal with a reason
    rather than a traceback.
    """

    dataset_dir: Path
    curation_dir: Path


class OpsActions:
    """Every state change the console can make."""

    def __init__(self, context: OpsContext) -> None:
        self.context = context
        self.orchestrator = context.orchestrator
        self.registry = context.registry
        self.read = OpsReadModel(context)

    # ── helpers ──────────────────────────────────────────────────────
    def _run(self, run_id: str) -> TrainingRun:
        try:
            return self.orchestrator.get_run(run_id)
        except Exception as exc:
            raise ActionRefused(f"No run {run_id} is registered.") from exc

    def _run_directory(self, run: TrainingRun) -> Path:
        directory = Path(run.output_directory or self.orchestrator.artifacts_root / run.run_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _write_ops_artifact(self, run: TrainingRun, name: str, payload: dict[str, Any]) -> None:
        """Persist a report the console generated, beside the run.

        In a subdirectory of its own rather than among Phase 25's files:
        that package owns the layout of a run directory, and a console
        dropping files into it is one convention change away from a
        collision.
        """
        directory = self._run_directory(run) / OPS_ARTIFACT_DIR
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _resolve_builds(self, dataset_ref: TrainingDatasetRef) -> ResolvedBuilds:
        dataset_entry = next(
            (
                entry
                for entry in self.context.datasets.entries()
                if str(entry.lock.get("dataset_id", "")) == dataset_ref.dataset_id
            ),
            None,
        )
        curation_entry = next(
            (
                entry
                for entry in self.context.curations.entries()
                if str(entry.lock.get("curation_id", "")) == dataset_ref.curation_id
            ),
            None,
        )
        if dataset_entry is None:
            raise ActionRefused(
                f"No dataset build for {dataset_ref.dataset_id!r} is available to this "
                "deployment, so its lock cannot be verified."
            )
        if curation_entry is None:
            raise ActionRefused(
                f"No curation build for {dataset_ref.curation_id!r} is available to this "
                "deployment, so its lock cannot be verified."
            )
        try:
            return ResolvedBuilds(
                dataset_dir=self.context.datasets.directory_for(dataset_entry.build_id),
                curation_dir=self.context.curations.directory_for(curation_entry.build_id),
            )
        except OpsConfigurationError as exc:
            raise ActionRefused(str(exc)) from exc

    def _gate_inputs(self, builds: ResolvedBuilds) -> GateInputs:
        """Paths, exactly as Phase 25 defines them.

        The gates read from disk on every run rather than from state
        loaded earlier, which is what makes a re-validation describe the
        files as they are now rather than as they were on Monday.
        """
        return GateInputs(
            dataset_lock_path=builds.dataset_dir / "dataset_lock.json",
            dataset_manifest_path=builds.dataset_dir / "dataset_manifest.jsonl",
            curation_lock_path=builds.curation_dir / "curation_lock.json",
            curated_manifest_path=builds.curation_dir / "curated_manifest.jsonl",
        )

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
    ) -> Experiment:
        """Record a hypothesis. Nothing is trained and nothing is queued.

        Creation is deliberately inert: an experiment is a claim about
        what might be true, and the runs that test it are created
        separately and explicitly.
        """
        try:
            return self.orchestrator.create_experiment(
                name=name,
                hypothesis=hypothesis,
                base_model_id=base_model_id,
                description=description,
                operator=operator,
                tags=list(tags or []),
            )
        except Exception as exc:
            raise ActionRefused(
                f"The experiment could not be created: {exc}. A base model must be registered "
                "before an experiment can be built on it."
            ) from exc

    # ── runs ─────────────────────────────────────────────────────────
    def create_run(
        self,
        *,
        experiment_id: str,
        dataset_build_id: str,
        curation_build_id: str,
        preset_name: str,
        execution_backend: str,
        worker_id: str | None = None,
        parent_run_id: str | None = None,
        resume_from_checkpoint_id: str | None = None,
    ) -> TrainingRun:
        """Create a DRAFT run from two builds the deployment offers.

        The dataset reference is *derived* from the two locks rather
        than accepted from the request. A caller cannot state a digest:
        every hash on the run is read from the file it describes, which
        is the only version of this that a gate can later disagree with
        usefully.
        """
        try:
            dataset_lock = self.context.datasets.lock_for(dataset_build_id)
            curation_lock = self.context.curations.lock_for(curation_build_id)
        except OpsConfigurationError as exc:
            raise ActionRefused(str(exc)) from exc

        source_digest = str(curation_lock.get("source_dataset_lock_sha256") or "")
        dataset_digest = str(
            dataset_lock.get("lock_sha256") or dataset_lock.get("dataset_lock_sha256") or ""
        )
        if source_digest and dataset_digest and source_digest != dataset_digest:
            # Caught here rather than left to the gate. The gate would
            # also catch it, but only after a run exists and has been
            # marked FAILED — and a run that failed because two
            # selections did not belong together is noise in an
            # experiment's history.
            raise ActionRefused(
                f"Curation {curation_build_id!r} was built from a different dataset lock "
                f"({source_digest[:12]}) than {dataset_build_id!r} ({dataset_digest[:12]}). "
                "Select the pair that belong together."
            )

        dataset_ref = TrainingDatasetRef(
            dataset_id=str(dataset_lock.get("dataset_id", "")),
            dataset_lock_sha256=source_digest or dataset_digest,
            curation_id=str(curation_lock.get("curation_id", "")),
            curation_lock_sha256=str(curation_lock.get("curated_manifest_sha256", "")),
            curated_manifest_sha256=str(curation_lock.get("curated_manifest_sha256", "")),
            manifest_artifact_ref=f"curation://{curation_lock.get('curation_id')}/curated_manifest",
            sampling_weights_sha256=curation_lock.get("sampling_weights_sha256"),
            selected_track_count=int(curation_lock.get("selected_track_count") or 0),
            selected_hours=float(curation_lock.get("selected_hours") or 0.0),
        )

        if execution_backend not in {DRY_RUN, REMOTE_GPU}:
            raise ActionRefused(f"{execution_backend!r} is not a backend this build knows.")

        if worker_id:
            compatibility = next(
                (
                    item
                    for item in self.read.worker_compatibility(execution_backend=execution_backend)
                    if item.worker.worker_id == worker_id
                ),
                None,
            )
            if compatibility is None:
                raise ActionRefused(f"No worker {worker_id} is registered.")
            if not compatibility.compatible:
                raise ActionRefused(
                    f"Worker {compatibility.worker.name} cannot take this run: "
                    + "; ".join(compatibility.reasons)
                )

        try:
            config = preset(preset_name)
        except Exception as exc:
            raise ActionRefused(str(exc)) from exc

        try:
            return self.orchestrator.create_run(
                experiment_id=experiment_id,
                dataset_ref=dataset_ref,
                config=config,
                execution_backend=execution_backend,
                worker_id=worker_id,
                parent_run_id=parent_run_id,
                resume_from_checkpoint_id=resume_from_checkpoint_id,
            )
        except OrchestrationError as exc:
            raise ActionRefused(str(exc)) from exc
        except Exception as exc:
            raise ActionRefused(f"The run could not be created: {exc}") from exc

    def create_retry_run(self, run_id: str) -> ActionResponse:
        """A new run citing the old one. The old one is not touched.

        Everything that decided what was trained is carried across —
        experiment, dataset reference, configuration, backend, worker —
        so the retry is the same experiment repeated rather than a
        different one wearing its name. Changing a setting is a
        deliberate act on the new run, not a side effect of retrying.
        """
        run = self._run(run_id)
        if run.status not in {
            RunStatus.FAILED.value,
            RunStatus.CANCELLED.value,
            RunStatus.LOST.value,
        }:
            raise ActionRefused(
                f"Run {run_id} is {run.status}. A retry is only created once a run has stopped."
            )
        if run.status == RunStatus.LOST.value:
            raise ActionRefused(
                f"Run {run_id} is LOST: the worker stopped reporting and the trainer may still "
                "be running. Reconcile it before creating a retry — launching a second trainer "
                "against one checkpoint directory corrupts both."
            )

        try:
            retry = self.orchestrator.create_run(
                experiment_id=run.experiment_id,
                dataset_ref=run.dataset_ref,
                config=run.config,
                execution_backend=run.execution_backend,
                worker_id=run.worker_id,
                parent_run_id=run.run_id,
                resume_from_checkpoint_id=run.resume_from_checkpoint_id,
            )
        except Exception as exc:
            raise ActionRefused(f"The retry run could not be created: {exc}") from exc

        return ActionResponse(
            action="create_retry_run",
            performed=True,
            run_status=retry.status,
            detail=(
                f"Created {retry.run_id} as a retry of {run_id}. It is a DRAFT and nothing has "
                "been validated or dispatched."
            ),
            created_id=retry.run_id,
        )

    def validate_run(self, run_id: str) -> ActionResponse:
        """Run every gate, and record what they said.

        The gate report is written whether it passed or failed. A
        rights refusal that existed only in the terminal of whoever
        happened to run it is a refusal the next operator repeats.
        """
        run = self._run(run_id)
        if run.status != RunStatus.DRAFT.value:
            raise ActionRefused(f"Run {run_id} is {run.status}, not DRAFT.")

        builds = self._resolve_builds(run.dataset_ref)
        inputs = self._gate_inputs(builds)
        self.registry.append_audit(RUN_VALIDATION_REQUESTED, run_id, "run", source="console")

        try:
            run, report = self.orchestrator.validate_run(run_id, inputs, worker_id=run.worker_id)
        except OrchestrationError as exc:
            raise ActionRefused(str(exc)) from exc

        self._write_ops_artifact(run, GATE_REPORT_NAME, report.to_dict())

        if not report.passed:
            failure = report.first_failure
            return ActionResponse(
                action="validate",
                performed=True,
                run_status=run.status,
                detail=(
                    f"{failure.name if failure else 'A gate'} failed: "
                    f"{failure.detail if failure else 'see the gate report'}. The run is FAILED "
                    "and there is no override — resolve the cause and create a new run."
                ),
                outcome=report.failure_code(),
            )

        detail = f"Every gate passed. Run {run_id} is QUEUED."
        try:
            plan = self.orchestrator.compile_plan(run_id)
        except OrchestrationError as exc:
            return ActionResponse(
                action="validate",
                performed=True,
                run_status=run.status,
                detail=f"{detail} The plan could not be compiled: {exc}",
                outcome="PLAN_NOT_COMPILED",
            )

        worker = self.read.worker_entity(run.worker_id) if run.worker_id else None
        if worker is not None:
            backend = LocalDryRunBackend()
            preflight = self.orchestrator.preflight(
                run_id,
                plan,
                worker,
                backend,
                report,
                # A dry run trains nothing, so there is no artifact whose
                # provenance an unidentified revision would compromise.
                # Every other backend requires a clean tree.
                require_clean_repository=run.execution_backend != DRY_RUN,
            )
            self._write_ops_artifact(run, CONTROL_PREFLIGHT_NAME, preflight.to_dict())
            if not preflight.ok:
                detail += (
                    f" Control-plane preflight reported {len(preflight.problems)} problem(s); "
                    "see the preflight panel."
                )
            elif preflight.unknown:
                detail += (
                    f" Control-plane preflight passed with {len(preflight.unknown)} unknown(s), "
                    "which are not a pass."
                )
        else:
            detail += " No worker is assigned, so control-plane preflight was not run."

        return ActionResponse(
            action="validate",
            performed=True,
            run_status=run.status,
            detail=detail,
            outcome="PASSED",
        )

    def dispatch_run(self, run_id: str) -> ActionResponse:
        """Start a validated run, where this console is able to.

        Only the dry-run backend. That is not a limitation of the state
        machine — `start_run` would happily take a remote plan — but of
        what this process is trusted with: a remote launch needs
        credentials the console does not hold, and manufacturing them
        here would move a boundary Phase 27 set deliberately.

        The whole dispatch is held under the registry lock. Checking the
        state and then acting on it are two operations, and two
        concurrent dispatches — a double click, two tabs — can both pass
        the check before either acts. `start_run` alone is not enough:
        its own guard has the same gap, and the caller that loses the
        race would still go on to record metrics and complete a run it
        did not start.
        """
        with self.registry.lock():
            return self._dispatch_locked(run_id)

    def _dispatch_locked(self, run_id: str) -> ActionResponse:
        run = self._run(run_id)
        if run.status in {RunStatus.STARTING.value, RunStatus.RUNNING.value}:
            # Idempotent by state, not by a flag. A double click finds
            # the run already past QUEUED and is told so.
            return ActionResponse(
                action="dispatch",
                performed=False,
                run_status=run.status,
                detail=f"Run {run_id} is already {run.status}. Nothing was started.",
                outcome="ALREADY_STARTED",
            )
        if run.status != RunStatus.QUEUED.value:
            raise ActionRefused(
                f"Run {run_id} is {run.status}; only a QUEUED run can be dispatched. "
                + (
                    "Validate it first."
                    if run.status == RunStatus.DRAFT.value
                    else "A run that has stopped is never restarted — create a retry run."
                )
            )
        if run.execution_backend == REMOTE_GPU:
            raise ActionRefused(
                "Remote dispatch needs SSH credentials, which this console deliberately does "
                "not hold. Dispatch from the operator CLI: `luber-training remote run dispatch`."
            )
        if run.execution_backend != DRY_RUN:
            raise ActionRefused(
                f"No backend named {run.execution_backend!r} can be executed from this console."
            )
        if not run.worker_id:
            raise ActionRefused(
                f"Run {run_id} has no worker assigned; a run is dispatched to a worker."
            )

        # Gates decide whether the *data* may be trained on; preflight
        # decides whether this *machine* can execute the plan. Both, and
        # in that order — a run whose gates passed can still be pointed
        # at a worker that has never demonstrated CUDA, and finding that
        # out by renting the hardware is the expensive way.
        preflight = self.read.control_preflight_for(run_id)
        if not preflight.available:
            raise ActionRefused(
                "No control-plane preflight has been recorded for this run. Validate it "
                "against an assigned worker first."
            )
        if preflight.status == "FAIL":
            raise ActionRefused(
                "Control-plane preflight failed: " + "; ".join(preflight.problems[:3])
            )

        worker = self.read.worker_entity(run.worker_id)
        if worker is None:
            raise ActionRefused(f"Worker {run.worker_id} is no longer registered.")

        try:
            plan = self.orchestrator.compile_plan(run_id)
            # One backend instance for the whole lifecycle. It is
            # stateful — it remembers which runs it started and which
            # were cancelled — and a second instance would collect
            # metrics for a run it had never seen begin.
            backend = LocalDryRunBackend()
            self.orchestrator.start_run(run_id, plan, worker, backend)
            events = backend.collect_metrics(plan)
            self.orchestrator.record_metrics(run_id, events)
            finished = self.orchestrator.complete_run(run_id)
        except OrchestrationError as exc:
            raise ActionRefused(str(exc)) from exc

        return ActionResponse(
            action="dispatch",
            performed=True,
            run_status=finished.status,
            detail=(
                f"The dry run completed and recorded {len(events)} metric event(s), every one "
                "marked SIMULATED. Nothing was trained and no checkpoint was produced."
            ),
            outcome="DRY_RUN_COMPLETED",
        )

    def cancel_run(self, run_id: str) -> ActionResponse:
        """Stop a run, or record that stopping it has been asked for.

        The distinction is the point. A dry run stops here and now. A
        remote run does not, because this console cannot signal the
        worker — and a run displayed CANCELLED is a GPU an operator
        believes they have stopped paying for.
        """
        run = self._run(run_id)
        if run.is_terminal:
            raise ActionRefused(f"Run {run_id} is already {run.status}.")
        if run.status == RunStatus.DRAFT.value:
            raise ActionRefused(
                f"Run {run_id} is a DRAFT and has started nothing; there is nothing to cancel."
            )
        if run.status == RunStatus.LOST.value:
            raise ActionRefused(
                f"Run {run_id} is LOST. Reconcile it first: cancelling a run whose worker is "
                "not answering records an intention nothing can act on."
            )

        self.registry.append_audit(
            RUN_CANCEL_REQUESTED,
            run_id,
            "run",
            source="console",
            backend=run.execution_backend,
            status_at_request=run.status,
        )

        if run.execution_backend == DRY_RUN:
            plan = self.orchestrator.compile_plan(run_id)
            cancelled = self.orchestrator.cancel_run(run_id, plan, LocalDryRunBackend())
            return ActionResponse(
                action="cancel",
                performed=True,
                run_status=cancelled.status,
                detail=(
                    "The dry run was cancelled. Metrics, logs and any finished checkpoints "
                    "were kept."
                ),
                outcome="CANCELLED",
            )

        return ActionResponse(
            action="cancel",
            performed=False,
            run_status=run.status,
            detail=(
                "Graceful cancellation requested and recorded in the audit log. The run "
                f"remains {run.status}: this console holds no transport to the worker, so "
                "nothing has confirmed the trainer stopped. Deliver the signal with "
                "`luber-training remote run cancel`, then reconcile."
            ),
            outcome="CANCEL_REQUESTED",
        )

    def reconcile_run(self, run_id: str) -> ActionResponse:
        """Ask the worker what is actually happening. Changes nothing there.

        Safe to repeat, which is what makes it the right first move
        after any ambiguity. What it does with the answer is
        conservative: a legal Phase 25 transition is made, and anything
        else is reported rather than forced.
        """
        run = self._run(run_id)
        reason = self.context.transport_unavailable_reason()
        if reason is not None:
            raise ActionRefused(reason)

        try:
            backend = self._remote_backend()
        except OpsConfigurationError as exc:
            raise ActionRefused(str(exc)) from exc

        report = backend.reconcile(run_id)
        self.registry.append_audit(
            RUN_RECONCILED,
            run_id,
            "run",
            source="console",
            outcome=report.outcome,
            worker_state=report.worker_state,
        )

        applied: str | None = None
        target = report.run_status
        if target and target != run.status and run.can_transition_to(target):
            updated = self.orchestrator.transition_run(
                run_id,
                target,
                error_code=report.failure_code,
                error_message=report.detail or None,
            )
            applied = updated.status
        elif target and target != run.status:
            applied = None

        detail = report.detail or f"The worker reported {report.outcome}."
        if applied:
            detail += f" The run moved to {applied}."
        elif target and target != run.status:
            detail += (
                f" The worker's view implies {target}, but a run that is {run.status} has no "
                f"legal transition to it. The record is left as it is rather than rewritten."
            )

        return ActionResponse(
            action="reconcile",
            performed=True,
            run_status=applied or run.status,
            detail=detail,
            outcome=report.outcome,
        )

    def _remote_backend(self) -> Any:
        from luber_training.remote.backend import RemoteGpuBackend
        from luber_training.remote.client import LocalWorkerClient
        from luber_training.remote.transport import LocalArtifactTransport
        from luber_training.remote.worker import RemoteWorker

        root = self.context.worker_root()
        worker = RemoteWorker(root)
        return RemoteGpuBackend(
            LocalWorkerClient(root),
            LocalArtifactTransport(Path(worker.config().roots.run_root)),
            liveness=self.context.liveness,
        )


__all__ = ["ActionRefused", "OpsActions"]
