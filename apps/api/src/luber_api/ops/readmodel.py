"""Normalised operator views over the training and evaluation registries.

The browser never reads a registry file. Everything it sees is built
here, from records the Phase 25 and Phase 26 packages own, and the
translation is where four rules are enforced that a direct read could
not enforce at all.

**Nothing is invented.** Every field traces to a record, a lock or a
measurement. Where a fact was never established the view carries
``None`` and the UI renders UNKNOWN — never zero, never a default, never
a green tick. A run whose disk requirement nobody measured says so.

**Local and remote stay apart.** The control plane's ``RunStatus`` and
the worker's ``WorkerState`` are separate fields on separate models,
because they can legitimately disagree and the disagreement is the
information. A worker reporting RUNNING while the registry says LOST is
precisely the case reconciliation exists for; a single merged status
would delete it.

**Secrets have nowhere to go.** Credential references are reduced to a
boolean before they leave, log text is redacted here rather than in the
browser, and no view carries an SSH endpoint, a key reference or an
absolute filesystem path.

**Reading is cheap and bounded.** Lists are paginated and filtered
server-side, metric series are thinned, logs are read from an offset. A
registry with ten thousand runs must not become one response, and the
console must not become the reason a run's directory is read end to end
every three seconds.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypeVar

from luber_api.ops.context import OpsContext
from luber_api.ops.failures import meaning_for
from luber_api.ops.redaction import redact_document, redact_text
from luber_api.ops.schemas import (
    ActionAvailability,
    AuditEvent,
    BaselineResponse,
    BuildOption,
    CandidateSummary,
    CatalogueResponse,
    CheckpointComparisonResponse,
    CheckpointComparisonRow,
    CheckpointDetail,
    CheckpointListResponse,
    CheckpointSummary,
    ComparisonView,
    CostView,
    CountBreakdown,
    DatasetRefView,
    EvaluationDetail,
    EvaluationListResponse,
    EvaluationSummary,
    ExperimentDetail,
    ExperimentListResponse,
    ExperimentSummary,
    FailureView,
    GateOutcomeView,
    GateView,
    GpuTelemetryView,
    HeartbeatView,
    HumanReviewView,
    LogView,
    ModelBaselineView,
    OverviewResponse,
    Page,
    PreflightCheckView,
    PreflightView,
    PresetOption,
    PromotionReviewView,
    QualificationSummary,
    RemoteStateView,
    ReproducibilityView,
    RunDetail,
    RunListResponse,
    RunProgress,
    RunSummary,
    StagingView,
    SystemCheck,
    TimelineEntry,
    TrainingConfigView,
    WorkerCapabilitiesView,
    WorkerCompatibility,
    WorkerDetail,
    WorkerListResponse,
    WorkerSummary,
)
from luber_api.ops.streams import (
    TELEMETRY_METRICS,
    build_series,
    latest_value,
    load_metrics,
    read_stream,
    tail_lines,
)
from luber_training.backends import DRY_RUN, REMOTE_GPU
from luber_training.config import PRESET_INTENT, PRESETS, TrainingConfig, preset
from luber_training.entities import (
    CheckpointKind,
    CheckpointStatus,
    RunStatus,
    TrainingWorker,
    WorkerClass,
)
from luber_training.remote.protocol import REMOTE_PROTOCOL_VERSION, run_status_for

#: An audit event the console itself appends. Named in the Phase 25
#: style and stored in the same append-only log, so an operator asking
#: "what happened to this run" sees a console action in the same
#: ordered history as everything else.
RUN_CANCEL_REQUESTED = "RUN_CANCEL_REQUESTED"
RUN_RECONCILED = "RUN_RECONCILED"
RUN_VALIDATION_REQUESTED = "RUN_VALIDATION_REQUESTED"

#: Where the console keeps the reports it generates for a run. A
#: subdirectory of its own rather than files beside Phase 25's, so
#: nothing here can collide with a convention that package adds later.
OPS_ARTIFACT_DIR = "ops"
GATE_REPORT_NAME = "gate_report.json"
CONTROL_PREFLIGHT_NAME = "preflight.json"

#: The order of the Phase 25 run state machine, for the timeline.
LINEAR_STATES: tuple[str, ...] = (
    RunStatus.DRAFT.value,
    RunStatus.VALIDATING.value,
    RunStatus.QUEUED.value,
    RunStatus.STARTING.value,
    RunStatus.RUNNING.value,
)

TERMINAL_STATES: tuple[str, ...] = (
    RunStatus.COMPLETED.value,
    RunStatus.FAILED.value,
    RunStatus.CANCELLED.value,
    RunStatus.LOST.value,
)

T = TypeVar("T")


def _now() -> datetime:
    return datetime.now(UTC)


def _parse(timestamp: str | None) -> datetime | None:
    if not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _paginate(items: list[T], limit: int, offset: int) -> tuple[list[T], Page]:
    total = len(items)
    window = items[offset : offset + limit]
    return window, Page(total=total, limit=limit, offset=offset, returned=len(window))


def _counts(values: list[str]) -> CountBreakdown:
    breakdown: dict[str, int] = {}
    for value in values:
        breakdown[value] = breakdown.get(value, 0) + 1
    return CountBreakdown(total=len(values), by_state=dict(sorted(breakdown.items())))


class OpsReadModel:
    """Every operator view, built from the registries on demand.

    Constructed per request. The registries are files, reading them is a
    directory listing and a handful of `json.loads`, and a cache would
    buy microseconds at the cost of a console that disagrees with the
    CLI an operator just ran in the next terminal.
    """

    def __init__(self, context: OpsContext) -> None:
        self.context = context
        self.registry = context.registry
        self.evaluations_registry = context.evaluations
        self.orchestrator = context.orchestrator

    # ── caches for one request ───────────────────────────────────────
    def _all(self, collection: str) -> list[dict[str, Any]]:
        cache: dict[str, list[dict[str, Any]]] = self.__dict__.setdefault("_collection_cache", {})
        if collection not in cache:
            try:
                cache[collection] = self.registry.list_all(collection)
            except Exception:
                cache[collection] = []
        return cache[collection]

    def _all_evaluations(self, collection: str) -> list[dict[str, Any]]:
        cache: dict[str, list[dict[str, Any]]] = self.__dict__.setdefault("_eval_cache", {})
        if collection not in cache:
            try:
                cache[collection] = self.evaluations_registry.list_all(collection)
            except Exception:
                cache[collection] = []
        return cache[collection]

    def _experiment_names(self) -> dict[str, str]:
        return {
            str(record.get("experiment_id")): str(record.get("name", ""))
            for record in self._all("experiments")
        }

    def _worker_names(self) -> dict[str, str]:
        return {
            str(record.get("worker_id")): str(record.get("name", ""))
            for record in self._all("workers")
        }

    def _checkpoint_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self._all("checkpoints"):
            run_id = str(record.get("run_id", ""))
            counts[run_id] = counts.get(run_id, 0) + 1
        return counts

    def _cancel_requests(self) -> dict[str, str]:
        """Runs the console has asked to stop, and when.

        Read from the audit log rather than a field on the run: Phase 25
        owns the run record, and a cancellation the console requested
        but nothing has confirmed is not a change to the run's state.
        """
        requested: dict[str, str] = {}
        for event in self.registry.audit_events():
            if event.get("event") == RUN_CANCEL_REQUESTED:
                requested[str(event.get("entity_id"))] = str(event.get("timestamp", ""))
        return requested

    # ── paths ────────────────────────────────────────────────────────
    def run_directory(self, run: dict[str, Any]) -> Path | None:
        directory = run.get("output_directory")
        if not directory:
            return None
        path = Path(str(directory))
        return path if path.is_dir() else None

    def worker_run_directory(self, run_id: str) -> Path | None:
        """This run's directory on a worker, when one is reachable.

        Only ever a *local* worker root — see `OpsContext`. Reading it is
        a filesystem read of the same files Phase 27 writes, not a
        command invocation, so opening a run page never starts a
        subprocess on a machine that might be busy training.
        """
        if self.context.transport_unavailable_reason() is not None:
            return None
        try:
            from luber_training.remote.paths import RunLayout
            from luber_training.remote.worker import RemoteWorker

            worker = RemoteWorker(self.context.worker_root())
            layout = RunLayout.for_run(Path(worker.config().roots.run_root), run_id)
        except Exception:
            return None
        return layout.root if layout.root.is_dir() else None

    # ── overview ─────────────────────────────────────────────────────
    def overview(self) -> OverviewResponse:
        experiments = self._all("experiments")
        runs = self._all("runs")
        workers = self._all("workers")
        checkpoints = self._all("checkpoints")
        evaluations = self._all_evaluations("evaluations")
        qualifications = self._all_evaluations("qualifications")

        liveness = [self._liveness_for(record)[0] for record in workers]

        empty_reason: str | None = None
        if not any((experiments, runs, workers, checkpoints, evaluations)):
            empty_reason = (
                "This registry holds no experiments, runs, workers or checkpoints yet. "
                "Register a model baseline and create an experiment to begin, or point "
                "OPS_REGISTRY_ROOT at the registry you meant."
            )

        return OverviewResponse(
            generated_at=_now().isoformat(),
            registry_present=self.registry.root.is_dir(),
            experiments=_counts([str(item.get("status", "")) for item in experiments]),
            runs=_counts([str(item.get("status", "")) for item in runs]),
            workers=_counts(liveness),
            worker_classes=_counts([str(item.get("worker_class", "")) for item in workers]),
            checkpoints=_counts([str(item.get("status", "")) for item in checkpoints]),
            checkpoint_kinds=_counts([str(item.get("kind", "")) for item in checkpoints]),
            evaluations=_counts([str(item.get("status", "")) for item in evaluations]),
            qualifications=_counts([str(item.get("outcome", "")) for item in qualifications]),
            system=self._system_checks(workers, runs),
            empty_reason=empty_reason,
        )

    def _system_checks(
        self, workers: list[dict[str, Any]], runs: list[dict[str, Any]]
    ) -> list[SystemCheck]:
        checks: list[SystemCheck] = []

        registry_ok = self.registry.root.is_dir()
        checks.append(
            SystemCheck(
                name="training registry",
                status="OK" if registry_ok else "UNAVAILABLE",
                detail=(
                    "readable" if registry_ok else "the configured registry directory is absent"
                ),
            )
        )

        evaluations_dir = self.registry.root / "evaluations"
        checks.append(
            SystemCheck(
                name="evaluation registry",
                status="OK" if evaluations_dir.is_dir() else "UNKNOWN",
                detail=(
                    "readable"
                    if evaluations_dir.is_dir()
                    else "no evaluation records have been written in this registry"
                ),
            )
        )

        transport_reason = self.context.transport_unavailable_reason()
        checks.append(
            SystemCheck(
                name="remote worker transport",
                status="UNAVAILABLE" if transport_reason else "OK",
                detail=transport_reason or f"{self.context.worker_transport} transport configured",
            )
        )

        # Deliberately not "GPU READY". No capability figure this
        # project holds about NVIDIA hardware came from a machine it
        # rented, and a probe is the only thing that could change that.
        verified = [
            worker
            for worker in workers
            if str(worker.get("worker_class")) == WorkerClass.GPU_TRAINING_READY.value
        ]
        if not workers:
            checks.append(
                SystemCheck(
                    name="training capability",
                    status="UNAVAILABLE",
                    detail="no workers are registered",
                )
            )
        elif not verified:
            checks.append(
                SystemCheck(
                    name="training capability",
                    status="DEGRADED",
                    detail=(
                        f"{len(workers)} worker(s) registered, none verified as "
                        "GPU_TRAINING_READY by a probe"
                    ),
                )
            )
        else:
            fresh = [w for w in verified if self._liveness_for(w)[0] == "ONLINE"]
            checks.append(
                SystemCheck(
                    name="training capability",
                    status="OK" if fresh else "DEGRADED",
                    detail=(
                        f"{len(fresh)} of {len(verified)} probe-verified worker(s) "
                        "reporting within the liveness window"
                    ),
                )
            )

        active = [
            run
            for run in runs
            if str(run.get("status"))
            in {RunStatus.RUNNING.value, RunStatus.STARTING.value, RunStatus.QUEUED.value}
        ]
        lost = [run for run in runs if str(run.get("status")) == RunStatus.LOST.value]
        checks.append(
            SystemCheck(
                name="control plane",
                status="DEGRADED" if lost else "OK",
                detail=(
                    f"{len(active)} active run(s); {len(lost)} run(s) LOST and awaiting "
                    "reconciliation"
                    if lost
                    else f"{len(active)} active run(s)"
                ),
            )
        )
        return checks

    # ── experiments ──────────────────────────────────────────────────
    def _experiment_summary(
        self, record: dict[str, Any], runs_by_experiment: dict[str, list[dict[str, Any]]]
    ) -> ExperimentSummary:
        experiment_id = str(record.get("experiment_id", ""))
        runs = sorted(
            runs_by_experiment.get(experiment_id, []),
            key=lambda item: str(item.get("created_at", "")),
        )
        latest = runs[-1] if runs else None
        return ExperimentSummary(
            experiment_id=experiment_id,
            name=str(record.get("name", "")),
            hypothesis=str(record.get("hypothesis", "")),
            description=str(record.get("description", "")),
            base_model_id=str(record.get("base_model_id", "")),
            status=str(record.get("status", "")),
            blocked_reason=str(record.get("blocked_reason", "")),
            dataset_lock_ref=record.get("dataset_lock_ref"),
            curation_lock_ref=record.get("curation_lock_ref"),
            operator=str(record.get("operator", "")),
            tags=[str(tag) for tag in record.get("tags", []) or []],
            created_at=str(record.get("created_at", "")),
            run_count=len(runs),
            latest_run_id=str(latest.get("run_id")) if latest else None,
            latest_run_status=str(latest.get("status")) if latest else None,
        )

    def _runs_by_experiment(self) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for run in self._all("runs"):
            grouped.setdefault(str(run.get("experiment_id", "")), []).append(run)
        return grouped

    def experiments(
        self,
        *,
        status: str | None = None,
        base_model_id: str | None = None,
        tag: str | None = None,
        query: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> ExperimentListResponse:
        records = self._all("experiments")
        grouped = self._runs_by_experiment()
        summaries = [self._experiment_summary(record, grouped) for record in records]

        # Search covers the name and the id only. Not the dataset: the
        # console does not search training material, and a text box that
        # matched track titles would be one.
        needle = (query or "").strip().lower()
        filtered = [
            item
            for item in summaries
            if (status is None or item.status == status)
            and (base_model_id is None or item.base_model_id == base_model_id)
            and (tag is None or tag in item.tags)
            and (not needle or needle in item.name.lower() or needle in item.experiment_id.lower())
        ]
        filtered.sort(key=lambda item: item.created_at, reverse=True)
        window, page = _paginate(filtered, limit, offset)
        return ExperimentListResponse(
            items=window,
            page=page,
            available_statuses=sorted({item.status for item in summaries if item.status}),
            available_base_models=sorted(
                {item.base_model_id for item in summaries if item.base_model_id}
            ),
            available_tags=sorted({tag for item in summaries for tag in item.tags}),
        )

    def experiment_detail(self, experiment_id: str) -> ExperimentDetail | None:
        record = next(
            (
                item
                for item in self._all("experiments")
                if str(item.get("experiment_id")) == experiment_id
            ),
            None,
        )
        if record is None:
            return None

        grouped = self._runs_by_experiment()
        summary = self._experiment_summary(record, grouped)
        runs = [self._run_summary(run) for run in grouped.get(experiment_id, [])]
        runs.sort(key=lambda item: item.created_at, reverse=True)

        candidates = [
            CandidateSummary(
                candidate_id=str(item.get("candidate_id", "")),
                run_id=str(item.get("run_id", "")),
                checkpoint_id=str(item.get("checkpoint_id", "")),
                experiment_id=str(item.get("experiment_id", "")),
                status=str(item.get("status", "")),
                created_at=str(item.get("created_at", "")),
                notes=str(item.get("notes", "")),
            )
            for item in self._all("candidates")
            if str(item.get("experiment_id")) == experiment_id
        ]

        candidate_ids = {item.candidate_id for item in candidates}
        evaluations = [
            self._evaluation_summary(item)
            for item in self._all_evaluations("evaluations")
            if str((item.get("lineage") or {}).get("experiment_id", "")) == experiment_id
        ]
        qualifications = [
            self._qualification_summary(item)
            for item in self._all_evaluations("qualifications")
            if str(item.get("candidate_id", "")) in candidate_ids
        ]

        return ExperimentDetail(
            experiment=summary,
            base_model=self._baseline_view(summary.base_model_id),
            runs=runs,
            candidates=candidates,
            evaluations=evaluations,
            qualifications=qualifications,
            audit_events=self._audit_events(experiment_id),
        )

    # ── models ───────────────────────────────────────────────────────
    def _baseline_view(self, model_id: str) -> ModelBaselineView | None:
        record = next(
            (item for item in self._all("models") if str(item.get("model_id")) == model_id),
            None,
        )
        return self._model_view(record) if record else None

    def _model_view(self, record: dict[str, Any]) -> ModelBaselineView:
        return ModelBaselineView(
            model_id=str(record.get("model_id", "")),
            provider=str(record.get("provider", "")),
            model_family=str(record.get("model_family", "")),
            model_name=str(record.get("model_name", "")),
            model_version=str(record.get("model_version", "")),
            upstream_commit=str(record.get("upstream_commit", "")),
            architecture=str(record.get("architecture", "")),
            training_strategy_support=[
                str(item) for item in record.get("training_strategy_support", []) or []
            ],
            checkpoint_sha256=record.get("checkpoint_sha256"),
            identity_basis=str(record.get("identity_basis", "")),
            stage=str(record.get("stage", "")),
            created_at=str(record.get("created_at", "")),
        )

    def baseline(self) -> BaselineResponse:
        models = [self._model_view(record) for record in self._all("models")]
        return BaselineResponse(
            production=[model for model in models if model.stage == "PRODUCTION"],
            all_models=sorted(models, key=lambda item: item.created_at, reverse=True),
            note=(
                "Nothing in this console changes a model stage. Promotion review may approve "
                "a checkpoint for staging; activating a model in production is a runtime "
                "deployment decision made elsewhere."
            ),
        )

    # ── runs ─────────────────────────────────────────────────────────
    def _failure_view(self, record: dict[str, Any]) -> FailureView | None:
        code = record.get("error_code")
        meaning = meaning_for(str(code) if code else None)
        if meaning is None:
            return None
        message = record.get("error_message")
        return FailureView(
            code=str(code),
            headline=meaning.headline,
            guidance=meaning.guidance,
            raw_message=redact_text(str(message)) if message else None,
            confident=meaning.confident,
        )

    def _duration(self, record: dict[str, Any]) -> float | None:
        started = _parse(record.get("started_at"))
        if started is None:
            return None
        finished = (
            _parse(record.get("completed_at"))
            or _parse(record.get("failed_at"))
            or _parse(record.get("cancelled_at"))
        )
        return ((finished or _now()) - started).total_seconds()

    def _run_summary(
        self,
        record: dict[str, Any],
        *,
        experiment_names: dict[str, str] | None = None,
        worker_names: dict[str, str] | None = None,
        checkpoint_counts: dict[str, int] | None = None,
        cancel_requests: dict[str, str] | None = None,
        include_metrics: bool = False,
    ) -> RunSummary:
        run_id = str(record.get("run_id", ""))
        experiment_names = (
            experiment_names if experiment_names is not None else self._experiment_names()
        )
        worker_names = worker_names if worker_names is not None else self._worker_names()
        checkpoint_counts = (
            checkpoint_counts if checkpoint_counts is not None else self._checkpoint_counts()
        )
        cancel_requests = cancel_requests if cancel_requests is not None else {}

        latest_point = None
        latest_name = None
        if include_metrics:
            events = load_metrics(self.run_directory(record), self.worker_run_directory(run_id))
            series = build_series(events, exclude=TELEMETRY_METRICS, limit=1)
            preferred = next(
                (item for item in series if item.metric_name == "train_loss"),
                series[0] if series else None,
            )
            if preferred and preferred.points:
                latest_point = preferred.points[-1]
                latest_name = preferred.metric_name

        worker_id = record.get("worker_id")
        return RunSummary(
            run_id=run_id,
            experiment_id=str(record.get("experiment_id", "")),
            experiment_name=experiment_names.get(str(record.get("experiment_id", "")), ""),
            base_model_id=str(record.get("base_model_id", "")),
            status=str(record.get("status", "")),
            execution_backend=str(record.get("execution_backend", "")),
            worker_id=str(worker_id) if worker_id else None,
            worker_name=worker_names.get(str(worker_id)) if worker_id else None,
            created_at=str(record.get("created_at", "")),
            queued_at=record.get("queued_at"),
            started_at=record.get("started_at"),
            completed_at=record.get("completed_at"),
            failed_at=record.get("failed_at"),
            cancelled_at=record.get("cancelled_at"),
            duration_seconds=self._duration(record),
            parent_run_id=record.get("parent_run_id"),
            checkpoint_count=checkpoint_counts.get(run_id, 0),
            latest_metric=latest_point,
            latest_metric_name=latest_name,
            failure=self._failure_view(record),
            cancel_requested_at=cancel_requests.get(run_id),
        )

    def runs(
        self,
        *,
        status: str | None = None,
        experiment_id: str | None = None,
        worker_id: str | None = None,
        backend: str | None = None,
        limit: int = 50,
        offset: int = 0,
        with_metrics: bool = False,
    ) -> RunListResponse:
        records = self._all("runs")
        filtered = [
            record
            for record in records
            if (status is None or str(record.get("status")) == status)
            and (experiment_id is None or str(record.get("experiment_id")) == experiment_id)
            and (worker_id is None or str(record.get("worker_id")) == worker_id)
            and (backend is None or str(record.get("execution_backend")) == backend)
        ]
        filtered.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        window, page = _paginate(filtered, limit, offset)

        # The lookup tables are built once for the page rather than per
        # row. With a thousand runs the difference is a thousand
        # directory listings against one.
        experiment_names = self._experiment_names()
        worker_names = self._worker_names()
        checkpoint_counts = self._checkpoint_counts()
        cancel_requests = self._cancel_requests()

        return RunListResponse(
            items=[
                self._run_summary(
                    record,
                    experiment_names=experiment_names,
                    worker_names=worker_names,
                    checkpoint_counts=checkpoint_counts,
                    cancel_requests=cancel_requests,
                    # Reading every run's metrics file to fill one column
                    # would make a list page O(runs) file reads. Opt-in.
                    include_metrics=with_metrics,
                )
                for record in window
            ],
            page=page,
            available_statuses=sorted({str(item.get("status", "")) for item in records if item}),
            available_backends=sorted(
                {str(item.get("execution_backend", "")) for item in records if item}
            ),
        )

    def _timeline(self, record: dict[str, Any]) -> list[TimelineEntry]:
        status = str(record.get("status", ""))
        queued_at = record.get("queued_at")
        started_at = record.get("started_at")

        reached: dict[str, bool] = {
            RunStatus.DRAFT.value: True,
            # Validation happened if the run got past it, or is in it, or
            # failed with a code only a gate produces.
            RunStatus.VALIDATING.value: bool(queued_at)
            or status in {RunStatus.VALIDATING.value, RunStatus.QUEUED.value}
            or bool(started_at)
            or (status == RunStatus.FAILED.value and bool(record.get("error_code"))),
            RunStatus.QUEUED.value: bool(queued_at),
            # STARTING carries no stamp of its own; a run that started is
            # a run that passed through it.
            RunStatus.STARTING.value: bool(started_at) or status == RunStatus.STARTING.value,
            RunStatus.RUNNING.value: bool(started_at),
        }
        stamps: dict[str, str | None] = {
            RunStatus.DRAFT.value: record.get("created_at"),
            RunStatus.VALIDATING.value: None,
            RunStatus.QUEUED.value: queued_at,
            RunStatus.STARTING.value: None,
            RunStatus.RUNNING.value: started_at,
            RunStatus.COMPLETED.value: record.get("completed_at"),
            RunStatus.FAILED.value: record.get("failed_at"),
            RunStatus.CANCELLED.value: record.get("cancelled_at"),
            RunStatus.LOST.value: None,
        }

        entries = [
            TimelineEntry(
                state=state,
                reached=reached[state],
                at=stamps.get(state),
                current=status == state,
            )
            for state in LINEAR_STATES
        ]
        entries.extend(
            TimelineEntry(
                state=state,
                reached=status == state,
                at=stamps.get(state),
                current=status == state,
                terminal=True,
            )
            for state in TERMINAL_STATES
        )
        return entries

    def _config_view(self, payload: dict[str, Any]) -> TrainingConfigView:
        data = dict(payload)
        data["target_modules"] = [str(item) for item in data.get("target_modules", []) or []]
        known = set(TrainingConfigView.model_fields)
        return TrainingConfigView(**{key: value for key, value in data.items() if key in known})

    def _dataset_view(self, payload: dict[str, Any]) -> DatasetRefView:
        return DatasetRefView(
            dataset_id=str(payload.get("dataset_id", "")),
            dataset_lock_sha256=str(payload.get("dataset_lock_sha256", "")),
            curation_id=str(payload.get("curation_id", "")),
            curation_lock_sha256=str(payload.get("curation_lock_sha256", "")),
            curated_manifest_sha256=str(payload.get("curated_manifest_sha256", "")),
            manifest_artifact_ref=str(payload.get("manifest_artifact_ref", "")),
            sampling_weights_sha256=payload.get("sampling_weights_sha256"),
            selected_track_count=int(payload.get("selected_track_count") or 0),
            selected_hours=float(payload.get("selected_hours") or 0.0),
        )

    def _gates_for(self, run_directory: Path | None) -> tuple[list[GateView], bool, str | None]:
        """The Phase 25 gate results for a run, if any were recorded.

        Absent is a real answer: a run that has not been validated has
        no rights verdict, and showing an empty list as though every
        gate passed would be the worst possible reading of it.
        """
        payload = (
            _read_json(run_directory / OPS_ARTIFACT_DIR / GATE_REPORT_NAME)
            if run_directory is not None
            else None
        )
        if payload is None:
            # One answer for both causes. From an operator's side "the
            # directory does not exist" and "the report is not in it"
            # are the same fact — nobody has run the gates — and the
            # next move is the same either way.
            return (
                [],
                False,
                "No gate report has been recorded for this run. Validate it to produce one.",
            )
        views: list[GateView] = []
        for gate in payload.get("gates", []) or []:
            evidence = gate.get("evidence") or {}
            skipped = bool(evidence.get("skipped"))
            views.append(
                GateView(
                    name=str(gate.get("name", "")),
                    status=(
                        "PASS" if gate.get("passed") else ("NOT_EVALUATED" if skipped else "FAIL")
                    ),
                    detail=str(gate.get("detail", "")),
                    failure_code=gate.get("failure_code"),
                    offending_count=int(gate.get("offending_count") or 0),
                    offending_ids=[str(item) for item in gate.get("offending_ids", []) or []],
                )
            )
        return views, True, None

    def _control_preflight(self, run_directory: Path | None) -> PreflightView:
        if run_directory is None:
            return PreflightView(
                available=False,
                unavailable_reason="This run has no artifact directory on this machine.",
            )
        payload = _read_json(run_directory / OPS_ARTIFACT_DIR / CONTROL_PREFLIGHT_NAME)
        if payload is None:
            return PreflightView(
                available=False,
                unavailable_reason=("Control-plane preflight has not been run for this run."),
            )
        checks = [
            PreflightCheckView(
                name=name,
                status=("PASS" if value == "PASS" else "FAIL" if value == "FAIL" else "UNKNOWN"),
                detail="",
            )
            for name, value in sorted((payload.get("checks") or {}).items())
        ]
        problems = [redact_text(str(item)) for item in payload.get("problems", []) or []]
        unknown = [str(item) for item in payload.get("unknown", []) or []]
        status: Literal["PASS", "BLOCKED", "FAIL", "UNKNOWN"] = (
            "PASS" if payload.get("ok") else "FAIL"
        )
        if payload.get("ok") and unknown:
            # Unknowns do not fail a run, and they must not read as a
            # clean pass either.
            status = "BLOCKED"
        return PreflightView(
            available=True,
            status=status,
            checks=checks,
            problems=problems,
            unknown=unknown,
            generated_at=payload.get("generated_at"),
        )

    def _remote_preflight(self, worker_directory: Path | None) -> PreflightView:
        if worker_directory is None:
            return PreflightView(
                available=False,
                unavailable_reason=(
                    self.context.transport_unavailable_reason()
                    or "No worker directory for this run is visible from this machine."
                ),
            )
        payload = _read_json(worker_directory / "remote_preflight.json")
        if payload is None:
            return PreflightView(
                available=False,
                unavailable_reason="The worker has not recorded a preflight for this run.",
            )
        checks = [
            PreflightCheckView(
                name=str(check.get("name", "")),
                status=(
                    "PASS"
                    if check.get("status") == "PASS"
                    else "FAIL"
                    if check.get("status") == "FAIL"
                    else "UNKNOWN"
                ),
                detail=redact_text(str(check.get("detail", ""))),
                severity=str(check.get("severity", "REQUIRED")),
                expected=check.get("expected"),
                observed=check.get("observed"),
            )
            for check in payload.get("checks", []) or []
        ]
        raw_status = str(payload.get("status", "UNKNOWN"))
        status = raw_status if raw_status in {"PASS", "BLOCKED", "FAIL"} else "UNKNOWN"
        return PreflightView(
            available=True,
            status=status,  # type: ignore[arg-type]
            checks=checks,
            problems=[redact_text(str(item)) for item in payload.get("blocking_reasons", []) or []],
            unknown=[check.name for check in checks if check.status == "UNKNOWN"],
            generated_at=payload.get("created_at"),
        )

    def _remote_state(self, run_id: str, worker_directory: Path | None) -> RemoteStateView:
        if worker_directory is None:
            return RemoteStateView(
                available=False,
                unavailable_reason=(
                    self.context.transport_unavailable_reason()
                    or "No worker directory for this run is visible from this machine."
                ),
            )
        payload = _read_json(worker_directory / "status.json")
        if payload is None:
            return RemoteStateView(
                available=False,
                unavailable_reason="The worker holds no execution record for this run.",
            )
        process = payload.get("process") or {}
        state = str(payload.get("state", ""))
        alive: bool | None = None
        if process:
            finished = process.get("finished_at")
            alive = finished is None
        lease = _read_json(worker_directory / "lease.json") or {}
        return RemoteStateView(
            available=True,
            worker_state=state,
            implied_run_status=run_status_for(state) if state else None,
            detail=redact_text(str(payload.get("detail", ""))),
            exit_code=payload.get("exit_code"),
            failure_code=payload.get("failure_code"),
            lease_id=payload.get("lease_id"),
            process_alive=alive,
            updated_at=payload.get("updated_at"),
            started_at=payload.get("started_at"),
            completed_at=payload.get("completed_at"),
            cancel_requested_at=process.get("cancel_requested_at"),
            protocol_version=lease.get("protocol_version") or REMOTE_PROTOCOL_VERSION,
            plan_sha256=payload.get("training_plan_sha256") or None,
        )

    def _staging(self, worker_directory: Path | None) -> StagingView:
        if worker_directory is None:
            return StagingView(
                available=False,
                unavailable_reason=(
                    self.context.transport_unavailable_reason()
                    or "No worker directory for this run is visible from this machine."
                ),
            )
        payload = _read_json(worker_directory / "artifact_manifest.json")
        if payload is None:
            return StagingView(
                available=False,
                unavailable_reason="No artifact manifest has been received for this run.",
            )
        entries = payload.get("entries", []) or []
        roles: dict[str, int] = {}
        digests: set[str] = set()
        total_bytes = 0
        present = 0
        missing = 0
        for entry in entries:
            role = str(entry.get("role", "UNKNOWN"))
            roles[role] = roles.get(role, 0) + 1
            total_bytes += int(entry.get("size_bytes") or 0)
            digest = entry.get("sha256")
            if digest:
                digests.add(str(digest))
            target = entry.get("target_path")
            if target:
                if (worker_directory / str(target)).exists():
                    present += 1
                else:
                    missing += 1
        return StagingView(
            available=True,
            total_entries=len(entries),
            unique_contents=len(digests),
            total_bytes=total_bytes,
            presence_checked=True,
            present_entries=present,
            missing_entries=missing,
            roles=dict(sorted(roles.items())),
            built_at=payload.get("built_at"),
        )

    def _liveness_for(self, record: dict[str, Any]) -> tuple[str, float | None]:
        last = record.get("last_heartbeat")
        return (
            self.context.liveness.evaluate(last),
            self.context.liveness.age_seconds(last),
        )

    def _heartbeat_view(
        self, worker_record: dict[str, Any] | None, worker_directory: Path | None
    ) -> HeartbeatView:
        """Liveness from the registry, enriched by the worker's own file.

        The registry timestamp is authoritative for how long it has been
        since the control plane heard anything. The worker's heartbeat
        file adds telemetry the registry does not carry, and is used
        only when this machine can see it.
        """
        if worker_record is None:
            return HeartbeatView(
                available=False,
                unavailable_reason="This run has no worker assigned.",
            )
        liveness, age = self._liveness_for(worker_record)
        payload: dict[str, Any] = {}
        if worker_directory is not None:
            payload = _read_json(worker_directory.parent.parent / "heartbeat.json") or {}

        gpu = [
            GpuTelemetryView(
                index=int(reading.get("index", 0)),
                utilization_pct=reading.get("utilization_pct"),
                memory_used_mb=reading.get("memory_used_mb"),
                memory_total_mb=reading.get("memory_total_mb"),
                temperature_c=reading.get("temperature_c"),
                power_w=reading.get("power_w"),
                sampled_at=reading.get("sampled_at"),
            )
            for reading in payload.get("gpu", []) or []
        ]
        return HeartbeatView(
            available=bool(worker_record.get("last_heartbeat")) or bool(payload),
            unavailable_reason=(
                None
                if worker_record.get("last_heartbeat") or payload
                else "This worker has never reported."
            ),
            timestamp=worker_record.get("last_heartbeat") or payload.get("timestamp"),
            age_seconds=age,
            liveness=liveness,  # type: ignore[arg-type]
            worker_state=payload.get("worker_state"),
            active_run_id=payload.get("active_run_id"),
            health=payload.get("health"),
            uptime_seconds=payload.get("uptime_seconds"),
            free_disk_mb=payload.get("free_disk_mb"),
            gpu=gpu,
            detail=redact_text(str(payload.get("detail", ""))),
        )

    def _environment_lock(self, run_directory: Path | None) -> dict[str, Any]:
        if run_directory is None:
            return {}
        payload = _read_json(run_directory / "environment_lock.json") or {}
        cleaned = redact_document(payload)
        return cleaned if isinstance(cleaned, dict) else {}

    def _reproducibility(
        self,
        record: dict[str, Any],
        environment: dict[str, Any],
        worker_record: dict[str, Any] | None,
        baseline: ModelBaselineView | None,
    ) -> ReproducibilityView:
        dataset = record.get("dataset_ref") or {}
        config = record.get("config") or {}
        code = environment.get("code") if isinstance(environment.get("code"), dict) else {}
        software = (worker_record or {}).get("software_environment") or {}
        return ReproducibilityView(
            luber_commit=(code or {}).get("commit") or environment.get("luber_commit"),
            luber_dirty=(code or {}).get("dirty"),
            ace_step_commit=config.get("ace_step_commit") or environment.get("ace_step_commit"),
            base_model_id=str(record.get("base_model_id", "")),
            base_model_upstream_commit=baseline.upstream_commit if baseline else None,
            dataset_lock_sha256=str(dataset.get("dataset_lock_sha256", "")),
            curation_lock_sha256=str(dataset.get("curation_lock_sha256", "")),
            curated_manifest_sha256=str(dataset.get("curated_manifest_sha256", "")),
            training_config_sha256=str(record.get("config_sha256", "")),
            training_plan_sha256=record.get("training_plan_sha256"),
            environment_lock_digest=environment.get("digest"),
            worker_capability_signature=software.get("capability_signature"),
            python_version=environment.get("python_version")
            or ((worker_record or {}).get("capabilities") or {}).get("python_version"),
            torch_version=((worker_record or {}).get("capabilities") or {}).get("torch_version"),
        )

    def _cost(self, record: dict[str, Any], worker_record: dict[str, Any] | None) -> CostView:
        """Cost, assembled only from what was recorded.

        Nothing here consults a price list. Where a rate was never
        recorded the field is None and `unknown` names it, because an
        estimated spend derived from a guessed hourly rate is a number
        an operator would put in a budget.
        """
        software = (worker_record or {}).get("software_environment") or {}
        wall = self._duration(record)
        unknown: list[str] = []
        rate_raw = software.get("hourly_rate")
        rate: float | None
        try:
            rate = float(rate_raw) if rate_raw is not None else None
        except (TypeError, ValueError):
            rate = None
        if rate is None:
            unknown.append("no hourly rate has been recorded for this worker")
        provider = software.get("provider")
        if not provider:
            unknown.append("no provider has been recorded for this worker")
        estimated = (rate * wall / 3600.0) if (rate is not None and wall is not None) else None
        if estimated is None and rate is not None:
            unknown.append("the run has no measured wall time yet")
        return CostView(
            provider=str(provider) if provider else None,
            instance_type=str(software["instance_type"]) if software.get("instance_type") else None,
            hourly_rate=rate,
            currency=str(software["currency"]) if software.get("currency") else None,
            wall_seconds=wall,
            gpu_seconds=None,
            estimated_cost=estimated,
            actual_cost=None,
            unknown=unknown,
        )

    def _progress(
        self,
        record: dict[str, Any],
        events: list[Any],
        checkpoints: list[CheckpointSummary],
    ) -> RunProgress:
        config = record.get("config") or {}
        steps = [event.step for event in events if event.step is not None]
        epochs = [event.epoch for event in events if event.epoch is not None]
        latest_checkpoint = max(
            checkpoints,
            key=lambda item: (item.step or -1, item.created_at),
            default=None,
        )
        return RunProgress(
            latest_step=max(steps) if steps else None,
            latest_epoch=max(epochs) if epochs else None,
            total_epochs=int(config["epochs"]) if config.get("epochs") is not None else None,
            elapsed_seconds=self._duration(record),
            latest_train_loss=latest_value(events, "train_loss"),
            latest_learning_rate=latest_value(events, "learning_rate"),
            latest_checkpoint_id=latest_checkpoint.checkpoint_id if latest_checkpoint else None,
            # No ETA. The trainer counts epochs and not steps, nothing
            # records a step total, and an ETA derived from an assumed
            # one is the number an operator plans a day around.
            eta_seconds=None,
            eta_reason=(
                "The installed trainer measures length in epochs and records no step total, "
                "so remaining time cannot be derived from what has been measured."
            ),
        )

    def run_detail(self, run_id: str) -> RunDetail | None:
        record = next(
            (item for item in self._all("runs") if str(item.get("run_id")) == run_id), None
        )
        if record is None:
            return None

        run_directory = self.run_directory(record)
        worker_directory = self.worker_run_directory(run_id)
        worker_id = record.get("worker_id")
        worker_record = (
            next(
                (
                    item
                    for item in self._all("workers")
                    if str(item.get("worker_id")) == str(worker_id)
                ),
                None,
            )
            if worker_id
            else None
        )

        events = load_metrics(run_directory, worker_directory)
        checkpoints = [
            self._checkpoint_summary(item)
            for item in self._all("checkpoints")
            if str(item.get("run_id")) == run_id
        ]
        checkpoints.sort(key=lambda item: (item.step or 0, item.created_at))

        experiment_record = next(
            (
                item
                for item in self._all("experiments")
                if str(item.get("experiment_id")) == str(record.get("experiment_id"))
            ),
            None,
        )
        baseline = self._baseline_view(str(record.get("base_model_id", "")))
        environment = self._environment_lock(run_directory)
        gates, gates_available, gates_reason = self._gates_for(run_directory)

        checkpoint_ids = {item.checkpoint_id for item in checkpoints}
        evaluations = [
            self._evaluation_summary(item)
            for item in self._all_evaluations("evaluations")
            if str((item.get("lineage") or {}).get("checkpoint_id", "")) in checkpoint_ids
        ]

        # The latest metric is filled from the events already loaded
        # above rather than by re-reading the file: the failure panel
        # needs it, and reading the same jsonl twice to fill one field
        # is the kind of waste that makes a detail page slow.
        summary = self._run_summary(record, cancel_requests=self._cancel_requests())
        latest_series = build_series(events, exclude=TELEMETRY_METRICS, limit=1)
        preferred = next(
            (item for item in latest_series if item.metric_name == "train_loss"),
            latest_series[0] if latest_series else None,
        )
        if preferred and preferred.points:
            summary.latest_metric = preferred.points[-1]
            summary.latest_metric_name = preferred.metric_name

        return RunDetail(
            run=summary,
            experiment=(
                self._experiment_summary(experiment_record, self._runs_by_experiment())
                if experiment_record
                else None
            ),
            base_model=baseline,
            timeline=self._timeline(record),
            config=self._config_view(record.get("config") or {}),
            config_sha256=str(record.get("config_sha256", "")),
            training_plan_sha256=record.get("training_plan_sha256"),
            dataset=self._dataset_view(record.get("dataset_ref") or {}),
            dataset_available=bool((record.get("dataset_ref") or {}).get("dataset_id")),
            worker=self._worker_summary(worker_record) if worker_record else None,
            heartbeat=self._heartbeat_view(worker_record, worker_directory),
            remote=self._remote_state(run_id, worker_directory),
            staging=self._staging(worker_directory),
            control_preflight=self._control_preflight(run_directory),
            remote_preflight=self._remote_preflight(worker_directory),
            gates=gates,
            gates_available=gates_available,
            gates_unavailable_reason=gates_reason,
            metrics=build_series(events, exclude=TELEMETRY_METRICS),
            telemetry=build_series(events, names=TELEMETRY_METRICS),
            progress=self._progress(record, events, checkpoints),
            checkpoints=checkpoints,
            evaluations=evaluations,
            reproducibility=self._reproducibility(record, environment, worker_record, baseline),
            cost=self._cost(record, worker_record),
            audit_events=self._audit_events(run_id),
            actions=self.run_actions(record),
            environment_lock=environment,
        )

    def control_preflight_for(self, run_id: str) -> PreflightView:
        """The recorded control-plane preflight for a run, by id.

        Public because the action layer needs the same answer the run
        page shows. Two readings of "did preflight pass" would
        eventually disagree, and the one that mattered would be the one
        the button used.
        """
        record = next(
            (item for item in self._all("runs") if str(item.get("run_id")) == run_id), None
        )
        if record is None:
            return PreflightView(available=False, unavailable_reason="No such run.")
        return self._control_preflight(self.run_directory(record))

    def failure_diagnostics(self, run_id: str) -> list[str]:
        record = next(
            (item for item in self._all("runs") if str(item.get("run_id")) == run_id), None
        )
        if record is None:
            return []
        return tail_lines(
            self.run_directory(record), self.worker_run_directory(run_id), stream="stderr"
        )

    # ── actions ──────────────────────────────────────────────────────
    def run_actions(self, record: dict[str, Any]) -> list[ActionAvailability]:
        """What the console will let an operator do to this run, and why.

        Every entry here is recomputed by the endpoint that performs the
        action. This list exists so the UI does not have to predict the
        answer, not so the UI can decide it.
        """
        status = str(record.get("status", ""))
        run_id = str(record.get("run_id", ""))
        backend = str(record.get("execution_backend", ""))
        terminal = status in {
            RunStatus.COMPLETED.value,
            RunStatus.FAILED.value,
            RunStatus.CANCELLED.value,
        }
        transport_reason = self.context.transport_unavailable_reason()

        actions: list[ActionAvailability] = []

        validate_ok = status == RunStatus.DRAFT.value
        actions.append(
            ActionAvailability(
                action="validate",
                label="Validate",
                available=validate_ok and self.context.datasets.configured,
                reason=(
                    ""
                    if validate_ok and self.context.datasets.configured
                    else (
                        f"Only a DRAFT run can be validated; this run is {status}."
                        if not validate_ok
                        else "No dataset build root is configured, so the gates cannot be run."
                    )
                ),
                confirmation=(
                    f"Run every Phase 25 gate against run {run_id} — dataset lock, curation "
                    "lock, rights, evaluation leakage and self-generated audio — and move it "
                    "to QUEUED if all of them pass. Nothing is transferred and no trainer "
                    "starts."
                ),
            )
        )

        preflight = self._control_preflight(self.run_directory(record))
        dispatch_ok = status == RunStatus.QUEUED.value
        dispatch_reason = ""
        if not dispatch_ok:
            dispatch_reason = (
                f"Only a QUEUED run can be dispatched; this run is {status}."
                if status != RunStatus.DRAFT.value
                else "This run has not been validated. Validate it first."
            )
        elif backend == REMOTE_GPU:
            # Said before anything about preflight: this refusal is about
            # what the console is trusted with, and it does not change
            # however green the checks are.
            dispatch_ok = False
            dispatch_reason = (
                "Remote dispatch needs SSH credentials, which this console deliberately does "
                "not hold. Dispatch from the operator CLI: "
                "`luber-training remote run dispatch`."
            )
        elif backend != DRY_RUN:
            dispatch_ok = False
            dispatch_reason = f"No backend named {backend!r} can be executed from this console."
        elif not preflight.available:
            # Validation runs the gates; preflight checks the machine. A
            # run whose gates passed can still be pointed at a worker
            # that cannot execute it, and dispatching on the strength of
            # the gates alone is how that gets discovered by renting the
            # hardware.
            dispatch_ok = False
            dispatch_reason = (
                "No control-plane preflight has been recorded for this run. Validate it "
                "against an assigned worker first."
            )
        elif preflight.status == "FAIL":
            dispatch_ok = False
            dispatch_reason = "Control-plane preflight failed: " + "; ".join(preflight.problems[:2])
        actions.append(
            ActionAvailability(
                action="dispatch",
                label="Dispatch",
                available=dispatch_ok,
                reason=dispatch_reason,
                confirmation=(
                    f"Start run {run_id} on the {backend} backend. The dry-run backend trains "
                    "nothing and produces metrics marked SIMULATED and a checkpoint of kind "
                    "MOCK, which can never become an evaluation candidate."
                ),
                destructive=True,
            )
        )

        cancel_ok = not terminal and status != RunStatus.DRAFT.value
        actions.append(
            ActionAvailability(
                action="cancel",
                label="Cancel",
                available=cancel_ok,
                reason=("" if cancel_ok else f"A run that is {status} cannot be cancelled."),
                confirmation=(
                    f"Request graceful cancellation of run {run_id}. Metrics, logs and any "
                    "finished checkpoints are kept. "
                    + (
                        "The dry-run backend stops immediately."
                        if backend == DRY_RUN
                        else "The run stays as it is until a worker confirms it stopped — this "
                        "console records the request, it does not deliver the signal."
                    )
                ),
                destructive=True,
            )
        )

        reconcile_ok = transport_reason is None and status in {
            RunStatus.QUEUED.value,
            RunStatus.STARTING.value,
            RunStatus.RUNNING.value,
            RunStatus.LOST.value,
        }
        actions.append(
            ActionAvailability(
                action="reconcile",
                label="Reconcile remote state",
                available=reconcile_ok,
                reason=(
                    ""
                    if reconcile_ok
                    else (
                        transport_reason or f"A run that is {status} has nothing left to reconcile."
                    )
                ),
                confirmation=(
                    f"Ask the worker what is actually happening to run {run_id}. This changes "
                    "nothing on the worker and may be repeated safely; it reports what it "
                    "finds, including that it could not tell."
                ),
            )
        )

        # LOST is deliberately excluded. The trainer may still be
        # running, and a retry launched beside it puts two of them in one
        # checkpoint directory. The action layer refuses this too; this
        # is the same rule, stated where the operator reads it.
        retry_ok = status in {RunStatus.FAILED.value, RunStatus.CANCELLED.value}
        actions.append(
            ActionAvailability(
                action="create_retry_run",
                label="Create retry run",
                available=retry_ok,
                reason=(
                    ""
                    if retry_ok
                    else "Reconcile this run first: the trainer may still be running, and a "
                    "second one against the same checkpoint directory corrupts both."
                    if status == RunStatus.LOST.value
                    else "A retry is only offered once a run has stopped."
                ),
                confirmation=(
                    f"Create a new DRAFT run citing {run_id} as its parent, with the same "
                    "experiment, dataset reference and configuration. Run "
                    f"{run_id} is not modified — a retry is a new run, and its history stays "
                    "as it happened."
                ),
            )
        )
        return actions

    # ── workers ──────────────────────────────────────────────────────
    def _capabilities_view(self, payload: dict[str, Any]) -> WorkerCapabilitiesView:
        known = set(WorkerCapabilitiesView.model_fields)
        return WorkerCapabilitiesView(
            **{key: value for key, value in payload.items() if key in known}
        )

    def _active_runs_for(self, worker_id: str) -> list[str]:
        active = {RunStatus.STARTING.value, RunStatus.RUNNING.value}
        return [
            str(run.get("run_id"))
            for run in self._all("runs")
            if str(run.get("worker_id")) == worker_id and str(run.get("status")) in active
        ]

    def _worker_summary(self, record: dict[str, Any]) -> WorkerSummary:
        liveness, age = self._liveness_for(record)
        software = record.get("software_environment") or {}
        worker_id = str(record.get("worker_id", ""))
        return WorkerSummary(
            worker_id=worker_id,
            name=str(record.get("name", "")),
            backend_type=str(record.get("backend_type", "")),
            host_identity=str(record.get("host_identity", "")),
            worker_class=str(record.get("worker_class", "")),
            remote_classification=software.get("remote_classification"),
            status=str(record.get("status", "")),
            liveness=liveness,  # type: ignore[arg-type]
            last_heartbeat=record.get("last_heartbeat"),
            heartbeat_age_seconds=age,
            max_concurrent_runs=int(record.get("max_concurrent_runs") or 1),
            active_run_ids=self._active_runs_for(worker_id),
            capabilities=self._capabilities_view(record.get("capabilities") or {}),
            protocol_version=software.get("remote_protocol_version"),
            capability_signature=software.get("capability_signature"),
            created_at=str(record.get("created_at", "")),
            # The *name* of a key reference is not sent. An operator
            # needs to know a credential is configured; what the
            # deployment calls it is not information a browser needs.
            has_credentials=bool(record.get("ssh_key_ref") or record.get("credential_ref")),
        )

    def workers(
        self,
        *,
        worker_class: str | None = None,
        liveness: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> WorkerListResponse:
        summaries = [self._worker_summary(record) for record in self._all("workers")]
        filtered = [
            item
            for item in summaries
            if (worker_class is None or item.worker_class == worker_class)
            and (liveness is None or item.liveness == liveness)
        ]
        filtered.sort(key=lambda item: item.name or item.worker_id)
        window, page = _paginate(filtered, limit, offset)
        return WorkerListResponse(
            items=window,
            page=page,
            available_classes=sorted(
                {item.worker_class for item in summaries if item.worker_class}
            ),
            available_liveness=sorted({item.liveness for item in summaries}),
        )

    def worker_detail(self, worker_id: str) -> WorkerDetail | None:
        record = next(
            (item for item in self._all("workers") if str(item.get("worker_id")) == worker_id),
            None,
        )
        if record is None:
            return None
        summary = self._worker_summary(record)
        capabilities = record.get("capabilities") or {}
        unknown = sorted(
            key
            for key, value in capabilities.items()
            if value is None and key not in {"reported_at"}
        )
        runs = [
            self._run_summary(run)
            for run in self._all("runs")
            if str(run.get("worker_id")) == worker_id
        ]
        runs.sort(key=lambda item: item.created_at, reverse=True)

        worker_directory = None
        for run in runs[:1]:
            worker_directory = self.worker_run_directory(run.run_id)

        software = record.get("software_environment") or {}
        cleaned = redact_document({str(k): str(v) for k, v in software.items()})
        return WorkerDetail(
            worker=summary,
            heartbeat=self._heartbeat_view(record, worker_directory),
            software_environment=cleaned if isinstance(cleaned, dict) else {},
            recent_runs=runs[:20],
            audit_events=self._audit_events(worker_id),
            unknown_capabilities=unknown,
        )

    def worker_compatibility(
        self, *, execution_backend: str, resume_checkpoint_id: str | None = None
    ) -> list[WorkerCompatibility]:
        """Which workers may be chosen, with the reasons they may not.

        Every reason is derived from a reported fact. Where a capability
        was never measured the worker is reported unverified rather than
        assumed adequate or assumed inadequate — Step 51 is explicit
        that an unknown VRAM figure is not a requirement to guess at.
        """
        results: list[WorkerCompatibility] = []
        for record in self._all("workers"):
            summary = self._worker_summary(record)
            reasons: list[str] = []

            if execution_backend == REMOTE_GPU:
                if summary.worker_class != WorkerClass.GPU_TRAINING_READY.value:
                    reasons.append(
                        f"class is {summary.worker_class}; only a worker a probe verified as "
                        "GPU_TRAINING_READY may take a remote training run"
                    )
                if summary.capabilities.cuda_available is None:
                    reasons.append("CUDA availability has never been measured on this machine")
                elif not summary.capabilities.cuda_available:
                    reasons.append("torch reports no CUDA device on this machine")
                if summary.protocol_version and summary.protocol_version != REMOTE_PROTOCOL_VERSION:
                    reasons.append(
                        f"speaks {summary.protocol_version}; this build speaks "
                        f"{REMOTE_PROTOCOL_VERSION}"
                    )
                if summary.capabilities.vram_total_mb is None:
                    reasons.append(
                        "VRAM has never been measured, so nothing can confirm this plan fits"
                    )

            if summary.backend_type != execution_backend:
                reasons.append(
                    f"registered for the {summary.backend_type} backend, not {execution_backend}"
                )
            if len(summary.active_run_ids) >= summary.max_concurrent_runs:
                reasons.append(
                    f"already running {len(summary.active_run_ids)} of "
                    f"{summary.max_concurrent_runs} permitted run(s)"
                )
            if summary.liveness in {"OFFLINE", "UNKNOWN"}:
                reasons.append(f"liveness is {summary.liveness}")

            results.append(
                WorkerCompatibility(worker=summary, compatible=not reasons, reasons=reasons)
            )
        results.sort(key=lambda item: (not item.compatible, item.worker.name))
        return results

    # ── checkpoints ──────────────────────────────────────────────────
    def _candidate_by_checkpoint(self) -> dict[str, str]:
        return {
            str(item.get("checkpoint_id")): str(item.get("candidate_id"))
            for item in self._all("candidates")
        }

    def _checkpoint_summary(self, record: dict[str, Any]) -> CheckpointSummary:
        kind = str(record.get("kind", ""))
        status = str(record.get("status", ""))
        is_real = kind in {CheckpointKind.ADAPTER.value, CheckpointKind.FULL_MODEL.value}

        blocked: str | None = None
        if kind == CheckpointKind.MOCK.value:
            blocked = (
                "This is a MOCK artifact from a dry run. It contains no trained weights and "
                "can never become an evaluation candidate."
            )
        elif status != CheckpointStatus.READY.value:
            blocked = f"Only a READY checkpoint may be evaluated; this one is {status}."

        reference = record.get("reference")
        scheme: str | None = None
        present: bool | None = None
        if reference:
            text = str(reference)
            if "://" in text:
                scheme = text.split("://", 1)[0]
            else:
                scheme = "file"
                present = Path(text).exists()

        run_id = str(record.get("run_id", ""))
        experiment_id = next(
            (
                str(run.get("experiment_id", ""))
                for run in self._all("runs")
                if str(run.get("run_id")) == run_id
            ),
            "",
        )
        return CheckpointSummary(
            checkpoint_id=str(record.get("checkpoint_id", "")),
            run_id=run_id,
            experiment_id=experiment_id,
            kind=kind,
            is_real_model=is_real,
            status=status,
            step=record.get("step"),
            epoch=record.get("epoch"),
            size_bytes=record.get("size_bytes"),
            sha256=record.get("sha256"),
            checkpoint_format=str(record.get("checkpoint_format", "")),
            created_at=str(record.get("created_at", "")),
            finalized_at=record.get("finalized_at"),
            metrics_snapshot={
                str(key): float(value)
                for key, value in (record.get("metrics_snapshot") or {}).items()
            },
            can_evaluate=blocked is None,
            evaluate_blocked_reason=blocked,
            candidate_id=self._candidate_by_checkpoint().get(str(record.get("checkpoint_id"))),
            location_scheme=scheme,
            location_present=present,
        )

    def checkpoints(
        self,
        *,
        status: str | None = None,
        kind: str | None = None,
        run_id: str | None = None,
        experiment_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> CheckpointListResponse:
        records = self._all("checkpoints")
        summaries = [self._checkpoint_summary(record) for record in records]
        filtered = [
            item
            for item in summaries
            if (status is None or item.status == status)
            and (kind is None or item.kind == kind)
            and (run_id is None or item.run_id == run_id)
            and (experiment_id is None or item.experiment_id == experiment_id)
        ]
        filtered.sort(key=lambda item: item.created_at, reverse=True)
        window, page = _paginate(filtered, limit, offset)
        return CheckpointListResponse(
            items=window,
            page=page,
            available_statuses=sorted({item.status for item in summaries if item.status}),
            available_kinds=sorted({item.kind for item in summaries if item.kind}),
        )

    def checkpoint_detail(self, checkpoint_id: str) -> CheckpointDetail | None:
        record = next(
            (
                item
                for item in self._all("checkpoints")
                if str(item.get("checkpoint_id")) == checkpoint_id
            ),
            None,
        )
        if record is None:
            return None
        summary = self._checkpoint_summary(record)
        run_record = next(
            (item for item in self._all("runs") if str(item.get("run_id")) == summary.run_id),
            None,
        )
        experiment_record = next(
            (
                item
                for item in self._all("experiments")
                if str(item.get("experiment_id")) == summary.experiment_id
            ),
            None,
        )
        evaluations = [
            self._evaluation_summary(item)
            for item in self._all_evaluations("evaluations")
            if str((item.get("lineage") or {}).get("checkpoint_id", "")) == checkpoint_id
        ]
        qualifications = [
            self._qualification_summary(item)
            for item in self._all_evaluations("qualifications")
            if str(item.get("candidate_id", "")) == (summary.candidate_id or "\x00")
        ]
        return CheckpointDetail(
            checkpoint=summary,
            run=self._run_summary(run_record) if run_record else None,
            experiment=(
                self._experiment_summary(experiment_record, self._runs_by_experiment())
                if experiment_record
                else None
            ),
            evaluations=evaluations,
            qualifications=qualifications,
            audit_events=self._audit_events(checkpoint_id),
        )

    def compare_checkpoints(self, checkpoint_ids: list[str]) -> CheckpointComparisonResponse:
        rows: list[CheckpointComparisonRow] = []
        metric_names: set[str] = set()
        for checkpoint_id in checkpoint_ids:
            detail = self.checkpoint_detail(checkpoint_id)
            if detail is None:
                continue
            evaluation = detail.evaluations[0] if detail.evaluations else None
            qualification = detail.qualifications[0] if detail.qualifications else None
            metrics = self._evaluation_metrics(evaluation.evaluation_id) if evaluation else {}
            metric_names.update(metrics)
            rows.append(
                CheckpointComparisonRow(
                    checkpoint=detail.checkpoint,
                    evaluation=evaluation,
                    qualification=qualification,
                    metrics=metrics,
                    training_context=detail.checkpoint.metrics_snapshot,
                )
            )
        return CheckpointComparisonResponse(
            rows=rows,
            metric_names=sorted(metric_names),
            note=(
                "Training loss appears only as context. A lower training loss is not evidence "
                "that a model is better, and reading it beside evaluation metrics invites "
                "exactly that conclusion."
            ),
        )

    # ── evaluations ──────────────────────────────────────────────────
    def _qualification_by_evaluation(self) -> dict[str, dict[str, Any]]:
        return {
            str(item.get("evaluation_id")): item for item in self._all_evaluations("qualifications")
        }

    def _evaluation_summary(self, record: dict[str, Any]) -> EvaluationSummary:
        lineage = record.get("lineage") or {}
        cost = record.get("cost") or {}
        qualification = self._qualification_by_evaluation().get(str(record.get("evaluation_id")))
        return EvaluationSummary(
            evaluation_id=str(record.get("evaluation_id", "")),
            status=str(record.get("status", "")),
            mode=str(record.get("mode", "")),
            suite_id=str(record.get("suite_id", "")),
            suite_version=str(record.get("suite_version", "")),
            suite_digest=str(record.get("suite_digest", "")),
            policy_digest=str(record.get("policy_digest", "")),
            candidate_id=str(lineage.get("candidate_id", "")),
            checkpoint_id=str(lineage.get("checkpoint_id", "")),
            run_id=str(lineage.get("run_id", "")),
            experiment_id=str(lineage.get("experiment_id", "")),
            baseline_label=str((record.get("baseline") or {}).get("label", "")),
            candidate_label=str((record.get("candidate") or {}).get("label", "")),
            experiment_hypothesis=str(record.get("experiment_hypothesis", "")),
            seeds=[int(seed) for seed in record.get("seeds", []) or []],
            started_at=record.get("started_at"),
            completed_at=record.get("completed_at"),
            failed_at=record.get("failed_at"),
            cancelled_at=record.get("cancelled_at"),
            error=redact_text(str(record["error"])) if record.get("error") else None,
            qualification_outcome=(str(qualification.get("outcome")) if qualification else None),
            wall_seconds=cost.get("wall_seconds"),
            gpu_seconds=cost.get("gpu_seconds"),
        )

    def _qualification_summary(self, record: dict[str, Any]) -> QualificationSummary:
        return QualificationSummary(
            evaluation_id=str(record.get("evaluation_id", "")),
            candidate_id=str(record.get("candidate_id", "")),
            outcome=str(record.get("outcome", "")),
            policy_id=str(record.get("policy_id", "")),
            policy_version=str(record.get("policy_version", "")),
            policy_digest=str(record.get("policy_digest", "")),
            reasons=[str(item) for item in record.get("reasons", []) or []],
            passed_gates=[str(item) for item in record.get("passed_gates", []) or []],
            failed_gates=[str(item) for item in record.get("failed_gates", []) or []],
            inconclusive_gates=[str(item) for item in record.get("inconclusive_gates", []) or []],
            gate_outcomes=[
                GateOutcomeView(
                    name=str(gate.get("name", "")),
                    passed=bool(gate.get("passed")),
                    detail=str(gate.get("detail", "")),
                    severity=str(gate.get("severity", "NONE")),
                    inconclusive=bool(gate.get("inconclusive")),
                )
                for gate in record.get("gate_outcomes", []) or []
            ],
            hypothesis_status=str(record.get("hypothesis_status", "")),
            human_review_required_for=[
                str(item) for item in record.get("human_review_required_for", []) or []
            ],
            decided_at=str(record.get("decided_at", "")),
        )

    def evaluation_artifacts(self, evaluation_id: str) -> Path | None:
        try:
            directory = self.evaluations_registry.artifacts(evaluation_id).directory
        except Exception:
            return None
        return directory if directory.is_dir() else None

    def _evaluation_metrics(self, evaluation_id: str) -> dict[str, float]:
        directory = self.evaluation_artifacts(evaluation_id)
        if directory is None:
            return {}
        payload = _read_json(directory / "comparisons.json")
        if payload is None:
            return {}
        metrics: dict[str, float] = {}
        for item in payload.get("comparisons", []) or []:
            value = item.get("candidate_value")
            if isinstance(value, int | float):
                metrics[str(item.get("metric_name", ""))] = float(value)
        return metrics

    def _comparisons(self, evaluation_id: str) -> list[ComparisonView]:
        directory = self.evaluation_artifacts(evaluation_id)
        if directory is None:
            return []
        payload = _read_json(directory / "comparisons.json")
        if payload is None:
            return []
        return [
            ComparisonView(
                metric_name=str(item.get("metric_name", "")),
                verdict=str(item.get("verdict", "")),
                baseline_value=item.get("baseline_value"),
                candidate_value=item.get("candidate_value"),
                delta=item.get("delta"),
                severity=str(item.get("severity", "NONE")),
                detail=str(item.get("detail", "")),
            )
            for item in payload.get("comparisons", []) or []
        ]

    def evaluations(
        self,
        *,
        status: str | None = None,
        outcome: str | None = None,
        experiment_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> EvaluationListResponse:
        records = self._all_evaluations("evaluations")
        summaries = [self._evaluation_summary(record) for record in records]
        filtered = [
            item
            for item in summaries
            if (status is None or item.status == status)
            and (outcome is None or item.qualification_outcome == outcome)
            and (experiment_id is None or item.experiment_id == experiment_id)
        ]
        filtered.sort(key=lambda item: item.started_at or "", reverse=True)
        window, page = _paginate(filtered, limit, offset)
        return EvaluationListResponse(
            items=window,
            page=page,
            available_statuses=sorted({item.status for item in summaries if item.status}),
            available_outcomes=sorted(
                {item.qualification_outcome for item in summaries if item.qualification_outcome}
            ),
        )

    def evaluation_detail(self, evaluation_id: str) -> EvaluationDetail | None:
        record = next(
            (
                item
                for item in self._all_evaluations("evaluations")
                if str(item.get("evaluation_id")) == evaluation_id
            ),
            None,
        )
        if record is None:
            return None
        summary = self._evaluation_summary(record)
        qualification_record = self._qualification_by_evaluation().get(evaluation_id)
        qualification = (
            self._qualification_summary(qualification_record) if qualification_record else None
        )
        comparisons = self._comparisons(evaluation_id)

        checkpoint = None
        if summary.checkpoint_id:
            detail = self.checkpoint_detail(summary.checkpoint_id)
            checkpoint = detail.checkpoint if detail else None

        run_record = next(
            (item for item in self._all("runs") if str(item.get("run_id")) == summary.run_id),
            None,
        )
        experiment_record = next(
            (
                item
                for item in self._all("experiments")
                if str(item.get("experiment_id")) == summary.experiment_id
            ),
            None,
        )

        human: HumanReviewView | None = None
        if qualification and qualification.outcome == "HUMAN_REVIEW_REQUIRED":
            directory = self.evaluation_artifacts(evaluation_id)
            request = _read_json(directory / "human_review" / "request.json") if directory else None
            human = HumanReviewView(
                required=True,
                mode=str((request or {}).get("mode", "LIGHT_AB")),
                reason="; ".join(qualification.reasons[:3]),
                case_count=len((request or {}).get("case_ids", []) or []),
                dimensions=qualification.human_review_required_for,
                status=str((request or {}).get("status", "PENDING")),
                package_available=bool(request),
            )

        directory = self.evaluation_artifacts(evaluation_id)
        return EvaluationDetail(
            evaluation=summary,
            lineage={
                str(key): str(value)
                for key, value in (record.get("lineage") or {}).items()
                if value is not None
            },
            qualification=qualification,
            comparisons=comparisons,
            regressions=[
                item
                for item in comparisons
                if item.verdict == "REGRESSED" or item.severity not in {"NONE", "INFO"}
            ],
            promotion_reviews=[
                PromotionReviewView(
                    review_id=str(item.get("review_id", "")),
                    candidate_id=str(item.get("candidate_id", "")),
                    evaluation_id=str(item.get("evaluation_id", "")),
                    qualification_outcome=str(item.get("qualification_outcome", "")),
                    decision=str(item.get("decision", "")),
                    decided_by=str(item.get("decided_by", "")),
                    rationale=str(item.get("rationale", "")),
                    decided_at=str(item.get("decided_at", "")),
                )
                for item in self._all_evaluations("promotion_reviews")
                if str(item.get("evaluation_id")) == evaluation_id
            ],
            human_review=human,
            checkpoint=checkpoint,
            run=self._run_summary(run_record) if run_record else None,
            experiment=(
                self._experiment_summary(experiment_record, self._runs_by_experiment())
                if experiment_record
                else None
            ),
            audit_events=self._audit_events(evaluation_id),
            report_available=bool(directory and (directory / "report.md").is_file()),
        )

    # ── logs, audit, catalogue ───────────────────────────────────────
    def logs(
        self,
        run_id: str,
        *,
        stream: str = "stdout",
        offset: int | None = None,
        limit: int | None = None,
    ) -> LogView:
        record = next(
            (item for item in self._all("runs") if str(item.get("run_id")) == run_id), None
        )
        if record is None:
            return LogView(available=False, unavailable_reason="No such run.")
        from luber_api.ops.streams import DEFAULT_LOG_BYTES

        return read_stream(
            self.run_directory(record),
            self.worker_run_directory(run_id),
            stream=stream,
            offset=offset,
            limit=min(limit or DEFAULT_LOG_BYTES, DEFAULT_LOG_BYTES),
        )

    def _audit_events(self, entity_id: str) -> list[AuditEvent]:
        events = self.registry.audit_events(entity_id)
        return [
            AuditEvent(
                timestamp=str(event.get("timestamp", "")),
                event=str(event.get("event", "")),
                entity_id=str(event.get("entity_id", "")),
                entity_kind=str(event.get("entity_kind", "")),
                metadata=(
                    redact_document(event.get("metadata") or {})
                    if isinstance(event.get("metadata"), dict)
                    else {}
                ),
            )
            for event in sorted(events, key=lambda item: str(item.get("timestamp", "")))
        ]

    def _build_option(self, build_id: str, lock: dict[str, Any], *, curation: bool) -> BuildOption:
        if curation:
            return BuildOption(
                build_id=build_id,
                identity=str(lock.get("curation_id", "")),
                lock_sha256=str(lock.get("curation_lock_sha256") or lock.get("lock_sha256") or ""),
                manifest_sha256=str(lock.get("curated_manifest_sha256", "")),
                track_count=(
                    int(lock["selected_track_count"])
                    if lock.get("selected_track_count") is not None
                    else None
                ),
                hours=(
                    float(lock["selected_hours"])
                    if lock.get("selected_hours") is not None
                    else None
                ),
                created_at=lock.get("created_at") or lock.get("built_at"),
                source_dataset_lock_sha256=lock.get("source_dataset_lock_sha256"),
            )
        return BuildOption(
            build_id=build_id,
            identity=str(lock.get("dataset_id", "")),
            lock_sha256=str(lock.get("lock_sha256") or lock.get("dataset_lock_sha256") or ""),
            manifest_sha256=str(lock.get("manifest_sha256", "")),
            track_count=(int(lock["track_count"]) if lock.get("track_count") is not None else None),
            hours=float(lock["total_hours"]) if lock.get("total_hours") is not None else None,
            created_at=lock.get("created_at") or lock.get("built_at"),
        )

    def catalogue(self) -> CatalogueResponse:
        presets: list[PresetOption] = []
        for name in sorted(PRESETS):
            config: TrainingConfig = preset(name)
            presets.append(
                PresetOption(
                    name=name,
                    intent=PRESET_INTENT.get(name, ""),
                    config=self._config_view(config.to_dict()),
                )
            )
        return CatalogueResponse(
            datasets=[
                self._build_option(entry.build_id, entry.lock, curation=False)
                for entry in self.context.datasets.entries()
            ],
            curations=[
                self._build_option(entry.build_id, entry.lock, curation=True)
                for entry in self.context.curations.entries()
            ],
            dataset_problems=self.context.datasets.problems(),
            curation_problems=self.context.curations.problems(),
            presets=presets,
            backends=[DRY_RUN, REMOTE_GPU],
            base_models=[self._model_view(record) for record in self._all("models")],
        )

    # ── helpers used by the action layer ─────────────────────────────
    def worker_entity(self, worker_id: str) -> TrainingWorker | None:
        try:
            return self.orchestrator.get_worker(worker_id)
        except Exception:
            return None


__all__ = [
    "CONTROL_PREFLIGHT_NAME",
    "GATE_REPORT_NAME",
    "OPS_ARTIFACT_DIR",
    "RUN_CANCEL_REQUESTED",
    "RUN_RECONCILED",
    "RUN_VALIDATION_REQUESTED",
    "OpsReadModel",
]
