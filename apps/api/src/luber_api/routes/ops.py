"""The operator training console's HTTP surface.

One router, one gate. Every route below sits behind
:func:`require_operator` and :func:`enforce_operator_origin`, applied at
the router rather than per route, so an endpoint added later is
protected by having been added rather than by somebody remembering to
protect it.

Reads are `GET` and change nothing. Actions are `POST` under
``/actions/`` and re-validate the world before touching it — the UI's
disabled button is a courtesy, and this is the control.

Errors are given three shapes deliberately. A request for something
absent is 404. A request that was well formed but is not legal in the
current state is 409, because "you cannot cancel a run that already
finished" is the world's answer and not a fault in the request. A
configuration the deployment has not supplied — no dataset root, no
worker transport — is 409 as well, with the reason, because the operator
can fix it and a 500 would suggest the console had broken.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from luber_api.ops.actions import ActionRefused, OpsActions
from luber_api.ops.context import OpsConfigurationError, OpsContext, get_ops_context
from luber_api.ops.readmodel import OpsReadModel
from luber_api.ops.schemas import (
    ActionResponse,
    BaselineResponse,
    CatalogueResponse,
    CheckpointComparisonResponse,
    CheckpointDetail,
    CheckpointListResponse,
    ComparisonRequest,
    EvaluationDetail,
    EvaluationListResponse,
    ExperimentCreateRequest,
    ExperimentDetail,
    ExperimentListResponse,
    ExperimentSummary,
    LogView,
    OverviewResponse,
    RunCreateRequest,
    RunDetail,
    RunListResponse,
    WorkerCompatibility,
    WorkerDetail,
    WorkerListResponse,
)
from luber_api.ops.security import enforce_operator_origin, require_operator
from luber_api.settings import ApiSettings, get_settings

router = APIRouter(
    prefix="/v1/ops/training",
    tags=["operator-training-console"],
    dependencies=[Depends(require_operator), Depends(enforce_operator_origin)],
)


def _read_model(context: Annotated[OpsContext, Depends(get_ops_context)]) -> OpsReadModel:
    return OpsReadModel(context)


def _actions(context: Annotated[OpsContext, Depends(get_ops_context)]) -> OpsActions:
    return OpsActions(context)


ReadModel = Annotated[OpsReadModel, Depends(_read_model)]
Actions = Annotated[OpsActions, Depends(_actions)]
Settings = Annotated[ApiSettings, Depends(get_settings)]


def _limit(requested: int, settings: ApiSettings) -> int:
    """Clamp a caller's page size to what the deployment allows.

    A ceiling rather than a validation error: a caller asking for ten
    thousand rows gets the maximum and a `page` block that tells them
    there are more, which is more useful than a 422.
    """
    return max(1, min(requested, settings.ops_page_size_limit))


def _not_found(kind: str, entity_id: str) -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, f"No {kind} {entity_id} is registered.")


# ── overview ─────────────────────────────────────────────────────────


@router.get("/overview", response_model=OverviewResponse)
def overview(read: ReadModel) -> OverviewResponse:
    return read.overview()


@router.get("/baseline", response_model=BaselineResponse)
def baseline(read: ReadModel) -> BaselineResponse:
    return read.baseline()


@router.get("/catalogue", response_model=CatalogueResponse)
def catalogue(read: ReadModel) -> CatalogueResponse:
    """What a run may be built from on this deployment.

    Builds are named by identifier, never by path. An operator selects
    from what the configured roots contain; there is no field anywhere
    in this API that accepts a directory.
    """
    return read.catalogue()


# ── experiments ──────────────────────────────────────────────────────


@router.get("/experiments", response_model=ExperimentListResponse)
def list_experiments(
    read: ReadModel,
    settings: Settings,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    base_model_id: str | None = None,
    tag: str | None = None,
    q: Annotated[str | None, Query(max_length=200)] = None,
    limit: int = 50,
    offset: int = 0,
) -> ExperimentListResponse:
    return read.experiments(
        status=status_filter,
        base_model_id=base_model_id,
        tag=tag,
        query=q,
        limit=_limit(limit, settings),
        offset=max(0, offset),
    )


@router.post("/experiments", status_code=status.HTTP_201_CREATED, response_model=ExperimentSummary)
def create_experiment(
    payload: ExperimentCreateRequest, actions: Actions, read: ReadModel
) -> ExperimentSummary:
    try:
        experiment = actions.create_experiment(
            name=payload.name,
            hypothesis=payload.hypothesis,
            base_model_id=payload.base_model_id,
            description=payload.description,
            operator=payload.operator,
            tags=payload.tags,
        )
    except ActionRefused as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    detail = read.experiment_detail(experiment.experiment_id)
    if detail is None:  # pragma: no cover - written and immediately read back
        raise _not_found("experiment", experiment.experiment_id)
    return detail.experiment


@router.get("/experiments/{experiment_id}", response_model=ExperimentDetail)
def experiment_detail(experiment_id: str, read: ReadModel) -> ExperimentDetail:
    detail = read.experiment_detail(experiment_id)
    if detail is None:
        raise _not_found("experiment", experiment_id)
    return detail


# ── runs ─────────────────────────────────────────────────────────────


@router.get("/runs", response_model=RunListResponse)
def list_runs(
    read: ReadModel,
    settings: Settings,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    experiment_id: str | None = None,
    worker_id: str | None = None,
    backend: str | None = None,
    limit: int = 50,
    offset: int = 0,
    with_metrics: bool = False,
) -> RunListResponse:
    """Runs, filtered and paginated on the server.

    ``with_metrics`` is opt-in because filling the latest-metric column
    means opening one metrics file per row. At a thousand runs that is
    the difference between a directory listing and a thousand reads, and
    a list page must not become the most expensive thing the console
    does.
    """
    return read.runs(
        status=status_filter,
        experiment_id=experiment_id,
        worker_id=worker_id,
        backend=backend,
        limit=_limit(limit, settings),
        offset=max(0, offset),
        with_metrics=with_metrics,
    )


@router.post("/runs", status_code=status.HTTP_201_CREATED, response_model=RunDetail)
def create_run(payload: RunCreateRequest, actions: Actions, read: ReadModel) -> RunDetail:
    try:
        run = actions.create_run(
            experiment_id=payload.experiment_id,
            dataset_build_id=payload.dataset_build_id,
            curation_build_id=payload.curation_build_id,
            preset_name=payload.preset,
            execution_backend=payload.execution_backend,
            worker_id=payload.worker_id,
            parent_run_id=payload.parent_run_id,
            resume_from_checkpoint_id=payload.resume_from_checkpoint_id,
        )
    except ActionRefused as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except OpsConfigurationError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    detail = read.run_detail(run.run_id)
    if detail is None:  # pragma: no cover - written and immediately read back
        raise _not_found("run", run.run_id)
    return detail


@router.get("/runs/{run_id}", response_model=RunDetail)
def run_detail(run_id: str, read: ReadModel) -> RunDetail:
    detail = read.run_detail(run_id)
    if detail is None:
        raise _not_found("run", run_id)
    return detail


@router.get("/runs/{run_id}/logs", response_model=LogView)
def run_logs(
    run_id: str,
    read: ReadModel,
    stream: Literal["stdout", "stderr"] = "stdout",
    offset: int | None = None,
    limit: int | None = None,
) -> LogView:
    """One incremental read of one stream, redacted server-side.

    Omit ``offset`` for the first read and the response starts at the
    tail of a large file. Send back ``next_offset`` to receive only what
    has arrived since — the same cursor Phase 27 uses between the
    control plane and a worker.
    """
    return read.logs(run_id, stream=stream, offset=offset, limit=limit)


@router.get("/runs/{run_id}/diagnostics", response_model=list[str])
def run_diagnostics(run_id: str, read: ReadModel) -> list[str]:
    """The last lines of stderr, for the failure panel.

    Step 46: an operator should not have to open a log viewer to find
    out why a run failed.
    """
    return read.failure_diagnostics(run_id)


@router.get("/runs/{run_id}/bundle")
def run_bundle(run_id: str, read: ReadModel, response: Response) -> dict[str, object]:
    """The Phase 25 reproducibility bundle for one run.

    References and digests, not copies. Served through this endpoint so
    the browser never reads a registry file, and redacted on the way out
    like everything else.
    """
    from luber_api.ops.redaction import redact_document

    try:
        bundle = read.orchestrator.run_bundle(run_id)
    except Exception as exc:
        raise _not_found("run", run_id) from exc
    response.headers["Content-Disposition"] = f'attachment; filename="{run_id}-bundle.json"'
    cleaned = redact_document(bundle)
    return cleaned if isinstance(cleaned, dict) else {}


def _perform(call: Callable[[], ActionResponse]) -> ActionResponse:
    """Turn a refusal into a 409, and let everything else surface.

    409 rather than 400: the request was well formed and the world said
    no. An operator reading "cannot cancel a run that already finished"
    has learned something about the run, not about their request.
    """
    try:
        return call()
    except ActionRefused as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except OpsConfigurationError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


@router.post("/runs/{run_id}/actions/validate", response_model=ActionResponse)
def validate_run(run_id: str, actions: Actions) -> ActionResponse:
    return _perform(lambda: actions.validate_run(run_id))


@router.post("/runs/{run_id}/actions/dispatch", response_model=ActionResponse)
def dispatch_run(run_id: str, actions: Actions) -> ActionResponse:
    return _perform(lambda: actions.dispatch_run(run_id))


@router.post("/runs/{run_id}/actions/cancel", response_model=ActionResponse)
def cancel_run(run_id: str, actions: Actions) -> ActionResponse:
    return _perform(lambda: actions.cancel_run(run_id))


@router.post("/runs/{run_id}/actions/reconcile", response_model=ActionResponse)
def reconcile_run(run_id: str, actions: Actions) -> ActionResponse:
    return _perform(lambda: actions.reconcile_run(run_id))


@router.post("/runs/{run_id}/actions/retry", response_model=ActionResponse)
def create_retry_run(run_id: str, actions: Actions) -> ActionResponse:
    return _perform(lambda: actions.create_retry_run(run_id))


# ── workers ──────────────────────────────────────────────────────────


@router.get("/workers", response_model=WorkerListResponse)
def list_workers(
    read: ReadModel,
    settings: Settings,
    worker_class: str | None = None,
    liveness: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> WorkerListResponse:
    return read.workers(
        worker_class=worker_class,
        liveness=liveness,
        limit=_limit(limit, settings),
        offset=max(0, offset),
    )


@router.get("/workers/compatibility", response_model=list[WorkerCompatibility])
def worker_compatibility(
    read: ReadModel, execution_backend: str = "remote-gpu"
) -> list[WorkerCompatibility]:
    """Which workers may take a run on this backend, and why not.

    Ahead of the run-create form rather than inside it, so the reasons
    are computed from the same reported capabilities the API will check
    again when the run is created.
    """
    return read.worker_compatibility(execution_backend=execution_backend)


@router.get("/workers/{worker_id}", response_model=WorkerDetail)
def worker_detail(worker_id: str, read: ReadModel) -> WorkerDetail:
    detail = read.worker_detail(worker_id)
    if detail is None:
        raise _not_found("worker", worker_id)
    return detail


# ── checkpoints ──────────────────────────────────────────────────────


@router.get("/checkpoints", response_model=CheckpointListResponse)
def list_checkpoints(
    read: ReadModel,
    settings: Settings,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    kind: str | None = None,
    run_id: str | None = None,
    experiment_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> CheckpointListResponse:
    return read.checkpoints(
        status=status_filter,
        kind=kind,
        run_id=run_id,
        experiment_id=experiment_id,
        limit=_limit(limit, settings),
        offset=max(0, offset),
    )


@router.post("/checkpoints/compare", response_model=CheckpointComparisonResponse)
def compare_checkpoints(
    payload: ComparisonRequest, read: ReadModel
) -> CheckpointComparisonResponse:
    """Place checkpoints side by side. Reads only; changes nothing."""
    return read.compare_checkpoints(payload.checkpoint_ids)


@router.get("/checkpoints/{checkpoint_id}", response_model=CheckpointDetail)
def checkpoint_detail(checkpoint_id: str, read: ReadModel) -> CheckpointDetail:
    detail = read.checkpoint_detail(checkpoint_id)
    if detail is None:
        raise _not_found("checkpoint", checkpoint_id)
    return detail


# ── evaluations ──────────────────────────────────────────────────────


@router.get("/evaluations", response_model=EvaluationListResponse)
def list_evaluations(
    read: ReadModel,
    settings: Settings,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    outcome: str | None = None,
    experiment_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> EvaluationListResponse:
    return read.evaluations(
        status=status_filter,
        outcome=outcome,
        experiment_id=experiment_id,
        limit=_limit(limit, settings),
        offset=max(0, offset),
    )


@router.get("/evaluations/{evaluation_id}", response_model=EvaluationDetail)
def evaluation_detail(evaluation_id: str, read: ReadModel) -> EvaluationDetail:
    detail = read.evaluation_detail(evaluation_id)
    if detail is None:
        raise _not_found("evaluation", evaluation_id)
    return detail


@router.get("/evaluations/{evaluation_id}/report", response_class=Response)
def evaluation_report(evaluation_id: str, read: ReadModel) -> Response:
    """The Phase 26 markdown report, if one was written."""
    from luber_api.ops.redaction import redact_text

    directory = read.evaluation_artifacts(evaluation_id)
    path = (directory / "report.md") if directory else None
    if path is None or not path.is_file():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"No report has been written for {evaluation_id}."
        )
    return Response(
        content=redact_text(path.read_text(encoding="utf-8")),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{evaluation_id}-report.md"'},
    )
