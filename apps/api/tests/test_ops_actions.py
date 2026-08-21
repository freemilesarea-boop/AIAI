"""What the console changes, and every case where it refuses to.

The interesting assertions are the refusals. A console that can start
training is easy; one that cannot be talked into starting the same run
twice, cannot cancel a finished run, cannot evaluate a placeholder, and
cannot be persuaded by a stale browser tab that a rights failure was a
misunderstanding, is the one worth having.

Every action here is driven through HTTP rather than by calling the
action layer directly, because the disabled button and the state check
have to disagree in the same test for the check to be worth anything.
"""

from __future__ import annotations

import asyncio

from httpx import AsyncClient
from ops_fixtures import Scenario

BASE = "/v1/ops/training"


def _action(payload: dict, name: str) -> dict:
    return next(item for item in payload["actions"] if item["action"] == name)


# ── experiments ──────────────────────────────────────────────────────


async def test_create_experiment_records_a_hypothesis_and_starts_nothing(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    response = await ops_client.post(
        f"{BASE}/experiments",
        json={
            "name": "Ballad warmth",
            "hypothesis": "Weighting ballads improves sustained vowel warmth.",
            "base_model_id": scenario.model_id,
            "description": "",
            "operator": "operator",
            "tags": ["ballad"],
        },
    )
    assert response.status_code == 201
    created = response.json()
    assert created["status"] == "DRAFT"
    assert created["run_count"] == 0

    runs = (await ops_client.get(f"{BASE}/runs")).json()
    assert runs["page"]["total"] == len(scenario.run_ids)  # unchanged


async def test_an_experiment_on_an_unknown_model_is_refused(
    ops_client: AsyncClient,
) -> None:
    response = await ops_client.post(
        f"{BASE}/experiments",
        json={
            "name": "Nowhere",
            "hypothesis": "Something.",
            "base_model_id": "model_does_not_exist",
        },
    )
    assert response.status_code == 409
    assert "base model" in response.json()["detail"]


async def test_experiment_fields_are_validated(ops_client: AsyncClient, scenario: Scenario) -> None:
    response = await ops_client.post(
        f"{BASE}/experiments",
        json={"name": "", "hypothesis": "x", "base_model_id": scenario.model_id},
    )
    assert response.status_code == 422


# ── run creation ─────────────────────────────────────────────────────


async def test_create_run_derives_every_digest_from_the_locks(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    """A caller selects builds; it does not state hashes.

    Every digest on the run is read from the file it describes, which is
    the only version of this that a gate can later disagree with
    usefully.
    """
    response = await ops_client.post(
        f"{BASE}/runs",
        json={
            "experiment_id": scenario.experiment_id,
            "dataset_build_id": scenario.dataset_build_id,
            "curation_build_id": scenario.curation_build_id,
            "preset": "SMOKE",
            "execution_backend": "dry-run",
            "worker_id": scenario.worker_ids["mac"],
        },
    )
    assert response.status_code == 201
    detail = response.json()
    assert detail["run"]["status"] == "DRAFT"
    assert detail["dataset"]["dataset_id"] == "ds-test-001"
    assert len(detail["dataset"]["curated_manifest_sha256"]) == 64
    assert detail["config"]["rank"] == 4


async def test_run_creation_never_accepts_a_path(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    """An identifier that is not in the catalogue is refused, not resolved."""
    for attempt in ("../../etc", "/etc/passwd", "not-a-build"):
        response = await ops_client.post(
            f"{BASE}/runs",
            json={
                "experiment_id": scenario.experiment_id,
                "dataset_build_id": attempt,
                "curation_build_id": scenario.curation_build_id,
                "preset": "SMOKE",
                "execution_backend": "dry-run",
            },
        )
        assert response.status_code == 409, attempt
        detail = response.json()["detail"]
        # Refused by name, not resolved and then found to be missing:
        # the identifier never becomes a path at all.
        assert "unsafe build identifier" in detail or "is not a build" in detail


async def test_an_incompatible_worker_is_refused_with_its_reason(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    response = await ops_client.post(
        f"{BASE}/runs",
        json={
            "experiment_id": scenario.experiment_id,
            "dataset_build_id": scenario.dataset_build_id,
            "curation_build_id": scenario.curation_build_id,
            "preset": "SMOKE",
            "execution_backend": "remote-gpu",
            # The Mac: DEVELOPMENT_ONLY, and nothing has measured a GPU.
            "worker_id": scenario.worker_ids["mac"],
        },
    )
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "DEVELOPMENT_ONLY" in detail or "CUDA" in detail


async def test_an_unknown_preset_is_refused(ops_client: AsyncClient, scenario: Scenario) -> None:
    response = await ops_client.post(
        f"{BASE}/runs",
        json={
            "experiment_id": scenario.experiment_id,
            "dataset_build_id": scenario.dataset_build_id,
            "curation_build_id": scenario.curation_build_id,
            "preset": "MAKE_IT_GOOD",
            "execution_backend": "dry-run",
        },
    )
    assert response.status_code == 409
    assert "unknown preset" in response.json()["detail"]


# ── validation ───────────────────────────────────────────────────────


async def test_validation_runs_the_real_gates_and_records_them(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    run_id = scenario.run_ids["draft"]
    response = await ops_client.post(f"{BASE}/runs/{run_id}/actions/validate")
    assert response.status_code == 200
    result = response.json()
    assert result["outcome"] == "PASSED"
    assert result["run_status"] == "QUEUED"

    detail = (await ops_client.get(f"{BASE}/runs/{run_id}")).json()
    assert detail["gates_available"] is True
    names = {gate["name"] for gate in detail["gates"]}
    assert {"dataset_lock", "curation_lock", "rights", "evaluation_leakage"} <= names
    assert all(gate["status"] == "PASS" for gate in detail["gates"])
    assert detail["training_plan_sha256"]
    assert detail["control_preflight"]["available"] is True


async def test_validating_a_run_that_is_not_a_draft_is_refused(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    response = await ops_client.post(
        f"{BASE}/runs/{scenario.run_ids['completed']}/actions/validate"
    )
    assert response.status_code == 409
    assert "not DRAFT" in response.json()["detail"]


async def test_validation_is_recorded_in_the_audit_log(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    run_id = scenario.run_ids["draft"]
    await ops_client.post(f"{BASE}/runs/{run_id}/actions/validate")
    events = [
        event["event"]
        for event in (await ops_client.get(f"{BASE}/runs/{run_id}")).json()["audit_events"]
    ]
    assert "RUN_VALIDATION_REQUESTED" in events
    assert "RUN_VALIDATED" in events
    assert "RUN_QUEUED" in events


# ── dispatch ─────────────────────────────────────────────────────────


async def test_an_unvalidated_run_cannot_be_dispatched(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    run_id = scenario.run_ids["draft"]
    detail = (await ops_client.get(f"{BASE}/runs/{run_id}")).json()
    assert _action(detail, "dispatch")["available"] is False

    # And the server says the same thing to a caller who ignores that.
    response = await ops_client.post(f"{BASE}/runs/{run_id}/actions/dispatch")
    assert response.status_code == 409
    assert "Validate it first" in response.json()["detail"]


async def test_remote_dispatch_is_refused_and_says_where_it_happens(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    """The console holds no SSH credentials, and does not pretend to."""
    run_id = scenario.run_ids["queued_remote"]
    detail = (await ops_client.get(f"{BASE}/runs/{run_id}")).json()
    dispatch = _action(detail, "dispatch")
    assert dispatch["available"] is False
    assert "SSH credentials" in dispatch["reason"]

    response = await ops_client.post(f"{BASE}/runs/{run_id}/actions/dispatch")
    assert response.status_code == 409
    assert "luber-training remote run dispatch" in response.json()["detail"]


async def test_dispatch_is_refused_when_preflight_failed(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    """Gates are about the data; preflight is about the machine.

    A run whose gates passed can still be pointed at a worker that has
    never demonstrated CUDA, and dispatching on the gates alone is how
    that gets discovered by renting the hardware.
    """
    # Create a dry-run run against the development-only Mac, validate it
    # — the gates pass, the machine cannot execute the plan.
    created = await ops_client.post(
        f"{BASE}/runs",
        json={
            "experiment_id": scenario.experiment_id,
            "dataset_build_id": scenario.dataset_build_id,
            "curation_build_id": scenario.curation_build_id,
            "preset": "SMOKE",
            "execution_backend": "dry-run",
            "worker_id": scenario.worker_ids["mac"],
        },
    )
    run_id = created.json()["run"]["run_id"]

    validated = await ops_client.post(f"{BASE}/runs/{run_id}/actions/validate")
    assert validated.json()["outcome"] == "PASSED"

    detail = (await ops_client.get(f"{BASE}/runs/{run_id}")).json()
    assert detail["control_preflight"]["status"] == "FAIL"
    assert _action(detail, "dispatch")["available"] is False

    response = await ops_client.post(f"{BASE}/runs/{run_id}/actions/dispatch")
    assert response.status_code == 409
    detail_text = response.json()["detail"]
    assert "preflight failed" in detail_text
    assert "DEVELOPMENT_ONLY" in detail_text or "CUDA" in detail_text


async def test_a_dry_run_dispatch_completes_and_labels_its_metrics(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    run_id = scenario.run_ids["draft"]
    await ops_client.post(f"{BASE}/runs/{run_id}/actions/validate")

    response = await ops_client.post(f"{BASE}/runs/{run_id}/actions/dispatch")
    assert response.status_code == 200
    result = response.json()
    assert result["performed"] is True
    assert result["run_status"] == "COMPLETED"
    assert "Nothing was trained" in result["detail"]

    detail = (await ops_client.get(f"{BASE}/runs/{run_id}")).json()
    assert detail["checkpoints"] == []  # a dry run produces none
    for series in detail["metrics"]:
        assert series["sources"] == ["SIMULATED"]


async def test_dispatching_twice_starts_one_thing(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    """Idempotent by state rather than by a flag.

    The second call finds the run past QUEUED, which is not a special
    case to handle but a transition that does not exist.
    """
    run_id = scenario.run_ids["draft"]
    await ops_client.post(f"{BASE}/runs/{run_id}/actions/validate")

    responses = await asyncio.gather(
        ops_client.post(f"{BASE}/runs/{run_id}/actions/dispatch"),
        ops_client.post(f"{BASE}/runs/{run_id}/actions/dispatch"),
    )

    # Exactly one of them did something. The other either found the run
    # already past QUEUED (200, performed=False) or found it finished
    # (409) — both are the state machine answering, not a flag.
    performed = [
        response
        for response in responses
        if response.status_code == 200 and response.json()["performed"]
    ]
    assert len(performed) == 1, [(response.status_code, response.text) for response in responses]

    detail = (await ops_client.get(f"{BASE}/runs/{run_id}")).json()
    assert detail["run"]["status"] == "COMPLETED"
    started = [event for event in detail["audit_events"] if event["event"] == "RUN_STARTED"]
    assert len(started) == 1


# ── cancel ───────────────────────────────────────────────────────────


async def test_cancelling_a_finished_run_is_refused(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    for label in ("completed", "cancelled", "oom"):
        response = await ops_client.post(f"{BASE}/runs/{scenario.run_ids[label]}/actions/cancel")
        assert response.status_code == 409, label
        assert "already" in response.json()["detail"]


async def test_cancelling_a_draft_is_refused(ops_client: AsyncClient, scenario: Scenario) -> None:
    response = await ops_client.post(f"{BASE}/runs/{scenario.run_ids['draft']}/actions/cancel")
    assert response.status_code == 409
    assert "nothing to cancel" in response.json()["detail"]


async def test_cancelling_a_remote_run_records_a_request_and_says_so(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    """The console must not report a GPU released that it cannot release."""
    run_id = scenario.run_ids["running"]
    response = await ops_client.post(f"{BASE}/runs/{run_id}/actions/cancel")
    assert response.status_code == 200
    result = response.json()
    assert result["performed"] is False
    assert result["outcome"] == "CANCEL_REQUESTED"
    assert result["run_status"] == "RUNNING"
    assert "no transport" in result["detail"]

    detail = (await ops_client.get(f"{BASE}/runs/{run_id}")).json()
    assert detail["run"]["status"] == "RUNNING"
    assert detail["run"]["cancel_requested_at"] is not None
    assert any(event["event"] == "RUN_CANCEL_REQUESTED" for event in detail["audit_events"])


async def test_cancelling_a_dry_run_actually_cancels_it(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    run_id = scenario.run_ids["draft"]
    await ops_client.post(f"{BASE}/runs/{run_id}/actions/validate")

    response = await ops_client.post(f"{BASE}/runs/{run_id}/actions/cancel")
    assert response.status_code == 200
    result = response.json()
    assert result["performed"] is True
    assert result["run_status"] == "CANCELLED"

    detail = (await ops_client.get(f"{BASE}/runs/{run_id}")).json()
    assert detail["run"]["status"] == "CANCELLED"
    assert detail["run"]["failure"]["code"] == "CANCELLED_BY_OPERATOR"


async def test_cancelling_a_lost_run_asks_for_reconciliation_first(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    response = await ops_client.post(
        f"{BASE}/runs/{scenario.run_ids['worker_lost']}/actions/cancel"
    )
    assert response.status_code == 409
    assert "Reconcile it first" in response.json()["detail"]


# ── reconcile ────────────────────────────────────────────────────────


async def test_reconcile_without_a_transport_says_why(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    response = await ops_client.post(
        f"{BASE}/runs/{scenario.run_ids['worker_lost']}/actions/reconcile"
    )
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "no transport" in detail
    assert "operator CLI" in detail


async def test_reconcile_against_a_real_worker_reports_what_it_finds(
    ops_client: AsyncClient, ops_app, scenario: Scenario, tmp_path
) -> None:
    """The real Phase 27 reconciliation, over a local worker root.

    A worker that has never heard of this run answers NOT_PRESENT, and
    the console reports exactly that rather than deciding the run must
    therefore have failed.
    """
    from luber_training.remote.paths import RemoteRoots
    from luber_training.remote.worker import RemoteWorker

    worker_root = tmp_path / "luber-worker"
    RemoteWorker(worker_root).initialise(
        worker_name="test-worker", roots=RemoteRoots.under(worker_root)
    )

    ops_app.state.ops_context.settings.ops_worker_transport = "local"
    ops_app.state.ops_context.settings.ops_worker_root = str(worker_root)

    response = await ops_client.post(
        f"{BASE}/runs/{scenario.run_ids['worker_lost']}/actions/reconcile"
    )
    assert response.status_code == 200
    result = response.json()
    assert result["outcome"] == "NOT_PRESENT"
    # LOST is not rewritten into something tidier on the strength of a
    # worker that has no record of the run.
    assert result["run_status"] == "LOST"

    detail = (await ops_client.get(f"{BASE}/runs/{scenario.run_ids['worker_lost']}")).json()
    assert any(event["event"] == "RUN_RECONCILED" for event in detail["audit_events"])


# ── retry ────────────────────────────────────────────────────────────


async def test_a_retry_is_a_new_run_that_cites_its_parent(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    parent = scenario.run_ids["oom"]
    response = await ops_client.post(f"{BASE}/runs/{parent}/actions/retry")
    assert response.status_code == 200
    result = response.json()
    assert result["created_id"]
    assert result["run_status"] == "DRAFT"

    child = (await ops_client.get(f"{BASE}/runs/{result['created_id']}")).json()
    assert child["run"]["parent_run_id"] == parent
    assert child["run"]["experiment_id"] == scenario.experiment_id

    # The original is untouched: history is not edited to look clean.
    original = (await ops_client.get(f"{BASE}/runs/{parent}")).json()
    assert original["run"]["status"] == "FAILED"
    assert original["run"]["failure"]["code"] == "OOM"


async def test_retrying_a_running_run_is_refused(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    response = await ops_client.post(f"{BASE}/runs/{scenario.run_ids['running']}/actions/retry")
    assert response.status_code == 409
    assert "has stopped" in response.json()["detail"]


async def test_a_lost_run_does_not_offer_a_retry_button(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    """The button and the endpoint agree.

    An offered retry that the server refuses is worse than no button:
    the operator learns the rule by being stopped rather than by
    reading it.
    """
    detail = (await ops_client.get(f"{BASE}/runs/{scenario.run_ids['worker_lost']}")).json()
    retry = _action(detail, "create_retry_run")
    assert retry["available"] is False
    assert "Reconcile this run first" in retry["reason"]


async def test_retrying_a_lost_run_demands_reconciliation(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    """The case that corrupts checkpoints if it is allowed.

    Two trainers writing into one checkpoint directory produce artifacts
    that are individually well-formed and jointly worthless, so a retry
    of a LOST run is refused until somebody has established what the
    first one is doing.
    """
    response = await ops_client.post(f"{BASE}/runs/{scenario.run_ids['worker_lost']}/actions/retry")
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "Reconcile it" in detail
    assert "a second trainer against one checkpoint directory" in detail


# ── things that are simply not offered ───────────────────────────────


async def test_there_is_no_endpoint_that_forces_a_blocked_run(
    ops_client: AsyncClient, ops_app
) -> None:
    paths = [path for path in ops_app.openapi()["paths"] if path.startswith("/v1/ops/")]
    assert not [
        path
        for path in paths
        if any(word in path.lower() for word in ("force", "override", "skip", "bypass"))
    ]


async def test_there_is_no_endpoint_that_promotes_a_model(ops_client: AsyncClient, ops_app) -> None:
    """Phase 28 displays a promotion review and cannot make one true.

    No route activates a model, changes a stage, or touches the
    inference runtime.
    """
    paths = [path for path in ops_app.openapi()["paths"] if path.startswith("/v1/ops/")]
    assert not [
        path
        for path in paths
        if any(word in path.lower() for word in ("promote", "activate", "deploy", "stage"))
    ]


async def test_the_only_mutating_verbs_are_the_expected_ones(ops_app) -> None:
    document = ops_app.openapi()["paths"]
    mutating = {
        (path, method.upper())
        for path, operations in document.items()
        if path.startswith("/v1/ops/")
        for method in operations
        if method.upper() not in {"GET", "HEAD", "OPTIONS"}
    }
    assert mutating == {
        ("/v1/ops/training/experiments", "POST"),
        ("/v1/ops/training/runs", "POST"),
        ("/v1/ops/training/runs/{run_id}/actions/validate", "POST"),
        ("/v1/ops/training/runs/{run_id}/actions/dispatch", "POST"),
        ("/v1/ops/training/runs/{run_id}/actions/cancel", "POST"),
        ("/v1/ops/training/runs/{run_id}/actions/reconcile", "POST"),
        ("/v1/ops/training/runs/{run_id}/actions/retry", "POST"),
        # A comparison reads; it is a POST only because a list of
        # checkpoint ids does not belong in a URL.
        ("/v1/ops/training/checkpoints/compare", "POST"),
    }
