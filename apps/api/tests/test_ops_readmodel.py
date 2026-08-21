"""What the console shows, and the things it must never show.

The assertions here are mostly about restraint. It is easy to write a
dashboard that fills every field; the work is making sure that a figure
nobody measured arrives as UNKNOWN, that a dry run's numbers stay
labelled, that a worker the registry calls ONLINE is reported STALE when
it has not spoken, and that a credential in a trainer log does not reach
a browser.
"""

from __future__ import annotations

from httpx import AsyncClient
from ops_fixtures import Scenario

BASE = "/v1/ops/training"


# ── overview ─────────────────────────────────────────────────────────


async def test_overview_counts_come_from_the_registry(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    payload = (await ops_client.get(f"{BASE}/overview")).json()

    assert payload["runs"]["total"] == len(scenario.run_ids)
    assert payload["runs"]["by_state"]["RUNNING"] == 1
    assert payload["runs"]["by_state"]["LOST"] == 1
    assert payload["experiments"]["by_state"]["BLOCKED"] == 1
    assert payload["checkpoint_kinds"]["by_state"]["MOCK"] == 1
    assert payload["qualifications"]["by_state"]["HUMAN_REVIEW_REQUIRED"] == 1
    assert payload["empty_reason"] is None


async def test_overview_never_claims_gpu_ready(ops_client: AsyncClient) -> None:
    """A capability check that reports on evidence, not on hope.

    The scenario has probe-verified workers and one reporting inside the
    liveness window, so this is OK — but the *wording* must stay about
    what a probe established, never "GPU READY".
    """
    payload = (await ops_client.get(f"{BASE}/overview")).json()
    capability = next(
        check for check in payload["system"] if check["name"] == "training capability"
    )
    assert capability["status"] in {"OK", "DEGRADED"}
    assert "probe" in capability["detail"]
    assert "GPU READY" not in capability["detail"].upper()


async def test_overview_reports_the_missing_transport(ops_client: AsyncClient) -> None:
    transport = next(
        check
        for check in (await ops_client.get(f"{BASE}/overview")).json()["system"]
        if check["name"] == "remote worker transport"
    )
    assert transport["status"] == "UNAVAILABLE"
    assert "operator CLI" in transport["detail"]


async def test_an_empty_registry_says_which_kind_of_empty(
    ops_client: AsyncClient, ops_app, tmp_path
) -> None:
    """Nothing has happened, or we are pointed at the wrong directory.

    An empty dashboard that cannot distinguish those two is the one an
    operator stares at while their real registry sits untouched.
    """
    from luber_api.ops.context import BuildCatalogue, OpsContext
    from luber_api.settings import get_settings
    from luber_evaluation.registry import EvaluationRegistry
    from luber_training.orchestrator import Orchestrator
    from luber_training.registry import Registry
    from luber_training.remote.identity import LivenessPolicy

    registry = Registry(tmp_path / "empty")
    ops_app.state.ops_context = OpsContext(
        settings=get_settings(),
        registry=registry,
        evaluations=EvaluationRegistry(registry),
        orchestrator=Orchestrator(registry),
        datasets=BuildCatalogue(root=None, lock_name="dataset_lock.json"),
        curations=BuildCatalogue(root=None, lock_name="curation_lock.json"),
        liveness=LivenessPolicy(),
    )
    payload = (await ops_client.get(f"{BASE}/overview")).json()
    assert payload["runs"]["total"] == 0
    assert "OPS_REGISTRY_ROOT" in payload["empty_reason"]


# ── experiments ──────────────────────────────────────────────────────


async def test_experiment_list_filters_and_searches(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    everything = (await ops_client.get(f"{BASE}/experiments")).json()
    assert everything["page"]["total"] == 2
    assert "korean" in everything["available_tags"]

    blocked = (await ops_client.get(f"{BASE}/experiments?status=BLOCKED")).json()
    assert [item["experiment_id"] for item in blocked["items"]] == [scenario.blocked_experiment_id]
    assert blocked["items"][0]["blocked_reason"]

    found = (await ops_client.get(f"{BASE}/experiments?q=korean")).json()
    assert [item["experiment_id"] for item in found["items"]] == [scenario.experiment_id]


async def test_experiment_detail_carries_its_lineage(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    payload = (await ops_client.get(f"{BASE}/experiments/{scenario.experiment_id}")).json()
    assert payload["experiment"]["run_count"] == len(scenario.run_ids)
    assert payload["base_model"]["upstream_commit"]
    assert len(payload["candidates"]) == 1
    assert {item["qualification_outcome"] for item in payload["evaluations"]} == {
        "QUALIFIED",
        "REJECTED",
        "HUMAN_REVIEW_REQUIRED",
    }
    assert any(event["event"] == "EXPERIMENT_CREATED" for event in payload["audit_events"])


async def test_unknown_ids_are_404(ops_client: AsyncClient) -> None:
    for path in ("experiments/exp_missing", "runs/run_missing", "workers/wrk_missing"):
        assert (await ops_client.get(f"{BASE}/{path}")).status_code == 404


# ── runs ─────────────────────────────────────────────────────────────


async def test_run_list_filters_server_side(ops_client: AsyncClient, scenario: Scenario) -> None:
    failed = (await ops_client.get(f"{BASE}/runs?status=FAILED")).json()
    assert {item["run_id"] for item in failed["items"]} == {
        scenario.run_ids["oom"],
        scenario.run_ids["rights_blocked"],
    }

    dry = (await ops_client.get(f"{BASE}/runs?backend=dry-run")).json()
    assert all(item["execution_backend"] == "dry-run" for item in dry["items"])

    by_worker = (
        await ops_client.get(f"{BASE}/runs?worker_id={scenario.worker_ids['stale']}")
    ).json()
    assert [item["run_id"] for item in by_worker["items"]] == [scenario.run_ids["worker_lost"]]


async def test_run_detail_keeps_local_and_remote_apart(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    """The registry's status and the worker's state are separate fields.

    With no transport the remote block is explicitly unavailable rather
    than absent or, worse, inferred from the run status.
    """
    payload = (await ops_client.get(f"{BASE}/runs/{scenario.run_ids['running']}")).json()
    assert payload["run"]["status"] == "RUNNING"
    assert payload["remote"]["available"] is False
    assert payload["remote"]["worker_state"] is None
    assert "operator CLI" in payload["remote"]["unavailable_reason"]


async def test_run_detail_shows_hashes_for_reproducibility(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    payload = (await ops_client.get(f"{BASE}/runs/{scenario.run_ids['running']}")).json()
    repro = payload["reproducibility"]
    assert repro["training_config_sha256"] == payload["config_sha256"]
    assert repro["dataset_lock_sha256"]
    assert repro["curation_lock_sha256"]
    assert repro["luber_commit"] == "a6b4a7fafdd99f12e78fcda1d9096a6ac5bf0374"
    assert repro["ace_step_commit"]
    assert payload["config"]["rank"] == 4  # the SMOKE preset, as recorded


async def test_run_timeline_uses_the_real_state_machine(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    payload = (await ops_client.get(f"{BASE}/runs/{scenario.run_ids['running']}")).json()
    states = {entry["state"]: entry for entry in payload["timeline"]}

    assert states["DRAFT"]["reached"] and states["RUNNING"]["reached"]
    assert states["RUNNING"]["current"] is True
    assert states["COMPLETED"]["reached"] is False
    assert states["LOST"]["terminal"] is True

    draft = (await ops_client.get(f"{BASE}/runs/{scenario.run_ids['draft']}")).json()
    draft_states = {entry["state"]: entry for entry in draft["timeline"]}
    assert draft_states["QUEUED"]["reached"] is False
    assert draft_states["VALIDATING"]["reached"] is False


async def test_metrics_keep_their_source(ops_client: AsyncClient, scenario: Scenario) -> None:
    """A dry run's numbers stay marked, and telemetry stays separate."""
    simulated = (await ops_client.get(f"{BASE}/runs/{scenario.run_ids['completed']}")).json()
    loss = next(item for item in simulated["metrics"] if item["metric_name"] == "train_loss")
    assert loss["sources"] == ["SIMULATED"]

    real = (await ops_client.get(f"{BASE}/runs/{scenario.run_ids['running']}")).json()
    real_loss = next(item for item in real["metrics"] if item["metric_name"] == "train_loss")
    assert real_loss["sources"] == ["TRAINER"]
    assert real_loss["total_points"] == 40
    assert real_loss["sampled"] is False

    telemetry = {item["metric_name"] for item in real["telemetry"]}
    assert telemetry == {"gpu_memory_mb"}
    assert "gpu_memory_mb" not in {item["metric_name"] for item in real["metrics"]}


async def test_progress_refuses_to_invent_an_eta(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    progress = (await ops_client.get(f"{BASE}/runs/{scenario.run_ids['running']}")).json()[
        "progress"
    ]
    assert progress["latest_step"] == 40
    assert progress["latest_train_loss"] is not None
    assert progress["eta_seconds"] is None
    assert "epochs" in progress["eta_reason"]


async def test_gates_absent_is_not_gates_passed(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    """A run nobody validated has no rights verdict, and says so."""
    draft = (await ops_client.get(f"{BASE}/runs/{scenario.run_ids['draft']}")).json()
    assert draft["gates_available"] is False
    assert draft["gates"] == []
    assert "Validate it" in draft["gates_unavailable_reason"]


async def test_rights_failure_is_visible_with_no_override(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    payload = (await ops_client.get(f"{BASE}/runs/{scenario.run_ids['rights_blocked']}")).json()
    rights = next(gate for gate in payload["gates"] if gate["name"] == "rights")
    assert rights["status"] == "FAIL"
    assert rights["offending_count"] == 2
    assert payload["run"]["failure"]["code"] == "RIGHTS_GATE_FAILED"
    assert "no override" in payload["run"]["failure"]["guidance"]

    # There is no action anywhere on this run that would run it anyway.
    assert all(
        action["action"] != "force" and "force" not in action["label"].lower()
        for action in payload["actions"]
    )


async def test_leakage_gate_is_shown(ops_client: AsyncClient, scenario: Scenario) -> None:
    payload = (await ops_client.get(f"{BASE}/runs/{scenario.run_ids['running']}")).json()
    leakage = next(gate for gate in payload["gates"] if gate["name"] == "evaluation_leakage")
    assert leakage["status"] == "PASS"


async def test_preflight_unknown_is_not_a_pass(ops_client: AsyncClient, scenario: Scenario) -> None:
    payload = (await ops_client.get(f"{BASE}/runs/{scenario.run_ids['running']}")).json()
    control = payload["control_preflight"]
    assert control["available"] is True
    # Everything required passed, but a disk figure nobody measured
    # keeps it out of a clean PASS.
    assert control["status"] == "BLOCKED"
    assert control["unknown"]
    assert any(check["status"] == "UNKNOWN" for check in control["checks"])


async def test_oom_is_claimed_only_where_it_was_established(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    payload = (await ops_client.get(f"{BASE}/runs/{scenario.run_ids['oom']}")).json()
    failure = payload["run"]["failure"]
    assert failure["code"] == "OOM"
    assert failure["headline"] == "CUDA out of memory"
    assert failure["confident"] is True
    assert "batch size" in failure["guidance"]
    # And it does not offer to change the configuration by itself.
    assert "automatically" not in failure["guidance"]


async def test_worker_lost_never_reads_as_training_failed(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    """The most important sentence in the console.

    An operator told "training failed" retries; a retry against a
    trainer that is still running puts two of them in one checkpoint
    directory.
    """
    payload = (await ops_client.get(f"{BASE}/runs/{scenario.run_ids['worker_lost']}")).json()
    assert payload["run"]["status"] == "LOST"
    failure = payload["run"]["failure"]
    assert failure["headline"] == "Worker connection lost"
    assert failure["confident"] is False
    assert "Reconcile" in failure["guidance"]
    assert "failed" not in failure["headline"].lower()

    # No retry is offered either: launching one beside a trainer that may
    # still be running is the mistake this whole state exists to prevent.
    retry = next(item for item in payload["actions"] if item["action"] == "create_retry_run")
    assert retry["available"] is False
    assert "Reconcile this run first" in retry["reason"]
    reconcile = next(item for item in payload["actions"] if item["action"] == "reconcile")
    assert reconcile["available"] is False  # no transport in this deployment


async def test_failure_diagnostics_surface_without_opening_a_log(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    lines = (await ops_client.get(f"{BASE}/runs/{scenario.run_ids['oom']}/diagnostics")).json()
    assert any("out of memory" in line.lower() for line in lines)


# ── logs ─────────────────────────────────────────────────────────────


async def test_logs_are_incremental(ops_client: AsyncClient, scenario: Scenario) -> None:
    run_id = scenario.run_ids["running"]
    first = (await ops_client.get(f"{BASE}/runs/{run_id}/logs")).json()
    assert first["available"] is True
    assert first["text"]
    assert first["eof"] is True

    # Continuing from the cursor returns nothing new, rather than the
    # whole file again.
    again = (
        await ops_client.get(f"{BASE}/runs/{run_id}/logs?offset={first['next_offset']}")
    ).json()
    assert again["text"] == ""
    assert again["next_offset"] == first["next_offset"]


async def test_logs_are_redacted_server_side(ops_client: AsyncClient, scenario: Scenario) -> None:
    """The secret never reaches the browser, so the browser cannot leak it."""
    payload = (await ops_client.get(f"{BASE}/runs/{scenario.run_ids['running']}/logs")).json()
    text = payload["text"]
    assert "hf_liveTokenValueThatMustNotLeak0000" not in text
    assert "sk-live-abcdefghijklmnop" not in text
    assert "«redacted»" in text
    # And the surrounding diagnostic survives.
    assert "loading dataset manifest" in text
    assert "step 1 loss 2.38" in text


async def test_stderr_is_a_separate_stream(ops_client: AsyncClient, scenario: Scenario) -> None:
    payload = (
        await ops_client.get(f"{BASE}/runs/{scenario.run_ids['oom']}/logs?stream=stderr")
    ).json()
    assert payload["stream"] == "stderr"
    assert "out of memory" in payload["text"].lower()


async def test_a_run_with_no_log_says_so(ops_client: AsyncClient, scenario: Scenario) -> None:
    payload = (await ops_client.get(f"{BASE}/runs/{scenario.run_ids['draft']}/logs")).json()
    assert payload["available"] is False
    assert payload["unavailable_reason"]


# ── workers ──────────────────────────────────────────────────────────


async def test_worker_liveness_is_not_registry_status(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    """A record saying ONLINE proves only that somebody wrote ONLINE."""
    items = {
        item["worker_id"]: item
        for item in (await ops_client.get(f"{BASE}/workers")).json()["items"]
    }

    gpu = items[scenario.worker_ids["gpu"]]
    assert gpu["liveness"] == "ONLINE"
    assert gpu["heartbeat_age_seconds"] is not None

    stale = items[scenario.worker_ids["stale"]]
    assert stale["status"] == "ONLINE"
    assert stale["liveness"] == "STALE"

    never = items[scenario.worker_ids["unverified"]]
    assert never["liveness"] == "UNKNOWN"
    assert never["last_heartbeat"] is None


async def test_unmeasured_capabilities_stay_unknown(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    payload = (await ops_client.get(f"{BASE}/workers/{scenario.worker_ids['mac']}")).json()
    capabilities = payload["worker"]["capabilities"]
    assert capabilities["gpu_model"] is None
    assert capabilities["vram_total_mb"] is None
    assert capabilities["cuda_available"] is None  # never measured, not False
    assert capabilities["cpu_count"] == 12
    assert "vram_total_mb" in payload["unknown_capabilities"]
    assert payload["worker"]["worker_class"] == "DEVELOPMENT_ONLY"


async def test_worker_credentials_are_a_boolean_and_nothing_more(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    body = (await ops_client.get(f"{BASE}/workers/{scenario.worker_ids['gpu']}")).text
    assert '"has_credentials":true' in body.replace(" ", "")
    assert "operator-training-key" not in body
    assert "ssh_key_ref" not in body


async def test_worker_detail_reports_its_capability_signature(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    payload = (await ops_client.get(f"{BASE}/workers/{scenario.worker_ids['gpu']}")).json()
    assert payload["worker"]["capability_signature"].startswith("cap")
    assert payload["worker"]["protocol_version"] == "luber-remote/1"
    assert payload["worker"]["remote_classification"] == "CUDA_TRAINING"
    assert payload["worker"]["active_run_ids"] == [scenario.run_ids["running"]]


async def test_worker_compatibility_explains_every_refusal(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    rows = {
        item["worker"]["worker_id"]: item
        for item in (
            await ops_client.get(f"{BASE}/workers/compatibility?execution_backend=remote-gpu")
        ).json()
    }

    mac = rows[scenario.worker_ids["mac"]]
    assert mac["compatible"] is False
    assert any("DEVELOPMENT_ONLY" in reason for reason in mac["reasons"])

    unverified = rows[scenario.worker_ids["unverified"]]
    assert unverified["compatible"] is False
    assert any("never been measured" in reason for reason in unverified["reasons"])
    # Never a guessed requirement.
    assert not any("MB required" in reason for reason in unverified["reasons"])


# ── checkpoints ──────────────────────────────────────────────────────


async def test_mock_checkpoints_are_marked_and_cannot_be_evaluated(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    payload = (
        await ops_client.get(f"{BASE}/checkpoints/{scenario.checkpoint_ids['mock']}")
    ).json()["checkpoint"]
    assert payload["kind"] == "MOCK"
    assert payload["is_real_model"] is False
    assert payload["can_evaluate"] is False
    assert "no trained weights" in payload["evaluate_blocked_reason"]


async def test_a_real_adapter_can_be_evaluated(ops_client: AsyncClient, scenario: Scenario) -> None:
    payload = (
        await ops_client.get(f"{BASE}/checkpoints/{scenario.checkpoint_ids['adapter']}")
    ).json()
    checkpoint = payload["checkpoint"]
    assert checkpoint["kind"] == "ADAPTER"
    assert checkpoint["is_real_model"] is True
    assert checkpoint["can_evaluate"] is True
    assert checkpoint["sha256"]
    assert checkpoint["candidate_id"]
    assert len(payload["evaluations"]) == 3


async def test_checkpoint_location_is_a_scheme_not_a_path(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    body = (
        await ops_client.get(f"{BASE}/checkpoints/{scenario.checkpoint_ids['adapter']}")
    ).json()["checkpoint"]
    assert body["location_scheme"] == "file"
    assert "/" not in (body["location_scheme"] or "")


async def test_checkpoint_comparison_labels_training_loss_as_context(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    response = await ops_client.post(
        f"{BASE}/checkpoints/compare",
        json={
            "checkpoint_ids": [
                scenario.checkpoint_ids["adapter"],
                scenario.checkpoint_ids["mock"],
            ]
        },
    )
    payload = response.json()
    assert len(payload["rows"]) == 2
    assert "not evidence" in payload["note"]
    adapter = next(
        row
        for row in payload["rows"]
        if row["checkpoint"]["checkpoint_id"] == scenario.checkpoint_ids["adapter"]
    )
    assert adapter["training_context"]["train_loss"] == 2.02
    assert "train_loss" not in adapter["metrics"]


# ── evaluations and qualification ────────────────────────────────────


async def test_qualification_is_not_reduced_to_a_score(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    payload = (
        await ops_client.get(f"{BASE}/evaluations/{scenario.evaluation_ids['rejected']}")
    ).json()
    qualification = payload["qualification"]
    assert qualification["outcome"] == "REJECTED"
    assert qualification["failed_gates"] == ["lyric_intelligibility"]
    assert qualification["passed_gates"]
    assert "score" not in qualification
    assert payload["regressions"]
    assert payload["regressions"][0]["severity"] == "MAJOR"


async def test_human_review_required_is_a_real_outcome(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    payload = (
        await ops_client.get(f"{BASE}/evaluations/{scenario.evaluation_ids['human_review']}")
    ).json()
    assert payload["qualification"]["outcome"] == "HUMAN_REVIEW_REQUIRED"
    review = payload["human_review"]
    assert review["required"] is True
    assert review["mode"] == "LIGHT_AB"
    assert review["case_count"] == 3
    assert review["status"] == "PENDING"


async def test_qualified_does_not_mean_production(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    payload = (
        await ops_client.get(f"{BASE}/evaluations/{scenario.evaluation_ids['qualified']}")
    ).json()
    assert payload["qualification"]["outcome"] == "QUALIFIED"
    assert [item["decision"] for item in payload["promotion_reviews"]] == ["HOLD"]

    baseline = (await ops_client.get(f"{BASE}/baseline")).json()
    assert [item["model_id"] for item in baseline["production"]] == [scenario.production_model_id]
    assert "changes a model stage" in baseline["note"]


async def test_evaluation_filters(ops_client: AsyncClient, scenario: Scenario) -> None:
    payload = (await ops_client.get(f"{BASE}/evaluations?outcome=QUALIFIED")).json()
    assert [item["evaluation_id"] for item in payload["items"]] == [
        scenario.evaluation_ids["qualified"]
    ]
    assert set(payload["available_outcomes"]) == {
        "QUALIFIED",
        "REJECTED",
        "HUMAN_REVIEW_REQUIRED",
    }


async def test_evaluation_report_is_served_through_the_api(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    response = await ops_client.get(
        f"{BASE}/evaluations/{scenario.evaluation_ids['qualified']}/report"
    )
    assert response.status_code == 200
    assert "QUALIFIED" in response.text
    assert "attachment" in response.headers["content-disposition"]


# ── cost and bundles ─────────────────────────────────────────────────


async def test_cost_is_shown_only_where_it_is_known(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    known = (await ops_client.get(f"{BASE}/runs/{scenario.run_ids['running']}")).json()["cost"]
    assert known["provider"] == "example-gpu-cloud"
    assert known["hourly_rate"] == 1.29
    assert known["estimated_cost"] is not None
    assert known["actual_cost"] is None

    unknown = (await ops_client.get(f"{BASE}/runs/{scenario.run_ids['completed']}")).json()["cost"]
    assert unknown["hourly_rate"] is None
    assert unknown["estimated_cost"] is None
    assert any("hourly rate" in reason for reason in unknown["unknown"])


async def test_run_bundle_is_downloadable_and_redacted(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    response = await ops_client.get(f"{BASE}/runs/{scenario.run_ids['running']}/bundle")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    payload = response.json()
    assert payload["training_config_sha256"]
    # Neither the name of a key reference nor the deployment's directory
    # layout belongs in a browser.
    assert "operator-training-key" not in response.text
    assert "«redacted»" in response.text
    assert str(scenario.root) not in response.text
    assert payload["checkpoints"][0]["reference"].startswith("…/")


# ── catalogue ────────────────────────────────────────────────────────


async def test_catalogue_offers_builds_by_id_and_never_by_path(
    ops_client: AsyncClient, scenario: Scenario
) -> None:
    payload = (await ops_client.get(f"{BASE}/catalogue")).json()
    assert [item["build_id"] for item in payload["datasets"]] == ["primary"]
    assert [item["build_id"] for item in payload["curations"]] == ["primary"]
    assert payload["datasets"][0]["identity"] == "ds-test-001"
    assert payload["curations"][0]["source_dataset_lock_sha256"]
    assert "SMOKE" in {item["name"] for item in payload["presets"]}
    assert set(payload["backends"]) == {"dry-run", "remote-gpu"}

    body = (await ops_client.get(f"{BASE}/catalogue")).text
    assert str(scenario.root) not in body


# ── scale ────────────────────────────────────────────────────────────


async def test_a_thousand_runs_stay_paginated(ops_app, ops_client, tmp_path) -> None:
    """A registry at a scale a real project reaches.

    The assertion is not that it is fast — it is that one page is one
    page. A console that returned every run would be unusable long
    before it was slow.
    """
    from ops_fixtures import build_scenario, make_context

    big = build_scenario(tmp_path / "big", bulk_runs=1000)
    ops_app.state.ops_context = make_context(big)

    payload = (await ops_client.get(f"{BASE}/runs?limit=50")).json()
    assert payload["page"]["total"] == 1008
    assert payload["page"]["returned"] == 50
    assert len(payload["items"]) == 50

    # A caller asking for everything is clamped rather than obeyed.
    clamped = (await ops_client.get(f"{BASE}/runs?limit=100000")).json()
    assert clamped["page"]["returned"] == 200
    assert clamped["page"]["limit"] == 200

    second = (await ops_client.get(f"{BASE}/runs?limit=50&offset=50")).json()
    assert second["page"]["offset"] == 50
    first_ids = {item["run_id"] for item in payload["items"]}
    assert not first_ids & {item["run_id"] for item in second["items"]}


async def test_a_hundred_workers_stay_filterable(ops_app, ops_client, tmp_path) -> None:
    from ops_fixtures import build_scenario, make_context

    fleet = build_scenario(tmp_path / "fleet", bulk_workers=100)
    ops_app.state.ops_context = make_context(fleet)

    payload = (await ops_client.get(f"{BASE}/workers?limit=25")).json()
    assert payload["page"]["total"] == 105
    assert payload["page"]["returned"] == 25

    filtered = (await ops_client.get(f"{BASE}/workers?liveness=ONLINE")).json()
    assert filtered["page"]["total"] >= 100
    assert all(item["liveness"] == "ONLINE" for item in filtered["items"])
