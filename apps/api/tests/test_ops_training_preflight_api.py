"""The console's Phase 33 surface: preflight and canary, read-only.

Two properties this file exists to hold.

**Nothing here starts anything.** Collecting preflight evidence means
subprocesses against a trainer installation; running a canary starts one.
Neither belongs in a process a browser can reach, so both endpoints are
`GET` and both read a record the operator CLI produced.

**UNVERIFIED survives the wire.** A status that arrived at the browser
as anything other than UNVERIFIED — dropped, defaulted, coerced to
"BLOCKED" — would defeat the whole distinction, so it is asserted
explicitly rather than implied by a happy path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from ops_fixtures import Scenario

from luber_api.ops.readmodel import CANARY_RECORD_NAME, TRAINING_PREFLIGHT_NAME

pytestmark = pytest.mark.asyncio


def _run_directory(scenario: Scenario, run_key: str) -> Path:
    run_id = scenario.run_ids[run_key]
    record = scenario.registry.read("runs", run_id)
    directory = Path(str(record["output_directory"]))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _preflight(run_id: str, *, status: str = "READY", **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "luber-training-preflight/1",
        "policy_version": "training-preflight-v1",
        "status": status,
        "intent": "CANARY",
        "run_id": run_id,
        "plan_id": "plan_1",
        "plan_digest": "a" * 64,
        "execution_location": "LOCAL",
        "execution_device": "MPS",
        "torch_device": "mps",
        "resolved_precision": "bf16",
        "optimizer": "adamw",
        "worker_identity": None,
        "target_label": "Apple Silicon (arm64)",
        "capability_digest": "b" * 64,
        "hardware_snapshot": {"selected_device": "MPS", "mps_available": True},
        "capacity": {
            "device": "MPS",
            "evidence": [
                {
                    "name": "device_memory_mb",
                    "source": "MEASURED",
                    "value_mb": 24576,
                    "detail": "unified memory, shared with the operating system",
                    "derivation": "",
                    "unified_memory": True,
                },
                {
                    "name": "training_memory_requirement_mb",
                    "source": "UNKNOWN",
                    "value_mb": None,
                    "detail": "no LUBER configuration has a measured memory requirement",
                    "derivation": "",
                    "unified_memory": False,
                },
            ],
        },
        "checks": [
            {
                "name": "plan.execution_device",
                "group": "plan",
                "status": "PASS",
                "detail": "MPS (torch: mps)",
                "reason": None,
                "mandatory": True,
            }
        ],
        "blocking_reasons": [],
        "warnings": [],
        "unverified": [],
        "dataset_status": "PASS",
        "dependency_status": "PASS",
        "storage_status": "PASS",
        "checkpoint_status": "NOT_APPLICABLE",
        "canary_status": "NOT_RUN",
        "capacity_status": "UNKNOWN",
        "measured_at": "2026-08-22T12:00:00+00:00",
    }
    payload.update(overrides)
    return payload


def _canary(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "luber-training-canary/1",
        "mode": "ACE_STEP",
        "status": "PASSED",
        "detail": "the installed trainer took 1 optimizer step and wrote a checkpoint",
        "plan_digest": "c" * 64,
        "execution_location": "LOCAL",
        "execution_device": "MPS",
        "resolved_precision": "bf16",
        "optimizer": "adamw",
        "dataset_kind": "SYNTHETIC",
        "envelope": {
            "max_epochs": 1,
            "max_samples": 2,
            "max_optimizer_steps": 8,
            "wall_clock_seconds": 1500.0,
            "resume": True,
        },
        "exit_code": 0,
        "seconds": 12.9,
        "steps": 1,
        "checkpoint": {
            "ok": True,
            "step": 1,
            "provenance_plan_digest": "c" * 64,
            "problems": [],
        },
        "resume": {"ok": True, "detail": "training continued from step 1 to 2"},
    }
    payload.update(overrides)
    return payload


class TestTrainingPreflightEndpoint:
    async def test_it_is_absent_before_anybody_has_run_one(self, ops_client, scenario: Scenario):
        run_id = scenario.run_ids["completed"]
        response = await ops_client.get(f"/v1/ops/training/runs/{run_id}/training-preflight")
        assert response.status_code == 200
        payload = response.json()
        assert payload["available"] is False
        assert "No training preflight has been recorded" in payload["unavailable_reason"]

    async def test_a_ready_preflight_serialises_whole(self, ops_client, scenario: Scenario):
        run_id = scenario.run_ids["completed"]
        directory = _run_directory(scenario, "completed")
        (directory / TRAINING_PREFLIGHT_NAME).write_text(
            json.dumps(_preflight(run_id)), encoding="utf-8"
        )
        response = await ops_client.get(f"/v1/ops/training/runs/{run_id}/training-preflight")
        payload = response.json()
        assert payload["available"] is True
        assert payload["status"] == "READY"
        assert payload["execution_device"] == "MPS"
        assert payload["resolved_precision"] == "bf16"
        assert payload["checks"][0]["group"] == "plan"
        # Capacity keeps its provenance across the wire, including the
        # unified-memory caveat.
        memory = next(item for item in payload["capacity"] if item["name"] == "device_memory_mb")
        assert memory["source"] == "MEASURED"
        assert memory["unified_memory"] is True
        requirement = next(
            item for item in payload["capacity"] if item["name"].startswith("training_memory")
        )
        assert requirement["source"] == "UNKNOWN"
        assert requirement["value_mb"] is None

    async def test_a_blocked_preflight_carries_machine_readable_reasons(
        self, ops_client, scenario: Scenario
    ):
        run_id = scenario.run_ids["completed"]
        directory = _run_directory(scenario, "completed")
        (directory / TRAINING_PREFLIGHT_NAME).write_text(
            json.dumps(
                _preflight(
                    run_id,
                    status="BLOCKED",
                    blocking_reasons=[
                        "DEVICE_UNAVAILABLE: hardware.device: this machine does not offer CUDA"
                    ],
                    checks=[
                        {
                            "name": "hardware.device",
                            "group": "hardware",
                            "status": "FAIL",
                            "detail": "this machine does not offer CUDA",
                            "reason": "DEVICE_UNAVAILABLE",
                            "mandatory": True,
                        }
                    ],
                )
            ),
            encoding="utf-8",
        )
        payload = (
            await ops_client.get(f"/v1/ops/training/runs/{run_id}/training-preflight")
        ).json()
        assert payload["status"] == "BLOCKED"
        assert payload["checks"][0]["reason"] == "DEVICE_UNAVAILABLE"

    async def test_unverified_arrives_as_unverified(self, ops_client, scenario: Scenario):
        """Not coerced to BLOCKED, not softened to READY."""
        run_id = scenario.run_ids["completed"]
        directory = _run_directory(scenario, "completed")
        (directory / TRAINING_PREFLIGHT_NAME).write_text(
            json.dumps(
                _preflight(
                    run_id,
                    status="UNVERIFIED",
                    intent="FULL_TRAINING",
                    unverified=[
                        "CAPACITY_UNVERIFIED: capacity.training_requirement: nobody measured it"
                    ],
                )
            ),
            encoding="utf-8",
        )
        payload = (
            await ops_client.get(f"/v1/ops/training/runs/{run_id}/training-preflight")
        ).json()
        assert payload["status"] == "UNVERIFIED"
        assert payload["intent"] == "FULL_TRAINING"
        assert payload["unverified"]

    async def test_an_unknown_status_never_becomes_ready(self, ops_client, scenario: Scenario):
        run_id = scenario.run_ids["completed"]
        directory = _run_directory(scenario, "completed")
        (directory / TRAINING_PREFLIGHT_NAME).write_text(
            json.dumps(_preflight(run_id, status="SOMETHING_ELSE")), encoding="utf-8"
        )
        payload = (
            await ops_client.get(f"/v1/ops/training/runs/{run_id}/training-preflight")
        ).json()
        assert payload["status"] == "UNVERIFIED"

    async def test_an_unknown_run_is_404(self, ops_client):
        response = await ops_client.get("/v1/ops/training/runs/run_nope/training-preflight")
        assert response.status_code == 404

    async def test_it_is_behind_the_operator_gate(self, ops_app, scenario: Scenario):
        from httpx import ASGITransport, AsyncClient

        run_id = scenario.run_ids["completed"]
        async with AsyncClient(
            transport=ASGITransport(app=ops_app), base_url="http://ops.test"
        ) as client:
            response = await client.get(f"/v1/ops/training/runs/{run_id}/training-preflight")
        assert response.status_code in (401, 403)


class TestCanaryEndpoint:
    async def test_it_is_absent_before_a_canary_has_run(self, ops_client, scenario: Scenario):
        run_id = scenario.run_ids["completed"]
        payload = (await ops_client.get(f"/v1/ops/training/runs/{run_id}/canary")).json()
        assert payload["available"] is False
        assert payload["status"] == "NOT_RUN"

    async def test_a_passed_canary_carries_its_bounds(self, ops_client, scenario: Scenario):
        run_id = scenario.run_ids["completed"]
        directory = _run_directory(scenario, "completed")
        (directory / CANARY_RECORD_NAME).write_text(json.dumps(_canary()), encoding="utf-8")
        payload = (await ops_client.get(f"/v1/ops/training/runs/{run_id}/canary")).json()
        assert payload["status"] == "PASSED"
        assert payload["max_optimizer_steps"] == 8
        assert payload["max_samples"] == 2
        assert payload["dataset_kind"] == "SYNTHETIC"
        assert payload["checkpoint_ok"] is True
        assert payload["resume_ok"] is True

    async def test_a_failed_canary_reports_its_checkpoint_problems(
        self, ops_client, scenario: Scenario
    ):
        run_id = scenario.run_ids["completed"]
        directory = _run_directory(scenario, "completed")
        (directory / CANARY_RECORD_NAME).write_text(
            json.dumps(
                _canary(
                    status="FAILED",
                    checkpoint={
                        "ok": False,
                        "step": None,
                        "provenance_plan_digest": None,
                        "problems": ["every adapter tensor is zero"],
                    },
                    resume=None,
                )
            ),
            encoding="utf-8",
        )
        payload = (await ops_client.get(f"/v1/ops/training/runs/{run_id}/canary")).json()
        assert payload["status"] == "FAILED"
        assert payload["checkpoint_problems"] == ["every adapter tensor is zero"]
        assert payload["resume_ok"] is None

    async def test_both_appear_on_the_run_detail(self, ops_client, scenario: Scenario):
        run_id = scenario.run_ids["completed"]
        directory = _run_directory(scenario, "completed")
        (directory / TRAINING_PREFLIGHT_NAME).write_text(
            json.dumps(_preflight(run_id)), encoding="utf-8"
        )
        (directory / CANARY_RECORD_NAME).write_text(json.dumps(_canary()), encoding="utf-8")
        payload = (await ops_client.get(f"/v1/ops/training/runs/{run_id}")).json()
        assert payload["training_preflight"]["status"] == "READY"
        assert payload["canary"]["status"] == "PASSED"
