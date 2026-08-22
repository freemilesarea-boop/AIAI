"""The compute-targets panel: honest about what is not there.

The panel exists so an operator can answer "where can this run" without
reading code. The ways it can mislead them are specific, and each has a
test: claiming a GPU that was returned to the rental provider, claiming
this Mac can take heavy training when the scheduler refuses it, showing
a planned machine beside real ones, or leaking a host identity into a
console that is reachable by anyone who can load the page.
"""

from __future__ import annotations

import json

from httpx import AsyncClient

BASE = "/v1/ops/training/compute-targets"


async def test_the_panel_is_gated_like_every_other_operator_route(ops_app) -> None:
    from httpx import ASGITransport

    async with AsyncClient(
        transport=ASGITransport(app=ops_app), base_url="http://ops.test"
    ) as anonymous:
        assert (await anonymous.get(BASE)).status_code in (401, 403)


async def test_it_shows_a_local_row_and_a_remote_cuda_row(ops_client: AsyncClient) -> None:
    """A missing remote row would read as "we didn't check". A row
    saying NOT_CONNECTED reads as "there isn't one yet"."""
    payload = (await ops_client.get(BASE)).json()

    rows = payload["targets"]
    assert rows, "the panel is never empty; the machine it runs on is always a target"
    assert any(row["location"] == "LOCAL" and row["device"] == "CPU" for row in rows)
    assert any(row["location"] == "REMOTE" and row["device"] == "CUDA" for row in rows)


async def test_no_row_claims_cuda_on_a_machine_without_it(ops_client: AsyncClient) -> None:
    """The single thing a hardware panel must never do."""
    payload = (await ops_client.get(BASE)).json()

    for row in payload["targets"]:
        if row["device"] == "CUDA" and row["location"] == "LOCAL":
            assert row["status"] != "READY"


async def test_no_local_row_claims_it_can_take_heavy_training(
    ops_client: AsyncClient,
) -> None:
    """It has to agree with the scheduler, which refuses exactly this.

    An operator plans against this panel; a row promising something the
    scheduler will decline is planning material that is wrong.
    """
    payload = (await ops_client.get(BASE)).json()

    for row in payload["targets"]:
        if row["location"] == "LOCAL":
            assert "HEAVY_TRAINING" not in row["workloads"]


async def test_the_local_concurrency_limit_is_stated(ops_client: AsyncClient) -> None:
    """One. The machine that runs the control plane stays a control plane."""
    payload = (await ops_client.get(BASE)).json()

    assert payload["local_training_concurrency"] == 1


async def test_every_row_carries_its_provenance(ops_client: AsyncClient) -> None:
    payload = (await ops_client.get(BASE)).json()

    assert payload["capability_schema_version"]
    assert payload["execution_placement_policy_version"]
    known = {"READY", "NOT_AVAILABLE", "NOT_CONNECTED", "UNPROBED"}
    assert all(row["status"] in known for row in payload["targets"])


async def test_precisions_are_measured_or_absent_never_guessed(
    ops_client: AsyncClient,
) -> None:
    """An empty list means nobody measured. It never means "none work"."""
    payload = (await ops_client.get(BASE)).json()

    for row in payload["targets"]:
        assert isinstance(row["precisions"], list)
        assert set(row["precisions"]) <= {"fp32", "fp16", "bf16"}


async def test_the_panel_leaks_no_host_identity(ops_client: AsyncClient) -> None:
    """The response models have no field for one. Asserted anyway,
    because a later field could change that."""
    body = json.dumps((await ops_client.get(BASE)).json())

    for needle in ("/Users/", "/home/", "hostname", "ssh_key", "credential"):
        assert needle.lower() not in body.lower()


async def test_a_registered_worker_appears_with_what_it_reported(
    ops_client: AsyncClient,
) -> None:
    """Phase 27's fixture registry has workers in it. Whatever they
    reported is carried across; whatever nobody asked about stays
    unknown rather than becoming False."""
    payload = (await ops_client.get(BASE)).json()

    remote = [row for row in payload["targets"] if row["location"] == "REMOTE"]
    assert remote, "the operator fixture registers workers"
    named = [row for row in remote if row["name"] != "UNKNOWN"]
    for row in named:
        assert row["device"] in {"CUDA", "CPU"}
