"""The inference console's HTTP surface: what it shows, and what it cannot.

Two things are being proved here at once. That the numbers are right, and
that no route can be made to return something a user wrote. The second is
tested against a fixture whose generations carry real prompts, lyrics and
titles — including a `request_trace` holding the prompt in full, which is
the actual hazard — so a leak fails loudly rather than being absent
because the fixture was sanitised.
"""

from __future__ import annotations

import json

from httpx import ASGITransport, AsyncClient
from inference_fixtures import NOW, OPERATOR_TOKEN, assert_no_user_content

BASE = "/v1/ops/inference"


# ── the gate ─────────────────────────────────────────────────────────


async def test_every_route_refuses_without_the_operator_token(inference_app):
    """The gate is on the router, so a route added later is protected by
    having been added rather than by somebody remembering."""
    async with AsyncClient(
        transport=ASGITransport(app=inference_app), base_url="http://ops.test"
    ) as anonymous:
        for path in (
            "/overview",
            "/summary",
            "/trend?chart=retry",
            "/providers",
            "/incidents",
            "/segments",
            "/generations",
            "/regressions",
            "/ingest-status",
        ):
            response = await anonymous.get(f"{BASE}{path}")
            assert response.status_code in (401, 403), path


async def test_a_wrong_token_is_refused(inference_app):
    async with AsyncClient(
        transport=ASGITransport(app=inference_app),
        base_url="http://ops.test",
        headers={"X-Luber-Operator-Token": "not-the-token"},
    ) as wrong:
        assert (await wrong.get(f"{BASE}/overview")).status_code in (401, 403)


async def test_the_console_is_absent_when_it_is_switched_off(app):
    """The default app has no console, so the path does not exist.

    404 rather than 403: a 403 would confirm to a prober that this
    deployment has an inference console worth attacking.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://ops.test") as client:
        assert (await client.get(f"{BASE}/overview")).status_code == 404


# ── privacy ──────────────────────────────────────────────────────────


async def test_no_route_returns_anything_a_user_wrote(inference_client):
    """The privacy test, run over every route that returns data."""
    paths = [
        "/overview?window=24h",
        "/summary?window=7d",
        "/trend?chart=retry&window=24h",
        "/trend?chart=failure&window=24h",
        "/trend?chart=latency&window=24h",
        "/providers?window=7d",
        "/segments?window=7d",
        "/regressions?window=1h",
        "/incidents",
        "/generations?window=24h",
        "/ingest-status",
    ]
    for path in paths:
        response = await inference_client.get(f"{BASE}{path}")
        assert response.status_code == 200, f"{path}: {response.text}"
        assert_no_user_content(response.json())


async def test_the_generation_drilldown_shows_no_prompt(inference_client):
    """The one route that reads the generations table directly.

    It selects a single column by primary key, so there is no query that
    could return a prompt — and the response model has no field to hold
    one if there were.
    """
    listing = (await inference_client.get(f"{BASE}/generations?window=24h&limit=1")).json()
    generation_id = listing["items"][0]["generation_id"]

    response = await inference_client.get(f"{BASE}/generations/{generation_id}")

    assert response.status_code == 200
    body = response.json()
    assert_no_user_content(body)
    # It still explains the generation.
    assert body["qc_data_available"] is True
    assert body["attempts"]
    assert any("selected" in line.lower() for line in body["explanation"])


# ── health ───────────────────────────────────────────────────────────


async def test_the_overview_carries_counts_beside_every_rate(inference_client):
    """A percentage without its counts is unreadable: 2.86% could be
    12 of 420 or 2 of 70, and those need different responses."""
    body = (await inference_client.get(f"{BASE}/overview?window=7d")).json()

    assert body["summary"]["sample_count"] > 0
    for name, rate in body["summary"]["overview"].items():
        assert "numerator" in rate and "denominator" in rate, name
        assert rate["render"], name
        if rate["status"] == "OK":
            assert f"{rate['numerator']}/{rate['denominator']}" in rate["render"]


async def test_the_overview_states_that_nothing_is_automatic(inference_client):
    body = (await inference_client.get(f"{BASE}/overview?window=24h")).json()
    assert "none" in body["automatic_remediation"].lower()


async def test_an_empty_window_reads_as_no_data_rather_than_zero(empty_client):
    """0% failure on an hour when nothing ran is a green light for a
    system that was switched off."""
    body = (await empty_client.get(f"{BASE}/summary?window=1h")).json()

    assert body["sample_count"] == 0
    for rate in body["overview"].values():
        assert rate["status"] == "NO_DATA"
        assert rate["value"] is None
        assert "NO_DATA" in rate["render"]


async def test_latency_is_reported_as_quantiles(inference_client):
    body = (await inference_client.get(f"{BASE}/summary?window=7d")).json()
    total = body["latency"]["total_latency_seconds"]
    assert total["count"] > 0
    assert total["p50"] is not None and total["p95"] is not None
    assert total["p99"] is not None


# ── trends ───────────────────────────────────────────────────────────


async def test_a_trend_carries_a_sample_count_per_bucket(inference_client):
    body = (await inference_client.get(f"{BASE}/trend?chart=retry&window=24h")).json()
    assert body["has_data"] is True
    assert body["points"]
    for point in body["points"]:
        assert "sample_count" in point


async def test_an_empty_bucket_is_null_rather_than_zero(inference_client):
    """A chart that drew zero through a quiet night would show a
    recovery that never happened."""
    body = (await inference_client.get(f"{BASE}/trend?chart=failure&window=24h")).json()
    empty = [point for point in body["points"] if point["sample_count"] == 0]
    assert empty, "the fixture should have quiet buckets"
    for point in empty:
        assert all(value is None for value in point["values"].values())


async def test_an_unknown_chart_is_refused(inference_client):
    assert (await inference_client.get(f"{BASE}/trend?chart=nonsense")).status_code == 422


# ── providers and segments ───────────────────────────────────────────


async def test_providers_are_listed_with_their_own_numbers(inference_client):
    body = (await inference_client.get(f"{BASE}/providers?window=7d")).json()
    assert body["providers"]
    for provider in body["providers"]:
        assert provider["provider_revision"]
        assert provider["baseline_status"] in ("READY", "BASELINE_BUILDING")
        assert provider["sample_count"] > 0


async def test_a_segment_ranking_refuses_a_split_too_wide_to_support_a_finding(
    inference_client,
):
    response = await inference_client.get(
        f"{BASE}/segments?group_by=provider,duration_bucket,task_type,language"
    )
    assert response.status_code == 409
    assert "dimension" in response.json()["detail"]


async def test_segment_ranking_reports_what_it_dropped(inference_client):
    """A short list must read as "most segments are small", not as
    "only these segments exist"."""
    body = (await inference_client.get(f"{BASE}/segments?window=7d&minimum_samples=100000")).json()
    assert body["segments"] == []
    assert body["segments_below_minimum"] > 0


# ── regressions and incidents ────────────────────────────────────────


async def test_the_seeded_regression_is_reported_with_its_evidence(inference_client):
    body = (await inference_client.get(f"{BASE}/regressions?window=1h")).json()
    assert body, "the fixture seeds a collapse regression"
    finding = body[0]
    assert finding["baseline_denominator"] and finding["current_denominator"]
    assert finding["threshold_crossed"]
    assert finding["explanation"]
    assert finding["recommendations"]


async def test_a_regression_explains_what_moved_and_never_why(inference_client):
    body = (await inference_client.get(f"{BASE}/regressions?window=1h")).json()
    rendered = json.dumps(body).lower()
    for causal in ("caused by", "because of", "due to the deploy"):
        assert causal not in rendered


async def test_incidents_are_listed_and_openable(inference_client):
    listing = (await inference_client.get(f"{BASE}/incidents")).json()
    assert listing["total"] > 0
    incident_id = listing["items"][0]["incident_id"]

    detail = (await inference_client.get(f"{BASE}/incidents/{incident_id}")).json()

    assert detail["incident_id"] == incident_id
    assert detail["evidence"]
    assert detail["baseline_window"] and detail["current_window"]
    assert detail["segment_label"]


async def test_a_missing_incident_is_a_404(inference_client):
    assert (await inference_client.get(f"{BASE}/incidents/nope")).status_code == 404


async def test_acknowledging_records_the_operator_and_keeps_measuring(inference_client):
    listing = (await inference_client.get(f"{BASE}/incidents")).json()
    incident_id = listing["items"][0]["incident_id"]

    response = await inference_client.post(
        f"{BASE}/incidents/{incident_id}/acknowledge?operator=alex"
    )

    assert response.status_code == 200
    incident = response.json()["incident"]
    assert incident["status"] == "ACKNOWLEDGED"
    assert incident["acknowledged_by"] == "alex"
    # Acknowledgement is not suppression: the evidence is still there.
    assert incident["evidence"]


async def test_dismissal_requires_a_reason_and_keeps_the_history(inference_client):
    listing = (await inference_client.get(f"{BASE}/incidents")).json()
    incident_id = listing["items"][0]["incident_id"]

    missing = await inference_client.post(f"{BASE}/incidents/{incident_id}/dismiss?operator=alex")
    assert missing.status_code == 422

    response = await inference_client.post(
        f"{BASE}/incidents/{incident_id}/dismiss?operator=alex&reason=known+load+test"
    )

    assert response.status_code == 200
    incident = response.json()["incident"]
    assert incident["status"] == "DISMISSED"
    assert incident["dismissal_reason"] == "known load test"
    assert incident["evidence"], "dismissal must not delete the history"


async def test_reading_regressions_does_not_create_incidents(inference_client):
    """Refreshing a browser tab must not mint incidents.

    Opening them is the scheduled detector's job; this endpoint
    evaluates and reports.
    """
    before = (await inference_client.get(f"{BASE}/incidents?include_closed=true")).json()["total"]
    for _ in range(5):
        await inference_client.get(f"{BASE}/regressions?window=1h")
    after = (await inference_client.get(f"{BASE}/incidents?include_closed=true")).json()["total"]
    assert after == before


# ── pagination and staleness ─────────────────────────────────────────


async def test_generation_lists_are_paginated(inference_client):
    first = (await inference_client.get(f"{BASE}/generations?window=24h&limit=5")).json()
    assert len(first["items"]) == 5
    assert first["total"] > 5

    second = (await inference_client.get(f"{BASE}/generations?window=24h&limit=5&offset=5")).json()
    assert {item["generation_id"] for item in first["items"]}.isdisjoint(
        item["generation_id"] for item in second["items"]
    )


async def test_a_client_cannot_ask_for_everything(inference_client):
    body = (await inference_client.get(f"{BASE}/incidents?limit=100000")).json()
    assert body["limit"] <= 200


async def test_ingest_status_says_when_nothing_has_been_ingested(empty_client):
    body = (await empty_client.get(f"{BASE}/ingest-status")).json()
    assert body["observations"] == 0
    assert body["latest_observation_at"] is None
    assert body["stale"] is True
    assert "backfill" in (body["note"] or "")


async def test_ingest_status_reports_how_far_behind_it_is(inference_client):
    body = (await inference_client.get(f"{BASE}/ingest-status")).json()
    assert body["observations"] > 0
    assert body["latest_observation_at"] is not None
    assert body["seconds_behind"] is not None


# ── comparisons ──────────────────────────────────────────────────────


async def test_a_revision_comparison_says_it_is_not_an_experiment(inference_client):
    body = (
        await inference_client.get(
            f"{BASE}/providers/compare?left=acestep@v1&right=acestep@v2&window=7d"
        )
    ).json()
    assert "caveat" in body
    assert "controlled experiment" in body["caveat"]


async def test_a_before_after_comparison_claims_correlation_only(inference_client):
    body = (await inference_client.get(f"{BASE}/before-after?at={NOW.isoformat()}&hours=24")).json()
    assert "caveat" in body
    assert "no evidence of cause" in body["caveat"]


async def test_a_before_after_comparison_needs_a_real_timestamp(inference_client):
    assert (await inference_client.get(f"{BASE}/before-after?at=not-a-date")).status_code == 400


def test_the_operator_token_is_never_in_a_response(inference_app):
    """No response model has a field a secret could occupy."""
    from luber_api.ops import inference_schemas

    rendered = json.dumps(
        {
            name: sorted(model.model_fields)
            for name, model in vars(inference_schemas).items()
            if hasattr(model, "model_fields")
        }
    )
    for secret in ("token", "password", "api_key", "secret", "credential"):
        assert secret not in rendered.lower()
    assert OPERATOR_TOKEN not in rendered
