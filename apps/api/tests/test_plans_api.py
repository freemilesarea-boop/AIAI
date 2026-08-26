"""Plans, allowance and download entitlement, at the HTTP boundary.

`packages/database/tests/test_allowance.py` proves the ledger arithmetic
and the concurrency invariant. This file proves the part a user can
actually reach: that the limit is enforced by the server rather than by
the page, that a refusal is a refusal and not a charge, and that a
plan check never becomes a way to learn about somebody else's songs.

Every test here drives the real routes with a real session cookie. There
is no test-only bypass of the session dependency, for the same reason
Phase 4 refused one: a suite that skips the boundary cannot prove it.
"""

from __future__ import annotations

from fastapi import FastAPI
from httpx import AsyncClient
from plan_fixtures import set_plan

from luber_api.schemas import MAX_RESULT_COUNT
from luber_schemas.plans import PLANS, PlanId


def _payload(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "title": "Midnight Window",
        "prompt": "quiet synth pop, warm pads",
        "lyrics": "[verse]\nthe city keeps its light on\n",
        "vocal_gender": "female",
        "duration": 30,
    }
    body.update(overrides)
    return body


# ── the catalogue ────────────────────────────────────────────────────


async def test_the_catalogue_is_readable_without_a_session(anon_client: AsyncClient) -> None:
    """Pricing is public. Requiring a login to read it would only mean
    the marketing page kept its own copy of the numbers."""
    response = await anon_client.get("/v1/plans")

    assert response.status_code == 200
    plans = response.json()["plans"]
    assert [plan["plan_id"] for plan in plans] == ["free", "basic", "pro", "creator"]


async def test_the_published_prices_are_the_configured_prices(anon_client: AsyncClient) -> None:
    """The browser must render what the server will enforce. One source."""
    served = {
        plan["plan_id"]: plan for plan in (await anon_client.get("/v1/plans")).json()["plans"]
    }

    for plan_id, plan in PLANS.items():
        assert served[plan_id.value]["monthly_price_krw"] == plan.monthly_price_krw
        assert served[plan_id.value]["monthly_generation_limit"] == plan.monthly_generation_limit
        assert served[plan_id.value]["download_mp3"] == plan.download_mp3
        assert served[plan_id.value]["commercial_use"] == plan.commercial_use


async def test_checkout_is_advertised_as_unavailable(anon_client: AsyncClient) -> None:
    """No payment provider is connected. The page must not offer a
    checkout it cannot honour, and the server is where that is stated."""
    assert (await anon_client.get("/v1/plans")).json()["checkout_available"] is False


async def test_exactly_one_tier_is_recommended(anon_client: AsyncClient) -> None:
    plans = (await anon_client.get("/v1/plans")).json()["plans"]
    recommended = [plan["plan_id"] for plan in plans if plan["recommended"]]

    assert recommended == ["pro"]


async def test_there_is_no_route_that_grants_a_plan(app: FastAPI) -> None:
    """The load-bearing absence.

    Any endpoint that let an authenticated account choose its own tier
    would be a way to take Creator for nothing. Plans are assigned by a
    script that needs shell access, and this test fails the day someone
    adds the convenient thing.
    """

    # Walked recursively: this FastAPI version keeps each `include_router`
    # as a wrapper object in `app.routes` with no `path` of its own, so a
    # flat scan finds nothing and passes without checking anything.
    def walk(routes: object) -> list[tuple[str, str]]:
        found: list[tuple[str, str]] = []
        for route in routes:  # type: ignore[attr-defined]
            path = getattr(route, "path", None)
            for method in getattr(route, "methods", set()) or set():
                if path:
                    found.append((method, path))
            # `include_router` is kept as a wrapper holding the router it
            # mounted, rather than being flattened into `app.routes`.
            inner = getattr(route, "original_router", None)
            nested = getattr(inner, "routes", None) or getattr(route, "routes", None)
            if nested:
                found.extend(walk(nested))
        return found

    seen = walk(app.routes)
    # The scan works: it can see the endpoints that do exist.
    assert ("GET", "/v1/plans") in seen
    assert ("GET", "/v1/account/entitlement") in seen

    mutating = [
        (method, path)
        for method, path in seen
        if ("plan" in path or "subscription" in path)
        and method in {"POST", "PUT", "PATCH", "DELETE"}
    ]

    assert mutating == []


# ── entitlement ──────────────────────────────────────────────────────


async def test_entitlement_requires_a_session(anon_client: AsyncClient) -> None:
    assert (await anon_client.get("/v1/account/entitlement")).status_code == 401


async def test_a_new_account_is_free(app: FastAPI, client: AsyncClient) -> None:
    """No subscription row means Free, with no backfill anywhere."""
    await set_plan(app, client.user_id, PlanId.FREE)  # type: ignore[attr-defined]

    body = (await client.get("/v1/account/entitlement")).json()

    assert body["plan"]["plan_id"] == "free"
    assert body["generation_limit"] == 20
    assert body["generation_used"] == 0
    assert body["generation_remaining"] == 20
    assert body["download_mp3"] is False
    assert body["commercial_use"] is False


async def test_entitlement_carries_no_storage_or_billing_internals(client: AsyncClient) -> None:
    """The response is the minimum the frontend needs. Nothing about how
    it is stored, no row ids, no ledger."""
    body = (await client.get("/v1/account/entitlement")).json()

    assert set(body) == {
        "plan",
        "period_start",
        "period_end",
        "generation_limit",
        "generation_used",
        "generation_remaining",
        "download_mp3",
        "download_wav",
        "commercial_use",
    }
    assert "subscription_id" not in body
    assert "user_id" not in body


async def test_entitlement_reports_only_the_caller(
    app: FastAPI, client: AsyncClient, client_b: AsyncClient
) -> None:
    """There is no user id to supply, so there is nothing to tamper with."""
    await set_plan(app, client.user_id, PlanId.FREE)  # type: ignore[attr-defined]
    await set_plan(app, client_b.user_id, PlanId.CREATOR)  # type: ignore[attr-defined]

    assert (await client.get("/v1/account/entitlement")).json()["plan"]["plan_id"] == "free"
    assert (await client_b.get("/v1/account/entitlement")).json()["plan"]["plan_id"] == "creator"


async def test_a_generation_moves_the_counter(client: AsyncClient) -> None:
    before = (await client.get("/v1/account/entitlement")).json()

    created = await client.post("/v1/generations", json=_payload())
    assert created.status_code == 202, created.text

    after = (await client.get("/v1/account/entitlement")).json()
    assert after["generation_used"] == before["generation_used"] + 1
    assert after["generation_remaining"] == before["generation_remaining"] - 1


async def test_two_results_spend_two_slots(client: AsyncClient) -> None:
    """One request, two songs, two units. The user received two songs."""
    before = (await client.get("/v1/account/entitlement")).json()

    created = await client.post("/v1/generations", json=_payload(result_count=2))
    assert created.status_code == 202, created.text
    assert len(created.json()["generations"]) == 2

    after = (await client.get("/v1/account/entitlement")).json()
    assert after["generation_used"] == before["generation_used"] + 2


# ── the limit ────────────────────────────────────────────────────────


async def _exhaust(client: AsyncClient, limit: int) -> None:
    for _ in range(limit):
        response = await client.post("/v1/generations", json=_payload())
        assert response.status_code == 202, response.text


async def test_an_exhausted_account_is_refused_by_the_server(
    app: FastAPI, client: AsyncClient
) -> None:
    await set_plan(app, client.user_id, PlanId.FREE)  # type: ignore[attr-defined]
    await _exhaust(client, PLANS[PlanId.FREE].monthly_generation_limit)

    refused = await client.post("/v1/generations", json=_payload())

    assert refused.status_code == 402
    assert refused.json()["detail"] == "GENERATION_LIMIT_REACHED"


async def test_the_refusal_creates_nothing(app: FastAPI, client: AsyncClient) -> None:
    """A refused request must not leave a Library full of failed songs."""
    await set_plan(app, client.user_id, PlanId.FREE)  # type: ignore[attr-defined]
    limit = PLANS[PlanId.FREE].monthly_generation_limit
    await _exhaust(client, limit)
    before = (await client.get("/v1/generations")).json()["total"]

    await client.post("/v1/generations", json=_payload())

    assert (await client.get("/v1/generations")).json()["total"] == before


async def test_the_last_slot_is_still_usable(app: FastAPI, client: AsyncClient) -> None:
    """Off-by-one in the honest direction: twenty means twenty, not nineteen."""
    await set_plan(app, client.user_id, PlanId.FREE)  # type: ignore[attr-defined]
    limit = PLANS[PlanId.FREE].monthly_generation_limit
    await _exhaust(client, limit - 1)

    assert (await client.post("/v1/generations", json=_payload())).status_code == 202
    assert (await client.get("/v1/account/entitlement")).json()["generation_remaining"] == 0


async def test_a_bigger_plan_gets_a_bigger_allowance(app: FastAPI, client: AsyncClient) -> None:
    await set_plan(app, client.user_id, PlanId.FREE)  # type: ignore[attr-defined]
    await _exhaust(client, PLANS[PlanId.FREE].monthly_generation_limit)
    assert (await client.post("/v1/generations", json=_payload())).status_code == 402

    await set_plan(app, client.user_id, PlanId.PRO)  # type: ignore[attr-defined]

    assert (await client.post("/v1/generations", json=_payload())).status_code == 202


async def test_one_accounts_spending_does_not_touch_another(
    app: FastAPI, client: AsyncClient, client_b: AsyncClient
) -> None:
    await set_plan(app, client.user_id, PlanId.FREE)  # type: ignore[attr-defined]
    await set_plan(app, client_b.user_id, PlanId.FREE)  # type: ignore[attr-defined]
    await _exhaust(client, PLANS[PlanId.FREE].monthly_generation_limit)

    assert (await client.post("/v1/generations", json=_payload())).status_code == 402
    assert (await client_b.post("/v1/generations", json=_payload())).status_code == 202


async def test_a_partial_request_yields_what_was_left(app: FastAPI, client: AsyncClient) -> None:
    """Asking for two songs with one left gives one song, not an error.

    The user gets what their allowance can pay for; the extra result is
    refused on its own rather than the whole request being thrown away.
    """
    await set_plan(app, client.user_id, PlanId.FREE)  # type: ignore[attr-defined]
    await _exhaust(client, PLANS[PlanId.FREE].monthly_generation_limit - 1)

    created = await client.post("/v1/generations", json=_payload(result_count=MAX_RESULT_COUNT))

    assert created.status_code == 202
    queued = [item for item in created.json()["generations"] if item["status"] != "FAILED"]
    assert len(queued) == 1
    assert (await client.get("/v1/account/entitlement")).json()["generation_remaining"] == 0


# ── downloads ────────────────────────────────────────────────────────


async def _completed_generation(client: AsyncClient) -> str:
    """A finished song belonging to *client*.

    The mock provider completes inline, so the generation is COMPLETED by
    the time the create call returns and its audio is downloadable.
    """
    created = await client.post("/v1/generations", json=_payload())
    assert created.status_code == 202, created.text
    generation_id = str(created.json()["generation_id"])
    body = (await client.get(f"/v1/generations/{generation_id}")).json()
    assert body["status"] == "COMPLETED", body["status"]
    return generation_id


async def test_free_cannot_download(app: FastAPI, client: AsyncClient) -> None:
    """The line the pricing page draws, enforced where it counts."""
    generation_id = await _completed_generation(client)
    await set_plan(app, client.user_id, PlanId.FREE)  # type: ignore[attr-defined]

    refused = await client.get(
        f"/v1/generations/{generation_id}/audio",
        params={"asset": "master", "download": "true"},
    )

    assert refused.status_code == 402
    assert refused.json()["detail"] == "DOWNLOAD_NOT_IN_PLAN"


async def test_free_can_still_listen(app: FastAPI, client: AsyncClient) -> None:
    """Free accounts own what they made. They can play all of it — the
    plan gates the save, not the song."""
    generation_id = await _completed_generation(client)
    await set_plan(app, client.user_id, PlanId.FREE)  # type: ignore[attr-defined]

    streamed = await client.get(
        f"/v1/generations/{generation_id}/audio", params={"asset": "master"}
    )

    assert streamed.status_code in (200, 302, 307)


async def test_a_paid_plan_can_download(app: FastAPI, client: AsyncClient) -> None:
    generation_id = await _completed_generation(client)
    await set_plan(app, client.user_id, PlanId.BASIC)  # type: ignore[attr-defined]

    saved = await client.get(
        f"/v1/generations/{generation_id}/audio",
        params={"asset": "master", "download": "true"},
    )

    assert saved.status_code in (200, 302, 307)


async def test_a_foreign_download_is_404_not_402(
    app: FastAPI, client: AsyncClient, client_b: AsyncClient
) -> None:
    """Ownership is checked first, and the plan check never becomes an
    existence oracle.

    A Free account asking for somebody else's file must be told the file
    does not exist. Answering "your plan is too small" would confirm that
    the id names a real generation — which is precisely the leak the
    ownership rule exists to prevent.
    """
    generation_id = await _completed_generation(client)
    await set_plan(app, client_b.user_id, PlanId.FREE)  # type: ignore[attr-defined]

    response = await client_b.get(
        f"/v1/generations/{generation_id}/audio",
        params={"asset": "master", "download": "true"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "generation not found"


async def test_an_upgrade_unlocks_an_existing_song(app: FastAPI, client: AsyncClient) -> None:
    """Entitlement is evaluated per request, not stamped onto the file.

    A song made on Free becomes downloadable when the account upgrades —
    it was always the user's song.
    """
    generation_id = await _completed_generation(client)
    await set_plan(app, client.user_id, PlanId.FREE)  # type: ignore[attr-defined]
    url = f"/v1/generations/{generation_id}/audio"
    params = {"asset": "master", "download": "true"}
    assert (await client.get(url, params=params)).status_code == 402

    await set_plan(app, client.user_id, PlanId.PRO)  # type: ignore[attr-defined]

    assert (await client.get(url, params=params)).status_code in (200, 302, 307)


# ── failures are not charged ─────────────────────────────────────────


async def test_a_generation_that_never_reached_the_queue_is_refunded(
    app: FastAPI, client: AsyncClient
) -> None:
    """The one behaviour this system must never have is charging for the
    product not working.

    The slot is taken before the job is queued, so a queue outage happens
    with the reservation already held. Marking the generation failed has
    to give it back — and it does, because settlement lives in the
    repository method every failure path already calls.
    """
    before = (await client.get("/v1/account/entitlement")).json()["generation_used"]

    async def always_fails(generation_id: object) -> None:
        raise RuntimeError("queue unavailable")

    app.state.enqueuer.enqueue = always_fails
    refused = await client.post("/v1/generations", json=_payload())
    assert refused.status_code == 503

    after = (await client.get("/v1/account/entitlement")).json()["generation_used"]
    assert after == before


async def test_a_completed_generation_stays_charged(client: AsyncClient) -> None:
    """The mirror of the refund: a song the user received is spent."""
    before = (await client.get("/v1/account/entitlement")).json()["generation_used"]

    generation_id = await _completed_generation(client)
    assert generation_id

    after = (await client.get("/v1/account/entitlement")).json()["generation_used"]
    assert after == before + 1
