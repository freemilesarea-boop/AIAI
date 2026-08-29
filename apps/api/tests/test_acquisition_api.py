"""Acquisition capture and reporting at the HTTP boundary.

The ingestion endpoint is the only public one BOORDA has that writes,
so the questions asked here are about what it will accept, what it
refuses to store, and what it tells a caller. The answer to the last is
"nothing" — it returns 204 with a cookie and no body, whatever happened.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy import insert, select

from luber_api.routes.acquisition import VISITOR_COOKIE_NAME
from luber_database.models.acquisition import (
    AcquisitionAttribution,
    AcquisitionSession,
    AcquisitionVisitor,
)
from luber_database.models.payments import PAYMENT_SUCCEEDED, BillingPayment

VISIT = "/v1/acquisition/visit"


async def _sessions(app: FastAPI) -> list[AcquisitionSession]:
    factory = app.state.session_factory
    async with factory() as session:
        return list((await session.execute(select(AcquisitionSession))).scalars().all())


async def _visitors(app: FastAPI) -> list[AcquisitionVisitor]:
    factory = app.state.session_factory
    async with factory() as session:
        return list((await session.execute(select(AcquisitionVisitor))).scalars().all())


# ── ingestion ────────────────────────────────────────────────────────


async def test_a_visit_is_recorded_and_a_cookie_is_issued(
    app: FastAPI, anon_client: AsyncClient
) -> None:
    response = await anon_client.post(
        VISIT,
        json={
            "path": "/",
            "referrer": "https://www.instagram.com/",
            "params": {"utm_source": "instagram", "utm_medium": "paid_social"},
        },
    )

    assert response.status_code == 204
    assert VISITOR_COOKIE_NAME in response.cookies
    rows = await _sessions(app)
    assert len(rows) == 1
    assert (rows[0].source, rows[0].medium) == ("instagram", "paid_social")


async def test_the_endpoint_never_answers_with_a_body(anon_client: AsyncClient) -> None:
    """A caller learns nothing about our data from calling this."""
    response = await anon_client.post(VISIT, json={"path": "/"})

    assert response.status_code == 204
    assert response.content == b""


async def test_the_visitor_cookie_is_not_readable_by_scripts(anon_client: AsyncClient) -> None:
    response = await anon_client.post(VISIT, json={"path": "/"})

    header = response.headers.get("set-cookie", "")
    assert "httponly" in header.lower()
    assert "samesite=lax" in header.lower()


async def test_the_same_browser_stays_one_visitor(app: FastAPI, anon_client: AsyncClient) -> None:
    await anon_client.post(VISIT, json={"path": "/", "params": {"utm_source": "google"}})
    await anon_client.post(VISIT, json={"path": "/plans"})

    visitors = await _visitors(app)
    assert len(visitors) == 1
    assert len(await _sessions(app)) == 2


async def test_direct_return_does_not_overwrite_the_campaign(
    app: FastAPI, anon_client: AsyncClient
) -> None:
    await anon_client.post(
        VISIT,
        json={"path": "/", "params": {"utm_source": "instagram", "utm_medium": "paid_social"}},
    )
    await anon_client.post(VISIT, json={"path": "/"})

    visitor = (await _visitors(app))[0]
    assert visitor.last_source == "instagram"


async def test_a_sensitive_parameter_is_never_stored(
    app: FastAPI, anon_client: AsyncClient
) -> None:
    """A landing URL can carry a reset token. Analytics has no use for
    it and this table is not where it should end up."""
    await anon_client.post(
        VISIT,
        json={
            "path": "/reset?token=super-secret-value",
            "params": {"utm_source": "email", "access_token": "super-secret-value"},
        },
    )

    rows = await _sessions(app)
    assert len(rows) == 1
    stored = str(vars(rows[0]))
    assert "super-secret-value" not in stored
    assert rows[0].landing_path == "/reset"


async def test_a_self_referral_records_no_referrer_host(
    app: FastAPI, anon_client: AsyncClient
) -> None:
    await anon_client.post(VISIT, json={"path": "/plans", "referrer": "https://boorda.kr/"})

    rows = await _sessions(app)
    assert rows[0].referrer_host is None
    assert rows[0].is_direct is True


async def test_admin_navigation_is_not_acquisition(app: FastAPI, anon_client: AsyncClient) -> None:
    """Operator navigation and asset requests are people already here."""
    for path in ("/admin", "/admin/users", "/ops/training", "/api/v1/plans", "/_next/static/x.js"):
        await anon_client.post(VISIT, json={"path": path})

    assert await _sessions(app) == []


async def test_the_request_has_nowhere_to_put_a_user_id(anon_client: AsyncClient) -> None:
    """A visit says which browser. It can never say which account."""
    response = await anon_client.post(VISIT, json={"path": "/", "user_id": str(uuid.uuid4())})

    assert response.status_code == 422


async def test_an_unknown_referrer_is_classified_as_a_referral(
    app: FastAPI, anon_client: AsyncClient
) -> None:
    await anon_client.post(VISIT, json={"path": "/", "referrer": "https://some-blog.example/x"})

    rows = await _sessions(app)
    assert (rows[0].source, rows[0].medium) == ("some-blog.example", "referral")


# ── signup binding ───────────────────────────────────────────────────


async def test_signup_inherits_the_browser_attribution(
    app: FastAPI, anon_client: AsyncClient
) -> None:
    await anon_client.post(
        VISIT,
        json={
            "path": "/",
            "params": {
                "utm_source": "instagram",
                "utm_medium": "paid_social",
                "utm_campaign": "summer_launch",
            },
        },
    )

    created = await anon_client.post(
        "/v1/auth/signup",
        json={"email": "attributed@example.com", "password": "correct horse battery staple"},
    )
    assert created.status_code == 201

    factory = app.state.session_factory
    async with factory() as session:
        row = await session.get(AcquisitionAttribution, uuid.UUID(created.json()["id"]))
    assert row is not None
    assert row.first_source == "instagram"
    assert row.first_campaign == "summer_launch"


async def test_a_signup_without_a_visit_still_succeeds(anon_client: AsyncClient) -> None:
    """Anyone blocking cookies must still be able to create an account."""
    response = await anon_client.post(
        "/v1/auth/signup",
        json={"email": "no-cookie@example.com", "password": "correct horse battery staple"},
    )

    assert response.status_code == 201


async def test_a_forged_cookie_attributes_nothing(app: FastAPI, anon_client: AsyncClient) -> None:
    """A cookie naming a visitor we have never seen binds nothing —
    and cannot name somebody else's account, because the account id
    comes from the row the server just created."""
    anon_client.cookies.set(VISITOR_COOKIE_NAME, str(uuid.uuid4()))

    created = await anon_client.post(
        "/v1/auth/signup",
        json={"email": "forged@example.com", "password": "correct horse battery staple"},
    )
    assert created.status_code == 201

    factory = app.state.session_factory
    async with factory() as session:
        assert await session.get(AcquisitionAttribution, uuid.UUID(created.json()["id"])) is None


# ── admin reporting ──────────────────────────────────────────────────

ACQUISITION_PATHS = [
    "/v1/admin/acquisition/summary",
    "/v1/admin/acquisition/channels",
    "/v1/admin/acquisition/campaigns",
]


async def test_acquisition_reporting_refuses_anonymous_callers(
    anon_client: AsyncClient,
) -> None:
    for path in ACQUISITION_PATHS:
        assert (await anon_client.get(path)).status_code == 401, path


async def test_acquisition_reporting_refuses_an_ordinary_account(
    plain_client: AsyncClient,
) -> None:
    """No public analytics endpoint. Ingestion is public; reading is not."""
    for path in ACQUISITION_PATHS:
        assert (await plain_client.get(path)).status_code == 403, path


async def test_an_admin_may_read_acquisition(admin_client: AsyncClient) -> None:
    for path in ACQUISITION_PATHS:
        assert (await admin_client.get(path)).status_code == 200, path


async def test_the_empty_state_is_zeroes_rather_than_an_error(
    admin_client: AsyncClient,
) -> None:
    body = (await admin_client.get("/v1/admin/acquisition/summary")).json()

    assert body["visitors"] == 0
    assert body["signups"] == 0
    assert body["signup_rate"] is None
    assert (await admin_client.get("/v1/admin/acquisition/channels")).json() == []


async def test_accounts_from_before_this_existed_are_counted_separately(
    admin_client: AsyncClient,
) -> None:
    """Never presented as direct — see the migration note."""
    body = (await admin_client.get("/v1/admin/acquisition/summary")).json()

    assert body["unattributed_users"] >= 1
    assert (await admin_client.get("/v1/admin/acquisition/channels")).json() == []


async def test_the_range_and_mode_are_reported_back(admin_client: AsyncClient) -> None:
    body = (
        await admin_client.get(
            "/v1/admin/acquisition/summary",
            params={"start": "2026-08-01", "end": "2026-08-28", "mode": "last_touch"},
        )
    ).json()

    assert body["range"]["start"] == "2026-08-01"
    assert body["range"]["end"] == "2026-08-28"
    assert body["mode"] == "last_touch"


async def test_an_unknown_attribution_mode_is_refused(admin_client: AsyncClient) -> None:
    response = await admin_client.get("/v1/admin/acquisition/summary", params={"mode": "made_up"})

    assert response.status_code == 422


async def test_the_funnel_reaches_the_console(
    app: FastAPI, anon_client: AsyncClient, admin_client: AsyncClient
) -> None:
    """End to end: a campaign visit, a signup, a verified payment."""
    await anon_client.post(
        VISIT,
        json={
            "path": "/",
            "params": {
                "utm_source": "youtube",
                "utm_medium": "paid_video",
                "utm_campaign": "august_music_ai",
            },
        },
    )
    created = await anon_client.post(
        "/v1/auth/signup",
        json={"email": "converted@example.com", "password": "correct horse battery staple"},
    )
    user_id = uuid.UUID(created.json()["id"])

    # Written directly: the billing path is exercised by the billing
    # suite, and reaching PayApp from here would be a real payment.
    factory = app.state.session_factory
    async with factory() as session:
        await session.execute(
            insert(BillingPayment).values(
                id=uuid.uuid4(),
                user_id=user_id,
                subscription_id=None,
                provider_payment_id=f"acq-{uuid.uuid4().hex[:10]}",
                plan_id="basic",
                amount_krw=19_900,
                status=PAYMENT_SUCCEEDED,
                paid_at=datetime.now(UTC),
                created_at=datetime.now(UTC),
            )
        )
        await session.commit()

    channels = (await admin_client.get("/v1/admin/acquisition/channels")).json()
    row = next(c for c in channels if c["key"] == "youtube_ads")

    assert row["label"] == "YouTube 광고"
    assert (row["visitors"], row["signups"], row["conversions"]) == (1, 1, 1)
    assert row["revenue_krw"] == 19_900
    assert row["conversion_rate"] == 1.0


async def test_campaign_rows_carry_the_campaign_name(
    anon_client: AsyncClient, admin_client: AsyncClient
) -> None:
    await anon_client.post(
        VISIT,
        json={
            "path": "/",
            "params": {
                "utm_source": "instagram",
                "utm_medium": "paid_social",
                "utm_campaign": "creator_campaign_01",
            },
        },
    )

    rows = (
        await admin_client.get("/v1/admin/acquisition/campaigns", params={"mode": "last_touch"})
    ).json()

    assert any(r["campaign"] == "creator_campaign_01" for r in rows)


async def test_no_response_carries_customer_identifying_data(
    anon_client: AsyncClient, admin_client: AsyncClient
) -> None:
    """Channel and campaign rows are aggregates. No addresses, no ids."""
    await anon_client.post(VISIT, json={"path": "/", "params": {"utm_source": "google"}})

    for path in ACQUISITION_PATHS:
        body = str((await admin_client.get(path)).json()).lower()
        for forbidden in ("@", "email", "password", "user_id", "visitor_key", "token"):
            assert forbidden not in body, (path, forbidden)
