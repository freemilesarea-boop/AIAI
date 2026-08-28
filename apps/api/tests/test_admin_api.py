"""The operator console at the HTTP boundary.

The permission tests are the ones that matter, and they are written from
the attacker's side: not "does the console hide the admin nav from a
customer" but "can a customer's session reach an admin endpoint at all".
Every route is asked, by name, whether an ordinary account gets in —
because the failure mode this console has is one route added later
without the dependency, and a test that checks a representative sample
would not catch it.

The second theme is that nothing here can destroy anything. The console
has no route that deletes an account, cancels a subscription, refunds a
payment or charges a card, and there are tests asserting those routes do
not exist. That is a strange thing to test until you consider that the
way they would appear is somebody adding one.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from admin_fixtures import set_role, signed_up_client
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy import insert, update

from luber_api.routes.admin import MAX_RANGE_DAYS, NO_MAIL_PROVIDER, resolve_range
from luber_database.models.payments import PAYMENT_SUCCEEDED, BillingPayment
from luber_database.models.user import User
from luber_schemas.enums import UserRole
from luber_schemas.plans import PlanId

#: Every admin route, as a caller would reach it. Kept as one list so a
#: route added later is either in here or visibly missing from it.
ADMIN_GET_PATHS = [
    "/v1/admin/dashboard",
    "/v1/admin/analytics/revenue",
    "/v1/admin/analytics/generations",
    "/v1/admin/analytics/downloads",
    "/v1/admin/analytics/plans",
    "/v1/admin/analytics/users",
    "/v1/admin/users",
    "/v1/admin/support",
    "/v1/admin/email/campaigns",
    "/v1/admin/audit",
]

SUPER_ADMIN_GET_PATHS = ["/v1/admin/admins"]


# ── who may get in ───────────────────────────────────────────────────


@pytest.mark.parametrize("path", ADMIN_GET_PATHS + SUPER_ADMIN_GET_PATHS)
async def test_anonymous_callers_are_refused(anon_client: AsyncClient, path: str) -> None:
    assert (await anon_client.get(path)).status_code == 401


@pytest.mark.parametrize("path", ADMIN_GET_PATHS + SUPER_ADMIN_GET_PATHS)
async def test_an_ordinary_account_is_refused(plain_client: AsyncClient, path: str) -> None:
    """403, not 404.

    They are authenticated and the route exists; saying so leaks
    nothing, and a 404 here would only make a real operator's
    misconfiguration harder to diagnose.
    """
    assert (await plain_client.get(path)).status_code == 403


@pytest.mark.parametrize("path", ADMIN_GET_PATHS)
async def test_an_admin_is_admitted(admin_client: AsyncClient, path: str) -> None:
    assert (await admin_client.get(path)).status_code == 200


@pytest.mark.parametrize("path", SUPER_ADMIN_GET_PATHS)
async def test_an_admin_cannot_reach_the_super_admin_routes(
    admin_client: AsyncClient, path: str
) -> None:
    assert (await admin_client.get(path)).status_code == 403


@pytest.mark.parametrize("path", ADMIN_GET_PATHS + SUPER_ADMIN_GET_PATHS)
async def test_a_super_admin_is_admitted_everywhere(
    super_admin_client: AsyncClient, path: str
) -> None:
    assert (await super_admin_client.get(path)).status_code == 200


def _walk(routes: object) -> list[object]:
    """Every route, including those inside included routers.

    This FastAPI version keeps a wrapper object per `include_router`
    call, and the wrapper has no `path` of its own — iterating
    `app.routes` alone finds none of the real routes and would make any
    assertion below pass vacuously.
    """
    found: list[object] = []
    for route in routes:  # type: ignore[attr-defined]
        if getattr(route, "path", None):
            found.append(route)
        inner = getattr(route, "original_router", None)
        nested = getattr(inner, "routes", None) or getattr(route, "routes", None)
        if nested:
            found.extend(_walk(nested))
    return found


def _guards(route: object) -> set[str]:
    """Names of every dependency in a route's tree."""
    dependant = getattr(route, "dependant", None)
    if dependant is None:
        return set()

    names: set[str] = set()
    stack = list(dependant.dependencies)
    while stack:
        current = stack.pop()
        call = getattr(current, "call", None)
        if call is not None:
            names.add(getattr(call, "__name__", ""))
        stack.extend(current.dependencies)
    return names


ADMIN_DEPENDENCIES = {
    "require_admin",
    "require_super_admin",
    "get_admin_repository",
    "get_super_admin_repository",
}


async def test_every_admin_route_is_behind_the_dependency(app: FastAPI) -> None:
    """No route under /v1/admin escapes the permission check.

    A structural assertion rather than a behavioural one: the list above
    can go stale, and this cannot. It fails the moment someone adds a
    route to the module without a guard — which is the only way this
    console realistically springs a leak.
    """
    admin_routes = [r for r in _walk(app.routes) if r.path.startswith("/v1/admin")]  # type: ignore[attr-defined]

    for route in admin_routes:
        assert ADMIN_DEPENDENCIES & _guards(route), (
            f"{route.path} is not behind an admin dependency"  # type: ignore[attr-defined]
        )

    # The scan must have seen real routes; one that found nothing would
    # satisfy the loop above without proving anything.
    paths = {r.path for r in admin_routes}  # type: ignore[attr-defined]
    assert paths >= set(ADMIN_GET_PATHS + SUPER_ADMIN_GET_PATHS)


async def test_a_forged_role_in_the_body_changes_nothing(plain_client: AsyncClient) -> None:
    """The session is the only source of identity.

    A customer claiming to be an administrator in the request body is
    still a customer: nothing in the permission path reads the body.
    """
    response = await plain_client.patch(
        "/v1/admin/admins",
        json={"user_id": str(uuid.uuid4()), "role": "SUPER_ADMIN"},
    )

    assert response.status_code == 403


async def test_a_forged_user_header_changes_nothing(plain_client: AsyncClient) -> None:
    response = await plain_client.get(
        "/v1/admin/dashboard", headers={"X-User-Id": str(uuid.uuid4()), "X-Role": "SUPER_ADMIN"}
    )

    assert response.status_code == 403


async def test_an_unrecognised_role_is_not_an_administrator(
    app: FastAPI, plain_client: AsyncClient
) -> None:
    """A role nobody recognises defaults to the least privilege.

    The column is a string. If one ever holds `SUPERADMIN` — a typo in a
    hand-run SQL statement — it must not be treated as anything.
    """
    factory = app.state.session_factory
    async with factory() as session:
        await session.execute(
            update(User)
            .where(User.id == uuid.UUID(plain_client.user_id))  # type: ignore[attr-defined]
            .values(role="SUPERADMIN")
        )
        await session.commit()

    assert (await plain_client.get("/v1/admin/dashboard")).status_code == 403


# ── the console cannot destroy anything ──────────────────────────────


@pytest.mark.parametrize(
    "method,path",
    [
        ("delete", "/v1/admin/users/{uid}"),
        ("post", "/v1/admin/users/{uid}/refund"),
        ("post", "/v1/admin/users/{uid}/cancel-subscription"),
        ("post", "/v1/admin/billing/charge"),
        ("post", "/v1/admin/users/{uid}/impersonate"),
    ],
)
async def test_destructive_routes_do_not_exist(
    super_admin_client: AsyncClient, method: str, path: str
) -> None:
    """The console has no route that deletes, refunds, charges or
    impersonates.

    Deleting an account, cancelling a recurring payment and issuing a
    refund all have real consequences for a real customer, and each one
    belongs to a path with its own confirmation — not to a button on a
    dashboard. `DELETE /v1/admin/admins/{id}` revokes a *role*, which is
    a different thing and is tested below.
    """
    response = await super_admin_client.request(
        method.upper(), path.format(uid=str(uuid.uuid4())), json={}
    )

    assert response.status_code in {404, 405}


# ── roles ────────────────────────────────────────────────────────────


async def test_the_role_endpoints_do_not_claim_a_plan(
    super_admin_client: AsyncClient, admin_client: AsyncClient
) -> None:
    """Null, not "free".

    These endpoints never join to a subscription, and an administrator
    who pays for Basic must not be described as being on Free — a wrong
    answer nothing downstream could detect.
    """
    listed = (await super_admin_client.get("/v1/admin/admins")).json()

    assert listed, "fixture must produce at least one administrator"
    assert all(row["plan_id"] is None for row in listed)


async def test_the_member_list_does_resolve_a_plan(
    admin_client: AsyncClient, plain_client: AsyncClient
) -> None:
    """Where a plan is claimed, it was actually looked up."""
    body = (await admin_client.get("/v1/admin/users", params={"search": "customer@"})).json()

    assert body["items"][0]["plan_id"] == PlanId.FREE.value


async def test_granting_promotes_an_existing_account(
    app: FastAPI, super_admin_client: AsyncClient
) -> None:
    target = await signed_up_client(app, "promote-me@example.com")

    response = await super_admin_client.post(
        "/v1/admin/admins", json={"email": "promote-me@example.com", "role": "ADMIN"}
    )

    assert response.status_code == 201
    assert response.json()["role"] == "ADMIN"
    assert response.json()["id"] == target.user_id  # type: ignore[attr-defined]
    await target.aclose()


async def test_granting_does_not_create_accounts(super_admin_client: AsyncClient) -> None:
    """An operator who mistypes an address gets a 404, not a new user."""
    response = await super_admin_client.post(
        "/v1/admin/admins", json={"email": "nobody@example.com", "role": "ADMIN"}
    )

    assert response.status_code == 404


async def test_a_promoted_account_can_then_reach_the_console(
    app: FastAPI, super_admin_client: AsyncClient
) -> None:
    """The grant is what admits them, and it takes effect immediately."""
    target = await signed_up_client(app, "newly-admin@example.com")
    assert (await target.get("/v1/admin/dashboard")).status_code == 403

    await super_admin_client.post(
        "/v1/admin/admins", json={"email": "newly-admin@example.com", "role": "ADMIN"}
    )

    assert (await target.get("/v1/admin/dashboard")).status_code == 200
    await target.aclose()


async def test_revoking_returns_an_account_to_user(
    app: FastAPI, super_admin_client: AsyncClient, admin_client: AsyncClient
) -> None:
    response = await super_admin_client.delete(
        f"/v1/admin/admins/{admin_client.user_id}"  # type: ignore[attr-defined]
    )

    assert response.status_code == 200
    assert response.json()["role"] == "USER"
    assert (await admin_client.get("/v1/admin/dashboard")).status_code == 403


async def test_revoking_a_role_does_not_delete_the_account(
    super_admin_client: AsyncClient, admin_client: AsyncClient
) -> None:
    """The person still exists; they simply are not an operator."""
    await super_admin_client.delete(
        f"/v1/admin/admins/{admin_client.user_id}"  # type: ignore[attr-defined]
    )

    # Still a working session on a working account.
    assert (await admin_client.get("/v1/auth/me")).status_code == 200


async def test_the_last_super_admin_cannot_be_demoted(
    super_admin_client: AsyncClient,
) -> None:
    """Locking yourself out is locking everyone out.

    The recovery from an empty console is a hand-run migration by
    whoever still has database access, so the write is refused rather
    than merely discouraged.
    """
    response = await super_admin_client.patch(
        "/v1/admin/admins",
        json={"user_id": super_admin_client.user_id, "role": "USER"},  # type: ignore[attr-defined]
    )

    assert response.status_code == 409
    assert (await super_admin_client.get("/v1/admin/admins")).status_code == 200


async def test_the_last_super_admin_cannot_be_revoked_either(
    super_admin_client: AsyncClient,
) -> None:
    """The DELETE route goes through the same guard as the PATCH."""
    response = await super_admin_client.delete(
        f"/v1/admin/admins/{super_admin_client.user_id}"  # type: ignore[attr-defined]
    )

    assert response.status_code == 409


async def test_a_super_admin_may_step_down_when_another_remains(
    super_admin_client: AsyncClient, second_super_admin_client: AsyncClient
) -> None:
    """Self-demotion is legitimate — someone handing over should be able
    to. What is refused is being the last one."""
    response = await super_admin_client.patch(
        "/v1/admin/admins",
        json={"user_id": super_admin_client.user_id, "role": "ADMIN"},  # type: ignore[attr-defined]
    )

    assert response.status_code == 200
    assert response.json()["role"] == "ADMIN"
    assert (await second_super_admin_client.get("/v1/admin/admins")).status_code == 200


async def test_an_admin_cannot_promote_themselves(admin_client: AsyncClient) -> None:
    """The reason `ADMIN` and `SUPER_ADMIN` are separate roles."""
    response = await admin_client.patch(
        "/v1/admin/admins",
        json={"user_id": admin_client.user_id, "role": "SUPER_ADMIN"},  # type: ignore[attr-defined]
    )

    assert response.status_code == 403


async def test_changing_an_unknown_account_is_404(super_admin_client: AsyncClient) -> None:
    response = await super_admin_client.patch(
        "/v1/admin/admins", json={"user_id": str(uuid.uuid4()), "role": "ADMIN"}
    )

    assert response.status_code == 404


# ── audit ────────────────────────────────────────────────────────────


async def test_a_role_change_is_recorded(
    app: FastAPI, super_admin_client: AsyncClient, admin_client: AsyncClient
) -> None:
    await super_admin_client.patch(
        "/v1/admin/admins",
        json={"user_id": admin_client.user_id, "role": "USER"},  # type: ignore[attr-defined]
    )

    body = (await super_admin_client.get("/v1/admin/audit")).json()
    entry = next(e for e in body["items"] if e["action"] == "ADMIN_REVOKED")

    assert entry["actor_email"] == "super@example.com"
    assert entry["target_email"] == "admin@example.com"
    assert entry["metadata"] == {"from": "ADMIN", "to": "USER"}


async def test_a_refused_change_leaves_no_audit_entry(
    super_admin_client: AsyncClient,
) -> None:
    """The audit row and the change share a transaction.

    A log claiming an action that did not happen is worse than no log.
    """
    before = (await super_admin_client.get("/v1/admin/audit")).json()["total"]

    await super_admin_client.delete(
        f"/v1/admin/admins/{super_admin_client.user_id}"  # type: ignore[attr-defined]
    )

    after = (await super_admin_client.get("/v1/admin/audit")).json()["total"]
    assert after == before


async def test_the_audit_total_respects_the_filter(
    super_admin_client: AsyncClient, admin_client: AsyncClient
) -> None:
    """The count answers the same question as the rows beneath it."""
    await super_admin_client.patch(
        "/v1/admin/admins",
        json={"user_id": admin_client.user_id, "role": "USER"},  # type: ignore[attr-defined]
    )

    filtered = (
        await super_admin_client.get("/v1/admin/audit", params={"action": "ADMIN_GRANTED"})
    ).json()

    assert filtered["total"] == len(filtered["items"]) == 0


async def test_an_ordinary_admin_may_read_the_audit_log(
    admin_client: AsyncClient,
) -> None:
    """A log only some operators can read is a weaker deterrent."""
    assert (await admin_client.get("/v1/admin/audit")).status_code == 200


async def test_there_is_no_route_that_writes_the_audit_log(
    super_admin_client: AsyncClient,
) -> None:
    for method in ("POST", "PATCH", "PUT", "DELETE"):
        response = await super_admin_client.request(method, "/v1/admin/audit", json={})
        assert response.status_code == 405


# ── analytics ────────────────────────────────────────────────────────


async def _pay(app: FastAPI, user_id: str, amount: int, paid_at: datetime) -> None:
    """Write a successful payment directly.

    The billing path is exercised by the billing suite; here the payment
    is a fixture for the aggregation, and going through PayApp to make
    one would be testing the wrong module.
    """
    factory = app.state.session_factory
    async with factory() as session:
        await session.execute(
            insert(BillingPayment).values(
                id=uuid.uuid4(),
                user_id=uuid.UUID(user_id),
                subscription_id=None,
                provider_payment_id=f"test-{uuid.uuid4().hex[:12]}",
                plan_id=PlanId.BASIC.value,
                amount_krw=amount,
                status=PAYMENT_SUCCEEDED,
                paid_at=paid_at,
                created_at=paid_at,
            )
        )
        await session.commit()


async def test_revenue_counts_only_successful_payments(
    app: FastAPI, admin_client: AsyncClient
) -> None:
    now = datetime.now(UTC)
    await _pay(app, admin_client.user_id, 19_900, now)  # type: ignore[attr-defined]
    await _pay(app, admin_client.user_id, 29_900, now)  # type: ignore[attr-defined]

    body = (await admin_client.get("/v1/admin/analytics/revenue")).json()

    assert body["total_krw"] == 49_800
    assert body["payment_count"] == 2


async def test_revenue_is_zero_when_nothing_was_paid(admin_client: AsyncClient) -> None:
    """Zero is a correct answer and must render as one."""
    body = (await admin_client.get("/v1/admin/analytics/revenue")).json()

    assert body["total_krw"] == 0
    assert body["series"] == []


async def test_a_payment_in_the_korean_morning_counts_as_today(
    app: FastAPI, admin_client: AsyncClient
) -> None:
    """08:00 KST is 23:00 UTC the previous day.

    Bucketing on the raw UTC date would file a Korean morning's revenue
    under yesterday, and the operator reconciling against a bank
    statement would find them a day apart with no visible reason.
    """
    # A moment that is unambiguously "today" in Korea and "yesterday" in
    # UTC, positioned far enough inside the month that the month window
    # contains it regardless of when the suite runs.
    kst_today = (datetime.now(UTC) + timedelta(hours=9)).date()
    if kst_today.day < 3:
        pytest.skip("month boundary; the dedicated bucketing tests cover this")
    morning_kst = datetime(kst_today.year, kst_today.month, kst_today.day, 8, tzinfo=UTC)
    instant = morning_kst - timedelta(hours=9)
    assert instant.date() != kst_today, "fixture must straddle the boundary"

    await _pay(app, admin_client.user_id, 19_900, instant)  # type: ignore[attr-defined]

    body = (
        await admin_client.get("/v1/admin/analytics/revenue", params={"granularity": "day"})
    ).json()

    assert body["total_krw"] == 19_900, "a Korean morning belongs to the Korean day"
    assert body["series"][0]["day"] == kst_today.isoformat()


async def test_plan_distribution_covers_every_tier(admin_client: AsyncClient) -> None:
    body = (await admin_client.get("/v1/admin/analytics/plans")).json()

    assert [row["plan_id"] for row in body["distribution"]] == [
        PlanId.FREE.value,
        PlanId.BASIC.value,
        PlanId.PRO.value,
        PlanId.CREATOR.value,
    ]
    assert sum(row["count"] for row in body["distribution"]) == body["users"]["total"]


async def test_generation_analytics_are_zero_while_generation_is_off(
    admin_client: AsyncClient,
) -> None:
    body = (await admin_client.get("/v1/admin/analytics/generations")).json()

    assert body["totals"]["requested"] == 0
    assert body["series"] == []


async def test_the_dashboard_answers_in_one_request(admin_client: AsyncClient) -> None:
    body = (await admin_client.get("/v1/admin/dashboard")).json()

    for key in (
        "revenue_krw",
        "users",
        "generations",
        "downloads",
        "support",
        "plans",
        "revenue_series",
    ):
        assert key in body, key


async def test_a_backwards_range_is_refused(admin_client: AsyncClient) -> None:
    response = await admin_client.get(
        "/v1/admin/analytics/revenue", params={"start": "2026-08-20", "end": "2026-08-01"}
    )

    assert response.status_code == 422


async def test_an_unbounded_range_is_refused(admin_client: AsyncClient) -> None:
    """A chart is 365 points. Ten years is a scan nobody budgeted for."""
    response = await admin_client.get(
        "/v1/admin/analytics/revenue", params={"start": "2016-01-01", "end": "2026-01-01"}
    )

    assert response.status_code == 422


async def test_a_malformed_date_is_refused(admin_client: AsyncClient) -> None:
    response = await admin_client.get("/v1/admin/analytics/revenue", params={"start": "yesterday"})

    assert response.status_code == 422


async def test_a_custom_range_drives_every_figure(app: FastAPI, admin_client: AsyncClient) -> None:
    """The range the operator picked is the range the numbers describe."""
    inside = datetime(2026, 8, 10, 3, tzinfo=UTC)
    outside = datetime(2026, 8, 25, 3, tzinfo=UTC)
    await _pay(app, admin_client.user_id, 19_900, inside)  # type: ignore[attr-defined]
    await _pay(app, admin_client.user_id, 29_900, outside)  # type: ignore[attr-defined]

    body = (
        await admin_client.get(
            "/v1/admin/analytics/revenue", params={"start": "2026-08-01", "end": "2026-08-15"}
        )
    ).json()

    assert body["total_krw"] == 19_900, "the payment outside the window must not count"
    assert body["range"] == {
        "start": "2026-08-01",
        "end": "2026-08-15",
        "days": 15,
        "bucketing": "day",
    }


async def test_a_custom_range_survives_on_the_dashboard_too(
    app: FastAPI, admin_client: AsyncClient
) -> None:
    await _pay(app, admin_client.user_id, 19_900, datetime(2026, 8, 10, 3, tzinfo=UTC))  # type: ignore[attr-defined]

    body = (
        await admin_client.get(
            "/v1/admin/dashboard", params={"start": "2026-08-01", "end": "2026-08-15"}
        )
    ).json()

    assert body["revenue_krw"] == 19_900
    assert body["range"]["start"] == "2026-08-01"
    assert body["range"]["end"] == "2026-08-15"


async def test_a_single_day_range_is_one_day_not_zero(admin_client: AsyncClient) -> None:
    body = (
        await admin_client.get(
            "/v1/admin/analytics/revenue", params={"start": "2026-08-28", "end": "2026-08-28"}
        )
    ).json()

    assert body["range"]["days"] == 1


async def test_the_bucket_size_is_chosen_from_the_range(admin_client: AsyncClient) -> None:
    """Deterministic thresholds, reported so the console can label the
    axis rather than guess at it."""
    cases = [
        ("2026-08-01", "2026-08-28", "day"),
        ("2026-06-01", "2026-08-28", "week"),
        ("2025-09-01", "2026-08-28", "month"),
    ]
    for start, end, expected in cases:
        body = (
            await admin_client.get(
                "/v1/admin/analytics/revenue", params={"start": start, "end": end}
            )
        ).json()
        assert body["range"]["bucketing"] == expected, (start, end)


async def test_an_explicit_bucket_overrides_the_default(admin_client: AsyncClient) -> None:
    body = (
        await admin_client.get(
            "/v1/admin/analytics/revenue",
            params={"start": "2026-08-01", "end": "2026-08-07", "bucket": "month"},
        )
    ).json()

    assert body["range"]["bucketing"] == "month"


async def test_a_future_range_is_empty_rather_than_an_error(
    admin_client: AsyncClient,
) -> None:
    """An operator checking next month should get an empty chart."""
    response = await admin_client.get(
        "/v1/admin/analytics/revenue", params={"start": "2099-01-01", "end": "2099-01-31"}
    )

    assert response.status_code == 200
    assert response.json()["total_krw"] == 0
    assert response.json()["series"] == []


# ── previous-period comparison ───────────────────────────────────────


async def test_the_comparison_window_is_the_range_immediately_before(
    admin_client: AsyncClient,
) -> None:
    body = (
        await admin_client.get(
            "/v1/admin/dashboard", params={"start": "2026-08-15", "end": "2026-08-28"}
        )
    ).json()

    assert body["comparison"]["start"] == "2026-08-01"
    assert body["comparison"]["end"] == "2026-08-14"


async def test_a_percentage_is_reported_against_the_previous_period(
    app: FastAPI, admin_client: AsyncClient
) -> None:
    # ₩10,000 in the previous window, ₩15,000 in the selected one: +50%.
    await _pay(app, admin_client.user_id, 10_000, datetime(2026, 8, 5, 3, tzinfo=UTC))  # type: ignore[attr-defined]
    await _pay(app, admin_client.user_id, 15_000, datetime(2026, 8, 20, 3, tzinfo=UTC))  # type: ignore[attr-defined]

    body = (
        await admin_client.get(
            "/v1/admin/dashboard", params={"start": "2026-08-15", "end": "2026-08-28"}
        )
    ).json()

    assert body["revenue_krw"] == 15_000
    assert body["comparison"]["revenue_krw"] == 10_000
    assert body["comparison"]["revenue_delta_pct"] == 50.0


async def test_a_fall_is_reported_as_a_negative_percentage(
    app: FastAPI, admin_client: AsyncClient
) -> None:
    await _pay(app, admin_client.user_id, 20_000, datetime(2026, 8, 5, 3, tzinfo=UTC))  # type: ignore[attr-defined]
    await _pay(app, admin_client.user_id, 15_000, datetime(2026, 8, 20, 3, tzinfo=UTC))  # type: ignore[attr-defined]

    body = (
        await admin_client.get(
            "/v1/admin/dashboard", params={"start": "2026-08-15", "end": "2026-08-28"}
        )
    ).json()

    assert body["comparison"]["revenue_delta_pct"] == -25.0


async def test_a_zero_base_reports_no_percentage_at_all(
    app: FastAPI, admin_client: AsyncClient
) -> None:
    """There is no honest percentage from nothing.

    The change from ₩0 is undefined — not infinite, not 100% — and a
    number here is one an operator might act on.
    """
    await _pay(app, admin_client.user_id, 19_900, datetime(2026, 8, 20, 3, tzinfo=UTC))  # type: ignore[attr-defined]

    body = (
        await admin_client.get(
            "/v1/admin/dashboard", params={"start": "2026-08-15", "end": "2026-08-28"}
        )
    ).json()

    assert body["comparison"]["revenue_krw"] == 0
    assert body["comparison"]["revenue_delta_pct"] is None


async def test_comparison_is_present_on_the_revenue_endpoint_too(
    admin_client: AsyncClient,
) -> None:
    body = (await admin_client.get("/v1/admin/analytics/revenue")).json()

    assert body["comparison"] is not None
    assert "revenue_delta_pct" in body["comparison"]


def test_resolve_range_reports_its_own_length_and_bucketing() -> None:
    window = resolve_range(None, "2026-08-01", "2026-08-28")

    assert window.days == 28
    assert window.bucketing == "day"


def test_resolve_range_covers_the_whole_korean_day() -> None:
    """The window is half-open on UTC instants, nine hours shifted."""
    window = resolve_range(None, "2026-08-28", "2026-08-28")

    assert window.start.isoformat() == "2026-08-27T15:00:00+00:00"
    assert window.end.isoformat() == "2026-08-28T15:00:00+00:00"


def test_the_range_ceiling_is_a_year() -> None:
    assert MAX_RANGE_DAYS == 366, "a leap year of daily buckets"


# ── users ────────────────────────────────────────────────────────────


async def test_the_user_list_never_carries_a_password_hash(
    admin_client: AsyncClient, plain_client: AsyncClient
) -> None:
    """The console is exactly where this mistake would matter most."""
    body = (await admin_client.get("/v1/admin/users")).json()

    assert body["total"] >= 2
    serialised = str(body)
    assert "password" not in serialised.lower()
    assert "hash" not in serialised.lower()


async def test_users_can_be_searched_by_address(
    admin_client: AsyncClient, plain_client: AsyncClient
) -> None:
    body = (await admin_client.get("/v1/admin/users", params={"search": "customer@"})).json()

    assert [row["email"] for row in body["items"]] == ["customer@example.com"]


async def test_the_user_detail_shows_activity_counts_not_content(
    admin_client: AsyncClient, plain_client: AsyncClient
) -> None:
    body = (
        await admin_client.get(f"/v1/admin/users/{plain_client.user_id}")  # type: ignore[attr-defined]
    ).json()

    assert body["user"]["email"] == "customer@example.com"
    assert body["activity"] == {
        "generations": 0,
        "completed": 0,
        "downloads": 0,
        "payments": 0,
    }
    # Counts, not the songs themselves.
    assert "generations_list" not in body


async def test_an_unknown_account_is_404(admin_client: AsyncClient) -> None:
    response = await admin_client.get(f"/v1/admin/users/{uuid.uuid4()}")

    assert response.status_code == 404


async def test_a_closed_account_is_not_listed(
    app: FastAPI, admin_client: AsyncClient, plain_client: AsyncClient
) -> None:
    """Closing anonymises. A closed account is not a member."""
    factory = app.state.session_factory
    async with factory() as session:
        await session.execute(
            update(User)
            .where(User.id == uuid.UUID(plain_client.user_id))  # type: ignore[attr-defined]
            .values(deleted_at=datetime.now(UTC))
        )
        await session.commit()

    body = (await admin_client.get("/v1/admin/users")).json()

    assert "customer@example.com" not in [row["email"] for row in body["items"]]


# ── support ──────────────────────────────────────────────────────────


async def _file_ticket(client: AsyncClient) -> str:
    response = await client.post(
        "/v1/support/inquiries",
        json={
            "category": "BILLING",
            "subject": "결제 확인 요청",
            "message": "8월 결제가 정상 처리되었는지 확인 부탁드립니다.",
        },
    )
    assert response.status_code == 201, response.text
    return str(response.json()["reference"])


async def test_an_operator_sees_a_customers_ticket(
    admin_client: AsyncClient, plain_client: AsyncClient
) -> None:
    """An administrator reads other people's tickets by definition —
    that is the job, and the owner-scoping that protects customers from
    each other is deliberately not applied here."""
    reference = await _file_ticket(plain_client)

    body = (await admin_client.get(f"/v1/admin/support/{reference}")).json()

    assert body["reference"] == reference
    assert body["user_email"] == "customer@example.com"


async def test_moving_a_ticket_is_recorded(
    admin_client: AsyncClient, plain_client: AsyncClient
) -> None:
    reference = await _file_ticket(plain_client)

    response = await admin_client.patch(
        f"/v1/admin/support/{reference}", json={"status": "RESOLVED"}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "RESOLVED"
    assert response.json()["resolved_at"] is not None

    audit = (await admin_client.get("/v1/admin/audit")).json()
    assert any(e["action"] == "SUPPORT_STATUS_CHANGED" for e in audit["items"])


async def test_an_internal_note_is_never_shown_to_the_customer(
    admin_client: AsyncClient, plain_client: AsyncClient
) -> None:
    """The note lives on the ticket precisely because the reply thread
    is what the customer will eventually be shown."""
    reference = await _file_ticket(plain_client)

    await admin_client.patch(
        f"/v1/admin/support/{reference}", json={"admin_note": "환불 정책 확인 필요"}
    )

    customer_view = (await plain_client.get(f"/v1/support/inquiries/{reference}")).json()

    assert "admin_note" not in customer_view
    assert "환불" not in str(customer_view)


async def test_the_note_text_is_not_copied_into_the_audit_log(
    admin_client: AsyncClient, plain_client: AsyncClient
) -> None:
    """An audit log records actions, not a second copy of the content."""
    reference = await _file_ticket(plain_client)
    await admin_client.patch(
        f"/v1/admin/support/{reference}", json={"admin_note": "카드사 승인번호 확인"}
    )

    audit = (await admin_client.get("/v1/admin/audit")).json()

    assert any(e["action"] == "SUPPORT_NOTE_ADDED" for e in audit["items"])
    assert "승인번호" not in str(audit)


async def test_an_operator_cannot_rewrite_what_a_customer_wrote(
    admin_client: AsyncClient, plain_client: AsyncClient
) -> None:
    """A support record the operator can edit is not a record."""
    reference = await _file_ticket(plain_client)

    response = await admin_client.patch(
        f"/v1/admin/support/{reference}", json={"message": "제가 잘못 봤습니다"}
    )

    assert response.status_code == 422


async def test_an_empty_ticket_update_is_refused(
    admin_client: AsyncClient, plain_client: AsyncClient
) -> None:
    reference = await _file_ticket(plain_client)

    assert (await admin_client.patch(f"/v1/admin/support/{reference}", json={})).status_code == 422


async def test_an_unknown_ticket_is_404(admin_client: AsyncClient) -> None:
    assert (await admin_client.get("/v1/admin/support/SUP-DEADBEEF")).status_code == 404


# ── email campaigns ──────────────────────────────────────────────────


CAMPAIGN = {
    "subject": "BOORDA 8월 업데이트",
    "body": "새로운 기능을 소개합니다.",
    "audience_type": "ALL",
}


async def test_an_ordinary_account_cannot_compose_a_campaign(
    plain_client: AsyncClient,
) -> None:
    assert (await plain_client.post("/v1/admin/email/campaigns", json=CAMPAIGN)).status_code == 403


async def test_a_campaign_is_saved_as_a_draft_and_nothing_is_sent(
    admin_client: AsyncClient, plain_client: AsyncClient
) -> None:
    """There is no mail provider configured for BOORDA.

    The response says so on every campaign rather than leaving an
    operator to infer it from a status string — someone who believes
    they announced a price change and did not has a worse day than
    someone who was told plainly.
    """
    response = await admin_client.post("/v1/admin/email/campaigns", json=CAMPAIGN)

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "DRAFT"
    assert body["sent_at"] is None
    assert body["delivery_note"] == NO_MAIL_PROVIDER


async def test_the_recipient_count_is_resolved_server_side(
    admin_client: AsyncClient, plain_client: AsyncClient
) -> None:
    """What the operator confirms against must be a server-side fact."""
    response = await admin_client.post("/v1/admin/email/audience", json={"audience_type": "ALL"})

    listed = (await admin_client.get("/v1/admin/users")).json()["total"]
    assert response.json()["recipient_count"] == listed


async def test_the_audience_can_be_counted_before_anything_is_written(
    admin_client: AsyncClient, plain_client: AsyncClient
) -> None:
    """The preview asks about an audience, not about a draft.

    Requiring a subject and a body to ask "how many people is this?"
    would make the console invent both while the operator is still
    deciding what to say.
    """
    response = await admin_client.post(
        "/v1/admin/email/audience", json={"audience_type": "PLAN", "plan_id": "basic"}
    )

    assert response.status_code == 200
    assert response.json()["recipient_count"] == 0


async def test_a_plan_audience_needs_a_plan(admin_client: AsyncClient) -> None:
    response = await admin_client.post(
        "/v1/admin/email/campaigns", json={**CAMPAIGN, "audience_type": "PLAN"}
    )

    assert response.status_code == 422


async def test_a_user_audience_needs_at_least_one_account(
    admin_client: AsyncClient,
) -> None:
    response = await admin_client.post(
        "/v1/admin/email/campaigns", json={**CAMPAIGN, "audience_type": "USERS"}
    )

    assert response.status_code == 422


async def test_an_unknown_audience_is_refused(admin_client: AsyncClient) -> None:
    response = await admin_client.post(
        "/v1/admin/email/campaigns", json={**CAMPAIGN, "audience_type": "EVERYONE_EVER"}
    )

    assert response.status_code == 422


async def test_composing_a_campaign_is_recorded(admin_client: AsyncClient) -> None:
    await admin_client.post("/v1/admin/email/campaigns", json=CAMPAIGN)

    audit = (await admin_client.get("/v1/admin/audit")).json()

    assert any(e["action"] == "EMAIL_CAMPAIGN_CREATED" for e in audit["items"])


async def test_there_is_no_send_route(admin_client: AsyncClient) -> None:
    """A route that pretended to send would be worse than none."""
    created = (await admin_client.post("/v1/admin/email/campaigns", json=CAMPAIGN)).json()

    response = await admin_client.post(f"/v1/admin/email/campaigns/{created['id']}/send", json={})

    assert response.status_code in {404, 405}


# ── download tracking ────────────────────────────────────────────────


async def test_role_changes_survive_a_second_operator(
    app: FastAPI, super_admin_client: AsyncClient
) -> None:
    """Granting twice is idempotent, not an error."""
    await signed_up_client(app, "twice@example.com")

    first = await super_admin_client.post(
        "/v1/admin/admins", json={"email": "twice@example.com", "role": "ADMIN"}
    )
    second = await super_admin_client.post(
        "/v1/admin/admins", json={"email": "twice@example.com", "role": "ADMIN"}
    )

    assert first.status_code == second.status_code == 201
    assert second.json()["role"] == "ADMIN"


async def test_setting_a_role_through_the_grant_route_cannot_demote(
    app: FastAPI, super_admin_client: AsyncClient
) -> None:
    """`POST /admins` grants. Revoking has its own verb, so a mistyped
    role cannot quietly remove someone."""
    await signed_up_client(app, "grant-user@example.com")

    response = await super_admin_client.post(
        "/v1/admin/admins", json={"email": "grant-user@example.com", "role": "USER"}
    )

    assert response.status_code == 422


async def test_set_role_helper_matches_the_api(app: FastAPI) -> None:
    """The fixture writes the same column the bootstrap script does."""
    client = await signed_up_client(app, "helper@example.com")
    await set_role(app, client.user_id, UserRole.ADMIN)  # type: ignore[attr-defined]

    assert (await client.get("/v1/admin/dashboard")).status_code == 200
    await client.aclose()
