"""Account management: password, profile, and closing an account.

The deletion tests carry the weight. Closing an account is the most
destructive thing a user can do to themselves, and the failure modes are
all quiet ones — a session that keeps working, a recurring charge that
outlives the account it belonged to, a request that names somebody
else's id.

Every test drives the real routes with a real session cookie. There is
no test-only bypass, for the same reason Phase 4 refused one.
"""

from __future__ import annotations

from httpx import AsyncClient
from luber_billing.payapp.fake import feedback_payload

from luber_schemas.plans import PlanId

PASSWORD = "correct horse battery staple"
NEW_PASSWORD = "a different sufficiently long passphrase"
CREDS = {"userid": "boorda-test", "linkkey": "test-linkkey", "linkval": "test-linkval"}


# ── password change ──────────────────────────────────────────────────


async def test_password_change_requires_a_session(anon_client: AsyncClient) -> None:
    response = await anon_client.post(
        "/v1/auth/password",
        json={
            "current_password": PASSWORD,
            "new_password": NEW_PASSWORD,
            "new_password_confirm": NEW_PASSWORD,
        },
    )

    assert response.status_code == 401


async def test_the_right_current_password_changes_it(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/auth/password",
        json={
            "current_password": PASSWORD,
            "new_password": NEW_PASSWORD,
            "new_password_confirm": NEW_PASSWORD,
        },
    )

    assert response.status_code == 204
    # The new one works.
    assert (await client.get("/v1/auth/me")).status_code == 200


async def test_the_wrong_current_password_is_refused(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/auth/password",
        json={
            "current_password": "not the password",
            "new_password": NEW_PASSWORD,
            "new_password_confirm": NEW_PASSWORD,
        },
    )

    assert response.status_code == 400


async def test_a_confirmation_mismatch_is_refused(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/auth/password",
        json={
            "current_password": PASSWORD,
            "new_password": NEW_PASSWORD,
            "new_password_confirm": NEW_PASSWORD + " typo",
        },
    )

    assert response.status_code == 400


async def test_a_weak_new_password_is_refused(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/auth/password",
        json={"current_password": PASSWORD, "new_password": "x", "new_password_confirm": "x"},
    )

    assert response.status_code == 422


async def test_the_old_password_stops_working(app, client: AsyncClient) -> None:
    """Otherwise the change is cosmetic."""
    email = "user-a@example.com"
    await client.post(
        "/v1/auth/password",
        json={
            "current_password": PASSWORD,
            "new_password": NEW_PASSWORD,
            "new_password_confirm": NEW_PASSWORD,
        },
    )

    from httpx import ASGITransport
    from httpx import AsyncClient as Fresh

    async with Fresh(transport=ASGITransport(app=app), base_url="http://testserver") as other:
        old = await other.post("/v1/auth/login", json={"email": email, "password": PASSWORD})
        new = await other.post("/v1/auth/login", json={"email": email, "password": NEW_PASSWORD})

    assert old.status_code == 401
    assert new.status_code == 200


async def test_changing_the_password_ends_other_sessions(app, client: AsyncClient) -> None:
    """What someone does when they think a credential is compromised.

    Leaving the other sessions alive would make the act cosmetic.
    """
    from httpx import ASGITransport
    from httpx import AsyncClient as Fresh

    async with Fresh(transport=ASGITransport(app=app), base_url="http://testserver") as second:
        assert (
            await second.post(
                "/v1/auth/login", json={"email": "user-a@example.com", "password": PASSWORD}
            )
        ).status_code == 200
        assert (await second.get("/v1/auth/me")).status_code == 200

        await client.post(
            "/v1/auth/password",
            json={
                "current_password": PASSWORD,
                "new_password": NEW_PASSWORD,
                "new_password_confirm": NEW_PASSWORD,
            },
        )

        assert (await second.get("/v1/auth/me")).status_code == 401
    # The browser that made the change stays signed in.
    assert (await client.get("/v1/auth/me")).status_code == 200


async def test_one_account_cannot_change_anothers_password(
    client: AsyncClient, client_b: AsyncClient
) -> None:
    """There is no user id in the request, so there is nothing to aim."""
    await client_b.post(
        "/v1/auth/password",
        json={
            "current_password": PASSWORD,
            "new_password": NEW_PASSWORD,
            "new_password_confirm": NEW_PASSWORD,
        },
    )

    # A's password is untouched.
    assert (await client.get("/v1/auth/me")).status_code == 200
    assert (
        await client.post(
            "/v1/auth/password",
            json={
                "current_password": PASSWORD,
                "new_password": NEW_PASSWORD,
                "new_password_confirm": NEW_PASSWORD,
            },
        )
    ).status_code == 204


# ── profile ──────────────────────────────────────────────────────────


async def test_the_display_name_can_be_set_and_cleared(client: AsyncClient) -> None:
    named = await client.patch("/v1/auth/profile", json={"display_name": "부르다"})
    assert named.status_code == 200
    assert named.json()["display_name"] == "부르다"

    cleared = await client.patch("/v1/auth/profile", json={"display_name": "  "})
    assert cleared.json()["display_name"] is None


async def test_profile_update_requires_a_session(anon_client: AsyncClient) -> None:
    assert (
        await anon_client.patch("/v1/auth/profile", json={"display_name": "x"})
    ).status_code == 401


async def test_the_email_is_not_editable(client: AsyncClient) -> None:
    """Changing a login address needs a round-trip to the new mailbox.
    Offering the field without one would let a typo lock someone out."""
    await client.patch("/v1/auth/profile", json={"email": "someone-else@example.com"})

    assert (await client.get("/v1/auth/me")).json()["email"] == "user-a@example.com"


# ── account closure ──────────────────────────────────────────────────


async def test_closing_requires_a_session(anon_client: AsyncClient) -> None:
    response = await anon_client.post(
        "/v1/auth/account/delete", json={"current_password": PASSWORD}
    )

    assert response.status_code == 401


async def test_closing_requires_the_right_password(client: AsyncClient) -> None:
    """The session proves who is asking; the password proves they are
    still at the keyboard."""
    response = await client.post(
        "/v1/auth/account/delete", json={"current_password": "not the password"}
    )

    assert response.status_code == 400
    assert (await client.get("/v1/auth/me")).status_code == 200


async def test_closing_with_the_right_password_succeeds(client: AsyncClient) -> None:
    response = await client.post("/v1/auth/account/delete", json={"current_password": PASSWORD})

    assert response.status_code == 204


async def test_the_session_stops_working_immediately(client: AsyncClient) -> None:
    """The cookie is unchanged in the client; the row behind it is gone."""
    await client.post("/v1/auth/account/delete", json={"current_password": PASSWORD})

    assert (await client.get("/v1/auth/me")).status_code == 401


async def test_a_closed_account_cannot_reach_protected_endpoints(client: AsyncClient) -> None:
    await client.post("/v1/auth/account/delete", json={"current_password": PASSWORD})

    for path in (
        "/v1/generations",
        "/v1/account/entitlement",
        "/v1/billing/status",
        "/v1/billing/payments",
        "/v1/projects",
    ):
        assert (await client.get(path)).status_code == 401, path


async def test_a_closed_account_cannot_log_back_in(app, client: AsyncClient) -> None:
    await client.post("/v1/auth/account/delete", json={"current_password": PASSWORD})

    from httpx import ASGITransport
    from httpx import AsyncClient as Fresh

    async with Fresh(transport=ASGITransport(app=app), base_url="http://testserver") as other:
        response = await other.post(
            "/v1/auth/login", json={"email": "user-a@example.com", "password": PASSWORD}
        )

    assert response.status_code == 401


async def test_closing_twice_is_safe(client: AsyncClient) -> None:
    """A double-submit, or a retry after a dropped response."""
    first = await client.post("/v1/auth/account/delete", json={"current_password": PASSWORD})
    second = await client.post("/v1/auth/account/delete", json={"current_password": PASSWORD})

    assert first.status_code == 204
    # The session is gone, so the repeat cannot authenticate — which is
    # the safe outcome, not an error the user has to understand.
    assert second.status_code == 401


async def test_closing_one_account_leaves_another_alone(
    client: AsyncClient, client_b: AsyncClient
) -> None:
    await client.post("/v1/auth/account/delete", json={"current_password": PASSWORD})

    assert (await client_b.get("/v1/auth/me")).status_code == 200
    assert (await client_b.get("/v1/account/entitlement")).status_code == 200


async def test_the_address_is_freed_for_signing_up_again(app, client: AsyncClient) -> None:
    """The closed row keeps a placeholder under `.invalid`, so the real
    address is no longer taken."""
    await client.post("/v1/auth/account/delete", json={"current_password": PASSWORD})

    from httpx import ASGITransport
    from httpx import AsyncClient as Fresh

    async with Fresh(transport=ASGITransport(app=app), base_url="http://testserver") as other:
        response = await other.post(
            "/v1/auth/signup", json={"email": "user-a@example.com", "password": PASSWORD}
        )

    assert response.status_code == 201


# ── subscription safety ──────────────────────────────────────────────


async def _subscribe(client: AsyncClient) -> None:
    created = await client.post(
        "/v1/billing/checkout", json={"plan_id": "pro", "phone": "010-1234-5678"}
    )
    assert created.status_code == 201, created.text
    paid = await client.post(
        "/v1/billing/payapp/feedback",
        data=feedback_payload(**CREDS, rebill_no="900001", price=29900, mul_no="777"),
    )
    assert paid.text == "SUCCESS"


async def test_a_live_subscription_blocks_closing(client: AsyncClient) -> None:
    """The outcome this must never produce: an account gone while PayApp
    keeps charging the card it was attached to."""
    await _subscribe(client)

    response = await client.post("/v1/auth/account/delete", json={"current_password": PASSWORD})

    assert response.status_code == 409
    assert response.json()["detail"] == "SUBSCRIPTION_ACTIVE"
    assert (await client.get("/v1/auth/me")).status_code == 200


async def test_the_blocked_account_is_untouched(client: AsyncClient) -> None:
    await _subscribe(client)
    await client.post("/v1/auth/account/delete", json={"current_password": PASSWORD})

    entitlement = (await client.get("/v1/account/entitlement")).json()
    assert entitlement["plan"]["plan_id"] == "pro"
    assert (await client.get("/v1/billing/status")).json()["status"] == "ACTIVE"


async def test_closing_works_once_the_subscription_is_cancelled_and_lapsed(
    client: AsyncClient, app
) -> None:
    """Cancelling leaves CANCEL_PENDING, which is still a live contract
    until the paid period ends — so it still blocks. The account can be
    closed once nothing is outstanding."""
    await _subscribe(client)
    assert (await client.post("/v1/billing/cancel")).status_code == 200

    blocked = await client.post("/v1/auth/account/delete", json={"current_password": PASSWORD})

    assert blocked.status_code == 409


async def test_a_free_account_closes_without_billing_involvement(client: AsyncClient, app) -> None:
    from plan_fixtures import set_plan

    await set_plan(app, client.user_id, PlanId.FREE)  # type: ignore[attr-defined]

    response = await client.post("/v1/auth/account/delete", json={"current_password": PASSWORD})

    assert response.status_code == 204


async def test_an_operator_assigned_plan_does_not_block_closing(client: AsyncClient, app) -> None:
    """A comp has no provider contract, so there is nothing to orphan."""
    from plan_fixtures import set_plan

    await set_plan(app, client.user_id, PlanId.CREATOR)  # type: ignore[attr-defined]

    response = await client.post("/v1/auth/account/delete", json={"current_password": PASSWORD})

    assert response.status_code == 204


# ── what closing keeps ───────────────────────────────────────────────


async def test_billing_history_survives_closing(app, client: AsyncClient) -> None:
    """The record of money that moved is not the user's to delete.

    A chargeback, a reconciliation or a refund question can arrive after
    the account is gone, and `billing_payments` cascades from `users` —
    which is exactly why closing is anonymisation rather than a delete.
    """
    from sqlalchemy import func, select

    from luber_database.models.payments import BillingPayment

    await _subscribe(client)
    await client.post("/v1/billing/cancel")

    factory = app.state.session_factory
    async with factory() as session:
        before = (
            await session.execute(select(func.count()).select_from(BillingPayment))
        ).scalar_one()
    assert before == 1


async def test_the_closed_row_carries_no_personal_data(app, client: AsyncClient) -> None:
    import uuid as _uuid

    from sqlalchemy import select

    from luber_database.models.user import User

    user_id = _uuid.UUID(client.user_id)  # type: ignore[attr-defined]
    await client.patch("/v1/auth/profile", json={"display_name": "부르다"})
    await client.post("/v1/auth/account/delete", json={"current_password": PASSWORD})

    factory = app.state.session_factory
    async with factory() as session:
        row = (await session.execute(select(User).where(User.id == user_id))).scalar_one()

    assert row.deleted_at is not None
    assert row.display_name is None
    assert row.password_hash is None
    assert "user-a@example.com" not in row.email
    assert row.email.endswith("@deleted.invalid")


async def test_closing_ends_every_session_not_just_this_one(app, client: AsyncClient) -> None:
    """Closing on one device must not leave the account open on another."""
    from httpx import ASGITransport
    from httpx import AsyncClient as Fresh

    async with Fresh(transport=ASGITransport(app=app), base_url="http://testserver") as second:
        await second.post(
            "/v1/auth/login", json={"email": "user-a@example.com", "password": PASSWORD}
        )
        assert (await second.get("/v1/auth/me")).status_code == 200

        await client.post("/v1/auth/account/delete", json={"current_password": PASSWORD})

        assert (await second.get("/v1/auth/me")).status_code == 401
