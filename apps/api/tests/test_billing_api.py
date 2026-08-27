"""The billing endpoints as the internet meets them.

The repository tests prove the logic. These prove the boundary: that the
public notification endpoint refuses a forged POST, that no request body
anywhere can name a price or somebody else's contract, that reaching the
return URL activates nothing, and that PayApp gets exactly the `SUCCESS`
it retries without.

Every test here drives the real routes. The provider is a deterministic
fake — no automated run in this repository can reach api.payapp.kr, and
that is a safety property rather than a testing convenience: a suite that
could call `rebillRegist` with real credentials could register a real
recurring contract against a real person.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from luber_billing.payapp.fake import feedback_payload
from luber_billing.states import SubscriptionState
from plan_fixtures import set_plan

from luber_schemas.plans import PLANS, PlanId

CREDS = {"userid": "boorda-test", "linkkey": "test-linkkey", "linkval": "test-linkval"}
PRO = PLANS[PlanId.PRO]
CHECKOUT = {"plan_id": "pro", "phone": "010-1234-5678"}


async def _checkout(client: AsyncClient, **overrides: object) -> dict:
    body = {**CHECKOUT, **overrides}
    response = await client.post("/v1/billing/checkout", json=body)
    assert response.status_code == 201, response.text
    return response.json()


async def _pay(
    client: AsyncClient,
    *,
    rebill_no: str,
    price: int = PRO.monthly_price_krw,
    pay_state: int = 4,
    mul_no: str = "777",
    path: str = "/v1/billing/payapp/feedback",
    **overrides: str,
):
    return await client.post(
        path,
        data=feedback_payload(
            **CREDS,
            rebill_no=rebill_no,
            price=price,
            pay_state=pay_state,
            mul_no=mul_no,
            **overrides,
        ),
    )


# ── checkout ─────────────────────────────────────────────────────────


async def test_checkout_requires_a_session(anon_client: AsyncClient) -> None:
    assert (await anon_client.post("/v1/billing/checkout", json=CHECKOUT)).status_code == 401


async def test_checkout_returns_the_provider_url_and_nothing_secret(
    client: AsyncClient,
) -> None:
    body = await _checkout(client)

    assert body["payurl"].startswith("https://payapp.kr/")
    serialised = str(body)
    for secret in (CREDS["linkkey"], CREDS["linkval"]):
        assert secret not in serialised


async def test_the_price_sent_to_payapp_comes_from_the_plan_table(
    client: AsyncClient, payapp
) -> None:
    """The assertion the whole integration rests on.

    The request body carried a plan name. The amount PayApp was asked to
    charge came from the server's own table.
    """
    await _checkout(client)

    assert payapp.registrations[0].goodprice == PRO.monthly_price_krw


async def test_a_client_supplied_price_is_ignored(client: AsyncClient, payapp) -> None:
    """Amount tampering from the frontend.

    Extra fields are not part of the schema, so they cannot reach the
    provider call — but the test asserts the outcome rather than the
    mechanism, because the mechanism could change.
    """
    await client.post(
        "/v1/billing/checkout",
        json={**CHECKOUT, "amount_krw": 100, "price": 100, "monthly_price_krw": 100},
    )

    assert payapp.registrations[0].goodprice == PRO.monthly_price_krw


async def test_a_client_supplied_entitlement_is_ignored(client: AsyncClient, payapp) -> None:
    await client.post(
        "/v1/billing/checkout",
        json={
            **CHECKOUT,
            "monthly_generation_limit": 99999,
            "download_wav": True,
            "commercial_use": True,
        },
    )

    entitlement = (await client.get("/v1/account/entitlement")).json()
    # Registration is not payment: still Free, whatever was asked for.
    assert entitlement["plan"]["plan_id"] == "free"


async def test_free_has_no_checkout(client: AsyncClient) -> None:
    response = await client.post("/v1/billing/checkout", json={**CHECKOUT, "plan_id": "free"})

    assert response.status_code == 400
    assert response.json()["detail"] == "FREE_PLAN_HAS_NO_CHECKOUT"


async def test_an_unknown_plan_is_refused(client: AsyncClient) -> None:
    response = await client.post("/v1/billing/checkout", json={**CHECKOUT, "plan_id": "platinum"})

    assert response.status_code == 422


async def test_an_unusable_phone_number_is_refused(client: AsyncClient) -> None:
    response = await client.post("/v1/billing/checkout", json={**CHECKOUT, "phone": "02-123-4567"})

    assert response.status_code == 422


async def test_the_phone_reaches_payapp_normalised(client: AsyncClient, payapp) -> None:
    await _checkout(client)

    assert payapp.registrations[0].recvphone == "01012345678"


async def test_the_callback_urls_point_at_this_deployment(client: AsyncClient, payapp) -> None:
    await _checkout(client)

    call = payapp.registrations[0]
    assert call.feedbackurl.endswith("/v1/billing/payapp/feedback")
    assert call.failurl.endswith("/v1/billing/payapp/failure")
    assert call.returnurl.endswith("/billing/return")


async def test_the_correlation_id_carries_no_account_information(
    client: AsyncClient, payapp
) -> None:
    """It travels to a public endpoint and back. Encoding a user id in it
    would leak one; a guessable one would let a forged notification be
    aimed at a chosen account."""
    body = await _checkout(client)

    assert payapp.registrations[0].var1 == body["correlation_id"]
    assert client.user_id not in body["correlation_id"]  # type: ignore[attr-defined]


# ── A. registration is not payment ───────────────────────────────────


async def test_a_registered_checkout_grants_nothing(client: AsyncClient) -> None:
    """The single most important test in the phase."""
    await _checkout(client)

    entitlement = (await client.get("/v1/account/entitlement")).json()

    assert entitlement["plan"]["plan_id"] == "free"
    assert entitlement["generation_limit"] == 20
    assert entitlement["download_mp3"] is False


async def test_the_checkout_response_says_no_payment_has_happened(client: AsyncClient) -> None:
    body = await _checkout(client)

    assert body["status"] == SubscriptionState.PENDING_INITIAL_PAYMENT.value


async def test_status_reports_awaiting_payment(client: AsyncClient) -> None:
    await _checkout(client)

    status_body = (await client.get("/v1/billing/status")).json()

    assert status_body["awaiting_payment"] is True
    assert status_body["status"] == SubscriptionState.PENDING_INITIAL_PAYMENT.value


# ── H. the return URL proves nothing ─────────────────────────────────


async def test_reaching_the_return_url_activates_nothing(client: AsyncClient) -> None:
    """A user who pays, closes the tab, or types the URL sees the same
    thing: whatever the server's own records say."""
    await _checkout(client)

    # Whatever the browser does with query parameters, this is the only
    # thing the page can read — and it reads the database.
    status_body = (await client.get("/v1/billing/status")).json()

    assert status_body["plan_id"] == "pro"  # the plan requested
    assert status_body["awaiting_payment"] is True
    assert (await client.get("/v1/account/entitlement")).json()["plan"]["plan_id"] == "free"


async def test_query_parameters_cannot_grant_a_plan(client: AsyncClient) -> None:
    await _checkout(client)

    await client.get(
        "/v1/billing/status", params={"pay_state": "4", "status": "ACTIVE", "plan_id": "creator"}
    )

    assert (await client.get("/v1/account/entitlement")).json()["plan"]["plan_id"] == "free"


# ── I. checkout idempotency ──────────────────────────────────────────


async def test_a_second_checkout_click_does_not_register_twice(client: AsyncClient, payapp) -> None:
    await _checkout(client)

    second = await client.post("/v1/billing/checkout", json=CHECKOUT)

    assert second.status_code == 409
    assert len(payapp.registrations) == 1


async def test_an_active_subscriber_cannot_start_another_contract(
    client: AsyncClient, payapp
) -> None:
    body = await _checkout(client)
    await _pay(client, rebill_no=payapp.registrations[0].var1 and "900001")

    again = await client.post("/v1/billing/checkout", json={**CHECKOUT, "plan_id": "creator"})

    assert again.status_code == 409
    assert again.json()["detail"] == "SUBSCRIPTION_ALREADY_ACTIVE"
    assert body["plan_id"] == "pro"


# ── B/C. payment and idempotency at the boundary ─────────────────────


async def test_a_valid_notification_activates_the_plan(client: AsyncClient) -> None:
    await _checkout(client)

    response = await _pay(client, rebill_no="900001")

    assert response.status_code == 200
    assert response.text == "SUCCESS"
    entitlement = (await client.get("/v1/account/entitlement")).json()
    assert entitlement["plan"]["plan_id"] == "pro"
    assert entitlement["generation_limit"] == 500
    assert entitlement["download_wav"] is True


async def test_payapp_gets_the_exact_body_it_retries_without(client: AsyncClient) -> None:
    """Not JSON, not HTML, no redirect. PayApp retries anything else."""
    await _checkout(client)

    response = await _pay(client, rebill_no="900001")

    assert response.text == "SUCCESS"
    assert response.headers["content-type"].startswith("text/plain")
    assert response.status_code == 200


async def test_ten_deliveries_activate_once(client: AsyncClient) -> None:
    await _checkout(client)

    responses = [await _pay(client, rebill_no="900001") for _ in range(10)]

    assert {r.status_code for r in responses} == {200}
    assert {r.text for r in responses} == {"SUCCESS"}
    history = (await client.get("/v1/billing/payments")).json()["items"]
    assert len(history) == 1


# ── E/F. forged notifications ────────────────────────────────────────


async def test_a_notification_with_the_wrong_linkval_grants_nothing(
    client: AsyncClient,
) -> None:
    await _checkout(client)

    response = await client.post(
        "/v1/billing/payapp/feedback",
        data=feedback_payload(
            userid=CREDS["userid"],
            linkkey=CREDS["linkkey"],
            linkval="guessed",
            rebill_no="900001",
            price=PRO.monthly_price_krw,
        ),
    )

    assert response.status_code == 403
    assert response.text != "SUCCESS"
    assert (await client.get("/v1/account/entitlement")).json()["plan"]["plan_id"] == "free"


async def test_a_notification_with_the_wrong_userid_grants_nothing(client: AsyncClient) -> None:
    await _checkout(client)

    response = await client.post(
        "/v1/billing/payapp/feedback",
        data=feedback_payload(
            userid="someone-else",
            linkkey=CREDS["linkkey"],
            linkval=CREDS["linkval"],
            rebill_no="900001",
            price=PRO.monthly_price_krw,
        ),
    )

    assert response.status_code == 403
    assert (await client.get("/v1/account/entitlement")).json()["plan"]["plan_id"] == "free"


async def test_an_entirely_forged_notification_grants_nothing(client: AsyncClient) -> None:
    await _checkout(client)

    response = await client.post(
        "/v1/billing/payapp/feedback",
        data={"pay_state": "4", "rebill_no": "900001", "price": str(PRO.monthly_price_krw)},
    )

    assert response.status_code == 403
    assert (await client.get("/v1/account/entitlement")).json()["plan"]["plan_id"] == "free"


# ── D. amount validation at the boundary ─────────────────────────────


async def test_a_short_payment_is_acknowledged_but_grants_nothing(
    client: AsyncClient,
) -> None:
    """Acknowledged because it was durably recorded — a retry would
    change nothing — and granted nothing because the figure is wrong."""
    await _checkout(client)

    response = await _pay(client, rebill_no="900001", price=100)

    assert response.status_code == 200
    assert response.text == "SUCCESS"
    assert (await client.get("/v1/account/entitlement")).json()["plan"]["plan_id"] == "free"


async def test_paying_the_basic_price_does_not_buy_pro(client: AsyncClient) -> None:
    """The realistic attack: a genuine PayApp payment for the cheaper
    plan, replayed against a Pro subscription."""
    await _checkout(client)

    await _pay(client, rebill_no="900001", price=PLANS[PlanId.BASIC].monthly_price_krw)

    assert (await client.get("/v1/account/entitlement")).json()["plan"]["plan_id"] == "free"


# ── G. unknown contracts ─────────────────────────────────────────────


async def test_a_payment_naming_an_unknown_contract_grants_nothing(
    client: AsyncClient,
) -> None:
    await _checkout(client)

    response = await _pay(client, rebill_no="000000", mul_no="999")

    assert response.status_code == 200
    assert (await client.get("/v1/account/entitlement")).json()["plan"]["plan_id"] == "free"


# ── L. failures ──────────────────────────────────────────────────────


async def test_a_failure_notification_is_acknowledged(client: AsyncClient) -> None:
    await _checkout(client)
    await _pay(client, rebill_no="900001")

    response = await _pay(
        client,
        rebill_no="900001",
        pay_state=99,
        mul_no="888",
        path="/v1/billing/payapp/failure",
    )

    assert response.status_code == 200
    assert response.text == "SUCCESS"


async def test_a_failure_removes_paid_access_without_touching_the_library(
    client: AsyncClient,
) -> None:
    await _checkout(client)
    await _pay(client, rebill_no="900001")
    made = await client.post(
        "/v1/generations",
        json={
            "title": "Before the failure",
            "prompt": "quiet synth pop",
            "lyrics": "[verse]\nstill here\n",
            "vocal_gender": "female",
            "duration": 30,
        },
    )
    assert made.status_code == 202

    await _pay(
        client,
        rebill_no="900001",
        pay_state=99,
        mul_no="888",
        path="/v1/billing/payapp/failure",
    )

    assert (await client.get("/v1/account/entitlement")).json()["plan"]["plan_id"] == "free"
    # The songs stay. Losing a subscription is not losing your work.
    assert (await client.get("/v1/generations")).json()["total"] == 1


async def test_a_failure_shows_in_the_users_own_history(client: AsyncClient) -> None:
    await _checkout(client)
    await _pay(client, rebill_no="900001")

    await _pay(
        client,
        rebill_no="900001",
        pay_state=99,
        mul_no="888",
        path="/v1/billing/payapp/failure",
    )

    statuses = [i["status"] for i in (await client.get("/v1/billing/payments")).json()["items"]]
    assert "FAILED" in statuses
    assert "SUCCEEDED" in statuses


# ── N/O/Q. cancellation ──────────────────────────────────────────────


async def test_cancelling_calls_the_provider(client: AsyncClient, payapp) -> None:
    await _checkout(client)
    await _pay(client, rebill_no="900001")

    response = await client.post("/v1/billing/cancel")

    assert response.status_code == 200
    assert payapp.cancellations == ["900001"]


async def test_cancelling_preserves_the_period_already_paid_for(client: AsyncClient) -> None:
    await _checkout(client)
    await _pay(client, rebill_no="900001")

    await client.post("/v1/billing/cancel")

    entitlement = (await client.get("/v1/account/entitlement")).json()
    assert entitlement["plan"]["plan_id"] == "pro"
    status_body = (await client.get("/v1/billing/status")).json()
    assert status_body["auto_renew"] is False


async def test_a_provider_refusal_leaves_the_local_state_alone(client: AsyncClient, payapp) -> None:
    """A subscription marked cancelled here while PayApp still holds a
    live contract would charge the customer next month with the product
    telling them they had cancelled."""
    from luber_billing.payapp.client import PayAppError

    await _checkout(client)
    await _pay(client, rebill_no="900001")
    payapp.fail_cancel_with = PayAppError("provider down", transport=True)

    response = await client.post("/v1/billing/cancel")

    assert response.status_code == 502
    assert (await client.get("/v1/billing/status")).json()["auto_renew"] is True


async def test_cancellation_requires_a_session(anon_client: AsyncClient) -> None:
    assert (await anon_client.post("/v1/billing/cancel")).status_code == 401


async def test_an_account_with_nothing_to_cancel_gets_404(client: AsyncClient) -> None:
    assert (await client.post("/v1/billing/cancel")).status_code == 404


async def test_one_account_cannot_cancel_anothers_subscription(
    client: AsyncClient, client_b: AsyncClient, payapp
) -> None:
    """There is no argument through which a stranger's contract could be
    named — cancellation takes no parameters at all."""
    await _checkout(client)
    await _pay(client, rebill_no="900001")

    response = await client_b.post("/v1/billing/cancel")

    assert response.status_code == 404
    assert payapp.cancellations == []
    assert (await client.get("/v1/billing/status")).json()["auto_renew"] is True


async def test_a_client_supplied_rebill_no_is_not_accepted(
    client: AsyncClient, client_b: AsyncClient, payapp
) -> None:
    await _checkout(client)
    await _pay(client, rebill_no="900001")

    response = await client_b.post("/v1/billing/cancel", json={"rebill_no": "900001"})

    assert response.status_code in (404, 422)
    assert payapp.cancellations == []


# ── P. history isolation ─────────────────────────────────────────────


async def test_payment_history_requires_a_session(anon_client: AsyncClient) -> None:
    assert (await anon_client.get("/v1/billing/payments")).status_code == 401


async def test_one_account_cannot_see_anothers_payments(
    client: AsyncClient, client_b: AsyncClient
) -> None:
    await _checkout(client)
    await _pay(client, rebill_no="900001")

    assert (await client_b.get("/v1/billing/payments")).json()["items"] == []
    assert len((await client.get("/v1/billing/payments")).json()["items"]) == 1


async def test_payment_history_exposes_no_provider_identifiers(client: AsyncClient) -> None:
    await _checkout(client)
    await _pay(client, rebill_no="900001", mul_no="secret-mul")

    body = (await client.get("/v1/billing/payments")).text

    assert "secret-mul" not in body
    assert "900001" not in body


async def test_no_response_anywhere_leaks_a_payapp_secret(client: AsyncClient) -> None:
    await _checkout(client)
    await _pay(client, rebill_no="900001")

    for path in ("/v1/billing/status", "/v1/billing/payments", "/v1/account/entitlement"):
        body = (await client.get(path)).text
        assert CREDS["linkkey"] not in body
        assert CREDS["linkval"] not in body


# ── forward compatibility ────────────────────────────────────────────


async def test_an_unfamiliar_payapp_field_does_not_break_a_payment(
    client: AsyncClient,
) -> None:
    """PayApp documents that fields may be added over time."""
    await _checkout(client)

    response = await _pay(client, rebill_no="900001", some_new_field_2027="x", another_new_one="y")

    assert response.text == "SUCCESS"
    assert (await client.get("/v1/account/entitlement")).json()["plan"]["plan_id"] == "pro"


async def test_a_malformed_notification_is_refused_not_crashed(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/billing/payapp/feedback", data={**CREDS, "pay_state": "not-a-number"}
    )

    assert response.status_code == 403
    assert response.text != "SUCCESS"


# ── plan interaction with Phase 6 ────────────────────────────────────


async def test_a_paid_plan_from_payment_unlocks_downloads(client: AsyncClient, app) -> None:
    """Phase 6's download gate reads the entitlement, so a real payment
    is what opens it — no separate switch to keep in step."""
    await set_plan(app, client.user_id, PlanId.FREE)  # type: ignore[attr-defined]
    await _checkout(client)
    await _pay(client, rebill_no="900001")

    entitlement = (await client.get("/v1/account/entitlement")).json()

    assert entitlement["download_mp3"] is True
    assert entitlement["commercial_use"] is True


@pytest.mark.parametrize(
    ("plan_id", "price", "limit"),
    [("basic", 19900, 200), ("pro", 29900, 500), ("creator", 49900, 1000)],
)
async def test_each_paid_plan_charges_and_grants_its_own_figures(
    client: AsyncClient, payapp, plan_id: str, price: int, limit: int
) -> None:
    await _checkout(client, plan_id=plan_id)
    assert payapp.registrations[0].goodprice == price

    await _pay(client, rebill_no="900001", price=price)

    entitlement = (await client.get("/v1/account/entitlement")).json()
    assert entitlement["plan"]["plan_id"] == plan_id
    assert entitlement["generation_limit"] == limit


# ── operator-assigned plans are not subscriptions ────────────────────


async def test_an_operator_assigned_plan_is_not_reported_as_a_subscription(
    client: AsyncClient, app
) -> None:
    """The comp that must not look like a contract.

    Phase 6's `set_plan` script writes a subscription row with no
    provider linkage. Reporting it as a subscription would show the
    account 이용 중, 자동 갱신 켜짐 and a 구독 해지 button that cancels
    nothing — confidently wrong in the way this endpoint exists to avoid.
    """
    await set_plan(app, client.user_id, PlanId.PRO)  # type: ignore[attr-defined]

    body = (await client.get("/v1/billing/status")).json()

    assert body["status"] == "NONE"
    assert body["auto_renew"] is False
    assert body["next_renewal_at"] is None
    # The plan it grants is still real; only the contract is absent.
    assert body["plan_id"] == "pro"


async def test_an_operator_assigned_plan_offers_nothing_to_cancel(
    client: AsyncClient, app, payapp
) -> None:
    await set_plan(app, client.user_id, PlanId.PRO)  # type: ignore[attr-defined]

    response = await client.post("/v1/billing/cancel")

    assert response.status_code == 404
    assert payapp.cancellations == []


async def test_a_real_subscription_is_reported_as_one(client: AsyncClient) -> None:
    await _checkout(client)
    await _pay(client, rebill_no="900001")

    body = (await client.get("/v1/billing/status")).json()

    assert body["status"] == "ACTIVE"
    assert body["auto_renew"] is True
    assert body["next_renewal_at"] is not None


# ── the deployment kill switch ───────────────────────────────────────


def _payload() -> dict[str, object]:
    """A minimal valid generation request."""
    return {
        "title": "Midnight Window",
        "prompt": "quiet synth pop, warm pads",
        "lyrics": "[verse]\nthe city keeps its light on\n",
        "vocal_gender": "female",
        "duration": 30,
    }


async def test_generation_is_refused_when_disabled(client: AsyncClient, monkeypatch) -> None:
    """A deployment with no GPU and no durable storage must not accept
    generations — the row would sit QUEUED for ever and the reservation
    would hold a slot of the user's allowance against nothing."""
    from luber_api.settings import get_settings

    monkeypatch.setenv("GENERATION_ENABLED", "false")
    get_settings.cache_clear()
    try:
        response = await client.post("/v1/generations", json=_payload())
    finally:
        monkeypatch.delenv("GENERATION_ENABLED", raising=False)
        get_settings.cache_clear()

    assert response.status_code == 503
    assert response.json()["detail"] == "GENERATION_UNAVAILABLE"


async def test_a_refused_generation_spends_no_allowance(client: AsyncClient, monkeypatch) -> None:
    """The property the guard exists for: it refuses *before* reserving.

    A 503 that had already taken a slot would be worse than no guard at
    all — the user would lose a song from their monthly allowance and
    have nothing to show for it.
    """
    from luber_api.settings import get_settings

    before = (await client.get("/v1/account/entitlement")).json()

    monkeypatch.setenv("GENERATION_ENABLED", "false")
    get_settings.cache_clear()
    try:
        await client.post("/v1/generations", json=_payload())
    finally:
        monkeypatch.delenv("GENERATION_ENABLED", raising=False)
        get_settings.cache_clear()

    after = (await client.get("/v1/account/entitlement")).json()
    assert after["generation_used"] == before["generation_used"]
    assert after["generation_remaining"] == before["generation_remaining"]
    # And nothing was written to the Library either.
    assert (await client.get("/v1/generations")).json()["total"] == 0


async def test_generation_works_when_enabled(client: AsyncClient) -> None:
    """The switch defaults on, so no existing deployment changes."""
    response = await client.post("/v1/generations", json=_payload())

    assert response.status_code == 202


async def test_billing_still_works_while_generation_is_disabled(
    client: AsyncClient, monkeypatch
) -> None:
    """The kill switch is narrow on purpose: it stops generation and
    nothing else. Billing must keep working, since that is the whole
    reason this deployment exists."""
    from luber_api.settings import get_settings

    monkeypatch.setenv("GENERATION_ENABLED", "false")
    get_settings.cache_clear()
    try:
        checkout = await client.post("/v1/billing/checkout", json=CHECKOUT)
        assert checkout.status_code == 201
        paid = await _pay(client, rebill_no="900001")
        assert paid.text == "SUCCESS"
    finally:
        monkeypatch.delenv("GENERATION_ENABLED", raising=False)
        get_settings.cache_clear()

    assert (await client.get("/v1/account/entitlement")).json()["plan"]["plan_id"] == "pro"
