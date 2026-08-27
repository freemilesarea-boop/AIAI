"""Checkout, provider notifications, cancellation and history.

Five endpoints, and they do not share a security model — which is the
most important thing to understand about this file.

``POST /v1/billing/checkout``, ``POST /v1/billing/cancel`` and
``GET /v1/billing/payments`` are ordinary product routes: a session is
required, the account is taken from that session, and nothing the body
says about identity is believed.

``POST /v1/billing/payapp/feedback`` and ``.../failure`` are the
opposite. PayApp's servers call them and have no session, so they are
open to the internet by necessity. Their authenticity check is the
userid/linkkey/linkval comparison in `luber_billing.payapp.notification`,
done in constant time, and everything downstream of it is written on the
assumption that the caller may be hostile.

Three rules run through all of it:

**The browser never names a price.** It sends a plan id. The server
resolves that to an amount from the plan table, stores it on the
checkout, sends it to PayApp, and later compares PayApp's reported
amount against it. There is no request field anywhere in this file that
could make a subscription cost less.

**The browser never names a rebill_no.** Cancellation takes no
identifier at all — the authenticated account has one subscription and
the server looks it up. A caller cannot cancel a stranger's contract by
guessing a number, because there is nowhere to put the number.

**Registration is not payment.** `rebillRegist` succeeding means PayApp
accepted a request. The response to the browser says so, the
subscription sits in PENDING_INITIAL_PAYMENT, and only a validated
`pay_state=4` notification moves it.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from luber_billing.payapp.client import (
    HttpPayAppClient,
    PayAppClient,
    PayAppError,
)
from luber_billing.payapp.notification import (
    NotificationRejected,
    authenticate,
    parse,
    redact,
)
from luber_billing.policy import (
    InvalidPhoneNumber,
    billing_cycle_day,
    contract_expiry,
    mask_phone,
    normalise_phone,
)
from luber_billing.states import SubscriptionState
from pydantic import BaseModel, Field

from luber_api.dependencies import get_session_factory
from luber_api.rate_limit import RateLimitExceeded, enforce_rate_limit
from luber_api.session import require_current_user
from luber_api.settings import ApiSettings, get_settings
from luber_database.billing_repository import (
    BillingRepository,
    CheckoutAlreadyOpen,
    NoSubscriptionToCancel,
    NotificationProcessor,
)
from luber_database.models.payments import EVENT_FAILURE, EVENT_FEEDBACK
from luber_database.models.user import User
from luber_schemas.plans import PLANS, PlanId, plan_for

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/billing", tags=["billing"])

#: PayApp expects this exact body, and retries when it does not get it.
#: Not JSON, not HTML, no redirect — the string and nothing else.
ACK_BODY = "SUCCESS"


# ── request/response models ──────────────────────────────────────────


class CheckoutRequest(BaseModel):
    """Everything the client is allowed to say about a subscription.

    A plan name and a phone number. Not a price, not a generation limit,
    not a download entitlement, not a rebill_no — every one of those is
    resolved server-side, and adding a field here would be the way this
    system gets robbed.
    """

    plan_id: PlanId
    #: PayApp's `recvphone` is required for recurring registration and
    #: BOORDA has no phone on `User`. Collected per checkout rather than
    #: added to the profile: it is billing contact information, not
    #: public identity, and a field nobody needs is a field that leaks.
    phone: str = Field(min_length=9, max_length=20)


class CheckoutResponse(BaseModel):
    """Where to send the browser, and what is true so far.

    `payurl` is PayApp's own hosted page. It carries no credential of
    ours — it is a URL PayApp generated and told us to use.
    """

    payurl: str
    plan_id: str
    display_name: str
    amount_krw: int
    #: Always PENDING_INITIAL_PAYMENT here. Present so the client cannot
    #: read the 200 as "subscribed" — it is told, in the response body,
    #: that nothing has been paid.
    status: str
    correlation_id: str


class BillingStatusResponse(BaseModel):
    """What the return page and Settings read.

    Server-side truth, queried by the client after PayApp sends the user
    back. The return URL's own query parameters are not consulted by
    anything here.
    """

    plan_id: str
    display_name: str
    status: str
    auto_renew: bool
    period_start: str | None
    period_end: str | None
    next_renewal_at: str | None
    last_payment_at: str | None
    #: True while a payment is expected but unconfirmed, so the return
    #: page can say 결제 확인 중 rather than guessing either way.
    awaiting_payment: bool
    #: False when the deployment has no PayApp credentials, so the UI
    #: renders an honest unavailable state instead of a dead button.
    checkout_available: bool


class PaymentRecord(BaseModel):
    paid_at: str
    plan_id: str
    amount_krw: int
    status: str
    #: Provider-supplied text on a failure, already truncated. No
    #: identifiers, no secrets.
    failure_reason: str | None = None


class PaymentHistoryResponse(BaseModel):
    items: list[PaymentRecord]


# ── dependencies ─────────────────────────────────────────────────────


async def get_billing_repository(
    request: Request,
    user: Annotated[User, Depends(require_current_user)],
) -> Any:
    factory = get_session_factory(request)
    async with factory() as session:
        yield BillingRepository(session, user.id)


def get_payapp_client(request: Request) -> PayAppClient:
    """The provider client, or a refusal.

    Held on app state so tests substitute a deterministic fake and no
    automated run can reach PayApp. A suite that could call
    `rebillRegist` against real credentials is a suite that can register
    a real recurring contract against a real person.
    """
    existing = getattr(request.app.state, "payapp_client", None)
    if existing is not None:
        return existing  # type: ignore[no-any-return]

    settings = get_settings()
    if not settings.billing_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="BILLING_NOT_CONFIGURED",
        )
    client = HttpPayAppClient(
        userid=str(settings.payapp_userid),
        linkkey=str(settings.payapp_linkkey),
        api_url=settings.payapp_api_url,
    )
    request.app.state.payapp_client = client
    return client


# ── checkout ─────────────────────────────────────────────────────────


@router.post("/checkout", response_model=CheckoutResponse, status_code=status.HTTP_201_CREATED)
async def create_checkout(
    payload: CheckoutRequest,
    request: Request,
    repository: Annotated[BillingRepository, Depends(get_billing_repository)],
    client: Annotated[PayAppClient, Depends(get_payapp_client)],
) -> CheckoutResponse:
    """Register a recurring payment request with PayApp.

    Order of operations is the safety property: the local row is written
    and committed *before* PayApp is called. If this process dies at any
    point after that, the checkout is a visible CREATED row that
    reconciliation can find — rather than a recurring contract at PayApp
    that BOORDA has no record of.
    """
    settings = get_settings()
    plan = plan_for(payload.plan_id)
    if not plan.is_paid:
        # Free is not a purchase. Sending it to a payment provider would
        # ask a customer to authorise ₩0, which PayApp would reject and
        # which would be a confusing thing to have offered.
        raise HTTPException(status_code=400, detail="FREE_PLAN_HAS_NO_CHECKOUT")

    try:
        phone = normalise_phone(payload.phone)
    except InvalidPhoneNumber as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        checkout = await repository.create_checkout(plan_id=plan.plan_id, recvphone=phone)
    except CheckoutAlreadyOpen as exc:
        # Not an error the user caused twice over: a double click, or a
        # subscription they already have. Either way, refusing is what
        # keeps one account from holding two recurring contracts.
        detail = "SUBSCRIPTION_ALREADY_ACTIVE" if exc.checkout is None else "CHECKOUT_ALREADY_OPEN"
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail) from exc

    now = checkout.created_at
    base = settings.payapp_public_base_url.rstrip("/")
    try:
        registration = await client.register_recurring(
            goodname=f"BOORDA {plan.display_name}",
            # From the plan table, never from the request body.
            goodprice=plan.monthly_price_krw,
            recvphone=phone,
            cycle_day=billing_cycle_day(now),
            expire_date=contract_expiry(now),
            feedbackurl=f"{base}/v1/billing/payapp/feedback",
            failurl=f"{base}/v1/billing/payapp/failure",
            returnurl=f"{settings.payapp_return_base_url.rstrip('/')}/billing/return",
            # Opaque, random, and authorising nothing. It tells the
            # notification handler which row to look at first; the
            # decision is made against our own records regardless.
            var1=checkout.correlation_id,
        )
    except PayAppError as exc:
        await repository.mark_registration_failed(
            checkout.id, reason=f"{exc.errno or 'transport'}: {exc}"
        )
        logger.warning(
            "payapp recurring registration failed",
            extra={
                "correlation_id": checkout.correlation_id,
                "plan_id": plan.plan_id.value,
                "payapp_errno": exc.errno,
                "transport": exc.transport,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="PAYMENT_PROVIDER_UNAVAILABLE"
        ) from exc

    subscription = await repository.mark_registered(
        checkout.id, rebill_no=registration.rebill_no, payurl=registration.payurl
    )

    logger.info(
        "payapp recurring registration accepted; awaiting payment",
        extra={
            "correlation_id": checkout.correlation_id,
            "plan_id": plan.plan_id.value,
            "amount_krw": plan.monthly_price_krw,
            "recvphone_masked": mask_phone(phone),
        },
    )

    return CheckoutResponse(
        payurl=registration.payurl,
        plan_id=plan.plan_id.value,
        display_name=plan.display_name,
        amount_krw=plan.monthly_price_krw,
        status=subscription.status,
        correlation_id=checkout.correlation_id,
    )


# ── provider notifications ───────────────────────────────────────────


async def _form_payload(request: Request) -> dict[str, str]:
    """PayApp posts `application/x-www-form-urlencoded`."""
    form = await request.form()
    return {key: str(value) for key, value in form.items()}


async def _handle_notification(request: Request, *, kind: str) -> Response:
    """The shared body of both provider endpoints.

    Feedback and failure differ only in which endpoint PayApp chose and
    therefore in the event's `kind`; the authentication, the idempotency
    and the refusal-to-guess are identical, and writing them twice would
    be two places for them to drift apart.
    """
    settings = get_settings()
    payload = await _form_payload(request)

    if not settings.billing_available():
        # Nothing to validate against. Refusing loudly is right: a
        # deployment receiving payment notifications it cannot
        # authenticate has a configuration problem, and answering
        # SUCCESS would tell PayApp to stop retrying.
        logger.error("payapp notification received but billing is not configured")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="UNCONFIGURED")

    await _throttle(request, settings)

    factory = get_session_factory(request)
    async with factory() as session:
        processor = NotificationProcessor(session)
        try:
            authenticate(
                payload,
                expected_userid=str(settings.payapp_userid),
                expected_linkkey=str(settings.payapp_linkkey),
                expected_linkval=str(settings.payapp_linkval),
            )
            notification = parse(payload)
        except NotificationRejected as exc:
            await processor.record_rejection(reason=exc.reason, payload=redact(payload))
            logger.warning(
                "rejected payapp notification",
                extra={"payapp_event_kind": kind, "reason": exc.reason},
            )
            # Not SUCCESS. If this really was PayApp — a rotated key, a
            # misconfiguration — we want it to keep retrying and we want
            # the anomaly rows to accumulate until somebody looks.
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="REJECTED") from exc

        result = await processor.process(notification, kind=kind)

    if not result.acknowledge:
        # The event could not be durably recorded. Saying SUCCESS here
        # would tell PayApp the payment was handled when it was not, and
        # PayApp would never send it again.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="NOT_PROCESSED"
        )

    return Response(content=ACK_BODY, media_type="text/plain", status_code=status.HTTP_200_OK)


async def _throttle(request: Request, settings: ApiSettings) -> None:
    """Bound how fast one address can post notification-shaped requests.

    This endpoint has to be open, and every rejected request writes an
    anomaly row. Without a bound, the audit trail is also a way to fill
    the database.
    """
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        return
    client_host = request.client.host if request.client else "unknown"
    try:
        await enforce_rate_limit(
            redis,
            key=f"payapp-feedback:{client_host}",
            limit=settings.payapp_feedback_rate_limit,
            window_seconds=settings.payapp_feedback_rate_window_seconds,
        )
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="RATE_LIMITED",
            headers={"Retry-After": str(exc.retry_after_seconds)},
        ) from exc
    except Exception:
        # A rate limiter that is down must not become a payment outage.
        logger.warning("rate limiting unavailable for payapp feedback", exc_info=True)


@router.post("/payapp/feedback", include_in_schema=False)
async def payapp_feedback(request: Request) -> Response:
    """Payment notifications. Server-to-server; no session, by necessity."""
    return await _handle_notification(request, kind=EVENT_FEEDBACK)


@router.post("/payapp/failure", include_in_schema=False)
async def payapp_failure(request: Request) -> Response:
    """Recurring approval failures — `pay_state=99`.

    PayApp documents that a *first*-cycle failure is not notified here at
    all, so nothing in the system waits on this endpoint to discover that
    an initial payment did not happen. That is why an unpaid checkout
    resolves by timing out rather than by a failure that will never come.
    """
    return await _handle_notification(request, kind=EVENT_FAILURE)


# ── status, cancellation, history ────────────────────────────────────


@router.get("/status", response_model=BillingStatusResponse)
async def billing_status(
    repository: Annotated[BillingRepository, Depends(get_billing_repository)],
) -> BillingStatusResponse:
    """What the return page polls, and what Settings renders.

    The only thing the post-payment page is allowed to believe. A user
    who reaches `/billing/return` — by paying, by closing the window, or
    by typing the URL — sees whatever this says, and this reads the
    database.
    """
    settings = get_settings()
    subscription = await repository.subscription()
    # A row with no provider contract is not a subscription as far as
    # billing is concerned — it is an operator-assigned plan from Phase
    # 6, a comp. Reporting it as one would show a Free account "이용 중",
    # "자동 갱신 켜짐" and a 구독 해지 button that cancels nothing, which
    # is exactly the kind of confident wrongness this endpoint exists to
    # avoid. The plan it grants is still real; the *contract* is not.
    if subscription is None or subscription.provider_subscription_id is None:
        free = PLANS[PlanId.FREE]
        granted = plan_for(subscription.plan_id) if subscription is not None else free
        return BillingStatusResponse(
            plan_id=granted.plan_id.value,
            display_name=granted.display_name,
            status="NONE",
            auto_renew=False,
            period_start=None,
            period_end=None,
            next_renewal_at=None,
            last_payment_at=None,
            awaiting_payment=(await repository.open_checkout()) is not None,
            checkout_available=settings.billing_available(),
        )

    plan = plan_for(subscription.plan_id)
    state = subscription.status
    return BillingStatusResponse(
        plan_id=plan.plan_id.value,
        display_name=plan.display_name,
        status=state,
        auto_renew=bool(subscription.auto_renew),
        period_start=subscription.period_start.isoformat() if subscription.period_start else None,
        period_end=subscription.period_end.isoformat() if subscription.period_end else None,
        next_renewal_at=(
            subscription.next_renewal_at.isoformat() if subscription.next_renewal_at else None
        ),
        last_payment_at=(
            subscription.last_payment_at.isoformat() if subscription.last_payment_at else None
        ),
        awaiting_payment=state == SubscriptionState.PENDING_INITIAL_PAYMENT.value,
        checkout_available=settings.billing_available(),
    )


@router.post("/cancel", response_model=BillingStatusResponse)
async def cancel_subscription(
    repository: Annotated[BillingRepository, Depends(get_billing_repository)],
    client: Annotated[PayAppClient, Depends(get_payapp_client)],
) -> BillingStatusResponse:
    """Stop the next charge. Keep the period already paid for.

    Takes no arguments at all — deliberately. The account is the
    session's, the `rebill_no` is looked up from our own records, and
    there is therefore no field in this request a caller could use to
    reach another account's contract.

    PayApp is called *before* the local state changes. If PayApp refuses,
    nothing local moves: a subscription marked cancelled here while
    PayApp still holds a live contract would charge the customer again
    next month with the product telling them they had cancelled.
    """
    try:
        subscription = await repository.subscription_to_cancel()
    except NoSubscriptionToCancel as exc:
        raise HTTPException(status_code=404, detail="NO_ACTIVE_SUBSCRIPTION") from exc

    rebill_no = str(subscription.provider_subscription_id)
    try:
        await client.cancel_recurring(rebill_no=rebill_no)
    except PayAppError as exc:
        logger.error(
            "payapp cancellation failed; local subscription unchanged",
            extra={"payapp_errno": exc.errno, "transport": exc.transport},
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="PAYMENT_PROVIDER_UNAVAILABLE"
        ) from exc

    updated = await repository.mark_canceled()
    logger.info(
        "subscription cancelled at provider; paid period preserved",
        extra={"plan_id": updated.plan_id, "status": updated.status},
    )
    return await billing_status(repository)


@router.get("/payments", response_model=PaymentHistoryResponse)
async def payment_history(
    repository: Annotated[BillingRepository, Depends(get_billing_repository)],
) -> PaymentHistoryResponse:
    """This account's own payments. Scoped at the repository, not here.

    No provider identifiers are returned. `mul_no` and `rebill_no` are
    operational data: they identify a contract to PayApp, and there is
    nothing a customer can do with one that they cannot do better through
    this product.
    """
    rows = await repository.payments()
    return PaymentHistoryResponse(
        items=[
            PaymentRecord(
                paid_at=row.paid_at.isoformat(),
                plan_id=row.plan_id,
                amount_krw=row.amount_krw,
                status=row.status,
                failure_reason=row.failure_reason,
            )
            for row in rows
        ]
    )


__all__ = [
    "ACK_BODY",
    "BillingStatusResponse",
    "CheckoutRequest",
    "CheckoutResponse",
    "PaymentHistoryResponse",
    "get_payapp_client",
    "router",
]
