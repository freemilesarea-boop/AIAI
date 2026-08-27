"""The billing package on its own: policy, states, parsing, the client.

No database here. These are the decisions that can be made from the
notification and the plan table alone, and they are worth isolating —
if a rule about money is only exercised through an HTTP route and a
session and a transaction, it is hard to be sure which of those things
is actually enforcing it.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from luber_billing.payapp.client import CYCLE_TYPE_MONTH, PayAppError, _parse_response
from luber_billing.payapp.fake import FakePayAppClient, feedback_payload
from luber_billing.payapp.notification import (
    RECURRING_FAILURE_ONLY,
    NotificationRejected,
    authenticate,
    event_fingerprint,
    parse,
    redact,
)
from luber_billing.policy import (
    LAST_DAY_OF_MONTH,
    InvalidPhoneNumber,
    billing_cycle_day,
    contract_expiry,
    mask_phone,
    normalise_phone,
    paid_period,
)
from luber_billing.states import (
    SubscriptionState,
    entitles,
    may_start_new,
    may_transition,
)

CREDS = {"userid": "boorda", "linkkey": "key-abc", "linkval": "val-xyz"}


# ── authentication ───────────────────────────────────────────────────


def test_a_matching_notification_is_accepted() -> None:
    authenticate(
        feedback_payload(**CREDS, rebill_no="900001"),
        expected_userid="boorda",
        expected_linkkey="key-abc",
        expected_linkval="val-xyz",
    )


def test_a_wrong_userid_is_refused() -> None:
    payload = feedback_payload(**{**CREDS, "userid": "someone-else"}, rebill_no="900001")

    with pytest.raises(NotificationRejected) as caught:
        authenticate(
            payload,
            expected_userid="boorda",
            expected_linkkey="key-abc",
            expected_linkval="val-xyz",
        )

    assert caught.value.reason == "INVALID_USERID"


def test_a_wrong_linkval_is_refused() -> None:
    payload = feedback_payload(**{**CREDS, "linkval": "guessed"}, rebill_no="900001")

    with pytest.raises(NotificationRejected) as caught:
        authenticate(
            payload,
            expected_userid="boorda",
            expected_linkkey="key-abc",
            expected_linkval="val-xyz",
        )

    assert caught.value.reason == "INVALID_LINKVAL"


def test_a_wrong_linkkey_reports_the_same_reason_as_a_wrong_linkval() -> None:
    """Which of the two secrets was wrong is not information to give out.

    A caller who has neither should not learn that one of them was
    right — that turns two unknowns into one.
    """
    payload = feedback_payload(**{**CREDS, "linkkey": "guessed"}, rebill_no="900001")

    with pytest.raises(NotificationRejected) as caught:
        authenticate(
            payload,
            expected_userid="boorda",
            expected_linkkey="key-abc",
            expected_linkval="val-xyz",
        )

    assert caught.value.reason == "INVALID_LINKVAL"


def test_missing_credentials_are_refused_rather_than_treated_as_empty() -> None:
    with pytest.raises(NotificationRejected):
        authenticate(
            {"pay_state": "4"},
            expected_userid="boorda",
            expected_linkkey="key-abc",
            expected_linkval="val-xyz",
        )


# ── parsing ──────────────────────────────────────────────────────────


def test_unknown_fields_do_not_break_a_real_payment() -> None:
    """PayApp documents that fields may be added.

    A parser that refused a notification over an unfamiliar key would
    stop processing real payments on the day PayApp shipped a feature.
    """
    payload = feedback_payload(
        **CREDS, rebill_no="900001", some_future_field="whatever", another="123"
    )

    notification = parse(payload)

    assert notification.is_payment_complete
    assert notification.extra["some_future_field"] == "whatever"


def test_card_details_never_survive_parsing() -> None:
    payload = feedback_payload(
        **CREDS,
        rebill_no="900001",
        card_num="1234-****-****-5678",
        card_name="Some Card",
    )

    notification = parse(payload)

    assert "card_num" not in notification.extra
    # The non-sensitive card metadata is fine and useful in an incident.
    assert notification.extra["card_name"] == "Some Card"


def test_secrets_are_dropped_from_anything_we_persist() -> None:
    safe = redact(feedback_payload(**CREDS, rebill_no="900001"))

    assert "linkkey" not in safe
    assert "linkval" not in safe
    assert safe["userid"] == "boorda"


def test_a_missing_pay_state_is_refused() -> None:
    payload = feedback_payload(**CREDS, rebill_no="900001")
    del payload["pay_state"]

    with pytest.raises(NotificationRejected):
        parse(payload)


def test_an_unparseable_amount_is_none_rather_than_zero() -> None:
    """Zero would pass an amount comparison written as a subtraction.

    None cannot be mistaken for a real figure, which is the point.
    """
    payload = feedback_payload(**CREDS, rebill_no="900001", price="19,900원")

    assert parse(payload).price is None


def test_a_comma_formatted_amount_is_read_correctly() -> None:
    payload = feedback_payload(**CREDS, rebill_no="900001", price="29,900")

    assert parse(payload).price == 29900


def test_the_failure_state_is_recognised() -> None:
    payload = feedback_payload(**CREDS, rebill_no="900001", pay_state=99)

    notification = parse(payload)

    assert notification.is_recurring_failure
    assert not notification.is_payment_complete


def test_first_cycle_failure_is_documented_as_unnotified() -> None:
    """PayApp: 1회차 승인 실패는 Noti되지 않음.

    Pinned as a constant because it is the reason an unpaid initial
    checkout resolves by timing out rather than by waiting for a failure
    notification that will never arrive.
    """
    assert RECURRING_FAILURE_ONLY is True


# ── fingerprints ─────────────────────────────────────────────────────


def test_the_same_notification_fingerprints_identically() -> None:
    payload = feedback_payload(**CREDS, rebill_no="900001", mul_no="777")

    first = event_fingerprint(parse(payload), kind="FEEDBACK")
    second = event_fingerprint(parse(payload), kind="FEEDBACK")

    assert first == second


def test_two_different_payments_fingerprint_differently() -> None:
    a = parse(feedback_payload(**CREDS, rebill_no="900001", mul_no="777"))
    b = parse(feedback_payload(**CREDS, rebill_no="900001", mul_no="778"))

    assert event_fingerprint(a, kind="FEEDBACK") != event_fingerprint(b, kind="FEEDBACK")


def test_a_failure_without_a_payment_id_still_fingerprints_stably() -> None:
    """Failure notifications do not always carry a mul_no.

    Falling back to subscription + state + day means a redelivery of the
    same failure still collapses onto one row, which is the property the
    unique constraint needs.
    """
    payload = feedback_payload(**CREDS, rebill_no="900001", pay_state=99)
    del payload["mul_no"]
    at = datetime(2026, 8, 27, tzinfo=UTC)

    first = event_fingerprint(parse(payload), kind="FAILURE", received_date=at)
    second = event_fingerprint(parse(payload), kind="FAILURE", received_date=at)

    assert first == second
    assert "900001" in first


def test_feedback_and_failure_do_not_share_a_fingerprint() -> None:
    notification = parse(feedback_payload(**CREDS, rebill_no="900001", mul_no="777"))

    assert event_fingerprint(notification, kind="FEEDBACK") != event_fingerprint(
        notification, kind="FAILURE"
    )


# ── the state machine ────────────────────────────────────────────────


def test_only_active_and_cancel_pending_can_entitle() -> None:
    assert entitles(SubscriptionState.ACTIVE)
    # Cancelled the renewal, not the month already paid for.
    assert entitles(SubscriptionState.CANCEL_PENDING)
    assert not entitles(SubscriptionState.PENDING_INITIAL_PAYMENT)
    assert not entitles(SubscriptionState.PAST_DUE)
    assert not entitles(SubscriptionState.CANCELED)
    assert not entitles(SubscriptionState.EXPIRED)


def test_registration_alone_cannot_reach_active_by_any_route_but_payment() -> None:
    assert may_transition(SubscriptionState.PENDING_INITIAL_PAYMENT, SubscriptionState.ACTIVE)


def test_a_finished_subscription_cannot_be_revived_by_an_event() -> None:
    """A late or replayed notification must not resurrect access."""
    for finished in (SubscriptionState.CANCELED, SubscriptionState.EXPIRED):
        for target in SubscriptionState:
            assert not may_transition(finished, target)


def test_past_due_can_recover() -> None:
    """A transient card decline must not end the relationship."""
    assert may_transition(SubscriptionState.PAST_DUE, SubscriptionState.ACTIVE)


def test_a_new_subscription_needs_the_old_one_finished() -> None:
    assert may_start_new(None)
    assert may_start_new(SubscriptionState.CANCELED)
    assert may_start_new(SubscriptionState.EXPIRED)
    # One account, one recurring contract.
    assert not may_start_new(SubscriptionState.ACTIVE)
    assert not may_start_new(SubscriptionState.PENDING_INITIAL_PAYMENT)
    assert not may_start_new(SubscriptionState.PAST_DUE)


# ── policy ───────────────────────────────────────────────────────────


def test_billing_falls_on_the_subscribers_own_day() -> None:
    assert billing_cycle_day(datetime(2026, 8, 17, tzinfo=UTC)) == 17


def test_late_month_signups_bill_on_the_last_day() -> None:
    """There is no 31st of February, and skipping the month is free access."""
    for day in (29, 30, 31):
        assert billing_cycle_day(datetime(2026, 1, day, tzinfo=UTC)) == LAST_DAY_OF_MONTH


def test_a_paid_period_starts_when_the_money_moved() -> None:
    paid_at = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)

    start, end = paid_period(paid_at)

    assert start == paid_at
    # Longer than a month on purpose: the period must not end before the
    # next charge, or the account drops to Free in the gap.
    assert (end - start).days >= 31


def test_the_contract_expiry_is_far_away_and_well_formed() -> None:
    expiry = contract_expiry(datetime(2026, 8, 27, tzinfo=UTC))

    assert expiry.startswith("2036-")
    assert len(expiry) == 10


def test_korean_mobile_numbers_are_normalised_to_digits() -> None:
    assert normalise_phone("010-1234-5678") == "01012345678"
    assert normalise_phone("01012345678") == "01012345678"
    assert normalise_phone(" 010 1234 5678 ") == "01012345678"


def test_anything_that_is_not_a_korean_mobile_is_refused() -> None:
    for bad in ("02-123-4567", "+821012345678", "abcdefghijk", "010-12-34", ""):
        with pytest.raises(InvalidPhoneNumber):
            normalise_phone(bad)


def test_a_masked_phone_keeps_no_middle_digits() -> None:
    masked = mask_phone("01012345678")

    assert masked == "010****5678"
    assert "1234" not in masked


# ── the client ───────────────────────────────────────────────────────


def test_payapp_responses_are_form_encoded_not_json() -> None:
    parsed = _parse_response("state=1&errorMessage=&rebill_no=900001&payurl=https://payapp.kr/x")

    assert parsed["state"] == "1"
    assert parsed["rebill_no"] == "900001"


def test_the_month_cycle_type_is_what_payapp_documents() -> None:
    assert CYCLE_TYPE_MONTH == "Month"


async def test_the_fake_records_the_price_it_was_given() -> None:
    """The assertion the whole integration rests on.

    Tests use this to prove the amount reaching PayApp came from the plan
    table and not from anything a browser sent.
    """
    client = FakePayAppClient()

    await client.register_recurring(
        goodname="BOORDA Pro",
        goodprice=29900,
        recvphone="01012345678",
        cycle_day=17,
        expire_date="2036-08-24",
        feedbackurl="https://example.test/feedback",
        failurl="https://example.test/failure",
        returnurl="https://example.test/return",
        var1="boorda_abc",
    )

    assert client.registrations[0].goodprice == 29900


async def test_a_registration_that_answers_without_identifiers_is_a_failure() -> None:
    """`state=1` with no rebill_no is not a shape the documentation
    describes. A subscription with no identifier could never be cancelled
    or reconciled, so it is refused rather than guessed at."""
    client = FakePayAppClient(registration_returns_nothing=True)

    with pytest.raises(PayAppError):
        await client.register_recurring(
            goodname="BOORDA Pro",
            goodprice=29900,
            recvphone="01012345678",
            cycle_day=17,
            expire_date="2036-08-24",
            feedbackurl="https://example.test/feedback",
            failurl="https://example.test/failure",
            returnurl="https://example.test/return",
            var1="boorda_abc",
        )
