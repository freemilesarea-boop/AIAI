"""Reading, and refusing to trust, what PayApp sends us.

This is the security boundary of the whole billing system. Everything on
the other side of it is a form POST from the public internet claiming
that money moved. The endpoint that receives it cannot require a session
— PayApp's servers have none — so authenticity rests entirely on the
checks in this module.

PayApp's documentation is explicit about the rule:
「userid, linkkey, linkval 값을 비교 확인하고 동일한 경우에만」 — compare
userid, linkkey and linkval, and proceed only when they match. All three
are compared in constant time. A comparison that returned early on the
first wrong byte would hand the integration secret to a patient caller
one byte at a time, and this endpoint is reachable by anyone.

Two further rules that are not about authenticity but about correctness:

**Unknown fields are ignored, not rejected.** PayApp's documentation
says fields may be added over time. A parser that rejected a
notification because it carried a new unrelated key would stop
processing real payments on the day PayApp shipped a feature. Required
known fields are validated explicitly; everything else is carried along
as opaque metadata.

**Nothing here decides entitlement.** This module answers "is this
notification genuine and what does it say". Whether it grants anything
is decided by the service, against the account's own records — a
notification that is perfectly authentic can still be for the wrong
amount, an unknown subscription, or a checkout that is already paid.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any

#: The pay_state values PayApp documents, plus the failure value it
#: sends to `failurl`.
#:
#: Only COMPLETED grants anything. The others are recorded so that a
#: cancellation or a pending virtual-account deposit is visible in the
#: ledger rather than being silently dropped — "we ignored it" and "we
#: never received it" must not look the same afterwards.


class PayState(IntEnum):
    REQUESTED = 1
    COMPLETED = 4
    REQUEST_CANCELED = 8
    APPROVAL_CANCELED = 9
    AWAITING_DEPOSIT = 10
    REQUEST_CANCELED_ALT = 32
    APPROVAL_CANCELED_ALT = 64
    PARTIAL_CANCEL = 70
    PARTIAL_CANCEL_ALT = 71
    #: Sent to `failurl` when a recurring approval fails. Documented for
    #: second and later cycles only — see `RECURRING_FAILURE_ONLY` below.
    RECURRING_FAILED = 99


#: Values that mean money was taken. Exactly one.
GRANTING_STATES: frozenset[int] = frozenset({int(PayState.COMPLETED)})

#: Values that mean money was given back or the request died.
REVERSING_STATES: frozenset[int] = frozenset(
    {
        int(PayState.REQUEST_CANCELED),
        int(PayState.APPROVAL_CANCELED),
        int(PayState.REQUEST_CANCELED_ALT),
        int(PayState.APPROVAL_CANCELED_ALT),
        int(PayState.PARTIAL_CANCEL),
        int(PayState.PARTIAL_CANCEL_ALT),
    }
)

#: PayApp: 「1회차 승인 실패는 Noti되지 않음」 — a first-cycle approval
#: failure is not notified to `failurl` at all. Recorded here as a named
#: constant because it is the reason PENDING_INITIAL_PAYMENT can only be
#: resolved by a success notification or by the passage of time, never by
#: waiting for a failure that will not come.
RECURRING_FAILURE_ONLY = True


class NotificationRejected(Exception):
    """The notification is not from our PayApp account, or is unreadable.

    Carries a machine-readable reason so the endpoint can record an
    anomaly, and a message that is deliberately vague about *which*
    check failed — telling a prober that the userid matched but the
    linkval did not is telling them where to keep guessing.
    """

    def __init__(self, reason: str) -> None:
        super().__init__("payapp notification rejected")
        self.reason = reason


#: Keys never written to our own records, whatever PayApp sends. Card
#: data is the provider's to hold, not ours: we are not a card processor
#: and storing a PAN — even a masked one — turns a billing table into a
#: compliance problem for no product benefit.
SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "card_num",
        "cardno",
        "cardNo",
        "cardpw",
        "cardPw",
        "expmonth",
        "expMonth",
        "expyear",
        "expYear",
        "cvc",
        "cvv",
        "buyerauthno",
        "buyerAuthNo",
        "encbill",
        "encBill",
        "linkkey",
        "linkval",
        "userpwd",
    }
)


def redact(payload: dict[str, Any]) -> dict[str, str]:
    """A copy safe to persist and to log.

    Secrets and card data are dropped entirely rather than masked. A
    masked value still records that the field was present and how long it
    was, and there is no question we would ever answer with it.
    """
    return {
        key: str(value)
        for key, value in payload.items()
        if key.lower() not in {k.lower() for k in SENSITIVE_KEYS}
    }


@dataclass(frozen=True)
class PayAppNotification:
    """One authenticated notification, normalised.

    Frozen because a notification is a record of something that already
    happened. Nothing downstream should be able to adjust the amount it
    reports on the way to the amount check.
    """

    pay_state: int
    #: PayApp's payment identifier. Present on payment notifications;
    #: absent on some failure notifications, which is why the event
    #: fingerprint cannot rest on it alone.
    mul_no: str | None
    #: PayApp's recurring registration identifier — the link back to a
    #: subscription we registered.
    rebill_no: str | None
    #: Amount PayApp says was charged, in KRW. Compared against the plan
    #: price by the service; never used to *set* anything.
    price: int | None
    goodname: str | None
    pay_date: str | None
    pay_type: str | None
    #: Our own opaque correlation id, echoed back through var1.
    correlation_id: str | None
    #: Everything else PayApp sent, redacted. Kept for reconstructing an
    #: incident, not for making decisions.
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def is_payment_complete(self) -> bool:
        return self.pay_state in GRANTING_STATES

    @property
    def is_recurring_failure(self) -> bool:
        return self.pay_state == int(PayState.RECURRING_FAILED)

    @property
    def is_reversal(self) -> bool:
        return self.pay_state in REVERSING_STATES


def _text(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _amount(payload: dict[str, Any], key: str) -> int | None:
    """PayApp sends the amount as a string. Parse it strictly.

    Returns None rather than 0 on anything unparseable: a missing amount
    and a zero amount must not look alike to the check that compares it
    against the plan price, because zero would silently pass a
    comparison written as ``price < expected``.
    """
    raw = _text(payload, key)
    if raw is None:
        return None
    cleaned = raw.replace(",", "")
    try:
        # Tolerates "19900.00" from a provider that decides to send
        # decimals, while refusing "19900원" or an empty string.
        return int(float(cleaned))
    except ValueError:
        return None


def authenticate(
    payload: dict[str, Any],
    *,
    expected_userid: str,
    expected_linkkey: str,
    expected_linkval: str,
) -> None:
    """Prove the notification came from our PayApp account, or raise.

    All three comparisons run regardless of whether an earlier one
    failed, and the failures are combined at the end. Returning at the
    first mismatch would make the response time depend on which check
    failed, which is a signal worth denying a prober.
    """
    userid = _text(payload, "userid") or ""
    linkkey = _text(payload, "linkkey") or ""
    linkval = _text(payload, "linkval") or ""

    userid_ok = hmac.compare_digest(userid, expected_userid)
    linkkey_ok = hmac.compare_digest(linkkey, expected_linkkey)
    linkval_ok = hmac.compare_digest(linkval, expected_linkval)

    if not userid_ok:
        raise NotificationRejected("INVALID_USERID")
    if not linkkey_ok or not linkval_ok:
        # One reason for both: which of the two secrets was wrong is not
        # information a caller who has neither should be given.
        raise NotificationRejected("INVALID_LINKVAL")


def parse(payload: dict[str, Any]) -> PayAppNotification:
    """Normalise an authenticated payload.

    Call only after :func:`authenticate`. Unknown keys are preserved in
    ``extra`` rather than rejected — PayApp documents that fields may be
    added, and refusing a real payment over an unfamiliar key would be a
    self-inflicted outage.
    """
    raw_state = _text(payload, "pay_state")
    if raw_state is None:
        raise NotificationRejected("MISSING_PAY_STATE")
    try:
        pay_state = int(raw_state)
    except ValueError as exc:
        raise NotificationRejected("MALFORMED_PAY_STATE") from exc

    known = {
        "userid",
        "linkkey",
        "linkval",
        "pay_state",
        "mul_no",
        "rebill_no",
        "price",
        "goodname",
        "pay_date",
        "pay_type",
        "var1",
        "var2",
    }
    extra = {k: v for k, v in redact(payload).items() if k not in known}

    return PayAppNotification(
        pay_state=pay_state,
        mul_no=_text(payload, "mul_no"),
        rebill_no=_text(payload, "rebill_no"),
        price=_amount(payload, "price"),
        goodname=_text(payload, "goodname"),
        pay_date=_text(payload, "pay_date"),
        pay_type=_text(payload, "pay_type"),
        # var1 carries our correlation id. It is a *correlation aid*: it
        # tells us which checkout this probably belongs to, and grants
        # nothing on its own. Anyone who can post to this endpoint can
        # put anything in var1, so every entitlement decision is made
        # against our own records afterwards.
        correlation_id=_text(payload, "var1"),
        extra=extra,
    )


def event_fingerprint(
    notification: PayAppNotification, *, kind: str, received_date: datetime | None = None
) -> str:
    """A stable identity for one notification, for the dedupe constraint.

    Built from provider identifiers rather than from anything of ours, so
    the same PayApp event replayed after a restart, a retry, or a
    redelivery days later collapses onto the same row.

    ``mul_no`` is the natural key and is used alone when present. Failure
    notifications may arrive without one; those fall back to the
    subscription, the state and the day, which is the finest identity
    available. Two genuinely distinct failures for the same subscription
    on the same day therefore collapse into one recorded event — the
    correct trade, because the alternative is a fingerprint that changes
    on every redelivery and defeats the constraint entirely.
    """
    if notification.mul_no:
        return f"{kind}:{notification.mul_no}:{notification.pay_state}"
    day = (received_date or datetime.now()).date().isoformat()
    return f"{kind}:rebill={notification.rebill_no}:{notification.pay_state}:{day}"


__all__ = [
    "GRANTING_STATES",
    "RECURRING_FAILURE_ONLY",
    "REVERSING_STATES",
    "SENSITIVE_KEYS",
    "NotificationRejected",
    "PayAppNotification",
    "PayState",
    "authenticate",
    "event_fingerprint",
    "parse",
    "redact",
]
