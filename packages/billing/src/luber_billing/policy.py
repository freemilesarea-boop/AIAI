"""BOORDA's billing policy, written down rather than assumed.

PayApp's ``rebillRegist`` needs a ``rebillCycleMonth`` — which day of the
month to charge on — and a ``rebillExpire``. Neither has an obvious right
answer, and picking one silently inside a request handler would make the
policy an accident of whatever the first caller happened to pass.

So the decisions are here, each with the reason.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

#: PayApp's sentinel for "the last day of the month", whatever that is.
LAST_DAY_OF_MONTH = 90

#: How long a registered-but-unpaid checkout stays open.
#:
#: A user who opens the PayApp window and puts their phone down comes
#: back within minutes; one who has not paid in a day is not coming back.
#: Long enough that a slow bank authentication is never cut off, short
#: enough that an abandoned checkout does not block the account's single
#: subscription slot for a week.
CHECKOUT_ABANDON_HOURS = 24

#: How long after a period ends before a missing renewal is an anomaly.
#:
#: PayApp charges on a cycle day and notifies asynchronously; a
#: notification arriving a few hours late is normal operation, not an
#: incident. Two days is comfortably longer than any delay we should
#: tolerate silently and short enough that a genuinely missed renewal is
#: found the same week.
RENEWAL_GRACE_HOURS = 48

#: How long PayApp should keep the recurring contract alive.
#:
#: PayApp requires an end date; there is no "forever". Ten years is
#: chosen so no customer's subscription silently stops because a date we
#: picked arrived — the subscription ends when the user cancels or
#: payment fails, which are events we handle, not because a field
#: expired. Renewed well before then by any operator who is still here.
CONTRACT_YEARS = 10


def billing_cycle_day(started_at: datetime) -> int:
    """Which day of the month PayApp should charge on.

    The day the subscription started, so the customer is billed on their
    own anniversary rather than on a date we chose for them.

    Days 29, 30 and 31 collapse to PayApp's last-day sentinel. Someone
    who subscribes on the 31st cannot be billed on the 31st of February,
    and the alternatives are worse: skipping those months means free
    access, and clamping to the 28th moves everyone's billing date
    permanently earlier.
    """
    day = started_at.astimezone(UTC).day
    return LAST_DAY_OF_MONTH if day >= 29 else day


def contract_expiry(started_at: datetime) -> str:
    """The ``rebillExpire`` date, as PayApp's ``yyyy-mm-dd``."""
    at = started_at.astimezone(UTC)
    # Day-arithmetic rather than a calendar library: the exact date a
    # decade out does not matter, only that it is far away and valid.
    return (at + timedelta(days=365 * CONTRACT_YEARS)).date().isoformat()


def paid_period(paid_at: datetime) -> tuple[datetime, datetime]:
    """The window one confirmed payment buys.

    Anchored on the payment, not on the calendar month: the customer paid
    on the 17th, so their songs run to the 17th. Tying a paid allowance
    to calendar months would give a customer who subscribes on the 28th
    three days for a full month's price.

    Approximated as 31 days rather than "same day next month" on purpose.
    The period must never end *before* the next PayApp charge, or an
    account would drop to Free for the hours between its period ending
    and the renewal notification arriving. Erring long costs a few days
    of access at worst; erring short breaks the product for a paying
    customer.
    """
    start = paid_at.astimezone(UTC)
    return start, start + timedelta(days=31)


#: Korean mobile numbers, the only form PayApp needs for `recvphone`.
#:
#: Deliberately narrow. This field is transmitted to a payment provider
#: and stored against a billing record; accepting free text would mean
#: storing whatever a caller typed and discovering at charge time that
#: PayApp cannot use it.
_PHONE = re.compile(r"^01[016789]-?\d{3,4}-?\d{4}$")


class InvalidPhoneNumber(ValueError):
    """The phone number is not one PayApp can send a payment request to."""


def normalise_phone(raw: str) -> str:
    """Digits only, validated as a Korean mobile number.

    Stored and transmitted without hyphens so the same number typed two
    ways is one number — otherwise "010-1234-5678" and "01012345678"
    would be two different billing contacts for one person.
    """
    candidate = raw.strip().replace(" ", "")
    if not _PHONE.match(candidate):
        raise InvalidPhoneNumber("Enter a Korean mobile number, e.g. 010-1234-5678.")
    return candidate.replace("-", "")


def mask_phone(phone: str) -> str:
    """For display and for logs. Never the whole number."""
    digits = phone.replace("-", "")
    if len(digits) < 8:
        return "***"
    return f"{digits[:3]}****{digits[-4:]}"


__all__ = [
    "CHECKOUT_ABANDON_HOURS",
    "CONTRACT_YEARS",
    "LAST_DAY_OF_MONTH",
    "RENEWAL_GRACE_HOURS",
    "InvalidPhoneNumber",
    "billing_cycle_day",
    "contract_expiry",
    "mask_phone",
    "normalise_phone",
    "paid_period",
]
