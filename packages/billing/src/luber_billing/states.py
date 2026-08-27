"""What a subscription is doing, and what it is allowed to do next.

The single most dangerous thing this system could have is a boolean
called ``is_paid``. A boolean cannot distinguish "we asked PayApp to
register a recurring contract" from "PayApp told us money moved", and
those two facts arrive minutes apart through completely different
channels — one is our own outbound API call, the other is an inbound
notification from PayApp's servers. A system that conflates them grants
paid access to anyone who can start a checkout and close the tab.

So the states are explicit and the transitions are enumerated. A
transition that is not listed here does not happen; the attempt is
recorded as an anomaly rather than applied.

**Registration is not payment.** ``cmd=rebillRegist`` returning
``state=1`` means one thing only: PayApp accepted the registration of a
recurring payment *request*. The customer has not authenticated, no card
has been charged, and PayApp's own documentation notes that first-cycle
approval failure is not even notified to ``failurl``. The only fact that
moves a subscription to ACTIVE is a validated ``pay_state=4`` feedback
notification.

**Cancellation is not expiry.** PayApp's ``rebillCancel`` stops the
*next* charge; it does not refund the one already taken. A subscription
the user cancels therefore stays usable until the period they paid for
ends. Modelling that as an immediate CANCELED would take away access
somebody paid for.
"""

from __future__ import annotations

from enum import StrEnum


class SubscriptionState(StrEnum):
    """Where a subscription is in its life."""

    #: A checkout exists and PayApp accepted the registration, but no
    #: payment has been confirmed. Grants nothing.
    PENDING_INITIAL_PAYMENT = "PENDING_INITIAL_PAYMENT"
    #: A payment has been confirmed and the paid period contains now.
    ACTIVE = "ACTIVE"
    #: A renewal charge failed. The subscription keeps its plan record
    #: but grants nothing, and — critically — its allowance period is
    #: not advanced. A failed payment must never look like a paid one.
    PAST_DUE = "PAST_DUE"
    #: The user asked to cancel. Auto-renew is off at PayApp; the period
    #: already paid for is still theirs.
    CANCEL_PENDING = "CANCEL_PENDING"
    #: Cancelled and the paid period has ended.
    CANCELED = "CANCELED"
    #: Ran out without an explicit cancellation — a lapsed PAST_DUE, or
    #: a registration that was never paid for.
    EXPIRED = "EXPIRED"


class CheckoutState(StrEnum):
    """Where one attempt to start a subscription is."""

    #: Row written before PayApp was called, so a checkout that dies
    #: mid-flight is still visible rather than being a gap.
    CREATED = "CREATED"
    #: PayApp accepted the registration and returned a payurl. The user
    #: may or may not go on to authenticate. Grants nothing.
    REGISTERED = "REGISTERED"
    #: A payment was confirmed against this checkout.
    COMPLETED = "COMPLETED"
    #: PayApp refused the registration, or the call failed.
    REGISTRATION_FAILED = "REGISTRATION_FAILED"
    #: Nobody paid within the window. Closed so the account can try again.
    ABANDONED = "ABANDONED"
    #: The user started a different checkout, or cancelled this one.
    SUPERSEDED = "SUPERSEDED"


#: Checkout states that still occupy the account's one open slot. A
#: second checkout while one of these is in force would risk two live
#: recurring contracts against one account.
OPEN_CHECKOUT_STATES: frozenset[CheckoutState] = frozenset(
    {CheckoutState.CREATED, CheckoutState.REGISTERED}
)

#: Subscription states that count as "the account already has one".
LIVE_SUBSCRIPTION_STATES: frozenset[SubscriptionState] = frozenset(
    {
        SubscriptionState.PENDING_INITIAL_PAYMENT,
        SubscriptionState.ACTIVE,
        SubscriptionState.PAST_DUE,
        SubscriptionState.CANCEL_PENDING,
    }
)

#: States in which the paid plan is actually granted — *if* the paid
#: period also contains the current moment. Both conditions are required:
#: an ACTIVE row whose period ended grants nothing, which is what makes a
#: missed renewal fail closed instead of granting a free month.
#:
#: CANCEL_PENDING is here deliberately. The user cancelled the renewal,
#: not the month they already paid for.
ENTITLING_STATES: frozenset[SubscriptionState] = frozenset(
    {SubscriptionState.ACTIVE, SubscriptionState.CANCEL_PENDING}
)


#: Every transition the system may make. Anything absent is a bug or an
#: attack, and is recorded as INVALID_STATE_TRANSITION rather than being
#: quietly applied.
_ALLOWED: dict[SubscriptionState, frozenset[SubscriptionState]] = {
    SubscriptionState.PENDING_INITIAL_PAYMENT: frozenset(
        {
            # The only path to paid access.
            SubscriptionState.ACTIVE,
            # Nobody ever paid.
            SubscriptionState.EXPIRED,
            SubscriptionState.CANCELED,
        }
    ),
    SubscriptionState.ACTIVE: frozenset(
        {
            # A renewal was confirmed: ACTIVE → ACTIVE, with a new period.
            SubscriptionState.ACTIVE,
            SubscriptionState.PAST_DUE,
            SubscriptionState.CANCEL_PENDING,
            # Cancelling when the paid period has *already* lapsed. There
            # is nothing left to preserve, so it ends outright rather
            # than passing through CANCEL_PENDING. Without this the row
            # would stay ACTIVE with auto-renew off — and an ACTIVE row
            # accepts renewal notifications, so a late one would revive
            # access the user had ended.
            SubscriptionState.CANCELED,
            SubscriptionState.EXPIRED,
        }
    ),
    SubscriptionState.PAST_DUE: frozenset(
        {
            # A later attempt succeeded. Recovery must be possible, or a
            # transient card decline would end the relationship.
            SubscriptionState.ACTIVE,
            SubscriptionState.PAST_DUE,
            SubscriptionState.CANCEL_PENDING,
            SubscriptionState.CANCELED,
            SubscriptionState.EXPIRED,
        }
    ),
    SubscriptionState.CANCEL_PENDING: frozenset(
        {
            SubscriptionState.CANCELED,
            # A cancellation that PayApp had already billed past.
            SubscriptionState.ACTIVE,
        }
    ),
    # Terminal *as far as notifications are concerned*. No PayApp event
    # may revive a finished subscription — that is what stops a late or
    # replayed notification from resurrecting access somebody ended.
    # Starting again is a deliberate act by the user, and goes through
    # `may_start_new` below.
    SubscriptionState.CANCELED: frozenset(),
    SubscriptionState.EXPIRED: frozenset(),
}


#: States from which the account may begin a *new* subscription.
#:
#: Separate from `_ALLOWED` on purpose. The account holds one
#: subscription record (`subscriptions.user_id` is unique), so signing up
#: again reuses the row — but only through a checkout the user
#: deliberately started, never through an inbound notification. What was
#: paid is not lost by the reuse: `billing_payments` and `billing_events`
#: are append-only and keep the whole history.
_MAY_RESTART: frozenset[SubscriptionState] = frozenset(
    {SubscriptionState.CANCELED, SubscriptionState.EXPIRED}
)


def may_start_new(current: SubscriptionState | None) -> bool:
    """Whether a checkout may claim this account's subscription record.

    None means the account has never had one. Otherwise the existing
    subscription must be finished — an account with a live subscription
    cannot start a second one, which is what keeps two recurring
    contracts from being billed against one person.
    """
    return current is None or current in _MAY_RESTART


def may_transition(current: SubscriptionState, target: SubscriptionState) -> bool:
    return target in _ALLOWED[current]


def entitles(state: SubscriptionState) -> bool:
    """Whether this state can grant a paid plan at all.

    Necessary, not sufficient — the caller must also check that the paid
    period contains the moment being asked about.
    """
    return state in ENTITLING_STATES


__all__ = [
    "ENTITLING_STATES",
    "LIVE_SUBSCRIPTION_STATES",
    "OPEN_CHECKOUT_STATES",
    "CheckoutState",
    "SubscriptionState",
    "entitles",
    "may_start_new",
    "may_transition",
]
