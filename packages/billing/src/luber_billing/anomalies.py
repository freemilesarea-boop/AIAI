"""The names for things that went wrong with money.

A billing problem that is only a log line is a billing problem nobody
finds. Every one of these is written to a table an operator can query,
because the questions that get asked after an incident — "did this
happen to anyone else", "when did it start", "is it still happening" —
cannot be answered by grepping.

**An anomaly never grants entitlement.** That is the whole point of
recording one rather than making a judgement call in the moment: the
system stops, writes down exactly what it saw, and lets a person decide.
The alternative — guessing generously — means an amount mismatch becomes
free Creator access, and guessing meanly means a customer who paid does
not get what they paid for.
"""

from __future__ import annotations

from enum import StrEnum


class AnomalyKind(StrEnum):
    """Why a billing event could not be applied as it stood."""

    #: A notification names a rebill_no we never registered. Either PayApp
    #: sent us someone else's event, or a subscription exists at the
    #: provider that we lost. Both need a person.
    UNKNOWN_REBILL = "UNKNOWN_REBILL"
    #: A payment arrived that we cannot tie to any checkout or
    #: subscription of ours.
    UNKNOWN_PAYMENT = "UNKNOWN_PAYMENT"
    #: PayApp reported an amount that is not this plan's price. Never
    #: "corrected": the difference is the whole signal, and writing the
    #: expected number over the reported one destroys the evidence.
    AMOUNT_MISMATCH = "AMOUNT_MISMATCH"
    #: The same provider identifier arrived describing something
    #: different from what we already recorded for it.
    DUPLICATE_CONFLICT = "DUPLICATE_CONFLICT"
    #: An active subscription's period ended and no renewal payment was
    #: recorded. The failure mode webhooks alone cannot catch, because a
    #: notification that was never delivered leaves no trace.
    MISSING_EXPECTED_RENEWAL = "MISSING_EXPECTED_RENEWAL"
    #: Notification failed the linkkey/linkval check.
    INVALID_LINKVAL = "INVALID_LINKVAL"
    #: Notification failed the merchant id check.
    INVALID_USERID = "INVALID_USERID"
    #: A valid event asked for a subscription transition the state
    #: machine does not allow — an out-of-order or replayed event that
    #: would otherwise roll state backwards.
    INVALID_STATE_TRANSITION = "INVALID_STATE_TRANSITION"
    #: A checkout registered with PayApp that nobody ever paid for.
    ABANDONED_CHECKOUT = "ABANDONED_CHECKOUT"
    #: A subscription left in PAST_DUE with no resolution.
    UNRESOLVED_PAST_DUE = "UNRESOLVED_PAST_DUE"
    #: The notification could not be parsed at all.
    MALFORMED_NOTIFICATION = "MALFORMED_NOTIFICATION"


#: Anomalies that mean someone is probing the public endpoint rather than
#: that our own records are inconsistent. Separated because the response
#: differs: these want rate limiting and alerting, not reconciliation.
HOSTILE_KINDS: frozenset[AnomalyKind] = frozenset(
    {
        AnomalyKind.INVALID_LINKVAL,
        AnomalyKind.INVALID_USERID,
        AnomalyKind.MALFORMED_NOTIFICATION,
    }
)

#: Anomalies a person must look at before the account can be right again.
#: Everything else is either informational or self-resolving.
OPERATOR_ACTION_KINDS: frozenset[AnomalyKind] = frozenset(
    {
        AnomalyKind.UNKNOWN_REBILL,
        AnomalyKind.UNKNOWN_PAYMENT,
        AnomalyKind.AMOUNT_MISMATCH,
        AnomalyKind.DUPLICATE_CONFLICT,
        AnomalyKind.MISSING_EXPECTED_RENEWAL,
        AnomalyKind.INVALID_STATE_TRANSITION,
    }
)


__all__ = ["HOSTILE_KINDS", "OPERATOR_ACTION_KINDS", "AnomalyKind"]
