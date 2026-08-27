"""PayApp recurring billing: provider client, notification validation,
subscription state machine, billing policy and anomaly vocabulary.

Deliberately free of SQLAlchemy. What a payment *means* is decided here
and is testable without a database; where it is *stored* is
`luber_database.billing_repository`. The seam runs one way.
"""

from luber_billing.anomalies import AnomalyKind
from luber_billing.states import (
    CheckoutState,
    SubscriptionState,
    entitles,
    may_start_new,
    may_transition,
)

__all__ = [
    "AnomalyKind",
    "CheckoutState",
    "SubscriptionState",
    "entitles",
    "may_start_new",
    "may_transition",
]
