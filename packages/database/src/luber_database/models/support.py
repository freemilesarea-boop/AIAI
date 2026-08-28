"""Customer support inquiries, and room for the replies that follow.

Two tables. `support_tickets` is what a customer submitted;
`support_replies` is the conversation on it. The second is empty for now
— there is no operator interface yet — but it exists because retrofitting
a reply model onto a schema that assumed one message per ticket means
migrating live support history, and that is the kind of migration nobody
enjoys.

**The reference, not the id.** A ticket carries a short `reference` like
`SUP-3F9A2C71` alongside its UUID primary key. The customer sees the
reference and quotes it in email; the UUID never leaves the server. That
is not obfuscation standing in for authorisation — every query is scoped
to the owner regardless — it just means a support address in a shared
inbox is not also a list of valid identifiers to try.

**Deletion follows the account policy.** `user_id` is `ON DELETE
RESTRICT` by omission, matching `generations` and `projects`: closing an
account is anonymisation, and the ticket stays attached to a row that no
longer names anyone. Cascading here would delete the record of a
complaint at the moment the complainant left, which is exactly when it
matters most.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from luber_database.base import Base

#: Ceilings the API validates against and the columns enforce. Generous
#: enough for a real description of a real problem, bounded so a single
#: request cannot be a denial of service.
SUBJECT_MAX_LENGTH = 200
MESSAGE_MAX_LENGTH = 5000
#: Where the user was when the problem happened. Optional, and stored as
#: text the product never navigates to — it is a clue for whoever reads
#: the ticket, not a link the UI follows.
CONTEXT_URL_MAX_LENGTH = 500


def _utcnow() -> datetime:
    return datetime.now(UTC)


class SupportTicket(Base):
    """One inquiry, as the customer submitted it."""

    __tablename__ = "support_tickets"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    #: What the customer quotes. Short, unique, and unrelated to the
    #: primary key so that knowing one tells you nothing about another.
    reference: Mapped[str] = mapped_column(String(16), nullable=False, unique=True, index=True)

    #: Owner. RESTRICT by omission, as on `generations` and `projects` —
    #: see the module docstring.
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("users.id"), nullable=False, index=True
    )

    #: A `SupportCategory` value.
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    subject: Mapped[str] = mapped_column(String(SUBJECT_MAX_LENGTH), nullable=False)
    #: Plain text. Stored as submitted and rendered escaped — nothing in
    #: the product treats this as markup.
    message: Mapped[str] = mapped_column(Text, nullable=False)
    context_url: Mapped[str | None] = mapped_column(String(CONTEXT_URL_MAX_LENGTH))

    #: A `SupportStatus` value. Operator-owned: no request a customer can
    #: make carries this field.
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="OPEN", index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        server_default=func.now(),
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, server_default=func.now()
    )
    #: An operator's internal note. Never returned by any
    #: customer-facing route — it lives on the ticket rather than in the
    #: reply thread precisely because the reply thread is what the
    #: customer will eventually be shown.
    admin_note: Mapped[str | None] = mapped_column(Text)

    #: When it stopped needing attention. Null while open.
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        # The list query: this user's tickets, newest first.
        Index("ix_support_tickets_user_created", "user_id", "created_at"),
        # The operator queue, once one exists.
        Index("ix_support_tickets_status_created", "status", "created_at"),
    )


#: Who wrote a reply. Not an enum in `luber_schemas` because it describes
#: this table's rows rather than a value the product passes around.
AUTHOR_CUSTOMER = "CUSTOMER"
AUTHOR_OPERATOR = "OPERATOR"


class SupportReply(Base):
    """A message on a ticket, from either side.

    Nothing writes to this yet. It exists so that adding operator replies
    is a feature rather than a migration of live support history — the
    shape of a conversation is hard to add to a schema that assumed a
    single message, and support tickets are exactly the rows you least
    want to be migrating under load.
    """

    __tablename__ = "support_replies"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    ticket_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("support_tickets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    #: `AUTHOR_CUSTOMER` or `AUTHOR_OPERATOR`. Set by the server from who
    #: is authenticated, never from the request body.
    author_type: Mapped[str] = mapped_column(String(16), nullable=False)
    #: Null for an operator writing from a console with no BOORDA
    #: account of their own.
    author_user_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("users.id"))

    message: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        server_default=func.now(),
        index=True,
    )

    __table_args__ = (Index("ix_support_replies_ticket_created", "ticket_id", "created_at"),)


__all__ = [
    "AUTHOR_CUSTOMER",
    "AUTHOR_OPERATOR",
    "CONTEXT_URL_MAX_LENGTH",
    "MESSAGE_MAX_LENGTH",
    "SUBJECT_MAX_LENGTH",
    "SupportReply",
    "SupportTicket",
]
