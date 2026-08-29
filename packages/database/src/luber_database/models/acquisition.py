"""Where visitors came from, and which of them became customers.

Three tables, each answering one question.

**`acquisition_visitors`** — one anonymous browser, over time. Holds the
first touch (immutable once set) and the most recent *non-direct* touch.
Those two are the whole attribution model: everything the console
reports is one of them.

**`acquisition_sessions`** — one attributable arrival. Enough to see
traffic over time and to reconstruct how a visitor's attribution came to
be what it is. Deliberately not pageviews: this records entering, not
browsing, because a row per click buys volume rather than insight.

**`acquisition_attributions`** — the snapshot taken when a visitor
became a user. Separate from the visitor row because reporting must not
change retroactively: a customer who signed up from a Google search and
later clicked an Instagram ad was *acquired* by Google, and last month's
report has to keep saying so.

**Payments are deliberately absent.** There is no conversion table for
money. A paid conversion is derived at query time as the earliest
successful `billing_payments` row for an attributed user — so the
verified billing path is untouched, its idempotency is inherited whole,
and a retried PayApp callback cannot double-count something we never
write.

**Identity.** `visitor_key` is a random UUID minted by BOORDA and kept
in a first-party cookie. No IP address, no user agent, no fingerprint:
nothing here identifies a person, and nothing here works across domains.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from luber_database.base import Base

#: Matches `MAX_VALUE_LENGTH` in `luber_schemas.acquisition`, which
#: truncates to fit rather than rejecting.
VALUE_LENGTH = 120
PATH_LENGTH = 200


def _utcnow() -> datetime:
    return datetime.now(UTC)


class AcquisitionVisitor(Base):
    """One anonymous browser, and the two touches worth remembering."""

    __tablename__ = "acquisition_visitors"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    #: The value in the first-party cookie. Random, ours, meaningless
    #: anywhere else.
    visitor_key: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False, unique=True, index=True)

    #: First touch. Written once, never updated — see the module
    #: docstring for why that is the point rather than an omission.
    first_source: Mapped[str] = mapped_column(String(VALUE_LENGTH), nullable=False)
    first_medium: Mapped[str] = mapped_column(String(VALUE_LENGTH), nullable=False)
    first_campaign: Mapped[str | None] = mapped_column(String(VALUE_LENGTH))
    first_content: Mapped[str | None] = mapped_column(String(VALUE_LENGTH))
    first_term: Mapped[str | None] = mapped_column(String(VALUE_LENGTH))

    #: Last *non-direct* touch. Null until one exists: a visitor who has
    #: only ever arrived directly has no last-touch source, and writing
    #: "direct" here would erase the distinction between "came back on
    #: their own" and "came back through a campaign".
    last_source: Mapped[str | None] = mapped_column(String(VALUE_LENGTH))
    last_medium: Mapped[str | None] = mapped_column(String(VALUE_LENGTH))
    last_campaign: Mapped[str | None] = mapped_column(String(VALUE_LENGTH))
    last_content: Mapped[str | None] = mapped_column(String(VALUE_LENGTH))
    last_term: Mapped[str | None] = mapped_column(String(VALUE_LENGTH))

    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, server_default=func.now()
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        server_default=func.now(),
        index=True,
    )

    #: Set when this browser signed up. Nullable forever — most visitors
    #: never do. Cleared when the account is closed, so a closed account
    #: leaves no link from a live cookie to a person.
    user_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("users.id"), index=True)

    __table_args__ = (Index("ix_acq_visitors_first_seen", "first_seen_at"),)


class AcquisitionSession(Base):
    """One arrival, already classified.

    The normalised attribution is stored rather than recomputed, so a
    later change to the classification rules cannot silently rewrite
    history that an operator has already read.
    """

    __tablename__ = "acquisition_sessions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    visitor_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("acquisition_visitors.id", ondelete="CASCADE"), nullable=False, index=True
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        server_default=func.now(),
        index=True,
    )

    source: Mapped[str] = mapped_column(String(VALUE_LENGTH), nullable=False)
    medium: Mapped[str] = mapped_column(String(VALUE_LENGTH), nullable=False)
    campaign: Mapped[str | None] = mapped_column(String(VALUE_LENGTH))
    content: Mapped[str | None] = mapped_column(String(VALUE_LENGTH))
    term: Mapped[str | None] = mapped_column(String(VALUE_LENGTH))

    #: Where they landed, query string already discarded.
    landing_path: Mapped[str] = mapped_column(String(PATH_LENGTH), nullable=False, default="/")
    #: Host only. A full referring URL is someone else's page address and
    #: can carry parameters we have no business keeping.
    referrer_host: Mapped[str | None] = mapped_column(String(VALUE_LENGTH))
    #: Denormalised so the "was this attributable" filter is an index
    #: lookup rather than a comparison against two string columns.
    is_direct: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        Index("ix_acq_sessions_started_source", "started_at", "source"),
        Index("ix_acq_sessions_source_medium", "source", "medium"),
        Index("ix_acq_sessions_campaign", "campaign"),
    )


class AcquisitionAttribution(Base):
    """How one account was acquired, frozen at signup.

    Keyed by user, one row each. The snapshot is what makes historical
    reporting stable: the visitor row keeps evolving as that browser
    comes back, and this does not.
    """

    __tablename__ = "acquisition_attributions"

    #: The user *is* the key. One account has one acquisition story, and
    #: a second row would be a second answer to a question with one.
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id"), primary_key=True)
    visitor_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("acquisition_visitors.id", ondelete="SET NULL"), index=True
    )

    first_source: Mapped[str] = mapped_column(String(VALUE_LENGTH), nullable=False)
    first_medium: Mapped[str] = mapped_column(String(VALUE_LENGTH), nullable=False)
    first_campaign: Mapped[str | None] = mapped_column(String(VALUE_LENGTH))
    first_content: Mapped[str | None] = mapped_column(String(VALUE_LENGTH))
    first_term: Mapped[str | None] = mapped_column(String(VALUE_LENGTH))

    #: The last non-direct touch known at signup. Null when the visitor
    #: only ever arrived directly.
    last_source: Mapped[str | None] = mapped_column(String(VALUE_LENGTH))
    last_medium: Mapped[str | None] = mapped_column(String(VALUE_LENGTH))
    last_campaign: Mapped[str | None] = mapped_column(String(VALUE_LENGTH))
    last_content: Mapped[str | None] = mapped_column(String(VALUE_LENGTH))
    last_term: Mapped[str | None] = mapped_column(String(VALUE_LENGTH))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        server_default=func.now(),
        index=True,
    )

    __table_args__ = (
        Index("ix_acq_attr_first", "first_source", "first_medium"),
        Index("ix_acq_attr_last", "last_source", "last_medium"),
    )


__all__ = [
    "PATH_LENGTH",
    "VALUE_LENGTH",
    "AcquisitionAttribution",
    "AcquisitionSession",
    "AcquisitionVisitor",
]
