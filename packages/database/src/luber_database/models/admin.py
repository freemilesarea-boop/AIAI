"""Operator-facing tables: downloads, audit, email campaigns.

Three tables and one column, all of them new. The audit found no role
model, no download tracking and no email infrastructure, so none of this
is a reshaping of something that existed — which is worth stating,
because the temptation with an admin console is to bolt it onto
whatever is nearest.

**`download_events` counts deliveries, not requests.** The download route
already streams or redirects; a row here records that it did. What it
deliberately does not do is count a browser's range requests as separate
downloads — an audio element fetching a file in pieces is one download,
and a metric that says otherwise is worse than no metric.

**The audit log records the actor, never the secret.** Its `metadata`
column is JSON and is written by hand at each call site rather than
serialising whatever object was to hand, precisely so a password hash or
a provider key cannot arrive in it by accident.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from luber_database.base import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class DownloadEvent(Base):
    """One delivered audio download.

    Written by the audio route when `download=true` — the save, not the
    play. Streaming for the in-page player is not a download and is not
    counted, which is the difference between "how many files did people
    take away" and "how many HTTP requests did the player make".
    """

    __tablename__ = "download_events"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("users.id"), nullable=False, index=True
    )
    generation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("generations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    #: `master`, `preview`, … — which asset was taken.
    asset_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    #: Denormalised from the plan in force at the time. A ticket asking
    #: "was this download allowed" is answerable later even after the
    #: account changes tier.
    plan_id: Mapped[str] = mapped_column(String(32), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        server_default=func.now(),
        index=True,
    )

    __table_args__ = (
        Index("ix_download_events_user_created", "user_id", "created_at"),
        Index("ix_download_events_generation", "generation_id", "created_at"),
    )


#: The actions worth recording. Named for what an operator did, not for
#: which endpoint they hit — an audit log is read months later by someone
#: reconstructing a decision.
ACTION_ADMIN_GRANTED = "ADMIN_GRANTED"
ACTION_ADMIN_REVOKED = "ADMIN_REVOKED"
ACTION_ADMIN_ROLE_CHANGED = "ADMIN_ROLE_CHANGED"
ACTION_SUPPORT_STATUS_CHANGED = "SUPPORT_STATUS_CHANGED"
ACTION_SUPPORT_NOTE_ADDED = "SUPPORT_NOTE_ADDED"
ACTION_EMAIL_CAMPAIGN_CREATED = "EMAIL_CAMPAIGN_CREATED"


class AdminAuditLog(Base):
    """What an operator did, and to whom.

    Append-only in practice: nothing in the product updates or deletes a
    row here. An audit trail that can be edited by the people it audits
    is decoration.
    """

    __tablename__ = "admin_audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    #: Who acted. Never null — an unattributed audit entry answers the
    #: least interesting half of the question.
    actor_user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("users.id"), nullable=False, index=True
    )
    #: Who or what it was done to. Null for actions with no user subject.
    target_user_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("users.id"), index=True
    )
    action: Mapped[str] = mapped_column(String(48), nullable=False, index=True)

    #: Structured context — the before and after of a role change, the
    #: audience of a campaign. Written explicitly at each call site, not
    #: serialised from an object, so a hash or a provider key cannot
    #: arrive here by accident.
    meta: Mapped[dict[str, str]] = mapped_column("metadata", JSON, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        server_default=func.now(),
        index=True,
    )

    __table_args__ = (
        Index("ix_admin_audit_actor_created", "actor_user_id", "created_at"),
        Index("ix_admin_audit_action_created", "action", "created_at"),
    )


#: Who a campaign went to.
AUDIENCE_ALL = "ALL"
AUDIENCE_PLAN = "PLAN"
AUDIENCE_USERS = "USERS"

#: Where a campaign got to. There is no SENT until a provider exists —
#: see `docs/ADMIN_CONSOLE.md`.
CAMPAIGN_DRAFT = "DRAFT"
CAMPAIGN_QUEUED = "QUEUED"
CAMPAIGN_SENT = "SENT"
CAMPAIGN_FAILED = "FAILED"

#: Ceilings the API validates against, matching the columns below. A
#: subject is a line; a body is a message, generously bounded so one
#: request cannot be a denial of service.
SUBJECT_MAX_LENGTH = 200
BODY_MAX_LENGTH = 20_000


class AdminEmailCampaign(Base):
    """A message an operator composed for some set of customers.

    The row is created and the recipients are resolved and counted; the
    sending is a separate step that no provider currently backs. That
    split is deliberate rather than incidental — the count is the thing
    an operator confirms against before a send, and it has to be a
    server-side fact rather than a number the browser worked out.
    """

    __tablename__ = "admin_email_campaigns"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    created_by: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("users.id"), nullable=False, index=True
    )

    subject: Mapped[str] = mapped_column(String(SUBJECT_MAX_LENGTH), nullable=False)
    #: Plain text. Nothing renders this as HTML, here or in a mail body —
    #: see the security note in `docs/ADMIN_CONSOLE.md`.
    body: Mapped[str] = mapped_column(Text, nullable=False)

    audience_type: Mapped[str] = mapped_column(String(16), nullable=False)
    #: A `PlanId` when `audience_type` is PLAN, else null.
    audience_plan_id: Mapped[str | None] = mapped_column(String(32))
    #: Resolved server-side when the campaign is created. What the
    #: operator confirmed against.
    recipient_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    status: Mapped[str] = mapped_column(String(16), nullable=False, default=CAMPAIGN_DRAFT)
    #: Why it did not send, when it did not.
    failure_reason: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        server_default=func.now(),
        index=True,
    )
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("ix_admin_campaigns_created", "created_at"),)


__all__ = [
    "ACTION_ADMIN_GRANTED",
    "ACTION_ADMIN_REVOKED",
    "ACTION_ADMIN_ROLE_CHANGED",
    "ACTION_EMAIL_CAMPAIGN_CREATED",
    "ACTION_SUPPORT_NOTE_ADDED",
    "ACTION_SUPPORT_STATUS_CHANGED",
    "AUDIENCE_ALL",
    "AUDIENCE_PLAN",
    "AUDIENCE_USERS",
    "BODY_MAX_LENGTH",
    "CAMPAIGN_DRAFT",
    "CAMPAIGN_FAILED",
    "CAMPAIGN_QUEUED",
    "CAMPAIGN_SENT",
    "SUBJECT_MAX_LENGTH",
    "AdminAuditLog",
    "AdminEmailCampaign",
    "DownloadEvent",
]
