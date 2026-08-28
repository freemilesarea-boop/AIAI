from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from luber_database.base import Base


def _utcnow() -> datetime:
    """Python-side default so the models work on SQLite as well as
    PostgreSQL. The server default stays for rows inserted outside the
    ORM."""
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    #: Argon2id, produced by ``luber_api.security.hash_password``. The
    #: encoded string carries its own salt and parameters. Nullable so a
    #: row can exist without a usable password — that is what makes an
    #: account impossible to log into, rather than merely hard.
    password_hash: Mapped[str | None] = mapped_column(String(255))
    display_name: Mapped[str | None] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), default=_utcnow
    )
    #: What this account may do in the operator console. A `UserRole`
    #: value; defaults to USER, so nobody is an administrator until
    #: somebody deliberately makes them one.
    #:
    #: A column rather than an email comparison: `if email == "..."` is a
    #: permission that cannot be revoked, leaves no audit trail, and
    #: transfers with the mailbox.
    role: Mapped[str] = mapped_column(String(24), nullable=False, default="USER", index=True)

    #: When the account was closed. NULL means live.
    #:
    #: Closing is anonymisation rather than deletion: three tables
    #: reference this row with NO ACTION, so a hard delete fails for any
    #: account that has made a song, and `billing_payments` cascades, so
    #: a hard delete would erase the record of money that moved. The row
    #: stays, emptied of personal data; this column is what stops it
    #: authenticating. See migration 0020.
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)


class Session(Base):
    """One logged-in browser, authoritative on the server.

    The row is the session. A cookie that no longer matches a live row
    authenticates nothing, which is what makes logout and expiry real
    rather than advisory — a client that keeps its cookie gains nothing.

    ``token_hash`` and not the token: a dump of this table must not hand
    an attacker working sessions.
    """

    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    #: SHA-256 of the opaque token the browser holds. Unique, so two
    #: sessions can never collide, and indexed because every
    #: authenticated request looks up exactly this column.
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), default=_utcnow
    )
    #: Server-side expiry, and the authoritative one. The cookie carries
    #: its own Max-Age, but a cookie is a client-side hint that a client
    #: is free to ignore.
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        # Cleanup sweeps by expiry; authentication filters on it too.
        Index("ix_sessions_expires_at", "expires_at"),
        # "Log out everywhere" and cascade deletes both scan by user.
        Index("ix_sessions_user_id", "user_id"),
    )
