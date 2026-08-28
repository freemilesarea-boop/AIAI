"""Database access for users and sessions.

Separate from :class:`GenerationRepository` because it answers a
different question — who is this — and mixing identity lookups into the
product repository is how ownership checks end up scattered.

Nothing here hashes anything. Hashing lives in ``luber_api.security``;
this module stores and compares what it is given. That split keeps the
one place that decides *how* secrets are protected from being spread
across query code.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

from sqlalchemy import CursorResult, delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from luber_database.models.user import Session, User


class AuthRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ---- users -------------------------------------------------------

    async def get_user_by_email(self, email: str) -> User | None:
        """Look a user up by their normalised address.

        The caller normalises. Doing it here as well would hide a
        mismatch between what is stored and what is searched for.
        """
        result = await self._session.execute(
            # Closed accounts are excluded here, not filtered by the
            # caller: an anonymised row keeps a placeholder address, and
            # a login path that could find one would be a way back in.
            select(User).where(User.email == email, User.deleted_at.is_(None))
        )
        return result.scalar_one_or_none()

    async def get_user(self, user_id: UUID) -> User | None:
        return await self._session.get(User, user_id)

    async def create_user(
        self, *, email: str, password_hash: str, display_name: str | None = None
    ) -> User:
        """Insert a user. Uniqueness is the database's job, not a check.

        Deliberately no "does this email exist" query first: two
        concurrent signups would both pass it. The unique index is what
        actually prevents a duplicate, so the caller handles the
        integrity error instead.
        """
        user = User(email=email, password_hash=password_hash, display_name=display_name)
        self._session.add(user)
        await self._session.commit()
        await self._session.refresh(user)
        return user

    async def update_password_hash(self, user_id: UUID, password_hash: str) -> None:
        """Replace a stored hash — used to upgrade Argon2 parameters."""
        user = await self._session.get(User, user_id)
        if user is None:
            return
        user.password_hash = password_hash
        await self._session.commit()

    # ---- sessions ----------------------------------------------------

    async def update_display_name(self, user_id: UUID, name: str | None) -> User | None:
        """Set or clear the only profile field the schema has."""
        user = await self._session.get(User, user_id)
        if user is None or user.deleted_at is not None:
            return None
        user.display_name = name
        await self._session.commit()
        await self._session.refresh(user)
        return user

    async def close_account(self, user_id: UUID, *, now: datetime | None = None) -> User | None:
        """Close an account by emptying it, and end every session.

        Not a delete. `generations`, `projects` and `reference_audio`
        reference this row with NO ACTION, so removing it fails for any
        account that has made a song; and `billing_payments` cascades, so
        removing it would destroy the record of money that moved. Both
        reasons point the same way: keep the row, take the person out of
        it.

        What is cleared is everything that identifies a human — the
        address, the display name, the credential. What survives is a
        row that still satisfies the foreign keys and still carries the
        billing history someone may have to answer for later.

        The address becomes a placeholder under `.invalid`, which is
        reserved by RFC 2606 and can never be delivered to. It is derived
        from the user id so it is unique without a second lookup, and
        replacing the original frees it for signing up again.

        Idempotent: closing an already-closed account changes nothing and
        returns the row, so a retried request is safe.
        """
        at = now or datetime.now(UTC)
        user = await self._session.get(User, user_id)
        if user is None:
            return None
        if user.deleted_at is not None:
            return user

        user.deleted_at = at
        user.email = f"deleted-{user.id}@deleted.invalid"
        user.display_name = None
        # None, not a hash of something unguessable: the column is
        # nullable precisely so an account can be left with no usable
        # password rather than a hard-to-guess one.
        user.password_hash = None
        # Every session, not just the caller's — closing an account on
        # one device must not leave it open on another.
        await self._session.execute(delete(Session).where(Session.user_id == user_id))
        await self._session.commit()
        await self._session.refresh(user)
        return user

    async def create_session(
        self, *, token_hash: str, user_id: UUID, expires_at: datetime
    ) -> Session:
        session = Session(token_hash=token_hash, user_id=user_id, expires_at=expires_at)
        self._session.add(session)
        await self._session.commit()
        await self._session.refresh(session)
        return session

    async def get_session_user(self, token_hash: str, *, now: datetime) -> User | None:
        """The live user behind a token digest, or ``None``.

        Expiry is applied in the query rather than after it. A session
        that has passed ``expires_at`` must not authenticate even for the
        instant between loading it and checking it, and doing the
        comparison in SQL means there is no such instant.

        The join means a deleted user cannot authenticate either: the
        cascade removes the sessions, and a row without a live user
        simply does not come back.
        """
        result = await self._session.execute(
            select(User)
            .join(Session, Session.user_id == User.id)
            .where(
                Session.token_hash == token_hash,
                Session.expires_at > now,
                # A closed account authenticates nothing, even with a
                # cookie that was valid a moment ago.
                User.deleted_at.is_(None),
            )
        )
        return result.scalar_one_or_none()

    async def delete_session(self, token_hash: str) -> bool:
        """Invalidate one session. This is what logout actually is."""
        result = await self._session.execute(
            delete(Session).where(Session.token_hash == token_hash)
        )
        await self._session.commit()
        return bool(cast("CursorResult[Any]", result).rowcount)

    async def delete_sessions_for_user(self, user_id: UUID) -> int:
        """Every session for one user — "log out everywhere"."""
        result = await self._session.execute(delete(Session).where(Session.user_id == user_id))
        await self._session.commit()
        return int(cast("CursorResult[Any]", result).rowcount or 0)

    async def delete_expired_sessions(self, *, now: datetime) -> int:
        """Remove sessions past their expiry.

        Housekeeping only. These rows already fail to authenticate, so
        this reclaims space rather than enforcing anything — which is
        why it can never touch a user or any product data.
        """
        result = await self._session.execute(delete(Session).where(Session.expires_at <= now))
        await self._session.commit()
        return int(cast("CursorResult[Any]", result).rowcount or 0)

    async def count_sessions(self, user_id: UUID) -> int:
        result = await self._session.execute(select(Session).where(Session.user_id == user_id))
        return len(result.scalars().all())
