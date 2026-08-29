"""Recording where visitors came from, and binding that to accounts.

The attribution rules live here and nowhere else, because they are the
kind of thing that goes subtly wrong when restated:

**First touch is written once.** Once a visitor has an origin, it is
theirs permanently. A customer acquired by a Google search stays
acquired by Google even after they later click an ad, and last month's
report keeps saying what it said.

**Direct never overwrites a known source.** Somebody who arrives from an
Instagram ad, leaves, and comes back by typing the address was still
brought here by the ad. Treating that return as "direct" would credit
the campaign with nothing and hand the conversion to no-one — which is
how paid acquisition ends up looking worthless.

**Only a non-direct touch moves last-touch.** So last-touch answers "the
most recent thing that brought them back", which is the question it is
supposed to answer.

Signup binding takes a snapshot rather than a pointer. The visitor row
keeps changing; a conversion that already happened must not.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from luber_database.models.acquisition import (
    AcquisitionAttribution,
    AcquisitionSession,
    AcquisitionVisitor,
)
from luber_schemas.acquisition import Attribution


class AcquisitionRepository:
    """Writes for the acquisition tables. Reads live in `acquisition_analytics`."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record_visit(
        self,
        *,
        visitor_key: UUID,
        attribution: Attribution,
        landing_path: str,
        referrer_host: str | None,
        now: datetime | None = None,
    ) -> AcquisitionVisitor:
        """Record one arrival, creating the visitor if this is the first.

        Returns the visitor so the caller can confirm the cookie it
        should set. Safe to call concurrently: two first visits racing on
        the same key collide on the unique index, and the loser re-reads
        the winner's row rather than failing the request — a visitor
        whose analytics 500s is a visitor whose page broke for nothing.
        """
        at = now or datetime.now(UTC)

        visitor = await self._get(visitor_key)
        if visitor is None:
            visitor = AcquisitionVisitor(
                id=uuid.uuid4(),
                visitor_key=visitor_key,
                first_source=attribution.source,
                first_medium=attribution.medium,
                first_campaign=attribution.campaign,
                first_content=attribution.content,
                first_term=attribution.term,
                first_seen_at=at,
                last_seen_at=at,
            )
            # A first visit that is itself attributable is also the last
            # non-direct touch. Leaving it null would make a visitor who
            # signs up on their first visit have no last-touch at all.
            if not attribution.is_direct:
                _set_last(visitor, attribution)
            self._session.add(visitor)
            try:
                await self._session.flush()
            except IntegrityError:
                await self._session.rollback()
                existing = await self._get(visitor_key)
                if existing is None:
                    raise
                visitor = existing
                self._touch(visitor, attribution, at)
        else:
            self._touch(visitor, attribution, at)

        self._session.add(
            AcquisitionSession(
                id=uuid.uuid4(),
                visitor_id=visitor.id,
                started_at=at,
                source=attribution.source,
                medium=attribution.medium,
                campaign=attribution.campaign,
                content=attribution.content,
                term=attribution.term,
                landing_path=landing_path,
                referrer_host=referrer_host,
                is_direct=attribution.is_direct,
            )
        )
        await self._session.commit()
        await self._session.refresh(visitor)
        return visitor

    def _touch(self, visitor: AcquisitionVisitor, attribution: Attribution, at: datetime) -> None:
        """Apply a return visit to an existing visitor.

        `last_seen_at` always moves — the visit happened. The attribution
        moves only for a non-direct touch, which is the rule the whole
        model rests on.
        """
        visitor.last_seen_at = at
        if not attribution.is_direct:
            _set_last(visitor, attribution)

    async def _get(self, visitor_key: UUID) -> AcquisitionVisitor | None:
        result = await self._session.execute(
            select(AcquisitionVisitor).where(AcquisitionVisitor.visitor_key == visitor_key)
        )
        return result.scalar_one_or_none()

    async def bind_signup(
        self,
        *,
        visitor_key: UUID | None,
        user_id: UUID,
        now: datetime | None = None,
    ) -> AcquisitionAttribution | None:
        """Attach a new account to the browser that signed up.

        Returns None when there is nothing to attach — no cookie, or a
        cookie naming a visitor we have never seen. That is the ordinary
        case for anyone who blocks cookies, and it must not fail a
        signup: an account that cannot be created because analytics
        failed is an unforgivable trade.

        The identity comes from the caller's authenticated signup path,
        never from the request body. A cookie can only say *which
        browser*; it can never say which account.
        """
        if visitor_key is None:
            return None
        at = now or datetime.now(UTC)

        visitor = await self._get(visitor_key)
        if visitor is None:
            return None

        existing = await self._session.get(AcquisitionAttribution, user_id)
        if existing is not None:
            # One account, one acquisition story. A second signup on the
            # same user id cannot happen, but if it did, the first answer
            # stands.
            return existing

        attribution = AcquisitionAttribution(
            user_id=user_id,
            visitor_id=visitor.id,
            first_source=visitor.first_source,
            first_medium=visitor.first_medium,
            first_campaign=visitor.first_campaign,
            first_content=visitor.first_content,
            first_term=visitor.first_term,
            last_source=visitor.last_source,
            last_medium=visitor.last_medium,
            last_campaign=visitor.last_campaign,
            last_content=visitor.last_content,
            last_term=visitor.last_term,
            created_at=at,
        )
        self._session.add(attribution)

        # Only if this browser is not already somebody else's. Two
        # accounts from one browser is ordinary — a shared laptop — and
        # the second must not silently steal the first one's visitor.
        if visitor.user_id is None:
            visitor.user_id = user_id

        try:
            await self._session.commit()
        except IntegrityError:
            await self._session.rollback()
            return await self._session.get(AcquisitionAttribution, user_id)
        return attribution

    async def unlink_user(self, user_id: UUID) -> int:
        """Detach visitors from a closed account.

        Called when an account is closed. The attribution snapshot stays
        — it is aggregate marketing history about a row that no longer
        names anybody — but the live cookie stops pointing at the
        account, so a browser still carrying it is anonymous again.

        Returns how many visitor rows were detached.
        """
        result = await self._session.execute(
            update(AcquisitionVisitor)
            .where(AcquisitionVisitor.user_id == user_id)
            .values(user_id=None)
        )
        return int(getattr(result, "rowcount", 0) or 0)


def _set_last(visitor: AcquisitionVisitor, attribution: Attribution) -> None:
    """Copy a non-direct touch into the last-touch columns."""
    visitor.last_source = attribution.source
    visitor.last_medium = attribution.medium
    visitor.last_campaign = attribution.campaign
    visitor.last_content = attribution.content
    visitor.last_term = attribution.term


__all__ = ["AcquisitionRepository"]
