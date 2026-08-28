"""Support tickets, scoped to one account.

Same discipline as every other repository here: the owner is bound at
construction, not passed per method. A route cannot obtain an unscoped
support repository, so reading somebody else's ticket is not a mistake a
route is able to make.

That matters more here than usual. A support ticket contains whatever a
frustrated customer typed — order details, a phone number, a description
of something that went wrong with their payment — and it is exactly the
kind of record an enumeration attack goes looking for. So the lookup is
`reference AND user_id`, in one query. There is no "fetch then check"
anywhere in this module, because a fetch-then-check is one forgotten
line away from being a fetch.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from luber_database.models.support import SupportReply, SupportTicket
from luber_schemas.enums import SupportCategory, SupportStatus

#: Characters a reference is drawn from. No 0/O or 1/I: references get
#: read aloud and retyped from email, and a customer who mistypes theirs
#: cannot find their own ticket.
_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
_REFERENCE_LENGTH = 8
#: Collisions are vanishingly unlikely at 32^8, but "unlikely" is not
#: "handled" — the unique index is what guarantees it, and this bounds
#: the retry rather than looping forever.
_MAX_REFERENCE_ATTEMPTS = 5


class TicketNotFound(LookupError):
    """No such ticket for this account.

    Deliberately the same failure for "does not exist" and "belongs to
    someone else". A support system that distinguished them would answer
    "that ticket is not yours", which confirms the ticket exists.
    """


def new_reference() -> str:
    """A short identifier the customer can quote.

    `secrets`, not `random`: this is the string that appears in email
    subject lines and gets read over the phone, and a predictable
    sequence would let someone guess at the volume of tickets — and at
    which references are worth trying.
    """
    body = "".join(secrets.choice(_ALPHABET) for _ in range(_REFERENCE_LENGTH))
    return f"SUP-{body}"


class SupportRepository:
    """One account's support tickets."""

    def __init__(self, session: AsyncSession, owner: UUID) -> None:
        self._session = session
        self._owner = owner

    @property
    def owner(self) -> UUID:
        return self._owner

    async def create_ticket(
        self,
        *,
        category: SupportCategory,
        subject: str,
        message: str,
        context_url: str | None = None,
        now: datetime | None = None,
    ) -> SupportTicket:
        """File a ticket for this account.

        The owner comes from the repository, the status is always OPEN,
        and the reference is generated here. None of the three is
        reachable from a request body — a customer cannot file on
        someone else's behalf, cannot open a ticket already marked
        resolved, and cannot choose a reference.
        """
        at = now or datetime.now(UTC)

        for _ in range(_MAX_REFERENCE_ATTEMPTS):
            ticket = SupportTicket(
                reference=new_reference(),
                user_id=self._owner,
                category=category.value,
                subject=subject,
                message=message,
                context_url=context_url,
                status=SupportStatus.OPEN.value,
                created_at=at,
                updated_at=at,
            )
            self._session.add(ticket)
            try:
                await self._session.commit()
            except IntegrityError:
                # The unique index refused a reference collision. Roll
                # back and draw another rather than failing the customer.
                await self._session.rollback()
                continue
            stored = await self._session.execute(
                select(SupportTicket).where(SupportTicket.id == ticket.id)
            )
            return stored.scalar_one()

        raise RuntimeError("could not allocate a unique support reference")

    async def list_tickets(self, *, limit: int = 50, offset: int = 0) -> list[SupportTicket]:
        """This account's tickets, newest first."""
        result = await self._session.execute(
            select(SupportTicket)
            .where(SupportTicket.user_id == self._owner)
            .order_by(SupportTicket.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def count_tickets(self) -> int:
        result = await self._session.execute(
            select(func.count())
            .select_from(SupportTicket)
            .where(SupportTicket.user_id == self._owner)
        )
        return int(result.scalar_one())

    async def get_ticket(self, reference: str) -> SupportTicket:
        """One ticket of this account's, by reference.

        Both conditions in the same `WHERE`. Loading by reference and
        then comparing the owner would work today and would be one
        refactor away from not working — this cannot return another
        account's row at all.
        """
        result = await self._session.execute(
            select(SupportTicket).where(
                SupportTicket.reference == reference,
                SupportTicket.user_id == self._owner,
            )
        )
        ticket = result.scalar_one_or_none()
        if ticket is None:
            raise TicketNotFound(reference)
        return ticket

    async def replies_for(self, reference: str) -> list[SupportReply]:
        """The conversation on one of this account's tickets.

        Empty until an operator interface exists. Routed through
        `get_ticket` so the ownership check is the same one, rather than
        a second implementation of it that could drift.
        """
        ticket = await self.get_ticket(reference)
        result = await self._session.execute(
            select(SupportReply)
            .where(SupportReply.ticket_id == ticket.id)
            .order_by(SupportReply.created_at)
        )
        return list(result.scalars().all())


__all__ = ["SupportRepository", "TicketNotFound", "new_reference"]
