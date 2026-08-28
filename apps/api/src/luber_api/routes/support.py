"""Customer support: filing an inquiry and reading your own.

Three endpoints, all authenticated, all scoped to the caller.

The shape of the request is the security design. A ticket carries no
user id and no status: the owner comes from the session and the status
is always OPEN, so filing on someone else's behalf and opening a ticket
pre-marked resolved are both impossible rather than merely refused.
Reading is by `reference`, and the repository looks it up with the owner
in the same `WHERE` — a reference belonging to another account does not
resolve, so a valid identifier from somewhere else is a 404 like any
other.

Support text is stored and returned as plain text. Nothing in the
product renders it as markup, and the browser escapes it — a ticket
saying `<script>` is a customer describing a bug, not one.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, field_validator
from redis.asyncio import Redis

from luber_api.dependencies import get_redis, get_session_factory
from luber_api.rate_limit import RateLimitExceeded, enforce_rate_limit
from luber_api.session import enforce_trusted_origin, require_current_user
from luber_api.settings import ApiSettings, get_settings
from luber_database.models.support import (
    CONTEXT_URL_MAX_LENGTH,
    MESSAGE_MAX_LENGTH,
    SUBJECT_MAX_LENGTH,
    SupportTicket,
)
from luber_database.models.user import User
from luber_database.support_repository import SupportRepository, TicketNotFound
from luber_schemas.enums import SupportCategory, SupportStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/support", tags=["support"])

#: How many inquiries one account may file in a window.
#:
#: Generous for a person with a genuine problem — several attempts at
#: describing it, a follow-up, a second issue — and low enough that a
#: script cannot fill the operator queue. Keyed by account rather than
#: address: the endpoint is authenticated, so the account is the
#: meaningful identity and a shared office IP is not.
TICKET_RATE_LIMIT = 10
TICKET_RATE_WINDOW_SECONDS = 60 * 60


class TicketCreateRequest(BaseModel):
    """Everything a customer may say about a new inquiry.

    No user id, no status, no reference. Each of those is server-owned,
    and the way to keep them that way is for the schema to have nowhere
    to put them.
    """

    model_config = {"extra": "forbid"}

    category: SupportCategory
    subject: str = Field(min_length=1, max_length=SUBJECT_MAX_LENGTH)
    message: str = Field(min_length=1, max_length=MESSAGE_MAX_LENGTH)
    #: Where the customer was when it happened. A clue for whoever reads
    #: the ticket; the product never navigates to it.
    context_url: str | None = Field(default=None, max_length=CONTEXT_URL_MAX_LENGTH)

    @field_validator("subject", "message")
    @classmethod
    def _not_only_whitespace(cls, value: str) -> str:
        """A subject of three spaces is an empty subject.

        `min_length` counts characters, which is not the same question.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("This field cannot be empty.")
        return stripped

    @field_validator("context_url")
    @classmethod
    def _tidy_context(cls, value: str | None) -> str | None:
        return (value or "").strip() or None


class TicketResponse(BaseModel):
    """One inquiry as its owner sees it.

    `reference` and not `id`: the UUID is a database key and stays on
    the server. Built field by field so a column added to the table
    later cannot leak by default.
    """

    reference: str
    category: str
    subject: str
    message: str
    context_url: str | None
    status: str
    created_at: str
    updated_at: str
    resolved_at: str | None


class TicketSummary(BaseModel):
    """A row in the list. No message body — that is the detail view."""

    reference: str
    category: str
    subject: str
    status: str
    created_at: str


class TicketListResponse(BaseModel):
    items: list[TicketSummary]
    total: int


def _detail(ticket: SupportTicket) -> TicketResponse:
    return TicketResponse(
        reference=ticket.reference,
        category=ticket.category,
        subject=ticket.subject,
        message=ticket.message,
        context_url=ticket.context_url,
        status=ticket.status,
        created_at=ticket.created_at.isoformat(),
        updated_at=ticket.updated_at.isoformat(),
        resolved_at=ticket.resolved_at.isoformat() if ticket.resolved_at else None,
    )


def _summary(ticket: SupportTicket) -> TicketSummary:
    return TicketSummary(
        reference=ticket.reference,
        category=ticket.category,
        subject=ticket.subject,
        status=ticket.status,
        created_at=ticket.created_at.isoformat(),
    )


async def get_support_repository(
    request: Request,
    user: Annotated[User, Depends(require_current_user)],
) -> AsyncIterator[SupportRepository]:
    """A repository that can only see the caller's own tickets.

    Scoping is bound here, once. A route cannot obtain an unscoped
    support repository, so forgetting an ownership filter is not a
    mistake a route is able to make.
    """
    factory = get_session_factory(request)
    async with factory() as session:
        yield SupportRepository(session, user.id)


@router.post("/inquiries", response_model=TicketResponse, status_code=status.HTTP_201_CREATED)
async def create_inquiry(
    payload: TicketCreateRequest,
    request: Request,
    repository: Annotated[SupportRepository, Depends(get_support_repository)],
    redis: Annotated[Redis, Depends(get_redis)],
    settings: Annotated[ApiSettings, Depends(get_settings)],
    _origin: Annotated[None, Depends(enforce_trusted_origin)] = None,
) -> TicketResponse:
    """File an inquiry against the caller's own account."""
    await _throttle(redis, repository.owner)

    ticket = await repository.create_ticket(
        category=payload.category,
        subject=payload.subject,
        message=payload.message,
        context_url=payload.context_url,
    )
    logger.info(
        "support inquiry filed",
        # The reference and the category, not the subject or the body: a
        # customer's description of their problem is not log material.
        extra={"support_reference": ticket.reference, "support_category": ticket.category},
    )
    return _detail(ticket)


@router.get("/inquiries", response_model=TicketListResponse)
async def list_inquiries(
    repository: Annotated[SupportRepository, Depends(get_support_repository)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> TicketListResponse:
    """The caller's own inquiries, newest first."""
    tickets = await repository.list_tickets(limit=limit, offset=offset)
    return TicketListResponse(
        items=[_summary(t) for t in tickets], total=await repository.count_tickets()
    )


@router.get("/inquiries/{reference}", response_model=TicketResponse)
async def get_inquiry(
    reference: str,
    repository: Annotated[SupportRepository, Depends(get_support_repository)],
) -> TicketResponse:
    """One of the caller's own inquiries.

    A reference belonging to another account answers 404, the same as
    one that does not exist. Distinguishing them would confirm that
    somebody else's ticket is real.
    """
    try:
        ticket = await repository.get_ticket(reference)
    except TicketNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Inquiry not found.") from exc
    return _detail(ticket)


async def _throttle(redis: Redis, owner: object) -> None:
    """Bound how fast one account can file.

    A rate limiter that is down must not become a support outage: a
    customer with a real problem should still be able to reach us.
    """
    try:
        await enforce_rate_limit(
            redis,
            key=f"support:inquiry:{owner}",
            limit=TICKET_RATE_LIMIT,
            window_seconds=TICKET_RATE_WINDOW_SECONDS,
        )
    except RateLimitExceeded as exc:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Too many inquiries. Please wait and try again.",
            headers={"Retry-After": str(exc.retry_after_seconds)},
        ) from exc
    except HTTPException:
        raise
    except Exception:
        logger.warning("rate limiting unavailable for support inquiries", exc_info=True)


__all__ = [
    "TICKET_RATE_LIMIT",
    "SupportCategory",
    "SupportStatus",
    "TicketCreateRequest",
    "TicketListResponse",
    "TicketResponse",
    "router",
]
