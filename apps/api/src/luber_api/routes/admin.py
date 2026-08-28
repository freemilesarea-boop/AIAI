"""The operator console API.

Every route here is behind `require_admin`, and the routes that change
who can administer are behind `require_super_admin`. Both come from
`luber_api.admin_security`, so the permission decision exists in one
place rather than being restated per handler.

Three things shape this module.

**Reads are aggregates, not exports.** The dashboard asks the database
for sums and counts; it never streams tables of rows to the browser. An
operator looking at revenue does not need every payment, and building
the console that way would make it the slowest thing running against
production.

**Writes are few and each one is audited.** Changing a role, moving a
ticket, composing a campaign. Everything else is read-only. The console
cannot delete an account, cancel a subscription, issue a refund or
trigger a charge — those are billing operations with their own paths and
their own consequences, and putting a button on them here would be
putting the most dangerous actions in the product behind the least
specific intent.

**Sending email is not implemented.** A campaign is composed, its
audience is resolved to a count, and the row is stored as a draft.
BOORDA has no mail provider configured, and a route that pretended to
send would be worse than one that says it did not.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from pydantic import BaseModel, Field, field_validator

from luber_api.admin_security import (
    get_admin_repository,
    get_super_admin_repository,
    require_admin,
    require_super_admin,
)
from luber_api.session import enforce_trusted_origin
from luber_database import admin_analytics as analytics
from luber_database.admin_repository import (
    AdminRepository,
    LastSuperAdmin,
    TargetNotFound,
    UserRow,
)
from luber_database.models.admin import (
    AUDIENCE_ALL,
    AUDIENCE_PLAN,
    AUDIENCE_USERS,
)
from luber_database.models.admin import (
    BODY_MAX_LENGTH as CAMPAIGN_BODY_MAX,
)
from luber_database.models.admin import (
    SUBJECT_MAX_LENGTH as CAMPAIGN_SUBJECT_MAX,
)
from luber_database.models.user import User
from luber_schemas.enums import SupportStatus, UserRole
from luber_schemas.plans import PlanId

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/admin", tags=["admin"])

#: Longest window a chart may request, in days.
#:
#: A year of daily buckets is 365 rows — a chart. Ten years is a report
#: nobody reads and a scan nobody budgeted for.
MAX_RANGE_DAYS = 366

#: How many accounts one campaign may name explicitly.
#:
#: The `USERS` audience is for a handful of people — a beta group, the
#: accounts affected by one incident. Reaching everyone is what `ALL` is
#: for, and it goes through the same confirmation.
MAX_EXPLICIT_RECIPIENTS = 500


# ── shared query parsing ─────────────────────────────────────────────


class _Range(BaseModel):
    """A resolved reporting window, in UTC, over Korean days."""

    start: datetime
    end: datetime
    start_day: str
    end_day: str


def resolve_range(
    granularity: analytics.Granularity | None = None,
    start: str | None = None,
    end: str | None = None,
    now: datetime | None = None,
) -> _Range:
    """Turn what the browser asked for into UTC bounds.

    Explicit dates win; otherwise the named period. Both are Korean days
    converted to UTC instants here, so every downstream query filters on
    a raw timestamp column and the index does the work.
    """
    at = now or datetime.now(UTC)
    if start or end:
        try:
            first = datetime.fromisoformat(start).date() if start else analytics.kst_today(at)
            last = datetime.fromisoformat(end).date() if end else analytics.kst_today(at)
        except ValueError as exc:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY, "Dates must be ISO (YYYY-MM-DD)."
            ) from exc
        if last < first:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "End is before start.")
        if (last - first).days > MAX_RANGE_DAYS:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                f"Range cannot exceed {MAX_RANGE_DAYS} days.",
            )
        begin, _ = analytics.kst_day_bounds(first)
        _, finish = analytics.kst_day_bounds(last)
        return _Range(
            start=begin, end=finish, start_day=first.isoformat(), end_day=last.isoformat()
        )

    begin, finish = analytics.period_bounds(granularity or "month", at)
    return _Range(
        start=begin,
        end=finish,
        start_day=(begin + analytics.KST_OFFSET).date().isoformat(),
        end_day=analytics.kst_today(at).isoformat(),
    )


RangeQuery = Annotated[
    analytics.Granularity | None,
    Query(description="day | week | month | year. Ignored when start/end are given."),
]


# ── response shapes ──────────────────────────────────────────────────


class BucketResponse(BaseModel):
    day: str
    value: int
    secondary: int = 0


class RangeResponse(BaseModel):
    start: str
    end: str


class RevenueResponse(BaseModel):
    range: RangeResponse
    total_krw: int
    payment_count: int
    #: First payment for a subscription versus every later one. Split in
    #: SQL from the payment history, because nothing in the billing path
    #: records which a payment was.
    new_krw: int
    new_count: int
    renewal_krw: int
    renewal_count: int
    series: list[BucketResponse]


class UsersSummary(BaseModel):
    total: int
    paid: int
    free: int
    new_in_range: int


class GenerationSummary(BaseModel):
    requested: int
    completed: int
    failed: int
    creators: int
    average_per_creator: float


class DashboardResponse(BaseModel):
    """Everything the landing page needs, in one request.

    One round trip rather than six. The operator opens the console and
    sees a whole picture or an error — never five panels and a spinner.
    """

    range: RangeResponse
    generated_at: str
    revenue_krw: int
    revenue_today_krw: int
    payment_count: int
    users: UsersSummary
    generations: GenerationSummary
    downloads: int
    support: dict[str, int]
    plans: list[dict[str, Any]]
    revenue_series: list[BucketResponse]
    generation_series: list[BucketResponse]


class AdminUserResponse(BaseModel):
    id: str
    email: str
    display_name: str | None
    role: str
    created_at: str
    deleted_at: str | None
    #: Null where the endpoint did not resolve one.
    #:
    #: The member list and the detail page join to the live subscription
    #: and fill these in. The role endpoints do not, and answer null
    #: rather than defaulting to Free — an administrator who pays for
    #: Basic is not on Free, and a response that says so is wrong in a
    #: way nothing downstream can detect.
    plan_id: str | None = None
    subscription_status: str | None = None


class AdminUserListResponse(BaseModel):
    items: list[AdminUserResponse]
    total: int


class AdminUserDetailResponse(BaseModel):
    user: AdminUserResponse
    activity: dict[str, int]


class AdminTicketResponse(BaseModel):
    reference: str
    user_email: str
    category: str
    subject: str
    message: str
    context_url: str | None
    status: str
    admin_note: str | None
    created_at: str
    updated_at: str
    resolved_at: str | None


class AdminTicketSummary(BaseModel):
    reference: str
    user_email: str
    category: str
    subject: str
    status: str
    created_at: str


class AdminTicketListResponse(BaseModel):
    items: list[AdminTicketSummary]
    total: int


class TicketUpdateRequest(BaseModel):
    """What an operator may change on a ticket.

    Status and an internal note. Not the subject, not the message, not
    the owner — a support record that the operator can rewrite is not a
    record.
    """

    model_config = {"extra": "forbid"}

    status: SupportStatus | None = None
    admin_note: str | None = Field(default=None, max_length=5000)

    @field_validator("admin_note")
    @classmethod
    def _tidy(cls, value: str | None) -> str | None:
        return (value or "").strip() or None


class RoleChangeRequest(BaseModel):
    """A role change, named by account id.

    The target is a `user_id` in the body and the *actor* is the
    session — those are different things, and only the second is
    identity. Nothing here can name who is making the change.
    """

    model_config = {"extra": "forbid"}

    user_id: UUID
    role: UserRole


class AdminGrantRequest(BaseModel):
    """Promote an existing account, addressed by email.

    Email is a lookup here, not a permission: the address finds the row,
    and the row's `role` column is what any later request is checked
    against.
    """

    model_config = {"extra": "forbid"}

    email: str = Field(min_length=3, max_length=320)
    role: UserRole = UserRole.ADMIN

    @field_validator("email")
    @classmethod
    def _normalise(cls, value: str) -> str:
        return value.strip().lower()


class AudienceRequest(BaseModel):
    """Who a campaign would reach.

    Separate from the campaign itself so the console can ask "how many
    people is this?" while the operator is still deciding what to write.
    Folding it into the create schema would force the browser to invent a
    subject and a body to ask a question about neither.
    """

    model_config = {"extra": "forbid"}

    audience_type: str = Field(pattern=f"^({AUDIENCE_ALL}|{AUDIENCE_PLAN}|{AUDIENCE_USERS})$")
    plan_id: PlanId | None = None
    user_ids: list[UUID] = Field(default_factory=list, max_length=MAX_EXPLICIT_RECIPIENTS)


class CampaignCreateRequest(AudienceRequest):
    subject: str = Field(min_length=1, max_length=CAMPAIGN_SUBJECT_MAX)
    body: str = Field(min_length=1, max_length=CAMPAIGN_BODY_MAX)

    @field_validator("subject", "body")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        """A subject of three spaces is an empty subject.

        `min_length` counts characters, which is not the same question.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("This field cannot be empty.")
        return stripped


class CampaignResponse(BaseModel):
    id: str
    subject: str
    body: str
    audience_type: str
    audience_plan_id: str | None
    recipient_count: int
    status: str
    created_by_email: str
    created_at: str
    sent_at: str | None
    #: Why nothing was sent. Present and non-null in every current
    #: response — see the module docstring.
    delivery_note: str | None = None


class CampaignListResponse(BaseModel):
    items: list[CampaignResponse]


class AudiencePreviewResponse(BaseModel):
    recipient_count: int


class AuditEntryResponse(BaseModel):
    id: str
    action: str
    actor_email: str
    target_email: str | None
    metadata: dict[str, str]
    created_at: str


class AuditListResponse(BaseModel):
    items: list[AuditEntryResponse]
    total: int


#: Returned on every campaign, so no caller can mistake a stored draft
#: for a delivered message.
NO_MAIL_PROVIDER = (
    "No email provider is configured. The campaign is saved as a draft and nothing was sent."
)


def _user_response(row: UserRow) -> AdminUserResponse:
    return AdminUserResponse(**row.__dict__)


# ── dashboard ────────────────────────────────────────────────────────


@router.get("/dashboard", response_model=DashboardResponse)
async def dashboard(
    repository: Annotated[AdminRepository, Depends(get_admin_repository)],
    granularity: RangeQuery = None,
    start: str | None = None,
    end: str | None = None,
) -> DashboardResponse:
    """The whole overview in one query set."""
    session = repository.session
    now = datetime.now(UTC)
    window = resolve_range(granularity, start, end, now)

    revenue, payments = await analytics.revenue_total(session, start=window.start, end=window.end)
    today_start, today_end = analytics.period_bounds("day", now)
    revenue_today, _ = await analytics.revenue_total(session, start=today_start, end=today_end)

    users = await analytics.user_totals(session, now=now)
    generations = await analytics.generation_totals(session, start=window.start, end=window.end)

    return DashboardResponse(
        range=RangeResponse(start=window.start_day, end=window.end_day),
        generated_at=now.isoformat(),
        revenue_krw=revenue,
        revenue_today_krw=revenue_today,
        payment_count=payments,
        users=UsersSummary(
            **users,
            new_in_range=await analytics.new_users(session, start=window.start, end=window.end),
        ),
        generations=GenerationSummary(**vars(generations)),
        downloads=await analytics.download_total(session, start=window.start, end=window.end),
        support=await analytics.support_counts(session),
        plans=await analytics.plan_distribution(session, now=now),
        revenue_series=[
            BucketResponse(**b.__dict__)
            for b in await analytics.revenue_series(session, start=window.start, end=window.end)
        ],
        generation_series=[
            BucketResponse(**b.__dict__)
            for b in await analytics.generation_series(session, start=window.start, end=window.end)
        ],
    )


# ── analytics ────────────────────────────────────────────────────────


@router.get("/analytics/revenue", response_model=RevenueResponse)
async def revenue_analytics(
    repository: Annotated[AdminRepository, Depends(get_admin_repository)],
    granularity: RangeQuery = None,
    start: str | None = None,
    end: str | None = None,
) -> RevenueResponse:
    session = repository.session
    window = resolve_range(granularity, start, end)
    total, count = await analytics.revenue_total(session, start=window.start, end=window.end)
    split = await analytics.revenue_split(session, start=window.start, end=window.end)
    return RevenueResponse(
        range=RangeResponse(start=window.start_day, end=window.end_day),
        total_krw=total,
        payment_count=count,
        **split,
        series=[
            BucketResponse(**b.__dict__)
            for b in await analytics.revenue_series(session, start=window.start, end=window.end)
        ],
    )


@router.get("/analytics/generations")
async def generation_analytics(
    repository: Annotated[AdminRepository, Depends(get_admin_repository)],
    granularity: RangeQuery = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """Generation volume.

    All zeroes while `GENERATION_ENABLED` is false, which is the correct
    answer and not an error — the console renders an empty chart.
    """
    session = repository.session
    window = resolve_range(granularity, start, end)
    return {
        "range": {"start": window.start_day, "end": window.end_day},
        "totals": vars(
            await analytics.generation_totals(session, start=window.start, end=window.end)
        ),
        "series": [
            b.__dict__
            for b in await analytics.generation_series(session, start=window.start, end=window.end)
        ],
    }


@router.get("/analytics/downloads")
async def download_analytics(
    repository: Annotated[AdminRepository, Depends(get_admin_repository)],
    granularity: RangeQuery = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    session = repository.session
    window = resolve_range(granularity, start, end)
    return {
        "range": {"start": window.start_day, "end": window.end_day},
        "total": await analytics.download_total(session, start=window.start, end=window.end),
        "series": [
            b.__dict__
            for b in await analytics.download_series(session, start=window.start, end=window.end)
        ],
    }


@router.get("/analytics/plans")
async def plan_analytics(
    repository: Annotated[AdminRepository, Depends(get_admin_repository)],
) -> dict[str, Any]:
    session = repository.session
    now = datetime.now(UTC)
    return {
        "distribution": await analytics.plan_distribution(session, now=now),
        "users": await analytics.user_totals(session, now=now),
    }


@router.get("/analytics/users")
async def user_analytics(
    repository: Annotated[AdminRepository, Depends(get_admin_repository)],
    granularity: RangeQuery = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    session = repository.session
    window = resolve_range(granularity, start, end)
    return {
        "range": {"start": window.start_day, "end": window.end_day},
        "totals": await analytics.user_totals(session),
        "new_in_range": await analytics.new_users(session, start=window.start, end=window.end),
        "series": [
            b.__dict__
            for b in await analytics.user_series(session, start=window.start, end=window.end)
        ],
    }


# ── users ────────────────────────────────────────────────────────────


@router.get("/users", response_model=AdminUserListResponse)
async def list_users(
    repository: Annotated[AdminRepository, Depends(get_admin_repository)],
    search: Annotated[str | None, Query(max_length=320)] = None,
    plan: PlanId | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> AdminUserListResponse:
    """Accounts, newest first, with the plan each is on.

    No password hash and no session token: `UserRow` is assembled field
    by field, so a column added to `users` later does not appear here by
    default.
    """
    rows, total = await repository.list_users(search=search, plan=plan, limit=limit, offset=offset)
    return AdminUserListResponse(items=[_user_response(r) for r in rows], total=total)


@router.get("/users/{user_id}", response_model=AdminUserDetailResponse)
async def get_user(
    user_id: Annotated[UUID, Path()],
    repository: Annotated[AdminRepository, Depends(get_admin_repository)],
) -> AdminUserDetailResponse:
    """One account, with lifetime activity counts.

    Counts, not content. An operator can see that an account generated
    forty tracks; the console does not hand them the tracks.
    """
    try:
        row = await repository.user_detail(user_id)
    except TargetNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Account not found.") from exc

    return AdminUserDetailResponse(
        user=_user_response(row),
        activity=await analytics.user_activity(repository.session, user_id),
    )


# ── support ──────────────────────────────────────────────────────────


@router.get("/support", response_model=AdminTicketListResponse)
async def list_tickets(
    repository: Annotated[AdminRepository, Depends(get_admin_repository)],
    ticket_status: Annotated[SupportStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> AdminTicketListResponse:
    rows, total = await repository.list_tickets(status=ticket_status, limit=limit, offset=offset)
    return AdminTicketListResponse(
        items=[
            AdminTicketSummary(
                reference=t.reference,
                user_email=email,
                category=t.category,
                subject=t.subject,
                status=t.status,
                created_at=t.created_at.isoformat(),
            )
            for t, email in rows
        ],
        total=total,
    )


def _ticket_detail(ticket: Any, email: str) -> AdminTicketResponse:
    return AdminTicketResponse(
        reference=ticket.reference,
        user_email=email,
        category=ticket.category,
        subject=ticket.subject,
        message=ticket.message,
        context_url=ticket.context_url,
        status=ticket.status,
        admin_note=ticket.admin_note,
        created_at=ticket.created_at.isoformat(),
        updated_at=ticket.updated_at.isoformat(),
        resolved_at=ticket.resolved_at.isoformat() if ticket.resolved_at else None,
    )


@router.get("/support/{reference}", response_model=AdminTicketResponse)
async def get_ticket(
    reference: str,
    repository: Annotated[AdminRepository, Depends(get_admin_repository)],
) -> AdminTicketResponse:
    try:
        ticket, email = await repository.get_ticket(reference)
    except TargetNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Inquiry not found.") from exc
    return _ticket_detail(ticket, email)


@router.patch("/support/{reference}", response_model=AdminTicketResponse)
async def update_ticket(
    reference: str,
    payload: TicketUpdateRequest,
    repository: Annotated[AdminRepository, Depends(get_admin_repository)],
    _origin: Annotated[None, Depends(enforce_trusted_origin)] = None,
) -> AdminTicketResponse:
    """Move a ticket, or attach an internal note. Both are audited."""
    if payload.status is None and payload.admin_note is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Nothing to change.")
    try:
        ticket = await repository.update_ticket(
            reference, status=payload.status, admin_note=payload.admin_note
        )
        _, email = await repository.get_ticket(reference)
    except TargetNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Inquiry not found.") from exc
    return _ticket_detail(ticket, email)


# ── email campaigns ──────────────────────────────────────────────────


def _campaign_response(campaign: Any, email: str) -> CampaignResponse:
    return CampaignResponse(
        id=str(campaign.id),
        subject=campaign.subject,
        body=campaign.body,
        audience_type=campaign.audience_type,
        audience_plan_id=campaign.audience_plan_id,
        recipient_count=campaign.recipient_count,
        status=campaign.status,
        created_by_email=email,
        created_at=campaign.created_at.isoformat(),
        sent_at=campaign.sent_at.isoformat() if campaign.sent_at else None,
        delivery_note=NO_MAIL_PROVIDER,
    )


@router.post("/email/audience", response_model=AudiencePreviewResponse)
async def preview_audience(
    payload: AudienceRequest,
    repository: Annotated[AdminRepository, Depends(get_admin_repository)],
    _origin: Annotated[None, Depends(enforce_trusted_origin)] = None,
) -> AudiencePreviewResponse:
    """How many accounts a campaign would reach, counted server-side.

    The operator confirms against this number, so it must not be
    something the browser worked out from a page of results.
    """
    return AudiencePreviewResponse(
        recipient_count=await repository.resolve_audience(
            audience_type=payload.audience_type,
            plan=payload.plan_id,
            user_ids=payload.user_ids,
        )
    )


@router.post(
    "/email/campaigns", response_model=CampaignResponse, status_code=status.HTTP_201_CREATED
)
async def create_campaign(
    payload: CampaignCreateRequest,
    repository: Annotated[AdminRepository, Depends(get_admin_repository)],
    _origin: Annotated[None, Depends(enforce_trusted_origin)] = None,
) -> CampaignResponse:
    """Compose a campaign. Nothing is sent.

    Stored as a draft with its resolved recipient count. There is no
    mail provider configured for BOORDA, and the response says so on
    every campaign rather than leaving the operator to infer it from a
    status string.
    """
    if payload.audience_type == AUDIENCE_PLAN and payload.plan_id is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "A plan audience needs a plan.")
    if payload.audience_type == AUDIENCE_USERS and not payload.user_ids:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "A user audience needs at least one account."
        )

    recipients = await repository.resolve_audience(
        audience_type=payload.audience_type, plan=payload.plan_id, user_ids=payload.user_ids
    )
    campaign = await repository.create_campaign(
        subject=payload.subject,
        body=payload.body,
        audience_type=payload.audience_type,
        plan=payload.plan_id,
        recipient_count=recipients,
    )
    logger.info(
        "email campaign drafted (not sent: no provider configured)",
        extra={"campaign_id": str(campaign.id), "recipients": recipients},
    )
    return _campaign_response(campaign, repository.actor.email)


@router.get("/email/campaigns", response_model=CampaignListResponse)
async def list_campaigns(
    repository: Annotated[AdminRepository, Depends(get_admin_repository)],
) -> CampaignListResponse:
    return CampaignListResponse(
        items=[_campaign_response(c, e) for c, e in await repository.list_campaigns()]
    )


# ── administrators (SUPER_ADMIN only) ────────────────────────────────


@router.get("/admins", response_model=list[AdminUserResponse])
async def list_admins(
    repository: Annotated[AdminRepository, Depends(get_super_admin_repository)],
) -> list[AdminUserResponse]:
    """Who currently holds a role. Super administrators only.

    An `ADMIN` is refused here — not because the list is secret, but
    because the page that shows it is the page that changes it, and the
    boundary is cleaner drawn at the resource than at the button.
    """
    return [_role_result(u) for u in await repository.list_admins()]


@router.post("/admins", response_model=AdminUserResponse, status_code=status.HTTP_201_CREATED)
async def grant_admin(
    payload: AdminGrantRequest,
    repository: Annotated[AdminRepository, Depends(get_super_admin_repository)],
    _origin: Annotated[None, Depends(enforce_trusted_origin)] = None,
) -> AdminUserResponse:
    """Promote an existing account.

    Only an existing account: this does not create users. An operator
    who mistypes an address gets a 404 rather than an invitation sent
    into the void, and there is no path here that produces a login.
    """
    target = await repository.find_by_email(payload.email)
    if target is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No account with that address.")
    if payload.role is UserRole.USER:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Use DELETE to revoke a role.")
    return _role_result(await repository.set_role(target.id, payload.role))


@router.patch("/admins", response_model=AdminUserResponse)
async def change_role(
    payload: RoleChangeRequest,
    repository: Annotated[AdminRepository, Depends(get_super_admin_repository)],
    _origin: Annotated[None, Depends(enforce_trusted_origin)] = None,
) -> AdminUserResponse:
    """Change an account's role, including one's own.

    Self-demotion is allowed and audited — an operator handing over may
    legitimately step down. What is refused is doing so while being the
    only super administrator left, because that locks everyone out and
    the recovery is a migration run by hand.
    """
    try:
        return _role_result(await repository.set_role(payload.user_id, payload.role))
    except TargetNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Account not found.") from exc
    except LastSuperAdmin as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "At least one super administrator must remain.",
        ) from exc


@router.delete("/admins/{user_id}", response_model=AdminUserResponse)
async def revoke_admin(
    user_id: Annotated[UUID, Path()],
    repository: Annotated[AdminRepository, Depends(get_super_admin_repository)],
    _origin: Annotated[None, Depends(enforce_trusted_origin)] = None,
) -> AdminUserResponse:
    """Return an account to `USER`.

    The account keeps existing — this removes a role, not a person. No
    route in this module deletes an account.
    """
    try:
        return _role_result(await repository.set_role(user_id, UserRole.USER))
    except TargetNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Account not found.") from exc
    except LastSuperAdmin as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "At least one super administrator must remain.",
        ) from exc


def _role_result(user: User) -> AdminUserResponse:
    """One account as the role endpoints describe it.

    No plan: these endpoints are about permission, and resolving a
    subscription to answer a question nobody asked would be a join per
    row on the administrator list.
    """
    return AdminUserResponse(
        id=str(user.id),
        email=user.email,
        display_name=user.display_name,
        role=user.role,
        created_at=user.created_at.isoformat(),
        deleted_at=user.deleted_at.isoformat() if user.deleted_at else None,
    )


# ── audit ────────────────────────────────────────────────────────────


@router.get("/audit", response_model=AuditListResponse)
async def audit_log(
    repository: Annotated[AdminRepository, Depends(get_admin_repository)],
    action: Annotated[str | None, Query(max_length=48)] = None,
    actor_id: UUID | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> AuditListResponse:
    """What operators have done, newest first.

    Readable by any administrator, not just super administrators: a log
    only some operators can read is a weaker deterrent than one they can
    all see. There is no route that writes or edits an entry.
    """
    rows = await repository.audit_log(action=action, actor_id=actor_id, limit=limit, offset=offset)
    return AuditListResponse(
        items=[
            AuditEntryResponse(
                id=str(entry.id),
                action=entry.action,
                actor_email=actor_email,
                target_email=target_email,
                metadata=entry.meta or {},
                created_at=entry.created_at.isoformat(),
            )
            for entry, actor_email, target_email in rows
        ],
        total=await repository.count_audit(action=action, actor_id=actor_id),
    )


__all__ = [
    "MAX_EXPLICIT_RECIPIENTS",
    "MAX_RANGE_DAYS",
    "NO_MAIL_PROVIDER",
    "AudienceRequest",
    "require_admin",
    "require_super_admin",
    "resolve_range",
    "router",
]
