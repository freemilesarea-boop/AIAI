"""Operator actions: roles, users, support, campaigns, audit.

Not owner-scoped, and that is the point — an administrator reads other
people's accounts by definition. What replaces owner-scoping as the
safety property is that every entry point here is reached only through
`require_admin`, and that the destructive ones write an audit row in the
same transaction as their effect.

**The lockout guard is the load-bearing part.** A permission system that
can be emptied is one bad afternoon from nobody being able to administer
anything, and the fix for that is a database statement run by whoever
still has shell access.

Getting it right takes more than a check before the write. Two
administrators demoting each other at the same instant each read a count
that was true when they started and false by the time they committed —
so `set_role` locks the super-administrator rows and puts the same
question in the `WHERE` clause of the update itself. The database
answers it while holding the write lock, and the losing demotion updates
zero rows rather than emptying the console. The lock alone is not
enough (SQLite has no row locks) and the condition alone is not enough
(a lock makes the PostgreSQL re-read happen); both are there because
neither covers the other's case.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from typing import cast as typing_cast
from uuid import UUID

from sqlalchemy import CursorResult, Result, Select, String, and_, cast, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from luber_database.models.admin import (
    ACTION_ADMIN_GRANTED,
    ACTION_ADMIN_REVOKED,
    ACTION_ADMIN_ROLE_CHANGED,
    ACTION_EMAIL_CAMPAIGN_CREATED,
    ACTION_SUPPORT_NOTE_ADDED,
    ACTION_SUPPORT_STATUS_CHANGED,
    AUDIENCE_ALL,
    AUDIENCE_PLAN,
    AUDIENCE_USERS,
    CAMPAIGN_DRAFT,
    AdminAuditLog,
    AdminEmailCampaign,
)
from luber_database.models.billing import Subscription
from luber_database.models.support import SupportTicket
from luber_database.models.user import User
from luber_schemas.enums import ADMIN_ROLES, SupportStatus, UserRole
from luber_schemas.plans import PlanId


class LastSuperAdmin(RuntimeError):
    """Refused: this would leave the console with no super administrator."""


class TargetNotFound(LookupError):
    """No such account, or no such ticket."""


@dataclass(frozen=True)
class UserRow:
    """One account as the user list shows it.

    Assembled explicitly. Returning the ORM row would put
    `password_hash` one serialisation mistake away from an operator's
    browser, and the admin console is precisely where that mistake would
    matter most.
    """

    id: str
    email: str
    display_name: str | None
    role: str
    created_at: str
    deleted_at: str | None
    plan_id: str
    subscription_status: str | None


def _live() -> Any:
    return User.deleted_at.is_(None)


def cast_result(result: Result[Any]) -> CursorResult[Any]:
    """Narrow an executed DML result so `rowcount` is reachable.

    `Session.execute` is annotated as returning `Result`, but an `UPDATE`
    always produces a `CursorResult`. The cast says so once rather than
    scattering `type: ignore` at each call site.
    """
    return typing_cast("CursorResult[Any]", result)


class AdminRepository:
    """Operator-facing reads and writes. Reached only via `require_admin`."""

    def __init__(self, session: AsyncSession, actor: User) -> None:
        self._session = session
        #: Who is acting. Every audit row is attributed to this account,
        #: taken from the session rather than from any request field.
        self._actor = actor

    @property
    def actor(self) -> User:
        return self._actor

    @property
    def session(self) -> AsyncSession:
        """The transaction this repository is working in.

        Exposed so the read-only aggregates in `admin_analytics` run on
        the same session as everything else in a request. They are
        functions rather than methods because they only ever compute —
        keeping them outside the class is what makes that visible at the
        call site — and they still need somewhere to run.
        """
        return self._session

    # ── audit ──────────────────────────────────────────────────────

    def _audit(
        self, action: str, *, target: UUID | None = None, meta: dict[str, Any] | None = None
    ) -> AdminAuditLog:
        """Record an action. Added to the session, committed by the caller.

        Not committed here on purpose: the audit row and the change it
        describes belong to one transaction, so a rolled-back action
        leaves no entry claiming it happened.
        """
        row = AdminAuditLog(
            actor_user_id=self._actor.id,
            target_user_id=target,
            action=action,
            # Stringified explicitly. Whatever a call site passes becomes
            # text, so a stray object cannot serialise something private
            # into the log.
            meta={k: str(v) for k, v in (meta or {}).items()},
        )
        self._session.add(row)
        return row

    def _audit_filters(
        self,
        *,
        action: str | None,
        actor_id: UUID | None,
        start: datetime | None,
        end: datetime | None,
    ) -> list[Any]:
        """The `WHERE` terms for an audit query.

        Shared by the page and its count so the two cannot drift. A total
        that answers a different question from the rows beneath it is
        worse than no total: the operator paginates through a filtered
        list using a number describing the unfiltered table.
        """
        terms: list[Any] = []
        if action:
            terms.append(AdminAuditLog.action == action)
        if actor_id:
            terms.append(AdminAuditLog.actor_user_id == actor_id)
        if start:
            terms.append(AdminAuditLog.created_at >= start)
        if end:
            terms.append(AdminAuditLog.created_at < end)
        return terms

    async def audit_log(
        self,
        *,
        action: str | None = None,
        actor_id: UUID | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[AdminAuditLog, str, str | None]]:
        """Audit entries with the acting and targeted addresses.

        Joined here rather than resolved per row in the route, which
        would be one query per entry on a page of a hundred.
        """
        target = aliased(User)
        statement = (
            select(AdminAuditLog, User.email, target.email)
            .join(User, User.id == AdminAuditLog.actor_user_id)
            .outerjoin(target, target.id == AdminAuditLog.target_user_id)
            .where(*self._audit_filters(action=action, actor_id=actor_id, start=start, end=end))
            .order_by(AdminAuditLog.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return [(row[0], row[1], row[2]) for row in (await self._session.execute(statement)).all()]

    async def count_audit(
        self,
        *,
        action: str | None = None,
        actor_id: UUID | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> int:
        statement = (
            select(func.count())
            .select_from(AdminAuditLog)
            .where(*self._audit_filters(action=action, actor_id=actor_id, start=start, end=end))
        )
        return int((await self._session.execute(statement)).scalar_one())

    # ── users ──────────────────────────────────────────────────────

    def _user_query(self, *, search: str | None, plan: PlanId | None, now: datetime) -> Select[Any]:
        """Users with their current plan, resolved in SQL.

        The join is to the live subscription window, so "which plan is
        this account on" is answered by the database rather than by
        fetching every user and asking per row.
        """
        live_sub = and_(
            Subscription.user_id == User.id,
            Subscription.status.in_(("ACTIVE", "CANCEL_PENDING")),
            Subscription.period_start <= now,
            Subscription.period_end > now,
        )
        statement = (
            select(User, Subscription.plan_id, Subscription.status)
            .outerjoin(Subscription, live_sub)
            .where(_live())
        )
        if search:
            needle = f"%{search.strip().lower()}%"
            statement = statement.where(
                or_(
                    func.lower(User.email).like(needle),
                    func.lower(cast(User.id, String)).like(needle),
                )
            )
        if plan is not None:
            if plan is PlanId.FREE:
                # Free is the absence of a live paid subscription, not a
                # row saying "free".
                statement = statement.where(
                    or_(Subscription.plan_id.is_(None), Subscription.plan_id == PlanId.FREE.value)
                )
            else:
                statement = statement.where(Subscription.plan_id == plan.value)
        return statement

    async def list_users(
        self,
        *,
        search: str | None = None,
        plan: PlanId | None = None,
        limit: int = 50,
        offset: int = 0,
        now: datetime | None = None,
    ) -> tuple[list[UserRow], int]:
        at = now or datetime.now(UTC)
        base = self._user_query(search=search, plan=plan, now=at)

        total = int(
            (
                await self._session.execute(select(func.count()).select_from(base.subquery()))
            ).scalar_one()
        )
        rows = (
            await self._session.execute(
                base.order_by(User.created_at.desc()).limit(limit).offset(offset)
            )
        ).all()

        return [
            UserRow(
                id=str(user.id),
                email=user.email,
                display_name=user.display_name,
                role=user.role,
                created_at=user.created_at.isoformat(),
                deleted_at=user.deleted_at.isoformat() if user.deleted_at else None,
                plan_id=plan_id or PlanId.FREE.value,
                subscription_status=status,
            )
            for user, plan_id, status in rows
        ], total

    async def get_user(self, user_id: UUID) -> User:
        user = await self._session.get(User, user_id)
        if user is None:
            raise TargetNotFound(str(user_id))
        return user

    async def user_detail(self, user_id: UUID, *, now: datetime | None = None) -> UserRow:
        """One account in the same shape the list uses.

        Built from the same query as `list_users`, so the detail page and
        the row that led to it agree about which plan the account is on.
        A closed account is not found: it has been anonymised, and the
        console has nothing useful to show for it.
        """
        at = now or datetime.now(UTC)
        row = (
            await self._session.execute(
                self._user_query(search=None, plan=None, now=at).where(User.id == user_id)
            )
        ).first()
        if row is None:
            raise TargetNotFound(str(user_id))
        user, plan_id, status = row
        return UserRow(
            id=str(user.id),
            email=user.email,
            display_name=user.display_name,
            role=user.role,
            created_at=user.created_at.isoformat(),
            deleted_at=user.deleted_at.isoformat() if user.deleted_at else None,
            plan_id=plan_id or PlanId.FREE.value,
            subscription_status=status,
        )

    async def find_by_email(self, email: str) -> User | None:
        result = await self._session.execute(select(User).where(User.email == email, _live()))
        return result.scalar_one_or_none()

    # ── roles ──────────────────────────────────────────────────────

    async def list_admins(self) -> list[User]:
        result = await self._session.execute(
            select(User)
            .where(User.role.in_([r.value for r in ADMIN_ROLES]), _live())
            .order_by(User.created_at)
        )
        return list(result.scalars().all())

    async def _count_super_admins(self, *, excluding: UUID | None = None) -> int:
        statement = select(func.count(User.id)).where(
            User.role == UserRole.SUPER_ADMIN.value, _live()
        )
        if excluding is not None:
            statement = statement.where(User.id != excluding)
        return int((await self._session.execute(statement)).scalar_one())

    async def _lock_super_admins(self) -> None:
        """Serialise concurrent role changes against each other.

        `SELECT ... FOR UPDATE` over the super-administrator rows. A
        second transaction attempting the same demotion blocks here until
        the first commits, and then re-reads the committed rows — so its
        own count sees the change rather than the world as it was when it
        started.

        SQLAlchemy renders nothing for this on SQLite, which has no row
        locks. That is why it is not the only guard: the conditional
        write in `set_role` is what holds where this does not.
        """
        await self._session.execute(
            select(User.id)
            .where(User.role == UserRole.SUPER_ADMIN.value, _live())
            .with_for_update()
        )

    def _another_super_admin_exists(self, *, excluding: UUID) -> Any:
        """A condition, evaluated by the database at write time.

        The load-bearing half of the lockout guard. A count read before
        the write is a decision made about a world that may have changed
        by the time the write lands; putting the same question in the
        `WHERE` clause means the database answers it while holding the
        write lock, and a demotion that would empty the console updates
        zero rows instead.
        """
        other = aliased(User)
        return (
            select(other.id)
            .where(
                other.role == UserRole.SUPER_ADMIN.value,
                other.deleted_at.is_(None),
                other.id != excluding,
            )
            .exists()
        )

    async def set_role(self, user_id: UUID, role: UserRole) -> User:
        """Grant, change or revoke a role, and record who did it.

        Refuses to remove the last super administrator, including when
        the caller is removing their own — locking yourself out is still
        locking everyone out.
        """
        user = await self.get_user(user_id)
        previous = user.role
        if previous == role.value:
            return user

        losing_super = previous == UserRole.SUPER_ADMIN.value and role is not UserRole.SUPER_ADMIN

        conditions: list[Any] = [User.id == user_id, User.role == previous]
        if losing_super:
            # Two guards for one property, because neither is sufficient
            # alone. The lock serialises concurrent transactions where
            # the database has row locks; the condition below is checked
            # by the database at the moment of the write, which is what
            # holds on SQLite and what catches any interleaving the lock
            # does not.
            await self._lock_super_admins()
            if await self._count_super_admins(excluding=user_id) == 0:
                raise LastSuperAdmin("at least one super administrator must remain")
            conditions.append(self._another_super_admin_exists(excluding=user_id))

        result = await self._session.execute(
            update(User).where(*conditions).values(role=role.value)
        )
        # `rowcount` is how the database reports whether the guard above
        # held: zero means the condition was false when the write ran.
        if cast_result(result).rowcount == 0:
            # Somebody else changed this row first. If we were removing a
            # super administrator, theirs may have been the demotion that
            # made ours the last one.
            await self._session.rollback()
            if losing_super:
                raise LastSuperAdmin("at least one super administrator must remain")
            raise TargetNotFound(str(user_id))

        if role is UserRole.USER:
            action = ACTION_ADMIN_REVOKED
        elif previous == UserRole.USER.value:
            action = ACTION_ADMIN_GRANTED
        else:
            action = ACTION_ADMIN_ROLE_CHANGED
        self._audit(action, target=user_id, meta={"from": previous, "to": role.value})

        await self._session.commit()
        await self._session.refresh(user)
        return user

    # ── support ────────────────────────────────────────────────────

    async def list_tickets(
        self,
        *,
        status: SupportStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[tuple[SupportTicket, str]], int]:
        """Tickets with the reporter's address, newest first."""
        base = select(SupportTicket, User.email).join(User, User.id == SupportTicket.user_id)
        if status is not None:
            base = base.where(SupportTicket.status == status.value)

        total = int(
            (
                await self._session.execute(select(func.count()).select_from(base.subquery()))
            ).scalar_one()
        )
        rows = (
            await self._session.execute(
                base.order_by(SupportTicket.created_at.desc()).limit(limit).offset(offset)
            )
        ).all()
        return [(t, e) for t, e in rows], total

    async def get_ticket(self, reference: str) -> tuple[SupportTicket, str]:
        result = await self._session.execute(
            select(SupportTicket, User.email)
            .join(User, User.id == SupportTicket.user_id)
            .where(SupportTicket.reference == reference)
        )
        row = result.one_or_none()
        if row is None:
            raise TargetNotFound(reference)
        return row[0], row[1]

    async def update_ticket(
        self,
        reference: str,
        *,
        status: SupportStatus | None = None,
        admin_note: str | None = None,
        now: datetime | None = None,
    ) -> SupportTicket:
        at = now or datetime.now(UTC)
        ticket, _ = await self.get_ticket(reference)

        if status is not None and status.value != ticket.status:
            self._audit(
                ACTION_SUPPORT_STATUS_CHANGED,
                target=ticket.user_id,
                meta={"reference": reference, "from": ticket.status, "to": status.value},
            )
            ticket.status = status.value
            ticket.resolved_at = (
                at if status in {SupportStatus.RESOLVED, SupportStatus.CLOSED} else None
            )

        if admin_note is not None:
            # The note's text is not audited — only that one was written.
            # An audit log is a record of actions, not a second copy of
            # the content those actions were about.
            self._audit(
                ACTION_SUPPORT_NOTE_ADDED, target=ticket.user_id, meta={"reference": reference}
            )
            ticket.admin_note = admin_note or None

        ticket.updated_at = at
        await self._session.commit()
        await self._session.refresh(ticket)
        return ticket

    # ── email campaigns ────────────────────────────────────────────

    async def resolve_audience(
        self,
        *,
        audience_type: str,
        plan: PlanId | None = None,
        user_ids: list[UUID] | None = None,
        now: datetime | None = None,
    ) -> int:
        """How many live accounts a campaign would reach.

        A count, not a list. The operator confirms against this number
        before sending, so it has to be a server-side fact rather than
        something the browser worked out from a page of results.
        """
        at = now or datetime.now(UTC)
        if audience_type == AUDIENCE_ALL:
            statement = select(func.count(User.id)).where(_live())
        elif audience_type == AUDIENCE_USERS:
            statement = select(func.count(User.id)).where(_live(), User.id.in_(user_ids or []))
        elif audience_type == AUDIENCE_PLAN and plan is not None:
            base = self._user_query(search=None, plan=plan, now=at)
            statement = select(func.count()).select_from(base.subquery())
        else:
            return 0
        return int((await self._session.execute(statement)).scalar_one())

    async def create_campaign(
        self,
        *,
        subject: str,
        body: str,
        audience_type: str,
        plan: PlanId | None = None,
        recipient_count: int,
    ) -> AdminEmailCampaign:
        """Record a campaign. Sending is a separate, unimplemented step.

        Created as DRAFT because no mail provider is configured. The row
        is still worth writing: it is the record of what an operator
        composed and who it was aimed at, and it is what a send would
        later read from.
        """
        campaign = AdminEmailCampaign(
            created_by=self._actor.id,
            subject=subject,
            body=body,
            audience_type=audience_type,
            audience_plan_id=plan.value if plan else None,
            recipient_count=recipient_count,
            status=CAMPAIGN_DRAFT,
        )
        self._session.add(campaign)
        self._audit(
            ACTION_EMAIL_CAMPAIGN_CREATED,
            meta={
                "subject": subject,
                "audience": audience_type,
                "plan": plan.value if plan else "",
                "recipients": recipient_count,
            },
        )
        await self._session.commit()
        await self._session.refresh(campaign)
        return campaign

    async def list_campaigns(self, *, limit: int = 50) -> list[tuple[AdminEmailCampaign, str]]:
        result = await self._session.execute(
            select(AdminEmailCampaign, User.email)
            .join(User, User.id == AdminEmailCampaign.created_by)
            .order_by(AdminEmailCampaign.created_at.desc())
            .limit(limit)
        )
        return [(c, e) for c, e in result.all()]


__all__ = ["AdminRepository", "LastSuperAdmin", "TargetNotFound", "UserRow"]
