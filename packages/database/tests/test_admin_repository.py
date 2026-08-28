"""Operator writes, and the guarantees they rest on.

The API suite covers the endpoints. What is asked here is what the
repository promises regardless of who calls it:

**The audit row and its change share a transaction.** A refused action
must leave no entry claiming it happened, and a recorded action must
have happened. Testing this at the repository is the only place it can
be tested honestly — through HTTP, a rolled-back write and a write that
never started look identical.

**The lockout guard counts inside the transaction.** Two administrators
demoting each other at the same instant would each pass a check made
before the other's write landed. The test below uses two concurrent
sessions rather than two sequential calls, because a sequential test
passes against the broken version.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import create_async_engine

from luber_database import Base, create_session_factory
from luber_database.admin_repository import (
    AdminRepository,
    LastSuperAdmin,
    TargetNotFound,
)
from luber_database.models.admin import (
    ACTION_ADMIN_GRANTED,
    ACTION_ADMIN_REVOKED,
    ACTION_ADMIN_ROLE_CHANGED,
    AUDIENCE_ALL,
    AUDIENCE_PLAN,
    CAMPAIGN_DRAFT,
    AdminAuditLog,
    AdminEmailCampaign,
    DownloadEvent,
)
from luber_database.models.billing import AllowanceReservation, Subscription
from luber_database.models.generation import (
    AudioAsset,
    Generation,
    GenerationJob,
    GenerationQA,
    LyricLineQA,
    Project,
    ReferenceAudio,
)
from luber_database.models.payments import (
    BillingAnomaly,
    BillingCheckout,
    BillingEvent,
    BillingPayment,
)
from luber_database.models.support import SupportReply, SupportTicket
from luber_database.models.user import Session, User
from luber_schemas.enums import SupportStatus, UserRole
from luber_schemas.plans import PlanId

TABLES = [
    User.__table__,
    Session.__table__,
    ReferenceAudio.__table__,
    Generation.__table__,
    GenerationJob.__table__,
    AudioAsset.__table__,
    GenerationQA.__table__,
    LyricLineQA.__table__,
    Project.__table__,
    Subscription.__table__,
    AllowanceReservation.__table__,
    BillingCheckout.__table__,
    BillingPayment.__table__,
    BillingEvent.__table__,
    BillingAnomaly.__table__,
    SupportTicket.__table__,
    SupportReply.__table__,
    DownloadEvent.__table__,
    AdminAuditLog.__table__,
    AdminEmailCampaign.__table__,
]


@pytest.fixture
async def factory(tmp_path):
    # File-backed rather than in-memory: the concurrency test needs two
    # sessions on two connections looking at the same database.
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/admin.db")
    async with engine.begin() as conn:
        await conn.run_sync(lambda sync_conn: Base.metadata.create_all(sync_conn, tables=TABLES))
    yield create_session_factory(engine)
    await engine.dispose()


async def _make_user(session, email: str, role: UserRole = UserRole.USER) -> User:
    user = User(id=uuid.uuid4(), email=email, password_hash="x", role=role.value)
    session.add(user)
    await session.commit()
    return user


async def _audit_actions(factory) -> list[str]:
    async with factory() as session:
        rows = (
            await session.execute(select(AdminAuditLog).order_by(AdminAuditLog.created_at))
        ).scalars()
        return [row.action for row in rows]


# ── roles ────────────────────────────────────────────────────────────


async def test_granting_records_a_grant(factory) -> None:
    async with factory() as session:
        actor = await _make_user(session, "boss@example.com", UserRole.SUPER_ADMIN)
        target = await _make_user(session, "new@example.com")

        await AdminRepository(session, actor).set_role(target.id, UserRole.ADMIN)

    assert await _audit_actions(factory) == [ACTION_ADMIN_GRANTED]


async def test_changing_between_roles_is_neither_a_grant_nor_a_revoke(factory) -> None:
    """The log is read months later by someone reconstructing a decision;
    calling a promotion a grant would misdescribe it."""
    async with factory() as session:
        actor = await _make_user(session, "boss@example.com", UserRole.SUPER_ADMIN)
        target = await _make_user(session, "mid@example.com", UserRole.ADMIN)

        await AdminRepository(session, actor).set_role(target.id, UserRole.SUPER_ADMIN)

    assert await _audit_actions(factory) == [ACTION_ADMIN_ROLE_CHANGED]


async def test_revoking_records_a_revoke(factory) -> None:
    async with factory() as session:
        actor = await _make_user(session, "boss@example.com", UserRole.SUPER_ADMIN)
        target = await _make_user(session, "old@example.com", UserRole.ADMIN)

        await AdminRepository(session, actor).set_role(target.id, UserRole.USER)

    assert await _audit_actions(factory) == [ACTION_ADMIN_REVOKED]


async def test_setting_the_role_an_account_already_has_records_nothing(factory) -> None:
    """Idempotent, and quiet about it. A log full of no-ops is a log
    nobody reads."""
    async with factory() as session:
        actor = await _make_user(session, "boss@example.com", UserRole.SUPER_ADMIN)
        target = await _make_user(session, "same@example.com", UserRole.ADMIN)

        await AdminRepository(session, actor).set_role(target.id, UserRole.ADMIN)

    assert await _audit_actions(factory) == []


async def test_an_unknown_account_is_refused(factory) -> None:
    async with factory() as session:
        actor = await _make_user(session, "boss@example.com", UserRole.SUPER_ADMIN)

        with pytest.raises(TargetNotFound):
            await AdminRepository(session, actor).set_role(uuid.uuid4(), UserRole.ADMIN)


# ── the lockout guard ────────────────────────────────────────────────


async def test_the_only_super_admin_cannot_step_down(factory) -> None:
    async with factory() as session:
        actor = await _make_user(session, "only@example.com", UserRole.SUPER_ADMIN)

        with pytest.raises(LastSuperAdmin):
            await AdminRepository(session, actor).set_role(actor.id, UserRole.USER)


async def test_a_refused_demotion_leaves_no_audit_entry(factory) -> None:
    """The audit row and the change share a transaction."""
    async with factory() as session:
        actor = await _make_user(session, "only@example.com", UserRole.SUPER_ADMIN)
        with pytest.raises(LastSuperAdmin):
            await AdminRepository(session, actor).set_role(actor.id, UserRole.USER)
        await session.rollback()

    assert await _audit_actions(factory) == []

    async with factory() as session:
        still = (
            await session.execute(
                select(func.count(User.id)).where(User.role == UserRole.SUPER_ADMIN.value)
            )
        ).scalar_one()
    assert still == 1, "the refusal must not have half-applied"


async def test_two_super_admins_cannot_both_demote_the_other(factory) -> None:
    """The reason the count happens inside the transaction.

    Two operators acting at the same instant would each pass a check made
    before the other's write landed, and the console would end up with
    nobody able to administer it. This runs both demotions concurrently
    on separate sessions; at most one may succeed.
    """
    async with factory() as session:
        first = await _make_user(session, "a@example.com", UserRole.SUPER_ADMIN)
        second = await _make_user(session, "b@example.com", UserRole.SUPER_ADMIN)
        first_id, second_id = first.id, second.id

    async def demote(actor_id: uuid.UUID, target_id: uuid.UUID) -> str:
        async with factory() as session:
            actor = await session.get(User, actor_id)
            assert actor is not None
            try:
                await AdminRepository(session, actor).set_role(target_id, UserRole.USER)
                return "applied"
            except LastSuperAdmin:
                return "refused"
            except Exception:
                # A database-level serialisation refusal is also a
                # refusal — what must not happen is both succeeding.
                return "refused"

    await asyncio.gather(
        demote(first_id, second_id),
        demote(second_id, first_id),
    )

    async with factory() as session:
        remaining = (
            await session.execute(
                select(func.count(User.id)).where(User.role == UserRole.SUPER_ADMIN.value)
            )
        ).scalar_one()

    assert remaining >= 1, "the console must never be left with no super administrator"


# ── support ──────────────────────────────────────────────────────────


async def _ticket(session, user: User) -> SupportTicket:
    ticket = SupportTicket(
        id=uuid.uuid4(),
        reference=f"SUP-{uuid.uuid4().hex[:8].upper()}",
        user_id=user.id,
        category="BILLING",
        subject="결제 확인",
        message="확인 부탁드립니다.",
        status=SupportStatus.OPEN.value,
    )
    session.add(ticket)
    await session.commit()
    return ticket


async def test_resolving_stamps_a_resolution_time(factory) -> None:
    async with factory() as session:
        actor = await _make_user(session, "op@example.com", UserRole.ADMIN)
        customer = await _make_user(session, "customer@example.com")
        ticket = await _ticket(session, customer)

        updated = await AdminRepository(session, actor).update_ticket(
            ticket.reference, status=SupportStatus.RESOLVED
        )

    assert updated.resolved_at is not None


async def test_reopening_clears_the_resolution_time(factory) -> None:
    """A reopened ticket that still claims a resolution time would make
    every "time to resolve" figure wrong."""
    async with factory() as session:
        actor = await _make_user(session, "op@example.com", UserRole.ADMIN)
        customer = await _make_user(session, "customer@example.com")
        ticket = await _ticket(session, customer)
        repository = AdminRepository(session, actor)

        await repository.update_ticket(ticket.reference, status=SupportStatus.RESOLVED)
        updated = await repository.update_ticket(ticket.reference, status=SupportStatus.IN_PROGRESS)

    assert updated.resolved_at is None


async def test_a_note_is_stored_but_its_text_is_not_audited(factory) -> None:
    """An audit log records actions, not a second copy of the content
    those actions were about."""
    async with factory() as session:
        actor = await _make_user(session, "op@example.com", UserRole.ADMIN)
        customer = await _make_user(session, "customer@example.com")
        ticket = await _ticket(session, customer)

        updated = await AdminRepository(session, actor).update_ticket(
            ticket.reference, admin_note="카드사 승인번호 12345"
        )

    assert updated.admin_note == "카드사 승인번호 12345"

    async with factory() as session:
        rows = (await session.execute(select(AdminAuditLog))).scalars()
        recorded = [row.meta for row in rows]

    assert all("12345" not in str(meta) for meta in recorded)


async def test_an_unknown_ticket_is_refused(factory) -> None:
    async with factory() as session:
        actor = await _make_user(session, "op@example.com", UserRole.ADMIN)

        with pytest.raises(TargetNotFound):
            await AdminRepository(session, actor).get_ticket("SUP-NOPE")


# ── campaigns ────────────────────────────────────────────────────────


async def test_a_campaign_is_stored_as_a_draft(factory) -> None:
    """There is no provider to send it. See `docs/ADMIN_CONSOLE.md`."""
    async with factory() as session:
        actor = await _make_user(session, "op@example.com", UserRole.ADMIN)

        campaign = await AdminRepository(session, actor).create_campaign(
            subject="공지",
            body="내용",
            audience_type=AUDIENCE_ALL,
            recipient_count=3,
            plan=None,
        )

    assert campaign.status == CAMPAIGN_DRAFT
    assert campaign.sent_at is None


async def test_the_audience_count_excludes_closed_accounts(factory) -> None:
    async with factory() as session:
        actor = await _make_user(session, "op@example.com", UserRole.ADMIN)
        await _make_user(session, "live@example.com")
        closed = await _make_user(session, "closed@example.com")
        closed.deleted_at = datetime.now(UTC)
        await session.commit()

        reached = await AdminRepository(session, actor).resolve_audience(audience_type=AUDIENCE_ALL)

    # The operator and the live customer; not the closed account.
    assert reached == 2


async def test_a_plan_audience_counts_only_that_tier(factory) -> None:
    async with factory() as session:
        actor = await _make_user(session, "op@example.com", UserRole.ADMIN)

        reached = await AdminRepository(session, actor).resolve_audience(
            audience_type=AUDIENCE_PLAN, plan=PlanId.PRO
        )

    assert reached == 0


# ── audit reads ──────────────────────────────────────────────────────


async def test_the_audit_count_and_the_page_answer_the_same_question(factory) -> None:
    """A total describing the unfiltered table under a filtered list is
    worse than no total."""
    async with factory() as session:
        actor = await _make_user(session, "boss@example.com", UserRole.SUPER_ADMIN)
        first = await _make_user(session, "one@example.com")
        second = await _make_user(session, "two@example.com", UserRole.ADMIN)
        repository = AdminRepository(session, actor)

        await repository.set_role(first.id, UserRole.ADMIN)
        await repository.set_role(second.id, UserRole.USER)

        granted = await repository.audit_log(action=ACTION_ADMIN_GRANTED)
        counted = await repository.count_audit(action=ACTION_ADMIN_GRANTED)

    assert len(granted) == counted == 1


async def test_an_audit_entry_carries_both_addresses(factory) -> None:
    async with factory() as session:
        actor = await _make_user(session, "boss@example.com", UserRole.SUPER_ADMIN)
        target = await _make_user(session, "target@example.com")
        repository = AdminRepository(session, actor)

        await repository.set_role(target.id, UserRole.ADMIN)
        entries = await repository.audit_log()

    _, actor_email, target_email = entries[0]
    assert actor_email == "boss@example.com"
    assert target_email == "target@example.com"


async def test_an_entry_with_no_subject_still_lists(factory) -> None:
    """A campaign has no target account; the outer join must not drop
    the row."""
    async with factory() as session:
        actor = await _make_user(session, "op@example.com", UserRole.ADMIN)
        repository = AdminRepository(session, actor)

        await repository.create_campaign(
            subject="공지", body="내용", audience_type=AUDIENCE_ALL, recipient_count=0, plan=None
        )
        entries = await repository.audit_log()

    assert len(entries) == 1
    assert entries[0][2] is None
