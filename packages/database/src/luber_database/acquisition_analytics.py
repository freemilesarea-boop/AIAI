"""Acquisition reporting, aggregated in the database.

Follows `admin_analytics`: every figure is a `GROUP BY`, days are Korean
days, and the range filter is a half-open interval on an indexed
timestamp so the index does the work.

**What the date range means.** Event-period reporting, not cohort
reporting, and the distinction is not cosmetic:

* 방문자 — visitors whose *session* started in the period
* 가입자 — accounts whose *signup* happened in the period
* 유료 전환 — accounts whose *first successful payment* happened in the
  period
* 매출 — successful payments *paid* in the period

So a visitor acquired in July who pays in August appears in July's
visitors and August's conversions. That is the honest reading of "what
happened this month"; cohort lifetime value is a different report and
is deliberately not pretended at here.

**What a paid conversion is.** The earliest successful payment for an
account, derived from `billing_payments` — the same source of truth the
revenue dashboard uses. Nothing writes a conversion record, so a retried
PayApp callback cannot double-count one: there is no second thing to
keep consistent. Renewals are excluded from conversions by construction
and remain in revenue, because a renewal is retention, not acquisition.

**Attribution mode.** First-touch groups by where an account originally
came from; last-touch by the most recent non-direct source before they
converted. Last-touch falls back to first-touch when a visitor only ever
arrived directly — they still came from somewhere, and dropping them
would make the two modes count different populations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from sqlalchemy import distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from luber_database.models.acquisition import (
    AcquisitionAttribution,
    AcquisitionSession,
    AcquisitionVisitor,
)
from luber_database.models.payments import PAYMENT_SUCCEEDED, BillingPayment
from luber_schemas.acquisition import DIRECT_MEDIUM, DIRECT_SOURCE, channel_of

AttributionMode = Literal["first_touch", "last_touch"]


def _attr_columns(mode: AttributionMode) -> tuple[Any, Any, Any]:
    """The (source, medium, campaign) an account is credited to.

    Last-touch coalesces onto first-touch: a visitor who only ever
    arrived directly has no last non-direct touch, and counting them in
    one mode but not the other would make the two modes describe
    different populations rather than the same one split differently.
    """
    if mode == "last_touch":
        return (
            func.coalesce(AcquisitionAttribution.last_source, AcquisitionAttribution.first_source),
            func.coalesce(AcquisitionAttribution.last_medium, AcquisitionAttribution.first_medium),
            func.coalesce(
                AcquisitionAttribution.last_campaign, AcquisitionAttribution.first_campaign
            ),
        )
    return (
        AcquisitionAttribution.first_source,
        AcquisitionAttribution.first_medium,
        AcquisitionAttribution.first_campaign,
    )


def _visit_columns(mode: AttributionMode) -> tuple[Any, Any, Any]:
    """The (source, medium, campaign) a *visit* is credited to.

    In last-touch mode a session is its own last touch — it is the thing
    that brought them back — so the session's own classification is
    used. In first-touch mode the visit is credited to wherever that
    visitor originally came from.
    """
    if mode == "last_touch":
        return (AcquisitionSession.source, AcquisitionSession.medium, AcquisitionSession.campaign)
    return (
        AcquisitionVisitor.first_source,
        AcquisitionVisitor.first_medium,
        AcquisitionVisitor.first_campaign,
    )


@dataclass(frozen=True)
class ChannelRow:
    """One reportable channel, with its funnel."""

    key: str
    source: str
    medium: str
    visitors: int = 0
    signups: int = 0
    conversions: int = 0
    revenue_krw: int = 0


@dataclass(frozen=True)
class CampaignRow:
    """One source / medium / campaign triple."""

    source: str
    medium: str
    campaign: str | None
    visitors: int = 0
    signups: int = 0
    conversions: int = 0
    revenue_krw: int = 0


def _first_payment_subquery() -> Any:
    """Each account's earliest successful payment.

    The definition of a paid conversion. A `MIN` per user rather than a
    stored flag, so a duplicate callback — which cannot create a second
    row anyway, the billing layer sees to that — could not create a
    second conversion even in principle.
    """
    return (
        select(
            BillingPayment.user_id.label("user_id"),
            func.min(BillingPayment.paid_at).label("converted_at"),
        )
        .where(BillingPayment.status == PAYMENT_SUCCEEDED)
        .group_by(BillingPayment.user_id)
        .subquery()
    )


def _in_window(column: Any, start: datetime, end: datetime) -> Any:
    """A half-open range on a timestamp column.

    Typed loosely for the same reason `kst_date` is: callers pass mapped
    attributes, which are `InstrumentedAttribute` rather than
    `ColumnElement`, and narrowing here would only be precise about a
    distinction SQLAlchemy does not make.
    """
    return (column >= start) & (column < end)


async def _grouped_visitors(
    session: AsyncSession, *, start: datetime, end: datetime, mode: AttributionMode
) -> dict[tuple[str, str, str | None], int]:
    """Distinct visitors per attribution, for sessions in the window."""
    src, med, camp = _visit_columns(mode)
    rows = (
        await session.execute(
            select(src, med, camp, func.count(distinct(AcquisitionSession.visitor_id)))
            .join(AcquisitionVisitor, AcquisitionVisitor.id == AcquisitionSession.visitor_id)
            .where(_in_window(AcquisitionSession.started_at, start, end))
            .group_by(src, med, camp)
        )
    ).all()
    return {(str(r[0]), str(r[1]), r[2]): int(r[3]) for r in rows}


async def _grouped_signups(
    session: AsyncSession, *, start: datetime, end: datetime, mode: AttributionMode
) -> dict[tuple[str, str, str | None], int]:
    src, med, camp = _attr_columns(mode)
    rows = (
        await session.execute(
            select(src, med, camp, func.count(AcquisitionAttribution.user_id))
            .where(_in_window(AcquisitionAttribution.created_at, start, end))
            .group_by(src, med, camp)
        )
    ).all()
    return {(str(r[0]), str(r[1]), r[2]): int(r[3]) for r in rows}


async def _grouped_conversions(
    session: AsyncSession, *, start: datetime, end: datetime, mode: AttributionMode
) -> dict[tuple[str, str, str | None], int]:
    """First paid conversions per attribution, in the window."""
    src, med, camp = _attr_columns(mode)
    first_payment = _first_payment_subquery()
    rows = (
        await session.execute(
            select(src, med, camp, func.count(AcquisitionAttribution.user_id))
            .join(first_payment, first_payment.c.user_id == AcquisitionAttribution.user_id)
            .where(_in_window(first_payment.c.converted_at, start, end))
            .group_by(src, med, camp)
        )
    ).all()
    return {(str(r[0]), str(r[1]), r[2]): int(r[3]) for r in rows}


async def _grouped_revenue(
    session: AsyncSession, *, start: datetime, end: datetime, mode: AttributionMode
) -> dict[tuple[str, str, str | None], int]:
    """Won from successful payments in the window, per attribution.

    Every payment, not only the first: revenue is revenue. Only the
    *conversion count* is restricted to the first one.
    """
    src, med, camp = _attr_columns(mode)
    rows = (
        await session.execute(
            select(src, med, camp, func.coalesce(func.sum(BillingPayment.amount_krw), 0))
            .join(AcquisitionAttribution, AcquisitionAttribution.user_id == BillingPayment.user_id)
            .where(
                BillingPayment.status == PAYMENT_SUCCEEDED,
                _in_window(BillingPayment.paid_at, start, end),
            )
            .group_by(src, med, camp)
        )
    ).all()
    return {(str(r[0]), str(r[1]), r[2]): int(r[3] or 0) for r in rows}


async def channel_breakdown(
    session: AsyncSession, *, start: datetime, end: datetime, mode: AttributionMode = "first_touch"
) -> list[ChannelRow]:
    """The funnel per channel, heaviest first.

    Channels with nothing at all are omitted rather than listed as
    zeroes: a table of empty rows is harder to read than a short one,
    and the console says so explicitly when everything is empty.
    """
    visitors = await _grouped_visitors(session, start=start, end=end, mode=mode)
    signups = await _grouped_signups(session, start=start, end=end, mode=mode)
    conversions = await _grouped_conversions(session, start=start, end=end, mode=mode)
    revenue = await _grouped_revenue(session, start=start, end=end, mode=mode)

    totals: dict[str, dict[str, Any]] = {}
    for bucket, values in (
        ("visitors", visitors),
        ("signups", signups),
        ("conversions", conversions),
        ("revenue_krw", revenue),
    ):
        for (src, med, _campaign), count in values.items():
            key = channel_of(src, med)
            entry = totals.setdefault(
                key,
                {
                    "source": src,
                    "medium": med,
                    "visitors": 0,
                    "signups": 0,
                    "conversions": 0,
                    "revenue_krw": 0,
                },
            )
            entry[bucket] += count

    rows = [
        ChannelRow(
            key=key,
            source=str(entry["source"]),
            medium=str(entry["medium"]),
            visitors=int(entry["visitors"]),
            signups=int(entry["signups"]),
            conversions=int(entry["conversions"]),
            revenue_krw=int(entry["revenue_krw"]),
        )
        for key, entry in totals.items()
    ]
    rows.sort(key=lambda r: (-r.visitors, -r.revenue_krw, r.key))
    return rows


async def campaign_breakdown(
    session: AsyncSession,
    *,
    start: datetime,
    end: datetime,
    mode: AttributionMode = "first_touch",
    limit: int = 200,
) -> list[CampaignRow]:
    """The same funnel, one row per source / medium / campaign."""
    visitors = await _grouped_visitors(session, start=start, end=end, mode=mode)
    signups = await _grouped_signups(session, start=start, end=end, mode=mode)
    conversions = await _grouped_conversions(session, start=start, end=end, mode=mode)
    revenue = await _grouped_revenue(session, start=start, end=end, mode=mode)

    keys = set(visitors) | set(signups) | set(conversions) | set(revenue)
    rows = [
        CampaignRow(
            source=src,
            medium=med,
            campaign=camp,
            visitors=visitors.get((src, med, camp), 0),
            signups=signups.get((src, med, camp), 0),
            conversions=conversions.get((src, med, camp), 0),
            revenue_krw=revenue.get((src, med, camp), 0),
        )
        for src, med, camp in keys
    ]
    rows.sort(key=lambda r: (-r.visitors, -r.revenue_krw, r.source, r.medium, r.campaign or ""))
    return rows[:limit]


async def summary(
    session: AsyncSession, *, start: datetime, end: datetime, mode: AttributionMode = "first_touch"
) -> dict[str, Any]:
    """Totals for the KPI row, plus the unattributed count.

    `unattributed_signups` is reported rather than hidden: accounts that
    predate this system have no acquisition record, and folding them
    into 직접 유입 would invent a channel for them.
    """
    channels = await channel_breakdown(session, start=start, end=end, mode=mode)
    visitors = sum(c.visitors for c in channels)
    signups = sum(c.signups for c in channels)
    conversions = sum(c.conversions for c in channels)
    revenue = sum(c.revenue_krw for c in channels)

    total_signups = int(
        (
            await session.execute(
                select(func.count()).select_from(
                    select(AcquisitionAttribution.user_id)
                    .where(_in_window(AcquisitionAttribution.created_at, start, end))
                    .subquery()
                )
            )
        ).scalar_one()
    )

    return {
        "visitors": visitors,
        "signups": signups,
        "conversions": conversions,
        "revenue_krw": revenue,
        # Rates against attributed visitors, which is the only
        # denominator both numerators share. Zero visitors means no
        # rate at all rather than a division.
        "signup_rate": round(signups / visitors, 4) if visitors else None,
        "conversion_rate": round(conversions / visitors, 4) if visitors else None,
        "attributed_signups": total_signups,
    }


async def unattributed_users(session: AsyncSession) -> int:
    """Accounts with no acquisition record at all.

    Everyone who signed up before this existed. Shown as 기존 회원, never
    as direct — there is no evidence either way, and inventing one is
    how a channel gets budget it did not earn.
    """
    from luber_database.models.user import User

    attributed = select(AcquisitionAttribution.user_id).scalar_subquery()
    return int(
        (
            await session.execute(
                select(func.count(User.id)).where(
                    User.deleted_at.is_(None), User.id.not_in(attributed)
                )
            )
        ).scalar_one()
    )


__all__ = [
    "DIRECT_MEDIUM",
    "DIRECT_SOURCE",
    "AttributionMode",
    "CampaignRow",
    "ChannelRow",
    "campaign_breakdown",
    "channel_breakdown",
    "summary",
    "unattributed_users",
]
