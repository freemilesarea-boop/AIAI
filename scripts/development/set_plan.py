#!/usr/bin/env python3
"""Put an account on a plan. Development and operations only.

BOORDA has no payment provider yet, which leaves a real question: how
does anyone get onto Basic, Pro or Creator in order to test the product?

The answer this repository gives is *not* an endpoint. A route that let a
signed-in account choose its own tier is a way to take Creator for
nothing, no matter how it is named, how it is documented, or which
environment variable is supposed to keep it out of production — the
switch that disables it is one misconfiguration away from being on. A
script is different in kind: it needs shell access to the machine holding
the database, and there is no misconfiguration that runs a command nobody
typed.

    uv run python scripts/development/set_plan.py --email you@example.com --plan pro
    uv run python scripts/development/set_plan.py --email you@example.com --show

Assigning a plan does not reset the allowance period: switching tiers
mid-month must not hand out a fresh twenty songs, and the repository is
where that rule lives, not here.

This writes a subscription row and nothing else. It records no payment,
issues no receipt and creates no billing history, because none of those
things happened.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from pathlib import Path

from sqlalchemy import select

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from luber_database import create_async_engine_from_url, create_session_factory  # noqa: E402
from luber_database.allowance_repository import AllowanceRepository  # noqa: E402
from luber_database.models.user import User  # noqa: E402
from luber_schemas.plans import PLAN_ORDER, PlanId  # noqa: E402


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit(
            "DATABASE_URL is not set. Run this from the repository root with the "
            "project's .env loaded, or export it explicitly."
        )
    return url


async def _run(email: str, plan: PlanId | None) -> int:
    engine = create_async_engine_from_url(_database_url())
    factory = create_session_factory(engine)
    try:
        async with factory() as session:
            found = await session.execute(select(User).where(User.email == email))
            user = found.scalar_one_or_none()
            if user is None:
                print(f"no account with email {email!r}", file=sys.stderr)
                return 1

            allowance = AllowanceRepository(session, uuid.UUID(str(user.id)))
            if plan is not None:
                await allowance.set_plan(plan)

            entitlement = await allowance.entitlement()
            print(f"account      {email}")
            print(
                f"plan         {entitlement.plan.display_name} ({entitlement.plan.plan_id.value})"
            )
            print(
                f"period       {entitlement.period_start.date()} → {entitlement.period_end.date()}"
            )
            print(
                f"generations  {entitlement.used}/{entitlement.limit} used, "
                f"{entitlement.remaining} remaining"
            )
            print(
                "downloads    "
                + ("mp3+wav" if entitlement.plan.can_download else "not included on this plan")
            )
            return 0
    finally:
        await engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--email", required=True, help="the account to act on")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--plan",
        choices=[plan.value for plan in PLAN_ORDER],
        help="the plan to assign",
    )
    group.add_argument(
        "--show", action="store_true", help="report the current plan without changing it"
    )
    args = parser.parse_args()
    plan = PlanId(args.plan) if args.plan else None
    return asyncio.run(_run(args.email, plan))


if __name__ == "__main__":
    raise SystemExit(main())
