#!/usr/bin/env python3
"""Create the first operator, and repair a locked-out console.

The admin console checks `users.role`, and the migration that added that
column defaulted every account to `USER`. So on a fresh database nobody
can reach `/v1/admin/*` — including the person who owns the service.
Something has to make the first super administrator, and that something
cannot itself be an admin route: an endpoint that grants the first role
without already having one is an endpoint that grants it to anyone.

This script is the answer. Its authorisation is possession of
`DATABASE_URL` and shell access to the machine holding the database —
whoever has that can already write the column by hand, so the script
adds no privilege. What it adds is doing it correctly: the same audit
row, the same last-super-admin guard, and no chance of a typo leaving
`role = 'SUPERADMIN'` in a column nothing matches.

Why an address is a lookup and never a rule
-------------------------------------------
`--email` finds a row. It does not confer anything by itself, and no
code path anywhere compares a signed-in address against a constant. That
distinction is the whole point of the column: a hardcoded
`if email == "..."` cannot be revoked, leaves no audit trail, appears in
source control, and silently transfers the console to whoever comes to
own that mailbox. A role can be granted, listed, audited and taken back.

What this refuses to do
-----------------------
It does not create accounts. The target must already have registered
through the product, with a password only they know — otherwise a
bootstrap tool would be a way to manufacture a login.

It does not remove the last super administrator, for the same reason the
API does not: the recovery from an empty console is a hand-written
migration.

It reads `DATABASE_URL` from the environment and never prints it.

    uv run python scripts/ops/grant_admin.py --list
    uv run python scripts/ops/grant_admin.py --email someone@example.com --dry-run
    uv run python scripts/ops/grant_admin.py --email someone@example.com --role SUPER_ADMIN
    uv run python scripts/ops/grant_admin.py --email someone@example.com --revoke
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import select  # noqa: E402

from luber_database import create_async_engine_from_url, create_session_factory  # noqa: E402
from luber_database.admin_repository import (  # noqa: E402
    AdminRepository,
    LastSuperAdmin,
    TargetNotFound,
)
from luber_database.models.user import User  # noqa: E402
from luber_schemas.enums import UserRole  # noqa: E402


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set.")
    return url


async def _find(session: object, email: str) -> User | None:
    result = await session.execute(  # type: ignore[attr-defined]
        select(User).where(User.email == email.strip().lower(), User.deleted_at.is_(None))
    )
    return result.scalar_one_or_none()


async def _run(email: str | None, role: UserRole, revoke: bool, list_only: bool) -> int:
    engine = create_async_engine_from_url(_database_url())
    factory = create_session_factory(engine)
    try:
        async with factory() as session:
            if list_only:
                rows = (
                    await session.execute(
                        select(User)
                        .where(
                            User.role.in_((UserRole.ADMIN.value, UserRole.SUPER_ADMIN.value)),
                            User.deleted_at.is_(None),
                        )
                        .order_by(User.created_at)
                    )
                ).scalars()
                found = list(rows)
                if not found:
                    print("no administrators — the console is unreachable until one is granted")
                    return 1
                print(f"{len(found)} administrator(s):\n")
                for user in found:
                    print(f"  {user.role:<12} {user.email}")
                return 0

            assert email is not None, "argparse guarantees this"
            target = await _find(session, email)
            if target is None:
                # Deliberately not created. See the module docstring.
                raise SystemExit(
                    f"No account for {email}. Register through the product first — "
                    "this script promotes existing accounts and never creates them."
                )

            wanted = UserRole.USER if revoke else role
            if target.role == wanted.value:
                print(f"{target.email} is already {wanted.value} — nothing to do")
                return 0

            print(f"{target.email}: {target.role} → {wanted.value}")

            # The actor is the target when bootstrapping the very first
            # administrator: there is no other account that could be
            # credited, and an audit row attributed to nobody is worse
            # than one attributed to the account the change was about.
            actor = target
            repository = AdminRepository(session, actor)
            try:
                await repository.set_role(target.id, wanted)
            except LastSuperAdmin:
                raise SystemExit(
                    "Refused: that would leave no super administrator. Grant another one first."
                ) from None
            except TargetNotFound:
                raise SystemExit("Account disappeared mid-change.") from None

            print("done — recorded in admin_audit_logs")
            return 0
    finally:
        await engine.dispose()


async def _dry_run(email: str, role: UserRole, revoke: bool) -> int:
    """Say what would change, and change nothing."""
    engine = create_async_engine_from_url(_database_url())
    factory = create_session_factory(engine)
    try:
        async with factory() as session:
            target = await _find(session, email)
            if target is None:
                print(f"would fail: no account for {email}")
                return 1
            wanted = UserRole.USER if revoke else role
            if target.role == wanted.value:
                print(f"would do nothing: {target.email} is already {wanted.value}")
            else:
                print(f"would change {target.email}: {target.role} → {wanted.value}")
            return 0
    finally:
        await engine.dispose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--email", help="the existing account to promote or demote")
    parser.add_argument(
        "--role",
        choices=[UserRole.ADMIN.value, UserRole.SUPER_ADMIN.value],
        default=UserRole.SUPER_ADMIN.value,
        help="role to grant (default: SUPER_ADMIN, for the first operator)",
    )
    parser.add_argument("--revoke", action="store_true", help="return the account to USER")
    parser.add_argument("--list", action="store_true", help="show who currently holds a role")
    parser.add_argument("--dry-run", action="store_true", help="report without changing anything")
    args = parser.parse_args(argv)

    if not args.list and not args.email:
        parser.error("--email is required unless --list is given")

    role = UserRole(args.role)
    if args.dry_run:
        if not args.email:
            parser.error("--dry-run needs --email")
        return asyncio.run(_dry_run(args.email, role, args.revoke))
    return asyncio.run(_run(args.email, role, args.revoke, args.list))


if __name__ == "__main__":
    raise SystemExit(main())
