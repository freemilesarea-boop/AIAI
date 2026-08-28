"""Who may use the operator console.

One module, three dependencies, and every admin route goes through them.
Centralised rather than a role comparison at each handler, because a
permission check that appears in twenty places is a permission check
that will be missing from the twenty-first.

Three properties this is built to have:

**The session is the only source of identity.** Nothing here reads a
user id, an email or a role from a request body, a header or a query
string. The account is whoever the session cookie resolves to, and the
role is the column on that row.

**Authorisation is server-side and unconditional.** Hiding `/admin` in
the frontend router is a convenience for the user, not a control — a
`USER` who types the URL reaches a page that renders nothing useful
because every API call behind it answers 403.

**Managing administrators is separate from being one.** An `ADMIN` runs
the service; only a `SUPER_ADMIN` changes who else can. Splitting them
means a compromised operator account cannot quietly grant itself
permanence.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from luber_api.dependencies import get_session_factory
from luber_api.session import require_current_user
from luber_database.admin_repository import AdminRepository
from luber_database.models.user import User
from luber_schemas.enums import ADMIN_ROLES, UserRole

logger = logging.getLogger(__name__)


def _role_of(user: User) -> UserRole:
    """The account's role, defaulting closed.

    An unrecognised value is not an administrator. The column is a
    string, and the safe reading of a string nobody recognises is the
    least privilege it could mean.
    """
    try:
        return UserRole(user.role)
    except ValueError:
        logger.warning("unrecognised role on account", extra={"user_id": str(user.id)})
        return UserRole.USER


def require_admin(
    request: Request,
    user: Annotated[User, Depends(require_current_user)],
) -> User:
    """An account that may reach the console at all.

    `require_current_user` answers 401 for anonymous callers, so by the
    time this runs there is a real session. What is left is whether that
    account has a role, and a plain `USER` gets 403 — a different answer
    from 401 on purpose: they are authenticated, and telling them so is
    not a leak.
    """
    if _role_of(user) not in ADMIN_ROLES:
        logger.warning(
            "non-admin reached an admin route",
            extra={"path": request.url.path, "user_id": str(user.id)},
        )
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Administrator access required.")
    return user


def require_super_admin(
    request: Request,
    user: Annotated[User, Depends(require_admin)],
) -> User:
    """An account that may change who else is an administrator.

    Layered on `require_admin` rather than repeating its check, so there
    is exactly one place that decides what an administrator is.
    """
    if _role_of(user) is not UserRole.SUPER_ADMIN:
        logger.warning(
            "admin attempted a super-admin action",
            extra={"path": request.url.path, "user_id": str(user.id)},
        )
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Super administrator access required.")
    return user


async def get_admin_repository(
    request: Request,
    actor: Annotated[User, Depends(require_admin)],
) -> AsyncIterator[AdminRepository]:
    """The operator repository, carrying who is acting.

    The actor is bound here from the session, so every audit row written
    downstream is attributed correctly without any route having to pass
    an identity along.
    """
    factory = get_session_factory(request)
    async with factory() as session:
        yield AdminRepository(session, actor)


async def get_super_admin_repository(
    request: Request,
    actor: Annotated[User, Depends(require_super_admin)],
) -> AsyncIterator[AdminRepository]:
    """The same repository, behind the stricter gate."""
    factory = get_session_factory(request)
    async with factory() as session:
        yield AdminRepository(session, actor)


__all__ = [
    "get_admin_repository",
    "get_super_admin_repository",
    "require_admin",
    "require_super_admin",
]
