"""The session cookie, and the one way to learn who is calling.

Every authenticated route resolves identity through
:func:`require_current_user` and nothing else. Cookie parsing lives here
once: repeating it per route is how one endpoint ends up with a subtly
weaker check than its neighbours.

``X-User-Id`` is deliberately not consulted. That header is a
pre-authentication placeholder, forgeable by anyone, and it must never
be able to produce a user from this module.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Response, status

from luber_api.security import hash_session_token
from luber_api.settings import ApiSettings, get_settings
from luber_database import AuthRepository
from luber_database.models.user import User

#: Named for the product, so it cannot collide with another service's
#: cookie on a shared development host.
SESSION_COOKIE_NAME = "luber_session"


def cookie_is_secure(settings: ApiSettings) -> bool:
    """Whether to mark the cookie ``Secure``.

    True in production, where the deployment terminates TLS. False in
    development, because a ``Secure`` cookie is silently dropped over
    plain HTTP on localhost — the login would appear to succeed and then
    every subsequent request would be anonymous, which is a confusing
    way to discover a config flag.
    """
    return settings.is_production


def set_session_cookie(response: Response, token: str, *, settings: ApiSettings) -> None:
    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        max_age=settings.session_lifetime_seconds,
        # Unreadable from JavaScript: an XSS bug should not also be a
        # session-theft bug.
        httponly=True,
        # Lax, not None. The browser reaches the API through the web
        # origin's own /api proxy, so every legitimate request is
        # same-site and Lax simply works — while cross-site form posts
        # from another origin arrive without the cookie.
        samesite="lax",
        secure=cookie_is_secure(settings),
        path="/",
        # No Domain: scoped to the exact host that set it. Sharing across
        # subdomains is not needed and would widen where it is sent.
    )


def clear_session_cookie(response: Response, *, settings: ApiSettings) -> None:
    """Expire the cookie with the same attributes it was set with.

    A browser only replaces a cookie when path and domain match, so the
    attributes here are not decorative — get them wrong and the stale
    cookie survives.
    """
    response.delete_cookie(
        SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        samesite="lax",
        secure=cookie_is_secure(settings),
    )


async def get_auth_repository(request: Request) -> AsyncIterator[AuthRepository]:
    # Read from app state directly rather than importing the dependency
    # module: that module needs require_current_user from here, and one
    # of the two edges has to go.
    factory = request.app.state.session_factory
    async with factory() as session:
        yield AuthRepository(session)


async def get_current_user(
    request: Request,
    repository: Annotated[AuthRepository, Depends(get_auth_repository)],
) -> User | None:
    """The authenticated user, or ``None``. Never raises.

    Every failure — no cookie, a token matching no row, an expired
    session, a user since deleted — produces the same ``None``. A caller
    cannot tell them apart, so the cookie cannot be used to probe for
    valid sessions.
    """
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return None
    return await repository.get_session_user(hash_session_token(token), now=datetime.now(UTC))


async def require_current_user(
    user: Annotated[User | None, Depends(get_current_user)],
) -> User:
    """The authenticated user, or 401. The canonical guard."""
    if user is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Authentication required.",
            headers={"WWW-Authenticate": "Cookie"},
        )
    return user


def enforce_trusted_origin(
    request: Request,
    settings: Annotated[ApiSettings, Depends(get_settings)],
) -> None:
    """Reject an unsafe request that came from an origin we do not serve.

    The CSRF layer that sits behind ``SameSite=Lax``. Lax already keeps
    the cookie off cross-site POSTs, so this is defence in depth rather
    than the primary control — but Lax is a browser behaviour, and a
    header check is ours.

    A missing ``Origin`` is allowed: non-browser clients (curl, the
    tests, server-to-server callers) do not send one, and they are not
    the threat CSRF describes — they have no ambient cookie to abuse.
    Browsers always send it on cross-origin unsafe requests, which is
    exactly the case this catches.
    """
    if request.method in {"GET", "HEAD", "OPTIONS", "TRACE"}:
        return
    origin = request.headers.get("origin")
    if origin is None:
        return
    if origin not in settings.cors_origins:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Origin not allowed.")


__all__ = [
    "SESSION_COOKIE_NAME",
    "clear_session_cookie",
    "cookie_is_secure",
    "enforce_trusted_origin",
    "get_auth_repository",
    "get_current_user",
    "require_current_user",
    "set_session_cookie",
]
