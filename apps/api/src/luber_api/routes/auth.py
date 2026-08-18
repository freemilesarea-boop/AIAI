"""Signup, login, logout and the current user.

The cookie is the whole design. The browser holds an opaque token it
cannot read from JavaScript; the server holds a hash of it and the row
that gives it meaning. Nothing about the user travels in the cookie, so
there is nothing in it to tamper with — a forged cookie matches no row
and authenticates nobody.

Two behaviours here are security decisions rather than conveniences and
are commented where they occur: login refuses to distinguish an unknown
email from a wrong password, and unsafe methods are rejected when their
``Origin`` is not one this deployment serves.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, EmailStr, Field
from redis.asyncio import Redis
from sqlalchemy.exc import IntegrityError

from luber_api.dependencies import get_redis
from luber_api.rate_limit import RateLimitExceeded, enforce_rate_limit, reset_rate_limit
from luber_api.security import (
    PasswordPolicyError,
    generate_session_token,
    hash_password,
    hash_session_token,
    needs_rehash,
    normalise_email,
    validate_password,
    verify_password,
)
from luber_api.session import (
    SESSION_COOKIE_NAME,
    clear_session_cookie,
    enforce_trusted_origin,
    get_auth_repository,
    get_current_user,
    require_current_user,
    set_session_cookie,
)
from luber_api.settings import ApiSettings, get_settings
from luber_database import AuthRepository
from luber_database.models.user import User

# Origin validation applies to the whole router. These are the only
# routes a session cookie currently authenticates, so this is the entire
# CSRF surface today; Part 3 extends it as product routes start reading
# the cookie.
router = APIRouter(prefix="/v1/auth", tags=["auth"], dependencies=[Depends(enforce_trusted_origin)])

#: One message for every authentication failure. See the note in login.
INVALID_CREDENTIALS = "Email or password is incorrect."


class SignupRequest(BaseModel):
    model_config = {"extra": "forbid"}

    email: EmailStr
    password: str = Field(min_length=1, max_length=1024)
    display_name: str | None = Field(default=None, max_length=120)


class LoginRequest(BaseModel):
    model_config = {"extra": "forbid"}

    email: EmailStr
    password: str = Field(min_length=1, max_length=1024)


class UserResponse(BaseModel):
    """Everything a client may know about a user.

    ``password_hash`` is absent and must stay absent. Building this by
    hand rather than from the ORM row means a column added later cannot
    leak by default.
    """

    model_config = {"extra": "forbid"}

    id: uuid.UUID
    email: str
    display_name: str | None
    created_at: datetime


def public_user(user: User) -> UserResponse:
    return UserResponse(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        created_at=user.created_at,
    )


async def _issue_session(
    repository: AuthRepository,
    response: Response,
    *,
    user: User,
    settings: ApiSettings,
) -> None:
    """Mint a session and attach its cookie."""
    token = generate_session_token()
    expires_at = datetime.now(UTC) + timedelta(seconds=settings.session_lifetime_seconds)
    await repository.create_session(
        token_hash=hash_session_token(token), user_id=user.id, expires_at=expires_at
    )
    set_session_cookie(response, token, settings=settings)


@router.post("/signup", status_code=status.HTTP_201_CREATED, response_model=UserResponse)
async def signup(
    payload: SignupRequest,
    request: Request,
    response: Response,
    repository: Annotated[AuthRepository, Depends(get_auth_repository)],
    redis: Annotated[Redis, Depends(get_redis)],
    settings: Annotated[ApiSettings, Depends(get_settings)],
) -> UserResponse:
    await _limit(redis, request, settings, action="signup")

    try:
        validate_password(payload.password)
    except PasswordPolicyError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

    email = normalise_email(str(payload.email))
    display_name = (payload.display_name or "").strip() or None

    try:
        user = await repository.create_user(
            email=email,
            password_hash=hash_password(payload.password),
            display_name=display_name,
        )
    except IntegrityError as exc:
        # The unique index is the authority. Reaching this means either a
        # genuine duplicate or a race between two signups, and both get
        # the same answer.
        raise HTTPException(
            status.HTTP_409_CONFLICT, "An account with that email already exists."
        ) from exc

    await _issue_session(repository, response, user=user, settings=settings)
    return public_user(user)


@router.post("/login", response_model=UserResponse)
async def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
    repository: Annotated[AuthRepository, Depends(get_auth_repository)],
    redis: Annotated[Redis, Depends(get_redis)],
    settings: Annotated[ApiSettings, Depends(get_settings)],
) -> UserResponse:
    await _limit(redis, request, settings, action="login")

    email = normalise_email(str(payload.email))
    user = await repository.get_user_by_email(email)

    # One failure mode, whatever went wrong. Telling a caller that an
    # email is unknown turns this endpoint into a membership oracle:
    # anyone could test a list of addresses against the service. The
    # cost of the ambiguity is a slightly less helpful error; the cost
    # of the alternative is every user's account being discoverable.
    if user is None or not user.password_hash:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, INVALID_CREDENTIALS)
    if not verify_password(payload.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, INVALID_CREDENTIALS)

    # A successful login is the only moment the plaintext exists, so it
    # is the only chance to upgrade a hash whose parameters have aged.
    if needs_rehash(user.password_hash):
        await repository.update_password_hash(user.id, hash_password(payload.password))

    # Success clears the penalty: two typos then the right password
    # should not leave the window spent.
    client = request.client.host if request.client else "unknown"
    await reset_rate_limit(redis, key=f"auth:login:{client}")

    # A fresh session per login rather than reviving the old one: the
    # previous token stays valid for its own browser, and a stolen one
    # gains nothing from someone else logging in.
    await _issue_session(repository, response, user=user, settings=settings)
    return public_user(user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    request: Request,
    response: Response,
    repository: Annotated[AuthRepository, Depends(get_auth_repository)],
    settings: Annotated[ApiSettings, Depends(get_settings)],
) -> Response:
    """Destroy the server-side session, then clear the cookie.

    Server first. Clearing only the cookie would leave a live session
    that any retained copy of the token could still use, which is the
    difference between logging out and hiding the key.
    """
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token:
        await repository.delete_session(hash_session_token(token))
    clear_session_cookie(response, settings=settings)
    response.status_code = status.HTTP_204_NO_CONTENT
    return response


@router.get("/me", response_model=UserResponse)
async def me(user: Annotated[User, Depends(require_current_user)]) -> UserResponse:
    return public_user(user)


async def _limit(redis: Redis, request: Request, settings: ApiSettings, *, action: str) -> None:
    client = request.client.host if request.client else "unknown"
    try:
        await enforce_rate_limit(
            redis,
            key=f"auth:{action}:{client}",
            limit=settings.auth_rate_limit_attempts,
            window_seconds=settings.auth_rate_limit_window_seconds,
        )
    except RateLimitExceeded as exc:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Too many attempts. Please wait and try again.",
            headers={"Retry-After": str(exc.retry_after_seconds)},
        ) from exc


__all__ = ["get_current_user", "public_user", "require_current_user", "router"]
