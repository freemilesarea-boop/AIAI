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

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from luber_billing.states import LIVE_SUBSCRIPTION_STATES, SubscriptionState
from pydantic import BaseModel, EmailStr, Field
from redis.asyncio import Redis
from sqlalchemy.exc import IntegrityError

from luber_api.dependencies import get_redis
from luber_api.rate_limit import RateLimitExceeded, enforce_rate_limit, reset_rate_limit
from luber_api.routes.billing import get_billing_repository
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
from luber_database.billing_repository import BillingRepository
from luber_database.models.user import User

# Origin validation applies to the whole router. These are the only
# routes a session cookie currently authenticates, so this is the entire
# CSRF surface today; Part 3 extends it as product routes start reading
# the cookie.
logger = logging.getLogger(__name__)

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
    #: Whether this account holds an operator role, so the web app knows
    #: whether to show a link to the console.
    #:
    #: Presentation only. Nothing is authorised by this field: every
    #: `/v1/admin/*` request is checked server-side against the session's
    #: own row, so a browser that lies about it gets a nav item and 403s
    #: behind every one of them.
    role: str


def public_user(user: User) -> UserResponse:
    return UserResponse(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        created_at=user.created_at,
        role=user.role,
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


# ── account management ───────────────────────────────────────────────


class PasswordChangeRequest(BaseModel):
    """Everything the client may say about a password change.

    No user id. The account is the session's, and there is deliberately
    no field through which another one could be named.
    """

    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=1, max_length=256)
    new_password_confirm: str = Field(min_length=1, max_length=256)


class ProfileUpdateRequest(BaseModel):
    #: The only profile field the schema has. Empty string clears it,
    #: which is why it is nullable rather than required.
    display_name: str | None = Field(default=None, max_length=120)


class AccountDeleteRequest(BaseModel):
    """Re-authentication for a destructive act.

    The session already proves who is asking; the password proves they
    are still at the keyboard. A session cookie left open on a shared
    machine should not be enough to close somebody's account.
    """

    current_password: str = Field(min_length=1, max_length=256)


@router.post("/password", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(
    payload: PasswordChangeRequest,
    request: Request,
    response: Response,
    user: Annotated[User, Depends(require_current_user)],
    repository: Annotated[AuthRepository, Depends(get_auth_repository)],
    redis: Annotated[Redis, Depends(get_redis)],
    settings: Annotated[ApiSettings, Depends(get_settings)],
    _origin: Annotated[None, Depends(enforce_trusted_origin)] = None,
) -> Response:
    """Change the caller's own password.

    Rate limited on the same ladder as login: this endpoint verifies a
    password, so it is an oracle for one, and an authenticated attacker
    on a borrowed session should not be able to test candidates freely.

    Every other session is ended on success. A password change is what
    someone does when they think a credential is compromised, and
    leaving the other sessions alive would make the act cosmetic.
    """
    await _limit(redis, request, settings, action="password")

    if user.password_hash is None or not verify_password(
        payload.current_password, user.password_hash
    ):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Your current password is incorrect.")

    if payload.new_password != payload.new_password_confirm:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The new passwords do not match.")

    try:
        new_password = validate_password(payload.new_password)
    except PasswordPolicyError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    await repository.update_password_hash(user.id, hash_password(new_password))
    # Drop every session, then issue a fresh one for this browser, so
    # the person who just changed it is not logged out of the tab they
    # are looking at while everyone else is.
    await repository.delete_sessions_for_user(user.id)
    await _issue_session(repository, response, user=user, settings=settings)

    logger.info("password changed", extra={"user_id": str(user.id)})
    response.status_code = status.HTTP_204_NO_CONTENT
    return response


@router.patch("/profile", response_model=UserResponse)
async def update_profile(
    payload: ProfileUpdateRequest,
    user: Annotated[User, Depends(require_current_user)],
    repository: Annotated[AuthRepository, Depends(get_auth_repository)],
    _origin: Annotated[None, Depends(enforce_trusted_origin)] = None,
) -> UserResponse:
    """Set the display name. The only profile field the schema has.

    Email is not editable here. Changing an address that logs in and
    receives notices needs a verification round-trip to the new mailbox,
    and offering the field without one would let a typo lock someone
    out.
    """
    name = (payload.display_name or "").strip() or None
    updated = await repository.update_display_name(user.id, name)
    if updated is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Account not found.")
    return public_user(updated)


@router.post("/account/delete", status_code=status.HTTP_204_NO_CONTENT)
async def delete_account(
    payload: AccountDeleteRequest,
    request: Request,
    response: Response,
    user: Annotated[User, Depends(require_current_user)],
    repository: Annotated[AuthRepository, Depends(get_auth_repository)],
    billing: Annotated[BillingRepository, Depends(get_billing_repository)],
    redis: Annotated[Redis, Depends(get_redis)],
    settings: Annotated[ApiSettings, Depends(get_settings)],
    _origin: Annotated[None, Depends(enforce_trusted_origin)] = None,
) -> Response:
    """Close the caller's own account.

    The account is the session's. There is no user id in the request, so
    there is nothing to tamper with and no way to aim this at somebody
    else — the IDOR does not exist rather than being checked for.

    **A live subscription blocks this.** Closing an account while PayApp
    still holds a recurring contract would keep charging a card for a
    product the person can no longer reach, and that is the one outcome
    this must never produce. Cancelling on their behalf is not done
    here: it is a provider call with financial effect, and the product
    already has a cancellation flow the user can see the consequences
    of. So the account stays open and the UI says why.

    Closing is anonymisation, not deletion — see
    `AuthRepository.close_account` for why the schema requires that.
    """
    await _limit(redis, request, settings, action="delete")

    if user.password_hash is None or not verify_password(
        payload.current_password, user.password_hash
    ):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Your password is incorrect.")

    subscription = await billing.subscription()
    if subscription is not None and subscription.provider_subscription_id is not None:
        state = SubscriptionState(subscription.status)
        if state in LIVE_SUBSCRIPTION_STATES:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                "SUBSCRIPTION_ACTIVE",
            )

    await repository.close_account(user.id)
    # The session is already gone server-side; clearing the cookie stops
    # the browser sending a token that now matches nothing.
    clear_session_cookie(response, settings=settings)
    logger.info("account closed", extra={"user_id": str(user.id)})
    response.status_code = status.HTTP_204_NO_CONTENT
    return response


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
