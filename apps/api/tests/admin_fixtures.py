"""Clients holding an operator role.

Every one of these signs up through the real route and then has its
`users.role` column set directly — which is exactly how the console is
bootstrapped in production, by `scripts/ops/grant_admin.py` writing the
same column. There is no test-only authorisation bypass anywhere here:
the requests under test carry an ordinary session cookie and are checked
by the ordinary dependency, so what the suite proves is what production
does.

Named distinctly rather than living in `conftest` because a
whole-repository run must not resolve one package's `conftest` import to
another's — the convention this directory already follows.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import update

from luber_database.models.user import User
from luber_schemas.enums import UserRole

TEST_PASSWORD = "correct horse battery staple"


async def set_role(app: FastAPI, user_id: str, role: UserRole) -> None:
    """Write the role column, as the bootstrap script does."""
    import uuid

    factory = app.state.session_factory
    async with factory() as session:
        await session.execute(
            update(User).where(User.id == uuid.UUID(user_id)).values(role=role.value)
        )
        await session.commit()


async def signed_up_client(app: FastAPI, email: str, role: UserRole | None = None) -> AsyncClient:
    """A registered account, optionally holding a role.

    The client is returned already used — the signup request opened it —
    so callers close it with `aclose()` rather than re-entering it as a
    context manager, which httpx refuses.
    """
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url="http://testserver")
    response = await client.post(
        "/v1/auth/signup", json={"email": email, "password": TEST_PASSWORD}
    )
    assert response.status_code == 201, response.text
    client.user_id = str(response.json()["id"])  # type: ignore[attr-defined]
    if role is not None:
        await set_role(app, client.user_id, role)  # type: ignore[attr-defined]
    return client


@pytest.fixture
async def admin_client(app: FastAPI):
    """An `ADMIN`: may run the service, may not change who else can."""
    client = await signed_up_client(app, "admin@example.com", UserRole.ADMIN)
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
async def super_admin_client(app: FastAPI):
    """A `SUPER_ADMIN`: the only role that may grant roles."""
    client = await signed_up_client(app, "super@example.com", UserRole.SUPER_ADMIN)
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
async def second_super_admin_client(app: FastAPI):
    """A second super administrator.

    Needed to prove the lockout guard blocks the *last* one and not
    merely any demotion of a super administrator.
    """
    client = await signed_up_client(app, "super2@example.com", UserRole.SUPER_ADMIN)
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
async def plain_client(app: FastAPI):
    """An ordinary customer. The adversary in every permission test."""
    client = await signed_up_client(app, "customer@example.com")
    try:
        yield client
    finally:
        await client.aclose()


__all__ = [
    "admin_client",
    "plain_client",
    "second_super_admin_client",
    "set_role",
    "signed_up_client",
    "super_admin_client",
]
