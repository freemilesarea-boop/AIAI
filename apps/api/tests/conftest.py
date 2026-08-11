"""API test fixtures.

Unit tests never require a live PostgreSQL/Redis: the lifespan is
bypassed and app.state resources are replaced with fakes.
"""

from __future__ import annotations

import pytest
from fakeredis.aioredis import FakeRedis
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine

from luber_api.main import create_app


class _FailingRedis:
    async def ping(self):
        raise ConnectionError("redis unreachable")


@pytest.fixture
def app() -> FastAPI:
    application = create_app()
    # In-memory stand-ins: SQLite for engine connectivity, fakeredis for Redis.
    application.state.db_engine = create_async_engine("sqlite+aiosqlite://")
    application.state.redis = FakeRedis()
    return application


@pytest.fixture
async def client(app: FastAPI):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


@pytest.fixture
async def degraded_client(app: FastAPI):
    app.state.redis = _FailingRedis()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c
