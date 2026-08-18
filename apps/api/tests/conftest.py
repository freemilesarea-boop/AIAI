"""API test fixtures.

Tests never require live PostgreSQL/Redis: the lifespan is bypassed and
``app.state`` resources are replaced — SQLite (in-memory) for the DB,
fakeredis for Redis, a temp-dir LocalAudioStorage, and the
InlineGenerationRunner driving the real GenerationService +
MockGenerationProvider with the committed fixture WAV.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fakeredis.aioredis import FakeRedis
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine

from luber_api.jobs import InlineGenerationRunner
from luber_api.main import create_app
from luber_audio_utils import LocalAudioStorage
from luber_database import Base, create_session_factory
from luber_database.models.generation import (
    AudioAsset,
    Generation,
    GenerationJob,
    GenerationQA,
    LyricLineQA,
    Project,
    ReferenceAudio,
)
from luber_database.models.user import Session, User
from luber_generation_client import MockGenerationProvider

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_WAV = REPO_ROOT / "tests" / "fixtures" / "mock_generation.wav"

GENERATION_TABLES = [
    # Auth tables first: sessions carries a foreign key to users.
    User.__table__,
    Session.__table__,
    ReferenceAudio.__table__,
    Generation.__table__,
    GenerationJob.__table__,
    AudioAsset.__table__,
    GenerationQA.__table__,
    LyricLineQA.__table__,
    Project.__table__,
]


class _FailingRedis:
    async def ping(self):
        raise ConnectionError("redis unreachable")


@pytest.fixture
async def app(tmp_path) -> FastAPI:
    application = create_app()
    # File-backed SQLite: each session gets its own connection, so
    # concurrent requests exercise the real unique-constraint race.
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/api-test.db")
    async with engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: Base.metadata.create_all(sync_conn, tables=GENERATION_TABLES)
        )
    session_factory = create_session_factory(engine)
    storage = LocalAudioStorage(tmp_path / "audio-store")

    application.state.db_engine = engine
    application.state.session_factory = session_factory
    application.state.redis = FakeRedis()
    application.state.audio_storage = storage
    # Held on app.state as well so tests can assert *which* provider
    # method a request reached — routing an edit to generate() is the
    # failure mode Phase 13B exists to prevent.
    provider = MockGenerationProvider(FIXTURE_WAV)
    application.state.provider = provider
    application.state.enqueuer = InlineGenerationRunner(
        session_factory,
        provider,
        storage,
    )
    yield application
    await engine.dispose()


#: Every authenticated fixture signs up through the real route, so the
#: session under test is one the product actually issues. There is no
#: test-only bypass: a suite that skips the session dependency cannot
#: prove the boundary it exists to prove.
TEST_PASSWORD = "correct horse battery staple"


async def _sign_up(http: AsyncClient, email: str) -> str:
    """Create an account and leave its session cookie on the client."""
    response = await http.post("/v1/auth/signup", json={"email": email, "password": TEST_PASSWORD})
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


@pytest.fixture
async def anon_client(app: FastAPI):
    """No session. For asserting that product routes refuse anonymity."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


@pytest.fixture
async def client(app: FastAPI):
    """The default product client: authenticated as user A.

    Product routes require a session, so the ordinary fixture carries
    one. Tests that care about anonymity ask for ``anon_client``.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        c.user_id = await _sign_up(c, "user-a@example.com")  # type: ignore[attr-defined]
        yield c


@pytest.fixture
async def client_b(app: FastAPI):
    """A second, unrelated account. The adversary in ownership tests."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        c.user_id = await _sign_up(c, "user-b@example.com")  # type: ignore[attr-defined]
        yield c


@pytest.fixture
async def degraded_client(app: FastAPI):
    app.state.redis = _FailingRedis()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c
