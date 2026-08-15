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
from luber_generation_client import MockGenerationProvider

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_WAV = REPO_ROOT / "tests" / "fixtures" / "mock_generation.wav"

GENERATION_TABLES = [
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
