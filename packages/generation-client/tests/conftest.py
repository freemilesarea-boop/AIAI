from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from luber_database import Base, GenerationRepository, create_session_factory
from luber_database.models.generation import (
    AudioAsset,
    Generation,
    GenerationJob,
    GenerationQA,
    LyricLineQA,
    Project,
)

GENERATION_TABLES = [
    Generation.__table__,
    GenerationJob.__table__,
    AudioAsset.__table__,
    GenerationQA.__table__,
    LyricLineQA.__table__,
    Project.__table__,
]


@pytest.fixture
async def engine():
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: Base.metadata.create_all(sync_conn, tables=GENERATION_TABLES)
        )
    yield engine
    await engine.dispose()


@pytest.fixture
async def repository(engine):
    factory = create_session_factory(engine)
    async with factory() as session:
        yield GenerationRepository(session)
