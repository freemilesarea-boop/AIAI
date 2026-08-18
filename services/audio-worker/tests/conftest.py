"""Database fixtures for the audio worker's maintenance tasks.

Same wiring as the other packages: an in-memory SQLite database with the
generation tables created from the models. Note that SQLite runs with
foreign keys disabled, which is precisely why the cleanup delete carries
its own NOT EXISTS condition rather than leaning on ON DELETE RESTRICT —
the guarantee has to hold here too, not only on PostgreSQL.
"""

from __future__ import annotations

import uuid

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
    ReferenceAudio,
)

GENERATION_TABLES = [
    ReferenceAudio.__table__,
    Generation.__table__,
    GenerationJob.__table__,
    AudioAsset.__table__,
    GenerationQA.__table__,
    LyricLineQA.__table__,
    Project.__table__,
]


#: A fixed owner for fixtures. Product rows now require one, so tests
#: create data as a stable pretend user rather than as nobody.
TEST_OWNER = uuid.UUID("11111111-1111-4111-8111-111111111111")


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
        yield GenerationRepository(session, owner=TEST_OWNER)
