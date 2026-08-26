"""Queue contract test: the ARQ ``generate`` task drives GenerationService.

Calls the task function directly with a worker-shaped ``ctx`` (no live
Redis needed) — the same code path a real ARQ worker executes when the
API enqueues a generation_id.
"""

import uuid
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from luber_database import Base, GenerationRepository, create_session_factory
from luber_database.models.billing import AllowanceReservation
from luber_database.models.generation import AudioAsset, Generation, GenerationJob
from luber_generation_worker.worker import (
    GENERATION_QUEUE_NAME,
    WorkerConfig,
    WorkerSettings,
    generate,
)
from luber_schemas import GenerationStatus

#: Fixture rows need an owner now; a stable pretend user.
OWNER_FOR_TESTS = uuid.UUID("11111111-1111-4111-8111-111111111111")

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_WAV = REPO_ROOT / "tests" / "fixtures" / "mock_generation.wav"

GENERATION_TABLES = [
    Generation.__table__,
    GenerationJob.__table__,
    AudioAsset.__table__,
    # The worker finishes generations, and finishing one settles its
    # allowance slot — so the ledger is part of this path, not a
    # billing-only table the worker never touches.
    AllowanceReservation.__table__,
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


def test_generate_task_is_registered():
    assert generate in WorkerSettings.functions
    assert WorkerSettings.queue_name == GENERATION_QUEUE_NAME == "luber:generation"


async def test_generate_task_completes_generation(engine, tmp_path):
    session_factory = create_session_factory(engine)
    async with session_factory() as session:
        repo = GenerationRepository(session, owner=OWNER_FOR_TESTS)
        gen = await repo.create_generation(
            title="TEST SONG",
            prompt="Dreamy Korean indie pop",
            lyrics="[Verse]\n테스트 가사",
            vocal_gender="female",
            duration_requested=30,
            status=GenerationStatus.QUEUED.value,
        )
        await repo.create_job(
            gen.id, queue_name=GENERATION_QUEUE_NAME, status=GenerationStatus.QUEUED.value
        )

    ctx = {
        "config": WorkerConfig(
            generation_provider="mock",
            mock_fixture_path=str(FIXTURE_WAV),
            audio_storage_dir=str(tmp_path / "worker-audio"),
        ),
        "session_factory": session_factory,
        "worker_id": "test-worker-1",
    }

    result = await generate(ctx, str(gen.id))
    assert result == "COMPLETED"

    async with session_factory() as session:
        repo = GenerationRepository(session, owner=OWNER_FOR_TESTS)
        fetched = await repo.get_generation(gen.id)
        assert fetched.status == GenerationStatus.COMPLETED.value
        assets = await repo.get_audio_assets(gen.id)
        kinds = {a.asset_type for a in assets}
        assert {"MASTER", "PREVIEW"} <= kinds
        assert kinds <= {"MASTER", "FINISHED_MASTER", "PREVIEW"}
        for asset in assets:
            stored = tmp_path / "worker-audio" / asset.storage_key
            assert stored.is_file()
        job = await repo.get_latest_job(gen.id)
        assert job.worker_id == "test-worker-1"
        assert job.status == GenerationStatus.COMPLETED.value
